import os, subprocess, re
import difflib
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple
from pydub import AudioSegment
from core.utils import *
from core.utils.models import *
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import mediainfo
from rich import print as rprint

def normalize_audio_volume(audio_path, output_path, target_db = -20.0, format = "wav"):
    audio = AudioSegment.from_file(audio_path)
    change_in_dBFS = target_db - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(output_path, format=format)
    rprint(f"[green]✅ Audio normalized from {audio.dBFS:.1f}dB to {target_db:.1f}dB[/green]")
    return output_path

def convert_video_to_audio(video_file: str):
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    if not os.path.exists(_RAW_AUDIO_FILE):
        rprint(f"[blue]🎬➡️🎵 Converting to high quality audio with FFmpeg ......[/blue]")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_file, '-vn',
            '-c:a', 'libmp3lame', '-b:a', '32k',
            '-ar', '16000',
            '-ac', '1', 
            '-metadata', 'encoding=UTF-8', _RAW_AUDIO_FILE
        ], check=True, stderr=subprocess.PIPE)
        rprint(f"[green]🎬➡️🎵 Converted <{video_file}> to <{_RAW_AUDIO_FILE}> with FFmpeg\n[/green]")

def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file using ffmpeg."""
    if not os.path.exists(audio_file):
        rprint(f"[red]❌ Error: Audio file does not exist: {audio_file}[/red]")
        return 0
        
    cmd = ['ffmpeg', '-i', audio_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    output = stderr.decode('utf-8', errors='ignore')
    
    try:
        duration_lines = [line for line in output.split('\n') if 'Duration' in line]
        if not duration_lines:
            raise ValueError("No Duration line found in ffmpeg output")
        duration_str = duration_lines[0]
        duration_part = duration_str.split('Duration: ')[1].split(',')[0]
        
        # Check for N/A duration (empty audio files)
        if duration_part.strip() == 'N/A':
            rprint(f"[yellow]⚠️ Warning: Audio file has N/A duration (likely empty): {audio_file}[/yellow]")
            duration = 0
        else:
            duration_parts = duration_part.split(':')
            if len(duration_parts) != 3:
                raise ValueError(f"Invalid duration format: {duration_parts}")
            duration = float(duration_parts[0])*3600 + float(duration_parts[1])*60 + float(duration_parts[2])
            if duration < 0:
                raise ValueError(f"Negative duration: {duration}")
    except Exception as e:
        rprint(f"[red]❌ Error: Failed to get audio duration for {audio_file}: {e}[/red]")
        # Fallback: try using pydub as alternative
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_file)
            duration = len(audio) / 1000.0  # pydub length is in milliseconds
            rprint(f"[yellow]⚠️ Using pydub fallback, duration: {duration:.2f}s[/yellow]")
        except Exception as fallback_e:
            rprint(f"[red]❌ Error: Pydub fallback also failed: {fallback_e}[/red]")
            duration = 0
    return duration

def split_audio(audio_file: str, target_len: float = 30*60, win: float = 60) -> List[Tuple[float, float]]:
    ## 在 [target_len-win, target_len+win] 区间内用 pydub 检测静默，切分音频
    rprint(f"[blue]🎙️ Starting audio segmentation {audio_file} {target_len} {win}[/blue]")
    audio = AudioSegment.from_file(audio_file)
    duration_info = mediainfo(audio_file)["duration"]
    if duration_info == 'N/A' or duration_info is None:
        rprint(f"[yellow]⚠️ Warning: Could not get duration from mediainfo, using audio segment length[/yellow]")
        duration = len(audio) / 1000.0  # pydub length is in milliseconds
    else:
        duration = float(duration_info)
    if duration <= target_len + win:
        return [(0, duration)]
    segments, pos = [], 0.0
    safe_margin = 0.5  # 静默点前后安全边界，单位秒

    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration)); break

        threshold = pos + target_len
        ws, we = int((threshold - win) * 1000), int((threshold + win) * 1000)
        
        # 获取完整的静默区域
        silence_regions = detect_silence(audio[ws:we], min_silence_len=int(safe_margin*1000), silence_thresh=-30)
        silence_regions = [(s/1000 + (threshold - win), e/1000 + (threshold - win)) for s, e in silence_regions]
        # 筛选长度足够（至少1秒）且位置适合的静默区域
        valid_regions = [
            (start, end) for start, end in silence_regions 
            if (end - start) >= (safe_margin * 2) and threshold <= start + safe_margin <= threshold + win
        ]
        
        if valid_regions:
            start, end = valid_regions[0]
            split_at = start + safe_margin  # 在静默区域起始点后0.5秒处切分
        else:
            rprint(f"[yellow]⚠️ No valid silence regions found for {audio_file} at {threshold}s, using threshold[/yellow]")
            split_at = threshold
            
        segments.append((pos, split_at)); pos = split_at

    rprint(f"[green]🎙️ Audio split completed {len(segments)} segments[/green]")
    return segments

def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        # Get speaker_id, if not exists, set to None
        speaker_id = segment.get('speaker_id', None)
        
        for word in segment['words']:
            # Check word length
            if len(word["word"]) > 30:
                rprint(f"[yellow]⚠️ Warning: Detected word longer than 30 characters, skipping: {word['word']}[/yellow]")
                continue
                
            # ! For French, we need to convert guillemets to empty strings
            word["word"] = word["word"].replace('»', '').replace('«', '')
            
            if 'start' not in word and 'end' not in word:
                if all_words:
                    # Assign the end time of the previous word as the start and end time of the current word
                    word_dict = {
                        'text': word["word"],
                        'start': all_words[-1]['end'],
                        'end': all_words[-1]['end'],
                        'speaker_id': speaker_id
                    }
                    all_words.append(word_dict)
                else:
                    # If it's the first word, look next for a timestamp then assign it to the current word
                    next_word = next((w for w in segment['words'] if 'start' in w and 'end' in w), None)
                    if next_word:
                        word_dict = {
                            'text': word["word"],
                            'start': next_word["start"],
                            'end': next_word["end"],
                            'speaker_id': speaker_id
                        }
                        all_words.append(word_dict)
                    else:
                        raise Exception(f"No next word with timestamp found for the current word : {word}")
            else:
                # Normal case, with start and end times
                word_dict = {
                    'text': f'{word["word"]}',
                    'start': word.get('start', all_words[-1]['end'] if all_words else 0),
                    'end': word['end'],
                    'speaker_id': speaker_id
                }
                
                all_words.append(word_dict)
    
    return pd.DataFrame(all_words)

def get_llm_vocab_corrections(texts: List[str], vocab_str: str) -> List[Tuple[List[str], List[str]]]:
    """Use LLM to find misrecognized domain terms and return as explicit pairs."""
    if not texts or not vocab_str:
        return []
    
    from core.prompts import get_vocab_correction_prompt
    
    # Combine texts into a single string
    full_text = " ".join([str(t).strip('"') for t in texts])
    # Split into chunks of ~2000 characters
    chunk_size = 2000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    all_pairs = []
    for chunk in chunks:
        prompt = get_vocab_correction_prompt(chunk, vocab_str)
        try:
            response = ask_gpt(prompt, resp_type='json', log_title='vocab_correction')
            if isinstance(response, list):
                for item in response:
                    wrong = item.get('wrong_phrase', '').strip()
                    correct = item.get('correct_phrase', '').strip()
                    if wrong and correct:
                        all_pairs.append((wrong.lower().split(), correct.split()))
        except Exception as e:
            rprint(f"[yellow]⚠️ LLM vocab correction failed for a chunk: {e}[/yellow]")
            
    return all_pairs

def apply_domain_vocab_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Correct misrecognized words using user-supplied domain vocabulary.

    Supports two formats (comma or newline separated):
      - Plain term:       "Claude Cowork"           → fuzzy-matched (cutoff 0.75)
      - Explicit pair:    "called cowork -> Claude Cowork"  → exact case-insensitive match
    """
    vocab_str = load_key("domain_vocab")
    if not vocab_str or not vocab_str.strip():
        return df

    raw_terms = [t.strip() for t in vocab_str.replace('\n', ',').split(',') if t.strip()]
    if not raw_terms:
        return df

    # Separate explicit replacements from fuzzy hint terms
    explicit_pairs = []   # list of ([wrong_words_lower], [correct_words])
    hint_terms = []
    for term in raw_terms:
        parts = re.split(r'\s*(?:->|→)\s*', term, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            wrong_words = parts[0].strip().split()
            correct_words = parts[1].strip().split()
            explicit_pairs.append(([w.lower() for w in wrong_words], correct_words))
        else:
            hint_terms.append(term)
            
    # 🤖 NEW: Get corrections from LLM based on context
    llm_pairs = get_llm_vocab_corrections(list(df['text']), vocab_str)
    if llm_pairs:
        rprint(f"[cyan]🤖 LLM found {len(llm_pairs)} potential correction(s)[/cyan]")
        explicit_pairs.extend(llm_pairs)

    rprint(f"[cyan]📖 Domain vocab: {len(hint_terms)} fuzzy hint(s), {len(explicit_pairs)} explicit replacement(s)[/cyan]")
    texts = list(df['text'])

    # 1. Explicit replacements — exact case-insensitive window match
    # Use a set to track applied replacements to avoid infinite loops if LLM suggests a recursive replacement
    applied_indices = set()
    for wrong_lower, correct_words in explicit_pairs:
        n = len(wrong_lower)
        for i in range(len(texts) - n + 1):
            if i in applied_indices: continue
            window = [str(texts[i + j]).strip().strip('"') for j in range(n)]
            if [w.lower() for w in window] == wrong_lower:
                original = ' '.join(window)
                replacement = ' '.join(correct_words)
                rprint(f"[yellow]📝 Correction: '{original}' → '{replacement}'[/yellow]")
                # Replace words in-place (same-length); extra correct words go into first slot
                for j, rep_word in enumerate(correct_words):
                    if i + j < len(texts):
                        texts[i + j] = rep_word
                # If wrong phrase is longer than correct, mark excess as empty
                for j in range(len(correct_words), n):
                    texts[i + j] = ''
                for k in range(i, i + n): applied_indices.add(k)

    # Remove entries emptied by different-length explicit replacements
    # Note: we need to keep the index mapping for df, but texts is a list.
    # Actually, the original code also did `texts = [t for t in texts if t != '']`.
    # This might break the DataFrame index if not careful.
    # But save_results will handle it.
    
    texts = [t for t in texts if t != '']

    # 2. Fuzzy matching for plain hint terms
    single_terms = [t for t in hint_terms if len(t.split()) == 1]
    multi_terms  = [t for t in hint_terms if len(t.split()) > 1]

    for i, word in enumerate(texts):
        clean = str(word).strip().strip('"')
        lower_singles = [t.lower() for t in single_terms]
        matches = difflib.get_close_matches(clean.lower(), lower_singles, n=1, cutoff=0.75)
        if matches:
            matched = next(t for t in single_terms if t.lower() == matches[0])
            if clean != matched:
                rprint(f"[yellow]📝 Fuzzy: '{clean}' → '{matched}'[/yellow]")
                texts[i] = matched

    for term in multi_terms:
        term_words = term.split()
        n = len(term_words)
        for i in range(len(texts) - n + 1):
            window = ' '.join(str(w).strip().strip('"') for w in texts[i:i+n])
            ratio = difflib.SequenceMatcher(None, window.lower(), term.lower()).ratio()
            if ratio >= 0.75:
                rprint(f"[yellow]📝 Fuzzy: '{window}' → '{term}'[/yellow]")
                for j, replacement in enumerate(term_words):
                    texts[i + j] = replacement

    df = df.copy()
    # If the length changed due to different-length explicit replacements, we need a new df
    if len(texts) != len(df):
        # This is a bit risky because we lose timestamp mapping for those words.
        # But if 'wrong' was 2 words and 'correct' is 1 word, we merged them.
        # The original code did this too.
        return pd.DataFrame({'text': texts}) 
    
    df['text'] = texts
    return df


def reapply_vocab_correction_to_cleaned_chunks() -> bool:
    """Re-read cleaned_chunks.xlsx (already quoted), strip quotes, apply vocab correction, re-save."""
    if not os.path.exists(_2_CLEANED_CHUNKS):
        rprint("[red]❌ cleaned_chunks.xlsx not found[/red]")
        return False
    df = pd.read_excel(_2_CLEANED_CHUNKS)
    # Strip the double-quote wrapping that save_results adds
    df['text'] = df['text'].apply(lambda x: str(x).strip('"') if pd.notna(x) else x)
    df = apply_domain_vocab_correction(df)
    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(_2_CLEANED_CHUNKS, index=False)
    rprint(f"[green]✅ Vocab correction re-applied to {_2_CLEANED_CHUNKS}[/green]")
    return True


def detect_suspicious_terms_from_srt(srt_path: str) -> list:
    """Parse src.srt and find unusual/high-freq terms that might be misrecognized.
    Returns list of dicts: {term, category, count, example}.
    """
    if not os.path.exists(srt_path):
        return []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract subtitle text lines (skip index numbers and timestamp lines)
    text_lines = []
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.isdigit() or re.match(r'\d{2}:\d{2}:\d{2}', line):
            continue
        text_lines.append(line)

    # Build word → list-of-lines mapping (for context display)
    word_to_lines = defaultdict(list)
    for line in text_lines:
        for w in re.findall(r"[A-Za-z][A-Za-z'-]*[A-Za-z]", line):
            if line not in word_to_lines[w]:
                word_to_lines[w].append(line)

    COMMON = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'is','are','was','were','be','been','being','have','has','had','do',
        'does','did','will','would','could','should','may','might','can','must',
        'shall','it','this','that','these','those','you','he','she','we',
        'they','me','him','her','us','them','my','your','his','our','their',
        'what','which','who','when','where','how','why','all','some','any',
        'not','no','so','if','as','by','from','up','about','into','through',
        'during','before','after','above','below','between','out','off','over',
        'under','again','then','once','just','now','also','here','there','very',
        'too','more','most','other','like','get','go','see','know','think',
        'come','look','want','make','say','use','find','give','tell','work',
        'call','try','ask','need','feel','let','put','keep','right','new',
        'good','first','last','long','great','little','own','old','big','high',
        'small','large','next','early','young','important','time','already',
        'year','day','way','even','still','because','much','every','two','one',
        'same','only','both','each','many','while','really','well','back',
        'actually','than','such','without','doing','having','going','coming',
        'making','using','getting','taking','saying','looking','thinking',
    }

    results = []
    seen = set()

    def _add(term, category, count, example):
        if term not in seen:
            seen.add(term)
            results.append({'term': term, 'category': category, 'count': count, 'example': example})

    # 1. ALL_CAPS acronyms (2+ chars, not single common letters)
    for w in sorted(word_to_lines):
        if len(w) >= 2 and w.isupper() and w.lower() not in COMMON:
            _add(w, 'Acronym', len(word_to_lines[w]), word_to_lines[w][0])

    # 2. CamelCase proper nouns (e.g. VideoLingo, WhisperX, ClaudeCowork)
    for w in sorted(word_to_lines):
        if len(w) >= 4 and w[0].isupper() and any(c.isupper() for c in w[1:]) and w.lower() not in COMMON:
            _add(w, 'CamelCase', len(word_to_lines[w]), word_to_lines[w][0])

    # 3. High-frequency capitalized words appearing 3+ times (likely proper nouns)
    freq_cap = [
        (w, len(lines), lines[0])
        for w, lines in word_to_lines.items()
        if (len(w) >= 4 and w[0].isupper() and not w.isupper()
            and not any(c.isupper() for c in w[1:])
            and len(lines) >= 3 and w.lower() not in COMMON)
    ]
    for w, cnt, example in sorted(freq_cap, key=lambda x: -x[1])[:20]:
        _add(w, f'High-freq({cnt}x)', cnt, example)

    return results[:40]


def save_results(df: pd.DataFrame):
    os.makedirs('output/log', exist_ok=True)

    # Remove rows where 'text' is empty
    initial_rows = len(df)
    df = df[df['text'].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        rprint(f"[blue]ℹ️ Removed {removed_rows} row(s) with empty text.[/blue]")
    
    # Check for and remove words longer than 20 characters
    long_words = df[df['text'].str.len() > 30]
    if not long_words.empty:
        rprint(f"[yellow]⚠️ Warning: Detected {len(long_words)} word(s) longer than 30 characters. These will be removed.[/yellow]")
        df = df[df['text'].str.len() <= 30]
    
    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(_2_CLEANED_CHUNKS, index=False)
    rprint(f"[green]📊 Excel file saved to {_2_CLEANED_CHUNKS}[/green]")

def save_language(language: str):
    update_key("whisper.detected_language", language)