import pandas as pd
import os
import re
from rich.panel import Panel
from rich.console import Console
import autocorrect_py as autocorrect
from core.utils import *
from core.utils.models import *
console = Console()

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ('src_subs_for_audio.srt', ['Source']),
    ('trans_subs_for_audio.srt', ['Translation'])
]

def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"

def remove_punctuation(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []
    
    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)
    
    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))
    
    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print("Position markers: " + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2)))))
    print(f"Difference indices: {diff_positions}")

def get_sentence_timestamps(df_words, df_sentences):
    time_stamp_list = []
    
    # Build complete string and position mapping
    full_words_str = ''
    position_to_word_idx = {}
    
    for idx, word in enumerate(df_words['text']):
        clean_word = remove_punctuation(word.lower())
        start_pos = len(full_words_str)
        full_words_str += clean_word
        for pos in range(start_pos, len(full_words_str)):
            position_to_word_idx[pos] = idx
    
    current_pos = 0
    for idx, sentence in df_sentences['Source'].items():
        clean_sentence = remove_punctuation(sentence.lower()).replace(" ", "")
        sentence_len = len(clean_sentence)
        
        match_found = False
        while current_pos <= len(full_words_str) - sentence_len:
            if full_words_str[current_pos:current_pos+sentence_len] == clean_sentence:
                start_word_idx = position_to_word_idx[current_pos]
                end_word_idx = position_to_word_idx[current_pos + sentence_len - 1]
                
                time_stamp_list.append((
                    float(df_words['start'][start_word_idx]),
                    float(df_words['end'][end_word_idx])
                ))
                
                current_pos += sentence_len
                match_found = True
                break
            current_pos += 1
            
        if not match_found:
            # 1. First Fallback: Adaptive Fuzzy Match near the current position
            import difflib
            search_window = full_words_str[current_pos : current_pos + sentence_len * 2 + 50]
            
            best_ratio = 0.0
            best_match_start = -1
            best_match_len = -1
            matcher = difflib.SequenceMatcher(None, clean_sentence)
            
            # Scan positions in search_window allowing length variation
            for length in range(max(1, sentence_len - 15), min(len(search_window), sentence_len + 15) + 1):
                for start in range(len(search_window) - length + 1):
                    sub_str = search_window[start:start+length]
                    matcher.set_seq2(sub_str)
                    ratio = matcher.quick_ratio()
                    if ratio > best_ratio:
                        ratio = matcher.ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match_start = start
                            best_match_len = length
                            
            if best_ratio > 0.6 and best_match_start != -1:
                matched_pos_in_full = current_pos + best_match_start
                start_word_idx = position_to_word_idx.get(matched_pos_in_full, len(df_words) - 1)
                end_word_idx = position_to_word_idx.get(matched_pos_in_full + best_match_len - 1, len(df_words) - 1)
                
                time_stamp_list.append((
                    float(df_words['start'][start_word_idx]),
                    float(df_words['end'][end_word_idx])
                ))
                
                current_pos = matched_pos_in_full + best_match_len
                match_found = True
                print(f"✨ Fuzzy Match Success! Sentence '{sentence}' matched with similarity {best_ratio:.2f}")
                
        if not match_found:
            # 2. Second Fallback: Sequential word fallback to ensure 100% crash prevention
            start_word_idx = position_to_word_idx.get(min(current_pos, len(full_words_str) - 1), len(df_words) - 1)
            estimated_word_count = len(sentence.split())
            if estimated_word_count == 0:
                estimated_word_count = max(1, len(sentence) // 5)
                
            end_word_idx = min(start_word_idx + estimated_word_count - 1, len(df_words) - 1)
            
            start_time = float(df_words['start'][start_word_idx])
            end_time = float(df_words['end'][end_word_idx])
            
            time_stamp_list.append((start_time, end_time))
            
            char_len = 0
            for w_idx in range(start_word_idx, end_word_idx + 1):
                char_len += len(remove_punctuation(df_words['text'][w_idx].lower()))
            
            current_pos = min(current_pos + char_len, len(full_words_str))
            if current_pos >= len(full_words_str):
                current_pos = len(full_words_str)
            else:
                current_pos += 1
                
            print(f"⚠️ Warning: Using sequential word fallback for '{sentence}' at timestamp {start_time:.2f}s - {end_time:.2f}s")
            
    return time_stamp_list

def align_timestamp(df_text, df_translate, subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    # Assign an ID to each word in df_text['text'] and create a new DataFrame
    words = df_text['text'].str.split(expand=True).stack().reset_index(level=1, drop=True).reset_index()
    words.columns = ['id', 'word']
    words['id'] = words['id'].astype(int)

    # Process timestamps ⏰
    time_stamp_list = get_sentence_timestamps(df_text, df_translate)
    df_trans_time['timestamp'] = time_stamp_list
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])

    # Remove gaps 🕳️ (but keep a tiny gap of 0.15s for breathing room to make both subtitles and dubbing comfortable)
    for i in range(len(df_trans_time)-1):
        delta_time = df_trans_time.loc[i+1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            tiny_gap = 0.15
            if delta_time > tiny_gap:
                df_trans_time.at[i, 'timestamp'] = (df_trans_time.loc[i, 'timestamp'][0], df_trans_time.loc[i+1, 'timestamp'][0] - tiny_gap)
            else:
                df_trans_time.at[i, 'timestamp'] = (df_trans_time.loc[i, 'timestamp'][0], df_trans_time.loc[i+1, 'timestamp'][0])

    # Convert start and end timestamps to SRT format
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    # Polish subtitles: replace punctuation in Translation if for_display
    if for_display:
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(lambda x: re.sub(r'[，。]', ' ', str(x)).strip() if pd.notna(x) else "")

    # Output subtitles 📜
    # When 'Corrected_Source' exists, use it in place of 'Source' for display (but never mutate the df)
    def generate_subtitle_string(df, columns):
        def _display_col(col):
            return 'Corrected_Source' if col == 'Source' and 'Corrected_Source' in df.columns else col
        
        def _get_val(row, col):
            val = row[col]
            if pd.isna(val):
                return ""
            return str(val).strip()

        disp = [_display_col(c) for c in columns]
        return ''.join([f"{i+1}\n{row['timestamp']}\n{_get_val(row, disp[0])}\n{_get_val(row, disp[1]) if len(disp) > 1 else ''}\n\n" for i, row in df.iterrows()]).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)
    
    return df_trans_time

# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。').strip('，')
    return autocorrect.format(cleaned)

def align_timestamp_main():
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.read_excel(_5_SPLIT_SUB)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # for audio
    df_translate_for_audio = pd.read_excel(_5_REMERGED) # use remerged file to avoid unmatched lines when dubbing
    df_translate_for_audio['Translation'] = df_translate_for_audio['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate_for_audio, AUDIO_SUBTITLE_OUTPUT_CONFIGS, _AUDIO_DIR)
    console.print(Panel(f"[bold green]🎉📝 Audio subtitles generation completed! Please check in the `{_AUDIO_DIR}` folder 👀[/bold green]"))
    

if __name__ == '__main__':
    align_timestamp_main()