"""
Parse an uploaded SRT / VTT subtitle file and write word-level data to
cleaned_chunks.xlsx, bypassing WhisperX transcription entirely.

Supported formats:
  - SRT  (standard sentence-level, including YouTube auto-generated)
  - VTT  (standard + YouTube karaoke-style with <HH:MM:SS.mmm><c>word</c>)

Word timestamps are either taken from VTT word-level cues or distributed
linearly within each subtitle line.
"""

import os
import re
import pandas as pd
from rich import print as rprint
from core.utils import *
from core.utils.models import *
from core.asr_backend.audio_preprocess import apply_domain_vocab_correction


# ─────────────────────────── Helpers ────────────────────────────────────────

def _ts_to_sec(ts: str) -> float:
    """Convert HH:MM:SS.mmm or HH:MM:SS,mmm to float seconds."""
    ts = ts.strip().replace(',', '.')
    parts = ts.split(':')
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return float(parts[0]) * 60 + float(parts[1])


def _strip_markup(text: str) -> str:
    """Remove VTT/HTML markup tags and decode basic HTML entities."""
    text = re.sub(r'<\d{2}:\d{2}:\d{2}[.,]\d{3}>', '', text)  # <00:00:01.000>
    text = re.sub(r'<[^>]+>', '', text)                         # <c>, </c>, <b> …
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text.strip()


# ─────────────────────────── VTT word-level cues ────────────────────────────

def _parse_vtt_words(text_lines: list, cue_start: float, cue_end: float) -> list:
    """
    Try to extract word-level entries from VTT cue text.
    YouTube auto-subs embed timestamps as: <HH:MM:SS.mmm><c>word</c>
    Returns list of (word_str, start_s, end_s) or empty list if not word-level.
    """
    raw = ' '.join(text_lines)
    # Pattern: <HH:MM:SS.mmm><c> word </c>  (the <c> body is the word)
    pattern = r'<(\d{2}:\d{2}:\d{2}[.,]\d{3})><c>([^<]+)</c>'
    hits = re.findall(pattern, raw)
    if not hits:
        return []

    words = []
    for i, (ts_str, word) in enumerate(hits):
        word = word.strip()
        if not word:
            continue
        w_start = _ts_to_sec(ts_str)
        # end = start of next word, or cue end for the last word
        w_end = _ts_to_sec(hits[i + 1][0]) if i + 1 < len(hits) else cue_end
        words.append((word, w_start, w_end))
    return words


# ─────────────────────────── File parsers ───────────────────────────────────

def _deduplicate_block_lines(text_lines: list) -> list:
    """
    Remove scrolling/karaoke redundant lines within a single VTT cue block.
    For example, if a line is a prefix/substring of another longer line in the same block,
    it is considered redundant.
    """
    cleaned = []
    for line in text_lines:
        line_clean = _strip_markup(line)
        if not line_clean:
            continue
        cleaned.append(line_clean)
        
    final_lines = []
    for i, line in enumerate(cleaned):
        is_sub = False
        for j, other in enumerate(cleaned):
            if i != j and len(other) > len(line):
                line_norm = "".join(line.lower().split())
                other_norm = "".join(other.lower().split())
                if line_norm in other_norm:
                    is_sub = True
                    break
        if not is_sub:
            final_lines.append(line)
    return final_lines

def _merge_vtt_rolling_duplicates(sentences: list) -> list:
    """
    Merge rolling/karaoke-style duplicate blocks in VTT.
    If block_i is a substring of block_i+1 (or vice versa) and they are close in time,
    we merge them by extending the next block's start to the current block's start,
    and dropping the redundant block.
    """
    if not sentences:
        return []
        
    merged = []
    i = 0
    n = len(sentences)
    while i < n:
        curr = list(sentences[i])  # [start, end, text, raw_lines]
        
        while i + 1 < n:
            nxt = sentences[i + 1]
            curr_text_norm = "".join(curr[2].lower().split())
            nxt_text_norm = "".join(nxt[2].lower().split())
            
            time_gap = nxt[0] - curr[1]
            if time_gap <= 1.5 and (curr_text_norm in nxt_text_norm or nxt_text_norm in curr_text_norm):
                # Keep the longer text
                if len(nxt_text_norm) >= len(curr_text_norm):
                    curr[2] = nxt[2]
                    curr[3] = nxt[3]
                curr[1] = nxt[1]  # Update end time
                i += 1  # Skip the next block
                rprint(f"[yellow]🔄 Merged rolling duplicate block: '{curr_text_norm}' into '{nxt_text_norm}'[/yellow]")
            else:
                break
                
        merged.append((curr[0], curr[1], curr[2], curr[3]))
        i += 1
        
    return merged

def _parse_vtt(file_path: str) -> list:
    """
    Parse VTT into list of (start_s, end_s, text_str) sentences.
    Handles YouTube karaoke duplicates by deduplicating internal lines and merging rolling blocks.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sentences = []
    seen = set()

    for block in re.split(r'\n\s*\n', content):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        if any(lines[0].startswith(k) for k in ('WEBVTT', 'NOTE', 'STYLE', 'REGION')):
            continue

        # Find timestamp line
        ts_match = ts_idx = None
        for i, line in enumerate(lines):
            m = re.match(r'([\d:.,]+)\s+-->\s+([\d:.,]+)', line)
            if m:
                ts_match, ts_idx = m, i
                break
        if ts_match is None:
            continue

        start = _ts_to_sec(ts_match.group(1))
        end   = _ts_to_sec(ts_match.group(2))
        
        # Deduplicate internal karaoke scrolling lines
        block_text_lines = lines[ts_idx + 1:]
        deduped_block_lines = _deduplicate_block_lines(block_text_lines)
        text = ' '.join(deduped_block_lines)
        # Remove music/sound markers like [music], (laughter)
        text = re.sub(r'\[[^\]]+\]|\([^)]+\)|（[^）]+）', '', text).strip()
        if not text:
            continue

        key = (round(start, 1), text)
        if key not in seen:
            seen.add(key)
            sentences.append((start, end, text, deduped_block_lines))

    # Merge cross-block rolling duplicates
    sentences = _merge_vtt_rolling_duplicates(sentences)
    return sentences


def _parse_srt(file_path: str) -> list:
    """Parse SRT into list of (start_s, end_s, text_str, [raw_text_line])."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sentences = []
    for block in re.split(r'\n\s*\n', content.strip()):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        ts_match = ts_idx = None
        for i, line in enumerate(lines):
            m = re.match(r'([\d:,]+)\s+-->\s+([\d:,]+)', line)
            if m:
                ts_match, ts_idx = m, i
                break
        if ts_match is None:
            continue
        start = _ts_to_sec(ts_match.group(1))
        end   = _ts_to_sec(ts_match.group(2))
        raw_text_lines = lines[ts_idx + 1:]
        text = _strip_markup(' '.join(raw_text_lines))
        # Remove music/sound markers like [music], (laughter)
        text = re.sub(r'\[[^\]]+\]|\([^)]+\)|（[^）]+）', '', text).strip()
        if text:
            sentences.append((start, end, text, raw_text_lines))

    return sentences


# ─────────────────────────── Sentence → word rows ───────────────────────────

def _sentence_to_rows(start: float, end: float, text: str,
                      raw_lines: list, lang: str) -> list:
    """
    Convert a single subtitle sentence to word-level rows.
    Prefers VTT word-level timestamps; falls back to linear distribution.
    """
    # Try VTT word-level timestamps first
    word_entries = _parse_vtt_words(raw_lines, start, end)
    if word_entries:
        # Filter out music/sound markers from word entries
        filtered_entries = []
        for w, s, e in word_entries:
            cleaned_w = re.sub(r'\[[^\]]+\]|\([^)]+\)|（[^）]+）', '', w).strip()
            if cleaned_w:
                filtered_entries.append((cleaned_w, s, e))
        return [{'text': w, 'start': s, 'end': e, 'speaker_id': None}
                for w, s, e in filtered_entries]

    # Linear distribution
    is_cjk = any(c in lang for c in ('zh', 'ja', 'ko'))
    tokens = [c for c in text if c.strip()] if is_cjk else text.split()
    if not tokens:
        return []

    n = len(tokens)
    step = (end - start) / n
    return [
        {'text': tok,
         'start': round(start + i * step, 3),
         'end':   round(start + (i + 1) * step, 3),
         'speaker_id': None}
        for i, tok in enumerate(tokens)
    ]


# ─────────────────────────── Public API ─────────────────────────────────────

def use_subtitle_file(subtitle_path: str, lang: str = 'en') -> bool:
    """
    Parse an SRT/VTT subtitle file and write word-level data to
    cleaned_chunks.xlsx (bypassing WhisperX). Returns True on success.
    """
    rprint(f"[cyan]📄 Parsing subtitle file: {subtitle_path}  (lang={lang})[/cyan]")

    ext = os.path.splitext(subtitle_path)[1].lower()
    if ext == '.vtt':
        sentences = _parse_vtt(subtitle_path)
    elif ext == '.srt':
        sentences = _parse_srt(subtitle_path)
    else:
        rprint(f"[red]❌ Unsupported subtitle format: {ext}[/red]")
        return False

    if not sentences:
        rprint("[red]❌ No subtitle content found in file[/red]")
        return False

    rprint(f"[green]✅ Parsed {len(sentences)} subtitle segments[/green]")

    # Build word-level DataFrame
    rows = []
    for start, end, text, raw_lines in sentences:
        rows.extend(_sentence_to_rows(start, end, text, raw_lines, lang))

    if not rows:
        rprint("[red]❌ Empty word list after parsing[/red]")
        return False

    df = pd.DataFrame(rows)

    # Apply domain vocab correction
    df = apply_domain_vocab_correction(df)

    # Save detected language to config
    lang_short = re.split(r'[-_]', lang)[0]   # 'en-US' → 'en'
    update_key("whisper.language", lang_short)

    # Save to cleaned_chunks.xlsx (same schema as WhisperX output)
    os.makedirs('output/log', exist_ok=True)
    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(_2_CLEANED_CHUNKS, index=False)
    rprint(f"[green]📊 Saved {len(df)} word rows to {_2_CLEANED_CHUNKS}[/green]")
    return True
