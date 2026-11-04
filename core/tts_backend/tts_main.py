from core.tts_backend.single_attempt_scope import single_attempt_active, single_attempt_call
from core.utils.cancel_utils import check_task_cancelled, is_task_cancelled, sleep_cancellable, TaskCancelledError
import os
import pandas as pd
import re
import json
import time
import hashlib
import threading
import subprocess
from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.utils import rprint
from core.tts_backend.doubao_tts import doubao_tts
from core.tts_backend.edge_tts import edge_tts
from core.prompts import get_correct_text_prompt
from core.utils import *

_SEGMENT_HASH_LOCK = threading.RLock()
_SEGMENT_HASH_FILENAME = ".segment_text_hashes.json"

def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    text = "" if text is None else str(text)
    chars_to_remove = ['&', '®', '™', '©']
    for char in chars_to_remove:
        text = text.replace(char, '')
    # Remove music/sound markers like [music], (laughter)
    text = re.sub(r'\[[^\]]+\]|\([^)]+\)|（[^）]+）', '', text)
    return text.strip()


def compute_tts_text_hash(text: str) -> str:
    normalized = clean_text_for_tts(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _extract_segment_slot(file_path: str):
    """Identify (audio_dir, bucket, seg_key) for tmp/{num}_{idx}_temp.wav or segs/{num}_{idx}.wav."""
    if not file_path:
        return None
    norm_path = os.path.abspath(file_path)
    fname = os.path.basename(norm_path)
    parent = os.path.dirname(norm_path)
    parent_name = os.path.basename(parent)
    audio_dir = os.path.dirname(parent)

    if parent_name == "tmp":
        m = re.match(r"^(\d+_\d+)_temp\.wav$", fname)
        if m:
            return audio_dir, "tmp", m.group(1)
    elif parent_name == "segs":
        m = re.match(r"^(\d+_\d+)\.wav$", fname)
        if m:
            return audio_dir, "segs", m.group(1)
    return None


def _load_hash_store(audio_dir: str) -> dict:
    meta_path = os.path.join(audio_dir, _SEGMENT_HASH_FILENAME)
    if not os.path.exists(meta_path):
        return {"tmp": {}, "segs": {}}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("tmp", {})
            data.setdefault("segs", {})
            return data
    except Exception:
        pass
    return {"tmp": {}, "segs": {}}


def _save_hash_store(audio_dir: str, data: dict) -> None:
    os.makedirs(audio_dir, exist_ok=True)
    meta_path = os.path.join(audio_dir, _SEGMENT_HASH_FILENAME)
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_meta, meta_path)


def ensure_segment_hashes_initialized(tasks_df, audio_dir: str = "output/audio") -> None:
    """Seed baseline text hashes for existing segment wav files once so unmodified segments stay cached
    while any subsequent text edit is detected even if wav files were not manually deleted."""
    import ast
    abs_audio_dir = os.path.abspath(audio_dir)
    meta_path = os.path.join(abs_audio_dir, _SEGMENT_HASH_FILENAME)
    with _SEGMENT_HASH_LOCK:
        if os.path.exists(meta_path) or tasks_df is None:
            return
        tmp_dir = os.path.join(abs_audio_dir, "tmp")
        segs_dir = os.path.join(abs_audio_dir, "segs")
        if not os.path.exists(tmp_dir) and not os.path.exists(segs_dir):
            return
        store = {"tmp": {}, "segs": {}}
        for _, row in tasks_df.iterrows():
            raw_lines = row.get("lines")
            if raw_lines is None or (isinstance(raw_lines, float) and str(raw_lines) == "nan"):
                continue
            try:
                lines_list = ast.literal_eval(raw_lines) if isinstance(raw_lines, str) else list(raw_lines)
            except Exception:
                continue
            num = row.get("number")
            for idx_l, ln in enumerate(lines_list):
                line_text = ln.get("text", "") if isinstance(ln, dict) else str(ln)
                h = compute_tts_text_hash(line_text)
                seg_key = f"{num}_{idx_l}"
                if os.path.exists(os.path.join(tmp_dir, f"{seg_key}_temp.wav")):
                    store["tmp"][seg_key] = h
                if os.path.exists(os.path.join(segs_dir, f"{seg_key}.wav")):
                    store["segs"][seg_key] = h
        if store["tmp"] or store["segs"]:
            _save_hash_store(abs_audio_dir, store)


def record_segment_text_hash(file_path: str, text: str) -> None:
    slot = _extract_segment_slot(file_path)
    if not slot:
        return
    audio_dir, bucket, seg_key = slot
    h = compute_tts_text_hash(text)
    with _SEGMENT_HASH_LOCK:
        store = _load_hash_store(audio_dir)
        store[bucket][seg_key] = h
        _save_hash_store(audio_dir, store)


def copy_segment_text_hash(src_path: str, dst_path: str) -> None:
    src_slot = _extract_segment_slot(src_path)
    dst_slot = _extract_segment_slot(dst_path)
    if not src_slot or not dst_slot:
        return
    src_dir, src_bucket, src_key = src_slot
    dst_dir, dst_bucket, dst_key = dst_slot
    with _SEGMENT_HASH_LOCK:
        src_store = _load_hash_store(src_dir)
        h = src_store.get(src_bucket, {}).get(src_key)
        if not h:
            return
        dst_store = src_store if src_dir == dst_dir else _load_hash_store(dst_dir)
        dst_store[dst_bucket][dst_key] = h
        _save_hash_store(dst_dir, dst_store)


def remove_segment_text_hashes_for_numbers(numbers: list, audio_dir: str = "output/audio") -> None:
    if not numbers:
        return
    abs_audio_dir = os.path.abspath(audio_dir)
    meta_path = os.path.join(abs_audio_dir, _SEGMENT_HASH_FILENAME)
    with _SEGMENT_HASH_LOCK:
        if not os.path.exists(meta_path):
            return
        store = _load_hash_store(abs_audio_dir)
        prefixes = tuple(f"{num}_" for num in numbers)
        exact_keys = {str(num) for num in numbers}
        changed = False
        for bucket in ("tmp", "segs"):
            b_map = store.get(bucket, {})
            for k in list(b_map.keys()):
                if k in exact_keys or k.startswith(prefixes):
                    del b_map[k]
                    changed = True
        if changed:
            _save_hash_store(abs_audio_dir, store)


def is_segment_text_hash_valid(file_path: str, text: str) -> bool:
    slot = _extract_segment_slot(file_path)
    if not slot:
        return True
    audio_dir, bucket, seg_key = slot
    meta_path = os.path.join(audio_dir, _SEGMENT_HASH_FILENAME)
    with _SEGMENT_HASH_LOCK:
        if not os.path.exists(meta_path):
            return True
        store = _load_hash_store(audio_dir)
        stored_hash = store.get(bucket, {}).get(seg_key)
    if stored_hash is None:
        return False
    return stored_hash == compute_tts_text_hash(text)


def is_audio_valid(file_path: str, text: str) -> bool:
    """Validate if the generated audio is healthy, valid, and matches the current text."""
    if not is_segment_text_hash_valid(file_path, text):
        rprint(f"[yellow]⚠️ Audio cache text mismatch detected for <{file_path}>, invalidating stale segment...[/yellow]")
        return False

    cleaned_text = re.sub(r'[^\w\s]', '', str(text or '')).strip()
    # For very short texts or single characters, it may generate tiny audio or silence
    if not cleaned_text or len(cleaned_text) <= 1:
        return True
        
    if not os.path.exists(file_path):
        return False
        
    if os.path.getsize(file_path) <= 100:
        return False
        
    duration = get_audio_duration(file_path)
    if duration <= 0:
        return False
        
    # Critical sanity check: if the text is long but the audio is extremely short, it is truncated/corrupted.
    # 28 Chinese characters in 0.4s is physically impossible.
    # As a safe threshold: if the cleaned text length is >= 3, duration must be > 0.5 seconds.
    if len(cleaned_text) >= 3 and duration <= 0.5:
        rprint(f"[yellow]⚠️ Audio sanity check failed: text length {len(cleaned_text)} is long but duration is only {duration:.2f}s for <{file_path}>[/yellow]")
        return False
        
    # Decoding validation: verify if ffmpeg can decode it without errors
    try:
        cmd = ['ffmpeg', '-y', '-i', file_path, '-f', 'null', '-']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        if result.returncode != 0:
            rprint(f"[yellow]⚠️ Audio validation failed: ffmpeg decode error for <{file_path}>[/yellow]")
            return False
    except Exception as e:
        rprint(f"[yellow]⚠️ Audio validation failed: ffmpeg validation timed out or crashed for <{file_path}>: {e}[/yellow]")
        return False
        
    return True

def tts_main(
    text,
    save_as,
    number=None,
    task_df=None,
    gender=None,
    voice=None,
    section_id=None,
    context_texts=None,
    return_words=False,
):
    check_task_cancelled()
    raw_input_text = "" if text is None else str(text)
    text = clean_text_for_tts(raw_input_text)
    # Check if text is empty or single character, single character voiceovers are prone to bugs
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if not cleaned_text or len(cleaned_text) <= 1:
        silence = AudioSegment.silent(duration=100, frame_rate=24000)  # 100ms = 0.1s at 24kHz
        silence.export(save_as, format="wav")
        record_segment_text_hash(save_as, raw_input_text)
        rprint(f"Created silent audio for empty/single-char text: {save_as}")
        return {"ok": True, "words": [], "sample_rate": 24000, "duration": 0.1} if return_words else None
    
    # Resolve gender and custom voice from task_df if not explicitly provided
    if task_df is not None and number is not None:
        try:
            matching_rows = task_df[task_df['number'] == number]
            if not matching_rows.empty:
                if gender is None and 'gender' in matching_rows.columns:
                    gender = matching_rows['gender'].values[0]
                if not voice and 'voice' in matching_rows.columns:
                    raw_v = matching_rows['voice'].values[0]
                    if pd.notna(raw_v) and str(raw_v).strip():
                        voice = str(raw_v).strip()
        except Exception:
            pass
    if voice and (pd.isna(voice) or not str(voice).strip()):
        voice = None
    
    # Skip if file exists, is non-empty and is valid (only when word timestamps aren't explicitly needed)
    if not return_words and os.path.exists(save_as) and os.path.getsize(save_as) > 0:
        if is_audio_valid(save_as, raw_input_text):
            return
        else:
            rprint(f"[yellow]⚠️ Detected invalid/corrupted audio cache <{save_as}> for text: '{text}'. Removing and regenerating...[/yellow]")
            try:
                os.remove(save_as)
            except Exception:
                pass
    
    print(f"Generating ({voice or gender or 'default'}) <{text}...>")
    TTS_METHOD = load_key("tts_method")
    
    scoped_once = single_attempt_active()
    max_retries = 1 if scoped_once else 3
    last_error_is_network = False
    tts_result = None
    for attempt in range(max_retries):
        try:
            if not scoped_once and attempt >= max_retries - 1 and not last_error_is_network:
                print("Asking GPT to correct text...")
                correct_text = ask_gpt(get_correct_text_prompt(text), resp_type="json", log_title='tts_correct_text')
                text = correct_text['text']
            
            if TTS_METHOD == 'edge_tts':
                edge_tts(text, save_as, voice=voice, gender=gender)
                tts_result = {"ok": True, "words": [], "sample_rate": 24000, "duration": 0.0}
            elif TTS_METHOD == 'doubao_tts':
                if scoped_once:
                    tts_result = single_attempt_call(
                        doubao_tts.__wrapped__,
                        text,
                        save_as,
                        number=number,
                        voice_type=voice,
                        gender=gender,
                        section_id=section_id,
                        context_texts=context_texts,
                        return_words=return_words,
                    )
                else:
                    tts_result = doubao_tts(
                        text,
                        save_as,
                        voice_type=voice,
                        gender=gender,
                        section_id=section_id,
                        context_texts=context_texts,
                        return_words=return_words,
                    )
            else:
                raise ValueError(f"Unsupported TTS method: {TTS_METHOD}")
                
            # Check generated audio duration
            duration = get_audio_duration(save_as)
            if duration > 0:
                record_segment_text_hash(save_as, raw_input_text)
                if return_words and isinstance(tts_result, dict):
                    tts_result["duration"] = duration
                break
            else:
                if scoped_once:
                    raise RuntimeError("Single-attempt TTS produced no audio; stopped without retry")
                if os.path.exists(save_as):
                    os.remove(save_as)
                if attempt == max_retries - 1:
                    print(f"Warning: Generated audio duration is 0 for text: {text}")
                    # Create silent audio file
                    silence = AudioSegment.silent(duration=100, frame_rate=24000)  # 100ms silence
                    silence.export(save_as, format="wav")
                    record_segment_text_hash(save_as, raw_input_text)
                    return {"ok": True, "words": [], "sample_rate": 24000, "duration": 0.1} if return_words else None
                print(f"Attempt {attempt + 1} failed, retrying...")
        except Exception as e:
            if is_task_cancelled():
                raise TaskCancelledError("任务已被手动停止") from e
            import subprocess as _sp
            stderr_info = e.stderr if isinstance(e, _sp.CalledProcessError) and e.stderr else ''
            err_msg = f"{str(e)}" + (f"\nstderr: {stderr_info}" if stderr_info else "")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {err_msg}")
            # Detect network errors and apply backoff; skip GPT text correction on next retry
            last_error_is_network = any(kw in err_msg for kw in ('ConnectionReset', 'ClientConnector', 'Cannot connect', 'TimeoutError', 'aiohttp'))
            wait = 3 * (attempt + 1)
            print(f"Attempt {attempt + 1} failed ({'network error, retrying' if last_error_is_network else err_msg}), retrying in {wait}s...")
            sleep_cancellable(wait)
    return tts_result if return_words else None