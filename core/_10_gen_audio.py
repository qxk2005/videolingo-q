from core.tts_backend.single_attempt_scope import single_attempt_active, single_attempt_call
from core.utils.cancel_utils import check_task_cancelled, sleep_cancellable
import ast
import difflib
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pydub import AudioSegment
from rich.console import Console

from core.asr_backend.audio_preprocess import get_audio_duration
from core.prompts import get_batch_trim_prompt
from core.tts_backend.doubao_tts import (
    DOUBAO_SAMPLE_RATE,
    get_doubao_style_prompt,
    resolve_doubao_voice,
)
from core.tts_backend.elevenlabs_tts import (
    get_elevenlabs_model_id,
    get_elevenlabs_voice_settings,
    resolve_elevenlabs_voice,
)
from core.tts_backend.tts_main import (
    clean_text_for_tts,
    copy_segment_text_hash,
    ensure_segment_hashes_initialized,
    is_audio_valid,
    record_segment_text_hash,
    remove_segment_text_hashes_for_numbers,
    tts_main,
)
from core.utils import *
from core.utils.models import *

console = Console()

TEMP_FILE_TEMPLATE = f"{_AUDIO_TMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5
TTS_SETTINGS_FILE = "output/audio/.tts_settings"
PARAGRAPH_MANIFEST_FILE = "output/audio/.paragraph_manifest.json"
VIDEO_SEGMENTS_MAP_FILE = "output/audio/video_segments_map.json"
RETIMED_BG_AUDIO_FILE = "output/audio/retimed_background.wav"

# Bidirectional elastic alignment parameters for paragraph-capable engines
PARAGRAPH_TTS_METHODS = ("doubao_tts", "elevenlabs_tts")
MAX_AUDIO_SPEED_DOUBAO = 1.08
MIN_AUDIO_SPEED_DOUBAO = 0.96
MIN_VIDEO_SPEED_FLOOR = 0.85
EMERGENCY_VIDEO_SPEED_FLOOR = 0.72
EMERGENCY_AUDIO_SPEED_CAP = 1.12
MAX_TAIL_BORROW_SEC = 0.35


def _resolve_paragraph_speaker_voice(row, tts_method: str) -> str:
    gen = str(row.get("gender", "") or "") if pd.notna(row.get("gender", "")) else ""
    raw_v = row.get("voice") if pd.notna(row.get("voice")) else None
    if tts_method == "elevenlabs_tts":
        return resolve_elevenlabs_voice(voice_type=raw_v, gender=gen or None)
    return resolve_doubao_voice(voice_type=raw_v, gender=gen or None)


def _parse_lines_cell(val) -> List[str]:
    if isinstance(val, list):
        return [str(x.get("text", "") if isinstance(x, dict) else x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return [""]
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x.get("text", "") if isinstance(x, dict) else x) for x in parsed]
        except Exception:
            pass
        return [s]
    return [""]


def get_tts_fingerprint():
    """Get a string fingerprint of current TTS settings to detect changes."""
    tts_method = load_key("tts_method")
    settings = {"tts_method": tts_method}
    try:
        if tts_method == "edge_tts":
            settings["voice"] = load_key("edge_tts.voice")
            settings["male_voice"] = load_key("edge_tts.male_voice")
            settings["female_voice"] = load_key("edge_tts.female_voice")
        elif tts_method == "doubao_tts":
            settings["voice"] = load_key("doubao_tts.voice")
            settings["male_voice"] = load_key("doubao_tts.male_voice")
            settings["female_voice"] = load_key("doubao_tts.female_voice")
            settings["style_prompt"] = get_doubao_style_prompt()
            settings["pipeline_ver"] = "v3_para_elastic_24k"
        elif tts_method == "elevenlabs_tts":
            el_model = get_elevenlabs_model_id()
            settings["model_id"] = el_model
            settings["voice"] = load_key("elevenlabs_tts.voice")
            settings["male_voice"] = load_key("elevenlabs_tts.male_voice")
            settings["female_voice"] = load_key("elevenlabs_tts.female_voice")
            settings["voice_settings"] = get_elevenlabs_voice_settings(model_id=el_model)
            settings["pipeline_ver"] = "el_para_elastic_24k"
        elif tts_method == "openai_tts":
            settings["voice"] = load_key("openai_tts.voice")
        elif tts_method == "azure_tts":
            settings["voice"] = load_key("azure_tts.voice")
        elif tts_method == "fish_tts":
            settings["character"] = load_key("fish_tts.character")
        elif tts_method == "sf_fish_tts":
            settings["voice"] = load_key("sf_fish_tts.voice")
        elif tts_method == "gpt_sovits":
            settings["character"] = load_key("gpt_sovits.character")
    except Exception:
        pass
    return json.dumps(settings, sort_keys=True, ensure_ascii=False)


def check_and_clear_audio_cache():
    """If TTS settings changed since last run, clear cached audio files."""
    current = get_tts_fingerprint()
    if os.path.exists(TTS_SETTINGS_FILE):
        with open(TTS_SETTINGS_FILE, "r", encoding="utf-8") as f:
            stored = f.read().strip()
        if stored != current:
            rprint(f"[yellow]⚠️ TTS settings changed ({stored} → {current}), clearing audio cache...[/yellow]")
            for d in [_AUDIO_TMP_DIR, _AUDIO_SEGS_DIR]:
                if os.path.exists(d):
                    shutil.rmtree(d)
            for meta_f in [
                os.path.join(os.path.dirname(_AUDIO_TMP_DIR), ".segment_text_hashes.json"),
                PARAGRAPH_MANIFEST_FILE,
                VIDEO_SEGMENTS_MAP_FILE,
                RETIMED_BG_AUDIO_FILE,
            ]:
                if os.path.exists(meta_f):
                    try:
                        os.remove(meta_f)
                    except Exception:
                        pass
    settings_dir = os.path.dirname(TTS_SETTINGS_FILE)
    if settings_dir:
        os.makedirs(settings_dir, exist_ok=True)
    with open(TTS_SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.write(current)


def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS.mmm or HH:MM:SS,mmm) to seconds."""
    s = str(time_str).strip().replace(",", ".")
    hours, minutes, seconds = s.split(":")
    if "." in seconds:
        sec_part, frac_part = seconds.split(".", 1)
        frac = float(f"0.{frac_part}")
    else:
        sec_part, frac = seconds, 0.0
    return int(hours) * 3600 + int(minutes) * 60 + int(sec_part) + frac


def adjust_audio_speed(
    input_file: str,
    output_file: str,
    speed_factor: float,
    allow_micro_trim: bool = True,
) -> None:
    """Adjust audio speed with high-precision WAV output (24kHz mono preserved)."""
    if abs(speed_factor - 1.0) < 0.002:
        shutil.copy2(input_file, output_file)
        copy_segment_text_hash(input_file, output_file)
        return

    if speed_factor <= 0:
        rprint(f"[red]❌ Error: Invalid speed factor {speed_factor} (must be positive)[/red]")
        shutil.copy2(input_file, output_file)
        copy_segment_text_hash(input_file, output_file)
        return

    min_speed, max_speed = 0.5, 4.0
    if speed_factor < min_speed or speed_factor > max_speed:
        rprint(
            f"[yellow]⚠️ Speed factor {speed_factor:.3f} outside [{min_speed}-{max_speed}], clamping[/yellow]"
        )
        speed_factor = max(min_speed, min(speed_factor, max_speed))

    # Chain atempo stages cleanly if outside [0.5, 2.0]
    rem = float(speed_factor)
    atempo_stages = []
    while rem > 2.0:
        atempo_stages.append("atempo=2.0")
        rem /= 2.0
    while rem < 0.5:
        atempo_stages.append("atempo=0.5")
        rem /= 0.5
    atempo_stages.append(f"atempo={rem:.6f}")
    af_str = ",".join(atempo_stages)

    cmd = ["ffmpeg", "-i", input_file, "-filter:a", af_str, "-y", output_file]
    input_duration = get_audio_duration(input_file)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output_duration = get_audio_duration(output_file)
            expected_duration = input_duration / speed_factor
            diff = output_duration - expected_duration
            if (
                allow_micro_trim
                and output_duration >= expected_duration * 1.02
                and input_duration < 3
                and diff <= 0.1
            ):
                audio = AudioSegment.from_wav(output_file)
                trimmed_audio = audio[: (expected_duration * 1000)]
                trimmed_audio.export(output_file, format="wav")
            copy_segment_text_hash(input_file, output_file)
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                raise e


def trim_tts_silence(input_file: str) -> None:
    """Trim leading and trailing silence from TTS audio in-place."""
    temp_file = input_file.replace(".wav", "_trimmed.wav")
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-af",
        (
            "silenceremove="
            "start_periods=1:start_silence=0.04:start_threshold=-46dB:"
            "stop_periods=-1:stop_silence=0.08:stop_threshold=-46dB"
        ),
        "-y",
        temp_file,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        new_dur = get_audio_duration(temp_file)
        orig_dur = get_audio_duration(input_file)
        if 0 < new_dur < orig_dur:
            shutil.move(temp_file, input_file)
        elif os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def _load_paragraph_manifest() -> Dict[str, List[int]]:
    if not os.path.exists(PARAGRAPH_MANIFEST_FILE):
        return {}
    try:
        with open(PARAGRAPH_MANIFEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("paragraphs"), dict):
            return {str(k): [int(x) for x in v] for k, v in data["paragraphs"].items()}
    except Exception:
        pass
    return {}


def _save_paragraph_manifest(tasks_df: pd.DataFrame) -> None:
    if tasks_df is None or "para_id" not in tasks_df.columns:
        return
    os.makedirs(os.path.dirname(PARAGRAPH_MANIFEST_FILE), exist_ok=True)
    paras = {}
    for pid, grp in tasks_df.groupby("para_id", sort=False):
        paras[str(int(pid))] = [int(n) for n in grp["number"].tolist()]
    tmp_p = PARAGRAPH_MANIFEST_FILE + ".tmp"
    with open(tmp_p, "w", encoding="utf-8") as f:
        json.dump({"paragraphs": paras}, f, ensure_ascii=False, indent=2)
    os.replace(tmp_p, PARAGRAPH_MANIFEST_FILE)


def expand_numbers_to_paragraphs(numbers: list) -> List[int]:
    """Expand edited subtitle numbers to include all sibling numbers in both old and new natural paragraphs."""
    if not numbers:
        return []
    target_set = {int(n) for n in numbers}
    expanded = set(target_set)

    tts_method = load_key("tts_method")
    if tts_method in PARAGRAPH_TTS_METHODS and not single_attempt_active():
        # 1. Include siblings from previously synthesized paragraph manifest
        old_paras = _load_paragraph_manifest()
        for _, member_nums in old_paras.items():
            m_set = set(member_nums)
            if m_set & target_set:
                expanded |= m_set

        # 2. Include siblings from current tts_tasks.xlsx `para_id`
        if os.path.exists(_8_1_AUDIO_TASK):
            try:
                df = pd.read_excel(_8_1_AUDIO_TASK)
                if "para_id" in df.columns and "number" in df.columns:
                    matched_pids = set(
                        df.loc[df["number"].astype(int).isin(target_set), "para_id"].dropna().astype(int).tolist()
                    )
                    if matched_pids:
                        cur_siblings = df.loc[
                            df["para_id"].dropna().astype(int).isin(matched_pids), "number"
                        ].astype(int).tolist()
                        expanded |= set(cur_siblings)
            except Exception:
                pass

    return sorted(expanded)


def invalidate_audio_cache_for_numbers(numbers: list) -> int:
    """
    Delete cached TTS audio files (both tmp and speed-adjusted segs) for the specified task numbers.
    For Doubao V3 paragraph synthesis, automatically expands invalidation to the enclosing natural
    paragraph(s) so re-synthesized audio remains 100% continuous without patch-in artifacts.
    """
    expanded_numbers = expand_numbers_to_paragraphs(numbers)
    if not expanded_numbers:
        return 0

    if len(expanded_numbers) > len(set(int(x) for x in numbers)):
        rprint(
            f"[cyan]🔗 Paragraph-linked cache invalidation: expanded {len(set(numbers))} edited line(s) "
            f"to {len(expanded_numbers)} paragraph sibling line(s)[/cyan]"
        )

    deleted = 0
    for num in expanded_numbers:
        for base_dir in [_AUDIO_TMP_DIR, _AUDIO_SEGS_DIR]:
            if not os.path.exists(base_dir):
                continue
            for pattern in [f"{num}_*.wav", f"{num}.wav"]:
                for fpath in glob.glob(os.path.join(base_dir, pattern)):
                    try:
                        os.remove(fpath)
                        deleted += 1
                    except Exception:
                        pass
    remove_segment_text_hashes_for_numbers(expanded_numbers, os.path.dirname(_AUDIO_TMP_DIR))
    for final_f in [
        "output/dub.wav",
        "output/dub.mp3",
        "output/output_dub.mp4",
        "output/normalized_dub.wav",
        VIDEO_SEGMENTS_MAP_FILE,
        RETIMED_BG_AUDIO_FILE,
    ]:
        if os.path.exists(final_f):
            try:
                os.remove(final_f)
            except Exception:
                pass
    return deleted


# ==============================================================================
# Doubao V3 Paragraph Synthesis + Word-Timestamp Zero-Crossing Slicer
# ==============================================================================

def _norm_align_chars(s: str) -> List[str]:
    """Extract normalized alphanumeric & CJK characters for timestamp alignment."""
    cleaned = re.sub(r"\[[^\]]+\]|\([^)]+\)|（[^）]+）", "", str(s or ""))
    return [ch.lower() for ch in cleaned if ch.isalnum() and ch != "_"]


def _read_wav_mono_int16(wav_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        arr = np.frombuffer(raw, dtype=np.int16).copy()
        if nch > 1:
            arr = arr.reshape(-1, nch)[:, 0].copy()
        return arr, sr
    audio = AudioSegment.from_file(wav_path).set_channels(1).set_sample_width(2).set_frame_rate(DOUBAO_SAMPLE_RATE)
    return np.frombuffer(audio.raw_data, dtype=np.int16).copy(), DOUBAO_SAMPLE_RATE


def _write_wav_mono_int16(samples: np.ndarray, wav_path: str, sr: int = DOUBAO_SAMPLE_RATE) -> float:
    parent = os.path.dirname(os.path.abspath(wav_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    pcm = np.asarray(samples, dtype=np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return len(pcm) / float(sr)


def _find_quiet_zero_crossing(samples: np.ndarray, sr: int, start_sec: float, end_sec: float) -> int:
    """Locate the lowest-energy zero-crossing sample index inside `[start_sec, end_sec]`."""
    n = len(samples)
    if n == 0:
        return 0
    center_sec = 0.5 * (start_sec + end_sec)
    if end_sec - start_sec < 0.030:
        lo_sec = center_sec - 0.020
        hi_sec = center_sec + 0.020
    else:
        # Stay slightly inside long silence gaps while searching the entire valley
        margin = min(0.015, 0.15 * (end_sec - start_sec))
        lo_sec = start_sec + margin
        hi_sec = end_sec - margin

    lo = max(1, min(n - 2, int(round(lo_sec * sr))))
    hi = max(lo + 1, min(n - 1, int(round(hi_sec * sr))))

    seg = samples[lo:hi].astype(np.float32)
    win = max(16, int(0.008 * sr))  # 8ms RMS window
    if len(seg) <= win:
        local_center = len(seg) // 2
    else:
        sq = seg * seg
        csum = np.concatenate(([0.0], np.cumsum(sq)))
        rms2 = (csum[win:] - csum[:-win]) / float(win)
        # Mild preference toward midpoint when energy is identically silent
        positions = np.linspace(-1.0, 1.0, len(rms2), dtype=np.float32)
        score = rms2 + (positions * positions) * 4.0
        local_center = int(np.argmin(score)) + win // 2

    cand = lo + local_center
    zc_radius = min(int(0.004 * sr), 96)
    z_lo = max(1, cand - zc_radius)
    z_hi = min(n - 1, cand + zc_radius)
    sub = samples[z_lo - 1 : z_hi]
    signs = np.signbit(sub[:-1]) != np.signbit(sub[1:])
    zc_indices = np.nonzero(signs)[0]
    if len(zc_indices) > 0:
        best_zc = zc_indices[int(np.argmin(np.abs((z_lo + zc_indices) - cand)))]
        return int(z_lo + best_zc)
    return int(cand)


def _trim_slice_dead_edges_and_fade(
    samples: np.ndarray,
    sr: int,
    thresh_db: float = -48.0,
    head_pad_sec: float = 0.022,
    tail_pad_sec: float = 0.032,
    fade_sec: float = 0.008,
) -> np.ndarray:
    """Trim only excess dead digital silence at slice edges while preserving soft consonants
    and applying an 8ms cosine fade-in/out to prevent clicks."""
    n = len(samples)
    min_keep = int(0.08 * sr)
    if n <= min_keep:
        return samples.copy()

    win = max(24, int(0.010 * sr))
    if n > win * 2:
        amp_thresh = 32768.0 * (10.0 ** (thresh_db / 20.0))
        sq = samples.astype(np.float32) ** 2
        csum = np.concatenate(([0.0], np.cumsum(sq)))
        rms = np.sqrt((csum[win:] - csum[:-win]) / float(win))
        active = np.nonzero(rms > amp_thresh)[0]
        if len(active) > 0:
            first_act = int(active[0])
            last_act = int(active[-1]) + win
            s_idx = max(0, first_act - int(head_pad_sec * sr))
            e_idx = min(n, last_act + int(tail_pad_sec * sr))
            if e_idx - s_idx >= min_keep:
                samples = samples[s_idx:e_idx].copy()
            else:
                samples = samples.copy()
        else:
            samples = samples.copy()
    else:
        samples = samples.copy()

    fade_n = min(int(fade_sec * sr), len(samples) // 4)
    if fade_n >= 4:
        ramp = (0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_n, dtype=np.float32))))
        samples[:fade_n] = np.round(samples[:fade_n].astype(np.float32) * ramp).astype(np.int16)
        samples[-fade_n:] = np.round(samples[-fade_n:].astype(np.float32) * ramp[::-1]).astype(np.int16)
    return samples


def _dp_energy_valley_cuts(samples: np.ndarray, sr: int, char_lens: List[int]) -> List[int]:
    """Fallback deterministic slicer using DP over RMS energy valleys when word timestamps are absent."""
    k_segs = len(char_lens)
    n = len(samples)
    if k_segs <= 1 or n == 0:
        return []

    total_chars = max(1, sum(char_lens))
    cum_ratios = []
    acc = 0
    for c in char_lens[:-1]:
        acc += c
        cum_ratios.append(acc / float(total_chars))

    # Compute 20ms frame RMS profile
    hop = max(1, int(0.020 * sr))
    n_frames = max(k_segs + 2, n // hop)
    padded = samples[: n_frames * hop].astype(np.float32).reshape(n_frames, hop)
    rms_db = 10.0 * np.log10(np.mean(padded * padded, axis=1) + 1.0)
    min_db, max_db = float(np.min(rms_db)), float(np.max(rms_db))
    norm_energy = (rms_db - min_db) / max(1.0, max_db - min_db)

    cuts = []
    prev_frame = 2
    for idx_b, target_r in enumerate(cum_ratios):
        rem_cuts = (k_segs - 1) - idx_b
        max_frame = n_frames - 2 * rem_cuts
        target_f = int(round(target_r * n_frames))
        search_lo = max(prev_frame + 3, int(round(target_f - 0.18 * n_frames)))
        search_hi = min(max_frame, max(search_lo + 1, int(round(target_f + 0.18 * n_frames))))
        f_candidates = np.arange(search_lo, search_hi)
        dist_pen = ((f_candidates - target_f) / max(1.0, 0.12 * n_frames)) ** 2
        costs = norm_energy[f_candidates] * 2.2 + dist_pen
        best_f = int(f_candidates[int(np.argmin(costs))])
        prev_frame = best_f
        sec_center = (best_f * hop) / float(sr)
        cuts.append(_find_quiet_zero_crossing(samples, sr, max(0.0, sec_center - 0.03), sec_center + 0.03))
    return cuts


def slice_paragraph_by_word_timestamps(
    combined_wav_path: str,
    speakable_texts: List[str],
    words_info: List[dict],
) -> List[Tuple[np.ndarray, int]]:
    """Slice a continuous 24kHz paragraph WAV into `len(speakable_texts)` clean segment arrays
    using Doubao V3's native word-level timestamps + zero-crossing valley alignment."""
    samples, sr = _read_wav_mono_int16(combined_wav_path)
    total_samples = len(samples)
    k_count = len(speakable_texts)

    if k_count == 0:
        return []
    if k_count == 1:
        return [(_trim_slice_dead_edges_and_fade(samples, sr), sr)]

    line_norm_chars = [(_norm_align_chars(t) or ["x"]) for t in speakable_texts]
    src_chars = [ch for sub in line_norm_chars for ch in sub]
    cut_char_positions = []
    acc = 0
    for sub in line_norm_chars[:-1]:
        acc += len(sub)
        cut_char_positions.append(acc)

    cut_samples = None
    if words_info and len(words_info) >= 2:
        tts_chars = []
        char_to_word = []
        for w_idx, w in enumerate(words_info):
            w_chars = _norm_align_chars(w.get("word", ""))
            for ch in w_chars:
                tts_chars.append(ch)
                char_to_word.append(w_idx)

        if len(tts_chars) >= 2:
            matcher = difflib.SequenceMatcher(None, src_chars, tts_chars, autojunk=False)
            blocks = matcher.get_matching_blocks()
            anchors = [(0, 0)]
            for b in blocks:
                if b.size > 0:
                    anchors.append((b.a, b.b))
                    anchors.append((b.a + b.size, b.b + b.size))
            anchors.append((len(src_chars), len(tts_chars)))

            # Deduplicate anchors monotonically
            clean_anchors = [anchors[0]]
            for a_i, b_i in anchors[1:]:
                if a_i >= clean_anchors[-1][0] and b_i >= clean_anchors[-1][1]:
                    if a_i == clean_anchors[-1][0]:
                        clean_anchors[-1] = (a_i, max(clean_anchors[-1][1], b_i))
                    else:
                        clean_anchors.append((a_i, b_i))

            def _map_src_to_tts(c_pos: int) -> int:
                for idx_a in range(len(clean_anchors) - 1):
                    a0, b0 = clean_anchors[idx_a]
                    a1, b1 = clean_anchors[idx_a + 1]
                    if a0 <= c_pos <= a1:
                        if a1 == a0:
                            return b0
                        ratio = (c_pos - a0) / float(a1 - a0)
                        return int(round(b0 + ratio * (b1 - b0)))
                return len(tts_chars)

            min_seg_samples = max(int(0.08 * sr), 1)
            proposed_cuts = []
            prev_cut = 0
            for b_idx, c_pos in enumerate(cut_char_positions):
                rem = (k_count - 1) - b_idx
                t_pos = max(1, min(len(tts_chars) - 1, _map_src_to_tts(c_pos)))
                left_w = char_to_word[t_pos - 1]
                right_w = char_to_word[min(t_pos, len(char_to_word) - 1)]
                if right_w <= left_w and left_w + 1 < len(words_info):
                    right_w = left_w + 1

                t_end = float(words_info[left_w]["end"])
                t_next_start = float(words_info[right_w]["start"])
                if t_next_start < t_end:
                    t_end, t_next_start = t_next_start, t_end

                c_samp = _find_quiet_zero_crossing(samples, sr, t_end, t_next_start)
                lo_bound = prev_cut + min_seg_samples
                hi_bound = total_samples - rem * min_seg_samples
                c_samp = max(lo_bound, min(hi_bound, c_samp))
                proposed_cuts.append(c_samp)
                prev_cut = c_samp

            if len(proposed_cuts) == k_count - 1:
                cut_samples = proposed_cuts

    if cut_samples is None:
        cut_samples = _dp_energy_valley_cuts(samples, sr, [len(x) for x in line_norm_chars])

    boundaries = [0] + cut_samples + [total_samples]
    out_slices = []
    for k in range(k_count):
        seg_arr = samples[boundaries[k] : boundaries[k + 1]]
        seg_clean = _trim_slice_dead_edges_and_fade(seg_arr, sr)
        out_slices.append((seg_clean, sr))
    return out_slices


def _build_paragraph_combined_text(speakable_specs: List[dict]) -> str:
    """Join consecutive lines of the same speaker with clause-aware punctuation so mid-sentence
    subtitle splits use natural commas rather than abrupt sentence-ending full stops."""
    display_lang = str(load_key("target_language") or load_key("display_language") or "zh-CN")
    is_cjk = any(k in display_lang for k in ("中", "zh", "日", "ja"))
    comma_mark = "，" if is_cjk else ","
    period_mark = "。" if is_cjk else "."
    terminal_set = set(".!?。！？;；:：…—")
    clause_set = set(",，、")

    parts = []
    n = len(speakable_specs)
    for idx, spec in enumerate(speakable_specs):
        txt = spec["clean_text"].strip()
        if not txt:
            continue
        is_last = idx == n - 1
        last_ch = txt[-1]
        if last_ch in terminal_set or last_ch in clause_set:
            if is_last and last_ch in clause_set:
                txt = txt[:-1] + period_mark
        else:
            if is_last or float(spec.get("gap_after", 0.0)) >= 0.40:
                txt = txt + period_mark
            else:
                txt = txt + comma_mark
        parts.append(txt)

    joiner = "" if is_cjk else " "
    return joiner.join(parts)


def _synthesize_single_doubao_paragraph(
    para_df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    section_id: str,
    context_texts: List[str],
    force_regenerate: bool = False,
) -> Dict[int, float]:
    """Synthesize one natural speech paragraph via Doubao V3 and slice it back into per-line WAVs.
    Returns `{row_number: row_real_dur}`."""
    check_task_cancelled()
    first_row = para_df.iloc[0]
    gender = first_row.get("gender", None)
    raw_voice = first_row.get("voice", None)
    voice = (
        str(raw_voice).strip()
        if (raw_voice is not None and pd.notna(raw_voice) and str(raw_voice).strip())
        else None
    )

    # Flatten all lines across rows in this paragraph
    all_specs = []
    for pos_in_para, (_, row) in enumerate(para_df.iterrows()):
        num = int(row["number"])
        row_lines = _parse_lines_cell(row["lines"])
        row_gap = float(row.get("gap", 0.0) or 0.0)
        for l_idx, raw_line in enumerate(row_lines):
            cleaned = clean_text_for_tts(raw_line)
            alnum_only = re.sub(r"[^\w]", "", cleaned).replace("_", "").strip()
            is_last_in_row = l_idx == len(row_lines) - 1
            gap_after = row_gap if is_last_in_row else 0.10
            all_specs.append(
                {
                    "number": num,
                    "line_idx": l_idx,
                    "raw_text": raw_line,
                    "clean_text": cleaned,
                    "speakable": len(alnum_only) >= 1,
                    "gap_after": gap_after,
                    "temp_file": TEMP_FILE_TEMPLATE.format(f"{num}_{l_idx}"),
                }
            )

    # Check if every slice in this paragraph is already cached and valid
    if not force_regenerate and len(all_specs) > 0:
        all_cached = True
        row_durs: Dict[int, float] = {}
        for spec in all_specs:
            tf = spec["temp_file"]
            if os.path.exists(tf) and os.path.getsize(tf) > 0 and is_audio_valid(tf, spec["raw_text"]):
                row_durs[spec["number"]] = row_durs.get(spec["number"], 0.0) + get_audio_duration(tf)
            else:
                all_cached = False
                break
        if all_cached and sum(row_durs.values()) > 0:
            return row_durs

    # Write 80ms silence for any empty/non-speakable slots
    speakable_specs = []
    for spec in all_specs:
        if not spec["speakable"]:
            sil = np.zeros(int(0.08 * DOUBAO_SAMPLE_RATE), dtype=np.int16)
            _write_wav_mono_int16(sil, spec["temp_file"], DOUBAO_SAMPLE_RATE)
            record_segment_text_hash(spec["temp_file"], spec["raw_text"])
        else:
            speakable_specs.append(spec)

    if not speakable_specs:
        return {int(r["number"]): 0.08 for _, r in para_df.iterrows()}

    # Single speakable line in paragraph -> direct V3 call with speaker's section_id
    if len(speakable_specs) == 1:
        sp = speakable_specs[0]
        if os.path.exists(sp["temp_file"]):
            try:
                os.remove(sp["temp_file"])
            except Exception:
                pass
        tts_main(
            sp["raw_text"],
            sp["temp_file"],
            number=sp["number"],
            task_df=tasks_df,
            gender=gender,
            voice=voice,
            section_id=section_id,
            context_texts=context_texts,
            return_words=True,
        )
        arr, sr = _read_wav_mono_int16(sp["temp_file"])
        arr_clean = _trim_slice_dead_edges_and_fade(arr, sr)
        dur = _write_wav_mono_int16(arr_clean, sp["temp_file"], sr)
        record_segment_text_hash(sp["temp_file"], sp["raw_text"])
        row_durs = {}
        for spec in all_specs:
            row_durs[spec["number"]] = row_durs.get(spec["number"], 0.0) + get_audio_duration(spec["temp_file"])
        return row_durs

    # Multi-line natural speech paragraph -> single continuous V3 synthesis + word-timestamp slicing!
    combined_text = _build_paragraph_combined_text(speakable_specs)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
        combined_wav_path = tmp_f.name

    para_ok = False
    try:
        res = tts_main(
            combined_text,
            combined_wav_path,
            number=speakable_specs[0]["number"],
            task_df=tasks_df,
            gender=gender,
            voice=voice,
            section_id=section_id,
            context_texts=context_texts,
            return_words=True,
        )
        words_info = res.get("words", []) if isinstance(res, dict) else []
        slices = slice_paragraph_by_word_timestamps(
            combined_wav_path,
            [s["clean_text"] for s in speakable_specs],
            words_info,
        )
        if len(slices) == len(speakable_specs):
            all_valid = True
            for sp, (seg_arr, sr) in zip(speakable_specs, slices):
                _write_wav_mono_int16(seg_arr, sp["temp_file"], sr)
                record_segment_text_hash(sp["temp_file"], sp["raw_text"])
                if not is_audio_valid(sp["temp_file"], sp["raw_text"]):
                    all_valid = False
                    break
            if all_valid:
                para_ok = True
                rprint(
                    f"[bold green]✨ Paragraph V3 Synthesis: para_id={int(first_row.get('para_id', 0))} "
                    f"({len(speakable_specs)} lines, words={len(words_info)}) sliced cleanly![/bold green]"
                )
    except Exception as e:
        rprint(f"[yellow]⚠️ Paragraph synthesis fallback ({e}), using sequential context-linked V3...[/yellow]")
    finally:
        if os.path.exists(combined_wav_path):
            try:
                os.remove(combined_wav_path)
            except Exception:
                pass

    # Graceful fallback: sequential line-by-line V3 calls STILL preserving speaker's `section_id` & `context_texts`
    if not para_ok:
        for sp in speakable_specs:
            if os.path.exists(sp["temp_file"]):
                try:
                    os.remove(sp["temp_file"])
                except Exception:
                    pass
            tts_main(
                sp["raw_text"],
                sp["temp_file"],
                number=sp["number"],
                task_df=tasks_df,
                gender=gender,
                voice=voice,
                section_id=section_id,
                context_texts=context_texts,
                return_words=True,
            )
            arr, sr = _read_wav_mono_int16(sp["temp_file"])
            arr_clean = _trim_slice_dead_edges_and_fade(arr, sr)
            _write_wav_mono_int16(arr_clean, sp["temp_file"], sr)
            record_segment_text_hash(sp["temp_file"], sp["raw_text"])

    row_durs = {}
    for spec in all_specs:
        row_durs[spec["number"]] = row_durs.get(spec["number"], 0.0) + get_audio_duration(spec["temp_file"])
    return row_durs


def generate_doubao_paragraphs(tasks_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Synthesize all Doubao TTS paragraphs sequentially per speaker so `section_id` carries
    continuous prosody state from beginning to end of the video."""
    from core._8_2_dub_chunks import assign_speech_paragraphs
    from core.utils.progress_utils import get_progress, update_st_progress

    if "para_id" not in tasks_df.columns or tasks_df["para_id"].isna().any():
        tasks_df = assign_speech_paragraphs(tasks_df)

    _save_paragraph_manifest(tasks_df)
    tasks_df["real_dur"] = 0.0

    tts_method = str(load_key("tts_method") or "doubao_tts")
    style_prompt = get_doubao_style_prompt() if tts_method == "doubao_tts" else ""
    context_texts = [style_prompt] if style_prompt else None

    # Persistent `section_id` per unique speaker identity across the entire video
    speaker_sessions: Dict[str, str] = {}

    def _speaker_key(row) -> str:
        spk = str(row.get("speaker_id", "") or "") if pd.notna(row.get("speaker_id", "")) else ""
        gen = str(row.get("gender", "") or "") if pd.notna(row.get("gender", "")) else ""
        voc = _resolve_paragraph_speaker_voice(row, tts_method)
        return f"{spk}|{gen}|{voc}"

    para_groups = list(tasks_df.groupby("para_id", sort=False))
    progress = get_progress()
    is_internal = not progress.live.is_started
    if is_internal:
        progress.start()

    engine_label = "ElevenLabs" if tts_method == "elevenlabs_tts" else "豆包 V3"
    task_desc = f"🎙️ 正在按自然语流段连续合成 {engine_label} 配音..."
    task = progress.add_task(task_desc, total=len(tasks_df))
    completed_rows = 0

    try:
        for _, para_df in para_groups:
            check_task_cancelled()
            s_key = _speaker_key(para_df.iloc[0])
            if s_key not in speaker_sessions:
                speaker_sessions[s_key] = str(uuid.uuid4())
            sec_id = speaker_sessions[s_key]

            row_durs = _synthesize_single_doubao_paragraph(
                para_df=para_df,
                tasks_df=tasks_df,
                section_id=sec_id,
                context_texts=context_texts,
                force_regenerate=False,
            )
            for num, r_dur in row_durs.items():
                tasks_df.loc[tasks_df["number"] == num, "real_dur"] = r_dur
            step_n = len(para_df)
            completed_rows += step_n
            progress.advance(task, advance=step_n)
            update_st_progress(completed_rows, len(tasks_df), task_desc)
    finally:
        if is_internal:
            progress.stop()
        else:
            progress.remove_task(task)

    return tasks_df, speaker_sessions


# ==============================================================================
# 3-Stage Bidirectional Elastic Audio-Video Sync Engine
# ==============================================================================

def _extract_subline_orig_anchors(tasks_df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    """Compute each `(number, line_idx)` sub-line's original start timestamp (in `tasks_df` scaled coordinates)
    so sentences inside a multi-sentence paragraph stay anchored to their visual cues."""
    anchors: Dict[Tuple[int, int], float] = {}
    for _, row in tasks_df.iterrows():
        num = int(row["number"])
        r_start = parse_df_srt_time(row["start_time"])
        r_end = parse_df_srt_time(row["end_time"])
        r_dur = max(0.05, r_end - r_start)
        row_lines = _parse_lines_cell(row["lines"])
        if len(row_lines) <= 1:
            anchors[(num, 0)] = r_start
        else:
            lens = [max(1, len(re.sub(r"\s+", "", l))) for l in row_lines]
            tot_l = float(sum(lens))
            cur = r_start
            for l_i, ch_l in enumerate(lens):
                anchors[(num, l_i)] = cur
                cur += r_dur * (ch_l / tot_l)
    return anchors


def _solve_anchored_subline_times(
    seg_durs: List[float],
    anchors_new: List[float],
    min_pauses: List[float],
    para_new_start: float,
    para_max_end: float,
) -> List[List[float]]:
    """Place `K` audio segments inside `[para_new_start, para_max_end]` anchored to their visual start
    cues `anchors_new` via Forward-Anchor + Backward-Pullback (zero overlap, zero visual drift)."""
    k_cnt = len(seg_durs)
    if k_cnt == 0:
        return []

    starts = [0.0] * k_cnt
    ends = [0.0] * k_cnt

    # 1. Forward pass: start no earlier than visual cue `anchors_new[k]` and after previous segment + min_pause
    for k in range(k_cnt):
        earliest = para_new_start if k == 0 else (ends[k - 1] + min_pauses[k - 1])
        s_k = max(earliest, anchors_new[k])
        starts[k] = s_k
        ends[k] = s_k + seg_durs[k]

    # 2. Backward pullback: if last segment exceeds `para_max_end`, shift trailing segments left smoothly
    if ends[-1] > para_max_end + 1e-4:
        ends[-1] = para_max_end
        starts[-1] = ends[-1] - seg_durs[-1]
        for k in range(k_cnt - 2, -1, -1):
            latest_end = starts[k + 1] - min_pauses[k]
            if ends[k] > latest_end:
                ends[k] = latest_end
                starts[k] = ends[k] - seg_durs[k]

        # 3. Left clamp guard if total audio fills the entire window tightly
        if starts[0] < para_new_start:
            starts[0] = para_new_start
            ends[0] = starts[0] + seg_durs[0]
            for k in range(1, k_cnt):
                req_start = ends[k - 1] + min_pauses[k - 1]
                if starts[k] < req_start:
                    starts[k] = req_start
                ends[k] = starts[k] + seg_durs[k]

    return [[round(s, 4), round(e, 4)] for s, e in zip(starts, ends)]


def _condense_and_resynthesize_paragraph(
    para_df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    section_id: str,
    context_texts: List[str],
) -> pd.DataFrame:
    """Stage 3 overflow guard: ask LLM to condense overly wordy lines in `para_df` by ~20-25%
    and re-synthesize the entire paragraph cleanly with `section_id` (zero audio truncation)."""
    from core._8_2_dub_chunks import sync_lines_with_text

    trim_items = []
    trim_row_indices = []
    for df_idx, row in para_df.iterrows():
        r_dur = float(row["duration"])
        r_real = float(tasks_df.at[df_idx, "real_dur"] or 0.0)
        txt = str(tasks_df.at[df_idx, "text"] or "").strip()
        if txt and r_real > r_dur * 1.08:
            target_sec = max(1.2, r_dur * 0.92)
            trim_items.append((txt, target_sec))
            trim_row_indices.append(df_idx)

    if not trim_items:
        return tasks_df

    rprint(
        f"[yellow]✂️ Stage-3 Extreme Overflow on para_id={int(para_df.iloc[0]['para_id'])}: "
        f"condensing {len(trim_items)} dense line(s) via LLM before re-synthesis...[/yellow]"
    )

    def _valid_trim(resp):
        for i in range(1, len(trim_items) + 1):
            k = str(i)
            if k not in resp or "result" not in resp.get(k, {}):
                return {"status": "error", "message": f"Missing key {k}"}
        return {"status": "success", "message": ""}

    changed_any = False
    try:
        resp = ask_gpt(
            get_batch_trim_prompt(trim_items),
            resp_type="json",
            log_title="dub_para_overflow_trim",
            valid_def=_valid_trim,
        )
        for local_i, df_idx in enumerate(trim_row_indices, 1):
            old_t = str(tasks_df.at[df_idx, "text"] or "").strip()
            new_t = str(resp[str(local_i)]["result"] or "").strip()
            if new_t and new_t != old_t and len(new_t) < len(old_t):
                tasks_df.at[df_idx, "text"] = new_t
                tasks_df.at[df_idx, "lines"] = sync_lines_with_text(tasks_df.at[df_idx, "lines"], new_t)
                changed_any = True
    except Exception as e:
        rprint(f"[yellow]⚠️ Stage-3 LLM condensation skipped ({e}), using elastic video floor instead.[/yellow]")

    if not changed_any:
        return tasks_df

    para_nums = [int(n) for n in para_df["number"].tolist()]
    for num in para_nums:
        for fpath in glob.glob(os.path.join(_AUDIO_TMP_DIR, f"{num}_*.wav")):
            try:
                os.remove(fpath)
            except Exception:
                pass
    remove_segment_text_hashes_for_numbers(para_nums, os.path.dirname(_AUDIO_TMP_DIR))

    refreshed_para = tasks_df.loc[para_df.index]
    row_durs = _synthesize_single_doubao_paragraph(
        para_df=refreshed_para,
        tasks_df=tasks_df,
        section_id=section_id,
        context_texts=context_texts,
        force_regenerate=True,
    )
    for num, r_dur in row_durs.items():
        tasks_df.loc[tasks_df["number"] == num, "real_dur"] = r_dur

    return tasks_df


def align_paragraphs_bidirectionally(
    tasks_df: pd.DataFrame,
    speaker_sessions: Dict[str, str],
) -> pd.DataFrame:
    """Align Doubao V3 speech paragraphs using 3-stage bidirectional elastic sync:
      - Stage 1: Uniform narrow audio tempo (`1.0x ~ 1.08x`) + internal pause elasticity + `<=0.35s` tail borrow
      - Stage 2: Local smooth video slowdown (`0.85x ~ 1.0x`) on active speech spans only
      - Stage 3: Automatic LLM condensation + paragraph re-synthesis on extreme overflow (zero tail chopping)
    """
    from core._8_2_dub_chunks import _is_same_speaker_row

    rprint("[bold blue]🔄 Running 3-stage bidirectional elastic audio-video alignment...[/bold blue]")
    tasks_df["new_sub_times"] = None

    try:
        global_slow_factor = float(load_key("video_slow_factor") or 1.0)
        if global_slow_factor <= 0:
            global_slow_factor = 1.0
    except Exception:
        global_slow_factor = 1.0

    raw_orig_dur = get_audio_duration(_RAW_AUDIO_FILE) if os.path.exists(_RAW_AUDIO_FILE) else 0.0
    scaled_total_dur = raw_orig_dur / global_slow_factor if raw_orig_dur > 0 else 0.0

    tol_cfg = float(load_key("tolerance") if load_key("tolerance") is not None else MAX_TAIL_BORROW_SEC)
    max_borrow_cap = min(MAX_TAIL_BORROW_SEC, max(0.0, tol_cfg))

    tts_method = str(load_key("tts_method") or "doubao_tts")
    style_prompt = get_doubao_style_prompt() if tts_method == "doubao_tts" else ""
    context_texts = [style_prompt] if style_prompt else None

    def _speaker_key(row) -> str:
        spk = str(row.get("speaker_id", "") or "") if pd.notna(row.get("speaker_id", "")) else ""
        gen = str(row.get("gender", "") or "") if pd.notna(row.get("gender", "")) else ""
        voc = _resolve_paragraph_speaker_voice(row, tts_method)
        return f"{spk}|{gen}|{voc}"

    para_indices_list = [grp.index.tolist() for _, grp in tasks_df.groupby("para_id", sort=False)]
    anchors_scaled = _extract_subline_orig_anchors(tasks_df)

    # Track piecewise intervals in `scaled_orig` coordinates; converted to raw video coordinates when saving
    retimed_spans = []  # list of (o_start, o_end, s_video)
    cum_time_shift = 0.0
    retimed_para_count = 0

    for p_pos, idx_list in enumerate(para_indices_list):
        para_df = tasks_df.loc[idx_list]
        first_row = para_df.iloc[0]
        last_row = para_df.iloc[-1]

        o_start = parse_df_srt_time(first_row["start_time"])
        o_end = parse_df_srt_time(last_row["end_time"])
        if o_end <= o_start + 0.02:
            o_end = o_start + max(0.2, float(last_row.get("duration", 0.5) or 0.5))
        d_orig = o_end - o_start

        # Compute safe trailing silence borrow before next paragraph starts
        if p_pos + 1 < len(para_indices_list):
            next_first_row = tasks_df.loc[para_indices_list[p_pos + 1][0]]
            o_next_start = parse_df_srt_time(next_first_row["start_time"])
            raw_tail_gap = max(0.0, o_next_start - o_end)
            same_next_spk = _is_same_speaker_row(last_row, next_first_row)
            guard = 0.02 if same_next_spk else 0.06
            safe_tail_gap = max(0.0, raw_tail_gap - guard)
        else:
            raw_tail_gap = max(0.0, scaled_total_dur - o_end) if scaled_total_dur > o_end else 0.35
            safe_tail_gap = raw_tail_gap

        b_max = min(max_borrow_cap, safe_tail_gap)

        def _collect_para_metrics(p_df: pd.DataFrame):
            items = []
            for _, r in p_df.iterrows():
                num = int(r["number"])
                r_lines = _parse_lines_cell(r["lines"])
                for l_i, l_txt in enumerate(r_lines):
                    tf = TEMP_FILE_TEMPLATE.format(f"{num}_{l_i}")
                    of = OUTPUT_FILE_TEMPLATE.format(f"{num}_{l_i}")
                    d_raw = get_audio_duration(tf)
                    t_clean = str(l_txt or "").strip()
                    ends_term = bool(t_clean) and t_clean[-1] in "。！？.!?"
                    items.append(
                        {
                            "number": num,
                            "line_idx": l_i,
                            "temp_file": tf,
                            "out_file": of,
                            "raw_dur": d_raw,
                            "orig_anchor": anchors_scaled.get((num, l_i), o_start),
                            "ends_term": ends_term,
                        }
                    )
            min_pauses = []
            for k in range(len(items) - 1):
                min_pauses.append(0.09 if items[k]["ends_term"] else 0.045)
            d_speech = sum(x["raw_dur"] for x in items)
            d_min_p = sum(min_pauses)
            return items, min_pauses, d_speech, d_min_p

        items, min_pauses, d_speech, d_min_p = _collect_para_metrics(para_df)
        d_raw_min = d_speech + d_min_p
        d_avail = d_orig + b_max

        # Stage 3 check: would video slowdown need to drop below MIN_VIDEO_SPEED_FLOOR (0.85x)?
        max_comfortable_raw = MAX_AUDIO_SPEED_DOUBAO * (d_orig / MIN_VIDEO_SPEED_FLOOR + b_max)
        if d_raw_min > max_comfortable_raw and d_speech > 1.2 and not single_attempt_active():
            s_key = _speaker_key(first_row)
            sec_id = speaker_sessions.get(s_key) or str(uuid.uuid4())
            tasks_df = _condense_and_resynthesize_paragraph(para_df, tasks_df, sec_id, context_texts)
            para_df = tasks_df.loc[idx_list]
            anchors_scaled = _extract_subline_orig_anchors(tasks_df)
            items, min_pauses, d_speech, d_min_p = _collect_para_metrics(para_df)
            d_raw_min = d_speech + d_min_p

        # Solve Stage 1 vs Stage 2 speeds
        if d_raw_min <= d_avail:
            s_audio = 1.0
            s_video = 1.0
        elif d_raw_min <= MAX_AUDIO_SPEED_DOUBAO * d_avail:
            s_audio = round(min(MAX_AUDIO_SPEED_DOUBAO, max(1.0, d_raw_min / max(0.05, d_avail))), 4)
            s_video = 1.0
        else:
            s_audio = MAX_AUDIO_SPEED_DOUBAO
            s_video = 1.0  # refined below from exact post-atempo WAV durations

        # Generate speed-adjusted segment WAVs with uniform paragraph `s_audio`
        adj_durs = []
        for it in items:
            adjust_audio_speed(it["temp_file"], it["out_file"], s_audio, allow_micro_trim=False)
            adj_durs.append(get_audio_duration(it["out_file"]))

        scaled_min_pauses = [p / s_audio for p in min_pauses]
        d_adj_total = sum(adj_durs) + sum(scaled_min_pauses)

        # If even after `s_audio` (up to 1.08x) + `b_max` tail borrow it exceeds `d_orig`, engage Stage 2 local video slowdown!
        if d_adj_total > d_orig + b_max + 0.015:
            needed_vid_span = max(d_orig, d_adj_total - b_max)
            s_video = d_orig / needed_vid_span
            if s_video < EMERGENCY_VIDEO_SPEED_FLOOR:
                # Emergency safety clamp (e.g. ultra-short 1s utterance): cap video at 0.72x & bump audio slightly <=1.12x
                s_video = EMERGENCY_VIDEO_SPEED_FLOOR
                max_vid_span = d_orig / s_video
                req_s_audio = min(
                    EMERGENCY_AUDIO_SPEED_CAP,
                    s_audio * (d_adj_total / max(0.05, max_vid_span + b_max)),
                )
                if req_s_audio > s_audio + 0.003:
                    s_audio = round(req_s_audio, 4)
                    adj_durs = []
                    for it in items:
                        adjust_audio_speed(it["temp_file"], it["out_file"], s_audio, allow_micro_trim=False)
                        adj_durs.append(get_audio_duration(it["out_file"]))
                    scaled_min_pauses = [p / s_audio for p in min_pauses]
                    d_adj_total = sum(adj_durs) + sum(scaled_min_pauses)
                needed_vid_span = max(d_orig, d_adj_total - b_max)
                s_video = d_orig / needed_vid_span

            s_video = round(max(0.60, min(1.0, s_video)), 5)
            d_vid_new = d_orig / s_video
            retimed_spans.append((o_start, o_end, s_video))
            retimed_para_count += 1
        else:
            s_video = 1.0
            d_vid_new = d_orig

        n_start = o_start + cum_time_shift
        n_end = n_start + d_vid_new
        para_max_end = n_end + b_max

        anchors_new = [
            n_start + max(0.0, (it["orig_anchor"] - o_start)) / s_video for it in items
        ]
        placed_times = _solve_anchored_subline_times(
            seg_durs=adj_durs,
            anchors_new=anchors_new,
            min_pauses=scaled_min_pauses,
            para_new_start=n_start,
            para_max_end=para_max_end,
        )

        # Write placed `new_sub_times` back per row
        by_num: Dict[int, List[List[float]]] = {}
        for it, t_pair in zip(items, placed_times):
            by_num.setdefault(it["number"], []).append(t_pair)
        for num, sub_t_list in by_num.items():
            main_idx = tasks_df[tasks_df["number"] == num].index[0]
            tasks_df.at[main_idx, "new_sub_times"] = sub_t_list

        cum_time_shift += d_vid_new - d_orig
        vid_tag = f", video={s_video:.2f}x (+{d_vid_new - d_orig:.2f}s)" if s_video < 0.999 else ""
        rprint(
            f"[cyan]⚡ Para {int(first_row['para_id'])} (rows {idx_list[0]}–{idx_list[-1]}): "
            f"audio={s_audio:.3f}x{vid_tag}[/cyan]"
        )

    # Persist complete piecewise video timeline map (converted back to raw source video timestamps)
    last_scaled_end = max(
        scaled_total_dur,
        parse_df_srt_time(tasks_df.iloc[-1]["end_time"]) + 0.5 if len(tasks_df) > 0 else 0.0,
    )
    intervals = []
    cur_o = 0.0
    cur_n = 0.0

    # Combine global `video_slow_factor` with per-segment `s_video`
    for sp_o_start, sp_o_end, sp_v_speed in retimed_spans:
        raw_s = sp_o_start * global_slow_factor
        raw_e = sp_o_end * global_slow_factor
        if raw_s > cur_o + 1e-4:
            eff_gap_speed = global_slow_factor
            dur_n = (raw_s - cur_o) / eff_gap_speed
            intervals.append(
                {
                    "orig_start": round(cur_o, 5),
                    "orig_end": round(raw_s, 5),
                    "new_start": round(cur_n, 5),
                    "new_end": round(cur_n + dur_n, 5),
                    "speed": round(eff_gap_speed, 6),
                }
            )
            cur_n += dur_n
            cur_o = raw_s

        eff_span_speed = global_slow_factor * sp_v_speed
        dur_n = (raw_e - raw_s) / eff_span_speed
        intervals.append(
            {
                "orig_start": round(raw_s, 5),
                "orig_end": round(raw_e, 5),
                "new_start": round(cur_n, 5),
                "new_end": round(cur_n + dur_n, 5),
                "speed": round(eff_span_speed, 6),
            }
        )
        cur_n += dur_n
        cur_o = raw_e

    raw_final_end = last_scaled_end * global_slow_factor
    if raw_final_end > cur_o + 1e-4:
        eff_tail_speed = global_slow_factor
        dur_n = (raw_final_end - cur_o) / eff_tail_speed
        intervals.append(
            {
                "orig_start": round(cur_o, 5),
                "orig_end": round(raw_final_end, 5),
                "new_start": round(cur_n, 5),
                "new_end": round(cur_n + dur_n, 5),
                "speed": round(eff_tail_speed, 6),
            }
        )
        cur_n += dur_n

    os.makedirs(os.path.dirname(VIDEO_SEGMENTS_MAP_FILE), exist_ok=True)
    seg_map_payload = {
        "has_dynamic_retiming": len(retimed_spans) > 0,
        "retimed_paragraph_count": retimed_para_count,
        "total_added_seconds": round(cum_time_shift, 4),
        "orig_total_duration": round(raw_final_end, 4),
        "new_total_duration": round(cur_n, 4),
        "intervals": intervals,
    }
    with open(VIDEO_SEGMENTS_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(seg_map_payload, f, ensure_ascii=False, indent=2)

    rprint(
        f"[bold green]✅ Bidirectional elastic sync complete: {retimed_para_count}/{len(para_indices_list)} "
        f"paragraph(s) locally retimed (net video delta: +{cum_time_shift:.2f}s, 0 audio tails truncated)![/bold green]"
    )
    return tasks_df


# ==============================================================================
# Legacy / Non-Doubao (edge_tts & single_attempt_scope) Pipeline Compatibility
# ==============================================================================

def process_row(row: pd.Series, tasks_df: pd.DataFrame) -> Tuple[int, float]:
    """Process single row for non-paragraph engines (e.g. edge_tts or single-attempt mode)."""
    check_task_cancelled()
    number = row["number"]
    lines = _parse_lines_cell(row["lines"])
    gender = row.get("gender", None)
    raw_voice = row.get("voice", None)
    voice = (
        str(raw_voice).strip()
        if (raw_voice is not None and pd.notna(raw_voice) and str(raw_voice).strip())
        else None
    )
    scoped_once = single_attempt_active()

    all_cached_valid = True
    cached_dur = 0.0
    for line_index, line in enumerate(lines):
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0 and is_audio_valid(temp_file, line):
            cached_dur += get_audio_duration(temp_file)
        else:
            all_cached_valid = False
            break
    if all_cached_valid and len(lines) > 0 and cached_dur > 0:
        return number, cached_dur

    real_dur = 0.0
    for line_index, line in enumerate(lines):
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
        if not scoped_once and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
        tts_main(line, temp_file, number, tasks_df, gender=gender, voice=voice)
        trim_tts_silence(temp_file)
        dur = get_audio_duration(temp_file)
        if dur <= 0 and scoped_once:
            raise RuntimeError("Single-attempt TTS invalid audio; stopped without regeneration")
        if dur <= 0 and os.path.exists(temp_file):
            os.remove(temp_file)
            tts_main(line, temp_file, number, tasks_df, gender=gender, voice=voice)
            trim_tts_silence(temp_file)
            dur = get_audio_duration(temp_file)
        real_dur += dur

    return number, real_dur


def generate_tts_audio(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate TTS audio for standard non-paragraph engines (e.g. edge_tts)."""
    tasks_df["real_dur"] = 0.0
    rprint("[bold green]🎯 Starting TTS audio generation...[/bold green]")

    from core.utils.progress_utils import get_progress, update_st_progress

    progress = get_progress()
    is_internal = not progress.live.is_started
    if is_internal:
        progress.start()

    task_desc = "🔄 正在生成 TTS 配音音频..."
    task = progress.add_task(task_desc, total=len(tasks_df))

    tts_method = load_key("tts_method")
    tts_max_workers = 1 if single_attempt_active() else int(load_key("tts_max_workers") or 1)
    if tts_method == "gpt_sovits":
        tts_max_workers = 1

    try:
        if tts_max_workers <= 1:
            for i, (_, row) in enumerate(tasks_df.iterrows()):
                check_task_cancelled()
                number, real_dur = process_row(row, tasks_df)
                tasks_df.loc[tasks_df["number"] == number, "real_dur"] = real_dur
                progress.advance(task)
                update_st_progress(i + 1, len(tasks_df), task_desc)
        else:
            executor = ThreadPoolExecutor(max_workers=tts_max_workers)
            futures = [executor.submit(process_row, row, tasks_df.copy()) for _, row in tasks_df.iterrows()]
            try:
                completed_count = 0
                for future in as_completed(futures):
                    try:
                        number, real_dur = future.result()
                        tasks_df.loc[tasks_df["number"] == number, "real_dur"] = real_dur
                        progress.advance(task)
                        completed_count += 1
                        update_st_progress(completed_count, len(tasks_df), task_desc)
                    except BaseException as e:
                        for f in futures:
                            f.cancel()
                        raise e
            finally:
                executor.shutdown(wait=False)
    finally:
        if is_internal:
            progress.stop()
        else:
            progress.remove_task(task)

    rprint("[bold green]✨ TTS audio generation completed![/bold green]")
    return tasks_df


def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor (legacy path for edge_tts)."""
    chunk_durs = chunk_df["real_dur"].sum()
    tol_durs = chunk_df["tol_dur"].sum()
    durations = tol_durs - chunk_df.iloc[-1]["tolerance"]
    all_gaps = chunk_df["gap"].sum() - chunk_df.iloc[-1]["gap"]

    keep_gaps = True
    speed_var_error = 0.1

    if tol_durs <= 0 or durations <= 0:
        return min_speed, False

    safe_durations = max(durations - speed_var_error, 0.01)
    safe_tol_durs = max(tol_durs - speed_var_error, 0.01)

    if (chunk_durs + all_gaps) / accept < durations:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / safe_durations)
    elif chunk_durs / accept < durations:
        speed_factor = max(min_speed, chunk_durs / safe_durations)
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / safe_tol_durs)
    else:
        speed_factor = chunk_durs / safe_tol_durs
        keep_gaps = False

    speed_factor = max(min_speed, min(speed_factor, 4.0))
    return round(speed_factor, 3), keep_gaps


def _apply_speed_to_chunk(chunk_df: pd.DataFrame, speed_factor: float, keep_gaps: bool, chunk_start_time: float):
    cur_time = chunk_start_time
    number_to_sub_times = {}
    for i, row in chunk_df.iterrows():
        if i != 0 and keep_gaps:
            cur_time += chunk_df.iloc[i - 1]["gap"] / speed_factor
        number = row["number"]
        lines = _parse_lines_cell(row["lines"])
        new_sub_times = []
        for line_index, _ in enumerate(lines):
            temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
            output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
            adjust_audio_speed(temp_file, output_file, speed_factor)
            ad_dur = get_audio_duration(output_file)
            new_sub_times.append([cur_time, cur_time + ad_dur])
            cur_time += ad_dur
        number_to_sub_times[number] = new_sub_times
    return cur_time, number_to_sub_times


def _truncate_chunk_tail(tasks_df: pd.DataFrame, index: int, chunk_end_time: float) -> None:
    last_number = tasks_df.iloc[index]["number"]
    last_lines = _parse_lines_cell(tasks_df.iloc[index]["lines"])
    last_file = OUTPUT_FILE_TEMPLATE.format(f"{last_number}_{len(last_lines) - 1}")
    audio = AudioSegment.from_wav(last_file)
    cur_end = tasks_df.at[index, "new_sub_times"][-1][1]
    keep_ms = max(100, (len(audio) / 1000 - (cur_end - chunk_end_time)) * 1000)
    audio[:keep_ms].export(last_file, format="wav")
    last_times = tasks_df.at[index, "new_sub_times"]
    last_times[-1][1] = chunk_end_time
    tasks_df.at[index, "new_sub_times"] = last_times


def merge_chunks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline (legacy path for edge_tts / single_attempt)."""
    rprint("[bold blue]🔄 Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0
    tasks_df["new_sub_times"] = None

    if os.path.exists(VIDEO_SEGMENTS_MAP_FILE):
        try:
            os.remove(VIDEO_SEGMENTS_MAP_FILE)
        except Exception:
            pass

    for index, row in tasks_df.iterrows():
        if row["cut_off"] != 1:
            continue

        chunk_df = tasks_df.iloc[chunk_start : index + 1].reset_index(drop=True)
        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]["start_time"])
        chunk_end_time = parse_df_srt_time(chunk_df.iloc[-1]["end_time"]) + chunk_df.iloc[-1]["tolerance"]

        cur_time, sub_times = _apply_speed_to_chunk(chunk_df, speed_factor, keep_gaps, chunk_start_time)

        if cur_time > chunk_end_time + 0.6:
            available = chunk_end_time - chunk_start_time
            actual = cur_time - chunk_start_time
            if available > 0 and actual > 0:
                corrected_sf = max(0.5, min(speed_factor * actual / available, 4.0))
                cur_time, sub_times = _apply_speed_to_chunk(chunk_df, corrected_sf, keep_gaps, chunk_start_time)
                speed_factor = corrected_sf

        for number, new_sub_times in sub_times.items():
            main_df_idx = tasks_df[tasks_df["number"] == number].index[0]
            tasks_df.at[main_df_idx, "new_sub_times"] = new_sub_times

        if cur_time > chunk_end_time:
            _truncate_chunk_tail(tasks_df, index, chunk_end_time)

        chunk_start = index + 1

    rprint("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df


def gen_audio() -> None:
    """Main function: Generate audio and process timeline."""
    rprint("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")

    check_and_clear_audio_cache()

    os.makedirs(_AUDIO_TMP_DIR, exist_ok=True)
    os.makedirs(_AUDIO_SEGS_DIR, exist_ok=True)

    tasks_df = pd.read_excel(_8_1_AUDIO_TASK)
    ensure_segment_hashes_initialized(tasks_df, os.path.dirname(_AUDIO_TMP_DIR))
    rprint("[green]📊 Loaded task file successfully[/green]")

    tts_method = load_key("tts_method")
    if tts_method in PARAGRAPH_TTS_METHODS and not single_attempt_active():
        tasks_df, speaker_sessions = generate_doubao_paragraphs(tasks_df)
        tasks_df = align_paragraphs_bidirectionally(tasks_df, speaker_sessions)
    else:
        tasks_df = generate_tts_audio(tasks_df)
        tasks_df = merge_chunks(tasks_df)

    tasks_df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint("[bold green]🎉 Audio generation completed successfully![/bold green]")


if __name__ == "__main__":
    gen_audio()
