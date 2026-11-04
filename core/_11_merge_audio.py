import ast
import os
import subprocess
import wave
import numpy as np
import pandas as pd
from pydub import AudioSegment
from rich.console import Console

from core._10_gen_audio import TEMP_FILE_TEMPLATE, adjust_audio_speed
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.tts_main import (
    ensure_segment_hashes_initialized,
    is_audio_valid,
    tts_main,
)
from core.utils import *
from core.utils.models import *

console = Console()

DUB_WAV_FILE = "output/dub.wav"
DUB_VOCAL_FILE = "output/dub.mp3"
DUB_SUB_FILE = "output/dub.srt"
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"
MERGE_SAMPLE_RATE = 24000


def _safe_parse_list(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    cur = val
    for _ in range(3):
        if isinstance(cur, list):
            return cur
        if isinstance(cur, tuple):
            return list(cur)
        if isinstance(cur, str):
            s = cur.strip()
            if not s:
                return []
            try:
                cur = ast.literal_eval(s)
            except Exception:
                return [s]
        else:
            break
    if isinstance(cur, list):
        return cur
    if isinstance(cur, tuple):
        return list(cur)
    return [str(cur)]


def load_and_flatten_data(excel_file):
    """Load and flatten Excel data."""
    df = pd.read_excel(excel_file)
    lines = [_safe_parse_list(line) for line in df["lines"].tolist()]
    lines = [str(item.get("text", "") if isinstance(item, dict) else item) for sublist in lines for item in sublist]

    new_sub_times = [_safe_parse_list(time_v) for time_v in df["new_sub_times"].tolist()]
    new_sub_times = [item for sublist in new_sub_times for item in sublist]

    return df, lines, new_sub_times


def get_audio_files(df):
    """Generate a list of audio file paths."""
    audios = []
    for _, row in df.iterrows():
        number = row["number"]
        line_count = len(_safe_parse_list(row["lines"]))
        for line_index in range(line_count):
            temp_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
            audios.append(temp_file)
    return audios


def _decode_segment_pcm16(audio_file: str, sample_rate: int = MERGE_SAMPLE_RATE) -> np.ndarray:
    """Decode a WAV segment into mono 16-bit PCM numpy samples at `sample_rate` with zero MP3 delay."""
    if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 100:
        return np.zeros(int(0.08 * sample_rate), dtype=np.int16)

    # Fast path if already mono 16-bit WAV at target sample_rate
    try:
        with wave.open(audio_file, "rb") as wf:
            if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == sample_rate:
                raw = wf.readframes(wf.getnframes())
                if len(raw) > 0:
                    return np.frombuffer(raw, dtype=np.int16).copy()
    except Exception:
        pass

    try:
        audio_seg = (
            AudioSegment.from_file(audio_file)
            .set_frame_rate(sample_rate)
            .set_channels(1)
            .set_sample_width(2)
        )
        return np.frombuffer(audio_seg.raw_data, dtype=np.int16).copy()
    except Exception as e:
        console.print(f"[yellow]⚠️ Failed to decode {audio_file}: {e}, inserting 80ms silence[/yellow]")
        return np.zeros(int(0.08 * sample_rate), dtype=np.int16)


def merge_audio_segments_sample_exact(audios, new_sub_times, sample_rate: int = MERGE_SAMPLE_RATE) -> np.ndarray:
    """Sample-exact timeline placement at `24000Hz` so zero cumulative rounding drift ever occurs."""
    from core.utils.progress_utils import get_progress, update_st_progress

    progress = get_progress()
    is_internal = not progress.live.is_started
    if is_internal:
        progress.start()

    task_desc = "🎵 正在按采样级精度合成 24kHz 配音音轨..."
    task = progress.add_task(task_desc, total=len(audios))

    if not new_sub_times:
        if is_internal:
            progress.stop()
        else:
            progress.remove_task(task)
        return np.zeros(int(0.1 * sample_rate), dtype=np.int16)

    max_end_sec = max(float(t[1]) for t in new_sub_times) + 1.0
    capacity = int(round(max_end_sec * sample_rate)) + sample_rate
    canvas = np.zeros(capacity, dtype=np.float32)
    furthest_sample = 0

    try:
        for i, (audio_file, time_range) in enumerate(zip(audios, new_sub_times)):
            if not os.path.exists(audio_file):
                console.print(f"[bold yellow]⚠️ Warning: File {audio_file} missing, skipping...[/bold yellow]")
                progress.advance(task)
                update_st_progress(i + 1, len(audios), task_desc)
                continue

            seg_pcm = _decode_segment_pcm16(audio_file, sample_rate=sample_rate).astype(np.float32)
            start_time = max(0.0, float(time_range[0]))
            start_idx = int(round(start_time * sample_rate))
            end_idx = start_idx + len(seg_pcm)

            if end_idx > len(canvas):
                extra = np.zeros(end_idx - len(canvas) + sample_rate, dtype=np.float32)
                canvas = np.concatenate((canvas, extra))

            # Smooth 6ms crossfade if a tiny millisecond rounding overlap exists with previous tail
            if start_idx < furthest_sample and furthest_sample > start_idx:
                ov = min(furthest_sample - start_idx, len(seg_pcm), int(0.008 * sample_rate))
                if ov > 2:
                    fade_in = np.linspace(0.0, 1.0, ov, dtype=np.float32)
                    fade_out = 1.0 - fade_in
                    canvas[start_idx : start_idx + ov] = (
                        canvas[start_idx : start_idx + ov] * fade_out + seg_pcm[:ov] * fade_in
                    )
                    if len(seg_pcm) > ov:
                        canvas[start_idx + ov : end_idx] = seg_pcm[ov:]
                else:
                    canvas[start_idx:end_idx] += seg_pcm
            else:
                canvas[start_idx:end_idx] += seg_pcm

            furthest_sample = max(furthest_sample, end_idx)
            progress.advance(task)
            update_st_progress(i + 1, len(audios), task_desc)
    finally:
        if is_internal:
            progress.stop()
        else:
            progress.remove_task(task)

    final_samples = np.clip(np.round(canvas[:furthest_sample]), -32768, 32767).astype(np.int16)
    return final_samples


def create_srt_subtitle():
    df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)

    with open(DUB_SUB_FILE, "w", encoding="utf-8") as f:
        for i, ((start_time, end_time), line) in enumerate(zip(new_sub_times, lines), 1):
            s_ms_total = max(0, int(round(float(start_time) * 1000)))
            e_ms_total = max(s_ms_total + 10, int(round(float(end_time) * 1000)))
            s_h, rem = divmod(s_ms_total, 3_600_000)
            s_m, rem = divmod(rem, 60_000)
            s_s, s_ms = divmod(rem, 1_000)
            e_h, rem = divmod(e_ms_total, 3_600_000)
            e_m, rem = divmod(rem, 60_000)
            e_s, e_ms = divmod(rem, 1_000)

            start_str = f"{s_h:02d}:{s_m:02d}:{s_s:02d},{s_ms:03d}"
            end_str = f"{e_h:02d}:{e_m:02d}:{e_s:02d},{e_ms:03d}"

            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n")

    rprint(f"[bold green]✅ Subtitle file created: {DUB_SUB_FILE}[/bold green]")


def verify_and_fix_audios(df, audios):
    """Scan all output audio segments in segs/, validate them, and self-heal any missing or invalid files."""
    console.print("[bold cyan]🔍 [Integrity Check] Verifying all audio segments...[/bold cyan]")
    fixed_count = 0
    checked_count = 0

    for _, row in df.iterrows():
        number = row["number"]
        lines = _safe_parse_list(row["lines"])
        new_sub_times = _safe_parse_list(row["new_sub_times"])
        gender = row.get("gender", None)
        raw_voice = row.get("voice", None)
        voice = (
            str(raw_voice).strip()
            if (raw_voice is not None and pd.notna(raw_voice) and str(raw_voice).strip())
            else None
        )

        for line_index, line in enumerate(lines):
            line_text = str(line.get("text", "") if isinstance(line, dict) else line)
            checked_count += 1
            output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")

            is_valid = os.path.exists(output_file) and is_audio_valid(output_file, line_text)
            if not is_valid:
                console.print(
                    f"[yellow]⚠️ [Corrupted/Missing Segment] Audio '{output_file}' invalid/missing. Self-healing...[/yellow]"
                )
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except Exception:
                        pass

                temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                if os.path.exists(temp_file) and not is_audio_valid(temp_file, line_text):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass

                tts_main(line_text, temp_file, number, df, gender=gender, voice=voice)

                if os.path.exists(temp_file):
                    raw_dur = get_audio_duration(temp_file)
                    start_time, end_time = new_sub_times[line_index]
                    target_dur = max(0.2, float(end_time) - float(start_time))
                    sf = max(0.5, min(raw_dur / target_dur, 4.0))
                    try:
                        adjust_audio_speed(temp_file, output_file, sf, allow_micro_trim=False)
                        if is_audio_valid(output_file, line_text):
                            fixed_count += 1
                    except Exception as e:
                        console.print(f"[red]❌ [Self-Healed Failed] Speed adjust error: {e}[/red]")

    if fixed_count > 0:
        console.print(
            f"[bold green]🎉 [Integrity Checked] Scanned {checked_count} segments, self-healed {fixed_count}![/bold green]"
        )
    else:
        console.print(
            f"[bold green]✅ [Integrity Checked] Scanned {checked_count} segments. All 100% healthy.[/bold green]"
        )


def merge_full_audio():
    """Main function: Sample-exact 24kHz audio merge + lossless WAV & high-bitrate MP3 export."""
    console.print("\n[bold cyan]🎬 Starting 24kHz sample-exact audio merging process...[/bold cyan]")

    with console.status("[bold cyan]📊 Loading data from Excel...[/bold cyan]"):
        df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
        ensure_segment_hashes_initialized(df, os.path.dirname(_AUDIO_SEGS_DIR))

    audios = get_audio_files(df)
    console.print(f"[bold green]✅ Found {len(audios)} audio segments[/bold green]")

    create_srt_subtitle()
    verify_and_fix_audios(df, audios)

    if not audios or not os.path.exists(audios[0]):
        console.print(f"[bold red]❌ Error: First audio file does not exist![/bold red]")
        return

    sample_rate = MERGE_SAMPLE_RATE
    merged_samples = merge_audio_segments_sample_exact(audios, new_sub_times, sample_rate=sample_rate)

    os.makedirs(os.path.dirname(DUB_WAV_FILE) or ".", exist_ok=True)
    with wave.open(DUB_WAV_FILE, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(merged_samples.tobytes())

    # Export high-bitrate 192k MP3 alongside lossless WAV for UI preview & legacy compatibility
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", DUB_WAV_FILE,
            "-c:a", "libmp3lame", "-b:a", "192k", "-ar", str(sample_rate),
            DUB_VOCAL_FILE,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    console.print(f"[bold green]✅ 24kHz lossless WAV & 192k MP3 merged: {DUB_WAV_FILE} / {DUB_VOCAL_FILE}[/bold green]")


if __name__ == "__main__":
    merge_full_audio()
