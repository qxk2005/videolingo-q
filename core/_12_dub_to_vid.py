import json
import os
import platform
import subprocess
import wave

import cv2
import numpy as np
from rich.console import Console

from core._10_gen_audio import RETIMED_BG_AUDIO_FILE, VIDEO_SEGMENTS_MAP_FILE
from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import convert_video_to_audio, normalize_audio_volume
from core.asr_backend.demucs_vl import demucs_audio
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = "output/dub.srt"
DUB_WAV = "output/dub.wav"
DUB_AUDIO = "output/dub.mp3"
FILTER_SCRIPT_FILE = "output/audio/dub_filter_complex.txt"

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = "Arial"
if platform.system() == "Linux":
    TRANS_FONT_NAME = "NotoSansCJK-Regular"
if platform.system() == "Darwin":
    TRANS_FONT_NAME = "Arial Unicode MS"

TRANS_FONT_COLOR = "&H00FFFF"
TRANS_OUTLINE_COLOR = "&H000000"
TRANS_OUTLINE_WIDTH = 1
TRANS_BACK_COLOR = "&H33000000"


def _build_pts_binary_tree(intervals: list) -> str:
    """Build a balanced O(log N)-depth FFmpeg `setpts` expression over piecewise linear intervals."""
    if len(intervals) == 1:
        iv = intervals[0]
        o_s = float(iv["orig_start"])
        n_s = float(iv["new_start"])
        spd = float(iv["speed"])
        if abs(spd - 1.0) < 1e-5:
            return f"({n_s:.6f}+(T-{o_s:.6f}))"
        return f"({n_s:.6f}+(T-{o_s:.6f})/{spd:.6f})"
    mid = len(intervals) // 2
    pivot = float(intervals[mid]["orig_start"])
    left_expr = _build_pts_binary_tree(intervals[:mid])
    right_expr = _build_pts_binary_tree(intervals[mid:])
    return f"if(lt(T,{pivot:.6f}),{left_expr},{right_expr})"


def _atempo_filter_str(speed: float) -> str:
    rem = float(speed)
    stages = []
    while rem > 2.0:
        stages.append("atempo=2.0")
        rem /= 2.0
    while rem < 0.5:
        stages.append("atempo=0.5")
        rem /= 0.5
    stages.append(f"atempo={rem:.6f}")
    return ",".join(stages)


def _build_retimed_background_wav(
    background_file: str,
    intervals: list,
    out_wav_path: str,
    sr: int = 24000,
) -> str:
    """Stretch background M&E audio piecewise in memory to match retimed video intervals sample-for-sample."""
    decode_cmd = [
        "ffmpeg", "-v", "error", "-i", background_file,
        "-ar", str(sr), "-ac", "2", "-f", "s16le", "-",
    ]
    proc = subprocess.run(decode_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_bytes = proc.stdout
    total_frames = len(raw_bytes) // 4
    if total_frames == 0:
        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        with wave.open(out_wav_path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00" * (sr * 4))
        return out_wav_path

    bg_stereo = np.frombuffer(raw_bytes[: total_frames * 4], dtype=np.int16).reshape(-1, 2)

    max_new_end = max(float(iv["new_end"]) for iv in intervals) if intervals else (total_frames / float(sr))
    out_len = int(round(max_new_end * sr)) + sr
    canvas = np.zeros((out_len, 2), dtype=np.float32)
    xfade_len = int(0.012 * sr)  # 12ms crossfade at speed boundaries
    furthest = 0

    for iv in intervals:
        o_s = max(0.0, float(iv["orig_start"]))
        o_e = max(o_s, float(iv["orig_end"]))
        n_s = max(0.0, float(iv["new_start"]))
        n_e = max(n_s, float(iv["new_end"]))
        spd = float(iv["speed"])

        s_i = min(total_frames, int(round(o_s * sr)))
        e_i = min(total_frames, max(s_i, int(round(o_e * sr))))
        target_frames = max(0, int(round((n_e - n_s) * sr)))
        if target_frames == 0:
            continue

        # Include tiny extra pad for crossfade continuity
        e_i_padded = min(total_frames, e_i + int(0.015 * sr))
        src_chunk = bg_stereo[s_i:e_i_padded]

        if len(src_chunk) == 0:
            stretched = np.zeros((target_frames, 2), dtype=np.float32)
        elif abs(spd - 1.0) < 0.002:
            stretched = src_chunk.astype(np.float32)
        else:
            af_chain = _atempo_filter_str(spd)
            st_proc = subprocess.run(
                [
                    "ffmpeg", "-v", "error",
                    "-f", "s16le", "-ar", str(sr), "-ac", "2", "-i", "-",
                    "-filter:a", af_chain,
                    "-f", "s16le", "-",
                ],
                input=src_chunk.tobytes(),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            st_frames = len(st_proc.stdout) // 4
            stretched = (
                np.frombuffer(st_proc.stdout[: st_frames * 4], dtype=np.int16)
                .reshape(-1, 2)
                .astype(np.float32)
            )

        dst_s = int(round(n_s * sr))
        use_n = min(len(stretched), target_frames + xfade_len)
        dst_e = dst_s + use_n
        if dst_e > len(canvas):
            extra = np.zeros((dst_e - len(canvas) + sr, 2), dtype=np.float32)
            canvas = np.concatenate((canvas, extra), axis=0)

        seg_f = stretched[:use_n]
        if dst_s < furthest:
            ov = min(furthest - dst_s, use_n, xfade_len)
            if ov > 2:
                ramp = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
                canvas[dst_s : dst_s + ov] = canvas[dst_s : dst_s + ov] * (1.0 - ramp) + seg_f[:ov] * ramp
                if use_n > ov:
                    canvas[dst_s + ov : dst_e] = seg_f[ov:]
            else:
                canvas[dst_s:dst_e] = seg_f
        else:
            canvas[dst_s:dst_e] = seg_f

        furthest = max(furthest, dst_s + min(use_n, target_frames + xfade_len // 2))

    final_frames = max(furthest, int(round(max_new_end * sr)))
    final_pcm = np.clip(np.round(canvas[: min(final_frames, len(canvas))]), -32768, 32767).astype(np.int16)

    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    with wave.open(out_wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(final_pcm.tobytes())
    return out_wav_path


def merge_video_audio():
    """Merge video, retimed background M&E track, and 24kHz dubbed vocals with frame-accurate sync."""
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE

    if not load_key("burn_subtitles"):
        rprint(
            "[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]"
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()
        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Prefer lossless 24kHz WAV (`output/dub.wav`), fallback to `output/dub.mp3`
    dub_source = DUB_WAV if os.path.exists(DUB_WAV) else DUB_AUDIO
    if not os.path.exists(dub_source):
        rprint(
            f"[bold red]Error: Dub audio file {dub_source} not found. Please ensure audio generation completed.[/bold red]"
        )
        return

    normalized_dub_audio = "output/normalized_dub.wav"
    normalize_audio_volume(dub_source, normalized_dub_audio)

    # Ensure background audio exists
    if not os.path.exists(background_file):
        if load_key("demucs"):
            rprint(
                f"[bold yellow]Warning: Background audio {background_file} missing, reconstructing via Demucs...[/bold yellow]"
            )
            if not os.path.exists(_RAW_AUDIO_FILE):
                convert_video_to_audio(VIDEO_FILE)
            try:
                demucs_audio()
            except Exception as e:
                rprint(f"[bold red]Error running Demucs: {e}[/bold red]")

        if not os.path.exists(background_file):
            background_file = _RAW_AUDIO_FILE
            if not os.path.exists(background_file):
                convert_video_to_audio(VIDEO_FILE)
                if not os.path.exists(background_file):
                    rprint(f"[bold red]Error: Failed to reconstruct original audio {background_file}.[/bold red]")
                    return

    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = float(video.get(cv2.CAP_PROP_FPS) or 30.0)
    if orig_fps <= 1.0 or orig_fps > 120.0:
        orig_fps = 30.0
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {orig_fps:.2f}fps[/bold green]")

    subtitle_filter = (
        f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
        f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
        f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
    )

    # Check if dynamic piecewise video retiming map exists from Doubao V3 elastic alignment
    seg_map = None
    if os.path.exists(VIDEO_SEGMENTS_MAP_FILE):
        try:
            with open(VIDEO_SEGMENTS_MAP_FILE, "r", encoding="utf-8") as f:
                loaded_map = json.load(f)
            if isinstance(loaded_map, dict) and loaded_map.get("intervals"):
                seg_map = loaded_map
        except Exception:
            seg_map = None

    has_piecewise_retiming = bool(seg_map and seg_map.get("has_dynamic_retiming"))

    try:
        video_slow_factor = float(load_key("video_slow_factor") or 1.0)
    except Exception:
        video_slow_factor = 1.0

    bg_input_path = background_file
    if has_piecewise_retiming:
        intervals = seg_map["intervals"]
        rprint(
            f"[bold cyan]🎬 Applying local smooth video retiming across {seg_map.get('retimed_paragraph_count', 0)} "
            f"speech paragraph(s) (+{seg_map.get('total_added_seconds', 0.0):.2f}s total)...[/bold cyan]"
        )
        bg_input_path = _build_retimed_background_wav(background_file, intervals, RETIMED_BG_AUDIO_FILE, sr=24000)
        pts_expr = _build_pts_binary_tree(intervals)
        video_pts_filter = f"setpts='({pts_expr})/TB',fps={orig_fps:.4f},"
        bg_audio_filter = ""
        bg_audio_input = "[1:a]"
    elif video_slow_factor != 1.0:
        rprint(f"[bold cyan]🎬 Applying global video slowdown: {video_slow_factor:.2f}x[/bold cyan]")
        video_pts_filter = f"setpts=PTS/{video_slow_factor},"
        bg_audio_filter = f"[1:a]{_atempo_filter_str(video_slow_factor)}[a_bg];"
        bg_audio_input = "[a_bg]"
    else:
        video_pts_filter = ""
        bg_audio_filter = ""
        bg_audio_input = "[1:a]"

    filter_complex_content = (
        f"[0:v]{video_pts_filter}scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"{subtitle_filter}[v];"
        f"{bg_audio_filter}"
        f"{bg_audio_input}[2:a]amix=inputs=2:duration=longest:dropout_transition=3[a]"
    )

    os.makedirs(os.path.dirname(FILTER_SCRIPT_FILE), exist_ok=True)
    with open(FILTER_SCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(filter_complex_content)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        VIDEO_FILE,
        "-i",
        bg_input_path,
        "-i",
        normalized_dub_audio,
        "-filter_complex_script",
        FILTER_SCRIPT_FILE,
    ]

    encoder = get_ffmpeg_video_encoder()
    if encoder:
        rprint(f"[bold green]Using hardware video encoder: {encoder}[/bold green]")
        cmd.extend(["-map", "[v]", "-map", "[a]", "-c:v", encoder])
        if encoder == "h264_videotoolbox":
            cmd.extend(["-b:v", "3500k"])
        elif encoder == "h264_nvenc":
            cmd.extend(["-cq", "26"])
    else:
        cmd.extend(["-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "23"])

    cmd.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "faststart",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-b:a",
            "192k",
            DUB_VIDEO,
        ]
    )

    try:
        from core.utils.cancel_utils import (
            TaskCancelledError,
            check_task_cancelled,
            is_task_cancelled,
            register_cancellable_process,
        )

        check_task_cancelled()
        proc = subprocess.Popen(cmd)
        with register_cancellable_process(proc):
            ret = proc.wait()
        if is_task_cancelled():
            if os.path.exists(DUB_VIDEO):
                try:
                    os.remove(DUB_VIDEO)
                except Exception:
                    pass
            raise TaskCancelledError("任务已被手动停止")
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
        rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")
    except TaskCancelledError:
        if os.path.exists(DUB_VIDEO):
            try:
                os.remove(DUB_VIDEO)
            except Exception:
                pass
        raise
    except subprocess.CalledProcessError as e:
        rprint(f"[bold red]Error: FFmpeg process failed with return code {e.returncode}[/bold red]")
        raise
    except Exception as e:
        rprint(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        raise


if __name__ == "__main__":
    merge_video_audio()
