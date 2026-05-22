import os
import platform
import subprocess

import cv2
import numpy as np
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume, convert_video_to_audio
from core.asr_backend.demucs_vl import demucs_audio
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE
    
    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Check if dub audio exists
    if not os.path.exists(DUB_AUDIO):
        rprint(f"[bold red]Error: Dub audio file {DUB_AUDIO} not found. Please ensure the audio generation step completed successfully.[/bold red]")
        return

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    
    # Check if background audio exists, fallback to raw audio if missing
    if not os.path.exists(background_file):
        if load_key("demucs"):
            rprint(f"[bold yellow]Warning: Background audio {background_file} not found but Demucs is enabled. Attempting to reconstruct...[/bold yellow]")
            if not os.path.exists(_RAW_AUDIO_FILE):
                convert_video_to_audio(VIDEO_FILE)
            try:
                demucs_audio()
            except Exception as e:
                rprint(f"[bold red]Error running Demucs: {e}[/bold red]")
        
        # Re-check after possible reconstruction
        if not os.path.exists(background_file):
            rprint(f"[bold yellow]Warning: Background audio {background_file} still not found. Falling back to original audio.[/bold yellow]")
            background_file = _RAW_AUDIO_FILE
            if not os.path.exists(background_file):
                rprint(f"[bold yellow]Warning: Original audio {background_file} not found. Reconstructing from video...[/bold yellow]")
                convert_video_to_audio(VIDEO_FILE)
                if not os.path.exists(background_file):
                    rprint(f"[bold red]Error: Failed to reconstruct original audio {background_file}.[/bold red]")
                    return

    # Merge video and audio with translated subtitles
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    subtitle_filter = (
        f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
        f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
        f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
    )

    try:
        video_slow_factor = load_key("video_slow_factor")
    except KeyError:
        video_slow_factor = 1.0
    if video_slow_factor != 1.0:
        # PTS/factor where factor<1 stretches PTS values → slower video playback.
        # Background audio is slowed by the same ratio using atempo so it stays in sync.
        # The dubbed audio was already generated against scaled timestamps, so it needs no change.
        rprint(f"[bold cyan]🎬 Applying video slowdown: {video_slow_factor:.2f}x (setpts=PTS/{video_slow_factor})[/bold cyan]")
        video_pts_filter = f'setpts=PTS/{video_slow_factor},'
        bg_audio_filter = f'[1:a]atempo={video_slow_factor}[a_bg];'
        bg_audio_input = '[a_bg]'
    else:
        video_pts_filter = ''
        bg_audio_filter = ''
        bg_audio_input = '[1:a]'

    cmd = [
        'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', normalized_dub_audio,
        '-filter_complex',
        f'[0:v]{video_pts_filter}scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
        f'{subtitle_filter}[v];'
        f'{bg_audio_filter}'
        f'{bg_audio_input}[2:a]amix=inputs=2:duration=first:dropout_transition=3[a]'
    ]

    encoder = get_ffmpeg_video_encoder()
    if encoder:
        rprint(f"[bold green]Using hardware video encoder: {encoder}[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', encoder])
        if encoder == 'h264_videotoolbox':
            cmd.extend(['-b:v', '3000k'])
        elif encoder == 'h264_nvenc':
            cmd.extend(['-cq', '28'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-crf', '23'])
    
    # 🌐 Add web-compatible parameters
    cmd.extend([
        '-pix_fmt', 'yuv420p',    # Essential for browser compatibility
        '-movflags', 'faststart', # Allow video to start playing before fully downloaded
        '-c:a', 'aac', '-b:a', '96k', DUB_VIDEO
    ])
    
    try:
        # Remove stderr=subprocess.PIPE to avoid buffer deadlock for long-running FFmpeg processes
        # This allows you to see the encoding progress in the console.
        subprocess.run(cmd, check=True)
        rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")
    except subprocess.CalledProcessError as e:
        rprint(f"[bold red]Error: FFmpeg process failed with return code {e.returncode}[/bold red]")
    except Exception as e:
        rprint(f"[bold red]An unexpected error occurred: {e}[/bold red]")


if __name__ == '__main__':
    merge_video_audio()
