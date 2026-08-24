"""
Speaker Diarization Pipeline Step (_2_5_diarize)

Optional pipeline step that runs pyannote speaker diarization after ASR transcription.
This pre-computes and caches speaker identity labels for each time segment.
The cached results are later used by classify_subtitles_gender() in _8_1_audio_task
to do cluster-level gender classification instead of per-sentence analysis.

This step gracefully skips if:
- pyannote is not installed
- HuggingFace token is not configured
- Diarization results are already cached

Usage in pipeline:
    _1_ytdlp → _2_asr → _2_5_diarize (this) → _3_split → ... → _8_1_audio_task
"""

import os

from core.utils import rprint, load_key
from core.utils.models import _VOCAL_AUDIO_FILE, _RAW_AUDIO_FILE


def diarize_main():
    """Run speaker diarization and cache results for downstream gender classification."""

    # Check if pyannote is configured
    try:
        hf_token = load_key("pyannote.hf_token")
    except Exception:
        hf_token = ""

    if not hf_token:
        rprint("[yellow]ℹ️ No HuggingFace token configured (pyannote.hf_token is empty).[/yellow]")
        rprint("[yellow]   Skipping speaker diarization. Gender classification will use GMM-only method.[/yellow]")
        rprint("[yellow]   To enable enhanced diarization, set pyannote.hf_token in config.yaml.[/yellow]")
        return

    # Check if pyannote is installed
    try:
        from core.utils.speaker_diarize import (
            run_diarization,
            save_diarization_result,
            load_diarization_cache,
            DIARIZE_CACHE_FILE,
        )
    except ImportError:
        rprint("[yellow]⚠️ pyannote.audio is not installed.[/yellow]")
        rprint("[yellow]   Install with: pip install 'pyannote.audio>=3.1'[/yellow]")
        rprint("[yellow]   Skipping speaker diarization.[/yellow]")
        return

    # Check if results are already cached
    cached = load_diarization_cache()
    if cached is not None:
        rprint(f"[yellow]⚠️ Diarization results already cached ({len(cached)} segments). Skipping.[/yellow]")
        return

    # Determine audio file
    audio_path = _VOCAL_AUDIO_FILE if os.path.exists(_VOCAL_AUDIO_FILE) else _RAW_AUDIO_FILE
    if not os.path.exists(audio_path):
        rprint(f"[red]❌ Audio file not found at {audio_path}. Cannot run diarization.[/red]")
        return

    # Read speaker count config
    try:
        min_speakers = int(load_key("pyannote.min_speakers"))
    except Exception:
        min_speakers = 1
    try:
        max_speakers = int(load_key("pyannote.max_speakers"))
    except Exception:
        max_speakers = 10

    # Run diarization
    rprint("[bold cyan]🎙️ Running speaker diarization...[/bold cyan]")
    try:
        diarization = run_diarization(audio_path, hf_token, min_speakers, max_speakers)
        save_diarization_result(diarization)
        rprint("[bold green]✅ Speaker diarization completed and cached.[/bold green]")
    except Exception as e:
        rprint(f"[red]❌ Speaker diarization failed: {e}[/red]")
        err_str = str(e).lower()
        if "gated" in err_str or "403" in err_str or "restricted" in err_str:
            rprint("[bold yellow]💡 HuggingFace gated model access required. Please make sure your token has access to:[/bold yellow]")
            rprint("[cyan]   1. https://huggingface.co/pyannote/speaker-diarization-3.1[/cyan]")
            rprint("[cyan]   2. https://huggingface.co/pyannote/segmentation-3.0[/cyan]")
            rprint("[cyan]   3. https://huggingface.co/pyannote/speaker-diarization-community-1[/cyan]")
        rprint("[yellow]   Gender classification will fall back to GMM-only method.[/yellow]")


if __name__ == "__main__":
    diarize_main()
