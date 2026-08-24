"""
Speaker Diarization Module using pyannote.audio

Provides speaker diarization to enhance voice gender classification accuracy.
Instead of classifying gender per-sentence (which is noisy due to intonation),
this module first groups sentences by speaker identity (voice embeddings),
then classifies gender at the speaker-cluster level for much more stable results.

Requires:
  - pyannote.audio >= 3.1
  - A HuggingFace token with access to pyannote/speaker-diarization-3.1
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from core.utils import rprint, load_key
from core.utils.models import _VOCAL_AUDIO_FILE, _RAW_AUDIO_FILE

# Cache file for diarization results
DIARIZE_CACHE_FILE = "output/audio/speaker_diarize.json"


def _get_audio_path() -> str:
    """Determine the best audio file to use for diarization."""
    if os.path.exists(_VOCAL_AUDIO_FILE):
        return _VOCAL_AUDIO_FILE
    return _RAW_AUDIO_FILE


def run_diarization(
    audio_path: str,
    hf_token: str,
    min_speakers: int = 1,
    max_speakers: int = 10,
) -> "pyannote.core.Annotation":
    """
    Run pyannote speaker diarization on the given audio file.

    Args:
        audio_path: Path to the audio file (WAV/MP3).
        hf_token: HuggingFace access token.
        min_speakers: Minimum expected number of speakers.
        max_speakers: Maximum expected number of speakers.

    Returns:
        pyannote.core.Annotation object with speaker segments.
    """
    # ── Network Setup for HuggingFace Model Download ──────────────────────────
    # pyannote models are "gated" on HuggingFace (require authentication).
    # hf-mirror.com CANNOT proxy gated models (redirects back to huggingface.co).
    # Therefore, we need a real network proxy to access huggingface.co directly.
    #
    # Priority:
    # 1. Use VideoLingo's configured proxy (config.yaml → proxy) to access huggingface.co
    # 2. Fall back to HF_ENDPOINT mirror (for non-gated models only)

    # Set up network proxy from VideoLingo config
    try:
        proxy = load_key("proxy")
    except Exception:
        proxy = ""
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        rprint(f"[cyan]🌐 Using network proxy for HuggingFace: {proxy}[/cyan]")

    # Also support HF_ENDPOINT mirror (may work for non-gated models)
    try:
        hf_endpoint = load_key("pyannote.hf_endpoint")
    except Exception:
        hf_endpoint = ""
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        try:
            import huggingface_hub.constants
            huggingface_hub.constants.ENDPOINT = hf_endpoint
        except Exception:
            pass
        rprint(f"[cyan]🌐 HF Endpoint: {hf_endpoint}[/cyan]")

    if not proxy and not hf_endpoint:
        rprint("[yellow]⚠️ No proxy configured. pyannote models require access to huggingface.co.[/yellow]")
        rprint("[yellow]   Set 'proxy' in config.yaml (e.g., http://127.0.0.1:7890) to download gated models.[/yellow]")

    from pyannote.audio import Pipeline

    rprint("[bold cyan]🎯 Loading pyannote speaker-diarization-3.1 model...[/bold cyan]")

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

    # Use GPU if available
    import torch
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        rprint("[green]✓ Using CUDA GPU for diarization[/green]")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        rprint("[green]✓ Using Apple MPS for diarization[/green]")
    else:
        rprint("[yellow]ℹ️ Using CPU for diarization (this may take a while)[/yellow]")

    rprint(f"[cyan]🔊 Running speaker diarization on {audio_path}...[/cyan]")

    diarization = pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # Log discovered speakers
    speakers = set()
    for _, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
    rprint(f"[bold green]✨ Diarization complete: found {len(speakers)} speaker(s): {sorted(speakers)}[/bold green]")

    return diarization


def assign_speakers_to_subtitles(
    diarization,
    df_tasks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign speaker IDs to each subtitle line based on temporal overlap (IoU).

    For each subtitle, finds the diarization segment with the greatest time
    overlap and assigns that segment's speaker label.

    Args:
        diarization: pyannote Annotation object from run_diarization().
        df_tasks: DataFrame with 'start_time' and 'end_time' columns.

    Returns:
        DataFrame with a new 'speaker_id' column.
    """
    from core.utils.audio_gender import parse_time_to_seconds

    # Build a list of (start_sec, end_sec, speaker_label) from diarization
    diar_segments: List[Tuple[float, float, str]] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        diar_segments.append((segment.start, segment.end, speaker))

    speaker_ids = []

    for _, row in df_tasks.iterrows():
        sub_start = parse_time_to_seconds(row["start_time"])
        sub_end = parse_time_to_seconds(row["end_time"])
        sub_duration = sub_end - sub_start

        best_speaker = None
        best_overlap = 0.0

        for seg_start, seg_end, speaker in diar_segments:
            # Calculate overlap
            overlap_start = max(sub_start, seg_start)
            overlap_end = min(sub_end, seg_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        # If no overlap found (rare edge case), assign to first speaker
        if best_speaker is None and diar_segments:
            best_speaker = diar_segments[0][2]
        elif best_speaker is None:
            best_speaker = "SPEAKER_00"

        speaker_ids.append(best_speaker)

    df_tasks["speaker_id"] = speaker_ids

    # Log speaker distribution
    speaker_counts = pd.Series(speaker_ids).value_counts()
    rprint("[cyan]📊 Speaker distribution in subtitles:[/cyan]")
    for speaker, count in speaker_counts.items():
        rprint(f"  {speaker}: {count} lines")

    return df_tasks


def save_diarization_result(diarization, cache_file: str = DIARIZE_CACHE_FILE) -> None:
    """Save diarization results to a JSON cache file."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "speaker": speaker,
        })

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    rprint(f"[green]✓ Diarization results cached to {cache_file} ({len(segments)} segments)[/green]")


def load_diarization_cache(cache_file: str = DIARIZE_CACHE_FILE) -> Optional[List[Dict]]:
    """Load cached diarization results if available and valid."""
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            segments = json.load(f)
        if isinstance(segments, list) and len(segments) > 0:
            rprint(f"[green]✓ Loaded cached diarization ({len(segments)} segments)[/green]")
            return segments
    except (json.JSONDecodeError, OSError) as e:
        rprint(f"[yellow]⚠️ Failed to load diarization cache: {e}[/yellow]")

    return None


def assign_speakers_from_cache(
    cached_segments: List[Dict],
    df_tasks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign speaker IDs from cached diarization segments (same logic as
    assign_speakers_to_subtitles but works with dict-based cache data).
    """
    from core.utils.audio_gender import parse_time_to_seconds

    diar_segments = [(s["start"], s["end"], s["speaker"]) for s in cached_segments]
    speaker_ids = []

    for _, row in df_tasks.iterrows():
        sub_start = parse_time_to_seconds(row["start_time"])
        sub_end = parse_time_to_seconds(row["end_time"])

        best_speaker = None
        best_overlap = 0.0

        for seg_start, seg_end, speaker in diar_segments:
            overlap_start = max(sub_start, seg_start)
            overlap_end = min(sub_end, seg_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        if best_speaker is None and diar_segments:
            best_speaker = diar_segments[0][2]
        elif best_speaker is None:
            best_speaker = "SPEAKER_00"

        speaker_ids.append(best_speaker)

    df_tasks["speaker_id"] = speaker_ids
    return df_tasks
