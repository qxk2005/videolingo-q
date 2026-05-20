"""
MLX-Whisper backend for Apple Silicon (M-series chips).

Transcription is handled by mlx-whisper which uses Apple's MLX framework
and runs on the M-chip Neural Engine / GPU, giving 3-5x speed gains over CPU.
Word-level timestamp alignment is then performed by whisperX on CPU (fast, ~seconds).

Install dependency: pip install mlx-whisper
"""
import os
import re
import time
import warnings

import librosa
import whisperx
from rich import print as rprint

from core.utils import *

warnings.filterwarnings("ignore")

# ------------
# mlx-community HuggingFace repos, keyed by whisper model name
# ------------
_MLX_MODEL_MAP = {
    "large-v3":        "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo":  "mlx-community/whisper-large-v3-turbo",
    "medium":          "mlx-community/whisper-medium-mlx",
    "small":           "mlx-community/whisper-small-mlx",
    "base":            "mlx-community/whisper-base-mlx",
    "tiny":            "mlx-community/whisper-tiny-mlx",
}


def _build_initial_prompt():
    domain_vocab_str = load_key("domain_vocab")
    if not domain_vocab_str or not domain_vocab_str.strip():
        return ""
    raw_terms = [t.strip() for t in domain_vocab_str.replace('\n', ',').split(',') if t.strip()]
    prompt_terms = []
    for term in raw_terms:
        parts = re.split(r'\s*(?:->|→)\s*', term, maxsplit=1)
        prompt_terms.append(parts[1].strip() if len(parts) == 2 and parts[1].strip() else term)
    prompt = ", ".join(prompt_terms) + "."
    rprint(f"[cyan]📖 MLX domain vocab prompt: {prompt}[/cyan]")
    return prompt


@except_handler("MLX Whisper processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    try:
        import mlx_whisper
    except ImportError:
        raise ImportError(
            "mlx-whisper is not installed. Run: pip install mlx-whisper\n"
            "Then restart and select the 'mlx' runtime."
        )

    WHISPER_LANGUAGE = load_key("whisper.language")

    # ------------
    # Chinese model has no MLX variant - fall back to CPU local backend
    # ------------
    if WHISPER_LANGUAGE == 'zh':
        rprint("[yellow]⚠️ Chinese Whisper model has no MLX variant, falling back to CPU local backend[/yellow]")
        from core.asr_backend.whisperX_local import transcribe_audio as cpu_transcribe
        return cpu_transcribe(raw_audio_file, vocal_audio_file, start, end)

    model_name = load_key("whisper.model")
    mlx_repo = _MLX_MODEL_MAP.get(model_name, _MLX_MODEL_MAP["large-v3"])
    whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
    initial_prompt = _build_initial_prompt() or None

    rprint(f"[bold cyan]🍎 MLX Whisper | model: {mlx_repo} | segment {start:.1f}s-{end:.1f}s[/bold cyan]")

    def load_segment(audio_file):
        audio, _ = librosa.load(audio_file, sr=16000, offset=start, duration=end - start, mono=True)
        return audio

    raw_segment   = load_segment(raw_audio_file)
    vocal_segment = load_segment(vocal_audio_file)

    # ------------
    # 1. Transcribe with mlx-whisper (Apple Silicon accelerated)
    # ------------
    t0 = time.time()
    result = mlx_whisper.transcribe(
        raw_segment,
        path_or_hf_repo=mlx_repo,
        language=whisper_language,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        verbose=False,
    )
    rprint(f"[cyan]⏱️ MLX transcribe: {time.time() - t0:.2f}s[/cyan]")

    detected_lang = result.get("language", WHISPER_LANGUAGE or "en")
    update_key("whisper.language", detected_lang)

    if detected_lang == 'zh' and WHISPER_LANGUAGE != 'zh':
        raise ValueError("Detected Chinese audio. Please set transcription language to 'zh' and retry.")

    # ------------
    # 2. Align word timestamps using whisperX on CPU (fast: typically < 10s)
    # ------------
    rprint("[cyan]🔧 WhisperX word alignment on CPU...[/cyan]")
    t1 = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device="cpu")
    aligned = whisperx.align(
        result["segments"], model_a, metadata,
        vocal_segment, "cpu",
        return_char_alignments=False,
    )
    del model_a
    rprint(f"[cyan]⏱️ Align: {time.time() - t1:.2f}s[/cyan]")

    # ------------
    # 3. Shift timestamps back to absolute file position
    # ------------
    for segment in aligned["segments"]:
        segment["start"] += start
        segment["end"]   += start
        for word in segment.get("words", []):
            if "start" in word: word["start"] += start
            if "end"   in word: word["end"]   += start

    aligned["language"] = detected_lang
    return aligned
