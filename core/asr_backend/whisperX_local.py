import os
import warnings
import time
import subprocess
import torch
import whisperx
import librosa
from rich import print as rprint
from core.utils import *

warnings.filterwarnings("ignore")
MODEL_DIR = load_key("model_dir")

@except_handler("failed to check hf mirror", default_return=None)
def check_hf_mirror():
    mirrors = {'Official': 'huggingface.co', 'Mirror': 'hf-mirror.com'}
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        if os.name == 'nt':
            cmd = ['ping', '-n', '1', '-w', '3000', domain]
        else:
            cmd = ['ping', '-c', '1', '-W', '3', domain]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        response_time = time.time() - start
        if result.returncode == 0:
            if response_time < best_time:
                best_time = response_time
                fastest_url = f"https://{domain}"
            rprint(f"[green]✓ {name}:[/green] {response_time:.2f}s")
    if best_time == float('inf'):
        rprint("[yellow]⚠️ All mirrors failed, using default[/yellow]")
    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url

@except_handler("WhisperX processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    os.environ['HF_ENDPOINT'] = check_hf_mirror()
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"🚀 Starting WhisperX using device: {device} ...")
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB, [cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    else:
        batch_size = 1
        compute_type = "int8"
        rprint(f"[cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    rprint(f"[green]▶️ Starting WhisperX for segment {start:.2f}s to {end:.2f}s...[/green]")
    
    if WHISPER_LANGUAGE == 'zh':
        model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
    else:
        model_name = load_key("whisper.model")
        local_model = os.path.join(MODEL_DIR, model_name)
        
    if os.path.exists(local_model):
        rprint(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
        model_name = local_model
    else:
        rprint(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

    # Build initial_prompt from domain vocabulary to bias Whisper toward specialized terms
    # For explicit pairs (wrong -> correct), only the correct term is included as a hint
    domain_vocab_str = load_key("domain_vocab")
    if domain_vocab_str and domain_vocab_str.strip():
        import re as _re
        raw_terms = [t.strip() for t in domain_vocab_str.replace('\n', ',').split(',') if t.strip()]
        prompt_terms = []
        for term in raw_terms:
            parts = _re.split(r'\s*(?:->|→)\s*', term, maxsplit=1)
            prompt_terms.append(parts[1].strip() if len(parts) == 2 and parts[1].strip() else term)
        initial_prompt = ", ".join(prompt_terms) + "."
        rprint(f"[cyan]📖 Using domain vocab as initial prompt: {initial_prompt}[/cyan]")
    else:
        initial_prompt = ""

    vad_options = {"vad_onset": 0.300,"vad_offset": 0.200}
    asr_options = {"temperatures": [0],"initial_prompt": initial_prompt,"no_speech_threshold": 0.3,}
    whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
    rprint("[bold yellow] You can ignore warning of `Model was trained with torch 1.10.0+cu102, yours is 2.0.0+cu118...`[/bold yellow]")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=whisper_language, vad_options=vad_options, asr_options=asr_options, download_root=MODEL_DIR)

    def load_audio_segment(audio_file, start, end):
        audio, _ = librosa.load(audio_file, sr=16000, offset=start, duration=end - start, mono=True)
        return audio
    raw_audio_segment = load_audio_segment(raw_audio_file, start, end)
    vocal_audio_segment = load_audio_segment(vocal_audio_file, start, end)
    
    # -------------------------
    # 1. transcribe raw audio
    # -------------------------
    transcribe_start_time = time.time()
    rprint("[bold green]Note: You will see Progress if working correctly ↓[/bold green]")
    result = model.transcribe(raw_audio_segment, batch_size=batch_size, print_progress=True)
    transcribe_time = time.time() - transcribe_start_time
    rprint(f"[cyan]⏱️ time transcribe:[/cyan] {transcribe_time:.2f}s")

    # -------------------------
    # 1b. Re-transcribe sparse segments
    # Whisper can stop early within a VAD-detected segment when audio context shifts
    # dramatically (e.g., transition to background music). Detect such segments by
    # checking word density, then independently re-transcribe the remaining portion.
    # -------------------------
    MIN_WORDS_PER_SEC = 0.8   # below this → likely stopped early
    MIN_SEG_DUR      = 5.0   # only check segments longer than this
    REFILL_EXTRA     = 5.0   # extend refill window by this many seconds for context
    filled = list(result.get('segments', []))
    i = 0
    while i < len(filled):
        seg = filled[i]
        dur = seg['end'] - seg['start']
        n_words = len(seg['text'].split())
        if dur > MIN_SEG_DUR and n_words / dur < MIN_WORDS_PER_SEC:
            refill_start = seg['start'] + start   # absolute file position
            refill_dur   = (seg['end'] - seg['start']) + REFILL_EXTRA
            rprint(f"[yellow]🔍 Sparse segment {seg['start']:.2f}s-{seg['end']:.2f}s "
                   f"({n_words}w/{dur:.1f}s = {n_words/dur:.2f} w/s), re-transcribing from file...[/yellow]")
            try:
                # Load directly from file to avoid array-slice acoustic artifacts
                gap_audio, _ = librosa.load(raw_audio_file, sr=16000,
                                            offset=refill_start, duration=refill_dur, mono=True)
                gap_result = model.transcribe(gap_audio, batch_size=batch_size, print_progress=False)
                new_segs = []
                for gs in gap_result.get('segments', []):
                    gs = dict(gs)
                    # Timestamps are relative to refill_start; shift to original time base
                    gs['start'] += seg['start']
                    gs['end']   += seg['start']
                    gs['words']  = [dict(w) for w in gs.get('words', [])]
                    for w in gs['words']:
                        if 'start' in w: w['start'] += seg['start']
                        if 'end'   in w: w['end']   += seg['start']
                    # Only keep segments that fall within this segment's range
                    if gs['start'] < seg['end'] + REFILL_EXTRA:
                        new_segs.append(gs)
                if new_segs and len(new_segs[0]['text'].split()) > n_words:
                    rprint(f"[green]✅ Refill got {len(new_segs)} segment(s): "
                           f"{[s['text'][:40] for s in new_segs]}[/green]")
                    filled[i] = new_segs[0]
                    for j, s in enumerate(new_segs[1:]):
                        filled.insert(i + 1 + j, s)
                    i += len(new_segs)
                    continue
                else:
                    rprint(f"[yellow]⚠️ Refill did not improve content, keeping original[/yellow]")
            except Exception as e:
                rprint(f"[red]❌ Sparse segment refill error: {e}[/red]")
        i += 1
    result['segments'] = filled

    # Free GPU resources
    del model
    torch.cuda.empty_cache()

    # Save language
    update_key("whisper.language", result['language'])
    if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
        raise ValueError("Please specify the transcription language as zh and try again!")

    # -------------------------
    # 2. align by vocal audio
    # -------------------------
    align_start_time = time.time()
    # Align timestamps using vocal audio
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, vocal_audio_segment, device, return_char_alignments=False)
    align_time = time.time() - align_start_time
    rprint(f"[cyan]⏱️ time align:[/cyan] {align_time:.2f}s")

    # Free GPU resources again
    torch.cuda.empty_cache()
    del model_a

    # Adjust timestamps
    for segment in result['segments']:
        segment['start'] += start
        segment['end'] += start
        for word in segment['words']:
            if 'start' in word:
                word['start'] += start
            if 'end' in word:
                word['end'] += start
    return result