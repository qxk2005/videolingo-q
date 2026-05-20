from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume, apply_domain_vocab_correction
from core._1_ytdlp import find_video_files
from core.utils.models import *

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # 2. Demucs vocal separation:
    if load_key("demucs"):
        demucs_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(_RAW_AUDIO_FILE)
    
    # 4. Transcribe audio by clips
    all_results = []
    runtime = load_key("whisper.runtime")
    if runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
    elif runtime == "mlx":
        from core.asr_backend.whisper_mlx import transcribe_audio as ts
        rprint("[cyan]🍎 Transcribing audio with MLX Whisper (Apple Silicon)...[/cyan]")
    elif runtime == "cloud":
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
        rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")

    # When demucs is enabled, use vocal-separated audio for transcription to eliminate
    # background music interference with VAD and ASR accuracy.
    transcription_audio = vocal_audio if load_key("demucs") else _RAW_AUDIO_FILE
    
    from core.utils.progress_utils import get_progress, update_st_progress
    progress = get_progress()
    # Check if progress is already running (e.g. from batch mode)
    is_internal = not progress.live.is_started
    
    if is_internal:
        progress.start()
        
    task_desc = "🎤 正在转录音频分段..."
    task = progress.add_task(task_desc, total=len(segments))
    for i, (start, end) in enumerate(segments):
        result = ts(transcription_audio, vocal_audio, start, end)
        all_results.append(result)
        progress.update(task, advance=1)
        update_st_progress(i + 1, len(segments), task_desc)
        
    if is_internal:
        progress.stop()
    else:
        progress.remove_task(task)
    
    # 5. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 6. Process df
    df = process_transcription(combined_result)
    df = apply_domain_vocab_correction(df)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()