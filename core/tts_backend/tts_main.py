import os
import re
import time
import subprocess
from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.utils import rprint
from core.tts_backend.doubao_tts import doubao_tts
from core.tts_backend.edge_tts import edge_tts
from core.prompts import get_correct_text_prompt
from core.utils import *

def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    chars_to_remove = ['&', '®', '™', '©']
    for char in chars_to_remove:
        text = text.replace(char, '')
    # Remove music/sound markers like [music], (laughter)
    text = re.sub(r'\[[^\]]+\]|\([^)]+\)|（[^）]+）', '', text)
    return text.strip()

def is_audio_valid(file_path: str, text: str) -> bool:
    """Validate if the generated audio is healthy and valid."""
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
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

def tts_main(text, save_as, number, task_df):
    text = clean_text_for_tts(text)
    # Check if text is empty or single character, single character voiceovers are prone to bugs
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if not cleaned_text or len(cleaned_text) <= 1:
        silence = AudioSegment.silent(duration=100)  # 100ms = 0.1s
        silence.export(save_as, format="wav")
        rprint(f"Created silent audio for empty/single-char text: {save_as}")
        return
    
    # Skip if file exists, is non-empty and is valid
    if os.path.exists(save_as) and os.path.getsize(save_as) > 0:
        if is_audio_valid(save_as, text):
            return
        else:
            rprint(f"[yellow]⚠️ Detected invalid/corrupted audio cache <{save_as}> for text: '{text}'. Removing and regenerating...[/yellow]")
            try:
                os.remove(save_as)
            except Exception:
                pass
    
    print(f"Generating <{text}...>")
    TTS_METHOD = load_key("tts_method")
    
    max_retries = 3
    last_error_is_network = False
    for attempt in range(max_retries):
        try:
            if attempt >= max_retries - 1 and not last_error_is_network:
                print("Asking GPT to correct text...")
                correct_text = ask_gpt(get_correct_text_prompt(text), resp_type="json", log_title='tts_correct_text')
                text = correct_text['text']
            
            if TTS_METHOD == 'edge_tts':
                edge_tts(text, save_as)
            elif TTS_METHOD == 'doubao_tts':
                doubao_tts(text, save_as)
            else:
                raise ValueError(f"Unsupported TTS method: {TTS_METHOD}")
                
            # Check generated audio duration
            duration = get_audio_duration(save_as)
            if duration > 0:
                break
            else:
                if os.path.exists(save_as):
                    os.remove(save_as)
                if attempt == max_retries - 1:
                    print(f"Warning: Generated audio duration is 0 for text: {text}")
                    # Create silent audio file
                    silence = AudioSegment.silent(duration=100)  # 100ms silence
                    silence.export(save_as, format="wav")
                    return
                print(f"Attempt {attempt + 1} failed, retrying...")
        except Exception as e:
            import subprocess as _sp
            stderr_info = e.stderr if isinstance(e, _sp.CalledProcessError) and e.stderr else ''
            err_msg = f"{str(e)}" + (f"\nstderr: {stderr_info}" if stderr_info else "")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {err_msg}")
            # Detect network errors and apply backoff; skip GPT text correction on next retry
            last_error_is_network = any(kw in err_msg for kw in ('ConnectionReset', 'ClientConnector', 'Cannot connect', 'TimeoutError', 'aiohttp'))
            wait = 3 * (attempt + 1)
            print(f"Attempt {attempt + 1} failed ({'network error, retrying' if last_error_is_network else err_msg}), retrying in {wait}s...")
            time.sleep(wait)