import os
import re
import time
from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.doubao_tts import doubao_tts
from core.tts_backend.edge_tts import edge_tts
from core.prompts import get_correct_text_prompt
from core.utils import *

def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    chars_to_remove = ['&', '®', '™', '©']
    for char in chars_to_remove:
        text = text.replace(char, '')
    return text.strip()

def tts_main(text, save_as, number, task_df):
    text = clean_text_for_tts(text)
    # Check if text is empty or single character, single character voiceovers are prone to bugs
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if not cleaned_text or len(cleaned_text) <= 1:
        silence = AudioSegment.silent(duration=100)  # 100ms = 0.1s
        silence.export(save_as, format="wav")
        rprint(f"Created silent audio for empty/single-char text: {save_as}")
        return
    
    # Skip if file exists and is non-empty
    if os.path.exists(save_as) and os.path.getsize(save_as) > 0:
        return
    
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