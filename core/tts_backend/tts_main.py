import os
import re
import time
from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.doubao_tts import doubao_tts
from core.tts_backend.gemini_tts import gemini_tts
from core.tts_backend.gpt_sovits_tts import gpt_sovits_tts_for_videolingo
from core.tts_backend.sf_fishtts import siliconflow_fish_tts_for_videolingo
from core.tts_backend.openai_tts import openai_tts
from core.tts_backend.fish_tts import fish_tts
from core.tts_backend.azure_tts import azure_tts
from core.tts_backend.edge_tts import edge_tts
from core.tts_backend.sf_cosyvoice2 import cosyvoice_tts_for_videolingo
from core.tts_backend.custom_tts import custom_tts
from core.prompts import get_correct_text_prompt
from core.tts_backend._302_f5tts import f5_tts_for_videolingo
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
    
    # Skip if file exists
    if os.path.exists(save_as):
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
            if TTS_METHOD == 'openai_tts':
                openai_tts(text, save_as)
            elif TTS_METHOD == 'doubao_tts':
                doubao_tts(text, save_as)
            elif TTS_METHOD == 'gemini_tts':
                gemini_tts(text, save_as)
            elif TTS_METHOD == 'gpt_sovits':
                gpt_sovits_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'fish_tts':
                fish_tts(text, save_as)
            elif TTS_METHOD == 'azure_tts':
                azure_tts(text, save_as)
            elif TTS_METHOD == 'sf_fish_tts':
                siliconflow_fish_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'edge_tts':
                edge_tts(text, save_as)
            elif TTS_METHOD == 'custom_tts':
                custom_tts(text, save_as)
            elif TTS_METHOD == 'sf_cosyvoice2':
                cosyvoice_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'f5tts':
                f5_tts_for_videolingo(text, save_as, number, task_df)
                
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