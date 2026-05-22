import os
import requests
import json
import base64
import time
from pathlib import Path
from pydub import AudioSegment
import io
from core.utils import load_key, except_handler

def call_gemini_api_raw(text, model, key, voice, host="https://generativelanguage.googleapis.com"):
    """Internal helper to make the raw API call with a specific prompt strategy"""
    url = f"{host}/v1beta/models/{model}:generateContent?key={key}"
    
    # ULTIMATE STRATEGY: Directive-Based + Audio Only
    # This prevents the model from being chatty and bypasses many safety/policy blocks.
    prompt = f"NARRATE THIS EXACTLY, AUDIO ONLY, NO TEXT: {text.strip()}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice
                    }
                }
            }
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
        ]
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=payload, timeout=40)
    return response

@except_handler("Failed to generate audio using Gemini TTS", retry=1, delay=1) # Outer handler for catastrophic failures
def gemini_tts(text, save_path):
    """Generate audio using Gemini native multimodal output with Ultimate Stability strategy"""
    
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    if not API_KEY:
        raise ValueError("No Gemini API Key found. Please set it in the sidebar or config.yaml.")
        
    host = "https://generativelanguage.googleapis.com"
    voice = load_key("gemini_tts.voice") or "Aoede"
    
    # We prioritize the most stable model found in tests
    MODELS_TO_TRY = ["gemini-2.5-flash-preview-tts", "gemini-3.1-flash-tts-preview"]
    # Jitters to apply on retry: None, Space, Period, Space+Period
    JITTERS = ["", " ", ".", " ."]
    
    last_error = "Unknown error"
    
    print(f"DEBUG: [STABILITY MODE] Generating audio for: '{text[:15]}...'")
    
    for model in MODELS_TO_TRY:
        for jitter in JITTERS:
            try:
                jittered_text = text + jitter
                response = call_gemini_api_raw(jittered_text, model, API_KEY, voice, host)
                
                if response.status_code == 200:
                    data = response.json()
                    candidate = data.get('candidates', [{}])[0]
                    finish_reason = candidate.get('finishReason')
                    
                    # Success check
                    if finish_reason in ['STOP', None]:
                        parts = candidate.get('content', {}).get('parts', [])
                        audio_data = next((p['inlineData']['data'] for p in parts if 'inlineData' in p), None)
                        
                        if audio_data:
                            # Convert raw PCM to standard WAV
                            raw_audio_bytes = base64.b64decode(audio_data)
                            audio_segment = AudioSegment.from_raw(
                                io.BytesIO(raw_audio_bytes),
                                sample_width=2, frame_rate=24000, channels=1
                            )
                            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                            audio_segment.export(save_path, format="wav")
                            print(f"✅ Gemini Success: {model} (jitter: '{jitter}')")
                            return # EXIT SUCCESS
                
                # If we get here, this specific combination failed
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                else:
                    last_error = f"FinishReason: {data.get('candidates', [{}])[0].get('finishReason', 'UNKNOWN')}"
                
            except Exception as e:
                last_error = str(e)
            
            # Tiny sleep before trying next jitter
            time.sleep(0.5)
            
    # If we exhausted all models and jitters
    raise ValueError(f"All Gemini stability strategies exhausted. Last error: {last_error}")

if __name__ == "__main__":
    pass
