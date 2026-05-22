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
    """Internal helper to make the raw API call with Persona Anchoring strategy"""
    url = f"{host}/v1beta/models/{model}:generateContent?key={key}"
    
    # PERSONA ANCHORING: Giving the model a clear, consistent identity tag at the start of every request.
    # This helps the model "find" the same voice profile each time.
    full_prompt = f"[Speaker: {voice}, Professional Narrator] Text: {text.strip()}"
    
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice
                    }
                }
            },
            "temperature": 0.3, # Balanced for stability and successful generation
            "topP": 0.8,
            "topK": 40
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
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    return response

@except_handler("Failed to generate audio using Gemini TTS", retry=1, delay=1)
def gemini_tts(text, save_path):
    """Generate audio using Gemini native multimodal output with Persona Anchoring"""
    
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    if not API_KEY:
        raise ValueError("No Gemini API Key found. Please set it in the sidebar or config.yaml.")
        
    host = "https://generativelanguage.googleapis.com"
    voice = load_key("gemini_tts.voice") or "Aoede"
    
    # We use the most proven multimodal audio model
    PRIMARY_MODEL = "gemini-2.5-flash-preview-tts"
    
    print(f"DEBUG: [PERSONA ANCHORING] Narrating: '{text[:15]}...' with {PRIMARY_MODEL}/{voice}")
    
    try:
        response = call_gemini_api_raw(text, PRIMARY_MODEL, API_KEY, voice, host)
        
        if response.status_code == 200:
            data = response.json()
            candidate = data.get('candidates', [{}])[0]
            finish_reason = candidate.get('finishReason')
            
            if finish_reason in ['STOP', None]:
                parts = candidate.get('content', {}).get('parts', [])
                audio_data = next((p['inlineData']['data'] for p in parts if 'inlineData' in p), None)
                
                if audio_data:
                    raw_audio_bytes = base64.b64decode(audio_data)
                    audio_segment = AudioSegment.from_raw(
                        io.BytesIO(raw_audio_bytes),
                        sample_width=2, frame_rate=24000, channels=1
                    )
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    audio_segment.export(save_path, format="wav")
                    print(f"✅ Gemini Success: {PRIMARY_MODEL}")
                    return # SUCCESS
            
            error_msg = f"FinishReason: {finish_reason}"
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        error_msg = str(e)
            
    # Final fallback attempt with 3.1 if 2.5 fails
    FALLBACK_MODEL = "gemini-3.1-flash-tts-preview"
    try:
        response = call_gemini_api_raw(text, FALLBACK_MODEL, API_KEY, voice, host)
        if response.status_code == 200:
            data = response.json()
            candidate = data.get('candidates', [{}])[0]
            if candidate.get('finishReason') in ['STOP', None]:
                parts = candidate.get('content', {}).get('parts', [])
                audio_data = next((p['inlineData']['data'] for p in parts if 'inlineData' in p), None)
                if audio_data:
                    raw_audio_bytes = base64.b64decode(audio_data)
                    audio_segment = AudioSegment.from_raw(io.BytesIO(raw_audio_bytes), sample_width=2, frame_rate=24000, channels=1)
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    audio_segment.export(save_path, format="wav")
                    print(f"✅ Gemini Success: {FALLBACK_MODEL} (Fallback)")
                    return # SUCCESS
    except:
        pass

    raise ValueError(f"Gemini TTS consistency logic failed. Error: {error_msg}")

if __name__ == "__main__":
    pass
