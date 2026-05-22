import requests
import json
from core.utils import load_key

def test_gemini_exp_audio():
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    model = "gemini-2.0-flash-exp"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "Hello, testing audio output."
            }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": "Aoede"
                    }
                }
            }
        }
    }
    
    headers = {'Content-Type': 'application/json'}
    
    print(f"Directly testing {model} for AUDIO modality...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            audio_found = False
            for part in data.get('candidates', [{}])[0].get('content', {}).get('parts', []):
                if 'inlineData' in part:
                    audio_found = True
                    break
            if audio_found:
                print(f"✅ SUCCESS: {model} generated audio content successfully.")
                return True
            else:
                print(f"❌ FAILED: {model} returned 200 but NO inline audio data found.")
        else:
            print(f"❌ FAILED: Status {response.status_code} - {response.text}")
    except Exception as e:
        print(f"⚠️ EXCEPTION: {e}")
    return False

if __name__ == "__main__":
    test_gemini_exp_audio()
