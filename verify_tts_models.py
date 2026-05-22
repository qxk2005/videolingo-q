import requests
import json
from core.utils import load_key

def test_specific_model(model_name):
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    version = "v1beta"
    url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "Hello, this is a test."
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
    
    print(f"Testing model: {model_name}...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            # Check if audio content is actually in the response
            audio_found = False
            for part in data.get('candidates', [{}])[0].get('content', {}).get('parts', []):
                if 'inlineData' in part:
                    audio_found = True
                    break
            if audio_found:
                print(f"  SUCCESS: {model_name} generated audio!")
                return True
            else:
                print(f"  FAILED: {model_name} returned 200 but NO audio content found.")
                print(f"  Response: {json.dumps(data)[:200]}...")
        else:
            print(f"  FAILED: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
    return False

if __name__ == "__main__":
    # Test both 2.5 tts preview models found in the list
    test_specific_model("gemini-2.5-flash-preview-tts")
    test_specific_model("gemini-2.5-pro-preview-tts")
    # Also test the one that was working but had safety issues
    test_specific_model("gemini-3.1-flash-tts-preview")
