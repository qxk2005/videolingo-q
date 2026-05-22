import requests
import json
import base64
from core.utils import load_key

def test_modality():
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    models = ["gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-2.0-flash-lite", "gemini-2.0-flash-lite-001"]
    
    for model in models:
        for version in ["v1beta", "v1"]:
            print(f"Testing {model} with {version}...")
            url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": "Hello"
                    }]
                }],
                "generationConfig": {
                    "responseModalities": ["AUDIO"]
                }
            }
            
            headers = {'Content-Type': 'application/json'}
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    print(f"  SUCCESS! {model} {version} supports AUDIO.")
                    return model, version
                else:
                    print(f"  FAILED: {response.status_code} - {response.text[:100]}")
            except Exception as e:
                print(f"  EXCEPTION: {e}")
    return None, None

if __name__ == "__main__":
    model, version = test_modality()
    if model:
        print(f"Found working combination: {model} {version}")
    else:
        print("No working combination found for native audio output.")
