import requests
import json
from core.utils import load_key

def test_all_for_audio():
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    # Models to test based on user's ListModels output
    models = [
        "gemini-2.0-flash", 
        "gemini-2.5-flash-preview-tts", 
        "gemini-3.1-flash-tts-preview",
        "gemini-2.0-flash-001"
    ]
    
    results = []
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": "Test"}]}],
            "generationConfig": {"responseModalities": ["AUDIO"]}
        }
        print(f"Testing {model}...")
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                print(f"✅ {model} works!")
                results.append((model, True))
            else:
                print(f"❌ {model} failed ({response.status_code}): {response.text[:100]}")
                results.append((model, False))
        except Exception as e:
            print(f"⚠️ {model} error: {e}")
            results.append((model, False))
    return results

if __name__ == "__main__":
    test_all_for_audio()
