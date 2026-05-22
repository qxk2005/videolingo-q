import requests
import json
import base64
from core.utils import load_key

def test_creative_prompt(text):
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    model = "gemini-2.5-flash-preview-tts"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    
    # CREATIVE STRATEGY: 
    # Instead of sending raw text, we send it as a clear narration task.
    # This provides the model with more context and helps bypass sensitive word triggers.
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Please narrate the following text in a natural, professional voice. Ensure clear articulation of every word: \"{text}\""
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
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
        ]
    }
    
    print(f"Testing Creative Prompt for: '{text}'")
    try:
        response = requests.post(url, json=payload, timeout=20)
        data = response.json()
        if response.status_code == 200:
            finish_reason = data.get('candidates', [{}])[0].get('finishReason', 'SUCCESS')
            if finish_reason == 'STOP' or finish_reason == 'SUCCESS':
                # Check for audio data
                parts = data.get('candidates', [{}])[0].get('content', {}).get('parts', [])
                if any('inlineData' in p for p in parts):
                    print(f"  ✅ SUCCESS: Generated audio with Creative Prompt.")
                    return True
            print(f"  ❌ FAILED: Finish reason {finish_reason}. Response: {json.dumps(data)}")
        else:
            print(f"  ❌ FAILED: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  ⚠️ EXCEPTION: {e}")
    return False

if __name__ == "__main__":
    # Test with a sentence that might have caused issues before
    test_creative_prompt("现在我们回头看看进度")
