import requests
import json
import time
from core.utils import load_key

def call_gemini_tts_raw(text, model, key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": f"NARRATE THIS EXACTLY, AUDIO ONLY, NO TEXT: {text}"}]}],
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
        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_CIVIC_INTEGRITY"]]
    }
    response = requests.post(url, json=payload, timeout=25)
    return response

def test_ultimate_strategy(text_list):
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    # Models to try in sequence
    MODELS = ["gemini-2.0-flash", "gemini-2.5-flash-preview-tts"]
    
    success_count = 0
    for i, text in enumerate(text_list):
        print(f"Testing {i+1}/{len(text_list)}: '{text[:15]}...'")
        found = False
        
        # Strategy: Try each model with multiple jitters
        for model in MODELS:
            if found: break
            for jitter in ["", " ", ".", " ."]:
                jittered_text = text + jitter
                try:
                    resp = call_gemini_tts_raw(jittered_text, model, API_KEY)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidate = data.get('candidates', [{}])[0]
                        if candidate.get('finishReason') in ['STOP', None]:
                            if any('inlineData' in p for p in candidate.get('content', {}).get('parts', [])):
                                print(f"  ✅ SUCCESS with {model} (jitter: '{jitter}')")
                                found = True
                                success_count += 1
                                break
                    elif resp.status_code == 400 and "tried to generate text" in resp.text:
                        # Some models are too chatty, skip to next model
                        break
                except Exception as e:
                    pass
                time.sleep(0.5)
        
        if not found:
            print(f"  ❌ TOTAL FAILURE for '{text}'")
    
    print(f"\nFinal Result: {success_count}/{len(text_list)} success")
    return success_count == len(text_list)

if __name__ == "__main__":
    test_texts = [
        "现在我们回头看看进度",
        "Hello World",
        "这是一段普通的测试文本",
        "稍微长一点的句子，看看在处理较长内容时是否会因为超时或其它原因报错",
        "1234567890",
        "One more test sentence to be sure."
    ]
    test_ultimate_strategy(test_texts)
