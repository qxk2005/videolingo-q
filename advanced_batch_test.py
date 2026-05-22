import requests
import json
import time
from core.utils import load_key

def test_strategy(name, payload_factory, text_list):
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    model = "gemini-2.5-flash-preview-tts"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    
    success_count = 0
    for i, text in enumerate(text_list):
        payload = payload_factory(text)
        print(f"[{name}] Testing segment {i+1}/{len(text_list)}: '{text[:20]}...'")
        try:
            response = requests.post(url, json=payload, timeout=20)
            data = response.json()
            if response.status_code == 200:
                candidate = data.get('candidates', [{}])[0]
                finish_reason = candidate.get('finishReason')
                if finish_reason in ['STOP', None]: # None sometimes means success in some versions
                    if any('inlineData' in p for p in candidate.get('content', {}).get('parts', [])):
                        success_count += 1
                        print(f"  ✅ OK")
                    else:
                        print(f"  ❌ No audio data. Reason: {finish_reason}")
                else:
                    print(f"  ❌ Failed: {finish_reason}")
            else:
                print(f"  ❌ HTTP {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
        time.sleep(1) # Respect rate limits
    
    print(f"\nSummary for {name}: {success_count}/{len(text_list)} success\n" + "="*40)
    return success_count == len(text_list)

# Strategy A: System Instruction
def factory_system_instruction(text):
    return {
        "systemInstruction": {"parts": [{"text": "You are a professional voice actor. Speak the input text exactly."}]},
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"responseModalities": ["AUDIO"]},
        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_CIVIC_INTEGRITY"]]
    }

# Strategy B: Minimalist with Jitter
def factory_jitter(text):
    # Adding a trailing period or space to change hash
    return {
        "contents": [{"parts": [{"text": f"{text} "}]}],
        "generationConfig": {"responseModalities": ["AUDIO"]},
        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_CIVIC_INTEGRITY"]]
    }

# Strategy C: JSON Schema Output (Creative)
def factory_json_schema(text):
    return {
        "contents": [{"parts": [{"text": f"Output audio for this text: {text}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "responseMimeType": "application/json" # Some models allow this to force structured output including audio
        },
        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_CIVIC_INTEGRITY"]]
    }

if __name__ == "__main__":
    test_texts = [
        "现在我们回头看看进度",
        "这是一段普通的测试文本",
        "稍微长一点的句子，看看在处理较长内容时是否会因为超时或其它原因报错",
        "1234567890",
        "Hello World"
    ]
    
    print("🚀 Starting Advanced Batch Tests...\n")
    test_strategy("System Instruction", factory_system_instruction, test_texts)
    test_strategy("Text Jitter", factory_jitter, test_texts)
    # Strategy C often requires specific model support, might skip if not supported
