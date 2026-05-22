import sys
import os
sys.path.append(os.getcwd())
from core.tts_backend.gemini_tts import gemini_tts
from core.utils import load_key

def test():
    text = "你好，这是一段测试音频。"
    save_path = "test_gemini_debug.wav"
    try:
        print(f"Testing Gemini TTS with key: {load_key('gemini_tts.api_key')[:10]}...")
        gemini_tts(text, save_path)
        if os.path.exists(save_path):
            print(f"Success! File size: {os.path.getsize(save_path)} bytes")
        else:
            print("Failed: File not created")
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test()
