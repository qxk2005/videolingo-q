import os
import requests
import json
import uuid
import base64
from core.utils import load_key, except_handler

@except_handler("Failed to generate audio using Doubao TTS 2.0", retry=1, delay=1)
def doubao_tts(text, save_path):
    """
    Generate audio using Volcengine Doubao TTS Model 2.0 (V1 HTTP Interface)
    Reference: https://www.volcengine.com/docs/6561/1329505?lang=zh
    """
    
    # 1. Load minimalist configurations
    appid = str(load_key("doubao_tts.appid") or "").strip()
    access_token = str(load_key("doubao_tts.access_token") or "").strip()
    voice_type = load_key("doubao_tts.voice") or "zh_female_vv_uranus_bigtts"
    
    if not appid or not access_token:
        raise ValueError("Doubao AppID or Access Token not found. Please set them in the sidebar.")

    # 2. Setup V1 Request
    url = "https://openspeech.bytedance.com/api/v1/tts"
    
    # Header requires a SEMICOLON after Bearer
    headers = {
        "Authorization": f"Bearer;{access_token}",
        "Content-Type": "application/json"
    }
    
    # Minimalist Payload as per V1 documentation
    payload = {
        "app": {
            "appid": appid,
            "token": access_token,
            "cluster": "volcano_tts"
        },
        "user": {
            "uid": "videolingo_user"
        },
        "audio": {
            "voice_type": voice_type,
            "encoding": "mp3"
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": text.strip(),
            "operation": "query"
        }
    }
    
    print(f"DEBUG: [DOUBAO 2.0 V1] Generating with Voice: {voice_type}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        # Check HTTP status
        if response.status_code != 200:
            print(f"❌ Doubao V1 Error: HTTP {response.status_code}")
            raise Exception(f"HTTP {response.status_code}: {response.text}")
            
        res_json = response.json()
        
        # Check Business Logic Code (3000 is success)
        if res_json.get("code") == 3000:
            audio_data = base64.b64decode(res_json.get("data"))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(audio_data)
            print(f"✅ Doubao 2.0 V1 Success: {save_path}")
            return True
        else:
            error_msg = f"{res_json.get('message')} (Code: {res_json.get('code')})"
            print(f"❌ Doubao V1 Business Error: {error_msg}")
            raise Exception(error_msg)

    except Exception as e:
        print(f"❌ Doubao TTS 2.0 Failed: {e}")
        raise e

if __name__ == "__main__":
    pass
