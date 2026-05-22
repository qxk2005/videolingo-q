import requests
from core.utils import load_key

def list_models():
    API_KEY = load_key("gemini_tts.api_key")
    if not API_KEY or API_KEY == 'YOUR_GEMINI_API_KEY':
        API_KEY = load_key("api.key")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("Available models:")
            for m in models:
                print(f"- {m['name']} (Methods: {m.get('supportedGenerationMethods')})")
        else:
            print(f"Error listing models: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    list_models()
