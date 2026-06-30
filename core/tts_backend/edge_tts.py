from pathlib import Path
import edge_tts as edge_tts_sdk
from core.utils import *
import asyncio

def edge_tts(text, save_path):
    # Load settings from config file
    edge_set = load_key("edge_tts")
    voice = edge_set.get("voice", "en-US-JennyNeural")
    
    # Create output directory if it doesn't exist
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    communicate = edge_tts_sdk.Communicate(text, voice)
    
    # To run async function in a synchronous context, handle existing event loop if any
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Run in a separate thread to avoid "RuntimeError: asyncio.run() cannot be called from a running event loop"
        import threading
        result_exception = []
        
        def run_in_thread():
            try:
                asyncio.run(communicate.save(str(speech_file_path)))
            except Exception as e:
                result_exception.append(e)
                
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        if result_exception:
            raise result_exception[0]
    else:
        asyncio.run(communicate.save(str(speech_file_path)))
        
    print(f"Audio saved to {speech_file_path}")

if __name__ == "__main__":
    edge_tts("Today is a good day!", "edge_tts.wav")
