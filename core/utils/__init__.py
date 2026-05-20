# use try-except to avoid error when installing
try:
    from .ask_gpt import ask_gpt
    from .decorator import except_handler, check_file_exists
    from .config_utils import load_key, update_key, get_joiner
    from rich import print as rprint
except ImportError:
    pass

import platform
import subprocess

def get_ffmpeg_video_encoder():
    """
    Returns the best available hardware video encoder name, or None for software (CPU) encoding.
    On macOS (Apple Silicon / VideoToolbox): h264_videotoolbox
    On other platforms (NVIDIA):             h264_nvenc
    """
    try:
        if not load_key("ffmpeg_gpu"):
            return None
    except Exception:
        return None

    system = platform.system()
    candidate = 'h264_videotoolbox' if system == 'Darwin' else 'h264_nvenc'
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        if candidate in result.stdout:
            print(f"ffmpeg hw encoder available: {candidate}")
            return candidate
    except Exception:
        pass
    print(f"ffmpeg hw encoder '{candidate}' not found, falling back to software encoding")
    return None

__all__ = ["ask_gpt", "except_handler", "check_file_exists", "load_key", "update_key", "rprint", "get_joiner", "get_ffmpeg_video_encoder"]