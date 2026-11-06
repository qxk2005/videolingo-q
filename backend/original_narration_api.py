"""Task-specific native VideoLingo extension. Not an upstream feature or a standalone HTTP client."""
from pathlib import Path
import threading
from fastapi import HTTPException
from core.original_narration import NarrationJob, VOICE, read_json, digest

PROJECT = Path('/Users/niuzhidao/Documents/Codex/2026-11-05/task/20261105_001_claude_editing_ep4psgAFOqc/A_original')
INPUT = PROJECT / 'production_resume_20261106/narration-input.json'
EXPECTED_INPUT_SHA256 = '59d9a15d733af8aa009ca47e4a1d330ed4984c90fa5cf2d314093850852c8233'
_job = None
_lock = threading.Lock()


def service_provider(text, output):
    # Existing configured provider owns its authentication. No credentials cross this interface.
    from core.tts_backend.doubao_tts import original_narration_tts
    return original_narration_tts(text, output, voice_type=VOICE)


def get_job():
    global _job
    with _lock:
        if _job is None:
            if INPUT.is_symlink() or digest(INPUT) != EXPECTED_INPUT_SHA256:
                raise ValueError('Authorized manuscript changed; review a new version first')
            _job = NarrationJob(PROJECT / 'videolingo_narration_native_v1/jobs', read_json(INPUT), service_provider)
        return _job


def register_original_narration_routes(app, is_busy):
    @app.get('/api/original-narration/status')
    def original_narration_status():
        try:
            return get_job().status()
        except Exception:
            raise HTTPException(409, 'Original narration input or cache validation failed') from None

    @app.post('/api/original-narration/start')
    def original_narration_start():
        if is_busy():
            raise HTTPException(409, 'Studio or batch task is active; keep this workflow serial')
        try:
            return get_job().start()
        except Exception:
            raise HTTPException(409, 'Original narration input or cache validation failed') from None

    @app.post('/api/original-narration/continue')
    def original_narration_continue():
        if is_busy():
            raise HTTPException(409, 'Studio or batch task is active; keep this workflow serial')
        try:
            return get_job().start(continue_after_probe=True)
        except Exception:
            raise HTTPException(409, 'Original narration input or cache validation failed') from None
