"""Task-specific native VideoLingo extension. Not an upstream feature or a standalone HTTP client."""
from pathlib import Path
import threading
from fastapi import Body, HTTPException
from core.original_narration import NarrationJob, read_json, digest
from core.original_narration_voices import DEFAULT_VOICE, validate_voice

PROJECT = Path('/Users/niuzhidao/Documents/Codex/2026-11-05/task/20261105_001_claude_editing_ep4psgAFOqc/A_original')
INPUT = PROJECT / 'production_resume_20261106/narration-input.json'
EXPECTED_INPUT_SHA256 = '59d9a15d733af8aa009ca47e4a1d330ed4984c90fa5cf2d314093850852c8233'
_jobs = {}
_lock = threading.Lock()


def service_provider(text, output, voice_type):
    # Existing configured provider owns its authentication. No credentials cross this interface.
    from core.tts_backend.doubao_tts import original_narration_tts
    return original_narration_tts(text, output, voice_type=validate_voice(voice_type))


def get_job(voice=DEFAULT_VOICE):
    voice = validate_voice(voice)
    with _lock:
        if voice not in _jobs:
            if INPUT.is_symlink() or digest(INPUT) != EXPECTED_INPUT_SHA256:
                raise ValueError('Authorized manuscript changed; review a new version first')
            _jobs[voice] = NarrationJob(PROJECT / 'videolingo_narration_native_v1/jobs',
                                       read_json(INPUT), service_provider, voice=voice)
        return _jobs[voice]


def request_voice(payload=None):
    if payload is None:
        payload = {}
    if not isinstance(payload, dict) or set(payload) - {'voice'}:
        raise HTTPException(422, 'Only the optional voice field is supported')
    try:
        return validate_voice(payload.get('voice', DEFAULT_VOICE))
    except ValueError:
        raise HTTPException(422, 'Select an exact voice ID from the existing Doubao catalog') from None


def register_original_narration_routes(app, is_busy):
    @app.get('/api/original-narration/status')
    def original_narration_status(voice: str = DEFAULT_VOICE):
        voice = request_voice({'voice': voice})
        try:
            return get_job(voice).status()
        except Exception:
            raise HTTPException(409, 'Original narration input or cache validation failed') from None

    @app.post('/api/original-narration/start')
    def original_narration_start(payload: dict = Body(default=None)):
        voice = request_voice(payload)
        if is_busy():
            raise HTTPException(409, 'Studio or batch task is active; keep this workflow serial')
        try:
            return get_job(voice).start()
        except Exception:
            raise HTTPException(409, 'Original narration input or cache validation failed') from None

    @app.post('/api/original-narration/continue')
    def original_narration_continue(payload: dict = Body(default=None)):
        voice = request_voice(payload)
        if is_busy():
            raise HTTPException(409, 'Studio or batch task is active; keep this workflow serial')
        try:
            return get_job(voice).start(continue_after_probe=True)
        except Exception:
            raise HTTPException(409, 'Original narration input or cache validation failed') from None
