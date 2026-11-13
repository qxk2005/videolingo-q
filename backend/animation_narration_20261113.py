"""Task-confined 2026-11-13 original narration; no arbitrary paths or credentials."""
from pathlib import Path
import threading
from core.original_narration import NarrationJob, read_json, digest
from backend.original_narration_api import service_provider
PROJECT = Path('/Users/niuzhidao/Documents/Codex/2026-11-13/ai-animation-skill-A')
INPUT = PROJECT / 'narration-input.json'
EXPECTED = '27ce2b1f3f83ae095c4c64a41a787671b564e2db1b35a74b88aa9eafe087913c'
VOICE = 'zh_male_m191_uranus_bigtts'
_lock = threading.Lock()
_job = None

def execute(args, busy):
    global _job
    if not isinstance(args, dict) or set(args) != {'action'} or args['action'] not in {'status','start','continue'}:
        raise ValueError('Only action=status/start/continue is supported')
    if INPUT.is_symlink() or digest(INPUT) != EXPECTED:
        raise ValueError('Manuscript hash changed; no synthesis allowed')
    with _lock:
        if _job is None:
            _job = NarrationJob(PROJECT / 'audio-jobs', read_json(INPUT), service_provider, voice=VOICE)
        if args['action'] == 'status': return _job.status()
        if busy: raise ValueError('Existing Studio or batch task active')
        return _job.start(continue_after_probe=args['action']=='continue')
