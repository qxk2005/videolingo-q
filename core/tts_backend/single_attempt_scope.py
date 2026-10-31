"""Opt-in, local-video-scoped one-attempt TTS; other projects keep defaults."""
import hashlib
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlparse, parse_qs

_BATCH_MARKER = Path('output/tts-single-attempt-task.json')
_BATCH_IDS = {'gwduNjsOaNA', 'QER-0DaC-Gk', 'sNjFA7-_8k8', 'i0P56Pm1Q3U'}
_LEGACY_MARKER = Path('output/.tts-single-attempt-yija5xCPt2E.json')
_TASK_MARKER = Path('output/.tts-single-attempt-gwduNjsOaNA.json')

def _marker_path():
    if _BATCH_MARKER.exists():
        return _BATCH_MARKER
    # Explicit first-video opt-in; legacy marker and all other videos retain behavior.
    if _TASK_MARKER.exists():
        return _TASK_MARKER
    return _LEGACY_MARKER
_LOCK = threading.RLock()

def single_attempt_active():
    if not _marker_path().exists():
        return False
    policy = json.loads(_marker_path().read_text())
    if _marker_path() == _BATCH_MARKER and policy.get('video_id') not in _BATCH_IDS:
        raise RuntimeError('Unexpected batch video ID; stop for review')
    if _marker_path() == _TASK_MARKER and policy.get('video_id') != 'gwduNjsOaNA':
        raise RuntimeError('First-video opt-in marker has unexpected video ID; stop for review')
    if not policy.get('enabled'):
        return False
    url = urlparse(Path('output/youtube_url.txt').read_text().strip())
    video_id = url.path.strip('/') if url.hostname in ('youtu.be', 'www.youtu.be') else parse_qs(url.query).get('v', [''])[0]
    if video_id != policy['video_id']:
        if _marker_path() == _BATCH_MARKER:
            raise RuntimeError('TTS marker mismatch; stop for review')
        return False
    source = Path(policy['source_file'])
    st = source.stat()
    if st.st_size != policy['source_bytes'] or st.st_mtime_ns != policy['source_mtime_ns']:
        raise RuntimeError('Single-attempt scope media changed; stop for review')
    return True

def single_attempt_call(func, text, save_as, number=None, **kwargs):
    """Reserve BEFORE network; failures/unknown results must never be resubmitted."""
    with _LOCK:
        if not single_attempt_active():
            raise RuntimeError('Single-attempt scope no longer active; stop for review')
        policy = json.loads(_marker_path().read_text())
        ledger = Path(policy['ledger_dir']); ledger.mkdir(parents=True, exist_ok=True)
        if (ledger/'HALTED').exists():
            raise RuntimeError('Earlier TTS request failed or is uncertain; review required')
        key = hashlib.sha256((str(number)+'|'+Path(save_as).name).encode()).hexdigest()
        reservation = ledger/(key+'.json')
        record = {'number':str(number),'slot':Path(save_as).name,'characters':len(text),
                  'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'state':'reserved_before_request'}
        try:
            with reservation.open('x') as f:
                json.dump(record,f);f.flush();os.fsync(f.fileno())
        except FileExistsError:
            raise RuntimeError('This TTS segment already attempted; no automatic resubmission')
        try:
            result = func(text, save_as, **kwargs)
        except BaseException:
            (ledger/'HALTED').write_text('Request failed or outcome uncertain. No automatic retry.\n')
            raise
        record['state']='returned_success';reservation.write_text(json.dumps(record))
        return result
