"""Audio-first, project-confined narration jobs. No credentials, HTTP, or translation."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import threading
import uuid
import wave
from core.original_narration_voices import DEFAULT_VOICE, validate_voice

VOICE = DEFAULT_VOICE  # Legacy callers retain the original VV task identity.
RATE = 24000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_dir(path):
    path = Path(path)
    for item in [path, *path.parents]:
        if item.is_symlink():
            raise ValueError("Symlink directory rejected")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def read_json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_nlink != 1:
        raise ValueError("Aliased file rejected")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    safe_dir(path.parent)
    tmp = path.with_name("." + uuid.uuid4().hex + ".json")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_wav(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_nlink != 1:
        raise ValueError("Aliased audio rejected")
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) != (1, 2, RATE, "NONE"):
            raise ValueError("Expected mono PCM16 24000Hz")
        frames = w.getnframes()
        data = w.readframes(frames)
    if frames <= 0 or len(data) != frames * 2 or not any(data):
        raise ValueError("Empty, silent or truncated audio")
    return {"frames": frames, "sample_rate": RATE, "duration": frames / RATE,
            "sha256": digest(path)}


class NarrationJob:
    """One immutable manuscript, one request per segment, no automatic paid retry."""
    def __init__(self, root, segments, provider, voice=DEFAULT_VOICE):
        self._voice = validate_voice(voice)
        if not isinstance(segments, list) or len(segments) != 11:
            raise ValueError("Exactly 11 authorized segments required")
        ids = set()
        for s in segments:
            if set(s) - {"id", "narration", "title", "chapter"}:
                raise ValueError("Unexpected input fields")
            sid, text = s.get("id"), s.get("narration")
            if not isinstance(sid, str) or not re.fullmatch(r"S\d{2}", sid) or sid in ids:
                raise ValueError("Invalid or duplicate segment ID")
            if not isinstance(text, str) or not 1 <= len(text) <= 1000 or text != text.strip():
                raise ValueError("Invalid text; no implicit normalization")
            ids.add(sid)
        if sum(len(s["narration"]) for s in segments) > 12000:
            raise ValueError("Project text limit exceeded")
        self.segments = json.loads(json.dumps(segments))
        self.provider = provider
        key = json.dumps({"segments": segments, "voice": self.voice, "speed": 1.0,
                          "resource": "seed-tts-2.0", "version": 1}, sort_keys=True, ensure_ascii=False)
        self.key = hashlib.sha256(key.encode()).hexdigest()
        self.root = Path(root)
        for item in [self.root, *self.root.parents]:
            if item.is_symlink():
                raise ValueError("Symlink directory rejected")
        self.path = self.root / self.key
        self.thread = None

    @property
    def voice(self):
        return self._voice

    def status(self):
        if not self.path.exists():
            return {"status": "not_started", "job_id": self.key, "provider_attempts": 0,
                    "voice": self.voice, "speed": 1.0}
        safe_dir(self.path)
        if not (self.path / "state.json").exists():
            return {"status": "uncertain", "job_id": self.key, "reason": "Incomplete reservation; no automatic restart"}
        result = read_json(self.path / "state.json")
        if (result.get('job_id') != self.key or result.get('voice') != self.voice
                or result.get('speed') != 1.0
                or result.get('output_directory') != str(self.path)):
            raise ValueError('Cached task voice or identity mismatch')
        if result['status'] == 'ready_for_remaining':
            entries = self._verified_segments()
            if len(entries) != 1 or result['completed_segments'] != 1:
                raise ValueError('Invalid first-segment checkpoint')
        if result["status"] == "complete":
            timeline = read_json(self.path / "timeline.json")
            if timeline.get('voice', DEFAULT_VOICE) != self.voice:
                raise ValueError('Cached timeline voice mismatch')
            if len(timeline['segments']) != len(self.segments):
                raise ValueError('Cached timeline incomplete')
            for s, original in zip(timeline["segments"], self.segments):
                if s.get('voice', DEFAULT_VOICE) != self.voice:
                    raise ValueError('Cached segment voice mismatch')
                if s['id'] != original['id'] or s['file'] != original['id'] + '.wav' or s['text'] != original['narration']:
                    raise ValueError('Cached timeline identity mismatch')
                if validate_wav(self.path / s["file"])["sha256"] != s["sha256"]:
                    raise ValueError("Cached audio hash mismatch")
            if validate_wav(self.path / "master.wav")["sha256"] != timeline["master_sha256"]:
                raise ValueError("Cached master hash mismatch")
        return result

    def start(self, background=True, continue_after_probe=False):
        safe_dir(self.root)
        lockpath = self.root / ".start.lock"
        fd = os.open(lockpath, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                previous = self.status()
                if not continue_after_probe or previous['status'] != 'ready_for_remaining':
                    return previous
                self._assert_no_other_active_job()
                # An explicit continuation may reuse only verified completed probe audio.
                self._verified_segments()
                self._save('queued', previous['completed_segments'], previous['provider_attempts'])
            else:
                if continue_after_probe:
                    raise ValueError('Generate and verify the first segment before continuing')
                self._assert_no_other_active_job()
                self.path.mkdir(mode=0o700)
                self._save("queued", 0, 0)
            if background:
                self.thread = threading.Thread(target=self.run_reserved, args=(continue_after_probe,), daemon=True)
                self.thread.start()
            else:
                self.run_reserved(continue_after_probe)
        return self.status()

    def _assert_no_other_active_job(self):
        # The root start lock serializes reservations across voices and processes.
        for path in self.root.iterdir():
            if path == self.path or not re.fullmatch(r'[0-9a-f]{64}', path.name):
                continue
            if path.is_symlink() or not path.is_dir():
                raise ValueError('Invalid narration task directory')
            state_file = path / 'state.json'
            if not state_file.exists():
                raise ValueError('Another narration reservation is incomplete')
            state = read_json(state_file)
            if state.get('status') in {'queued', 'request_pending', 'running', 'uncertain'}:
                raise ValueError('Another narration task is active or uncertain')

    def _save(self, status, done, requests, **extra):
        atomic_json(self.path / "state.json", {"status": status, "job_id": self.key,
                    "completed_segments": done, "total_segments": 11, "provider_attempts": requests,
                    "voice": self.voice, "speed": 1.0, "output_directory": str(self.path), **extra})

    def _verified_segments(self):
        entries = read_json(self.path / 'segments.json')
        for s, original in zip(entries, self.segments):
            if s.get('voice', DEFAULT_VOICE) != self.voice:
                raise ValueError('Cached segment voice mismatch')
            if s['id'] != original['id'] or s['file'] != original['id'] + '.wav' or s['text'] != original['narration']:
                raise ValueError('Cached segment identity mismatch')
            if validate_wav(self.path / s['file'])['sha256'] != s['sha256']:
                raise ValueError('Cached audio hash mismatch')
        if not 1 <= len(entries) <= 11:
            raise ValueError('Invalid cached segment count')
        return entries

    def run_reserved(self, continue_after_probe=False):
        # Only the start() reservation creates a worker. Existing jobs never resume automatically.
        fd = os.open(self.path / ".worker.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+") as worker:
            try:
                fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            if self.status()["status"] != "queued":
                return
            state = self.status()
            done, requests, cumulative, timeline = 0, state['provider_attempts'], 0, []
            try:
                if state['completed_segments']:
                    timeline = self._verified_segments()
                    done = len(timeline)
                    if done != state['completed_segments']:
                        raise ValueError('Checkpoint identity mismatch')
                    cumulative = sum(s['frames'] for s in timeline)
                atomic_json(self.path / "input.json", self.segments)
                selected = self.segments[done:] if continue_after_probe else self.segments[:1]
                for s in selected:
                    tmp = self.path / (s["id"] + ".pending.wav")
                    if tmp.exists():
                        raise ValueError("Unexpected prior audio")
                    requests += 1
                    # Persist before dispatch: timeout/crash must not silently trigger another paid call.
                    self._save("request_pending", done, requests, active_segment=s["id"])
                    result = self.provider(s["narration"], str(tmp), voice_type=self.voice)
                    if not isinstance(result, dict) or result.get("ok") is not True:
                        raise ValueError("Provider did not confirm completion")
                    info = validate_wav(tmp)
                    final = self.path / (s["id"] + ".wav")
                    if final.exists():
                        raise ValueError("Refusing to overwrite audio")
                    os.rename(tmp, final)
                    timeline.append({"id": s["id"], "title": s.get("title", ""),
                        "voice": self.voice,
                        "text": s["narration"], "text_sha256": hashlib.sha256(s["narration"].encode()).hexdigest(),
                        "file": final.name, **info, "start_sample": cumulative,
                        "end_sample": cumulative + info["frames"]})
                    cumulative += info["frames"]
                    done += 1
                    atomic_json(self.path / "segments.json", timeline)
                    self._save("running", done, requests)
                if done < len(self.segments):
                    self._save('ready_for_remaining', done, requests,
                               next_action='Verify S01 audio, then explicitly continue. No automatic continuation.')
                    return
                master = self.path / "master.pending.wav"
                fd = os.open(master, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    with wave.open(stream, "wb") as out:
                        out.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                        for s in timeline:
                            with wave.open(str(self.path / s["file"]), "rb") as audio:
                                out.writeframes(audio.readframes(audio.getnframes()))
                validate_wav(master)
                os.rename(master, self.path / "master.wav")
                atomic_json(self.path / "timeline.json", {"sample_rate": RATE, "total_samples": cumulative,
                    "duration": cumulative / RATE, "master_sha256": digest(self.path / "master.wav"),
                    "speed": 1.0, "voice": self.voice, "segments": timeline})
                self._save("complete", done, requests)
            except Exception:
                # Exception messages can contain provider response data; never persist them.
                self._save("failed_or_uncertain", done, requests,
                           reason="Stopped; inspect receipts before any explicitly approved retry. No automatic retry.")
                # Keep completed audio and pending output for diagnosis; no source/other task deletion.

