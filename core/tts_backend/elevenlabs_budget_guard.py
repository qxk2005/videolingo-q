"""Opt-in Fable 5.2 HTTP request guard. No billing/credential API calls.

Reservations are durable before POST and never refunded. A crash, failed HTTP
request, local synthesis error, changed request or damaged ledger stops the task.
Only the named video's already-reviewed paragraph requests are permitted.
"""
import contextlib
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

from core.utils.cancel_utils import TaskCancelledError, check_task_cancelled, sleep_cancellable

VIDEO_ID = "0vNyVMDFdkg"
MARKER = "output/elevenlabs-cost-guard.json"
_MUTEX = threading.RLock()
_LOCAL = threading.local()


class BudgetStop(TaskCancelledError):
    """BaseException bypasses existing Exception-only retry/fallback handlers."""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".pending")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise BudgetStop("Budget guard file missing/corrupt; stop for review") from exc


def active():
    """Task selection only. Required policy is checked before cache or API work."""
    if getattr(_LOCAL, "task_guard", None) is not None:
        return True
    source = Path("output/youtube_url.txt")
    if not source.exists():
        if Path(MARKER).exists():
            raise BudgetStop("Budget marker present but source video missing")
        return False
    url = urlparse(source.read_text(encoding="utf-8").strip())
    video = url.path.strip("/") if url.hostname in ("youtu.be", "www.youtu.be") else parse_qs(url.query).get("v", [""])[0]
    if video != VIDEO_ID and Path(MARKER).exists():
        raise BudgetStop("Budget marker/source mismatch; no unguarded fallback")
    return video == VIDEO_ID


TRANSIENT_NETWORK_ERRORS = ("ReadTimeout", "ConnectTimeout", "ConnectionError", "ProxyError", "ChunkedEncodingError", "SSLError")


def safe_exception_chain(exc):
    """Exception CLASS names only; no messages, URLs, headers or local variables."""
    pending, seen, names = [exc], set(), []
    while pending and len(names) < 12:
        current_exc = pending.pop(0)
        if not isinstance(current_exc, BaseException) or id(current_exc) in seen:
            continue
        seen.add(id(current_exc))
        names.append(type(current_exc).__module__ + "." + type(current_exc).__name__)
        pending.extend([current_exc.__cause__, current_exc.__context__, getattr(current_exc, "reason", None)])
        pending.extend(value for value in current_exc.args if isinstance(value, BaseException))
    return names


def retryable_network_exception(exc):
    if type(exc).__name__ not in TRANSIENT_NETWORK_ERRORS:
        return False
    chain = safe_exception_chain(exc)
    if any("certificate" in name.lower() or "certverification" in name.lower() for name in chain):
        return False
    # Inspect only to prevent retrying certificate/proxy-auth denials; never save.
    message = str(exc).lower()
    if ("certificate" in message or "cert_verify" in message
            or any("tunnel connection failed: " + code in message for code in ("400", "401", "402", "403", "407", "429"))):
        return False
    if any("ssl" in name.lower() for name in chain):
        # Connection closed by TLS peer is transient, unlike certificate or
        # other TLS failures. No security setting or verification is changed.
        return any(name in ("ssl.SSLZeroReturnError", "ssl.SSLEOFError") for name in chain)
    return True


class Guard:
    def __init__(self, policy_path):
        self.policy_path = Path(policy_path).resolve()
        self.policy = _read_json(self.policy_path)
        self.ledger = Path(self.policy["ledger_path"])
        self.lock = self.ledger.with_suffix(".lock")
        self.owner = self.ledger.with_suffix(".owner.lock")
        self.policy_hash = digest(self.policy)
        if (self.policy.get("video_id") != VIDEO_ID or not self.policy.get("armed")
                or self.policy.get("model_id") != "eleven_v4"
                or self.policy.get("voice_id") != "MI36FIkp9wRP7cpWKPTl"
                or not 0 < self.policy.get("max_requests", 0) <= 128
                or not 0 < self.policy.get("max_characters", 0) <= 30000
                or self.policy.get("reserve_milli_cny_per_character") != 2
                or not 0 < self.policy.get("technical_cap_milli_cny", 0) <= 60000
                or self.policy.get("authorized_cap_cny") != 200):
            raise BudgetStop("Budget guard policy not armed or outside approved task limits")

    @contextlib.contextmanager
    def locked(self):
        with self.lock.open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def manual_retry_original(self, data, row):
        """Only the explicitly reviewed original slot67 request may stay unknown.

        Its reservation remains charged. This is not a general retry permission.
        """
        auth = data.get("slot67_one_time_manual_retry", {})
        return (auth.get("slot") == "67" and auth.get("original_sequence") == 8
                and auth.get("max_additional_attempts") == 1 and auth.get("new_attempt") == 2
                and row.get("slot") == "67" and row.get("sequence") == 8
                and row.get("attempt", 1) == 1 and row.get("state") == "uncertain"
                and row.get("characters") == 118 and row.get("reserve_milli_cny") == 236
                and auth.get("original_payload_sha256") == row.get("payload_sha256"))

    def network_retry_enabled(self, data):
        auth = data.get("bounded_network_retry_authorization", {})
        return (auth.get("video_id") == VIDEO_ID and auth.get("max_attempts_per_slot") == 2
                and auth.get("connect_timeout_seconds") == 60 and auth.get("read_timeout_seconds") == 180
                and auth.get("backoff_seconds") == 2 and auth.get("budget_limits_unchanged") is True
                and auth.get("allowed_errors") == list(TRANSIENT_NETWORK_ERRORS))

    def retry_original(self, data, row):
        if self.manual_retry_original(data, row):
            return True
        if (not self.network_retry_enabled(data) or row.get("attempt", 1) != 1
                or row.get("state") != "uncertain" or row.get("error_type") not in TRANSIENT_NETWORK_ERRORS):
            return False
        if row.get("retryable_transport") is True:
            return True
        # Only a specifically reviewed older timeout may lack the new classifier.
        reviewed = data["bounded_network_retry_authorization"].get("reviewed_existing_timeouts", [])
        return any(item.get("sequence") == row.get("sequence") and item.get("slot") == row.get("slot")
                   and item.get("payload_sha256") == row.get("payload_sha256")
                   and item.get("error_type") == row.get("error_type") for item in reviewed)

    def state(self):
        data = _read_json(self.ledger)
        if data.get("schema") != 1 or data.get("policy_sha256") != self.policy_hash:
            raise BudgetStop("Budget ledger policy/schema mismatch; stop for review")
        rows = data.get("requests")
        if not isinstance(rows, list) or len(rows) != data.get("sequence"):
            raise BudgetStop("Budget ledger sequence corrupt; stop for review")
        if len({(r.get("slot"), r.get("attempt", 1)) for r in rows}) != len(rows):
            raise BudgetStop("Duplicate ledger slot attempts; stop for review")
        for i, row in enumerate(rows, 1):
            if row.get("attempt", 1) != 1:
                original = next((r for r in rows if r.get("slot") == row.get("slot") and self.retry_original(data, r)), None)
                if row.get("attempt") != 2 or original is None:
                    raise BudgetStop("Unapproved extra attempt in ledger")
            if (row.get("sequence") != i or row.get("characters", 0) <= 0
                    or row.get("reserve_milli_cny") != row["characters"] * 2
                    or row.get("slot") not in self.policy["manifest"]
                    or row.get("characters") != self.policy["manifest"].get(row.get("slot"), {}).get("characters")
                    or row.get("state") not in ("reserved", "http_success", "success", "failed", "uncertain")):
                raise BudgetStop("Budget ledger record corrupt; stop for review")
        return data

    def stop(self, reason):
        with self.locked():
            data = self.state()
            if not data.get("halted"):
                data["halted"] = reason
            elif data["halted"] != reason:
                data.setdefault("stop_contexts", []).append({"time_unix": time.time(), "context": reason})
            first_reason = data["halted"]
            _atomic_json(self.ledger, data)
        raise BudgetStop("ElevenLabs task paused: " + first_reason)

    def check_state(self, data):
        if data.get("halted"):
            raise BudgetStop("ElevenLabs task already paused: " + str(data["halted"]))
        if any(r["state"] != "success" and not self.retry_original(data, r)
               for r in data["requests"]):
            raise BudgetStop("Earlier reserved/uncertain synthesis; no automatic retry")

    def check_pipeline(self, fingerprint, task_rows):
        p = self.policy
        if str(Path.cwd().resolve()) != p["repository_root"]:
            self.stop("Wrong repository workspace")
        if fingerprint != p["tts_fingerprint"] or digest(task_rows) != p["task_rows_sha256"]:
            self.stop("Reviewed text, speaker, timing or TTS settings changed")
        old = Path("output/audio/.tts_settings")
        if old.exists() and old.read_text().strip() != fingerprint:
            self.stop("Audio cache settings differ; retained without clearing")
        with self.locked():
            self.check_state(self.state())

    def check_source(self):
        source = Path("output/youtube_url.txt")
        marker = _read_json(MARKER)
        if not source.exists():
            self.stop("Source removed while task guard is selected")
        url = urlparse(source.read_text(encoding="utf-8").strip())
        video = url.path.strip("/") if url.hostname in ("youtu.be", "www.youtu.be") else parse_qs(url.query).get("v", [""])[0]
        if (video != VIDEO_ID or marker.get("video_id") != VIDEO_ID
                or Path(marker.get("policy_path", "")).resolve() != self.policy_path):
            self.stop("Source/marker changed while task guard is selected")

    def execute(self, func, text, save_path, slot, **kwargs):
        slot = str(slot)
        expected = self.policy["manifest"].get(slot)
        if not expected or text_hash(text.strip()) != expected["text_sha256"]:
            self.stop("Unlisted paragraph or changed text")
        with _MUTEX, self.owner.open("a+") as owner:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BudgetStop("Another process owns this video's synthesis")
            try:
                with self.locked():
                    data = self.state()
                    self.check_state(data)
                    existing = [r for r in data["requests"] if r["slot"] == slot]
                    first_attempt = 1
                    if existing:
                        if len(existing) == 1 and self.retry_original(data, existing[0]):
                            first_attempt = 2
                        else:
                            data["halted"] = "Paragraph already attempted; retain cache and review"
                            _atomic_json(self.ledger, data)
                            raise BudgetStop(data["halted"])
                for attempt in range(first_attempt, 3):
                    check_task_cancelled()
                    _LOCAL.guard, _LOCAL.slot, _LOCAL.attempt = self, slot, attempt
                    try:
                        result = getattr(func, "__wrapped__", func)(text, save_path, **kwargs)
                        if isinstance(result, dict) and result.get("ok") and Path(save_path).is_file():
                            cached = {"schema": 1, "source": "elevenlabs_response", "slot": slot, "attempt": attempt,
                                      "text_sha256": expected["text_sha256"],
                                      "audio_sha256": hashlib.sha256(Path(save_path).read_bytes()).hexdigest(),
                                      "result": {key: result.get(key) for key in
                                                 ("ok", "words", "sample_rate", "duration", "model_id")}}
                            _atomic_json(Path(str(save_path) + ".elevenlabs.json"), cached)
                        with self.locked():
                            data = self.state()
                            row = next((r for r in data["requests"] if r["slot"] == slot and r.get("attempt", 1) == attempt), None)
                            if row is None or row["state"] != "http_success":
                                raise BudgetStop("Synthesis did not have exactly one confirmed HTTP result")
                            row["state"] = "success"
                            _atomic_json(self.ledger, data)
                        return result
                    except BaseException as exc:
                        with self.locked():
                            data = self.state()
                            row = next((r for r in data["requests"] if r["slot"] == slot and r.get("attempt", 1) == attempt), None)
                            if row is not None:
                                row.setdefault("error_type", type(exc).__name__)
                                row.setdefault("error_stage", "local_synthesis")
                                row.setdefault("error_chain", safe_exception_chain(exc))
                                row.setdefault("finished_at_unix", time.time())
                            can_retry = (attempt == 1 and row is not None
                                         and self.network_retry_enabled(data) and self.retry_original(data, row)
                                         and row.get("retryable_transport") is True)
                            if can_retry:
                                data.setdefault("bounded_network_retry_events", []).append({
                                    "slot": slot, "original_sequence": row["sequence"], "new_attempt": 2,
                                    "time_unix": time.time(), "backoff_seconds": 2,
                                    "original_error_type": row["error_type"], "original_halt": data.get("halted"),
                                    "possible_duplicate_charge": True})
                                data["halted"] = None
                            _atomic_json(self.ledger, data)
                        if can_retry:
                            # A user stop interrupts the backoff and prevents POST.
                            sleep_cancellable(2)
                            continue
                        self.stop("Synthesis failed or uncertain (" + type(exc).__name__ + "); no further retry")
                    finally:
                        _LOCAL.guard, _LOCAL.slot, _LOCAL.attempt = None, None, None
                self.stop("Maximum two attempts exhausted")
            finally:
                fcntl.flock(owner, fcntl.LOCK_UN)

    def post(self, transport, url, **kwargs):
        p = self.policy
        body = kwargs.get("json")
        slot = getattr(_LOCAL, "slot", None)
        attempt = getattr(_LOCAL, "attempt", None)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        expected = p["manifest"].get(str(slot))
        if (not expected or attempt not in (1, 2) or getattr(_LOCAL, "guard", None) is not self
                or parsed.scheme != "https" or parsed.netloc != "api.elevenlabs.io"
                or parsed.path != "/v1/text-to-speech/" + p["voice_id"] + "/with-timestamps"
                or parsed.fragment or set(query) != {"output_format"}
                or query["output_format"] not in (["pcm_24000"], ["mp3_44100_128"])
                or not isinstance(body, dict)
                or set(body) - {"text", "model_id", "voice_settings", "language_code", "previous_request_ids"}
                or body.get("model_id") != p["model_id"]
                or body.get("voice_settings") != p["voice_settings"]
                or body.get("language_code") != p["language_code"]
                or not isinstance(body.get("text"), str)
                or text_hash(body["text"]) != expected["text_sha256"]
                or len(body["text"]) != expected["characters"]):
            self.stop("Unapproved HTTP operation, voice, model, text or payload")
        prev = body.get("previous_request_ids", [])
        if not isinstance(prev, list) or len(prev) > 3 or any(not isinstance(x, str) or len(x) > 256 for x in prev):
            self.stop("Invalid cross-paragraph request IDs")
        characters = len(body["text"])
        with self.locked():
            data = self.state()
            self.check_state(data)
            rows = data["requests"]
            if (any(r["slot"] == slot and r.get("attempt", 1) == attempt for r in rows)
                    or len(rows) + 1 > p["max_requests"]
                    or sum(r["characters"] for r in rows) + characters > p["max_characters"]
                    or sum(r["reserve_milli_cny"] for r in rows) + characters * 2 > p["technical_cap_milli_cny"]):
                data["halted"] = "Request/character/reserve ceiling or duplicate paragraph"
                _atomic_json(self.ledger, data)
                raise BudgetStop(data["halted"])
            data["sequence"] += 1
            rows.append({"sequence": data["sequence"], "slot": slot, "attempt": attempt, "characters": characters,
                         "reserved_at_unix": time.time(),
                         "timeout_seconds": {"connect": 60, "read": 180},
                         "possible_duplicate_charge": attempt == 2,
                         "reserve_milli_cny": characters * 2, "payload_sha256": digest(body),
                         "state": "reserved"})
            _atomic_json(self.ledger, data)
        try:
            # requests.post uses a fresh Session with default zero retries. Never
            # follow a redirect: even 307/308 could otherwise repeat a paid POST.
            kwargs["allow_redirects"] = False
            kwargs["timeout"] = (60, 180)
            resp = transport(url, **kwargs)
        except BaseException as exc:
            with self.locked():
                data = self.state()
                row = data["requests"][-1]
                row["state"] = "uncertain"
                row["error_type"] = type(exc).__name__
                row["error_chain"] = safe_exception_chain(exc)
                row["retryable_transport"] = retryable_network_exception(exc)
                row["error_stage"] = "http_transport"
                row["finished_at_unix"] = time.time()
                data["halted"] = "HTTP outcome uncertain (" + type(exc).__name__ + ")"
                _atomic_json(self.ledger, data)
            raise BudgetStop("HTTP outcome uncertain; reserved cost retained") from exc
        with self.locked():
            data = self.state()
            row = data["requests"][-1]
            row["http_status"] = int(resp.status_code)
            row["finished_at_unix"] = time.time()
            # Minimal non-secret response evidence; never store request headers.
            headers = getattr(resp, "headers", {})
            row["response_headers"] = {
                key: str(headers[key])[:256]
                for key in ("request-id", "x-request-id", "x-character-count", "character-cost")
                if key in headers
            }
            row["state"] = "http_success" if resp.status_code == 200 else "failed"
            if resp.status_code != 200:
                data["halted"] = "HTTP failure; no parameter/format fallback or retry"
            _atomic_json(self.ledger, data)
        if resp.status_code != 200:
            raise BudgetStop("ElevenLabs HTTP " + str(resp.status_code) + "; reserved cost retained")
        return resp


def current():
    existing = getattr(_LOCAL, "guard", None) or getattr(_LOCAL, "task_guard", None)
    if existing is not None:
        existing.check_source()
        return existing
    if not active():
        return None
    marker = _read_json(MARKER)
    if marker.get("video_id") != VIDEO_ID:
        raise BudgetStop("Missing/mismatched task budget marker")
    return Guard(marker["policy_path"])


def stop_if_active(reason):
    guard = current()
    if guard:
        guard.stop(reason)


def guarded_post(transport, url, **kwargs):
    guard = current()
    return guard.post(transport, url, **kwargs) if guard else transport(url, **kwargs)


def guarded_tts(func, text, save_path, number=None, **kwargs):
    guard = current()
    if not guard:
        return func(text, save_path, **kwargs)
    return guard.execute(func, text, save_path, slot=number, **kwargs)


def reviewed_rows(df):
    # Exclude generated result columns and XLSX metadata; include source timings
    # and paragraph boundaries so an authorized re-save is not a false mismatch.
    fields = ("number", "start_time", "end_time", "text", "gender", "speaker_id", "voice", "para_id", "lines", "gap")
    return [{k: str(row[k]) for k in fields} for _, row in df.iterrows()]


def preflight(fingerprint, tasks_df):
    guard = current()
    if guard:
        guard.check_pipeline(fingerprint, reviewed_rows(tasks_df))


def task_scope(func):
    """Pin scope for the full pipeline, including gaps between paragraph calls."""
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        guard = current()
        old = getattr(_LOCAL, "task_guard", None)
        _LOCAL.task_guard = guard
        try:
            return func(*args, **kwargs)
        finally:
            _LOCAL.task_guard = old
    return wrapped


def cached_synthesis(save_path, text, number):
    """Reuse a verified successful raw response without any network request."""
    guard = current()
    if guard is None:
        raise BudgetStop("Cached guarded synthesis requested outside task scope")
    slot = str(number)
    with guard.locked():
        data = guard.state()
        guard.check_state(data)
        row = next((r for r in reversed(data["requests"]) if r["slot"] == slot and r["state"] == "success"), None)
        if not row or row["state"] != "success":
            raise BudgetStop("No confirmed successful synthesis to reuse")
    cached = _read_json(str(save_path) + ".elevenlabs.json")
    if (cached.get("schema") != 1 or cached.get("source") != "elevenlabs_response"
            or cached.get("slot") != slot or cached.get("attempt", 1) != row.get("attempt", 1)
            or cached.get("text_sha256") != text_hash(text.strip())
            or cached.get("audio_sha256") != hashlib.sha256(Path(save_path).read_bytes()).hexdigest()
            or not isinstance(cached.get("result"), dict)
            or cached["result"].get("ok") is not True):
        guard.stop("Cached raw response changed; do not re-synthesize")
    return cached["result"]
