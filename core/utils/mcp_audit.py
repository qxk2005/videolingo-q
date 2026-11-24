"""
Persistent, cross-process MCP call audit logger with strict secret redaction.

Records MCP `initialize`, `tools/call`, `resources/read`, and `prompts/get` events
across all transports (`stdio`, `http`, `sse`) into `<APP_ROOT>/logs/mcp_audit.jsonl`
with automatic size rotation and `fcntl` file locking.
"""

from datetime import datetime
import fcntl
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from core.utils.config_utils import APP_ROOT, load_key

DEFAULT_AUDIT_LOG_PATH = os.path.join(APP_ROOT, "logs", "mcp_audit.jsonl")
MAX_AUDIT_FILE_BYTES = 5 * 1024 * 1024  # 5 MB per active file (+ .1 rotated backup)
MAX_STRING_VALUE_CHARS = 600
REDACTED_PLACEHOLDER = "***REDACTED***"

_AUDIT_THREAD_LOCK = threading.RLock()
_SESSION_CLIENTS: Dict[str, Dict[str, str]] = {}

# Exact lower-cased keys or key suffixes that hold credentials / cookies
_SENSITIVE_EXACT_KEYS = {
    "key",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "auth_token",
    "refresh_token",
    "hf_token",
    "antigravity_token_code",
    "whisperx_302_api_key",
    "elevenlabs_api_key",
    "secret",
    "secret_key",
    "client_secret",
    "password",
    "passwd",
    "authorization",
    "cookie",
    "cookies",
    "cookies_content",
    "credential",
    "credentials",
    "private_key",
}

_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_token",
    "_token_code",
    "_secret",
    "_password",
    "_passwd",
    "_private_key",
)

# Safe keys that contain words like "cookie" or "token" in config metadata but hold no secrets
_SAFE_EXACT_KEYS = {
    "cookies_path",
    "cookies_from_browser",
    "has_auth_cookie",
    "cookie_count",
    "yt_cookie_count",
    "effective_mode",
}

_INLINE_SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-+/=]{8,}", re.IGNORECASE),
    re.compile(r"\b(sk-[A-Za-z0-9_\-]{10,})\b"),
    re.compile(r"\b(xi-[A-Za-z0-9_\-]{10,})\b"),
    re.compile(r"\b(hf_[A-Za-z0-9]{10,})\b"),
]

_READ_ONLY_TOOLS = {
    "videolingo_get_status",
    "videolingo_get_config",
    "videolingo_get_tts_quota",
    "videolingo_read_subtitles",
    "videolingo_get_terminology",
    "videolingo_get_speakers",
    "videolingo_get_mcp_history",
}


def get_audit_log_path() -> str:
    env_override = os.environ.get("VIDEOLINGO_MCP_AUDIT_PATH", "").strip()
    if env_override:
        return env_override
    if "unittest" in sys.modules:
        return os.path.join(tempfile.gettempdir(), f"videolingo_unittest_mcp_audit_{os.getpid()}.jsonl")
    return DEFAULT_AUDIT_LOG_PATH


def _is_sensitive_key(key: Any) -> bool:
    raw = str(key or "").strip().lower()
    if not raw or raw in _SAFE_EXACT_KEYS:
        return False
    leaf = raw.split(".")[-1]
    if leaf in _SAFE_EXACT_KEYS:
        return False
    if raw in _SENSITIVE_EXACT_KEYS or leaf in _SENSITIVE_EXACT_KEYS:
        return True
    if any(leaf.endswith(sfx) for sfx in _SENSITIVE_KEY_SUFFIXES):
        return True
    return False


def _sanitize_string(val: str, max_chars: int = MAX_STRING_VALUE_CHARS) -> str:
    s = str(val)
    if not s:
        return ""
    # Detect Netscape cookies.txt or raw HTTP Cookie header strings
    if "# Netscape HTTP Cookie File" in s or ("\tTRUE\t" in s and ".youtube.com" in s):
        return REDACTED_PLACEHOLDER
    if re.search(r"(?:^|;\s*)(?:SID|SAPISID|__Secure-[13]PSID|LOGIN_INFO)=", s):
        return REDACTED_PLACEHOLDER
    for pat in _INLINE_SECRET_PATTERNS:
        if pat.pattern.startswith("(Bearer"):
            s = pat.sub(r"\1" + REDACTED_PLACEHOLDER, s)
        else:
            s = pat.sub(REDACTED_PLACEHOLDER, s)
    if len(s) > max_chars:
        return s[:max_chars] + f"...[truncated {len(s) - max_chars} chars]"
    return s


def redact_sensitive_data(value: Any, depth: int = 0) -> Any:
    """Return a deep, JSON-safe copy of `value` with all credentials & cookies masked."""
    if depth > 8:
        return "...[max_depth]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            k_str = str(k)
            if _is_sensitive_key(k_str):
                if v is None or v == "":
                    out[k_str] = ""
                else:
                    out[k_str] = REDACTED_PLACEHOLDER
            else:
                out[k_str] = redact_sensitive_data(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        items = list(value)
        limit = 30
        redacted_items = [redact_sensitive_data(item, depth + 1) for item in items[:limit]]
        if len(items) > limit:
            redacted_items.append(f"...[{len(items) - limit} more items]")
        return redacted_items
    return _sanitize_string(str(value))


def classify_mcp_call(event_type: str, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
    """Classify an MCP audit event as `'write'` (state/task mutation) or `'read'` (query/metadata)."""
    if event_type != "tool_call":
        return "read"
    args = arguments if isinstance(arguments, dict) else {}
    if name in _READ_ONLY_TOOLS:
        return "read"
    if name == "videolingo_batch_control" and args.get("action", "state") == "state":
        return "read"
    if name == "videolingo_manage_llm" and args.get("action") == "list_profiles":
        return "read"
    if name in ("videolingo_original_narration", "videolingo_animation_narration") and args.get("action") == "status":
        return "read"
    return "write"


def summarize_tool_result(tool_name: str, output: Any) -> Dict[str, Any]:
    """Extract a compact, high-signal summary from an MCP tool result without bloating JSONL."""
    if not isinstance(output, dict):
        return {"value": redact_sensitive_data(output)}

    summary: Dict[str, Any] = {}
    if "status" in output:
        summary["status"] = output.get("status")

    if tool_name == "videolingo_batch_add_tasks":
        action_res = output.get("action_result") if isinstance(output.get("action_result"), dict) else output
        tasks = action_res.get("tasks") if isinstance(action_res, dict) else None
        if isinstance(tasks, list):
            summary["queue_total"] = len(tasks)
            summary["recent_task_ids"] = [t.get("id") for t in tasks[-5:] if isinstance(t, dict) and t.get("id")]
        if "auto_started" in output:
            summary["auto_started"] = output.get("auto_started")
        return summary

    if tool_name in (
        "videolingo_one_shot_process",
        "videolingo_load_media",
        "videolingo_run_subtitle_pipeline",
        "videolingo_run_dubbing_pipeline",
    ):
        if "action_result" in output:
            summary["action_result"] = redact_sensitive_data(output.get("action_result"))
        ws = output.get("workspace") if isinstance(output.get("workspace"), dict) else {}
        if ws:
            summary["task_state"] = redact_sensitive_data(ws.get("task"))
            if isinstance(ws.get("video"), dict):
                summary["video_title"] = ws["video"].get("title") or ws["video"].get("file_name")
        return summary

    if tool_name == "videolingo_update_config":
        for k in ("status", "updated_keys", "tts_method"):
            if k in output:
                summary[k] = redact_sensitive_data(output[k])
        if not summary:
            summary = redact_sensitive_data(output)
        return summary

    if tool_name == "videolingo_read_subtitles":
        entries = output.get("entries") if isinstance(output.get("entries"), list) else []
        return {
            "srt_type": output.get("srt_type"),
            "total": output.get("total", len(entries)),
            "returned": len(entries),
        }

    if tool_name == "videolingo_get_config":
        cfg = output.get("config") if isinstance(output.get("config"), dict) else {}
        return {
            "tts_method": cfg.get("tts_method"),
            "target_language": cfg.get("target_language"),
            "active_profile": output.get("active_profile"),
        }

    if tool_name == "videolingo_get_status":
        task = output.get("task") if isinstance(output.get("task"), dict) else {}
        return {
            "running": task.get("running"),
            "stage": task.get("stage"),
            "step_name": task.get("step_name"),
            "percent": task.get("percent"),
        }

    if tool_name == "videolingo_get_mcp_history":
        return {
            "returned": len(output.get("items", [])) if isinstance(output.get("items"), list) else 0,
            "total_matched": output.get("total_matched", 0),
        }

    # Generic compact fallback
    compact = redact_sensitive_data(output)
    raw_json = json.dumps(compact, ensure_ascii=False)
    if len(raw_json) > 1200:
        return {"keys": list(output.keys())[:15], "preview": raw_json[:800] + "...[truncated]"}
    return compact


def remember_session_client(
    session_key: str,
    client_info: Optional[Dict[str, Any]] = None,
    transport: str = "",
    ip: str = "",
    user_agent: str = "",
) -> Dict[str, str]:
    """Store client identity reported during MCP `initialize` for subsequent calls in the same session."""
    info = client_info if isinstance(client_info, dict) else {}
    record = {
        "name": str(info.get("name") or "").strip(),
        "version": str(info.get("version") or "").strip(),
        "transport": str(transport or "").strip(),
        "ip": str(ip or "").strip(),
        "user_agent": _sanitize_string(str(user_agent or "").strip(), max_chars=160),
    }
    if session_key:
        with _AUDIT_THREAD_LOCK:
            existing = _SESSION_CLIENTS.get(session_key, {})
            merged = {
                "name": record["name"] or existing.get("name", ""),
                "version": record["version"] or existing.get("version", ""),
                "transport": record["transport"] or existing.get("transport", ""),
                "ip": record["ip"] or existing.get("ip", ""),
                "user_agent": record["user_agent"] or existing.get("user_agent", ""),
            }
            # Bound cache size
            if len(_SESSION_CLIENTS) > 256 and session_key not in _SESSION_CLIENTS:
                oldest_key = next(iter(_SESSION_CLIENTS))
                _SESSION_CLIENTS.pop(oldest_key, None)
            _SESSION_CLIENTS[session_key] = merged
            return dict(merged)
    return record


def resolve_session_client(
    session_key: str = "",
    transport: str = "",
    ip: str = "",
    user_agent: str = "",
) -> Dict[str, str]:
    with _AUDIT_THREAD_LOCK:
        cached = dict(_SESSION_CLIENTS.get(session_key, {})) if session_key else {}
    # Fallback to `"stdio_default"` if stdio process initialized without explicit session_id
    if not cached and transport == "stdio":
        with _AUDIT_THREAD_LOCK:
            cached = dict(_SESSION_CLIENTS.get("stdio_default", {}))
    elif not cached and transport in ("http", "sse") and ip:
        with _AUDIT_THREAD_LOCK:
            cached = dict(_SESSION_CLIENTS.get(f"http_ip:{ip}:{user_agent[:60]}", {}))
    return {
        "name": cached.get("name", ""),
        "version": cached.get("version", ""),
        "ip": str(ip or cached.get("ip") or ""),
        "user_agent": _sanitize_string(str(user_agent or cached.get("user_agent") or ""), max_chars=160),
    }


def _safe_read_persisted_tts() -> str:
    try:
        return str(load_key("tts_method", ignore_overrides=True) or "").strip()
    except Exception:
        return ""


def _build_tts_trace(
    event_type: str,
    name: str,
    arguments: Optional[Dict[str, Any]],
    tts_before: str,
    tts_after: str,
) -> Dict[str, str]:
    args = arguments if isinstance(arguments, dict) else {}
    task_override = ""
    if event_type == "tool_call" and name in ("videolingo_batch_add_tasks", "videolingo_one_shot_process"):
        task_override = str(args.get("tts_method") or "").strip()
    elif event_type == "tool_call" and name == "videolingo_update_config":
        updates = args.get("updates") if isinstance(args.get("updates"), dict) else {}
        if "tts_method" in updates:
            tts_after = str(updates.get("tts_method") or tts_after or "").strip()
    effective = task_override or tts_after or tts_before
    return {
        "global_before": tts_before,
        "global_after": tts_after,
        "task_override": task_override,
        "effective": effective,
    }


def _append_audit_entry(entry: Dict[str, Any]) -> None:
    log_path = get_audit_log_path()
    parent = os.path.dirname(os.path.abspath(log_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    encoded = line.encode("utf-8")

    with _AUDIT_THREAD_LOCK:
        try:
            # Rotate under exclusive lock if active file exceeds MAX_AUDIT_FILE_BYTES
            lock_path = log_path + ".lock"
            with open(lock_path, "a+", encoding="utf-8") as lock_fd:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                try:
                    if os.path.exists(log_path) and os.path.getsize(log_path) >= MAX_AUDIT_FILE_BYTES:
                        rotated = log_path + ".1"
                        os.replace(log_path, rotated)
                    with open(log_path, "ab") as f:
                        f.write(encoded)
                        f.flush()
                finally:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            # Audit logging must never crash user workloads if disk is read-only
            pass


def record_mcp_audit_event(
    *,
    event_type: str,
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    transport: str = "unknown",
    session_id: str = "",
    jsonrpc_id: Any = None,
    client_ip: str = "",
    user_agent: str = "",
    client_info: Optional[Dict[str, Any]] = None,
    status: str = "ok",
    duration_ms: float = 0.0,
    output: Any = None,
    error: str = "",
    tts_before: Optional[str] = None,
    tts_after: Optional[str] = None,
) -> Dict[str, Any]:
    """Build, redact, and persist a single MCP audit record to JSONL."""
    now = time.time()
    now_dt = datetime.fromtimestamp(now).astimezone()
    ts_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    iso_str = now_dt.isoformat(timespec="milliseconds")

    session_key = session_id or ("stdio_default" if transport == "stdio" else f"http_ip:{client_ip}:{user_agent[:60]}")
    if client_info:
        client_meta = remember_session_client(
            session_key=session_key,
            client_info=client_info,
            transport=transport,
            ip=client_ip,
            user_agent=user_agent,
        )
    else:
        client_meta = resolve_session_client(
            session_key=session_key,
            transport=transport,
            ip=client_ip,
            user_agent=user_agent,
        )

    before_m = _safe_read_persisted_tts() if tts_before is None else tts_before
    after_m = _safe_read_persisted_tts() if tts_after is None else tts_after
    category = classify_mcp_call(event_type, name, arguments)

    entry: Dict[str, Any] = {
        "audit_id": f"mcp_{int(now * 1000)}_{uuid.uuid4().hex[:6]}",
        "timestamp": ts_str,
        "iso_time": iso_str,
        "ts_epoch": round(now, 3),
        "event_type": event_type,
        "category": category,
        "transport": str(transport or "unknown"),
        "session_id": str(session_id or ""),
        "jsonrpc_id": jsonrpc_id if isinstance(jsonrpc_id, (str, int)) else None,
        "client": {
            "name": client_meta.get("name", ""),
            "version": client_meta.get("version", ""),
            "ip": client_meta.get("ip", ""),
            "user_agent": client_meta.get("user_agent", ""),
        },
        "name": str(name or ""),
        "arguments": redact_sensitive_data(arguments or {}),
        "tts_trace": _build_tts_trace(event_type, name, arguments, before_m, after_m),
        "status": "error" if error or status == "error" else "ok",
        "duration_ms": round(max(0.0, float(duration_ms or 0.0)), 1),
        "result_summary": summarize_tool_result(name, output) if output is not None else {},
        "error": _sanitize_string(error, max_chars=800) if error else "",
    }
    _append_audit_entry(entry)
    return entry


def _iter_audit_entries_newest_first() -> List[Dict[str, Any]]:
    log_path = get_audit_log_path()
    rotated_path = log_path + ".1"
    entries: List[Dict[str, Any]] = []
    for candidate in (rotated_path, log_path):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            entries.append(obj)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
    entries.reverse()
    return entries


def query_mcp_audit_history(
    *,
    limit: int = 100,
    offset: int = 0,
    category: str = "all",
    status: str = "all",
    tool_name: str = "",
    transport: str = "",
    query: str = "",
) -> Dict[str, Any]:
    """Query persisted MCP audit records (newest first) with filtering and aggregate stats."""
    safe_limit = max(1, min(500, int(limit or 100)))
    safe_offset = max(0, int(offset or 0))
    cat_filter = str(category or "all").strip().lower()
    status_filter = str(status or "all").strip().lower()
    tool_filter = str(tool_name or "").strip()
    transport_filter = str(transport or "").strip().lower()
    kw = str(query or "").strip().lower()

    all_entries = _iter_audit_entries_newest_first()

    stats = {
        "total": len(all_entries),
        "write_count": sum(1 for e in all_entries if e.get("category") == "write"),
        "read_count": sum(1 for e in all_entries if e.get("category") == "read"),
        "error_count": sum(1 for e in all_entries if e.get("status") == "error"),
    }
    tool_names = sorted({str(e.get("name") or "") for e in all_entries if e.get("name")})

    matched: List[Dict[str, Any]] = []
    for item in all_entries:
        if cat_filter in ("write", "read") and item.get("category") != cat_filter:
            continue
        if status_filter in ("ok", "error") and item.get("status") != status_filter:
            continue
        if tool_filter and item.get("name") != tool_filter:
            continue
        if transport_filter and str(item.get("transport") or "").lower() != transport_filter:
            continue
        if kw:
            haystack = json.dumps(
                {
                    "name": item.get("name"),
                    "client": item.get("client"),
                    "arguments": item.get("arguments"),
                    "tts_trace": item.get("tts_trace"),
                    "result_summary": item.get("result_summary"),
                    "error": item.get("error"),
                },
                ensure_ascii=False,
            ).lower()
            if kw not in haystack:
                continue
        matched.append(item)

    sliced = matched[safe_offset : safe_offset + safe_limit]
    return {
        "log_path": get_audit_log_path(),
        "stats": stats,
        "available_tools": tool_names,
        "total_matched": len(matched),
        "offset": safe_offset,
        "limit": safe_limit,
        "items": sliced,
    }
