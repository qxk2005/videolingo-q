"""Cooperative and preemptive task cancellation utilities across worker threads, HTTP clients, and subprocesses."""
from contextlib import contextmanager
import subprocess
import threading

_CANCEL_EVENT = threading.Event()
_REGISTRY_LOCK = threading.RLock()
_ACTIVE_CLIENTS = set()
_CANCELLED_CLIENT_IDS = set()
_ACTIVE_PROCS = set()


class TaskCancelledError(BaseException):
    """Inherits from BaseException so generic `except Exception:` retry/fallback handlers do not swallow it."""


def reset_task_cancel() -> None:
    _CANCEL_EVENT.clear()


def is_task_cancelled() -> bool:
    return _CANCEL_EVENT.is_set()


def check_task_cancelled(msg: str = "任务已被手动停止") -> None:
    if _CANCEL_EVENT.is_set():
        raise TaskCancelledError(msg)


def sleep_cancellable(seconds: float, msg: str = "任务已被手动停止") -> None:
    if _CANCEL_EVENT.wait(timeout=max(0.0, float(seconds))):
        raise TaskCancelledError(msg)


def _terminate_proc(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except Exception:
                if proc.poll() is None:
                    proc.kill()
    except Exception:
        pass


def is_client_cancelled(client) -> bool:
    with _REGISTRY_LOCK:
        return id(client) in _CANCELLED_CLIENT_IDS


@contextmanager
def register_cancellable_client(client):
    check_task_cancelled()
    cid = id(client)
    with _REGISTRY_LOCK:
        _ACTIVE_CLIENTS.add(client)
    try:
        yield client
    finally:
        with _REGISTRY_LOCK:
            _ACTIVE_CLIENTS.discard(client)
            _CANCELLED_CLIENT_IDS.discard(cid)
        try:
            client.close()
        except Exception:
            pass


@contextmanager
def register_cancellable_process(proc: subprocess.Popen):
    with _REGISTRY_LOCK:
        _ACTIVE_PROCS.add(proc)
    if _CANCEL_EVENT.is_set():
        _terminate_proc(proc)
        with _REGISTRY_LOCK:
            _ACTIVE_PROCS.discard(proc)
        raise TaskCancelledError("任务已被手动停止")
    try:
        yield proc
    finally:
        with _REGISTRY_LOCK:
            _ACTIVE_PROCS.discard(proc)


def trigger_task_cancel() -> None:
    """Signal cancellation and immediately abort any in-flight HTTP client sockets and child processes."""
    _CANCEL_EVENT.set()
    with _REGISTRY_LOCK:
        clients = list(_ACTIVE_CLIENTS)
        for c in clients:
            _CANCELLED_CLIENT_IDS.add(id(c))
        procs = list(_ACTIVE_PROCS)
    for client in clients:
        try:
            client.close()
        except Exception:
            pass
    for proc in procs:
        _terminate_proc(proc)
