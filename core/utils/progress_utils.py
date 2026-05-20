from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
import streamlit as st

_shared_progress = None
_console = Console()
_st_progress_placeholder = None
_st_progress_bar = None

def get_progress():
    global _shared_progress
    if _shared_progress is None:
        _shared_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=_console,
            transient=True
        )
    return _shared_progress

def set_st_progress_placeholder(placeholder):
    global _st_progress_placeholder, _st_progress_bar
    _st_progress_placeholder = placeholder
    _st_progress_bar = None # Reset bar so it's recreated in the new placeholder

def update_st_progress(current, total, description=""):
    global _st_progress_placeholder, _st_progress_bar
    if _st_progress_placeholder is not None:
        progress_val = min(1.0, current / total) if total > 0 else 0
        display_text = f"{description} ({current}/{total})"
        
        if _st_progress_bar is None:
            _st_progress_bar = _st_progress_placeholder.progress(progress_val, text=display_text)
        else:
            _st_progress_bar.progress(progress_val, text=display_text)

def clear_st_progress():
    global _st_progress_placeholder, _st_progress_bar
    if _st_progress_placeholder is not None:
        _st_progress_placeholder.empty()
    _st_progress_bar = None

def start_progress():
    p = get_progress()
    p.start()
    return p

def stop_progress():
    global _shared_progress
    if _shared_progress is not None:
        _shared_progress.stop()
        _shared_progress = None
    clear_st_progress()
