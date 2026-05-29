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
    try:
        # Prefer session state to avoid cross-session interference in multi-user settings
        st.session_state._st_progress_placeholder = placeholder
        st.session_state._st_progress_bar = None
    except Exception:
        # Fallback to module-level globals if run outside of Streamlit active session
        global _st_progress_placeholder, _st_progress_bar
        _st_progress_placeholder = placeholder
        _st_progress_bar = None

def update_st_progress(current, total, description=""):
    # Try session state first
    try:
        placeholder = st.session_state.get("_st_progress_placeholder")
        if placeholder is not None:
            progress_val = min(1.0, current / total) if total > 0 else 0
            display_text = f"{description} ({current}/{total})"
            
            progress_bar = st.session_state.get("_st_progress_bar")
            if progress_bar is None:
                try:
                    st.session_state._st_progress_bar = placeholder.progress(progress_val, text=display_text)
                except Exception:
                    pass
            else:
                try:
                    st.session_state._st_progress_bar.progress(progress_val, text=display_text)
                except Exception:
                    # In case of delta path mismatch (React setIn error), recreate the progress bar in placeholder
                    try:
                        st.session_state._st_progress_bar = placeholder.progress(progress_val, text=display_text)
                    except Exception:
                        pass
            return
    except Exception:
        pass

    # Fallback to module-level globals
    global _st_progress_placeholder, _st_progress_bar
    if _st_progress_placeholder is not None:
        progress_val = min(1.0, current / total) if total > 0 else 0
        display_text = f"{description} ({current}/{total})"
        
        if _st_progress_bar is None:
            try:
                _st_progress_bar = _st_progress_placeholder.progress(progress_val, text=display_text)
            except Exception:
                pass
        else:
            try:
                _st_progress_bar.progress(progress_val, text=display_text)
            except Exception:
                try:
                    _st_progress_bar = _st_progress_placeholder.progress(progress_val, text=display_text)
                except Exception:
                    pass

def clear_st_progress():
    try:
        placeholder = st.session_state.get("_st_progress_placeholder")
        if placeholder is not None:
            try:
                placeholder.empty()
            except Exception:
                pass
        st.session_state._st_progress_bar = None
    except Exception:
        pass

    global _st_progress_placeholder, _st_progress_bar
    if _st_progress_placeholder is not None:
        try:
            _st_progress_placeholder.empty()
        except Exception:
            pass
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

