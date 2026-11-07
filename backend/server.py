import os
import sys
import io
import re
import json
import time
import shutil
import zipfile
import mimetypes
import threading
import traceback
import subprocess
from collections import deque
from typing import Dict, Any, List, Optional

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import logging
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Silence Streamlit bare-mode ScriptRunContext warnings when calling shared core utilities from FastAPI threads
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

from core.utils import load_key, update_key, get_ffmpeg_video_encoder
from core.utils.config_utils import CONFIG_PATH, yaml as ruamel_yaml, lock as config_lock
from core.utils.onekeycleanup import cleanup
from core.utils.cancel_utils import (
    TaskCancelledError,
    reset_task_cancel,
    trigger_task_cancel,
)
from core.utils.models import (
    _RAW_AUDIO_FILE,
    _2_CLEANED_CHUNKS,
    _3_2_SPLIT_BY_MEANING,
    _4_2_TRANSLATION,
    _5_SPLIT_SUB,
    _8_1_AUDIO_TASK,
    _OUTPUT_DIR,
)
from core import (
    _1_ytdlp,
    _2_asr,
    _2_5_diarize,
    _3_1_split_nlp,
    _3_2_split_meaning,
    _4_1_summarize,
    _4_2_translate,
    _5_split_sub,
    _6_gen_sub,
    _7_sub_into_vid,
    _8_1_audio_task,
    _8_2_dub_chunks,
    _9_refer_audio,
    _10_gen_audio,
    _11_merge_audio,
    _12_dub_to_vid,
)
from core.asr_backend.audio_preprocess import (
    convert_video_to_audio,
    reapply_vocab_correction_to_cleaned_chunks,
    detect_suspicious_terms_from_srt,
)
from core.asr_backend.ytb_subtitle_asr import use_subtitle_file
from core.st_utils.download_video_section import (
    save_youtube_url,
    load_youtube_url,
    delete_existing_videos,
    srt_to_vtt,
)
from core.st_utils.sidebar_setting import (
    _load_profiles,
    _save_profiles,
    check_api,
    DOUBAO_MALE_VOICES,
    DOUBAO_FEMALE_VOICES,
    EDGE_MALE_VOICES,
    EDGE_FEMALE_VOICES,
    get_available_tts_voices,
)
from core.tts_backend.elevenlabs_tts import (
    ELEVENLABS_MODELS,
    add_elevenlabs_shared_voice,
    elevenlabs_tts as gen_elevenlabs_tts,
    fetch_elevenlabs_my_voices,
    fetch_elevenlabs_preview_audio,
    get_elevenlabs_default_config,
    get_elevenlabs_voice_catalogs,
    search_elevenlabs_shared_voices,
)
from core.utils.audio_gender import extract_speaker_sample_clips

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"
SRC_SRT = "output/src.srt"

_SRT_FILES = [
    "output/src.srt",
    "output/trans.srt",
    "output/src_trans.srt",
    "output/trans_src.srt",
    "output/dub.srt",
]
_SUBTITLE_CACHE_FILES = [
    "output/log/split_by_nlp.txt",
    "output/log/split_by_meaning.txt",
    "output/log/terminology.json",
    "output/log/translation_results.xlsx",
    "output/log/translation_results_for_subtitles.xlsx",
    "output/log/translation_results_remerged.xlsx",
    SUB_VIDEO,
] + _SRT_FILES
_SUBTITLE_CACHE_DIRS = ["output/gpt_log"]


# =====================================================================
# Real-time Stdout/Stderr Log Interceptor & Task State Manager
# =====================================================================
class _BinaryTap:
    def __init__(self, parent_tap: "LogTap"):
        self._parent = parent_tap

    def write(self, b):
        orig_buf = getattr(self._parent.original_stream, "buffer", None)
        if orig_buf:
            try:
                orig_buf.write(b)
            except Exception:
                pass
        if isinstance(b, (bytes, bytearray)):
            text = b.decode(self._parent.encoding or "utf-8", errors="replace")
        else:
            text = str(b)
        self._parent._ingest_text(text)
        return len(b) if b else 0

    def flush(self):
        self._parent.flush()


class LogTap(io.TextIOBase):
    def __init__(self, original_stream, log_deque: deque):
        self.original_stream = original_stream
        self._log_deque = log_deque
        self._line_buf = ""
        self._binary_tap = _BinaryTap(self)

    @property
    def buffer(self):
        return self._binary_tap

    @property
    def encoding(self):
        return getattr(self.original_stream, "encoding", "utf-8") or "utf-8"

    @property
    def errors(self):
        return getattr(self.original_stream, "errors", "replace") or "replace"

    def isatty(self):
        return False

    def writable(self):
        return True

    def fileno(self):
        if self.original_stream and hasattr(self.original_stream, "fileno"):
            return self.original_stream.fileno()
        raise io.UnsupportedOperation("fileno")

    def _ingest_text(self, s: str):
        if not s:
            return
        normalized = str(s).replace("\r\n", "\n").replace("\r", "\n")
        self._line_buf += normalized
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            if clean_line:
                ts = time.strftime("%H:%M:%S")
                self._log_deque.append(f"[{ts}] {clean_line}")
                try:
                    STATE.update_from_log(clean_line)
                except Exception:
                    pass

    def write(self, s: str):
        if self.original_stream:
            try:
                self.original_stream.write(s)
            except Exception:
                pass
        if not s:
            return 0
        self._ingest_text(s)
        return len(s)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass


LOG_BUFFER: deque = deque(maxlen=300)
sys.stdout = LogTap(sys.__stdout__, LOG_BUFFER)
sys.stderr = LogTap(sys.__stderr__, LOG_BUFFER)


class PipelineState:
    def __init__(self):
        self.running: bool = False
        self.stage: Optional[str] = None  # "download" | "subtitle" | "dubbing" | "batch"
        self.step_index: int = 0
        self.total_steps: int = 0
        self.step_name: str = ""
        self.sub_step: str = ""
        self.percent: float = 0.0
        self.speed: str = ""
        self.eta: str = ""
        self.total_size: str = ""
        self.started_at: float = 0.0
        self.error: Optional[str] = None
        self.toast: Optional[str] = None
        self.speaker_review_confirmed: bool = False
        self.stop_requested: bool = False
        self.lock = threading.Lock()

    def start_task(self, stage: str, total_steps: int, step_name: str):
        reset_task_cancel()
        self.running = True
        self.stage = stage
        self.step_index = 1
        self.total_steps = total_steps
        self.step_name = step_name
        self.sub_step = "正在初始化任务环境..."
        self.percent = 2.0
        self.speed = ""
        self.eta = ""
        self.total_size = ""
        self.started_at = time.time()
        self.error = None
        self.stop_requested = False

    def finish_task(self):
        self.running = False
        self.stop_requested = False
        self.stage = None
        self.step_index = 0
        self.total_steps = 0
        self.step_name = ""
        self.sub_step = ""
        self.percent = 0.0
        self.speed = ""
        self.eta = ""
        self.total_size = ""
        self.started_at = 0.0

    def to_dict(self) -> Dict[str, Any]:
        elapsed = int(max(0.0, time.time() - self.started_at)) if (self.running and self.started_at > 0) else 0
        return {
            "running": self.running,
            "stopping": bool(self.running and self.stop_requested),
            "stage": self.stage,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "step_name": self.step_name,
            "sub_step": self.sub_step,
            "percent": round(self.percent, 1),
            "speed": self.speed,
            "eta": self.eta,
            "total_size": self.total_size,
            "elapsed_sec": elapsed,
            "error": self.error,
            "toast": self.toast,
        }

    def update_from_log(self, line: str):
        if not self.running:
            return

        # 1. yt-dlp live download progress: [download]  68.4% of   69.48MiB at  212.86KiB/s ETA 01:45
        m_dl = re.search(
            r"\[download\]\s+(\d+(?:\.\d+)?)%\s+of\s+~?(\S+)(?:\s+at\s+(\S+))?(?:\s+ETA\s+(\S+))?",
            line,
        )
        if m_dl:
            pct = float(m_dl.group(1))
            self.percent = pct
            self.total_size = m_dl.group(2)
            spd = m_dl.group(3) or ""
            eta = m_dl.group(4) or ""
            if spd and spd != "Unknown":
                self.speed = spd
            if eta and eta != "Unknown":
                self.eta = eta
            if self.stage == "download":
                self.step_index = 2
                self.sub_step = f"正在下载媒体数据流 ({pct:.1f}%)"
            return

        # 2. yt-dlp lifecycle events
        if self.stage == "download":
            if "Checking and updating yt-dlp" in line:
                self.step_index = 1
                self.sub_step = "正在检查并更新 yt-dlp 下载引擎..."
                self.percent = max(self.percent, 4.0)
            elif "Extracting URL:" in line or "Downloading webpage" in line:
                self.step_index = 1
                self.sub_step = "正在连接 YouTube 并解析视频网页..."
                self.percent = max(self.percent, 8.0)
            elif "Solving JS challenges" in line or "player API JSON" in line:
                self.step_index = 1
                self.sub_step = "正在解析 YouTube 播放流签名与高码率通道..."
                self.percent = max(self.percent, 12.0)
            elif "Extracting cookies from" in line or "Extracted" in line and "cookies" in line:
                self.step_index = 1
                self.sub_step = f"已加载浏览器 Cookie 登录态 ({line})"
                self.percent = max(self.percent, 14.0)
            elif "Writing video thumbnail" in line or "Downloading video thumbnail" in line:
                self.step_index = 2
                self.sub_step = "已获取视频高清封面与标题，开始拉取音视频流..."
                self.percent = max(self.percent, 16.0)
            elif "[download] Destination:" in line:
                dest_file = os.path.basename(line.split("Destination:", 1)[1].strip())
                self.step_index = 2
                self.sub_step = f"正在下载流文件: {dest_file}"
            elif "[Merger]" in line or "Merging formats" in line:
                self.step_index = 3
                self.sub_step = "正在使用 FFmpeg 封装合并高清视频与音频流..."
                self.percent = 95.0
                self.eta = "即将完成"
            return

        # 3. tqdm progress bars (WhisperX / Translation / TTS)
        m_tqdm = re.search(
            r"(\d{1,3})%\|.*\|\s*(\d+/\d+)\s*\[([^<]+)<([^,\]]+)(?:,\s*([^\]]+))?\]",
            line,
        )
        if m_tqdm:
            self.percent = float(m_tqdm.group(1))
            self.sub_step = f"处理进度: {m_tqdm.group(2)} ({m_tqdm.group(1)}%)"
            self.eta = m_tqdm.group(4).strip()
            if m_tqdm.group(5):
                self.speed = m_tqdm.group(5).strip()
            return

        # 4. Informative stage logs
        if any(
            k in line
            for k in [
                "[🎬",
                "[📊",
                "[🔍",
                "[✂️",
                "[📝",
                "[✅",
                "[🎙️",
                "WhisperX",
                "Pyannote",
                "Translating",
                " Splitting",
                "Merging",
                "Converting",
            ]
        ):
            self.sub_step = line[:100]


STATE = PipelineState()

app = FastAPI(title="VideoLingo Q 2.0 Studio API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Helper Functions
# =====================================================================
def _read_full_config() -> Dict[str, Any]:
    with config_lock:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = ruamel_yaml.load(f)
    res = json.loads(json.dumps(dict(data), default=str))
    el_defaults = get_elevenlabs_default_config()
    if not isinstance(res.get("elevenlabs_tts"), dict):
        res["elevenlabs_tts"] = dict(el_defaults)
    else:
        for dk, dv in el_defaults.items():
            if dk not in res["elevenlabs_tts"] or res["elevenlabs_tts"][dk] is None:
                res["elevenlabs_tts"][dk] = dv
    return res


def _build_voice_catalogs_payload() -> Dict[str, Any]:
    el_male, el_female = get_elevenlabs_voice_catalogs()
    return {
        "doubao_male": DOUBAO_MALE_VOICES,
        "doubao_female": DOUBAO_FEMALE_VOICES,
        "edge_male": EDGE_MALE_VOICES,
        "edge_female": EDGE_FEMALE_VOICES,
        "elevenlabs_male": el_male,
        "elevenlabs_female": el_female,
        "elevenlabs_models": ELEVENLABS_MODELS,
        "current_available": get_available_tts_voices(),
    }


def _get_video_info(file_path: str) -> Optional[Dict[str, str]]:
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        abs_path = os.path.abspath(file_path)
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            abs_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
        return {
            "File": os.path.basename(file_path),
            "VideoCodec": v_stream.get("codec_name", "N/A"),
            "AudioCodec": a_stream.get("codec_name", "N/A"),
            "Resolution": f"{v_stream.get('width', 'N/A')}x{v_stream.get('height', 'N/A')}",
            "PixelFormat": v_stream.get("pix_fmt", "N/A"),
            "Size": f"{os.path.getsize(file_path) / (1024 * 1024):.1f} MB",
        }
    except Exception:
        return None


def _build_file_tree(root_dir: str = "output") -> List[Dict[str, Any]]:
    if not os.path.exists(root_dir):
        return []

    def _scan(curr_path: str) -> List[Dict[str, Any]]:
        nodes = []
        try:
            items = sorted(os.listdir(curr_path))
        except Exception:
            return nodes
        dirs = [
            d
            for d in items
            if os.path.isdir(os.path.join(curr_path, d))
            and not d.startswith(".")
            and d != "__pycache__"
        ]
        files = [
            f
            for f in items
            if os.path.isfile(os.path.join(curr_path, f))
            and not f.startswith(".")
            and f != "youtube_cookies.txt"
        ]
        for d in dirs:
            full = os.path.join(curr_path, d)
            rel = os.path.relpath(full, "output")
            children = _scan(full)
            nodes.append(
                {
                    "name": d,
                    "path": rel,
                    "type": "dir",
                    "children": children,
                    "count": len(children),
                }
            )
        for f in files:
            full = os.path.join(curr_path, f)
            rel = os.path.relpath(full, "output")
            try:
                size_bytes = os.path.getsize(full)
            except OSError:
                # Transient temporary file (e.g. output/audio/tmp/*_temp_trimmed.wav) was removed mid-scan
                continue
            if size_bytes >= 1024 * 1024:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            elif size_bytes >= 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes} B"
            nodes.append(
                {
                    "name": f,
                    "path": rel,
                    "type": "file",
                    "size": size_str,
                    "url": f"/api/media/{rel}",
                }
            )
        return nodes

    return _scan(root_dir)


def _format_sec(val: Any) -> str:
    try:
        td = pd.to_timedelta(val)
        total_sec = td.total_seconds()
        m, s = divmod(max(0.0, total_sec), 60)
        h, m = divmod(int(m), 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:05.2f}"
        return f"00:{m:02d}:{s:06.3f}"
    except Exception:
        return str(val)


def _to_sec_float(val: Any) -> float:
    try:
        return round(float(pd.to_timedelta(val).total_seconds()), 3)
    except Exception:
        return 0.0


def _build_speaker_review_payload() -> Dict[str, Any]:
    if not os.path.exists(_8_1_AUDIO_TASK):
        return {"available": False, "speakers": [], "lines": [], "voices": {}}

    try:
        df = pd.read_excel(_8_1_AUDIO_TASK)
    except Exception:
        return {"available": False, "speakers": [], "lines": [], "voices": {}}

    if df.empty:
        return {"available": False, "speakers": [], "lines": [], "voices": {}}

    if "gender" not in df.columns:
        df["gender"] = "female"
    if "speaker_id" not in df.columns:
        df["speaker_id"] = [
            "SPEAKER_00" if g == "male" else "SPEAKER_01" for g in df["gender"]
        ]
    if "voice" not in df.columns:
        df["voice"] = ""
    df["voice"] = df["voice"].fillna("").astype(str)
    df["speaker_id"] = df["speaker_id"].fillna("SPEAKER_00").astype(str)
    df["gender"] = df["gender"].fillna("female").astype(str)

    sample_clips = extract_speaker_sample_clips(df)
    voice_catalog = get_available_tts_voices()

    unique_speakers = list(dict.fromkeys(df["speaker_id"].tolist()))
    speakers_list = []
    for spk in unique_speakers:
        spk_rows = df[df["speaker_id"] == spk]
        line_count = len(spk_rows)
        mode_gender = (
            spk_rows["gender"].mode().iloc[0]
            if not spk_rows["gender"].mode().empty
            else "female"
        )
        non_empty_voices = [
            v for v in spk_rows["voice"].tolist() if str(v).strip()
        ]
        cur_voice = non_empty_voices[0] if non_empty_voices else ""
        sample_row = spk_rows.iloc[0]
        sample_text = "" if pd.isna(sample_row.get("text")) else str(sample_row.get("text", ""))
        sample_origin = "" if pd.isna(sample_row.get("origin")) else str(sample_row.get("origin", ""))
        sample_start = _to_sec_float(sample_row.get("start_time", 0))
        clip_path = sample_clips.get(spk)
        clip_url = None
        if clip_path and os.path.exists(clip_path):
            rel = os.path.relpath(clip_path, "output")
            clip_url = f"/api/media/{rel}"

        speakers_list.append(
            {
                "speaker_id": spk,
                "count": line_count,
                "gender": mode_gender,
                "voice": cur_voice,
                "sample_text": sample_text,
                "sample_origin": sample_origin,
                "sample_start": sample_start,
                "sample_timecode": _format_sec(sample_row.get("start_time", 0)),
                "clip_url": clip_url,
            }
        )

    lines_list = []
    for _, row in df.iterrows():
        lines_list.append(
            {
                "number": int(row["number"]),
                "start_sec": _to_sec_float(row.get("start_time", 0)),
                "end_sec": _to_sec_float(row.get("end_time", 0)),
                "time_range": f"{_format_sec(row.get('start_time'))} → {_format_sec(row.get('end_time'))}",
                "speaker_id": str(row.get("speaker_id", "SPEAKER_00")),
                "gender": str(row.get("gender", "female")),
                "voice": str(row.get("voice", "")),
                "cut_off": int(row.get("cut_off", 0)) if pd.notna(row.get("cut_off")) else 0,
                "text": "" if pd.isna(row.get("text")) else str(row.get("text", "")),
                "origin": "" if pd.isna(row.get("origin")) else str(row.get("origin", "")),
            }
        )

    return {
        "available": True,
        "speakers": speakers_list,
        "lines": lines_list,
        "voices": voice_catalog,
    }


# =====================================================================
# API Endpoints: Configuration, Cookies & Profiles
# =====================================================================
COOKIES_FILE_PATH = "output/youtube_cookies.txt"


def _normalize_cookies_content(raw_text: str) -> str:
    """
    Normalize Cookie content into a valid Netscape HTTP Cookie File format
    compatible with both yt-dlp and Python http.cookiejar.MozillaCookieJar.
    Supports:
    1) Netscape cookies.txt (with or without header, including #HttpOnly_ lines)
    2) JSON array exported by Cookie-Editor / EditThisCookie
    3) Raw HTTP 'Cookie: name=val; ...' header string copied from browser DevTools
    """
    text = (raw_text or "").strip().lstrip("\ufeff")
    if not text:
        raise ValueError("Cookie 内容为空，请上传或粘贴有效的 Cookie 内容")

    header = "# Netscape HTTP Cookie File\n# Normalized by VideoLingo Q\n\n"
    default_exp = int(time.time()) + 365 * 86400

    # 1. JSON format (Cookie-Editor / EditThisCookie)
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            cookie_list = parsed if isinstance(parsed, list) else parsed.get("cookies", [])
            if not isinstance(cookie_list, list) or not cookie_list:
                raise ValueError("JSON 中未找到有效的 Cookie 数组")
            lines = []
            for item in cookie_list:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                domain = str(item.get("domain") or ".youtube.com").strip() or ".youtube.com"
                include_subdomains = "FALSE" if item.get("hostOnly") is True and not domain.startswith(".") else "TRUE"
                if include_subdomains == "TRUE" and not domain.startswith("."):
                    domain = "." + domain
                path = str(item.get("path") or "/").strip() or "/"
                secure = "TRUE" if item.get("secure", True) else "FALSE"
                exp_raw = item.get("expirationDate") or item.get("expires") or default_exp
                try:
                    expires = max(0, int(float(exp_raw)))
                except Exception:
                    expires = default_exp
                name = str(item.get("name")).strip()
                value = str(item.get("value") if item.get("value") is not None else "").replace("\t", "").replace("\n", "").replace("\r", "")
                lines.append(f"{domain}\t{include_subdomains}\t{path}\t{secure}\t{expires}\t{name}\t{value}")
            if not lines:
                raise ValueError("JSON 中没有可解析的 Cookie 字段")
            return header + "\n".join(lines) + "\n"
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 格式 Cookie 解析失败: {e}")

    # 2. Netscape 7-column tab-separated format
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    netscape_lines = []
    for ln in raw_lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("#HttpOnly_"):
            stripped = stripped[len("#HttpOnly_"):]
        elif stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) >= 7:
            domain = parts[0].strip()
            flag = "TRUE" if parts[1].strip().upper() == "TRUE" or domain.startswith(".") else "FALSE"
            path = parts[2].strip() or "/"
            secure = "TRUE" if parts[3].strip().upper() == "TRUE" else "FALSE"
            try:
                expires = max(0, int(float(parts[4].strip())))
            except Exception:
                expires = default_exp
            name = parts[5].strip()
            value = "\t".join(parts[6:]).strip()
            if domain and name:
                netscape_lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}")

    if netscape_lines:
        return header + "\n".join(netscape_lines) + "\n"

    # 3. Raw HTTP Cookie header string (e.g. 'Cookie: SID=...; SAPISID=...; LOGIN_INFO=...')
    cookie_str = text
    if cookie_str.lower().startswith("cookie:"):
        cookie_str = cookie_str.split(":", 1)[1].strip()
    if "=" in cookie_str and (";" in cookie_str or len(cookie_str) > 12):
        pairs = [seg.strip() for seg in cookie_str.split(";") if seg.strip() and "=" in seg]
        kv_lines = []
        for pair in pairs:
            k, v = pair.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k or "\n" in k or " " in k or "\t" in k:
                continue
            kv_lines.append(f".youtube.com\tTRUE\t/\tTRUE\t{default_exp}\t{k}\t{v}")
        if kv_lines:
            return header + "\n".join(kv_lines) + "\n"

    raise ValueError(
        "无法识别该 Cookie 格式。支持：1) Netscape 格式 cookies.txt；2) Cookie-Editor 导出的 JSON；3) 浏览器 F12 复制的 Cookie 请求头字符串。"
    )


def _save_normalized_cookies(raw_text: str, clear_browser: bool = False) -> Dict[str, Any]:
    normalized = _normalize_cookies_content(raw_text)
    os.makedirs("output", exist_ok=True)
    with open(COOKIES_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(normalized)
    try:
        os.chmod(COOKIES_FILE_PATH, 0o600)
    except Exception:
        pass
    update_key("youtube.cookies_path", COOKIES_FILE_PATH)
    if clear_browser:
        update_key("youtube.cookies_from_browser", "")
    return _get_cookies_status()


def _get_cookies_status() -> Dict[str, Any]:
    configured_path = str(load_key("youtube.cookies_path") or "").strip()
    target_file = configured_path if (configured_path and os.path.exists(configured_path)) else COOKIES_FILE_PATH
    file_exists = os.path.exists(target_file) and os.path.getsize(target_file) > 0

    cookie_count = 0
    yt_cookie_count = 0
    has_auth_cookie = False
    updated_at = ""
    size_bytes = 0
    auth_names = {
        "SAPISID",
        "__Secure-3PAPISID",
        "__Secure-1PAPISID",
        "SID",
        "__Secure-3PSID",
        "__Secure-1PSID",
        "LOGIN_INFO",
    }

    if file_exists:
        try:
            st = os.stat(target_file)
            size_bytes = st.st_size
            updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    s = ln.strip()
                    if not s or (s.startswith("#") and not s.startswith("#HttpOnly_")):
                        continue
                    if s.startswith("#HttpOnly_"):
                        s = s[len("#HttpOnly_"):]
                    parts = s.split("\t")
                    if len(parts) >= 7:
                        cookie_count += 1
                        dom = parts[0].lower()
                        cname = parts[5].strip()
                        if "youtube.com" in dom or "google.com" in dom or "googlevideo.com" in dom:
                            yt_cookie_count += 1
                        if cname in auth_names and parts[6].strip():
                            has_auth_cookie = True
        except Exception:
            pass

    browser = ""
    try:
        browser = str(load_key("youtube.cookies_from_browser") or "").strip()
    except Exception:
        browser = ""

    if file_exists:
        effective_mode = "file"
    elif browser:
        effective_mode = "browser"
    else:
        effective_mode = "none"

    return {
        "effective_mode": effective_mode,
        "file_exists": file_exists,
        "file_path": target_file if file_exists else COOKIES_FILE_PATH,
        "size_bytes": size_bytes,
        "updated_at": updated_at,
        "cookie_count": cookie_count,
        "yt_cookie_count": yt_cookie_count,
        "has_auth_cookie": has_auth_cookie,
        "cookies_from_browser": browser,
    }


@app.get("/api/config")
def get_config():
    cfg = _read_full_config()
    profiles = _load_profiles()
    custom_terms = str(load_key("domain_vocab") or "")
    cookies_status = _get_cookies_status()
    return {
        "config": cfg,
        "profiles": profiles,
        "custom_terms": custom_terms,
        "cookies_exists": cookies_status["file_exists"],
        "cookies_status": cookies_status,
        "voice_catalogs": _build_voice_catalogs_payload(),
    }


class ConfigUpdatePayload(BaseModel):
    updates: Dict[str, Any] = {}
    custom_terms: Optional[str] = None
    cookies_content: Optional[str] = None


@app.post("/api/config")
def update_config(payload: ConfigUpdatePayload):
    for k, v in payload.updates.items():
        update_key(k, v)
    if payload.custom_terms is not None:
        update_key("domain_vocab", payload.custom_terms)
    if payload.cookies_content and payload.cookies_content.strip():
        try:
            _save_normalized_cookies(payload.cookies_content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    cookies_status = _get_cookies_status()
    return {
        "status": "ok",
        "config": _read_full_config(),
        "voice_catalogs": _build_voice_catalogs_payload(),
        "cookies_exists": cookies_status["file_exists"],
        "cookies_status": cookies_status,
    }


class ProfileActionPayload(BaseModel):
    action: str  # "save" | "load" | "delete"
    name: str


@app.post("/api/llm/profiles")
def manage_llm_profile(payload: ProfileActionPayload):
    profiles = _load_profiles()
    if payload.action == "save":
        profiles[payload.name] = {
            "key": load_key("api.key") or "",
            "base_url": load_key("api.base_url") or "",
            "model": load_key("api.model") or "",
            "llm_support_json": bool(load_key("api.llm_support_json")),
            "use_antigravity_cli": bool(load_key("api.use_antigravity_cli")),
        }
        _save_profiles(profiles)
        return {"status": "ok", "profiles": profiles, "message": f"已保存预设: {payload.name}"}
    elif payload.action == "load":
        if payload.name not in profiles:
            raise HTTPException(status_code=404, detail="预设不存在")
        p = profiles[payload.name]
        for sub_k in ["key", "base_url", "model", "llm_support_json", "use_antigravity_cli"]:
            if sub_k in p:
                update_key(f"api.{sub_k}", p[sub_k])
        return {"status": "ok", "config": _read_full_config(), "message": f"已加载预设: {payload.name}"}
    elif payload.action == "delete":
        if payload.name in profiles:
            del profiles[payload.name]
            _save_profiles(profiles)
        return {"status": "ok", "profiles": profiles, "message": f"已删除预设: {payload.name}"}
    raise HTTPException(status_code=400, detail="未知操作")


class LlmCheckPayload(BaseModel):
    use_antigravity_cli: Optional[bool] = None
    antigravity_token_code: Optional[str] = None
    key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    llm_support_json: Optional[bool] = None
    auto_save_on_success: bool = True


@app.post("/api/llm/check")
def check_llm_connectivity(payload: Optional[LlmCheckPayload] = None):
    from openai import OpenAI
    from core.utils.ask_gpt import ask_antigravity_cli

    t0 = time.perf_counter()
    use_cli = payload.use_antigravity_cli if (payload and payload.use_antigravity_cli is not None) else bool(load_key("api.use_antigravity_cli"))
    token_code = payload.antigravity_token_code if (payload and payload.antigravity_token_code is not None) else (load_key("api.antigravity_token_code") or "")
    api_key = (payload.key if (payload and payload.key is not None) else (load_key("api.key") or "")).strip()
    raw_base_url = (payload.base_url if (payload and payload.base_url is not None) else (load_key("api.base_url") or "")).strip()
    model_name = (payload.model if (payload and payload.model is not None) else (load_key("api.model") or "")).strip()
    support_json = payload.llm_support_json if (payload and payload.llm_support_json is not None) else bool(load_key("api.llm_support_json"))
    auto_save = True if payload is None else bool(payload.auto_save_on_success)

    test_prompt = 'This is a connectivity test, please respond with {"message": "success"} in json format.'
    try:
        if use_cli:
            if token_code:
                update_key("api.antigravity_token_code", token_code)
            resp_content = ask_antigravity_cli(test_prompt)
            target_label = "antigravity-cli (agy)"
        else:
            if not api_key:
                return {
                    "valid": False,
                    "saved": False,
                    "latency_ms": 0,
                    "model": model_name or "未填写",
                    "error": "API_KEY 不能为空，请先填写密钥",
                    "message": "API_KEY 未填写",
                }
            if not raw_base_url:
                return {
                    "valid": False,
                    "saved": False,
                    "latency_ms": 0,
                    "model": model_name or "未填写",
                    "error": "BASE_URL 不能为空，请先填写接口地址",
                    "message": "BASE_URL 未填写",
                }
            if not model_name:
                return {
                    "valid": False,
                    "saved": False,
                    "latency_ms": 0,
                    "model": "未填写",
                    "error": "MODEL 不能为空，请先填写模型名称",
                    "message": "MODEL 未填写",
                }

            if "ark" in raw_base_url:
                norm_base_url = "https://ark.cn-beijing.volces.com/api/v3"
            elif "v1" not in raw_base_url:
                norm_base_url = raw_base_url.rstrip("/") + "/v1"
            else:
                norm_base_url = raw_base_url

            client = OpenAI(api_key=api_key, base_url=norm_base_url, max_retries=0)
            resp_raw = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                timeout=25,
            )
            resp_content = (resp_raw.choices[0].message.content or "").strip()
            target_label = model_name

        latency_ms = int((time.perf_counter() - t0) * 1000)
        if not resp_content:
            return {
                "valid": False,
                "saved": False,
                "latency_ms": latency_ms,
                "model": target_label,
                "error": "模型返回内容为空，请确认模型名称与接口协议兼容",
                "message": "模型返回为空",
            }

        saved = False
        if auto_save:
            update_key("api.use_antigravity_cli", bool(use_cli))
            if token_code is not None:
                update_key("api.antigravity_token_code", token_code)
            if not use_cli:
                update_key("api.key", api_key)
                update_key("api.base_url", raw_base_url)
                update_key("api.model", model_name)
                update_key("api.llm_support_json", bool(support_json))
            saved = True

        clean_preview = re.sub(r"<think>.*?</think>", "", resp_content, flags=re.DOTALL).strip()
        clean_preview = re.sub(r"\s+", " ", clean_preview)[:120]
        return {
            "valid": True,
            "saved": saved,
            "latency_ms": latency_ms,
            "model": target_label,
            "reply_preview": clean_preview or resp_content[:120],
            "message": "API 连接测试通过，已自动保存 LLM 配置" if saved else "API 连接测试通过",
        }
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        err_detail = str(e).strip() or repr(e)
        return {
            "valid": False,
            "saved": False,
            "latency_ms": latency_ms,
            "model": "antigravity-cli (agy)" if use_cli else (model_name or "未知模型"),
            "error": err_detail,
            "message": "API 连接失败",
        }


class VoicePreviewPayload(BaseModel):
    tts_method: str
    voice_id: str
    api_key: Optional[str] = None
    model_id: Optional[str] = None


@app.post("/api/tts/preview")
def preview_tts_voice(payload: VoicePreviewPayload):
    import tempfile
    preview_text = "这是您的配音音色预览，听起来自然清晰吧？"
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        if payload.tts_method == "doubao_tts":
            from core.tts_backend.doubao_tts import doubao_tts as gen_doubao_tts
            gen_doubao_tts(preview_text, tmp_path, voice_type=payload.voice_id)
        elif payload.tts_method == "elevenlabs_tts":
            if payload.api_key and payload.api_key.strip():
                update_key("elevenlabs_tts.api_key", payload.api_key.strip())
            if payload.model_id and payload.model_id.strip():
                update_key("elevenlabs_tts.model_id", payload.model_id.strip())
            gen_elevenlabs_tts(preview_text, tmp_path, voice_type=payload.voice_id)
        else:
            from core.tts_backend.edge_tts import edge_tts
            edge_tts(preview_text, tmp_path, voice=payload.voice_id)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


class ElevenLabsSyncPayload(BaseModel):
    api_key: Optional[str] = None


@app.post("/api/elevenlabs/voices/sync")
def sync_elevenlabs_voices_api(payload: ElevenLabsSyncPayload):
    try:
        override_k = payload.api_key.strip() if payload.api_key and payload.api_key.strip() else None
        if override_k:
            update_key("elevenlabs_tts.api_key", override_k)
        sync_res = fetch_elevenlabs_my_voices(api_key=override_k)
        return {
            "status": "ok",
            "total": sync_res["total"],
            "chinese_count": sync_res["chinese_count"],
            "voice_catalogs": _build_voice_catalogs_payload(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/elevenlabs/shared-voices")
def list_elevenlabs_shared_voices_api(
    search: str = "",
    gender: str = "",
    accent: str = "",
    page_size: int = 24,
    page: int = 0,
    api_key: str = "",
):
    try:
        override_k = api_key.strip() if api_key and api_key.strip() else None
        if override_k:
            update_key("elevenlabs_tts.api_key", override_k)
        res = search_elevenlabs_shared_voices(
            search=search,
            gender=gender,
            accent=accent,
            page_size=page_size,
            page=page,
            api_key=override_k,
        )
        return {"status": "ok", **res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/elevenlabs/shared-voices/preview")
def preview_elevenlabs_shared_voice_api(url: str):
    try:
        audio_bytes, content_type = fetch_elevenlabs_preview_audio(url)
        return StreamingResponse(io.BytesIO(audio_bytes), media_type=content_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class ElevenLabsAddSharedVoicePayload(BaseModel):
    public_owner_id: str = ""
    voice_id: str
    name: str
    gender: str = ""
    accent: str = ""
    preview_url: str = ""
    set_as: str = ""  # "male" | "female" | ""
    api_key: Optional[str] = None


@app.post("/api/elevenlabs/shared-voices/add")
def add_elevenlabs_shared_voice_api(payload: ElevenLabsAddSharedVoicePayload):
    try:
        override_k = payload.api_key.strip() if payload.api_key and payload.api_key.strip() else None
        if override_k:
            update_key("elevenlabs_tts.api_key", override_k)
        res = add_elevenlabs_shared_voice(
            public_owner_id=payload.public_owner_id,
            voice_id=payload.voice_id,
            name=payload.name,
            gender=payload.gender,
            accent=payload.accent,
            preview_url=payload.preview_url,
            set_as=payload.set_as,
            api_key=override_k,
        )
        return {
            "status": "ok",
            "voice_id": res["voice_id"],
            "name": res["name"],
            "set_as": res["set_as"],
            "config": _read_full_config(),
            "voice_catalogs": _build_voice_catalogs_payload(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class CookiesPastePayload(BaseModel):
    content: str
    clear_browser: bool = False


@app.get("/api/cookies/status")
def get_cookies_status_api():
    st = _get_cookies_status()
    return {"status": "ok", "cookies_exists": st["file_exists"], "cookies_status": st}


@app.post("/api/cookies/upload")
async def upload_cookies_file(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    try:
        st = _save_normalized_cookies(raw_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "status": "ok",
        "cookies_exists": st["file_exists"],
        "cookies_status": st,
        "config": _read_full_config(),
    }


@app.post("/api/cookies/paste")
def paste_cookies_content(payload: CookiesPastePayload):
    try:
        st = _save_normalized_cookies(payload.content, clear_browser=payload.clear_browser)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "status": "ok",
        "cookies_exists": st["file_exists"],
        "cookies_status": st,
        "config": _read_full_config(),
    }


@app.delete("/api/cookies")
def delete_cookies():
    configured_path = str(load_key("youtube.cookies_path") or "").strip()
    for candidate in {COOKIES_FILE_PATH, configured_path}:
        if candidate and os.path.abspath(candidate).startswith(os.path.abspath("output")) and os.path.exists(candidate):
            try:
                os.remove(candidate)
            except Exception:
                pass
    if configured_path == COOKIES_FILE_PATH:
        update_key("youtube.cookies_path", "")
    st = _get_cookies_status()
    return {
        "status": "ok",
        "cookies_exists": st["file_exists"],
        "cookies_status": st,
        "config": _read_full_config(),
    }


# =====================================================================
# API Endpoints: Workspace Live State
# =====================================================================
@app.get("/api/workspace/state")
def get_workspace_state():
    orig_video = None
    try:
        orig_video = _1_ytdlp.find_video_files()
    except Exception:
        orig_video = None

    vtt_url = None
    if orig_video and os.path.exists(orig_video):
        srt_path = "output/youtube_subtitle.srt"
        vtt_path = "output/youtube_subtitle.vtt"
        if os.path.exists(srt_path):
            if not os.path.exists(vtt_path) or os.path.getmtime(srt_path) > os.path.getmtime(vtt_path):
                srt_to_vtt(srt_path, vtt_path)
        if os.path.exists(vtt_path):
            vtt_url = "/api/media/youtube_subtitle.vtt"

    title_path = "output/video_title.txt"
    video_title = "暂未获取到视频标题"
    if os.path.exists(title_path):
        try:
            with open(title_path, "r", encoding="utf-8") as f:
                video_title = f.read().strip() or video_title
        except Exception:
            pass
    elif orig_video:
        video_title = os.path.splitext(os.path.basename(orig_video))[0]

    cover_path = "output/video_cover.jpg"
    cover_url = "/api/media/video_cover.jpg" if os.path.exists(cover_path) else None
    if not cover_url and os.path.exists("output"):
        for fname in sorted(os.listdir("output")):
            if fname.lower().endswith((".webp", ".jpg", ".jpeg", ".png")):
                cover_url = f"/api/media/{fname}"
                if video_title == "暂未获取到视频标题":
                    video_title = os.path.splitext(fname)[0]
                break

    orig_rel = os.path.relpath(orig_video, "output") if orig_video and os.path.exists(orig_video) else None

    # Subtitle steps completion
    text_steps = [
        {"id": 1, "name": "1. WhisperX / 官方字幕直达", "done": os.path.exists(_2_CLEANED_CHUNKS)},
        {"id": 2, "name": "2. NLP + LLM 语义分句", "done": os.path.exists(_3_2_SPLIT_BY_MEANING)},
        {"id": 3, "name": "3. 摘要与三步法翻译", "done": os.path.exists(_4_2_TRANSLATION)},
        {"id": 4, "name": "4. 长字幕切分与对齐", "done": os.path.exists(_5_SPLIT_SUB)},
        {"id": 5, "name": "5. 生成时间轴与双语字幕", "done": os.path.exists(SRC_SRT)},
        {"id": 6, "name": "6. 烧录字幕到视频", "done": os.path.exists(SUB_VIDEO)},
    ]

    suspicious_terms = []
    if os.path.exists(SRC_SRT):
        try:
            suspicious_terms = detect_suspicious_terms_from_srt(SRC_SRT)[:25]
        except Exception:
            suspicious_terms = []

    # Dubbing steps completion
    pause_before_dub = bool(load_key("pause_before_dub"))
    has_audio_tasks = os.path.exists(_8_1_AUDIO_TASK)
    has_dub_mp3 = os.path.exists("output/dub.mp3") or os.path.exists("output/audio/dub.mp3")
    has_dub_video = os.path.exists(DUB_VIDEO)

    waiting_confirm = (
        has_audio_tasks
        and not has_dub_video
        and pause_before_dub
        and not STATE.speaker_review_confirmed
        and not STATE.running
    )

    dub_steps = [
        {"id": 1, "name": "1. 音频任务与说话人识别", "done": has_audio_tasks},
        {
            "id": 2,
            "name": "2. 说话人音色确认",
            "done": has_audio_tasks and (not pause_before_dub or STATE.speaker_review_confirmed or has_dub_video),
            "waiting": waiting_confirm,
        },
        {"id": 3, "name": "3. 生成并对齐配音", "done": has_dub_mp3 or has_dub_video},
        {"id": 4, "name": "4. 音画混流合成成片", "done": has_dub_video},
    ]

    # Encoding comparison
    encoding_rows = []
    for stage_label, path in [
        ("Original", orig_video),
        ("Subtitled", SUB_VIDEO if os.path.exists(SUB_VIDEO) else None),
        ("Dubbed", DUB_VIDEO if os.path.exists(DUB_VIDEO) else None),
    ]:
        if path and os.path.exists(path):
            info = _get_video_info(path)
            if info:
                info["Stage"] = stage_label
                encoding_rows.append(info)

    return {
        "video": {
            "exists": bool(orig_video and os.path.exists(orig_video)),
            "file_name": os.path.basename(orig_video) if orig_video else None,
            "orig_url": f"/api/media/{orig_rel}" if orig_rel else None,
            "vtt_url": vtt_url,
            "sub_url": "/api/media/output_sub.mp4" if os.path.exists(SUB_VIDEO) else None,
            "dub_url": "/api/media/output_dub.mp4" if os.path.exists(DUB_VIDEO) else None,
            "title": video_title,
            "cover_url": cover_url,
            "youtube_url": load_youtube_url(),
        },
        "stage_b": {
            "steps": text_steps,
            "src_srt_exists": os.path.exists(SRC_SRT),
            "sub_video_exists": os.path.exists(SUB_VIDEO),
            "suspicious_terms": suspicious_terms,
        },
        "stage_c": {
            "steps": dub_steps,
            "waiting_confirm": waiting_confirm,
            "dub_video_exists": has_dub_video,
            "speaker_review": _build_speaker_review_payload(),
        },
        "task": STATE.to_dict(),
        "files": _build_file_tree("output"),
        "encoding": encoding_rows,
        "logs": list(LOG_BUFFER)[-80:],
        "studio_batch_task": BATCH_MANAGER.get_studio_loaded_task_info() if "BATCH_MANAGER" in globals() else None,
        "has_single_stash": BATCH_MANAGER.has_single_stash() if "BATCH_MANAGER" in globals() else False,
    }


# =====================================================================
# API Endpoints: Stage A (Download / Upload Video & Official Subtitles)
# =====================================================================
class YoutubeDownloadPayload(BaseModel):
    url: str
    resolution: str = "1080"
    subtitle_only: bool = False


def _wipe_output_workspace():
    """Clean all current task files in output/ while preserving youtube_cookies.txt if present."""
    STATE.speaker_review_confirmed = False
    delete_existing_videos()
    if os.path.exists("output"):
        for item in os.listdir("output"):
            if item == "youtube_cookies.txt":
                continue
            full_path = os.path.join("output", item)
            try:
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)
            except Exception:
                pass


@app.post("/api/video/download")
def download_youtube(payload: YoutubeDownloadPayload):
    if STATE.running:
        raise HTTPException(status_code=409, detail="当前已有任务正在运行")

    def _worker():
        with STATE.lock:
            STATE.start_task(
                stage="download",
                total_steps=4,
                step_name="正在解析并下载 YouTube 媒体..." if not payload.subtitle_only else "正在下载 YouTube 官方字幕...",
            )

        try:
            url = payload.url.strip()
            if not payload.subtitle_only:
                if "BATCH_MANAGER" in globals():
                    BATCH_MANAGER.detach_studio_task()
                _wipe_output_workspace()
                STATE.step_name = f"正在下载 YouTube 视频 ({payload.resolution}p)..."
                STATE.sub_step = "正在启动 yt-dlp 解析链接与 Cookie 登录态..."
                _1_ytdlp.download_video_ytdlp(url, resolution=payload.resolution)
                save_youtube_url(url)

            STATE.step_index = 4
            STATE.percent = 97.0
            STATE.speed = ""
            STATE.eta = "即将完成"
            STATE.step_name = "正在检测并下载 YouTube 官方字幕..."
            STATE.sub_step = "正在探测原片 CC 字幕或自动生成字幕轨道..."
            sub_path, sub_type = _1_ytdlp.download_subtitle_ytdlp(url)
            if sub_path:
                save_youtube_url(url)
                lang = load_key("whisper.language")
                if lang == "auto":
                    lang = "en"
                STATE.sub_step = f"正在清洗并应用官方字幕 ({sub_type})..."
                use_subtitle_file(sub_path, lang)
                STATE.percent = 100.0
                STATE.toast = f"🎉 已成功下载并应用官方字幕 ({sub_type})，已自动跳过 WhisperX！"
            else:
                STATE.percent = 100.0
                STATE.toast = "✅ 视频下载完成（未检测到英文字幕，后续将使用 WhisperX 转录）"
        except (InterruptedError, TaskCancelledError):
            STATE.toast = "⏹️ 视频下载任务已手动停止"
        except Exception as e:
            traceback.print_exc()
            STATE.error = str(e)
        finally:
            if "BATCH_MANAGER" in globals():
                BATCH_MANAGER.sync_from_studio_if_loaded()
            STATE.finish_task()

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


@app.post("/api/video/upload")
def upload_local_video(file: UploadFile = File(...)):
    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.detach_studio_task()
    _wipe_output_workspace()
    os.makedirs("output", exist_ok=True)
    raw_name = file.filename or "uploaded_video.mp4"
    ext = raw_name.rsplit(".", 1)[-1].lower() if "." in raw_name else "mp4"
    base_name = re.sub(r"[^\w\-_\.]", "_", raw_name.rsplit(".", 1)[0])
    clean_filename = f"{base_name}.{ext}"
    save_path = os.path.join("output", clean_filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with open(os.path.join("output", "video_title.txt"), "w", encoding="utf-8") as f:
        f.write(clean_filename)

    if ext in ["mp3", "wav", "m4a", "flac"]:
        vcodec = get_ffmpeg_video_encoder() or "libx264"
        mp4_path = os.path.join("output", f"{base_name}.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1280x720:r=1",
                "-i",
                save_path,
                "-c:v",
                vcodec,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-shortest",
                mp4_path,
            ],
            check=False,
        )
        if os.path.exists(mp4_path):
            os.remove(save_path)
    return {"status": "ok", "file": raw_name}


@app.delete("/api/video")
def delete_video():
    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.detach_studio_task()
    _wipe_output_workspace()
    return {"status": "ok"}


@app.post("/api/subtitle/upload")
def upload_subtitle_bypass(file: UploadFile = File(...)):
    os.makedirs("output", exist_ok=True)
    ext = (file.filename or "sub.srt").rsplit(".", 1)[-1].lower()
    save_path = f"output/uploaded_subtitle.{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    lang = load_key("whisper.language")
    if lang == "auto":
        lang = "en"
    ok = use_subtitle_file(save_path, lang)
    if not ok:
        raise HTTPException(status_code=400, detail="字幕文件解析失败，请检查格式")
    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.sync_from_studio_if_loaded()
    return {"status": "ok"}


# =====================================================================
# API Endpoints: Stage B (Subtitle Translation Pipeline)
# =====================================================================
@app.post("/api/pipeline/subtitle/start")
def start_subtitle_pipeline(burn_only: bool = False):
    if STATE.running:
        raise HTTPException(status_code=409, detail="当前已有任务正在运行")

    def _worker():
        with STATE.lock:
            STATE.start_task(
                stage="subtitle",
                total_steps=6,
                step_name="6. 正在烧录字幕到视频..." if burn_only else "1. 正在准备字幕与三步翻译流水线...",
            )

        try:
            if burn_only:
                STATE.step_index = 6
                STATE.percent = 85.0
                STATE.step_name = "6. 正在烧录字幕到视频..."
                STATE.sub_step = "正在调用 FFmpeg 将双语字幕渲染进视频画面..."
                _7_sub_into_vid.merge_subtitles_to_video()
                STATE.percent = 100.0
                STATE.toast = "🎉 字幕烧录完成！"
                return

            if not os.path.exists(_2_CLEANED_CHUNKS):
                STATE.step_index = 1
                STATE.percent = 10.0
                STATE.step_name = "1. 正在执行 WhisperX 转录与说话人分离..."
                STATE.sub_step = "正在提取人声并执行 WhisperX 语音识别..."
                _2_asr.transcribe()
                STATE.percent = 22.0
                STATE.sub_step = "正在执行 Pyannote 多说话人声纹分离..."
                _2_5_diarize.diarize_main()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if not os.path.exists(_3_2_SPLIT_BY_MEANING):
                STATE.step_index = 2
                STATE.percent = 35.0
                STATE.step_name = "2. 正在使用 NLP + LLM 进行语义分句..."
                STATE.sub_step = "正在执行 SpaCy 语法切分与 LLM 语义断句..."
                _3_1_split_nlp.split_by_spacy()
                _3_2_split_meaning.split_sentences_by_meaning()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if not os.path.exists(_4_2_TRANSLATION):
                STATE.step_index = 3
                STATE.percent = 55.0
                STATE.step_name = "3. 正在提取术语并执行三步翻译..."
                STATE.sub_step = "正在提取视频主题摘要与专业领域术语..."
                _4_1_summarize.get_summary()
                STATE.percent = 65.0
                STATE.sub_step = "正在并发执行「直译 -> 反思 -> 意译」三步翻译..."
                _4_2_translate.translate_all()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if not os.path.exists(_5_SPLIT_SUB):
                STATE.step_index = 4
                STATE.percent = 80.0
                STATE.step_name = "4. 正在切分与对齐长字幕..."
                STATE.sub_step = "正在根据屏幕单行字数限制自适应切分长句..."
                _5_split_sub.split_for_sub_main()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if not os.path.exists(SRC_SRT):
                STATE.step_index = 5
                STATE.percent = 90.0
                STATE.step_name = "5. 正在生成双语字幕时间轴..."
                STATE.sub_step = "正在对齐词级时间戳并执行去重叠平滑..."
                _6_gen_sub.align_timestamp_main()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if load_key("burn_subtitles") and load_key("auto_burn_subtitles") and not os.path.exists(SUB_VIDEO):
                STATE.step_index = 6
                STATE.percent = 95.0
                STATE.step_name = "6. 正在烧录字幕到视频..."
                STATE.sub_step = "正在调用 FFmpeg 硬字幕压制合成 output_sub.mp4..."
                _7_sub_into_vid.merge_subtitles_to_video()

            STATE.percent = 100.0
            STATE.toast = "🎉 字幕翻译与生成全部完成！"
        except (InterruptedError, TaskCancelledError):
            STATE.toast = "⏹️ 字幕任务已手动停止，已完成步骤缓存已保留"
        except Exception as e:
            traceback.print_exc()
            STATE.error = str(e)
        finally:
            if "BATCH_MANAGER" in globals():
                BATCH_MANAGER.sync_from_studio_if_loaded()
            STATE.finish_task()

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


@app.post("/api/pipeline/subtitle/reapply-terms")
def reapply_terms():
    reset_task_cancel()
    reapply_vocab_correction_to_cleaned_chunks()
    for f in _SUBTITLE_CACHE_FILES:
        if os.path.exists(f):
            os.remove(f)
    for d in _SUBTITLE_CACHE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)
    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.sync_from_studio_if_loaded()
    return {"status": "ok", "message": "已重新应用词汇表并清空下游字幕缓存"}


@app.post("/api/pipeline/subtitle/reset")
def reset_subtitles():
    for f in _SUBTITLE_CACHE_FILES:
        if os.path.exists(f):
            os.remove(f)
    for d in _SUBTITLE_CACHE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)
    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.sync_from_studio_if_loaded()
    return {"status": "ok"}


@app.get("/api/subtitle/download-zip")
def download_subtitle_zip():
    zip_buffer = io.BytesIO()
    if os.path.exists("output"):
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for fname in os.listdir("output"):
                if fname.endswith(".srt"):
                    fpath = os.path.join("output", fname)
                    with open(fpath, "rb") as f:
                        zf.writestr(fname, f.read())
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="subtitles.zip"'},
    )


@app.post("/api/pipeline/archive")
def archive_to_history():
    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.detach_studio_task()
    cleanup()
    STATE.speaker_review_confirmed = False
    return {"status": "ok"}


# =====================================================================
# API Endpoints: Stage C (Multi-Speaker Dubbing & Speaker Review)
# =====================================================================
@app.post("/api/pipeline/dubbing/start")
def start_dubbing_pipeline(force_continue: bool = False):
    if STATE.running:
        raise HTTPException(status_code=409, detail="当前已有任务正在运行")

    if force_continue:
        STATE.speaker_review_confirmed = True

    def _worker():
        with STATE.lock:
            STATE.start_task(
                stage="dubbing",
                total_steps=4,
                step_name="1. 正在准备音频并执行说话人性别识别与配音分块...",
            )

        try:
            # Ensure output/audio/raw.mp3 exists (especially when Stage B used official subtitles and skipped WhisperX)
            if not os.path.exists(_RAW_AUDIO_FILE):
                STATE.step_index = 1
                STATE.percent = 8.0
                STATE.sub_step = "正在从原视频提取基础音轨 (raw.mp3)..."
                video_file = _1_ytdlp.find_video_files()
                if video_file and os.path.exists(video_file):
                    convert_video_to_audio(video_file)

            if not os.path.exists(_8_1_AUDIO_TASK):
                STATE.step_index = 1
                STATE.percent = 15.0
                STATE.step_name = "1. 正在执行说话人性别识别与配音分块..."
                STATE.sub_step = "正在使用 Wav2Vec2 + 基频特征判定每个说话人的性别与边界..."
                _8_1_audio_task.gen_audio_task_main()
                STATE.percent = 30.0
                STATE.sub_step = "正在计算跨角色隔离的配音切分点..."
                _8_2_dub_chunks.gen_dub_chunks()
            else:
                sync_res = _8_1_audio_task.sync_subtitles_to_audio_files()
                df_existing = pd.read_excel(_8_1_AUDIO_TASK)
                has_lines = (
                    "lines" in df_existing.columns
                    and len(df_existing) > 0
                    and df_existing["lines"].notna().all()
                )
                needs_rechunk = (
                    bool(sync_res.get("updated_task_rows", 0) > 0)
                    or "cut_off" not in df_existing.columns
                    or "gap" not in df_existing.columns
                    or "para_id" not in df_existing.columns
                    or (len(df_existing) > 0 and float(df_existing["gap"].iloc[-1]) <= 0)
                )
                if not has_lines:
                    STATE.step_index = 1
                    STATE.percent = 30.0
                    STATE.sub_step = "正在重新计算字幕时间容差与跨角色配音切分点..."
                    _8_2_dub_chunks.gen_dub_chunks()
                elif needs_rechunk:
                    STATE.step_index = 1
                    STATE.percent = 30.0
                    STATE.sub_step = "正在重新计算字幕时间容差与跨角色配音切分点..."
                    df_existing = _8_2_dub_chunks.analyze_subtitle_timing_and_speed(df_existing)
                    df_existing = _8_2_dub_chunks.process_cutoffs(df_existing)
                    df_existing.to_excel(_8_1_AUDIO_TASK, index=False)

            if STATE.stop_requested:
                raise InterruptedError("任务已被手动停止")

            pause_before_dub = bool(load_key("pause_before_dub"))
            if pause_before_dub and not STATE.speaker_review_confirmed:
                STATE.percent = 50.0
                STATE.toast = "⏸️ 已完成说话人识别，请确认角色性别与音色后点击「确认说话人并继续配音」"
                return

            STATE.step_index = 2
            STATE.percent = 55.0
            STATE.step_name = "2. 正在准备参考音频..."
            if not load_key("skip_refer"):
                STATE.sub_step = "正在分离人声并为各说话人抽取清晰参考音频切片..."
                _9_refer_audio.extract_refer_audio_main()
            else:
                STATE.sub_step = "当前 TTS 无需提取参考音频，跳过 Demucs 分离..."

            if STATE.stop_requested:
                raise InterruptedError("任务已被手动停止")

            STATE.step_index = 3
            STATE.percent = 65.0
            STATE.step_name = "3. 正在并发生成多角色配音并对齐时间轴..."
            STATE.sub_step = "正在调用 TTS 引擎并发合成各角色语音片段..."
            _10_gen_audio.gen_audio()
            STATE.percent = 85.0
            STATE.sub_step = "正在拼接完整配音轨道与对齐时间轴..."
            _11_merge_audio.merge_full_audio()

            if STATE.stop_requested:
                raise InterruptedError("任务已被手动停止")

            STATE.step_index = 4
            STATE.percent = 95.0
            STATE.step_name = "4. 正在混流配音与背景音合成最终视频..."
            STATE.sub_step = "正在调用 FFmpeg 混流配音、背景伴奏与画面..."
            _12_dub_to_vid.merge_video_audio()
            STATE.percent = 100.0
            STATE.toast = "🎉 多角色配音视频合成完成！"
        except (InterruptedError, TaskCancelledError):
            STATE.toast = "⏹️ 配音任务已手动停止，已生成片段缓存已保留"
        except Exception as e:
            traceback.print_exc()
            STATE.error = str(e)
        finally:
            if "BATCH_MANAGER" in globals():
                BATCH_MANAGER.sync_from_studio_if_loaded(
                    speaker_confirmed=bool(STATE.speaker_review_confirmed or os.path.exists(DUB_VIDEO))
                )
            STATE.finish_task()

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


class SpeakerOverrideItem(BaseModel):
    speaker_id: str
    gender: str
    voice: str = ""


class LineOverrideItem(BaseModel):
    number: int
    speaker_id: str
    gender: str
    voice: str = ""
    text: Optional[str] = None


class SaveSpeakerReviewPayload(BaseModel):
    speakers: List[SpeakerOverrideItem] = []
    lines: List[LineOverrideItem] = []
    incremental_rebuild: bool = False
    force_sync_numbers: List[int] = []


@app.post("/api/speakers/save")
def save_speaker_overrides(payload: SaveSpeakerReviewPayload):
    if not os.path.exists(_8_1_AUDIO_TASK):
        raise HTTPException(status_code=404, detail="配音任务表尚未生成")

    if not os.path.exists(_RAW_AUDIO_FILE):
        video_file = _1_ytdlp.find_video_files()
        if video_file and os.path.exists(video_file):
            convert_video_to_audio(video_file)

    df = pd.read_excel(_8_1_AUDIO_TASK)
    from core.tts_backend.tts_main import ensure_segment_hashes_initialized
    ensure_segment_hashes_initialized(df, os.path.dirname(_8_1_AUDIO_TASK))

    for col, default in [("gender", "female"), ("speaker_id", "SPEAKER_00"), ("voice", "")]:
        if col not in df.columns:
            df[col] = default
    df["voice"] = df["voice"].fillna("").astype(str)
    df["speaker_id"] = df["speaker_id"].fillna("SPEAKER_00").astype(str)
    df["gender"] = df["gender"].fillna("female").astype(str)

    affected_numbers = set()
    force_sync_set = set(payload.force_sync_numbers or [])
    spk_map = {s.speaker_id: s for s in payload.speakers}
    line_map = {l.number: l for l in payload.lines}

    for idx, row in df.iterrows():
        num = int(row["number"])
        orig_spk = str(row["speaker_id"])
        orig_gender = str(row["gender"])
        orig_voice = str(row["voice"])
        orig_text = "" if pd.isna(row.get("text")) else str(row.get("text"))

        new_spk = orig_spk
        new_gender = orig_gender
        new_voice = orig_voice
        new_text = orig_text

        # 1. Speaker-level card override
        if orig_spk in spk_map:
            target = spk_map[orig_spk]
            new_gender = target.gender
            new_voice = target.voice

        # 2. Line-level override (only overrides speaker card if the line itself changed relative to original disk state)
        if num in line_map:
            l_item = line_map[num]
            if l_item.speaker_id != orig_spk:
                new_spk = l_item.speaker_id
            if l_item.gender != orig_gender:
                new_gender = l_item.gender
            if l_item.voice != orig_voice:
                new_voice = l_item.voice
            if l_item.text is not None and l_item.text != orig_text:
                new_text = l_item.text

        text_changed = (new_text != orig_text) or (num in force_sync_set)
        if (
            new_spk != orig_spk
            or new_gender != orig_gender
            or new_voice != orig_voice
            or text_changed
        ):
            df.at[idx, "speaker_id"] = new_spk
            df.at[idx, "gender"] = new_gender
            df.at[idx, "voice"] = new_voice
            df.at[idx, "text"] = new_text
            if text_changed:
                df.at[idx, "lines"] = _8_2_dub_chunks.sync_lines_with_text(
                    row.get("lines"), new_text
                )
            affected_numbers.add(num)

    # Re-evaluate timing and cut_off boundaries across speakers
    df = _8_2_dub_chunks.analyze_subtitle_timing_and_speed(df)
    df = _8_2_dub_chunks.process_cutoffs(df)
    df.to_excel(_8_1_AUDIO_TASK, index=False)

    deleted_count = _10_gen_audio.invalidate_audio_cache_for_numbers(list(affected_numbers))

    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.sync_from_studio_if_loaded(speaker_confirmed=True)

    if payload.incremental_rebuild:
        STATE.speaker_review_confirmed = True
        start_dubbing_pipeline(force_continue=True)

    return {
        "status": "ok",
        "affected_lines": len(affected_numbers),
        "deleted_cache_files": deleted_count,
    }


@app.delete("/api/pipeline/dubbing")
def delete_dubbing_files():
    STATE.speaker_review_confirmed = False
    for f in [
        DUB_VIDEO,
        "output/dub.wav",
        "output/dub.mp3",
        "output/audio/dub.mp3",
        "output/dub.srt",
        "output/audio/video_segments_map.json",
        "output/audio/retimed_background.wav",
        "output/audio/.paragraph_manifest.json",
        "output/audio/dub_filter_complex.txt",
        _8_1_AUDIO_TASK,
    ]:
        if os.path.exists(f):
            os.remove(f)
    for d in ["output/audio/segs", "output/audio/tmp"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.sync_from_studio_if_loaded(speaker_confirmed=False)
    return {"status": "ok"}


def _request_and_wait_stop(timeout_sec: float = 8.0) -> bool:
    """Signal immediate cooperative + preemptive cancellation and wait for worker threads to save checkpoints and exit."""
    was_active = bool(STATE.running or ("BATCH_MANAGER" in globals() and BATCH_MANAGER.running))
    STATE.stop_requested = True
    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.stop_requested = True

    if not was_active:
        STATE.stop_requested = False
        if "BATCH_MANAGER" in globals():
            BATCH_MANAGER.stop_requested = False
        STATE.toast = "⏹️ 当前没有正在运行的任务"
        return True

    STATE.toast = "⏹️ 正在中断当前步骤并保存任务断点..."
    trigger_task_cancel()

    deadline = time.time() + max(0.2, float(timeout_sec))
    while time.time() < deadline:
        batch_running = bool("BATCH_MANAGER" in globals() and BATCH_MANAGER.running)
        if not STATE.running and not batch_running:
            STATE.toast = "⏹️ 任务与批量队列已停止，已完成步骤断点已保存"
            return True
        time.sleep(0.05)

    trigger_task_cancel()
    batch_running = bool("BATCH_MANAGER" in globals() and BATCH_MANAGER.running)
    stopped = (not STATE.running) and (not batch_running)
    if stopped:
        STATE.toast = "⏹️ 任务与批量队列已停止，已完成步骤断点已保存"
    else:
        STATE.toast = "⏹️ 已发送强制中断信号，正在等待当前底层算子收尾保存断点..."
    return stopped


@app.post("/api/task/stop")
def stop_current_task():
    stopped = _request_and_wait_stop(timeout_sec=8.0)
    queue_info = BATCH_MANAGER.get_state_payload().get("queue") if "BATCH_MANAGER" in globals() else None
    return {
        "status": "ok",
        "stopped": stopped,
        "state": "stopped" if stopped else "stopping",
        "task": STATE.to_dict(),
        "queue": queue_info,
    }


@app.post("/api/task/clear-error")
def clear_task_error():
    STATE.error = None
    return {"status": "ok"}


# =====================================================================
# Web Native Batch Queue Manager & Endpoints
# =====================================================================
BATCH_STATE_FILE = "batch/queue_state.json"
BATCH_TASKS_DIR = "batch/tasks"
SINGLE_STASH_DIR = "batch/_single_workspace_stash"


def _copy_workspace_files(src_dir: str, dst_dir: str, clear_dst: bool = True, preserve_cookies: bool = False):
    """Synchronize workspace files between output/ and batch/tasks/<id>/ safely."""
    os.makedirs(dst_dir, exist_ok=True)
    if clear_dst:
        for item in os.listdir(dst_dir):
            if preserve_cookies and item == "youtube_cookies.txt":
                continue
            p = os.path.join(dst_dir, item)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            except Exception:
                pass

    if not os.path.exists(src_dir):
        return

    for item in os.listdir(src_dir):
        if item == "youtube_cookies.txt":
            continue
        s_path = os.path.join(src_dir, item)
        d_path = os.path.join(dst_dir, item)
        try:
            if os.path.isdir(s_path):
                if clear_dst and os.path.exists(d_path):
                    shutil.rmtree(d_path)
                shutil.copytree(s_path, d_path, dirs_exist_ok=not clear_dst)
            else:
                shutil.copy2(s_path, d_path)
        except Exception:
            pass


def _find_task_orig_video(task_dir: str) -> Optional[str]:
    if not os.path.exists(task_dir):
        return None
    try:
        allowed = [ext.lower() for ext in (load_key("allowed_video_formats") or ["mp4", "mov", "mkv", "webm", "avi", "flv"])]
    except Exception:
        allowed = ["mp4", "mov", "mkv", "webm", "avi", "flv"]
    candidates = []
    for fname in os.listdir(task_dir):
        fpath = os.path.join(task_dir, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1][1:].lower()
            if ext in allowed and not fname.startswith("output_"):
                candidates.append(fname)
    return candidates[0] if candidates else None


class BatchQueueManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.tasks: List[Dict[str, Any]] = []
        self.running: bool = False
        self.pause_after_current: bool = False
        self.stop_requested: bool = False
        self.active_task_id: Optional[str] = None
        self.studio_loaded_task_id: Optional[str] = None
        self.revision: int = 1
        os.makedirs(BATCH_TASKS_DIR, exist_ok=True)
        self._load_or_migrate()

    def _bump_rev(self):
        self.revision += 1

    def _now_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _load_or_migrate(self):
        with self.lock:
            if os.path.exists(BATCH_STATE_FILE):
                try:
                    with open(BATCH_STATE_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.tasks = data.get("tasks", [])
                    self.studio_loaded_task_id = data.get("studio_loaded_task_id")
                    for t in self.tasks:
                        if t.get("status") == "running":
                            if t.get("id") and t.get("id") == self.studio_loaded_task_id and os.path.exists("output"):
                                _copy_workspace_files("output", self.get_task_dir(t["id"]), clear_dst=False)
                            t["status"] = "paused"
                            t["current_step"] = "上次运行中断，可断点续跑"
                        elif t.get("status") == "probing":
                            t["status"] = "pending"
                    self._save()
                except Exception as e:
                    print(f"Warning: Failed to read {BATCH_STATE_FILE}: {e}")
                    self.tasks = []
            elif os.path.exists("batch/tasks_setting.xlsx"):
                # One-time seamless migration from legacy Excel sheet
                try:
                    df = pd.read_excel("batch/tasks_setting.xlsx").fillna("")
                    migrated = []
                    for idx, row in df.iterrows():
                        vfile = str(row.get("Video File", "")).strip()
                        if not vfile:
                            continue
                        slang = str(row.get("Source Language", "")).strip()
                        tlang = str(row.get("Target Language", "")).strip()
                        dub_raw = row.get("Dubbing", "")
                        is_dub = str(dub_raw).strip() in ("1", "1.0", "True", "true")
                        raw_status = str(row.get("Status", "")).strip()
                        status = "pending"
                        if raw_status.lower() == "done":
                            status = "completed"
                        elif "error" in raw_status.lower():
                            status = "error"
                        tid = f"task_{int(time.time())}_{idx + 1}"
                        os.makedirs(os.path.join(BATCH_TASKS_DIR, tid), exist_ok=True)
                        migrated.append(
                            {
                                "id": tid,
                                "url": vfile,
                                "title": vfile,
                                "thumbnail": "",
                                "uploader": "",
                                "duration_str": "",
                                "source_lang": slang,
                                "target_lang": tlang,
                                "mode": "dub_video" if is_dub else "sub_video",
                                "resolution": str(load_key("ytb_resolution") or "1080"),
                                "prefer_official_sub": True,
                                "tts_method": "",
                                "male_voice": "",
                                "female_voice": "",
                                "pause_before_dub": False,
                                "speaker_confirmed": False,
                                "status": status,
                                "current_step": raw_status if status == "error" else ("已完成" if status == "completed" else "排队等待中"),
                                "step_index": 0,
                                "total_steps": 10 if is_dub else 6,
                                "progress": 100.0 if status == "completed" else 0.0,
                                "error_msg": raw_status if status == "error" else "",
                                "created_at": self._now_str(),
                                "updated_at": self._now_str(),
                            }
                        )
                    self.tasks = migrated
                    self._save()
                    # Probe metadata for migrated URL tasks in background
                    probe_ids = [t["id"] for t in self.tasks if t["url"].startswith("http")]
                    if probe_ids:
                        threading.Thread(target=self._probe_tasks_worker, args=(probe_ids,), daemon=True).start()
                except Exception as e:
                    print(f"Warning: Failed to migrate legacy Excel: {e}")

    def _save(self):
        with self.lock:
            os.makedirs(os.path.dirname(BATCH_STATE_FILE), exist_ok=True)
            payload = {
                "studio_loaded_task_id": self.studio_loaded_task_id,
                "tasks": self.tasks,
            }
            tmp_file = BATCH_STATE_FILE + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, BATCH_STATE_FILE)
            self._bump_rev()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            for t in self.tasks:
                if t["id"] == task_id:
                    return t
            return None

    def get_task_dir(self, task_id: str) -> str:
        return os.path.join(BATCH_TASKS_DIR, task_id)

    def has_single_stash(self) -> bool:
        if not os.path.exists(SINGLE_STASH_DIR):
            return False
        items = [x for x in os.listdir(SINGLE_STASH_DIR) if not x.startswith(".")]
        return len(items) > 0

    def get_studio_loaded_task_info(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            if not self.studio_loaded_task_id:
                return None
            t = self.get_task(self.studio_loaded_task_id)
            if not t:
                self.studio_loaded_task_id = None
                return None
            return {
                "id": t["id"],
                "title": t.get("title") or t.get("url"),
                "url": t.get("url"),
                "status": t.get("status"),
                "mode": t.get("mode"),
            }

    def detach_studio_task(self):
        with self.lock:
            if self.studio_loaded_task_id:
                self.studio_loaded_task_id = None
                self._save()

    def _stash_single_workspace_if_needed(self):
        """If output/ has files that do NOT belong to a batch task, stash them into SINGLE_STASH_DIR."""
        with self.lock:
            if self.studio_loaded_task_id:
                prev_dir = self.get_task_dir(self.studio_loaded_task_id)
                _copy_workspace_files("output", prev_dir, clear_dst=True)
                self.studio_loaded_task_id = None
                self._save()
                return

            if not os.path.exists("output"):
                return
            meaningful_items = [
                x for x in os.listdir("output")
                if x not in ("youtube_cookies.txt", ".DS_Store")
            ]
            if meaningful_items:
                _copy_workspace_files("output", SINGLE_STASH_DIR, clear_dst=True)

    def restore_single_stash(self) -> bool:
        with self.lock:
            if not self.has_single_stash():
                return False
            if self.studio_loaded_task_id:
                prev_dir = self.get_task_dir(self.studio_loaded_task_id)
                _copy_workspace_files("output", prev_dir, clear_dst=True)
                self.studio_loaded_task_id = None
            _copy_workspace_files(SINGLE_STASH_DIR, "output", clear_dst=True, preserve_cookies=True)
            shutil.rmtree(SINGLE_STASH_DIR, ignore_errors=True)
            self._save()
            return True

    def load_task_to_studio(self, task_id: str) -> Dict[str, Any]:
        with self.lock:
            t = self.get_task(task_id)
            if not t:
                raise HTTPException(status_code=404, detail="找不到该批量任务")
            self._stash_single_workspace_if_needed()
            task_dir = self.get_task_dir(task_id)
            _copy_workspace_files(task_dir, "output", clear_dst=True, preserve_cookies=True)
            if t.get("url") and t["url"].startswith("http"):
                save_youtube_url(t["url"])
            self.studio_loaded_task_id = task_id
            STATE.speaker_review_confirmed = bool(t.get("speaker_confirmed"))
            self._save()
            return t

    def sync_from_studio_if_loaded(self, speaker_confirmed: Optional[bool] = None):
        with self.lock:
            if not self.studio_loaded_task_id:
                return
            t = self.get_task(self.studio_loaded_task_id)
            if not t:
                self.studio_loaded_task_id = None
                return
            task_dir = self.get_task_dir(t["id"])
            _copy_workspace_files("output", task_dir, clear_dst=True)
            if speaker_confirmed is not None:
                t["speaker_confirmed"] = speaker_confirmed
            # Update title if video_title.txt exists
            t_file = os.path.join(task_dir, "video_title.txt")
            if os.path.exists(t_file):
                try:
                    with open(t_file, "r", encoding="utf-8") as f:
                        new_title = f.read().strip()
                        if new_title:
                            t["title"] = new_title
                except Exception:
                    pass
            # Inspect completion based on task mode
            mode = t.get("mode", "dub_video")
            has_dub = os.path.exists(os.path.join(task_dir, "output_dub.mp4"))
            has_sub_vid = os.path.exists(os.path.join(task_dir, "output_sub.mp4"))
            has_srt = os.path.exists(os.path.join(task_dir, "src.srt"))
            has_audio_task = os.path.exists(os.path.join(task_dir, "log", "tts_tasks.xlsx")) or os.path.exists(
                os.path.join(task_dir, "audio", "tts_tasks.xlsx")
            )
            if mode == "dub_video" and has_dub:
                t["status"] = "completed"
                t["progress"] = 100.0
                t["current_step"] = "🎉 已在单视频工作台完成配音合成"
                t["error_msg"] = ""
            elif mode == "sub_video" and has_sub_vid:
                t["status"] = "completed"
                t["progress"] = 100.0
                t["current_step"] = "🎉 已完成字幕烧录视频"
                t["error_msg"] = ""
            elif mode == "sub_only" and has_srt:
                t["status"] = "completed"
                t["progress"] = 100.0
                t["current_step"] = "🎉 已完成双语字幕生成"
                t["error_msg"] = ""
            elif t.get("status") == "waiting_review" and t.get("speaker_confirmed"):
                t["status"] = "pending"
                t["current_step"] = "✅ 已确认说话人音色，等待继续合成配音"
            elif has_audio_task and t.get("pause_before_dub") and not t.get("speaker_confirmed"):
                t["status"] = "waiting_review"
                t["current_step"] = "⏸️ 待确认说话人音色"
            t["updated_at"] = self._now_str()
            self._save()

    def _enrich_task(self, t: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(t)
        tid = item["id"]
        # Check live directory (if currently running or loaded in studio, inspect output/ as well)
        task_dir = self.get_task_dir(tid)
        inspect_dir = "output" if (tid == self.active_task_id or tid == self.studio_loaded_task_id) and os.path.exists("output") else task_dir

        orig_vid = _find_task_orig_video(inspect_dir) or _find_task_orig_video(task_dir)
        has_sub_vid = os.path.exists(os.path.join(inspect_dir, "output_sub.mp4")) or os.path.exists(os.path.join(task_dir, "output_sub.mp4"))
        has_dub_vid = os.path.exists(os.path.join(inspect_dir, "output_dub.mp4")) or os.path.exists(os.path.join(task_dir, "output_dub.mp4"))
        has_src_srt = os.path.exists(os.path.join(inspect_dir, "src.srt")) or os.path.exists(os.path.join(task_dir, "src.srt"))
        has_trans_srt = os.path.exists(os.path.join(inspect_dir, "trans.srt")) or os.path.exists(os.path.join(task_dir, "trans.srt"))
        has_bi_srt = os.path.exists(os.path.join(inspect_dir, "src_trans.srt")) or os.path.exists(os.path.join(task_dir, "src_trans.srt"))
        has_cover = os.path.exists(os.path.join(inspect_dir, "video_cover.jpg")) or os.path.exists(os.path.join(task_dir, "video_cover.jpg"))

        has_checkpoint = bool(
            orig_vid
            or has_src_srt
            or os.path.exists(os.path.join(task_dir, "log", "cleaned_chunks.xlsx"))
            or os.path.exists(os.path.join(task_dir, "audio", "tts_tasks.xlsx"))
        )

        if has_cover:
            item["thumbnail"] = f"/api/batch/media/{tid}/video_cover.jpg"

        item["outputs"] = {
            "has_checkpoint": has_checkpoint,
            "orig_video_url": f"/api/batch/media/{tid}/{orig_vid}" if orig_vid else None,
            "sub_video_url": f"/api/batch/media/{tid}/output_sub.mp4" if has_sub_vid else None,
            "dub_video_url": f"/api/batch/media/{tid}/output_dub.mp4" if has_dub_vid else None,
            "src_srt_url": f"/api/batch/media/{tid}/src.srt" if has_src_srt else None,
            "trans_srt_url": f"/api/batch/media/{tid}/trans.srt" if has_trans_srt else None,
            "bilingual_srt_url": f"/api/batch/media/{tid}/src_trans.srt" if has_bi_srt else None,
            "zip_srt_url": f"/api/batch/tasks/{tid}/download-subtitles" if (has_src_srt or has_trans_srt) else None,
        }
        return item

    def get_state_payload(self) -> Dict[str, Any]:
        with self.lock:
            enriched = [self._enrich_task(t) for t in self.tasks]
            stats = {
                "total": len(enriched),
                "pending": sum(1 for t in enriched if t["status"] in ("pending", "probing")),
                "running": sum(1 for t in enriched if t["status"] == "running"),
                "waiting_review": sum(1 for t in enriched if t["status"] == "waiting_review"),
                "paused": sum(1 for t in enriched if t["status"] == "paused"),
                "completed": sum(1 for t in enriched if t["status"] == "completed"),
                "error": sum(1 for t in enriched if t["status"] == "error"),
            }
            return {
                "tasks": enriched,
                "queue": {
                    "running": self.running,
                    "stopping": bool(self.running and self.stop_requested),
                    "pause_after_current": self.pause_after_current,
                    "active_task_id": self.active_task_id,
                    "studio_loaded_task_id": self.studio_loaded_task_id,
                    "has_single_stash": self.has_single_stash(),
                    "revision": self.revision,
                    "stats": stats,
                },
            }

    def add_tasks(self, payload: "BatchAddTasksPayload") -> List[Dict[str, Any]]:
        raw_urls = []
        for line in payload.urls:
            for sub in str(line).splitlines():
                u = sub.strip()
                if u:
                    raw_urls.append(u)

        if not raw_urls:
            raise HTTPException(status_code=400, detail="请至少输入一个有效的视频链接")

        added_ids = []
        now_ts = int(time.time() * 1000)
        with self.lock:
            for idx, url in enumerate(raw_urls):
                tid = f"task_{now_ts}_{idx + 1}"
                t_dir = self.get_task_dir(tid)
                os.makedirs(t_dir, exist_ok=True)
                total_steps = 5 if payload.mode == "sub_only" else (6 if payload.mode == "sub_video" else 10)
                yt_id = _1_ytdlp.get_youtube_video_id(url)
                initial_thumb = f"https://i.ytimg.com/vi/{yt_id}/hqdefault.jpg" if yt_id else ""
                new_task = {
                    "id": tid,
                    "url": url,
                    "title": f"正在解析视频标题... ({url})",
                    "thumbnail": initial_thumb,
                    "uploader": "",
                    "duration_str": "",
                    "source_lang": payload.source_lang.strip(),
                    "target_lang": payload.target_lang.strip(),
                    "mode": payload.mode,
                    "resolution": payload.resolution or "1080",
                    "prefer_official_sub": payload.prefer_official_sub,
                    "tts_method": payload.tts_method.strip(),
                    "male_voice": payload.male_voice.strip(),
                    "female_voice": payload.female_voice.strip(),
                    "pause_before_dub": bool(payload.pause_before_dub),
                    "speaker_confirmed": False,
                    "status": "probing",
                    "current_step": "🔍 正在嗅探视频标题与封面...",
                    "step_index": 0,
                    "total_steps": total_steps,
                    "progress": 0.0,
                    "error_msg": "",
                    "created_at": self._now_str(),
                    "updated_at": self._now_str(),
                }
                self.tasks.append(new_task)
                added_ids.append(tid)
            self._save()

        threading.Thread(target=self._probe_tasks_worker, args=(added_ids,), daemon=True).start()
        return added_ids

    def _probe_tasks_worker(self, task_ids: List[str]):
        import requests

        for tid in task_ids:
            t = self.get_task(tid)
            if not t:
                continue
            url = t["url"]
            task_dir = self.get_task_dir(tid)
            os.makedirs(task_dir, exist_ok=True)

            title = ""
            thumb_url = t.get("thumbnail", "")
            uploader = ""
            duration_str = ""

            if os.path.isfile(os.path.expanduser(url)):
                local_p = os.path.abspath(os.path.expanduser(url))
                title = os.path.basename(local_p)
                with self.lock:
                    curr = self.get_task(tid)
                    if curr:
                        curr["title"] = title
                        if curr["status"] == "probing":
                            curr["status"] = "pending"
                            curr["current_step"] = "排队等待中（本地文件）"
                        curr["updated_at"] = self._now_str()
                        self._save()
                continue

            # 1. Fast path: YouTube oEmbed API (< 300ms)
            yt_id = _1_ytdlp.get_youtube_video_id(url)
            proxies = None
            try:
                proxy_cfg = load_key("proxy")
                if proxy_cfg:
                    proxies = {"http": str(proxy_cfg), "https": str(proxy_cfg)}
            except Exception:
                pass

            if yt_id:
                try:
                    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={yt_id}&format=json"
                    resp = requests.get(oembed_url, timeout=5, proxies=proxies)
                    if resp.status_code == 200:
                        odata = resp.json()
                        title = odata.get("title") or ""
                        uploader = odata.get("author_name") or ""
                        thumb_url = odata.get("thumbnail_url") or f"https://i.ytimg.com/vi/{yt_id}/hqdefault.jpg"
                except Exception:
                    pass

            # 2. Full yt-dlp metadata extraction for duration or non-YouTube URLs
            if not title or not duration_str:
                try:
                    from yt_dlp import YoutubeDL
                    ydl_opts = {
                        "skip_download": True,
                        "noplaylist": True,
                        "nocheckcertificate": True,
                        "quiet": True,
                        "no_warnings": True,
                    }
                    if proxies and proxies.get("http"):
                        ydl_opts["proxy"] = proxies["http"]
                    _1_ytdlp._add_cookies_options(ydl_opts)
                    with YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        if info:
                            title = info.get("title") or title
                            uploader = info.get("uploader") or info.get("channel") or uploader
                            thumb_url = info.get("thumbnail") or thumb_url
                            dur = info.get("duration")
                            if dur:
                                m, s = divmod(int(dur), 60)
                                h, m = divmod(m, 60)
                                duration_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
                except Exception:
                    pass

            # Cache thumbnail image locally so frontend doesn't need proxy to view cover
            if thumb_url:
                try:
                    r_img = requests.get(thumb_url, timeout=6, proxies=proxies)
                    if r_img.status_code == 200 and len(r_img.content) > 500:
                        cover_file = os.path.join(task_dir, "video_cover.jpg")
                        with open(cover_file, "wb") as f:
                            f.write(r_img.content)
                        thumb_url = f"/api/batch/media/{tid}/video_cover.jpg"
                except Exception:
                    pass

            if title:
                try:
                    with open(os.path.join(task_dir, "video_title.txt"), "w", encoding="utf-8") as f:
                        f.write(title)
                except Exception:
                    pass

            with self.lock:
                curr = self.get_task(tid)
                if curr:
                    if title:
                        curr["title"] = title
                    elif curr["title"].startswith("正在解析视频标题"):
                        curr["title"] = url
                    if thumb_url:
                        curr["thumbnail"] = thumb_url
                    if uploader:
                        curr["uploader"] = uploader
                    if duration_str:
                        curr["duration_str"] = duration_str
                    if curr["status"] == "probing":
                        curr["status"] = "pending"
                        curr["current_step"] = "排队等待中"
                    curr["updated_at"] = self._now_str()
                    self._save()

    def update_task(self, task_id: str, updates: Dict[str, Any]):
        with self.lock:
            t = self.get_task(task_id)
            if not t:
                raise HTTPException(status_code=404, detail="找不到该任务")
            allowed_keys = [
                "title",
                "url",
                "source_lang",
                "target_lang",
                "mode",
                "resolution",
                "prefer_official_sub",
                "tts_method",
                "male_voice",
                "female_voice",
                "pause_before_dub",
            ]
            for k in allowed_keys:
                if k in updates and updates[k] is not None:
                    t[k] = updates[k]
            if "mode" in updates:
                t["total_steps"] = 5 if t["mode"] == "sub_only" else (6 if t["mode"] == "sub_video" else 10)
            t["updated_at"] = self._now_str()
            self._save()
            return t

    def delete_task(self, task_id: str):
        with self.lock:
            t = self.get_task(task_id)
            if not t:
                raise HTTPException(status_code=404, detail="找不到该任务")
            if t["status"] == "running" or self.active_task_id == task_id:
                raise HTTPException(status_code=409, detail="该任务正在执行中，请先停止后再删除")
            if self.studio_loaded_task_id == task_id:
                self.studio_loaded_task_id = None
            self.tasks = [x for x in self.tasks if x["id"] != task_id]
            t_dir = self.get_task_dir(task_id)
            if os.path.exists(t_dir):
                shutil.rmtree(t_dir, ignore_errors=True)
            self._save()

    def move_task(self, task_id: str, direction: str):
        with self.lock:
            idx = next((i for i, x in enumerate(self.tasks) if x["id"] == task_id), -1)
            if idx < 0:
                raise HTTPException(status_code=404, detail="找不到该任务")
            if direction == "up" and idx > 0:
                self.tasks[idx - 1], self.tasks[idx] = self.tasks[idx], self.tasks[idx - 1]
            elif direction == "down" and idx < len(self.tasks) - 1:
                self.tasks[idx + 1], self.tasks[idx] = self.tasks[idx], self.tasks[idx + 1]
            elif direction == "top" and idx > 0:
                item = self.tasks.pop(idx)
                self.tasks.insert(0, item)
            self._save()

    def reset_task_checkpoint(self, task_id: str):
        with self.lock:
            t = self.get_task(task_id)
            if not t:
                raise HTTPException(status_code=404, detail="找不到该任务")
            if t["status"] == "running":
                raise HTTPException(status_code=409, detail="任务正在运行中，无法重置")
            t_dir = self.get_task_dir(task_id)
            # Keep only cover & title metadata if present
            if os.path.exists(t_dir):
                for item in os.listdir(t_dir):
                    if item in ("video_cover.jpg", "video_title.txt"):
                        continue
                    p = os.path.join(t_dir, item)
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            if self.studio_loaded_task_id == task_id:
                _wipe_output_workspace()
                self.studio_loaded_task_id = None
            t["status"] = "pending"
            t["progress"] = 0.0
            t["step_index"] = 0
            t["current_step"] = "已清空缓存，等待重新执行"
            t["error_msg"] = ""
            t["speaker_confirmed"] = False
            t["updated_at"] = self._now_str()
            self._save()

    def clear_tasks(self, target: str = "completed"):
        with self.lock:
            kept = []
            for t in self.tasks:
                if t["status"] == "running" or t["id"] == self.active_task_id:
                    kept.append(t)
                    continue
                if target == "completed" and t["status"] != "completed":
                    kept.append(t)
                    continue
                # Delete task directory
                t_dir = self.get_task_dir(t["id"])
                if os.path.exists(t_dir):
                    shutil.rmtree(t_dir, ignore_errors=True)
                if self.studio_loaded_task_id == t["id"]:
                    self.studio_loaded_task_id = None
            self.tasks = kept
            self._save()

    def _apply_task_config_overrides(self, t: Dict[str, Any]) -> Dict[str, Any]:
        """Snapshot global config keys and apply per-task overrides."""
        keys_to_save = [
            "whisper.language",
            "target_language",
            "tts_method",
            "doubao_tts.male_voice",
            "doubao_tts.female_voice",
            "doubao_tts.voice",
            "doubao_tts.style_prompt",
            "edge_tts.male_voice",
            "edge_tts.female_voice",
            "edge_tts.voice",
            "elevenlabs_tts.male_voice",
            "elevenlabs_tts.female_voice",
            "elevenlabs_tts.voice",
        ]
        snapshot = {}
        for k in keys_to_save:
            try:
                snapshot[k] = load_key(k)
            except Exception:
                snapshot[k] = None

        if t.get("source_lang"):
            update_key("whisper.language", t["source_lang"])
        if t.get("target_lang"):
            update_key("target_language", t["target_lang"])

        tts_m = t.get("tts_method") or snapshot.get("tts_method") or "doubao_tts"
        if t.get("tts_method"):
            update_key("tts_method", t["tts_method"])

        if t.get("male_voice"):
            if tts_m == "edge_tts":
                update_key("edge_tts.male_voice", t["male_voice"])
            elif tts_m == "elevenlabs_tts":
                update_key("elevenlabs_tts.male_voice", t["male_voice"])
            else:
                update_key("doubao_tts.male_voice", t["male_voice"])

        if t.get("female_voice"):
            if tts_m == "edge_tts":
                update_key("edge_tts.female_voice", t["female_voice"])
                update_key("edge_tts.voice", t["female_voice"])
            elif tts_m == "elevenlabs_tts":
                update_key("elevenlabs_tts.female_voice", t["female_voice"])
                update_key("elevenlabs_tts.voice", t["female_voice"])
            else:
                update_key("doubao_tts.female_voice", t["female_voice"])
                update_key("doubao_tts.voice", t["female_voice"])

        if t.get("style_prompt") is not None and str(t.get("style_prompt")).strip():
            update_key("doubao_tts.style_prompt", str(t["style_prompt"]).strip())

        return snapshot

    def _restore_config_snapshot(self, snapshot: Dict[str, Any]):
        for k, val in snapshot.items():
            if val is not None:
                try:
                    update_key(k, val)
                except Exception:
                    pass

    def _update_running_task_progress(self, t: Dict[str, Any], step_idx: int, total_steps: int, step_title: str, sub_step: str, pct: float):
        with self.lock:
            t["step_index"] = step_idx
            t["total_steps"] = total_steps
            t["current_step"] = f"[{step_idx}/{total_steps}] {step_title}"
            t["progress"] = round(pct, 1)
            t["updated_at"] = self._now_str()
            self._save()
        STATE.step_index = step_idx
        STATE.total_steps = total_steps
        STATE.step_name = f"📦 [{t.get('title') or t['url']}] {step_title}"
        STATE.sub_step = sub_step
        STATE.percent = pct

    def _execute_single_task(self, t: Dict[str, Any]):
        tid = t["id"]
        task_dir = self.get_task_dir(tid)
        os.makedirs(task_dir, exist_ok=True)

        # Swap task workspace into output/
        with self.lock:
            self._stash_single_workspace_if_needed()
            _copy_workspace_files(task_dir, "output", clear_dst=True, preserve_cookies=True)
            self.active_task_id = tid
            self.studio_loaded_task_id = tid
            t["status"] = "running"
            t["error_msg"] = ""
            t["updated_at"] = self._now_str()
            self._save()

        mode = t.get("mode", "dub_video")
        total_steps = 5 if mode == "sub_only" else (6 if mode == "sub_video" else 10)
        cfg_snapshot = self._apply_task_config_overrides(t)

        try:
            url = t["url"].strip()
            is_local_file = os.path.isfile(os.path.expanduser(url))
            resolution = str(t.get("resolution") or load_key("ytb_resolution") or "1080")

            # Step 1: Download video (or import local file) & probe official subtitles
            orig_video = None
            try:
                orig_video = _1_ytdlp.find_video_files()
            except Exception:
                orig_video = None

            if not orig_video or not os.path.exists(orig_video):
                if is_local_file:
                    self._update_running_task_progress(
                        t, 1, total_steps, "导入本地媒体文件", "正在复制本地视频/音频到工作区...", 5.0
                    )
                    imported_name = _import_local_media_to_output(os.path.expanduser(url), wipe_first=False)
                    t["title"] = imported_name
                else:
                    self._update_running_task_progress(
                        t, 1, total_steps, f"下载视频 ({resolution}p)", "正在调用 yt-dlp 下载视频流与封面...", 5.0
                    )
                    _1_ytdlp.download_video_ytdlp(url, resolution=resolution)
                    save_youtube_url(url)
                    # Sync title & cover if fetched
                    if os.path.exists("output/video_title.txt"):
                        with open("output/video_title.txt", "r", encoding="utf-8") as f:
                            fetched_title = f.read().strip()
                            if fetched_title:
                                t["title"] = fetched_title
                _copy_workspace_files("output", task_dir, clear_dst=True)

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            if (not is_local_file) and t.get("prefer_official_sub", True) and not os.path.exists(_2_CLEANED_CHUNKS):
                self._update_running_task_progress(
                    t, 1, total_steps, "检测官方字幕", "正在探测原视频 CC 字幕或自动字幕轨道...", 12.0
                )
                sub_path, sub_type = _1_ytdlp.download_subtitle_ytdlp(url)
                if sub_path:
                    lang = load_key("whisper.language")
                    if lang == "auto":
                        lang = "en"
                    use_subtitle_file(sub_path, lang)
                    _copy_workspace_files("output", task_dir, clear_dst=True)

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            # Step 2: WhisperX + Pyannote Diarization (if not bypassed by official subtitle)
            if not os.path.exists(_2_CLEANED_CHUNKS):
                self._update_running_task_progress(
                    t, 2, total_steps, "WhisperX 语音转录与声纹分离", "正在执行语音识别与多说话人分离...", 20.0
                )
                _2_asr.transcribe()
                _2_5_diarize.diarize_main()
                _copy_workspace_files("output", task_dir, clear_dst=True)

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            # Step 3: NLP + LLM Semantic Sentence Splitting
            if not os.path.exists(_3_2_SPLIT_BY_MEANING):
                self._update_running_task_progress(
                    t, 3, total_steps, "NLP + LLM 语义分句", "正在执行 SpaCy 语法切分与大模型断句...", 35.0
                )
                _3_1_split_nlp.split_by_spacy()
                _3_2_split_meaning.split_sentences_by_meaning()
                _copy_workspace_files("output", task_dir, clear_dst=True)

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            # Step 4: Summarize & 3-Step Translation
            if not os.path.exists(_4_2_TRANSLATION):
                self._update_running_task_progress(
                    t, 4, total_steps, "术语提取与三步法翻译", "正在并发执行「直译 -> 反思 -> 意译」...", 50.0
                )
                _4_1_summarize.get_summary()
                _4_2_translate.translate_all()
                _copy_workspace_files("output", task_dir, clear_dst=True)

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            # Step 5: Subtitle Splitting & Timestamp Alignment
            if not os.path.exists(_5_SPLIT_SUB):
                self._update_running_task_progress(
                    t, 5, total_steps, "长字幕切分与时间轴对齐", "正在自适应切分长句并对齐双语时间戳...", 62.0
                )
                _5_split_sub.split_for_sub_main()
            if not os.path.exists(SRC_SRT):
                self._update_running_task_progress(
                    t, 5, total_steps, "生成双语 SRT 字幕", "正在生成并平滑无重叠字幕文件...", 68.0
                )
                _6_gen_sub.align_timestamp_main()
                _copy_workspace_files("output", task_dir, clear_dst=True)

            if mode == "sub_only":
                with self.lock:
                    t["status"] = "completed"
                    t["progress"] = 100.0
                    t["current_step"] = "✅ 双语 SRT 字幕生成完成"
                    t["updated_at"] = self._now_str()
                    self._save()
                return

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            # Step 6: Burn Subtitles into Video
            should_burn = (mode == "sub_video") or bool(load_key("burn_subtitles"))
            if should_burn and not os.path.exists(SUB_VIDEO):
                self._update_running_task_progress(
                    t, 6, total_steps, "烧录双语字幕到视频", "正在使用 FFmpeg 压制双语字幕视频 (output_sub.mp4)...", 75.0
                )
                _7_sub_into_vid.merge_subtitles_to_video()
                _copy_workspace_files("output", task_dir, clear_dst=True)

            if mode == "sub_video":
                with self.lock:
                    t["status"] = "completed"
                    t["progress"] = 100.0
                    t["current_step"] = "✅ 字幕压制视频合成完成"
                    t["updated_at"] = self._now_str()
                    self._save()
                return

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            # Step 7: Dubbing - Audio extraction & Speaker gender/chunking
            if not os.path.exists(_RAW_AUDIO_FILE):
                self._update_running_task_progress(
                    t, 7, total_steps, "提取原片基础音轨", "正在从原视频提取 raw.mp3 供声纹与性别分析...", 78.0
                )
                v_file = _1_ytdlp.find_video_files()
                if v_file and os.path.exists(v_file):
                    convert_video_to_audio(v_file)

            if not os.path.exists(_8_1_AUDIO_TASK):
                self._update_running_task_progress(
                    t, 7, total_steps, "说话人性别识别与配音分块", "正在识别角色性别并计算跨角色切分边界...", 82.0
                )
                _8_1_audio_task.gen_audio_task_main()
                _8_2_dub_chunks.gen_dub_chunks()
                _copy_workspace_files("output", task_dir, clear_dst=True)
            else:
                sync_res = _8_1_audio_task.sync_subtitles_to_audio_files()
                df_existing = pd.read_excel(_8_1_AUDIO_TASK)
                has_lines = (
                    "lines" in df_existing.columns
                    and len(df_existing) > 0
                    and df_existing["lines"].notna().all()
                )
                if not has_lines:
                    _8_2_dub_chunks.gen_dub_chunks()
                    _copy_workspace_files("output", task_dir, clear_dst=True)
                elif (
                    sync_res.get("updated_task_rows", 0) > 0
                    or "cut_off" not in df_existing.columns
                    or "gap" not in df_existing.columns
                    or "para_id" not in df_existing.columns
                ):
                    df_existing = _8_2_dub_chunks.analyze_subtitle_timing_and_speed(df_existing)
                    df_existing = _8_2_dub_chunks.process_cutoffs(df_existing)
                    df_existing.to_excel(_8_1_AUDIO_TASK, index=False)
                    _copy_workspace_files("output", task_dir, clear_dst=True)

            # Check per-task pause_before_dub
            if t.get("pause_before_dub") and not t.get("speaker_confirmed"):
                _copy_workspace_files("output", task_dir, clear_dst=True)
                with self.lock:
                    t["status"] = "waiting_review"
                    t["progress"] = 84.0
                    t["current_step"] = "⏸️ 待确认说话人音色（已挂起，可载入工作台确认）"
                    t["updated_at"] = self._now_str()
                    self._save()
                STATE.toast = f"⏸️ 任务《{t.get('title') or t['id']}》已挂起等待确认说话人，队列继续执行下一任务"
                return

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            # Step 8: Reference Audio Extraction
            if not load_key("skip_refer"):
                self._update_running_task_progress(
                    t, 8, total_steps, "提取说话人参考音频", "正在为各角色抽取清晰参考音频切片...", 87.0
                )
                _9_refer_audio.extract_refer_audio_main()
                _copy_workspace_files("output", task_dir, clear_dst=True)

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            # Step 9: Multi-speaker TTS Generation & Timeline Merge
            if not os.path.exists(DUB_VIDEO):
                self._update_running_task_progress(
                    t, 9, total_steps, "并发合成多角色配音", "正在调用 TTS 引擎并发合成并对齐语音轨道...", 92.0
                )
                _10_gen_audio.gen_audio()
                _11_merge_audio.merge_full_audio()
                _copy_workspace_files("output", task_dir, clear_dst=True)

            if self.stop_requested or STATE.stop_requested:
                raise InterruptedError("任务已被手动暂停")

            # Step 10: Final Dubbed Video Merge
            if not os.path.exists(DUB_VIDEO):
                self._update_running_task_progress(
                    t, 10, total_steps, "音画混流合成配音成片", "正在调用 FFmpeg 混流配音、伴奏与视频画面...", 97.0
                )
                _12_dub_to_vid.merge_video_audio()
                _copy_workspace_files("output", task_dir, clear_dst=True)

            with self.lock:
                t["status"] = "completed"
                t["progress"] = 100.0
                t["current_step"] = "🎉 字幕与多角色配音成片全部完成"
                t["error_msg"] = ""
                t["updated_at"] = self._now_str()
                self._save()

        except (InterruptedError, TaskCancelledError):
            _copy_workspace_files("output", task_dir, clear_dst=True)
            with self.lock:
                prev_step = str(t.get("current_step") or "可随时断点续跑")
                if prev_step.startswith("⏸️ 已暂停 (") and prev_step.endswith(")"):
                    prev_step = prev_step[len("⏸️ 已暂停 (") : -1]
                t["status"] = "paused"
                t["current_step"] = f"⏸️ 已暂停 ({prev_step})"
                t["error_msg"] = ""
                t["updated_at"] = self._now_str()
                self._save()
        except Exception as e:
            traceback.print_exc()
            _copy_workspace_files("output", task_dir, clear_dst=True)
            with self.lock:
                t["status"] = "error"
                t["error_msg"] = str(e)
                t["current_step"] = f"❌ 异常中断 ({t.get('current_step', '')})"
                t["updated_at"] = self._now_str()
                self._save()
        finally:
            self._restore_config_snapshot(cfg_snapshot)
            with self.lock:
                if self.active_task_id == tid:
                    self.active_task_id = None
                self._save()

    def start_queue(self, single_task_id: Optional[str] = None, from_scratch: bool = False):
        if STATE.running or self.running:
            raise HTTPException(status_code=409, detail="当前已有任务或队列正在运行中")

        if single_task_id and from_scratch:
            self.reset_task_checkpoint(single_task_id)

        def _queue_worker():
            reset_task_cancel()
            with self.lock:
                self.running = True
                self.pause_after_current = False
                self.stop_requested = False
                self._bump_rev()

            with STATE.lock:
                STATE.start_task(
                    stage="batch",
                    total_steps=10,
                    step_name="📦 正在启动批量队列处理引擎...",
                )

            try:
                if single_task_id:
                    t = self.get_task(single_task_id)
                    if t:
                        if t["status"] == "waiting_review":
                            t["speaker_confirmed"] = True
                        self._execute_single_task(t)
                else:
                    while not self.stop_requested and not STATE.stop_requested:
                        # Pick next runnable task in current queue order
                        next_task = None
                        with self.lock:
                            for candidate in self.tasks:
                                if candidate["status"] in ("pending", "paused", "error", "probing"):
                                    next_task = candidate
                                    break
                                if candidate["status"] == "waiting_review" and candidate.get("speaker_confirmed"):
                                    next_task = candidate
                                    break
                        if not next_task:
                            break

                        self._execute_single_task(next_task)

                        if self.stop_requested or STATE.stop_requested:
                            break
                        if self.pause_after_current:
                            STATE.toast = "⏸️ 当前视频已处理完毕，批量队列已按计划暂停"
                            break

                if not self.stop_requested and not STATE.stop_requested and not self.pause_after_current:
                    STATE.percent = 100.0
                    STATE.toast = "🎉 批量队列当前可执行任务已全部处理完毕！"
            finally:
                with self.lock:
                    self.running = False
                    self.stop_requested = False
                    self.pause_after_current = False
                    self.active_task_id = None
                    self._bump_rev()
                STATE.finish_task()

        threading.Thread(target=_queue_worker, daemon=True).start()


BATCH_MANAGER = BatchQueueManager()


class BatchAddTasksPayload(BaseModel):
    urls: List[str]
    source_lang: str = ""
    target_lang: str = ""
    mode: str = "dub_video"  # "sub_only" | "sub_video" | "dub_video"
    resolution: str = "1080"
    prefer_official_sub: bool = True
    tts_method: str = ""
    male_voice: str = ""
    female_voice: str = ""
    pause_before_dub: bool = False


class BatchUpdateTaskPayload(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    mode: Optional[str] = None
    resolution: Optional[str] = None
    prefer_official_sub: Optional[bool] = None
    tts_method: Optional[str] = None
    male_voice: Optional[str] = None
    female_voice: Optional[str] = None
    pause_before_dub: Optional[bool] = None


class BatchMovePayload(BaseModel):
    direction: str = "up"  # "up" | "down" | "top"


class BatchRunSinglePayload(BaseModel):
    from_scratch: bool = False


class BatchClearPayload(BaseModel):
    target: str = "completed"  # "completed" | "all"


@app.get("/api/batch/state")
def get_batch_state():
    return BATCH_MANAGER.get_state_payload()


@app.post("/api/batch/tasks")
def add_batch_tasks(payload: BatchAddTasksPayload):
    BATCH_MANAGER.add_tasks(payload)
    return BATCH_MANAGER.get_state_payload()


@app.put("/api/batch/tasks/{task_id}")
def update_batch_task(task_id: str, payload: BatchUpdateTaskPayload):
    BATCH_MANAGER.update_task(task_id, payload.model_dump(exclude_none=True))
    return BATCH_MANAGER.get_state_payload()


@app.delete("/api/batch/tasks/{task_id}")
def delete_batch_task(task_id: str):
    BATCH_MANAGER.delete_task(task_id)
    return BATCH_MANAGER.get_state_payload()


@app.post("/api/batch/tasks/{task_id}/move")
def move_batch_task(task_id: str, payload: BatchMovePayload):
    BATCH_MANAGER.move_task(task_id, payload.direction)
    return BATCH_MANAGER.get_state_payload()


@app.post("/api/batch/tasks/{task_id}/reset")
def reset_batch_task(task_id: str):
    BATCH_MANAGER.reset_task_checkpoint(task_id)
    return BATCH_MANAGER.get_state_payload()


@app.post("/api/batch/tasks/{task_id}/run")
def run_single_batch_task(task_id: str, payload: BatchRunSinglePayload):
    BATCH_MANAGER.start_queue(single_task_id=task_id, from_scratch=payload.from_scratch)
    return BATCH_MANAGER.get_state_payload()


@app.post("/api/batch/tasks/{task_id}/load-to-studio")
def load_batch_task_to_studio(task_id: str):
    if STATE.running or BATCH_MANAGER.running:
        raise HTTPException(status_code=409, detail="当前有任务正在运行，请先停止后再切换工作台任务")
    t = BATCH_MANAGER.load_task_to_studio(task_id)
    return {"status": "ok", "task": t}


@app.post("/api/batch/tasks/{task_id}/sync-from-studio")
def sync_batch_task_from_studio(task_id: str, confirm_speaker: bool = False):
    with BATCH_MANAGER.lock:
        BATCH_MANAGER.studio_loaded_task_id = task_id
        BATCH_MANAGER.sync_from_studio_if_loaded(speaker_confirmed=True if confirm_speaker else None)
    return BATCH_MANAGER.get_state_payload()


@app.post("/api/batch/start")
def start_batch_queue():
    BATCH_MANAGER.start_queue()
    return BATCH_MANAGER.get_state_payload()


@app.post("/api/batch/pause")
def toggle_pause_after_current():
    with BATCH_MANAGER.lock:
        BATCH_MANAGER.pause_after_current = not BATCH_MANAGER.pause_after_current
        BATCH_MANAGER._bump_rev()
    return BATCH_MANAGER.get_state_payload()


@app.post("/api/batch/stop")
def stop_batch_queue():
    stopped = _request_and_wait_stop(timeout_sec=8.0)
    payload = BATCH_MANAGER.get_state_payload()
    payload["stopped"] = stopped
    return payload


@app.post("/api/batch/clear")
def clear_batch_tasks(payload: BatchClearPayload):
    BATCH_MANAGER.clear_tasks(payload.target)
    return BATCH_MANAGER.get_state_payload()


@app.post("/api/batch/restore-single-stash")
def restore_single_workspace_stash():
    if STATE.running or BATCH_MANAGER.running:
        raise HTTPException(status_code=409, detail="当前有任务正在运行，无法恢复工作区")
    ok = BATCH_MANAGER.restore_single_stash()
    if not ok:
        raise HTTPException(status_code=404, detail="没有可恢复的单视频工作台暂存备份")
    return {"status": "ok"}


@app.get("/api/batch/tasks/{task_id}/download-subtitles")
def download_batch_task_subtitles(task_id: str):
    t = BATCH_MANAGER.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="找不到该任务")
    task_dir = BATCH_MANAGER.get_task_dir(task_id)
    inspect_dir = "output" if (task_id == BATCH_MANAGER.active_task_id or task_id == BATCH_MANAGER.studio_loaded_task_id) and os.path.exists("output") else task_dir
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for d in [inspect_dir, task_dir]:
            if os.path.exists(d):
                for fname in os.listdir(d):
                    if fname.endswith(".srt") and fname not in zf.namelist():
                        fpath = os.path.join(d, fname)
                        with open(fpath, "rb") as f:
                            zf.writestr(fname, f.read())
    zip_buffer.seek(0)
    safe_name = re.sub(r"[^\w\-]", "_", t.get("title") or task_id)[:40]
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_subtitles.zip"'},
    )


@app.get("/api/batch/media/{task_id}/{file_path:path}")
def stream_batch_task_media(task_id: str, file_path: str, request: Request):
    task_dir = os.path.abspath(BATCH_MANAGER.get_task_dir(task_id))
    target = os.path.abspath(os.path.join(task_dir, file_path))
    if not target.startswith(task_dir):
        raise HTTPException(status_code=403, detail="非法路径访问")

    # Fallback to output/ if task is currently active in studio/runner
    if not os.path.exists(target) and task_id in (BATCH_MANAGER.active_task_id, BATCH_MANAGER.studio_loaded_task_id):
        out_base = os.path.abspath("output")
        alt_target = os.path.abspath(os.path.join(out_base, file_path))
        if alt_target.startswith(out_base) and os.path.exists(alt_target):
            target = alt_target

    if not os.path.exists(target) or not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="文件不存在")

    content_type, _ = mimetypes.guess_type(target)
    if target.endswith(".vtt"):
        content_type = "text/vtt; charset=utf-8"
    elif target.endswith(".srt"):
        content_type = "text/plain; charset=utf-8"
    content_type = content_type or "application/octet-stream"

    file_size = os.path.getsize(target)
    range_header = request.headers.get("range")
    if range_header:
        match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = max(0, end - start + 1)

            def _iter_range():
                with open(target, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            }
            return StreamingResponse(_iter_range(), status_code=206, media_type=content_type, headers=headers)

    return FileResponse(target, media_type=content_type, headers={"Accept-Ranges": "bytes"})


# =====================================================================
# Media Streaming Endpoint with HTTP Range Support (Video / Audio Scrubbing)
# =====================================================================
@app.get("/api/media/{file_path:path}")
def stream_output_media(file_path: str, request: Request):
    safe_base = os.path.abspath("output")
    target = os.path.abspath(os.path.join(safe_base, file_path))
    if os.path.basename(target) == "youtube_cookies.txt":
        raise HTTPException(status_code=403, detail="禁止直接读取敏感凭证文件")
    if not target.startswith(safe_base) or not os.path.exists(target) or not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="文件不存在")

    content_type, _ = mimetypes.guess_type(target)
    if target.endswith(".vtt"):
        content_type = "text/vtt; charset=utf-8"
    elif target.endswith(".srt"):
        content_type = "text/plain; charset=utf-8"
    content_type = content_type or "application/octet-stream"

    file_size = os.path.getsize(target)
    range_header = request.headers.get("range")

    if range_header:
        match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = max(0, end - start + 1)

            def _iter_range():
                with open(target, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            }
            return StreamingResponse(_iter_range(), status_code=206, media_type=content_type, headers=headers)

    return FileResponse(target, media_type=content_type, headers={"Accept-Ranges": "bytes"})


# =====================================================================
# WebSocket Live Console Log Stream
# =====================================================================
@app.websocket("/ws/logs")
async def websocket_logs(ws: WebSocket):
    import asyncio
    await ws.accept()
    try:
        while True:
            logs_Snapshot = list(LOG_BUFFER)
            await ws.send_json(
                {
                    "logs": logs_Snapshot[-80:],
                    "task": STATE.to_dict(),
                    "batch_rev": BATCH_MANAGER.revision,
                }
            )
            if STATE.toast:
                STATE.toast = None
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# =====================================================================
# Agent / MCP Deep Collaboration & One-Shot Endpoints
# =====================================================================
class LocalFileImportPayload(BaseModel):
    file_path: str


def _import_local_media_to_output(file_path: str, wipe_first: bool = True) -> str:
    abs_src = os.path.abspath(os.path.expanduser(file_path.strip()))
    if not os.path.isfile(abs_src):
        raise HTTPException(status_code=404, detail=f"找不到本地文件: {abs_src}")

    if wipe_first:
        if "BATCH_MANAGER" in globals():
            BATCH_MANAGER.detach_studio_task()
        _wipe_output_workspace()

    os.makedirs("output", exist_ok=True)
    raw_name = os.path.basename(abs_src)
    ext = raw_name.rsplit(".", 1)[-1].lower() if "." in raw_name else "mp4"
    base_name = re.sub(r"[^\w\-_\.]", "_", raw_name.rsplit(".", 1)[0]) or "imported_media"
    clean_filename = f"{base_name}.{ext}"
    save_path = os.path.join("output", clean_filename)
    if os.path.abspath(save_path) != abs_src:
        shutil.copy2(abs_src, save_path)

    with open(os.path.join("output", "video_title.txt"), "w", encoding="utf-8") as f:
        f.write(raw_name)

    if ext in ["mp3", "wav", "m4a", "flac"]:
        vcodec = get_ffmpeg_video_encoder() or "libx264"
        mp4_path = os.path.join("output", f"{base_name}.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1280x720:r=1",
                "-i",
                save_path,
                "-c:v",
                vcodec,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-shortest",
                mp4_path,
            ],
            check=False,
        )
        if os.path.exists(mp4_path):
            if os.path.exists(save_path) and os.path.abspath(save_path) != os.path.abspath(mp4_path):
                os.remove(save_path)
            clean_filename = os.path.basename(mp4_path)
    return clean_filename


@app.post("/api/video/import-local")
def import_local_video_by_path(payload: LocalFileImportPayload):
    if STATE.running:
        raise HTTPException(status_code=409, detail="当前已有任务正在运行")
    fname = _import_local_media_to_output(payload.file_path, wipe_first=True)
    return {"status": "ok", "file": fname}


@app.post("/api/subtitle/import-local")
def import_local_subtitle_by_path(payload: LocalFileImportPayload):
    abs_src = os.path.abspath(os.path.expanduser(payload.file_path.strip()))
    if not os.path.isfile(abs_src):
        raise HTTPException(status_code=404, detail=f"找不到本地字幕文件: {abs_src}")
    os.makedirs("output", exist_ok=True)
    ext = os.path.basename(abs_src).rsplit(".", 1)[-1].lower() if "." in os.path.basename(abs_src) else "srt"
    save_path = f"output/uploaded_subtitle.{ext}"
    if os.path.abspath(save_path) != abs_src:
        shutil.copy2(abs_src, save_path)
    lang = load_key("whisper.language")
    if lang == "auto":
        lang = "en"
    ok = use_subtitle_file(save_path, lang)
    if not ok:
        raise HTTPException(status_code=400, detail="字幕文件解析失败，请检查格式")
    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.sync_from_studio_if_loaded()
    return {"status": "ok", "file": os.path.basename(abs_src)}


class OneShotProcessPayload(BaseModel):
    url: Optional[str] = None
    local_file_path: Optional[str] = None
    mode: str = "dub_video"  # "dub_video" | "sub_video" | "sub_only"
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    resolution: str = "1080"
    prefer_official_sub: bool = True
    tts_method: Optional[str] = None
    male_voice: Optional[str] = None
    female_voice: Optional[str] = None
    pause_before_dub: bool = False


@app.post("/api/pipeline/one-shot")
def start_one_shot_pipeline(payload: OneShotProcessPayload):
    if STATE.running:
        raise HTTPException(status_code=409, detail="当前已有任务正在运行")

    if not payload.url and not payload.local_file_path:
        try:
            existing_vid = _1_ytdlp.find_video_files()
        except Exception:
            existing_vid = None
        if not existing_vid or not os.path.exists(existing_vid):
            raise HTTPException(status_code=400, detail="请提供视频 url 或 local_file_path，或先在工作台加载视频")

    def _one_shot_worker():
        mode = payload.mode if payload.mode in ("dub_video", "sub_video", "sub_only") else "dub_video"
        total_steps = 5 if mode == "sub_only" else (6 if mode == "sub_video" else 10)

        with STATE.lock:
            STATE.start_task(
                stage="one_shot",
                total_steps=total_steps,
                step_name="1. 正在初始化一键全流程处理任务...",
            )

        try:
            # Apply optional config overrides
            if payload.source_lang:
                update_key("whisper.language", payload.source_lang.strip())
            if payload.target_lang:
                update_key("target_language", payload.target_lang.strip())
            if payload.tts_method:
                update_key("tts_method", payload.tts_method.strip())
            tts_m = (payload.tts_method or load_key("tts_method") or "doubao_tts").strip()
            if payload.male_voice:
                mv = payload.male_voice.strip()
                if tts_m == "edge_tts":
                    update_key("edge_tts.male_voice", mv)
                elif tts_m == "elevenlabs_tts":
                    update_key("elevenlabs_tts.male_voice", mv)
                else:
                    update_key("doubao_tts.male_voice", mv)
            if payload.female_voice:
                fv = payload.female_voice.strip()
                if tts_m == "edge_tts":
                    update_key("edge_tts.female_voice", fv)
                    update_key("edge_tts.voice", fv)
                elif tts_m == "elevenlabs_tts":
                    update_key("elevenlabs_tts.female_voice", fv)
                    update_key("elevenlabs_tts.voice", fv)
                else:
                    update_key("doubao_tts.female_voice", fv)
                    update_key("doubao_tts.voice", fv)
            update_key("pause_before_dub", bool(payload.pause_before_dub))

            # Stage A: Media Acquisition
            if payload.local_file_path:
                STATE.step_index = 1
                STATE.percent = 5.0
                STATE.step_name = "1. 正在导入本地媒体文件..."
                STATE.sub_step = f"正在导入 {os.path.basename(payload.local_file_path)}..."
                _import_local_media_to_output(payload.local_file_path, wipe_first=True)
            elif payload.url:
                url = payload.url.strip()
                if os.path.isfile(os.path.expanduser(url)):
                    STATE.step_index = 1
                    STATE.percent = 5.0
                    STATE.step_name = "1. 正在导入本地媒体文件..."
                    _import_local_media_to_output(url, wipe_first=True)
                else:
                    if "BATCH_MANAGER" in globals():
                        BATCH_MANAGER.detach_studio_task()
                    _wipe_output_workspace()
                    STATE.step_index = 1
                    STATE.percent = 5.0
                    STATE.step_name = f"1. 正在下载在线视频 ({payload.resolution}p)..."
                    STATE.sub_step = "正在调用 yt-dlp 下载音视频流..."
                    _1_ytdlp.download_video_ytdlp(url, resolution=payload.resolution)
                    save_youtube_url(url)
                    if payload.prefer_official_sub:
                        STATE.percent = 12.0
                        STATE.sub_step = "正在探测并下载官方字幕轨道..."
                        sub_path, sub_type = _1_ytdlp.download_subtitle_ytdlp(url)
                        if sub_path:
                            lang = load_key("whisper.language")
                            if lang == "auto":
                                lang = "en"
                            use_subtitle_file(sub_path, lang)

            if STATE.stop_requested:
                raise InterruptedError("任务已被手动停止")

            # Stage B: Subtitle Pipeline
            if not os.path.exists(_2_CLEANED_CHUNKS):
                STATE.step_index = 2
                STATE.percent = 20.0
                STATE.step_name = "2. 正在执行 WhisperX 转录与说话人声纹分离..."
                STATE.sub_step = "正在提取音频并运行 WhisperX..."
                _2_asr.transcribe()
                STATE.percent = 28.0
                STATE.sub_step = "正在执行 Pyannote 多说话人分离..."
                _2_5_diarize.diarize_main()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if not os.path.exists(_3_2_SPLIT_BY_MEANING):
                STATE.step_index = 3
                STATE.percent = 36.0
                STATE.step_name = "3. 正在执行 NLP + LLM 语义分句..."
                _3_1_split_nlp.split_by_spacy()
                _3_2_split_meaning.split_sentences_by_meaning()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if not os.path.exists(_4_2_TRANSLATION):
                STATE.step_index = 4
                STATE.percent = 48.0
                STATE.step_name = "4. 正在提取专业术语并执行三步翻译..."
                _4_1_summarize.get_summary()
                STATE.percent = 55.0
                _4_2_translate.translate_all()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if not os.path.exists(_5_SPLIT_SUB):
                STATE.step_index = 5
                STATE.percent = 65.0
                STATE.step_name = "5. 正在切分长字幕并生成双语时间轴..."
                _5_split_sub.split_for_sub_main()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if not os.path.exists(SRC_SRT):
                _6_gen_sub.align_timestamp_main()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if mode == "sub_only":
                STATE.percent = 100.0
                STATE.toast = "🎉 双语字幕全部生成完成！"
                return

            if not os.path.exists(SUB_VIDEO):
                STATE.step_index = 6
                STATE.percent = 72.0
                STATE.step_name = "6. 正在将双语字幕烧录至视频..."
                _7_sub_into_vid.merge_subtitles_to_video()
                if STATE.stop_requested:
                    raise InterruptedError("任务已被手动停止")

            if mode == "sub_video":
                STATE.percent = 100.0
                STATE.toast = "🎉 带字幕视频 (output_sub.mp4) 烧录完成！"
                return

            # Stage C: Dubbing Pipeline
            if not os.path.exists(_RAW_AUDIO_FILE):
                STATE.step_index = 7
                STATE.percent = 76.0
                STATE.step_name = "7. 正在提取基础音频并执行说话人性别识别..."
                video_file = _1_ytdlp.find_video_files()
                if video_file and os.path.exists(video_file):
                    convert_video_to_audio(video_file)

            if not os.path.exists(_8_1_AUDIO_TASK):
                STATE.step_index = 7
                STATE.percent = 78.0
                STATE.step_name = "7. 正在执行说话人性别识别与配音切分..."
                _8_1_audio_task.gen_audio_task_main()
                _8_2_dub_chunks.gen_dub_chunks()
            else:
                sync_res = _8_1_audio_task.sync_subtitles_to_audio_files()
                df_existing = pd.read_excel(_8_1_AUDIO_TASK)
                has_lines = (
                    "lines" in df_existing.columns
                    and len(df_existing) > 0
                    and df_existing["lines"].notna().all()
                )
                if not has_lines:
                    _8_2_dub_chunks.gen_dub_chunks()
                elif (
                    sync_res.get("updated_task_rows", 0) > 0
                    or "cut_off" not in df_existing.columns
                    or "gap" not in df_existing.columns
                    or "para_id" not in df_existing.columns
                ):
                    df_existing = _8_2_dub_chunks.analyze_subtitle_timing_and_speed(df_existing)
                    df_existing = _8_2_dub_chunks.process_cutoffs(df_existing)
                    df_existing.to_excel(_8_1_AUDIO_TASK, index=False)

            if STATE.stop_requested:
                raise InterruptedError("任务已被手动停止")

            if payload.pause_before_dub and not STATE.speaker_review_confirmed:
                STATE.percent = 80.0
                STATE.toast = "⏸️ 已完成说话人识别，等待确认角色性别与音色后继续合成配音"
                return

            STATE.step_index = 8
            STATE.percent = 82.0
            STATE.step_name = "8. 正在准备各角色参考音频..."
            if not load_key("skip_refer"):
                _9_refer_audio.extract_refer_audio_main()

            if STATE.stop_requested:
                raise InterruptedError("任务已被手动停止")

            STATE.step_index = 9
            STATE.percent = 88.0
            STATE.step_name = "9. 正在并发调用 TTS 合成多角色配音并对齐时间轴..."
            _10_gen_audio.gen_audio()
            _11_merge_audio.merge_full_audio()

            if STATE.stop_requested:
                raise InterruptedError("任务已被手动停止")

            STATE.step_index = 10
            STATE.percent = 96.0
            STATE.step_name = "10. 正在混流配音与背景伴奏合成最终视频..."
            _12_dub_to_vid.merge_video_audio()
            STATE.percent = 100.0
            STATE.toast = "🎉 一键全自动字幕翻译与多角色配音视频合成完成！"
        except (InterruptedError, TaskCancelledError):
            STATE.toast = "⏹️ 一键处理任务已手动停止，已完成步骤缓存已保留"
        except Exception as e:
            traceback.print_exc()
            STATE.error = str(e)
        finally:
            if "BATCH_MANAGER" in globals():
                BATCH_MANAGER.sync_from_studio_if_loaded()
            STATE.finish_task()

    threading.Thread(target=_one_shot_worker, daemon=True).start()
    return {"status": "started", "mode": payload.mode}


def _parse_srt_blocks(srt_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(srt_path):
        return []
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()
    if not content:
        return []
    raw_blocks = re.split(r"\r?\n\r?\n+", content)
    entries = []
    for block in raw_blocks:
        lines = [line.strip("\r") for line in block.splitlines() if line.strip("\r") != ""]
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
            timecode = lines[1].strip()
            text = "\n".join(lines[2:])
        except ValueError:
            idx = len(entries) + 1
            timecode = lines[0].strip()
            text = "\n".join(lines[1:])
        entries.append({"index": idx, "timecode": timecode, "text": text})
    return entries


def _write_srt_blocks(srt_path: str, entries: List[Dict[str, Any]]):
    out_blocks = []
    for i, item in enumerate(entries, start=1):
        idx = item.get("index") or i
        tc = item.get("timecode", "00:00:00,000 --> 00:00:02,000")
        txt = item.get("text", "")
        out_blocks.append(f"{idx}\n{tc}\n{txt}")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(out_blocks) + "\n")


@app.get("/api/content/subtitles")
def get_subtitle_content(
    srt_type: str = "src_trans",
    task_id: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
    query: Optional[str] = None,
):
    valid_types = {"src", "trans", "src_trans", "trans_src", "dub"}
    if srt_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"srt_type 必须是 {sorted(valid_types)} 之一")

    base_dir = "output"
    if task_id and "BATCH_MANAGER" in globals():
        t_dir = BATCH_MANAGER.get_task_dir(task_id)
        if not os.path.exists(os.path.join(base_dir, f"{srt_type}.srt")) and os.path.exists(t_dir):
            base_dir = t_dir
        elif task_id != BATCH_MANAGER.active_task_id and task_id != BATCH_MANAGER.studio_loaded_task_id:
            base_dir = t_dir

    srt_path = os.path.join(base_dir, f"{srt_type}.srt")
    available_files = [
        t for t in ["src", "trans", "src_trans", "trans_src", "dub"] if os.path.exists(os.path.join(base_dir, f"{t}.srt"))
    ]
    if not os.path.exists(srt_path):
        return {
            "exists": False,
            "srt_type": srt_type,
            "file_path": srt_path,
            "available_srt_types": available_files,
            "total": 0,
            "entries": [],
        }

    entries = _parse_srt_blocks(srt_path)
    if query and query.strip():
        q_lower = query.strip().lower()
        entries = [e for e in entries if q_lower in e["text"].lower()]

    total = len(entries)
    safe_offset = max(0, int(offset))
    safe_limit = max(1, min(500, int(limit)))
    paged = entries[safe_offset : safe_offset + safe_limit]

    return {
        "exists": True,
        "srt_type": srt_type,
        "file_path": srt_path,
        "available_srt_types": available_files,
        "total": total,
        "offset": safe_offset,
        "limit": safe_limit,
        "entries": paged,
    }


class SubtitleEditItem(BaseModel):
    index: int
    text: str


class SubtitleUpdatePayload(BaseModel):
    srt_type: str = "src_trans"
    edits: List[SubtitleEditItem]
    reburn_video: bool = False


def _sync_sibling_display_srts(edited_srt_type: str, base_dir: str = "output") -> List[str]:
    """Keep `src.srt`, `trans.srt`, `src_trans.srt`, and `trans_src.srt` consistent when one is edited."""
    synced = []
    if edited_srt_type not in {"src", "trans", "src_trans", "trans_src"}:
        return synced

    src_p = os.path.join(base_dir, "src.srt")
    trans_p = os.path.join(base_dir, "trans.srt")
    st_p = os.path.join(base_dir, "src_trans.srt")
    ts_p = os.path.join(base_dir, "trans_src.srt")

    if edited_srt_type in {"src_trans", "trans_src"}:
        bi_path = st_p if edited_srt_type == "src_trans" else ts_p
        bi_entries = _parse_srt_blocks(bi_path)
        src_entries = _parse_srt_blocks(src_p)
        trans_entries = _parse_srt_blocks(trans_p)
        if bi_entries and len(src_entries) == len(bi_entries) and len(trans_entries) == len(bi_entries):
            for idx, b in enumerate(bi_entries):
                parts = [ln.strip() for ln in str(b["text"]).splitlines()]
                if len(parts) >= 2:
                    if edited_srt_type == "src_trans":
                        s_txt, t_txt = parts[0], " ".join(parts[1:])
                    else:
                        t_txt, s_txt = parts[0], " ".join(parts[1:])
                    src_entries[idx]["text"] = s_txt
                    trans_entries[idx]["text"] = t_txt
            _write_srt_blocks(src_p, src_entries)
            _write_srt_blocks(trans_p, trans_entries)
            synced.extend(["src.srt", "trans.srt"])

    src_entries = _parse_srt_blocks(src_p)
    trans_entries = _parse_srt_blocks(trans_p)
    if src_entries and len(src_entries) == len(trans_entries):
        if os.path.exists(st_p) and edited_srt_type != "src_trans":
            st_entries = [
                {"index": s["index"], "timecode": s["timecode"], "text": f"{s['text']}\n{t['text']}"}
                for s, t in zip(src_entries, trans_entries)
            ]
            _write_srt_blocks(st_p, st_entries)
            synced.append("src_trans.srt")
        if os.path.exists(ts_p) and edited_srt_type != "trans_src":
            ts_entries = [
                {"index": s["index"], "timecode": s["timecode"], "text": f"{t['text']}\n{s['text']}"}
                for s, t in zip(src_entries, trans_entries)
            ]
            _write_srt_blocks(ts_p, ts_entries)
            synced.append("trans_src.srt")

    return synced


@app.post("/api/content/subtitles/update")
def update_subtitle_content(payload: SubtitleUpdatePayload):
    valid_types = {"src", "trans", "src_trans", "trans_src", "dub"}
    if payload.srt_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"srt_type 必须是 {sorted(valid_types)} 之一")

    srt_path = os.path.join("output", f"{payload.srt_type}.srt")
    if not os.path.exists(srt_path):
        raise HTTPException(status_code=404, detail=f"字幕文件不存在: {srt_path}")

    entries = _parse_srt_blocks(srt_path)
    edit_map = {item.index: item.text for item in payload.edits}
    updated_count = 0
    for entry in entries:
        if entry["index"] in edit_map:
            entry["text"] = edit_map[entry["index"]]
            updated_count += 1

    _write_srt_blocks(srt_path, entries)
    synced_srts = _sync_sibling_display_srts(payload.srt_type, "output")
    audio_sync = {}
    if payload.srt_type in {"src", "trans", "src_trans", "trans_src"}:
        audio_sync = _8_1_audio_task.sync_subtitles_to_audio_files()

    if "BATCH_MANAGER" in globals():
        BATCH_MANAGER.sync_from_studio_if_loaded()

    reburn_started = False
    if payload.reburn_video and not STATE.running:
        start_subtitle_pipeline(burn_only=True)
        reburn_started = True

    return {
        "status": "ok",
        "srt_path": srt_path,
        "updated_entries": updated_count,
        "synced_srt_files": synced_srts,
        "audio_task_sync": audio_sync,
        "reburn_started": reburn_started,
    }


@app.get("/api/content/terminology")
def get_terminology_content():
    term_path = "output/log/terminology.json"
    term_data = {"theme": "", "terms": []}
    if os.path.exists(term_path):
        try:
            with open(term_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    term_data = loaded
        except Exception:
            pass

    suspicious = []
    if os.path.exists(SRC_SRT):
        try:
            suspicious = detect_suspicious_terms_from_srt(SRC_SRT)[:25]
        except Exception:
            suspicious = []

    return {
        "terminology_file_exists": os.path.exists(term_path),
        "theme": term_data.get("theme", ""),
        "terms": term_data.get("terms", []),
        "suspicious_terms": suspicious,
        "domain_vocab": str(load_key("domain_vocab") or ""),
    }


class TerminologyUpdatePayload(BaseModel):
    theme: Optional[str] = None
    terms: Optional[List[Dict[str, str]]] = None
    domain_vocab: Optional[str] = None
    reapply_to_chunks: bool = False


@app.post("/api/content/terminology")
def update_terminology_content(payload: TerminologyUpdatePayload):
    term_path = "output/log/terminology.json"
    os.makedirs(os.path.dirname(term_path), exist_ok=True)
    term_data: Dict[str, Any] = {"theme": "", "terms": []}
    if os.path.exists(term_path):
        try:
            with open(term_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    term_data = loaded
        except Exception:
            pass

    if payload.theme is not None:
        term_data["theme"] = payload.theme

    if payload.terms is not None:
        existing_terms = term_data.get("terms", [])
        by_src = {str(t.get("src", "")).strip().lower(): t for t in existing_terms if t.get("src")}
        for item in payload.terms:
            src_val = str(item.get("src", "")).strip()
            if not src_val:
                continue
            by_src[src_val.lower()] = {
                "src": src_val,
                "tgt": str(item.get("tgt", "")).strip(),
                "note": str(item.get("note", "")).strip(),
            }
        term_data["terms"] = list(by_src.values())

    with open(term_path, "w", encoding="utf-8") as f:
        json.dump(term_data, f, ensure_ascii=False, indent=4)

    if payload.domain_vocab is not None:
        update_key("domain_vocab", payload.domain_vocab)

    reapplied = False
    if payload.reapply_to_chunks:
        reapply_terms()
        reapplied = True

    return {
        "status": "ok",
        "theme": term_data.get("theme", ""),
        "terms_count": len(term_data.get("terms", [])),
        "domain_vocab": str(load_key("domain_vocab") or ""),
        "reapplied_to_chunks": reapplied,
    }


# =====================================================================
# Serve Vue 3 SPA Frontend & Mount MCP Endpoints
# =====================================================================
from backend.original_narration_api import register_original_narration_routes

register_original_narration_routes(
    app, is_busy=lambda: STATE.running or BATCH_MANAGER.running
)

from backend.mcp_server import register_mcp_routes

register_mcp_routes(app)


@app.get("/", response_class=HTMLResponse)
def serve_vue_studio():
    dist_index = os.path.join(ROOT_DIR, "frontend", "dist", "index.html")
    if os.path.exists(dist_index):
        with open(dist_index, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    raise HTTPException(status_code=404, detail="Frontend dist/index.html not found")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting VideoLingo Q 2.0 Vue Studio Server on http://127.0.0.1:8502 ...")
    uvicorn.run(
        "backend.server:app",
        host="127.0.0.1",
        port=8502,
        reload=True,
        reload_dirs=[os.path.join(ROOT_DIR, "backend"), os.path.join(ROOT_DIR, "core")],
    )
