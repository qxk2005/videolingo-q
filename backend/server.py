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
    return json.loads(json.dumps(dict(data), default=str))


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
            if os.path.isfile(os.path.join(curr_path, f)) and not f.startswith(".")
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
        sample_text = str(sample_row.get("text", ""))
        sample_origin = str(sample_row.get("origin", ""))
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
                "text": str(row.get("text", "")),
                "origin": str(row.get("origin", "")),
            }
        )

    return {
        "available": True,
        "speakers": speakers_list,
        "lines": lines_list,
        "voices": voice_catalog,
    }


# =====================================================================
# API Endpoints: Configuration & Profiles
# =====================================================================
@app.get("/api/config")
def get_config():
    cfg = _read_full_config()
    profiles = _load_profiles()
    custom_terms = str(load_key("domain_vocab") or "")
    cookies_file = "output/youtube_cookies.txt"
    return {
        "config": cfg,
        "profiles": profiles,
        "custom_terms": custom_terms,
        "cookies_exists": os.path.exists(cookies_file),
        "voice_catalogs": {
            "doubao_male": DOUBAO_MALE_VOICES,
            "doubao_female": DOUBAO_FEMALE_VOICES,
            "edge_male": EDGE_MALE_VOICES,
            "edge_female": EDGE_FEMALE_VOICES,
            "current_available": get_available_tts_voices(),
        },
    }


class ConfigUpdatePayload(BaseModel):
    updates: Dict[str, Any] = {}
    custom_terms: Optional[str] = None


@app.post("/api/config")
def update_config(payload: ConfigUpdatePayload):
    for k, v in payload.updates.items():
        update_key(k, v)
    if payload.custom_terms is not None:
        update_key("domain_vocab", payload.custom_terms)
    return {"status": "ok", "config": _read_full_config()}


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


@app.post("/api/llm/check")
def check_llm_connectivity():
    ok = check_api()
    return {"valid": bool(ok)}


class VoicePreviewPayload(BaseModel):
    tts_method: str
    voice_id: str


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


@app.delete("/api/cookies")
def delete_cookies():
    cookies_file = "output/youtube_cookies.txt"
    if os.path.exists(cookies_file):
        os.remove(cookies_file)
    if load_key("youtube.cookies_path") == cookies_file:
        update_key("youtube.cookies_path", "")
    return {"status": "ok"}


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
        except Exception as e:
            traceback.print_exc()
            STATE.error = str(e)
        finally:
            STATE.finish_task()

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


@app.post("/api/video/upload")
def upload_local_video(file: UploadFile = File(...)):
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
                    return

            if not os.path.exists(_3_2_SPLIT_BY_MEANING):
                STATE.step_index = 2
                STATE.percent = 35.0
                STATE.step_name = "2. 正在使用 NLP + LLM 进行语义分句..."
                STATE.sub_step = "正在执行 SpaCy 语法切分与 LLM 语义断句..."
                _3_1_split_nlp.split_by_spacy()
                _3_2_split_meaning.split_sentences_by_meaning()
                if STATE.stop_requested:
                    return

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
                    return

            if not os.path.exists(_5_SPLIT_SUB):
                STATE.step_index = 4
                STATE.percent = 80.0
                STATE.step_name = "4. 正在切分与对齐长字幕..."
                STATE.sub_step = "正在根据屏幕单行字数限制自适应切分长句..."
                _5_split_sub.split_for_sub_main()
                if STATE.stop_requested:
                    return

            if not os.path.exists(SRC_SRT):
                STATE.step_index = 5
                STATE.percent = 90.0
                STATE.step_name = "5. 正在生成双语字幕时间轴..."
                STATE.sub_step = "正在对齐词级时间戳并执行去重叠平滑..."
                _6_gen_sub.align_timestamp_main()
                if STATE.stop_requested:
                    return

            if load_key("burn_subtitles") and load_key("auto_burn_subtitles") and not os.path.exists(SUB_VIDEO):
                STATE.step_index = 6
                STATE.percent = 95.0
                STATE.step_name = "6. 正在烧录字幕到视频..."
                STATE.sub_step = "正在调用 FFmpeg 硬字幕压制合成 output_sub.mp4..."
                _7_sub_into_vid.merge_subtitles_to_video()

            STATE.percent = 100.0
            STATE.toast = "🎉 字幕翻译与生成全部完成！"
        except Exception as e:
            traceback.print_exc()
            STATE.error = str(e)
        finally:
            STATE.finish_task()

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


@app.post("/api/pipeline/subtitle/reapply-terms")
def reapply_terms():
    reapply_vocab_correction_to_cleaned_chunks()
    for f in _SUBTITLE_CACHE_FILES:
        if os.path.exists(f):
            os.remove(f)
    for d in _SUBTITLE_CACHE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)
    return {"status": "ok", "message": "已重新应用词汇表并清空下游字幕缓存"}


@app.post("/api/pipeline/subtitle/reset")
def reset_subtitles():
    for f in _SUBTITLE_CACHE_FILES:
        if os.path.exists(f):
            os.remove(f)
    for d in _SUBTITLE_CACHE_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)
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
                df_existing = pd.read_excel(_8_1_AUDIO_TASK)
                needs_rechunk = (
                    "cut_off" not in df_existing.columns
                    or "gap" not in df_existing.columns
                    or (len(df_existing) > 0 and float(df_existing["gap"].iloc[-1]) <= 0)
                )
                if needs_rechunk:
                    STATE.step_index = 1
                    STATE.percent = 30.0
                    STATE.sub_step = "正在重新计算字幕时间容差与跨角色配音切分点..."
                    _8_2_dub_chunks.gen_dub_chunks()

            if STATE.stop_requested:
                return

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
                return

            STATE.step_index = 3
            STATE.percent = 65.0
            STATE.step_name = "3. 正在并发生成多角色配音并对齐时间轴..."
            STATE.sub_step = "正在调用 TTS 引擎并发合成各角色语音片段..."
            _10_gen_audio.gen_audio()
            STATE.percent = 85.0
            STATE.sub_step = "正在拼接完整配音轨道与对齐时间轴..."
            _11_merge_audio.merge_full_audio()

            if STATE.stop_requested:
                return

            STATE.step_index = 4
            STATE.percent = 95.0
            STATE.step_name = "4. 正在混流配音与背景音合成最终视频..."
            STATE.sub_step = "正在调用 FFmpeg 混流配音、背景伴奏与画面..."
            _12_dub_to_vid.merge_video_audio()
            STATE.percent = 100.0
            STATE.toast = "🎉 多角色配音视频合成完成！"
        except Exception as e:
            traceback.print_exc()
            STATE.error = str(e)
        finally:
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


@app.post("/api/speakers/save")
def save_speaker_overrides(payload: SaveSpeakerReviewPayload):
    if not os.path.exists(_8_1_AUDIO_TASK):
        raise HTTPException(status_code=404, detail="配音任务表尚未生成")

    if not os.path.exists(_RAW_AUDIO_FILE):
        video_file = _1_ytdlp.find_video_files()
        if video_file and os.path.exists(video_file):
            convert_video_to_audio(video_file)

    df = pd.read_excel(_8_1_AUDIO_TASK)
    for col, default in [("gender", "female"), ("speaker_id", "SPEAKER_00"), ("voice", "")]:
        if col not in df.columns:
            df[col] = default
    df["voice"] = df["voice"].fillna("").astype(str)
    df["speaker_id"] = df["speaker_id"].fillna("SPEAKER_00").astype(str)
    df["gender"] = df["gender"].fillna("female").astype(str)

    affected_numbers = set()
    spk_map = {s.speaker_id: s for s in payload.speakers}
    line_map = {l.number: l for l in payload.lines}

    for idx, row in df.iterrows():
        num = int(row["number"])
        orig_spk = str(row["speaker_id"])
        orig_gender = str(row["gender"])
        orig_voice = str(row["voice"])
        orig_text = str(row["text"])

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

        if (
            new_spk != orig_spk
            or new_gender != orig_gender
            or new_voice != orig_voice
            or new_text != orig_text
        ):
            df.at[idx, "speaker_id"] = new_spk
            df.at[idx, "gender"] = new_gender
            df.at[idx, "voice"] = new_voice
            df.at[idx, "text"] = new_text
            affected_numbers.add(num)

    # Re-evaluate timing and cut_off boundaries across speakers
    df = _8_2_dub_chunks.analyze_subtitle_timing_and_speed(df)
    df = _8_2_dub_chunks.process_cutoffs(df)
    df.to_excel(_8_1_AUDIO_TASK, index=False)

    deleted_count = _10_gen_audio.invalidate_audio_cache_for_numbers(list(affected_numbers))

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
    for f in [DUB_VIDEO, "output/dub.mp3", "output/audio/dub.mp3", "output/dub.srt", _8_1_AUDIO_TASK]:
        if os.path.exists(f):
            os.remove(f)
    segs_dir = "output/audio/segs"
    if os.path.exists(segs_dir):
        shutil.rmtree(segs_dir)
    return {"status": "ok"}


@app.post("/api/task/stop")
def stop_current_task():
    STATE.stop_requested = True
    STATE.running = False
    STATE.toast = "⏹️ 已发送任务停止信号"
    return {"status": "ok"}


@app.post("/api/task/clear-error")
def clear_task_error():
    STATE.error = None
    return {"status": "ok"}


# =====================================================================
# API Endpoints: Batch Queue Monitor
# =====================================================================
@app.get("/api/batch/state")
def get_batch_state():
    excel_path = "batch/tasks_setting.xlsx"
    if not os.path.exists(excel_path):
        return {"tasks": []}
    try:
        df = pd.read_excel(excel_path).fillna("")
        return {"tasks": df.to_dict(orient="records")}
    except Exception as e:
        return {"tasks": [], "error": str(e)}


# =====================================================================
# Media Streaming Endpoint with HTTP Range Support (Video / Audio Scrubbing)
# =====================================================================
@app.get("/api/media/{file_path:path}")
def stream_output_media(file_path: str, request: Request):
    safe_base = os.path.abspath("output")
    target = os.path.abspath(os.path.join(safe_base, file_path))
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
# Serve Vue 3 SPA Frontend
# =====================================================================
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
