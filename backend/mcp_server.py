#!/usr/bin/env python3
"""
VideoLingo Q 2.0 - Model Context Protocol (MCP) Server
Supports both:
1. Local stdio transport (for Claude Desktop, Antigravity, Cursor, ChatGPT Desktop, OpenAI Agents SDK)
   - Automatically starts the VideoLingo Vue Studio FastAPI backend (127.0.0.1:8502) if not running.
2. FastAPI HTTP / Streamable HTTP (`/mcp`) & SSE (`/mcp/sse` + `/mcp/messages`) endpoints mounted directly on 127.0.0.1:8502.
"""

import os
import io
import sys
import json
import time
import uuid
import queue
import socket
import asyncio
import threading
import traceback
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, List, Optional, Callable

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

STUDIO_BASE_URL = os.environ.get("VIDEOLINGO_API_URL", "http://127.0.0.1:8502").rstrip("/")
MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "videolingo-q-mcp",
    "version": "2.0.0",
}


def _log_stderr(msg: str):
    """Write diagnostic logs strictly to stderr so stdio JSON-RPC is never polluted."""
    try:
        sys.stderr.write(f"[VideoLingo-MCP] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


# =====================================================================
# Backend Auto-Starter & HTTP Bridge Client
# =====================================================================
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen_local_aware(req: urllib.request.Request, timeout: float):
    host = (urllib.parse.urlparse(req.full_url).hostname or "").lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        return _NO_PROXY_OPENER.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _is_backend_alive(base_url: str = STUDIO_BASE_URL, timeout: float = 1.5) -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/api/workspace/state", method="GET")
        with _urlopen_local_aware(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


_BACKEND_START_LOCK = threading.Lock()


def ensure_backend_running(base_url: str = STUDIO_BASE_URL, max_wait: float = 20.0) -> bool:
    """Ensure the VideoLingo FastAPI Studio server on 127.0.0.1:8502 is running."""
    if _is_backend_alive(base_url, timeout=1.0):
        return True

    with _BACKEND_START_LOCK:
        if _is_backend_alive(base_url, timeout=1.0):
            return True

        _log_stderr(f"Backend at {base_url} is not running. Auto-starting backend/server.py ...")
        os.makedirs(os.path.join(ROOT_DIR, "output"), exist_ok=True)
        log_path = os.path.join(ROOT_DIR, "output", "mcp_backend_autostart.log")

        python_bin = sys.executable
        pyenv_candidate = "/Users/niuzhidao/.pyenv/versions/Project/bin/python"
        if os.path.exists(pyenv_candidate) and os.access(pyenv_candidate, os.X_OK):
            python_bin = pyenv_candidate

        server_script = os.path.join(ROOT_DIR, "backend", "server.py")
        try:
            log_file = open(log_path, "a", encoding="utf-8")
            subprocess.Popen(
                [python_bin, server_script],
                cwd=ROOT_DIR,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        except Exception as e:
            _log_stderr(f"Failed to spawn backend/server.py: {e}")
            return False

        deadline = time.time() + max_wait
        while time.time() < deadline:
            if _is_backend_alive(base_url, timeout=1.0):
                _log_stderr(f"Backend at {base_url} is now ready.")
                return True
            time.sleep(0.5)

        _log_stderr(f"Timed out waiting for backend at {base_url} after {max_wait}s.")
        return False


def _api_request(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    query: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Send an HTTP request to the local VideoLingo Studio backend API."""
    if not ensure_backend_running():
        raise RuntimeError(
            f"无法连接或自动启动 VideoLingo 后端服务 ({STUDIO_BASE_URL})，请检查环境或运行 ./start_vue_studio.sh"
        )

    url = f"{STUDIO_BASE_URL}{path}"
    if query:
        filtered_q = {k: v for k, v in query.items() if v is not None}
        if filtered_q:
            url += "?" + urllib.parse.urlencode(filtered_q)

    data_bytes = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
    try:
        with _urlopen_local_aware(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {"status": "ok"}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            err_json = json.loads(err_body)
            detail = err_json.get("detail") or err_body
        except Exception:
            detail = err_body or str(e)
        raise RuntimeError(f"HTTP {e.code} 错误 ({path}): {detail}")
    except Exception as e:
        raise RuntimeError(f"请求后端接口 {path} 失败: {e}")


def _wait_and_summarize_status(wait_seconds: float = 0.0, include_logs: int = 15) -> Dict[str, Any]:
    """Optionally poll task progress for up to wait_seconds (capped at 55s to prevent MCP client timeout)."""
    wait_sec = max(0.0, min(float(wait_seconds or 0.0), 55.0))
    if wait_sec > 0:
        deadline = time.time() + wait_sec
        # Brief initial sleep so newly spawned worker threads register their running=True state
        time.sleep(min(0.8, wait_sec))
        while time.time() < deadline:
            state = _api_request("GET", "/api/workspace/state")
            batch_state = _api_request("GET", "/api/batch/state")
            task = state.get("task", {})
            queue_running = bool(batch_state.get("queue", {}).get("running"))
            if (not task.get("running") and not queue_running) or task.get("error"):
                break
            time.sleep(1.0)

    state = _api_request("GET", "/api/workspace/state")
    batch_state = _api_request("GET", "/api/batch/state")
    task = state.get("task", {})
    video = state.get("video", {})
    stage_b = state.get("stage_b", {})
    stage_c = state.get("stage_c", {})
    logs = state.get("logs", [])[-max(1, int(include_logs)) :]

    return {
        "studio_url": STUDIO_BASE_URL,
        "task": task,
        "video": {
            "exists": video.get("exists"),
            "title": video.get("title"),
            "file_name": video.get("file_name"),
            "youtube_url": video.get("youtube_url"),
            "orig_file": f"output/{video['file_name']}" if video.get("file_name") else None,
            "sub_video_file": "output/output_sub.mp4" if stage_b.get("sub_video_exists") else None,
            "dub_video_file": "output/output_dub.mp4" if stage_c.get("dub_video_exists") else None,
            "src_srt_file": "output/src.srt" if stage_b.get("src_srt_exists") else None,
            "bilingual_srt_file": "output/src_trans.srt" if os.path.exists(os.path.join(ROOT_DIR, "output/src_trans.srt")) else None,
        },
        "subtitle_pipeline": {
            "steps": stage_b.get("steps", []),
            "src_srt_exists": stage_b.get("src_srt_exists"),
            "sub_video_exists": stage_b.get("sub_video_exists"),
            "suspicious_terms_count": len(stage_b.get("suspicious_terms", [])),
        },
        "dubbing_pipeline": {
            "steps": stage_c.get("steps", []),
            "waiting_speaker_confirm": stage_c.get("waiting_confirm"),
            "dub_video_exists": stage_c.get("dub_video_exists"),
            "speakers_count": len(stage_c.get("speaker_review", {}).get("speakers", [])),
        },
        "batch_queue": batch_state.get("queue", {}),
        "recent_logs": logs,
    }


# =====================================================================
# MCP Tool Definitions & Handlers
# =====================================================================
MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "videolingo_one_shot_process",
        "description": (
            "【一键全自动处理视频】提交在线视频链接（YouTube/Bilibili等）或本地音视频文件绝对路径，"
            "在 VideoLingo 工作台中全自动串联完成：1.下载/导入视频与探测官方字幕 -> 2.WhisperX语音转录与声纹分离 -> "
            "3.NLP+LLM语义断句与三步翻译 -> 4.生成双语字幕并烧录视频 -> 5.多角色TTS配音与合成最终视频。"
            "默认异步启动并可通过 wait_seconds 等待初始进展。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "在线视频 URL（如 YouTube 链接）。与 local_file_path 二选一。",
                },
                "local_file_path": {
                    "type": "string",
                    "description": "本地视频或音频文件的绝对路径（如 /Users/.../video.mp4）。与 url 二选一。",
                },
                "mode": {
                    "type": "string",
                    "enum": ["dub_video", "sub_video", "sub_only"],
                    "default": "dub_video",
                    "description": "处理模式：'dub_video'（生成双语字幕+多角色配音成片）、'sub_video'（生成并烧录双语字幕视频）、'sub_only'（仅生成SRT字幕文件不压制视频）",
                },
                "source_lang": {
                    "type": "string",
                    "description": "源视频语言代码（如 'en', 'ja', 'zh'，留空则保持当前配置）",
                },
                "target_lang": {
                    "type": "string",
                    "description": "翻译目标语言（如 '简体中文', 'English', '日本語'，留空则保持当前配置）",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["1080", "720", "480", "best"],
                    "default": "1080",
                    "description": "下载在线视频时的目标分辨率",
                },
                "prefer_official_sub": {
                    "type": "boolean",
                    "default": True,
                    "description": "下载在线视频时是否优先使用官方/CC字幕以跳过 WhisperX",
                },
                "tts_method": {
                    "type": "string",
                    "description": "配音 TTS 引擎（如 'doubao_tts' 或 'edge_tts'，留空则使用当前配置）",
                },
                "male_voice": {
                    "type": "string",
                    "description": "男声角色默认 TTS 音色 ID（留空则使用当前配置）",
                },
                "female_voice": {
                    "type": "string",
                    "description": "女声角色默认 TTS 音色 ID（留空则使用当前配置）",
                },
                "pause_before_dub": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否在说话人识别后暂停等待人工/智能体确认音色后再合成配音（全自动模式默认 False）",
                },
                "wait_seconds": {
                    "type": "number",
                    "default": 5,
                    "description": "启动后等待并返回进度的秒数（0~55秒，默认等待 5 秒以观察任务启动情况）",
                },
            },
        },
    },
    {
        "name": "videolingo_get_status",
        "description": (
            "【查询状态与实时进度】获取 VideoLingo 当前单视频工作台（Studio）与批量队列的完整状态，"
            "包括当前运行阶段、进度百分比（percent）、子步骤（sub_step）、预计剩余时间（ETA）、报错信息、"
            "最新控制台日志以及已生成的视频和字幕文件路径。支持设置 wait_seconds 等待任务推进后再返回。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "wait_seconds": {
                    "type": "number",
                    "default": 0,
                    "description": "可选：等待指定秒数（0~55秒）后再返回状态，适合在轮询长耗时任务时使用（如设为 10 或 20）",
                },
                "include_logs": {
                    "type": "integer",
                    "default": 20,
                    "description": "返回的最新控制台日志行数（默认 20 行）",
                },
            },
        },
    },
    {
        "name": "videolingo_stop_task",
        "description": "【停止任务/清除错误】向当前正在运行的单视频流水线或批量队列发送停止信号，或清除任务报错状态。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "clear_error_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "若为 True，则仅清除错误提示状态而不终止任务",
                }
            },
        },
    },
    {
        "name": "videolingo_get_config",
        "description": (
            "【读取系统配置与音色列表】获取 VideoLingo 的完整配置信息（config.yaml）、YouTube 会员 Cookies 状态（cookies_status）、"
            "已保存的 LLM 预设列表、自定义领域术语表（custom_terms）以及所有可用的 TTS 男女声音色目录（豆包 TTS / Edge TTS）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "videolingo_update_config",
        "description": (
            "【修改系统配置】更新 VideoLingo 的配置项、自定义专业术语或 YouTube Cookies。"
            "常用配置键包括：'target_language'（目标语言，如 '简体中文'）、'whisper.language'（源语言，如 'en'）、"
            "'burn_subtitles'（是否烧录字幕，bool）、'auto_burn_subtitles'（翻译完自动压制视频，bool）、"
            "'tts_method'（'doubao_tts' 或 'edge_tts'）、'pause_before_dub'（配音前暂停确认说话人，bool）、"
            "'youtube.cookies_from_browser'（浏览器自动提取 Cookie，如 'chrome'/'edge'/'safari'/'firefox'/''）、"
            "'doubao_tts.male_voice'、'doubao_tts.female_voice'、'edge_tts.male_voice'、'edge_tts.female_voice'、"
            "'max_workers'（并发数）等。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "object",
                    "description": "要修改的配置键值对字典，支持点号路径（如 {'target_language': '简体中文', 'tts_method': 'edge_tts'}）",
                },
                "custom_terms": {
                    "type": "string",
                    "description": "可选：更新全局领域词汇修正表（domain_vocab 文本）",
                },
                "cookies_content": {
                    "type": "string",
                    "description": "可选：直接写入并持久化 YouTube Cookie 内容（支持 Netscape cookies.txt、Cookie-Editor JSON 或请求头 Cookie 字符串）",
                },
            },
        },
    },
    {
        "name": "videolingo_manage_llm",
        "description": "【管理大模型预设与测试连通性】加载、保存、删除 LLM API 预设（Profile），或测试当前 LLM 接口连通性与延迟。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["check_connectivity", "load_profile", "save_profile", "delete_profile"],
                    "description": "执行的操作类型",
                },
                "profile_name": {
                    "type": "string",
                    "description": "当 action 为 load_profile / save_profile / delete_profile 时的预设名称",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "videolingo_load_media",
        "description": (
            "【阶段 A：加载视频或字幕】向单视频工作台（Studio）加载媒体资源。支持三种方式："
            "1. 通过 url 下载在线视频（YouTube等）并自动探测官方字幕；"
            "2. 通过 local_video_path 直接导入本地视频或音频文件（.mp4/.mov/.mkv/.mp3/.wav等）；"
            "3. 通过 local_subtitle_path 直接导入本地外挂字幕文件（.srt/.vtt）以跳过 WhisperX 语音识别。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "在线视频 URL（如 YouTube 链接）",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["1080", "720", "480", "best"],
                    "default": "1080",
                    "description": "在线视频下载分辨率",
                },
                "subtitle_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "仅下载在线视频的官方字幕而不重新下载视频文件",
                },
                "local_video_path": {
                    "type": "string",
                    "description": "本地视频或音频文件的绝对路径（直接导入工作台）",
                },
                "local_subtitle_path": {
                    "type": "string",
                    "description": "本地外挂字幕文件的绝对路径（直接导入并替代 ASR）",
                },
                "wait_seconds": {
                    "type": "number",
                    "default": 5,
                    "description": "下载在线视频时等待并返回进度的秒数（0~55秒）",
                },
            },
        },
    },
    {
        "name": "videolingo_run_subtitle_pipeline",
        "description": (
            "【阶段 B：启动字幕翻译与压制流水线】在当前工作台启动字幕生成流程："
            "1.WhisperX转录与声纹分离 -> 2.NLP+LLM语义分句 -> 3.提取术语与三步翻译 -> 4.长句切分 -> 5.生成双语SRT -> 6.烧录字幕视频。"
            "支持断点续跑、仅烧录视频（burn_only）或重置缓存后重跑（reset_first）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "burn_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "若为 True，则跳过前面翻译步骤，直接将现有字幕烧录进视频（生成 output_sub.mp4）",
                },
                "reset_first": {
                    "type": "boolean",
                    "default": False,
                    "description": "若为 True，则在启动前清空已有翻译与字幕缓存（保留原视频和WhisperX结果）重新翻译",
                },
                "wait_seconds": {
                    "type": "number",
                    "default": 5,
                    "description": "启动后等待并返回进度的秒数（0~55秒）",
                },
            },
        },
    },
    {
        "name": "videolingo_run_dubbing_pipeline",
        "description": (
            "【阶段 C：启动多角色配音流水线】在当前工作台启动配音流程："
            "1.说话人性别识别与配音分块 -> 2.提取参考音频 -> 3.并发调用TTS生成多角色语音并对齐时间轴 -> 4.音画混流合成 output_dub.mp4。"
            "若处于等待说话人确认状态，设置 force_continue=True 可确认并继续合成。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "force_continue": {
                    "type": "boolean",
                    "default": False,
                    "description": "若为 True，则标记已完成说话人音色确认并继续执行后续 TTS 合成",
                },
                "reset_first": {
                    "type": "boolean",
                    "default": False,
                    "description": "若为 True，则先删除已有配音文件与配音任务表，从头执行配音流水线",
                },
                "wait_seconds": {
                    "type": "number",
                    "default": 5,
                    "description": "启动后等待并返回进度的秒数（0~55秒）",
                },
            },
        },
    },
    {
        "name": "videolingo_workspace_cleanup",
        "description": (
            "【工作区归档与清理】管理当前工作区文件：支持将已完成的作品归档至 history 目录（archive）、"
            "彻底清空当前工作区（wipe）、重置字幕翻译缓存（reset_subtitles）或删除配音产物（delete_dubbing）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["archive", "wipe", "reset_subtitles", "delete_dubbing"],
                    "description": "'archive'（归档到 history 并清理）、'wipe'（清空当前视频与所有产物）、'reset_subtitles'（仅清空字幕翻译缓存）、'delete_dubbing'（仅清空配音缓存）",
                }
            },
            "required": ["action"],
        },
    },
    {
        "name": "videolingo_read_subtitles",
        "description": (
            "【读取字幕内容】读取当前工作台或指定批量任务已生成的 SRT 字幕条目（包含序号、时间轴、字幕文本）。"
            "支持选择原文（src）、译文（trans）、上原文下译文双语（src_trans）、上译文下原文（trans_src）或配音字幕（dub），"
            "支持分页与关键词过滤，便于智能体总结视频内容或校对翻译质量。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "srt_type": {
                    "type": "string",
                    "enum": ["src_trans", "trans", "src", "trans_src", "dub"],
                    "default": "src_trans",
                    "description": "要读取的字幕文件类型",
                },
                "task_id": {
                    "type": "string",
                    "description": "可选：批量任务 ID（若不填则读取当前 Studio 工作台 output/ 下的字幕）",
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "分页起始条目索引（从 0 开始）",
                },
                "limit": {
                    "type": "integer",
                    "default": 100,
                    "description": "单次返回的最大字幕条数（默认 100，最大 500）",
                },
                "query": {
                    "type": "string",
                    "description": "可选：按关键词过滤包含特定文本的字幕条目",
                },
            },
        },
    },
    {
        "name": "videolingo_edit_subtitles",
        "description": (
            "【修改字幕文本】按字幕序号（index）批量修改已生成的 SRT 字幕文本内容，"
            "并可选在修改后自动重新压制烧录字幕视频（reburn_video=True）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "srt_type": {
                    "type": "string",
                    "enum": ["trans", "src_trans", "trans_src", "src", "dub"],
                    "default": "src_trans",
                    "description": "要修改的目标字幕类型",
                },
                "edits": {
                    "type": "array",
                    "description": "修改项列表，每项包含字幕序号 index（从1开始）与新文本 text",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer", "description": "SRT 字幕块序号（1-based）"},
                            "text": {"type": "string", "description": "替换后的字幕文本（双语字幕可用换行符分隔两行）"},
                        },
                        "required": ["index", "text"],
                    },
                },
                "reburn_video": {
                    "type": "boolean",
                    "default": False,
                    "description": "修改保存后是否立即启动 FFmpeg 重新烧录 output_sub.mp4",
                },
            },
            "required": ["edits"],
        },
    },
    {
        "name": "videolingo_get_terminology",
        "description": (
            "【读取视频摘要与专业术语表】获取 AI 从视频中提取的主题摘要（theme）、专业术语对照表（terminology.json）、"
            "字幕中检测到的可疑未译术语（suspicious_terms）以及全局自定义词表（domain_vocab）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "videolingo_update_terminology",
        "description": (
            "【更新专业术语并重新应用】修改或新增视频专业术语对照表（terms）、修改视频主题描述（theme）或更新全局领域词表（domain_vocab）。"
            "可设置 reapply_to_chunks=True 将词汇修正重新应用到分词块并清空下游翻译缓存以便重译。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "description": "可选：更新视频主题摘要描述",
                },
                "terms": {
                    "type": "array",
                    "description": "可选：要新增或覆盖的术语列表，每项包含 src（原文）、tgt（译文）、note（解释/备注）",
                    "items": {
                        "type": "object",
                        "properties": {
                            "src": {"type": "string"},
                            "tgt": {"type": "string"},
                            "note": {"type": "string"},
                        },
                        "required": ["src", "tgt"],
                    },
                },
                "domain_vocab": {
                    "type": "string",
                    "description": "可选：更新全局 domain_vocab 自定义术语配置字符串",
                },
                "reapply_to_chunks": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否立即将术语修正应用回转录分词块并清除下游字幕缓存",
                },
            },
        },
    },
    {
        "name": "videolingo_get_speakers",
        "description": (
            "【获取说话人识别与配音台词表】获取阶段 C 识别出的所有说话人角色列表（speaker_id、性别、音色、台词数、样本台词）、"
            "可用的 TTS 音色目录，以及逐句配音台词列表（支持分页）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "配音台词列表分页起始索引",
                },
                "limit": {
                    "type": "integer",
                    "default": 100,
                    "description": "返回的配音台词最大条数（默认 100）",
                },
            },
        },
    },
    {
        "name": "videolingo_update_speakers",
        "description": (
            "【修改说话人音色与配音台词】批量设置说话人角色（如 SPEAKER_00）的性别（male/female）和指定 TTS 音色（voice），"
            "或按台词句号（number）微调单句的说话人归属、性别、音色与配音文本。"
            "设置 incremental_rebuild=True 可在保存后自动清除受影响句子的语音缓存并立即启动/继续配音合成！"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "speakers": {
                    "type": "array",
                    "description": "按说话人角色整体设置性别与音色",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker_id": {"type": "string", "description": "如 'SPEAKER_00'"},
                            "gender": {"type": "string", "enum": ["male", "female"]},
                            "voice": {"type": "string", "description": "指定 TTS 音色 ID，留空则使用该性别的默认音色"},
                        },
                        "required": ["speaker_id", "gender"],
                    },
                },
                "lines": {
                    "type": "array",
                    "description": "按单句编号（number）覆盖特定台词的说话人、性别、音色或配音文本",
                    "items": {
                        "type": "object",
                        "properties": {
                            "number": {"type": "integer", "description": "台词编号（从 1 开始）"},
                            "speaker_id": {"type": "string"},
                            "gender": {"type": "string", "enum": ["male", "female"]},
                            "voice": {"type": "string"},
                            "text": {"type": "string", "description": "修改后的配音台词文本"},
                        },
                        "required": ["number", "speaker_id", "gender"],
                    },
                },
                "incremental_rebuild": {
                    "type": "boolean",
                    "default": True,
                    "description": "保存后是否立即确认说话人并启动/增量重跑配音合成流水线",
                },
                "wait_seconds": {
                    "type": "number",
                    "default": 5,
                    "description": "若触发重跑，等待并返回进度的秒数（0~55秒）",
                },
            },
        },
    },
    {
        "name": "videolingo_batch_add_tasks",
        "description": (
            "【批量队列：添加视频任务】向批量任务队列添加一个或多个视频 URL（或本地视频路径），"
            "支持自定义每个批次的翻译语言、处理模式（dub_video/sub_video/sub_only）、分辨率和配音音色，"
            "并支持设置 auto_start=True 在添加后立即启动批量队列自动处理。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要添加的视频 URL 列表",
                },
                "mode": {
                    "type": "string",
                    "enum": ["dub_video", "sub_video", "sub_only"],
                    "default": "dub_video",
                    "description": "处理模式：'dub_video'（字幕+配音）、'sub_video'（仅烧录字幕视频）、'sub_only'（仅生成字幕文件）",
                },
                "source_lang": {"type": "string", "default": "", "description": "源语言代码（如 'en'，留空遵循全局配置）"},
                "target_lang": {"type": "string", "default": "", "description": "目标语言（如 '简体中文'，留空遵循全局配置）"},
                "resolution": {"type": "string", "default": "1080", "description": "下载分辨率（'1080', '720', '480', 'best'）"},
                "prefer_official_sub": {"type": "boolean", "default": True},
                "tts_method": {"type": "string", "default": ""},
                "male_voice": {"type": "string", "default": ""},
                "female_voice": {"type": "string", "default": ""},
                "pause_before_dub": {"type": "boolean", "default": False},
                "auto_start": {
                    "type": "boolean",
                    "default": True,
                    "description": "添加任务后是否立即启动批量队列开始执行",
                },
            },
            "required": ["urls"],
        },
    },
    {
        "name": "videolingo_batch_control",
        "description": (
            "【批量队列：查询与控制】查询批量队列所有任务状态（state），或对批量队列执行启动（start）、"
            "当前完成后暂停（pause）、立即停止（stop）、清理任务（clear）、单独运行某任务（run_single）、"
            "重置某任务缓存（reset_task）、删除某任务（delete_task）或加载某任务到主工作台（load_to_studio）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "state",
                        "start",
                        "pause",
                        "stop",
                        "clear_completed",
                        "clear_all",
                        "run_single",
                        "reset_task",
                        "delete_task",
                        "load_to_studio",
                    ],
                    "default": "state",
                    "description": "要执行的批量队列操作",
                },
                "task_id": {
                    "type": "string",
                    "description": "当 action 为 run_single / reset_task / delete_task / load_to_studio 时的目标任务 ID",
                },
                "from_scratch": {
                    "type": "boolean",
                    "default": False,
                    "description": "当 action='run_single' 时，是否清空该任务缓存从头运行",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "videolingo_original_narration",
        "description": "本机新增：本次A原创稿音频先行任务。固定11段已审稿、VV普通音色、1.00x、独立目录；不翻译、不缩写、不读取原片时间轴。start仅合成S01供验收；continue复用S01并合成其余10段。两者会调用已有豆包并可能计费；status只读。失败不自动重试。",
        "inputSchema": {
            "type": "object", "additionalProperties": False,
            "properties": {"action": {"type": "string", "enum": ["start", "continue", "status"]}},
            "required": ["action"]
        }
    },

]


def execute_mcp_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dispatch and execute an MCP tool by name and return structured data."""
    args = arguments or {}

    if name == "videolingo_original_narration":
        if set(args) != {"action"} or args.get("action") not in {"start", "continue", "status"}:
            raise ValueError("Only action=start/continue/status is supported")
        if args["action"] == "status":
            return _api_request("GET", "/api/original-narration/status")
        return _api_request("POST", "/api/original-narration/" + args["action"], payload={}, timeout=10.0)

    elif name == "videolingo_one_shot_process":
        wait_sec = float(args.pop("wait_seconds", 5) or 0)
        res = _api_request("POST", "/api/pipeline/one-shot", payload=args, timeout=60.0)
        status_summary = _wait_and_summarize_status(wait_seconds=wait_sec)
        return {"action_result": res, "workspace": status_summary}

    elif name == "videolingo_get_status":
        wait_sec = float(args.get("wait_seconds", 0) or 0)
        include_logs = int(args.get("include_logs", 20) or 20)
        return _wait_and_summarize_status(wait_seconds=wait_sec, include_logs=include_logs)

    elif name == "videolingo_stop_task":
        if args.get("clear_error_only"):
            return _api_request("POST", "/api/task/clear-error")
        return _api_request("POST", "/api/task/stop")

    elif name == "videolingo_get_config":
        return _api_request("GET", "/api/config")

    elif name == "videolingo_update_config":
        payload = {
            "updates": args.get("updates") or {},
            "custom_terms": args.get("custom_terms"),
            "cookies_content": args.get("cookies_content"),
        }
        return _api_request("POST", "/api/config", payload=payload)

    elif name == "videolingo_manage_llm":
        act = args.get("action")
        if act == "check_connectivity":
            return _api_request("POST", "/api/llm/check", payload={}, timeout=35.0)
        elif act in ("load_profile", "save_profile", "delete_profile"):
            pname = args.get("profile_name")
            if not pname:
                raise ValueError("必须提供 profile_name 参数")
            mapped = act.replace("_profile", "")
            return _api_request("POST", "/api/llm/profiles", payload={"action": mapped, "name": pname})
        else:
            raise ValueError(f"未知的 action: {act}")

    elif name == "videolingo_load_media":
        wait_sec = float(args.get("wait_seconds", 5) or 0)
        results = {}
        if args.get("local_video_path"):
            results["import_video"] = _api_request(
                "POST",
                "/api/video/import-local",
                payload={"file_path": args["local_video_path"]},
                timeout=60.0,
            )
        elif args.get("url"):
            results["download_video"] = _api_request(
                "POST",
                "/api/video/download",
                payload={
                    "url": args["url"],
                    "resolution": str(args.get("resolution") or "1080"),
                    "subtitle_only": bool(args.get("subtitle_only", False)),
                },
            )
        if args.get("local_subtitle_path"):
            results["import_subtitle"] = _api_request(
                "POST",
                "/api/subtitle/import-local",
                payload={"file_path": args["local_subtitle_path"]},
            )
        if not results:
            raise ValueError("请提供 url、local_video_path 或 local_subtitle_path 中的至少一项")
        status_summary = _wait_and_summarize_status(wait_seconds=wait_sec if args.get("url") else 0.0)
        return {"results": results, "workspace": status_summary}

    elif name == "videolingo_run_subtitle_pipeline":
        if args.get("reset_first"):
            _api_request("POST", "/api/pipeline/subtitle/reset")
        burn_only = bool(args.get("burn_only", False))
        wait_sec = float(args.get("wait_seconds", 5) or 0)
        res = _api_request("POST", "/api/pipeline/subtitle/start", query={"burn_only": str(burn_only).lower()})
        status_summary = _wait_and_summarize_status(wait_seconds=wait_sec)
        return {"action_result": res, "workspace": status_summary}

    elif name == "videolingo_run_dubbing_pipeline":
        if args.get("reset_first"):
            _api_request("DELETE", "/api/pipeline/dubbing")
        force_continue = bool(args.get("force_continue", False))
        wait_sec = float(args.get("wait_seconds", 5) or 0)
        res = _api_request(
            "POST",
            "/api/pipeline/dubbing/start",
            query={"force_continue": str(force_continue).lower()},
        )
        status_summary = _wait_and_summarize_status(wait_seconds=wait_sec)
        return {"action_result": res, "workspace": status_summary}

    elif name == "videolingo_workspace_cleanup":
        act = args.get("action")
        if act == "archive":
            return _api_request("POST", "/api/pipeline/archive")
        elif act == "wipe":
            return _api_request("DELETE", "/api/video")
        elif act == "reset_subtitles":
            return _api_request("POST", "/api/pipeline/subtitle/reset")
        elif act == "delete_dubbing":
            return _api_request("DELETE", "/api/pipeline/dubbing")
        else:
            raise ValueError(f"不支持的清理动作: {act}")

    elif name == "videolingo_read_subtitles":
        return _api_request(
            "GET",
            "/api/content/subtitles",
            query={
                "srt_type": args.get("srt_type", "src_trans"),
                "task_id": args.get("task_id"),
                "offset": args.get("offset", 0),
                "limit": args.get("limit", 100),
                "query": args.get("query"),
            },
        )

    elif name == "videolingo_edit_subtitles":
        return _api_request(
            "POST",
            "/api/content/subtitles/update",
            payload={
                "srt_type": args.get("srt_type", "src_trans"),
                "edits": args.get("edits", []),
                "reburn_video": bool(args.get("reburn_video", False)),
            },
        )

    elif name == "videolingo_get_terminology":
        return _api_request("GET", "/api/content/terminology")

    elif name == "videolingo_update_terminology":
        return _api_request(
            "POST",
            "/api/content/terminology",
            payload={
                "theme": args.get("theme"),
                "terms": args.get("terms"),
                "domain_vocab": args.get("domain_vocab"),
                "reapply_to_chunks": bool(args.get("reapply_to_chunks", False)),
            },
        )

    elif name == "videolingo_get_speakers":
        state = _api_request("GET", "/api/workspace/state")
        spk_review = state.get("stage_c", {}).get("speaker_review", {})
        lines = spk_review.get("lines", [])
        offset = max(0, int(args.get("offset", 0) or 0))
        limit = max(1, min(500, int(args.get("limit", 100) or 100)))
        return {
            "available": spk_review.get("available", False),
            "waiting_confirm": state.get("stage_c", {}).get("waiting_confirm", False),
            "speakers": spk_review.get("speakers", []),
            "voices": spk_review.get("voices", {}),
            "total_lines": len(lines),
            "offset": offset,
            "limit": limit,
            "lines": lines[offset : offset + limit],
        }

    elif name == "videolingo_update_speakers":
        wait_sec = float(args.get("wait_seconds", 5) or 0)
        rebuild = bool(args.get("incremental_rebuild", True))
        res = _api_request(
            "POST",
            "/api/speakers/save",
            payload={
                "speakers": args.get("speakers", []),
                "lines": args.get("lines", []),
                "incremental_rebuild": rebuild,
            },
        )
        workspace = _wait_and_summarize_status(wait_seconds=wait_sec if rebuild else 0.0)
        return {"save_result": res, "workspace": workspace}

    elif name == "videolingo_batch_add_tasks":
        auto_start = bool(args.get("auto_start", True))
        payload = {
            "urls": args.get("urls", []),
            "mode": args.get("mode", "dub_video"),
            "source_lang": args.get("source_lang", ""),
            "target_lang": args.get("target_lang", ""),
            "resolution": str(args.get("resolution", "1080")),
            "prefer_official_sub": bool(args.get("prefer_official_sub", True)),
            "tts_method": args.get("tts_method", ""),
            "male_voice": args.get("male_voice", ""),
            "female_voice": args.get("female_voice", ""),
            "pause_before_dub": bool(args.get("pause_before_dub", False)),
        }
        state = _api_request("POST", "/api/batch/tasks", payload=payload)
        if auto_start:
            state = _api_request("POST", "/api/batch/start")
        return state

    elif name == "videolingo_batch_control":
        act = args.get("action", "state")
        tid = args.get("task_id")
        if act == "state":
            return _api_request("GET", "/api/batch/state")
        elif act == "start":
            return _api_request("POST", "/api/batch/start")
        elif act == "pause":
            return _api_request("POST", "/api/batch/pause")
        elif act == "stop":
            return _api_request("POST", "/api/batch/stop")
        elif act == "clear_completed":
            return _api_request("POST", "/api/batch/clear", payload={"target": "completed"})
        elif act == "clear_all":
            return _api_request("POST", "/api/batch/clear", payload={"target": "all"})
        elif act == "run_single":
            if not tid:
                raise ValueError("run_single 需要提供 task_id")
            return _api_request(
                "POST",
                f"/api/batch/tasks/{tid}/run",
                payload={"from_scratch": bool(args.get("from_scratch", False))},
            )
        elif act == "reset_task":
            if not tid:
                raise ValueError("reset_task 需要提供 task_id")
            return _api_request("POST", f"/api/batch/tasks/{tid}/reset")
        elif act == "delete_task":
            if not tid:
                raise ValueError("delete_task 需要提供 task_id")
            return _api_request("DELETE", f"/api/batch/tasks/{tid}")
        elif act == "load_to_studio":
            if not tid:
                raise ValueError("load_to_studio 需要提供 task_id")
            return _api_request("POST", f"/api/batch/tasks/{tid}/load-to-studio")
        else:
            raise ValueError(f"不支持的批量控制动作: {act}")

    else:
        raise ValueError(f"未找到名为 '{name}' 的 MCP 工具")


# =====================================================================
# MCP Resources & Prompts
# =====================================================================
MCP_RESOURCES = [
    {
        "uri": "videolingo://status",
        "name": "VideoLingo 当前工作台与队列状态",
        "description": "实时查看当前视频信息、字幕与配音步骤进度、批量队列统计及最近日志",
        "mimeType": "application/json",
    },
    {
        "uri": "videolingo://config",
        "name": "VideoLingo 全局配置与可用音色目录",
        "description": "当前 config.yaml 参数、LLM 预设、TTS 音色目录与自定义术语",
        "mimeType": "application/json",
    },
    {
        "uri": "videolingo://terminology",
        "name": "当前视频摘要与专业术语对照表",
        "description": "AI 提取的视频主题摘要与术语词表 (terminology.json)",
        "mimeType": "application/json",
    },
    {
        "uri": "videolingo://subtitles/bilingual",
        "name": "当前视频生成的双语字幕 (src_trans.srt)",
        "description": "读取当前已生成的双语字幕前 200 条内容",
        "mimeType": "application/json",
    },
]

MCP_PROMPTS = [
    {
        "name": "translate_and_dub_video",
        "description": "全自动下载、翻译并配音一个视频，同时跟进进度",
        "arguments": [
            {"name": "video_url_or_path", "description": "视频链接（YouTube等）或本地文件路径", "required": True},
            {"name": "target_lang", "description": "目标语言（默认 简体中文）", "required": False},
        ],
    },
    {
        "name": "proofread_and_fix_subtitles",
        "description": "读取当前视频提取的术语表与生成的双语字幕，检查错译/漏译并自动修正",
        "arguments": [],
    },
]


def read_mcp_resource(uri: str) -> Dict[str, Any]:
    if uri == "videolingo://status":
        data = _wait_and_summarize_status(0)
    elif uri == "videolingo://config":
        data = _api_request("GET", "/api/config")
    elif uri == "videolingo://terminology":
        data = _api_request("GET", "/api/content/terminology")
    elif uri == "videolingo://subtitles/bilingual":
        data = _api_request("GET", "/api/content/subtitles", query={"srt_type": "src_trans", "limit": 200})
    else:
        raise ValueError(f"未知的 Resource URI: {uri}")

    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(data, ensure_ascii=False, indent=2),
            }
        ]
    }


def get_mcp_prompt(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = arguments or {}
    if name == "translate_and_dub_video":
        target = args.get("video_url_or_path", "<视频链接或路径>")
        tlang = args.get("target_lang", "简体中文")
        text = (
            f"请使用 VideoLingo MCP 工具 `videolingo_one_shot_process` 处理以下视频：`{target}`，"
            f"目标语言设为 `{tlang}`，模式为 `dub_video`。\n"
            "启动后请使用 `videolingo_get_status`（可设置 wait_seconds=15）跟踪处理进度，并在完成后汇报生成的视频与字幕文件路径。"
        )
    elif name == "proofread_and_fix_subtitles":
        text = (
            "请依次调用 `videolingo_get_terminology` 和 `videolingo_read_subtitles` 检查当前视频的主题摘要、术语表与双语字幕，"
            "找出不准确的专业术语或不自然的翻译，然后调用 `videolingo_update_terminology` 或 `videolingo_edit_subtitles` 进行修正。"
        )
    else:
        raise ValueError(f"未知的 Prompt 名称: {name}")

    return {
        "description": f"VideoLingo 工作流提示词: {name}",
        "messages": [
            {
                "role": "user",
                "content": {"type": "text", "text": text},
            }
        ],
    }


# =====================================================================
# JSON-RPC 2.0 Message Processor (Shared by Stdio, Streamable HTTP & SSE)
# =====================================================================
def handle_jsonrpc_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single JSON-RPC 2.0 request/notification and return a response dict (or None for notifications)."""
    if not isinstance(msg, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request"},
        }

    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # Notifications (no id) do not expect a JSON-RPC response
    if msg_id is None and method.startswith("notifications/"):
        return None

    try:
        if method == "initialize":
            client_version = params.get("protocolVersion") or MCP_PROTOCOL_VERSION
            supported_versions = {"2024-11-05", "2025-03-26", "2025-06-18"}
            negotiated_version = client_version if client_version in supported_versions else MCP_PROTOCOL_VERSION
            result = {
                "protocolVersion": negotiated_version,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": SERVER_INFO,
                "instructions": (
                    "VideoLingo Q 2.0 视频字幕翻译与多角色配音工作站 MCP 服务。"
                    "支持一键全自动处理在线/本地视频（videolingo_one_shot_process）、"
                    "分阶段控制下载/字幕/配音、读取与校对双语字幕及术语表、管理说话人音色以及批量队列处理。"
                ),
            }
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        elif method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": MCP_TOOLS}}

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments") or {}
            try:
                output = execute_mcp_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(output, ensure_ascii=False, indent=2),
                            }
                        ],
                        "isError": False,
                    },
                }
            except Exception as tool_err:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "error": str(tool_err),
                                        "tool": tool_name,
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                            }
                        ],
                        "isError": True,
                    },
                }

        elif method == "resources/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": MCP_RESOURCES}}

        elif method == "resources/templates/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"resourceTemplates": []}}

        elif method == "resources/read":
            uri = params.get("uri", "")
            res_data = read_mcp_resource(uri)
            return {"jsonrpc": "2.0", "id": msg_id, "result": res_data}

        elif method == "prompts/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": MCP_PROMPTS}}

        elif method == "prompts/get":
            pname = params.get("name", "")
            pargs = params.get("arguments") or {}
            pdata = get_mcp_prompt(pname, pargs)
            return {"jsonrpc": "2.0", "id": msg_id, "result": pdata}

        else:
            if msg_id is None:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
    except Exception as e:
        _log_stderr(f"Error handling JSON-RPC method {method}: {traceback.format_exc()}")
        if msg_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32603, "message": str(e)},
        }


# =====================================================================
# FastAPI Route Mounting (Streamable HTTP `/mcp` & SSE `/mcp/sse`)
# =====================================================================
_SSE_SESSIONS: Dict[str, "queue.Queue[Optional[str]]"] = {}
_SSE_LOCK = threading.Lock()


def register_mcp_routes(app):
    """Mount MCP Streamable HTTP (/mcp) and HTTP+SSE (/mcp/sse, /mcp/messages) routes onto FastAPI app."""
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse

    @app.get("/mcp/info")
    def mcp_info():
        return {
            "server": SERVER_INFO,
            "protocol_version": MCP_PROTOCOL_VERSION,
            "endpoints": {
                "streamable_http": f"{STUDIO_BASE_URL}/mcp",
                "sse": f"{STUDIO_BASE_URL}/mcp/sse",
                "sse_messages": f"{STUDIO_BASE_URL}/mcp/messages",
                "stdio_command": f"{sys.executable} {os.path.abspath(__file__)}",
            },
            "tools_count": len(MCP_TOOLS),
            "tools": [{"name": t["name"], "description": t["description"]} for t in MCP_TOOLS],
            "resources": MCP_RESOURCES,
            "prompts": MCP_PROMPTS,
        }

    @app.post("/mcp")
    async def mcp_streamable_http_post(request: Request):
        """MCP Streamable HTTP endpoint (supports single or batch JSON-RPC messages)."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            )

        loop = asyncio.get_running_loop()
        if isinstance(body, list):
            responses = []
            for item in body:
                resp = await loop.run_in_executor(None, handle_jsonrpc_message, item)
                if resp is not None:
                    responses.append(resp)
            if not responses:
                return Response(status_code=202)
            return JSONResponse(content=responses)
        else:
            resp = await loop.run_in_executor(None, handle_jsonrpc_message, body)
            if resp is None:
                return Response(status_code=202)
            return JSONResponse(content=resp)

    @app.get("/mcp")
    async def mcp_streamable_http_get(request: Request):
        """GET /mcp: If client requests text/event-stream, open an SSE stream; otherwise return MCP server metadata."""
        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept:
            return await mcp_sse_endpoint(request)
        return mcp_info()

    @app.delete("/mcp")
    async def mcp_streamable_http_delete():
        return Response(status_code=200)

    @app.get("/mcp/sse")
    async def mcp_sse_endpoint(request: Request):
        """Legacy & standard MCP HTTP+SSE stream endpoint."""
        session_id = uuid.uuid4().hex
        msg_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        with _SSE_LOCK:
            _SSE_SESSIONS[session_id] = msg_queue

        async def _event_generator():
            try:
                endpoint_url = f"/mcp/messages?session_id={session_id}"
                yield f"event: endpoint\ndata: {endpoint_url}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        data = msg_queue.get_nowait()
                        if data is None:
                            break
                        yield f"event: message\ndata: {data}\n\n"
                    except queue.Empty:
                        await asyncio.sleep(0.15)
            finally:
                with _SSE_LOCK:
                    _SSE_SESSIONS.pop(session_id, None)

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/mcp/messages")
    async def mcp_sse_messages(request: Request, session_id: Optional[str] = None):
        """Receive JSON-RPC messages for an active SSE session and push response to SSE stream."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, handle_jsonrpc_message, body)

        if session_id:
            with _SSE_LOCK:
                q = _SSE_SESSIONS.get(session_id)
            if q and resp is not None:
                q.put(json.dumps(resp, ensure_ascii=False))
                return Response(status_code=202, content="Accepted")

        if resp is None:
            return Response(status_code=202)
        return JSONResponse(content=resp)


# =====================================================================
# Stdio Transport Runner
# =====================================================================
def run_stdio_server():
    """Run MCP server over stdin/stdout (newline-delimited JSON-RPC 2.0)."""
    _log_stderr("Starting VideoLingo Q 2.0 MCP Server in stdio mode...")
    # Ensure UTF-8 encoding on stdin/stdout
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdin, "buffer") else sys.stdin
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True) if hasattr(sys.stdout, "buffer") else sys.stdout

    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            stdout.write(json.dumps(err_resp, ensure_ascii=False) + "\n")
            stdout.flush()
            continue

        if isinstance(msg, list):
            batch_resps = []
            for item in msg:
                r = handle_jsonrpc_message(item)
                if r is not None:
                    batch_resps.append(r)
            if batch_resps:
                stdout.write(json.dumps(batch_resps, ensure_ascii=False) + "\n")
                stdout.flush()
        else:
            resp = handle_jsonrpc_message(msg)
            if resp is not None:
                stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                stdout.flush()


if __name__ == "__main__":
    import io
    import argparse

    parser = argparse.ArgumentParser(description="VideoLingo Q 2.0 MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP transport mode: 'stdio' (default) or 'http' (launches FastAPI server on port 8502)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (when --transport http)")
    parser.add_argument("--port", type=int, default=8502, help="HTTP port (when --transport http)")
    args = parser.parse_args()

    if args.transport == "http":
        import uvicorn
        from backend.server import app as studio_app

        _log_stderr(f"Starting VideoLingo Studio + MCP HTTP Server on http://{args.host}:{args.port}/mcp ...")
        uvicorn.run(studio_app, host=args.host, port=args.port)
    else:
        run_stdio_server()
