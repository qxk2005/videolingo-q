import base64
from collections import deque
import json
import os
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import wave

import requests

from core.utils import except_handler, load_key, update_key
from core.utils.config_utils import APP_ROOT

ELEVENLABS_SAMPLE_RATE = 24000
DEFAULT_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_MALE_VOICE = "nPczCjzI2devNBz1zQrb"    # Brian
DEFAULT_ELEVENLABS_FEMALE_VOICE = "pFZP5JQG7iQjIQuC4Bku"  # Lily

ELEVENLABS_VOICES_CACHE_PATH = os.path.join(APP_ROOT, "elevenlabs_voices.json")

ELEVENLABS_MODELS: Dict[str, str] = {
    "eleven_multilingual_v2": "Multilingual v2 (推荐·长视频中文最稳·支持跨段缝合)",
    "eleven_v3": "Eleven v3 (旗舰高表现力·支持普通话/粤语)",
    "eleven_flash_v2_5": "Flash v2.5 (极速低延迟·0.5倍积分消耗)",
    "eleven_turbo_v2_5": "Turbo v2.5 (均衡自然·0.5倍积分消耗)",
}

ELEVENLABS_DEFAULT_MALE_VOICES: Dict[str, str] = {
    "Brian (官方深沉解说·多语言男声)": "nPczCjzI2devNBz1zQrb",
    "George (官方温润叙事·多语言男声)": "JBFqnCBsd6RMkjVDRZzb",
    "Daniel (官方播音主持·多语言男声)": "onwK4e9ZLuTAKqWW03F9",
    "Liam (官方青年活力·多语言男声)": "TX3LPaxmHKxFdv7VOQHJ",
    "Chris (官方自然随和·多语言男声)": "iP95p4xoKVk53GoZ742B",
    "Roger (官方成熟自信·多语言男声)": "CwhRBWXzGAHq8TQ4Fs17",
}

ELEVENLABS_DEFAULT_FEMALE_VOICES: Dict[str, str] = {
    "Lily (官方清亮知性·多语言女声)": "pFZP5JQG7iQjIQuC4Bku",
    "Sarah (官方自然柔和·多语言女声)": "EXAVITQu4vr4xnSDxMaL",
    "Aria (官方生动表现·多语言女声)": "9BWtsMINqrJLrRacOk9x",
    "Charlotte (官方优雅叙事·多语言女声)": "XB0fDUnXU5powFXDhCwa",
    "Alice (官方清晰科教·多语言女声)": "Xb7hH8MSUJpSbSDYk0k2",
    "Jessica (官方轻快亲切·多语言女声)": "cgSgspJ2msm6clMCkdW9",
}

_PREVIEW_ALLOWED_HOST_SUFFIXES = (
    "elevenlabs.io",
    "googleapis.com",
    "cloudfront.net",
    "amazonaws.com",
)

_CACHE_LOCK = threading.RLock()
_REQ_HISTORY_LOCK = threading.RLock()
_SPEAKER_REQ_HISTORY: Dict[str, deque] = {}
_PCM_24K_FORBIDDEN_KEYS = set()


def get_elevenlabs_default_config() -> Dict[str, Any]:
    return {
        "api_key": "",
        "base_url": DEFAULT_ELEVENLABS_BASE_URL,
        "model_id": DEFAULT_ELEVENLABS_MODEL,
        "male_voice": DEFAULT_ELEVENLABS_MALE_VOICE,
        "female_voice": DEFAULT_ELEVENLABS_FEMALE_VOICE,
        "voice": DEFAULT_ELEVENLABS_FEMALE_VOICE,
        "stability": 0.55,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    }


def get_elevenlabs_api_key(override_key: Optional[str] = None) -> str:
    """Resolve dedicated `elevenlabs_tts.api_key` (strictly independent of `whisper.elevenlabs_api_key`)."""
    raw = override_key if override_key is not None else load_key("elevenlabs_tts.api_key")
    key = str(raw or "").strip()
    if not key or key.lower() in ("your_elevenlabs_api_key", "your_api_key"):
        raise ValueError("未配置 ElevenLabs 配音 API Key，请在配音设置中填写 elevenlabs_tts.api_key")
    return key


def get_elevenlabs_base_url() -> str:
    raw = str(load_key("elevenlabs_tts.base_url") or "").strip()
    if not raw:
        return DEFAULT_ELEVENLABS_BASE_URL
    url = raw.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url or DEFAULT_ELEVENLABS_BASE_URL


def _get_requests_proxies() -> Optional[Dict[str, str]]:
    try:
        proxy = str(load_key("proxy") or "").strip()
    except Exception:
        proxy = ""
    if proxy:
        return {"http": proxy, "https": proxy}
    return None


def get_elevenlabs_model_id() -> str:
    raw = str(load_key("elevenlabs_tts.model_id") or "").strip()
    return raw or DEFAULT_ELEVENLABS_MODEL


def get_elevenlabs_voice_settings(model_id: Optional[str] = None) -> Dict[str, Any]:
    """Build `voice_settings` payload, snapping `stability` to discrete tiers when `eleven_v3` is active."""
    mid = (model_id or get_elevenlabs_model_id()).strip()

    def _safe_float(val, default_v: float) -> float:
        if val is None or val == "":
            return default_v
        try:
            return max(0.0, min(1.0, float(val)))
        except Exception:
            return default_v

    stability = _safe_float(load_key("elevenlabs_tts.stability"), 0.55)
    similarity_boost = _safe_float(load_key("elevenlabs_tts.similarity_boost"), 0.75)
    style = _safe_float(load_key("elevenlabs_tts.style"), 0.0)
    raw_boost = load_key("elevenlabs_tts.use_speaker_boost")
    use_speaker_boost = True if raw_boost is None else bool(raw_boost)

    if mid == "eleven_v3":
        # `eleven_v3` strictly accepts discrete stability values: 0.0, 0.5, or 1.0
        if stability < 0.25:
            stability = 0.0
        elif stability < 0.75:
            stability = 0.5
        else:
            stability = 1.0
        return {
            "stability": stability,
            "similarity_boost": similarity_boost,
        }

    return {
        "stability": round(stability, 2),
        "similarity_boost": round(similarity_boost, 2),
        "style": round(style, 2),
        "use_speaker_boost": use_speaker_boost,
    }


def resolve_elevenlabs_language_code(model_id: str, text: str = "") -> Optional[str]:
    """Map `target_language` (or text script) to ISO language code for accurate Mandarin text normalization."""
    target = str(load_key("target_language") or "").strip().lower()
    if any(k in target for k in ("粤", "广东话", "cantonese", "yue")):
        return "yue" if model_id == "eleven_v3" else "zh"
    if any(k in target for k in ("中", "汉", "国语", "普通话", "chinese", "mandarin", "zh")):
        return "zh"
    if any(k in target for k in ("日", "japanese", "ja")):
        return "ja"
    if any(k in target for k in ("韩", "朝", "korean", "ko")):
        return "ko"
    if any(k in target for k in ("英", "english")) or target == "en":
        return "en"
    if any(k in target for k in ("法", "french", "français")) or target == "fr":
        return "fr"
    if any(k in target for k in ("德", "german", "deutsch")) or target == "de":
        return "de"
    if any(k in target for k in ("西", "spanish", "español")) or target == "es":
        return "es"
    if any(k in target for k in ("俄", "russian")) or target == "ru":
        return "ru"
    if re.search(r"[\u4e00-\u9fff]", str(text or "")):
        return "zh"
    return None


def resolve_elevenlabs_voice(voice_type: Optional[str] = None, gender: Optional[str] = None) -> str:
    if voice_type and str(voice_type).strip():
        return str(voice_type).strip()
    if gender == "male":
        return (
            str(load_key("elevenlabs_tts.male_voice") or "").strip()
            or str(load_key("elevenlabs_tts.voice") or "").strip()
            or DEFAULT_ELEVENLABS_MALE_VOICE
        )
    if gender == "female":
        return (
            str(load_key("elevenlabs_tts.female_voice") or "").strip()
            or str(load_key("elevenlabs_tts.voice") or "").strip()
            or DEFAULT_ELEVENLABS_FEMALE_VOICE
        )
    return (
        str(load_key("elevenlabs_tts.voice") or "").strip()
        or str(load_key("elevenlabs_tts.female_voice") or "").strip()
        or DEFAULT_ELEVENLABS_FEMALE_VOICE
    )


# ==============================================================================
# Voice Cache & Catalog Management (My Voices + Shared Chinese Voices)
# ==============================================================================

def _is_chinese_voice_meta(name: str, labels: Dict[str, Any], language: str = "", accent: str = "") -> bool:
    if re.search(r"[\u4e00-\u9fff]", str(name or "")):
        return True
    blob = " ".join(
        [
            str(language or ""),
            str(accent or ""),
            str((labels or {}).get("language", "")),
            str((labels or {}).get("accent", "")),
            str((labels or {}).get("description", "")),
        ]
    ).lower()
    return any(
        kw in blob
        for kw in ("zh", "cmn", "yue", "chinese", "mandarin", "cantonese", "taiwan", "beijing", "hong kong")
    )


def load_cached_elevenlabs_voices() -> List[Dict[str, Any]]:
    with _CACHE_LOCK:
        if not os.path.exists(ELEVENLABS_VOICES_CACHE_PATH):
            return []
        try:
            with open(ELEVENLABS_VOICES_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("voices"), list):
                return data["voices"]
        except Exception:
            pass
        return []


def save_cached_elevenlabs_voices(voices: List[Dict[str, Any]]) -> None:
    with _CACHE_LOCK:
        parent = os.path.dirname(ELEVENLABS_VOICES_CACHE_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = ELEVENLABS_VOICES_CACHE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"voices": voices}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, ELEVENLABS_VOICES_CACHE_PATH)


def _format_voice_label(v: Dict[str, Any]) -> str:
    name = str(v.get("name") or v.get("voice_id") or "Unnamed").strip()
    is_zh = bool(v.get("is_chinese"))
    accent = str(v.get("accent") or "").strip()
    category = str(v.get("category") or "").strip()
    tags = []
    if is_zh:
        tags.append("中文")
    if accent and accent.lower() not in ("", "standard"):
        tags.append(accent)
    if category in ("cloned", "generated", "professional"):
        tags.append(category)
    prefix = "🇨🇳 " if is_zh else ""
    suffix = f" ({'·'.join(tags)})" if tags else ""
    return f"{prefix}{name}{suffix}"


def get_elevenlabs_voice_catalogs() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return `(male_voices_dict, female_voices_dict)` merging cached user voices (Chinese first) + defaults."""
    cached = load_cached_elevenlabs_voices()
    # Sort Chinese voices first, then custom/cloned voices, then premade
    sorted_cached = sorted(
        cached,
        key=lambda item: (
            0 if item.get("is_chinese") else 1,
            0 if str(item.get("category", "")) != "premade" else 1,
            str(item.get("name", "")).lower(),
        ),
    )

    male_map: Dict[str, str] = {}
    female_map: Dict[str, str] = {}
    seen_male_ids = set()
    seen_female_ids = set()

    def _insert(target_map: Dict[str, str], seen_set: set, label: str, vid: str):
        vid = str(vid or "").strip()
        if not vid or vid in seen_set:
            return
        final_label = label
        idx = 2
        while final_label in target_map and target_map[final_label] != vid:
            final_label = f"{label} #{idx}"
            idx += 1
        target_map[final_label] = vid
        seen_set.add(vid)

    for v in sorted_cached:
        vid = str(v.get("voice_id") or "").strip()
        if not vid:
            continue
        label = _format_voice_label(v)
        g = str(v.get("gender") or "").strip().lower()
        if g == "male":
            _insert(male_map, seen_male_ids, label, vid)
        elif g == "female":
            _insert(female_map, seen_female_ids, label, vid)
        else:
            # Neutral / unspecified gender voices made available in both selectors
            _insert(male_map, seen_male_ids, label, vid)
            _insert(female_map, seen_female_ids, label, vid)

    for k, vid in ELEVENLABS_DEFAULT_MALE_VOICES.items():
        _insert(male_map, seen_male_ids, k, vid)
    for k, vid in ELEVENLABS_DEFAULT_FEMALE_VOICES.items():
        _insert(female_map, seen_female_ids, k, vid)

    # Ensure currently configured custom Voice IDs always appear in dropdowns
    try:
        cur_m = str(load_key("elevenlabs_tts.male_voice") or "").strip()
        if cur_m and cur_m not in seen_male_ids:
            _insert(male_map, seen_male_ids, f"🎯 自定义男声 ({cur_m[:10]}...)", cur_m)
        cur_f = str(load_key("elevenlabs_tts.female_voice") or "").strip()
        if cur_f and cur_f not in seen_female_ids:
            _insert(female_map, seen_female_ids, f"🎯 自定义女声 ({cur_f[:10]}...)", cur_f)
    except Exception:
        pass

    return male_map, female_map


def fetch_elevenlabs_my_voices(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetch personal `My Voices` from `GET /v1/voices` and persist into `elevenlabs_voices.json`."""
    key = get_elevenlabs_api_key(override_key=api_key)
    base_url = get_elevenlabs_base_url()
    resp = requests.get(
        f"{base_url}/v1/voices",
        headers={"xi-api-key": key},
        proxies=_get_requests_proxies(),
        timeout=25,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"获取 ElevenLabs 账号音色失败 (HTTP {resp.status_code}): {resp.text[:300]}")

    payload = resp.json()
    raw_voices = payload.get("voices") or []
    normalized: List[Dict[str, Any]] = []
    zh_count = 0
    for item in raw_voices:
        if not isinstance(item, dict):
            continue
        vid = str(item.get("voice_id") or "").strip()
        if not vid:
            continue
        labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
        # Check verified_languages as well for Chinese support
        verified_langs = item.get("verified_languages") or []
        has_verified_zh = any(
            isinstance(vl, dict) and str(vl.get("language", "")).lower() in ("zh", "cmn", "yue")
            for vl in verified_langs
        )
        name = str(item.get("name") or vid).strip()
        is_zh = has_verified_zh or _is_chinese_voice_meta(name, labels)
        if is_zh:
            zh_count += 1
        normalized.append(
            {
                "voice_id": vid,
                "name": name,
                "gender": str(labels.get("gender") or "").strip().lower(),
                "accent": str(labels.get("accent") or "").strip(),
                "language": str(labels.get("language") or ("zh" if has_verified_zh else "")).strip(),
                "category": str(item.get("category") or "").strip(),
                "description": str(labels.get("description") or item.get("description") or "").strip(),
                "preview_url": str(item.get("preview_url") or "").strip(),
                "is_chinese": is_zh,
            }
        )

    save_cached_elevenlabs_voices(normalized)
    male_cat, female_cat = get_elevenlabs_voice_catalogs()
    return {
        "total": len(normalized),
        "chinese_count": zh_count,
        "voices": normalized,
        "male_catalog": male_cat,
        "female_catalog": female_cat,
    }


def _classify_chinese_accent(item: Dict[str, Any]) -> str:
    blob = " ".join(
        [
            str(item.get("accent") or ""),
            str(item.get("language") or ""),
            str(item.get("name") or ""),
            str(item.get("description") or ""),
        ]
    ).lower()
    if any(k in blob for k in ("cantonese", "hong kong", "yue", "粤", "港")):
        return "cantonese"
    if any(k in blob for k in ("taiwan", "taipei", "台湾", "台灣", "台腔")):
        return "taiwan"
    return "mainland"


def search_elevenlabs_shared_voices(
    search: str = "",
    gender: str = "",
    accent: str = "",
    page_size: int = 30,
    page: int = 0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Query ElevenLabs Chinese Shared Voices Library (`GET /v1/shared-voices`) with accent/gender/keyword filtering."""
    key = get_elevenlabs_api_key(override_key=api_key)
    base_url = get_elevenlabs_base_url()
    proxies = _get_requests_proxies()
    headers = {"xi-api-key": key}

    clean_search = str(search or "").strip()
    clean_gender = str(gender or "").strip().lower()
    clean_accent = str(accent or "").strip().lower()  # "" | "mainland" | "taiwan" | "cantonese"

    def _query_once(lang_code: str, extra_search: str = "") -> Tuple[List[Dict[str, Any]], bool]:
        params: Dict[str, Any] = {
            "language": lang_code,
            "page_size": min(max(int(page_size) * 2 if clean_accent else int(page_size), 20), 100),
            "page": max(0, int(page)),
        }
        merged_q = " ".join(x for x in [clean_search, extra_search] if x).strip()
        if merged_q:
            params["search"] = merged_q
        if clean_gender in ("male", "female"):
            params["gender"] = clean_gender
        r = requests.get(
            f"{base_url}/v1/shared-voices",
            headers=headers,
            params=params,
            proxies=proxies,
            timeout=25,
        )
        if r.status_code != 200:
            detail = r.text[:300]
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"访问 ElevenLabs 共享音色库受限 (HTTP {r.status_code})：请确认 API Key 有效且账号具备付费订阅权限。详情: {detail}"
                )
            raise RuntimeError(f"查询 ElevenLabs 中文共享音色库失败 (HTTP {r.status_code}): {detail}")
        body = r.json()
        return body.get("voices") or [], bool(body.get("has_more", False))

    raw_items: List[Dict[str, Any]] = []
    has_more = False
    if clean_accent == "cantonese":
        try:
            yue_items, yue_more = _query_once("yue")
            raw_items.extend(yue_items)
            has_more = has_more or yue_more
        except Exception:
            pass
        zh_items, zh_more = _query_once("zh", extra_search="" if clean_search else "cantonese")
        raw_items.extend(zh_items)
        has_more = has_more or zh_more
    elif clean_accent == "taiwan":
        zh_items, zh_more = _query_once("zh", extra_search="" if clean_search else "taiwan")
        if len(zh_items) < 4 and not clean_search:
            more_zh, _ = _query_once("zh")
            zh_items.extend(more_zh)
        raw_items.extend(zh_items)
        has_more = zh_more
    else:
        raw_items, has_more = _query_once("zh")

    cached_ids = {str(v.get("voice_id", "")) for v in load_cached_elevenlabs_voices()}
    seen_vids = set()
    results: List[Dict[str, Any]] = []

    accent_label_map = {
        "mainland": "大陆普通话",
        "taiwan": "台湾国语",
        "cantonese": "粤语",
    }

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        vid = str(item.get("voice_id") or "").strip()
        owner_id = str(item.get("public_owner_id") or "").strip()
        if not vid or vid in seen_vids:
            continue
        seen_vids.add(vid)

        detected_region = _classify_chinese_accent(item)
        if clean_accent in ("mainland", "taiwan", "cantonese") and detected_region != clean_accent:
            continue

        results.append(
            {
                "voice_id": vid,
                "public_owner_id": owner_id,
                "name": str(item.get("name") or vid).strip(),
                "gender": str(item.get("gender") or "").strip().lower(),
                "accent": str(item.get("accent") or "").strip(),
                "region": detected_region,
                "region_label": accent_label_map.get(detected_region, "中文"),
                "age": str(item.get("age") or "").strip(),
                "use_case": str(item.get("use_case") or "").strip(),
                "description": str(item.get("description") or "").strip(),
                "preview_url": str(item.get("preview_url") or "").strip(),
                "cloned_by_count": int(item.get("cloned_by_count") or 0),
                "free_users_allowed": bool(item.get("free_users_allowed", False)),
                "in_my_voices": vid in cached_ids,
            }
        )
        if len(results) >= int(page_size):
            break

    return {
        "voices": results,
        "count": len(results),
        "has_more": has_more,
    }


def add_elevenlabs_shared_voice(
    public_owner_id: str,
    voice_id: str,
    name: str,
    gender: str = "",
    accent: str = "",
    preview_url: str = "",
    set_as: str = "",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a community shared voice into user's ElevenLabs account (`POST /v1/voices/add/...`),
    update local cache, and optionally bind as default `male` or `female` voice."""
    key = get_elevenlabs_api_key(override_key=api_key)
    base_url = get_elevenlabs_base_url()
    proxies = _get_requests_proxies()

    vid = str(voice_id or "").strip()
    owner = str(public_owner_id or "").strip()
    clean_name = str(name or vid).strip()
    if not vid:
        raise ValueError("缺少有效的 voice_id")

    cached = load_cached_elevenlabs_voices()
    existing = next((v for v in cached if str(v.get("voice_id", "")) == vid), None)
    final_voice_id = vid

    if existing is None and owner:
        url = f"{base_url}/v1/voices/add/{owner}/{vid}"
        resp = requests.post(
            url,
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json={"new_name": clean_name},
            proxies=proxies,
            timeout=25,
        )
        if resp.status_code == 200:
            res_data = resp.json()
            final_voice_id = str(res_data.get("voice_id") or vid).strip()
        else:
            body_txt = resp.text[:350]
            # If voice was already added previously in the account, sync `/v1/voices` to locate it
            if "already" in body_txt.lower() or resp.status_code == 409:
                sync_res = fetch_elevenlabs_my_voices(api_key=key)
                matched = next(
                    (
                        v
                        for v in sync_res["voices"]
                        if v["voice_id"] == vid or v["name"].lower() == clean_name.lower()
                    ),
                    None,
                )
                if matched:
                    final_voice_id = matched["voice_id"]
                else:
                    raise RuntimeError(f"添加共享音色失败 (HTTP {resp.status_code}): {body_txt}")
            else:
                raise RuntimeError(
                    f"添加共享音色至 ElevenLabs 账号失败 (HTTP {resp.status_code})，请检查账号音色槽位是否已满或订阅权限: {body_txt}"
                )

    # Upsert into local cache
    cached = load_cached_elevenlabs_voices()
    cached = [v for v in cached if str(v.get("voice_id", "")) != final_voice_id]
    cached.insert(
        0,
        {
            "voice_id": final_voice_id,
            "name": clean_name,
            "gender": str(gender or "").strip().lower(),
            "accent": str(accent or "zh").strip(),
            "language": "zh",
            "category": "shared_chinese",
            "description": "",
            "preview_url": str(preview_url or "").strip(),
            "is_chinese": True,
        },
    )
    save_cached_elevenlabs_voices(cached)

    clean_set_as = str(set_as or "").strip().lower()
    if clean_set_as == "male":
        update_key("elevenlabs_tts.male_voice", final_voice_id)
    elif clean_set_as == "female":
        update_key("elevenlabs_tts.female_voice", final_voice_id)
        update_key("elevenlabs_tts.voice", final_voice_id)

    male_cat, female_cat = get_elevenlabs_voice_catalogs()
    return {
        "voice_id": final_voice_id,
        "name": clean_name,
        "set_as": clean_set_as,
        "male_catalog": male_cat,
        "female_catalog": female_cat,
    }


def fetch_elevenlabs_preview_audio(preview_url: str) -> Tuple[bytes, str]:
    """Proxy-fetch voice preview sample from ElevenLabs / GCS CDN so mainland browsers play instantly."""
    raw_url = str(preview_url or "").strip()
    if not raw_url:
        raise ValueError("缺少试听音频地址 preview_url")
    parsed = urlparse(raw_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("https", "http") or not any(
        host == sfx or host.endswith("." + sfx) for sfx in _PREVIEW_ALLOWED_HOST_SUFFIXES
    ):
        raise ValueError(f"不安全的试听音频域名已拦截: {host}")

    resp = requests.get(raw_url, proxies=_get_requests_proxies(), timeout=20)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"拉取试听音频失败 (HTTP {resp.status_code})")
    content_type = resp.headers.get("Content-Type") or "audio/mpeg"
    return resp.content, content_type


# ==============================================================================
# Core Synthesis + Character-Timestamp Alignment + Cross-Paragraph Stitching
# ==============================================================================

def _write_pcm16_wav(pcm_bytes: bytes, save_path: str, sample_rate: int = ELEVENLABS_SAMPLE_RATE) -> float:
    even_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
    clean_pcm = bytes(pcm_bytes[:even_len])
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with wave.open(save_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clean_pcm)
    return even_len / float(sample_rate * 2)


def _decode_mp3_to_wav_24k(mp3_bytes: bytes, save_path: str) -> float:
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_mp3 = f"{save_path}.el_tmp.mp3"
    try:
        with open(tmp_mp3, "wb") as f:
            f.write(mp3_bytes)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_mp3,
                "-ar",
                str(ELEVENLABS_SAMPLE_RATE),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                save_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        if os.path.exists(tmp_mp3):
            try:
                os.remove(tmp_mp3)
            except Exception:
                pass
    try:
        with wave.open(save_path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate() or ELEVENLABS_SAMPLE_RATE)
    except Exception:
        return 0.0


def _parse_elevenlabs_alignment_to_words(alignment: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert ElevenLabs character-level `alignment` into `[{word, start, end, confidence}]`
    compatible with `slice_paragraph_by_word_timestamps`."""
    if not isinstance(alignment, dict):
        return []
    chars = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    n = min(len(chars), len(starts), len(ends))
    if n == 0:
        return []

    words: List[Dict[str, Any]] = []
    for i in range(n):
        ch = str(chars[i] or "")
        if not ch or ch.isspace():
            continue
        s_t = float(starts[i] or 0.0)
        e_t = max(s_t, float(ends[i] or s_t))
        words.append(
            {
                "word": ch,
                "start": s_t,
                "end": e_t,
                "confidence": 1.0,
            }
        )
    return words


def _get_previous_request_ids(section_id: Optional[str], voice_id: str, model_id: str) -> List[str]:
    if not section_id or model_id == "eleven_v3":
        return []
    bucket_key = f"{section_id}|{voice_id}|{model_id}"
    with _REQ_HISTORY_LOCK:
        dq = _SPEAKER_REQ_HISTORY.get(bucket_key)
        return list(dq) if dq else []


def _record_request_id(section_id: Optional[str], voice_id: str, model_id: str, request_id: Optional[str]) -> None:
    if not section_id or not request_id or model_id == "eleven_v3":
        return
    req_id = str(request_id).strip()
    if not req_id:
        return
    bucket_key = f"{section_id}|{voice_id}|{model_id}"
    with _REQ_HISTORY_LOCK:
        if bucket_key not in _SPEAKER_REQ_HISTORY:
            _SPEAKER_REQ_HISTORY[bucket_key] = deque(maxlen=3)
        if req_id not in _SPEAKER_REQ_HISTORY[bucket_key]:
            _SPEAKER_REQ_HISTORY[bucket_key].append(req_id)


def _elevenlabs_http_synthesize(
    api_key: str,
    base_url: str,
    text: str,
    save_path: str,
    voice_id: str,
    model_id: str,
    section_id: Optional[str] = None,
    return_words: bool = False,
) -> Dict[str, Any]:
    want_mp3 = str(save_path).lower().endswith(".mp3")
    proxies = _get_requests_proxies()
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "text": text.strip(),
        "model_id": model_id,
        "voice_settings": get_elevenlabs_voice_settings(model_id=model_id),
    }

    lang_code = resolve_elevenlabs_language_code(model_id=model_id, text=text)
    if lang_code:
        payload["language_code"] = lang_code

    prev_ids = _get_previous_request_ids(section_id=section_id, voice_id=voice_id, model_id=model_id)
    if prev_ids:
        payload["previous_request_ids"] = prev_ids

    # Fast binary path when previewing `.mp3` without timestamps
    if want_mp3 and not return_words:
        url = f"{base_url}/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
        resp = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=(15, 90))
        if resp.status_code in (400, 422) and ("language_code" in payload or "previous_request_ids" in payload):
            fallback_payload = {
                "text": text.strip(),
                "model_id": model_id,
                "voice_settings": get_elevenlabs_voice_settings(model_id=model_id),
            }
            resp = requests.post(url, headers=headers, json=fallback_payload, proxies=proxies, timeout=(15, 90))
        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs TTS HTTP {resp.status_code}: {resp.text[:350]}")
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return {
            "ok": True,
            "words": [],
            "sample_rate": ELEVENLABS_SAMPLE_RATE,
            "duration": 0.0,
            "model_id": model_id,
        }

    # Standard pipeline path with character-level timestamps (`/with-timestamps`)
    key_tag = api_key[-8:]
    use_pcm_24k = (not want_mp3) and (key_tag not in _PCM_24K_FORBIDDEN_KEYS)
    output_fmt = "pcm_24000" if use_pcm_24k else "mp3_44100_128"

    def _call_with_timestamps(fmt: str, body: Dict[str, Any]) -> requests.Response:
        endpoint = f"{base_url}/v1/text-to-speech/{voice_id}/with-timestamps?output_format={fmt}"
        return requests.post(endpoint, headers=headers, json=body, proxies=proxies, timeout=(15, 120))

    resp = _call_with_timestamps(output_fmt, payload)

    # Automatic tier fallback if `pcm_24000` is restricted on Free/Starter plans
    if resp.status_code == 403 and output_fmt == "pcm_24000" and "output_format" in resp.text.lower():
        _PCM_24K_FORBIDDEN_KEYS.add(key_tag)
        output_fmt = "mp3_44100_128"
        resp = _call_with_timestamps(output_fmt, payload)

    # Automatic payload fallback if optional stitching or language_code rejected by model/gateway
    if resp.status_code in (400, 422) and ("previous_request_ids" in payload or "language_code" in payload):
        minimal_payload = {
            "text": text.strip(),
            "model_id": model_id,
            "voice_settings": get_elevenlabs_voice_settings(model_id=model_id),
        }
        resp = _call_with_timestamps(output_fmt, minimal_payload)

    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs TTS HTTP {resp.status_code}: {resp.text[:350]}")

    req_id = resp.headers.get("request-id") or resp.headers.get("x-request-id")
    _record_request_id(section_id=section_id, voice_id=voice_id, model_id=model_id, request_id=req_id)

    data = resp.json()
    b64_audio = data.get("audio_base64")
    if not b64_audio:
        raise RuntimeError("ElevenLabs API 返回了空的音频数据 (audio_base64 missing)")

    audio_bytes = base64.b64decode(b64_audio)
    alignment = data.get("alignment") or data.get("normalized_alignment")
    words_info = _parse_elevenlabs_alignment_to_words(alignment)

    if want_mp3:
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(audio_bytes)
        dur_sec = 0.0
    elif output_fmt == "pcm_24000":
        dur_sec = _write_pcm16_wav(audio_bytes, save_path, sample_rate=ELEVENLABS_SAMPLE_RATE)
    else:
        dur_sec = _decode_mp3_to_wav_24k(audio_bytes, save_path)

    return {
        "ok": True,
        "words": words_info,
        "sample_rate": ELEVENLABS_SAMPLE_RATE,
        "duration": dur_sec,
        "model_id": model_id,
    }


@except_handler("Failed to generate audio using ElevenLabs TTS", retry=1, delay=1)
def elevenlabs_tts(
    text: str,
    save_path: str,
    voice_type: Optional[str] = None,
    gender: Optional[str] = None,
    section_id: Optional[str] = None,
    context_texts: Optional[List[str]] = None,
    return_words: bool = False,
):
    """
    Synthesize Mandarin / multilingual dubbing audio via ElevenLabs TTS API.
    Supports:
    - Character-level timestamps (`/with-timestamps`) for zero-crossing paragraph slicing.
    - Cross-paragraph prosody stitching (`previous_request_ids`) per speaker `section_id`.
    - Automatic `eleven_v3` discrete stability snapping & PCM/MP3 tier fallback.
    """
    api_key = get_elevenlabs_api_key()
    base_url = get_elevenlabs_base_url()
    model_id = get_elevenlabs_model_id()
    voice_id = resolve_elevenlabs_voice(voice_type=voice_type, gender=gender)

    ctx_tag = f", section={str(section_id)[:8]}" if section_id else ""
    print(f"DEBUG: [ELEVENLABS {model_id}] Generating with Voice: {voice_id}{ctx_tag}")

    res = _elevenlabs_http_synthesize(
        api_key=api_key,
        base_url=base_url,
        text=text,
        save_path=save_path,
        voice_id=voice_id,
        model_id=model_id,
        section_id=section_id,
        return_words=return_words,
    )
    print(
        f"✅ ElevenLabs TTS Success: {save_path} "
        f"(model={model_id}, chars_aligned={len(res['words'])}, sr={res['sample_rate']}Hz)"
    )
    return res if return_words else True


if __name__ == "__main__":
    pass
