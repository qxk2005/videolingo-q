import base64
import json
import os
import subprocess
import uuid
import wave
import requests
from core.utils import load_key, except_handler

DEFAULT_DOUBAO_STYLE_PROMPT = (
    "请用自然平稳、吐字清晰、语速均匀连贯的科普解说语气播讲，保持声线与情绪稳定克制。"
)
DOUBAO_SAMPLE_RATE = 24000


def get_doubao_style_prompt() -> str:
    """Resolve the global steady style prompt for Doubao TTS 2.0 `context_texts`."""
    raw = load_key("doubao_tts.style_prompt")
    if raw is None:
        return DEFAULT_DOUBAO_STYLE_PROMPT
    s = str(raw).strip()
    if not s:
        return DEFAULT_DOUBAO_STYLE_PROMPT
    if s.lower() in ("none", "off", "false", "disable"):
        return ""
    return s


def _is_foreign_voice_for_doubao(voice_id: str) -> bool:
    v = str(voice_id or "").strip()
    if not v:
        return True
    if "-" in v or v.endswith("Neural"):
        return True
    if "_" not in v and not v.startswith("BV"):
        return True
    return False


def resolve_doubao_voice(voice_type=None, gender=None) -> str:
    if voice_type and str(voice_type).strip() and not _is_foreign_voice_for_doubao(str(voice_type)):
        return str(voice_type).strip()
    if gender == "male":
        return (
            load_key("doubao_tts.male_voice")
            or load_key("doubao_tts.voice")
            or "zh_male_ruyayichen_uranus_bigtts"
        )
    if gender == "female":
        return (
            load_key("doubao_tts.female_voice")
            or load_key("doubao_tts.voice")
            or "zh_female_vv_uranus_bigtts"
        )
    return (
        load_key("doubao_tts.voice")
        or load_key("doubao_tts.female_voice")
        or "zh_female_vv_uranus_bigtts"
    )


def _write_pcm16_wav(pcm_bytes: bytes, save_path: str, sample_rate: int = DOUBAO_SAMPLE_RATE) -> float:
    """Write raw 16-bit little-endian mono PCM bytes into a standard RIFF WAV file."""
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


def _post_with_direct_fallback(url: str, **kwargs):
    """Attempt standard request first; if local proxy fails for domestic ByteDance endpoint, retry direct."""
    try:
        return requests.post(url, **kwargs)
    except requests.exceptions.ProxyError:
        kwargs_direct = dict(kwargs)
        kwargs_direct["proxies"] = {"http": None, "https": None}
        return requests.post(url, **kwargs_direct)


def _doubao_v3_stream_call(
    appid: str,
    access_token: str,
    text: str,
    save_path: str,
    voice_type: str,
    resource_id: str = "seed-tts-2.0",
    section_id: str = None,
    context_texts: list = None,
    single_request: bool = False,
):
    """Execute Volcengine V3 unidirectional streaming TTS call with optional context & word timestamps."""
    url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    want_mp3 = str(save_path).lower().endswith(".mp3")
    audio_fmt = "mp3" if want_mp3 else "pcm"

    headers = {
        "X-Api-App-Id": appid,
        "X-Api-Access-Key": access_token,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }

    additions_obj = {"disable_markdown_filter": True}
    if resource_id == "seed-tts-2.0":
        if context_texts:
            valid_ctx = [str(c).strip() for c in context_texts if str(c).strip()]
            if valid_ctx:
                additions_obj["context_texts"] = valid_ctx[:1]
        if section_id and str(section_id).strip():
            additions_obj["section_id"] = str(section_id).strip()

    payload = {
        "user": {"uid": "videolingo_user"},
        "req_params": {
            "text": text.strip(),
            "speaker": voice_type,
            "audio_params": {
                "format": audio_fmt,
                "sample_rate": DOUBAO_SAMPLE_RATE,
                "enable_timestamp": True,
                "enable_subtitle": True,
            },
            "additions": json.dumps(additions_obj, ensure_ascii=False),
        },
    }

    post = requests.post if single_request else _post_with_direct_fallback
    resp = post(url, headers=headers, json=payload, stream=True, timeout=(15, 120))
    try:
        if resp.status_code != 200:
            body_preview = resp.text[:400]
            raise RuntimeError(f"Doubao V3 HTTP {resp.status_code}: {body_preview}")

        audio_buf = bytearray()
        collected_words = []

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line_str = raw_line.decode("utf-8", errors="replace").strip()
            if line_str.startswith("data:"):
                line_str = line_str[5:].strip()
            if not line_str:
                continue
            try:
                event = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            code = event.get("code", 0)
            if code not in (0, 20000000):
                msg = event.get("message") or "Unknown V3 business error"
                err = RuntimeError(f"Doubao V3 Business Error: {msg} (Code: {code})")
                err.doubao_code = code
                raise err

            b64_audio = event.get("data")
            if b64_audio:
                audio_buf.extend(base64.b64decode(b64_audio))

            sentence_info = event.get("sentence")
            if isinstance(sentence_info, dict):
                words_raw = sentence_info.get("words")
                if isinstance(words_raw, list) and len(words_raw) > 0:
                    for w in words_raw:
                        if not isinstance(w, dict):
                            continue
                        w_text = str(w.get("word") if w.get("word") is not None else w.get("text", ""))
                        if not w_text:
                            continue
                        s_t = float(w.get("startTime", w.get("start_time", 0.0)) or 0.0)
                        e_t = float(w.get("endTime", w.get("end_time", s_t)) or s_t)
                        conf = float(w.get("confidence", 1.0) or 0.0)
                        collected_words.append(
                            {
                                "word": w_text,
                                "start": s_t,
                                "end": max(s_t, e_t),
                                "confidence": conf,
                            }
                        )

        if len(audio_buf) == 0:
            raise RuntimeError("Doubao V3 returned empty audio stream")

        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        if want_mp3:
            with open(save_path, "wb") as f:
                f.write(audio_buf)
            dur_sec = 0.0
        else:
            dur_sec = _write_pcm16_wav(audio_buf, save_path, sample_rate=DOUBAO_SAMPLE_RATE)

        return {
            "ok": True,
            "words": collected_words,
            "sample_rate": DOUBAO_SAMPLE_RATE,
            "duration": dur_sec,
            "api_version": "v3",
        }
    finally:
        resp.close()


def _doubao_v1_fallback(appid: str, access_token: str, text: str, save_path: str, voice_type: str):
    """Legacy V1 non-streaming fallback for older 1.0 custom voices without V3 entitlement."""
    url = "https://openspeech.bytedance.com/api/v1/tts"
    headers = {
        "Authorization": f"Bearer;{access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "app": {
            "appid": appid,
            "token": access_token,
            "cluster": "volcano_tts",
        },
        "user": {"uid": "videolingo_user"},
        "audio": {
            "voice_type": voice_type,
            "encoding": "mp3",
            "rate": DOUBAO_SAMPLE_RATE,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": text.strip(),
            "operation": "query",
        },
    }
    resp = _post_with_direct_fallback(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Doubao V1 HTTP {resp.status_code}: {resp.text[:300]}")
    res_json = resp.json()
    if res_json.get("code") != 3000:
        raise RuntimeError(f"{res_json.get('message')} (Code: {res_json.get('code')})")

    audio_data = base64.b64decode(res_json.get("data"))
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    if str(save_path).lower().endswith(".mp3"):
        with open(save_path, "wb") as f:
            f.write(audio_data)
    else:
        tmp_mp3 = f"{save_path}.v1_tmp.mp3"
        try:
            with open(tmp_mp3, "wb") as f:
                f.write(audio_data)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_mp3,
                    "-ar", str(DOUBAO_SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
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

    return {
        "ok": True,
        "words": [],
        "sample_rate": DOUBAO_SAMPLE_RATE,
        "duration": 0.0,
        "api_version": "v1",
    }


@except_handler("Failed to generate audio using Doubao TTS 2.0", retry=1, delay=1)
def doubao_tts(
    text,
    save_path,
    voice_type=None,
    gender=None,
    section_id=None,
    context_texts=None,
    return_words=False,
):
    """
    Generate audio using Volcengine Doubao TTS Model 2.0 (V3 Unidirectional Streaming API).
    Supports:
    - `section_id`: Multi-turn session context continuity across paragraphs of the same speaker.
    - `context_texts`: Natural-language voice style & prosody stabilization prompt.
    - `enable_subtitle` / `enable_timestamp`: Word-level timestamps (`start`, `end` in seconds).
    - 24000Hz lossless 16-bit PCM WAV output (when `save_path` ends with `.wav`).
    """
    appid = str(load_key("doubao_tts.appid") or "").strip()
    access_token = str(load_key("doubao_tts.access_token") or "").strip()
    if not appid or not access_token:
        raise ValueError("Doubao AppID or Access Token not found. Please set them in settings.")

    voice_type = resolve_doubao_voice(voice_type=voice_type, gender=gender)

    if context_texts is None:
        style_prompt = get_doubao_style_prompt()
        context_texts = [style_prompt] if style_prompt else None

    ctx_tag = f", section={str(section_id)[:8]}" if section_id else ""
    print(f"DEBUG: [DOUBAO 2.0 V3] Generating with Voice: {voice_type}{ctx_tag}")

    try:
        res = _doubao_v3_stream_call(
            appid=appid,
            access_token=access_token,
            text=text,
            save_path=save_path,
            voice_type=voice_type,
            resource_id="seed-tts-2.0",
            section_id=section_id,
            context_texts=context_texts,
        )
        print(
            f"✅ Doubao 2.0 V3 Success: {save_path} "
            f"(words={len(res['words'])}, sr={res['sample_rate']}Hz)"
        )
        return res if return_words else True
    except Exception as v3_err:
        err_code = getattr(v3_err, "doubao_code", None)
        err_str = str(v3_err)
        # Only fall back to 1.0 / V1 if resource ID mismatches a legacy 1.0 voice
        if err_code == 55000000 or "resource ID is mismatched" in err_str or "55000000" in err_str:
            print(f"⚠️ Voice '{voice_type}' mismatched seed-tts-2.0, trying seed-tts-1.0 / V1 fallback...")
            try:
                res = _doubao_v3_stream_call(
                    appid=appid,
                    access_token=access_token,
                    text=text,
                    save_path=save_path,
                    voice_type=voice_type,
                    resource_id="seed-tts-1.0",
                    section_id=None,
                    context_texts=None,
                )
                return res if return_words else True
            except Exception:
                res = _doubao_v1_fallback(appid, access_token, text, save_path, voice_type)
                return res if return_words else True
        print(f"❌ Doubao TTS 2.0 V3 Failed: {v3_err}")
        raise v3_err


def doubao_tts_with_timestamps(
    text: str,
    save_path: str,
    voice_type: str = None,
    gender: str = None,
    section_id: str = None,
    context_texts: list = None,
) -> dict:
    """Convenience wrapper that always returns `{ok, words, sample_rate, duration}`."""
    return doubao_tts(
        text=text,
        save_path=save_path,
        voice_type=voice_type,
        gender=gender,
        section_id=section_id,
        context_texts=context_texts,
        return_words=True,
    )


if __name__ == "__main__":
    pass


def original_narration_tts(text, save_path, voice_type="zh_female_vv_uranus_bigtts"):
    """Native additive service entry: one V3 request; no resource/proxy/voice fallback or retry.

    Authentication remains in this existing configured provider, never in task input,
    output, a client script or a new credential store. Legacy dubbing is unchanged.
    """
    from core.original_narration_voices import validate_voice
    voice_type = validate_voice(voice_type)
    if not isinstance(text, str) or not 1 <= len(text) <= 1000 or text != text.strip():
        raise ValueError("Invalid narration text")
    appid = str(load_key("doubao_tts.appid") or "").strip()
    access_token = str(load_key("doubao_tts.access_token") or "").strip()
    if not appid or not access_token:
        raise ValueError("Existing Doubao connection unavailable")
    return _doubao_v3_stream_call(
        appid=appid, access_token=access_token, text=text, save_path=save_path,
        voice_type=voice_type, resource_id="seed-tts-2.0", single_request=True,
        context_texts=[get_doubao_style_prompt()] if get_doubao_style_prompt() else None,
    )
