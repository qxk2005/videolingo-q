"""Read-only, credential-safe TTS quota probes. Never synthesize or change billing."""
from datetime import datetime, timezone
from pathlib import Path
import os

OFFICIAL_BASE = "https://api.elevenlabs.io"
SUBSCRIPTION_URL = OFFICIAL_BASE + "/v1/user/subscription"
PROVIDERS = ("elevenlabs", "doubao", "edge")


def _config():
    # Do not import config_utils: its import can initialize config and alter proxy state.
    from ruamel.yaml import YAML
    root = os.environ.get("VIDEOLINGO_ROOT")
    if not root or not Path(root).is_dir():
        root = Path(__file__).resolve().parents[2]
    with (Path(root) / "config.yaml").open(encoding="utf-8") as stream:
        value = YAML(typ="safe").load(stream)
    if not isinstance(value, dict):
        raise ValueError("Invalid configuration")
    return value


def _base(provider, status, note):
    return {"provider": provider, "status": status, "note": note,
            "quota": None, "balance": None}


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _elevenlabs(config, get=None):
    settings = config.get("elevenlabs_tts") or {}
    if not isinstance(settings, dict):
        return _base("elevenlabs", "configuration_error", "配音配置格式无效。")
    # Never forward the existing key to a different endpoint or follow redirects.
    base = settings.get("base_url") or OFFICIAL_BASE
    if not isinstance(base, str) or base.strip().rstrip("/") != OFFICIAL_BASE:
        return _base("elevenlabs", "unsupported_endpoint", "当前使用自定义服务地址，未向任何地址发送密钥；暂不支持该服务的余额查询。")
    key = settings.get("api_key")
    if not isinstance(key, str) or not key.strip() or key.strip().startswith("YOUR_"):
        return _base("elevenlabs", "not_configured", "未配置 ElevenLabs 密钥。")
    try:
        import requests
        transport = get or requests.get
        proxy = config.get("proxy")
        kwargs = {}
        if isinstance(proxy, str) and proxy.strip():
            kwargs["proxies"] = {"http": proxy.strip(), "https": proxy.strip()}
        response = transport(SUBSCRIPTION_URL, headers={"xi-api-key": key.strip()},
                             timeout=(5, 12), allow_redirects=False, **kwargs)
        try:
            code = response.status_code
            if code != 200:
                state, note = {
                    401: ("authentication_failed", "现有密钥被拒绝；未更换密钥。"),
                    403: ("permission_denied", "现有密钥无权读取订阅；如需继续，请用户单独授权 User Read 权限，未自动扩权。"),
                    429: ("rate_limited", "查询受到限流；未重试。"),
                }.get(code, ("upstream_error", "官方订阅查询失败；未重试，也未返回原始响应。"))
                if code in (401, 403):
                    # ElevenLabs may use HTTP 401 for missing scoped permissions.
                    # Inspect only a known error identifier, never echo its body.
                    try:
                        error = response.json()
                        detail = error.get("detail") if isinstance(error, dict) else None
                        if isinstance(detail, dict) and detail.get("status") == "missing_permissions":
                            state = "permission_denied"
                            note = "现有密钥缺少读取订阅权限；需要用户单独授权 User Read，未自动扩权。"
                    except Exception:
                        pass
                result = _base("elevenlabs", state, note)
                result["http_status"] = code
                return result
            data = response.json()
        finally:
            response.close()
    except Exception as exc:
        # Exceptions can include URLs, proxy credentials, headers or raw bodies.
        result = _base("elevenlabs", "request_failed", "额度查询网络或响应异常；未输出原始错误，未重试。")
        categories = {"ProxyError": "proxy_error", "SSLError": "tls_error",
                      "ConnectTimeout": "timeout", "ReadTimeout": "timeout",
                      "ConnectionError": "connection_error", "JSONDecodeError": "invalid_json",
                      "InvalidSchema": "transport_configuration_error"}
        result["error_category"] = categories.get(type(exc).__name__, "network_or_response_error")
        return result
    if (not isinstance(data, dict) or not _nonnegative_int(data.get("character_count"))
            or not _nonnegative_int(data.get("character_limit"))):
        return _base("elevenlabs", "invalid_response", "官方响应缺少有效额度数据；未知不能当作零。")
    used, limit = data["character_count"], data["character_limit"]
    result = _base("elevenlabs", "ok", "账号/工作区订阅额度，不是某个模型或 API Key 的独立配额；按官方 character 字段显示，不换算现金或可合成时长。")
    result["quota"] = {"unit": "provider_character_units", "used": used, "limit": limit,
                       "remaining": max(0, limit - used), "over_limit": max(0, used - limit)}
    result["balance_status"] = "not_exposed_by_subscription_endpoint"
    reset = data.get("next_character_count_reset_unix")
    result["resets_at_unix"] = reset if _nonnegative_int(reset) else None
    extension = data.get("max_credit_limit_extension")
    result["max_credit_limit_extension"] = extension if _nonnegative_int(extension) or extension == "unlimited" else None
    result["can_extend_character_limit"] = data.get("can_extend_character_limit") if type(data.get("can_extend_character_limit")) is bool else None
    result["source"] = SUBSCRIPTION_URL
    return result


def get_tts_quota(provider="all", *, config_loader=None, http_get=None):
    """Return only allowlisted counters; no configuration, identity or raw errors."""
    if not isinstance(provider, str) or provider not in (*PROVIDERS, "all"):
        raise ValueError("provider must be all, elevenlabs, doubao or edge")
    selected = PROVIDERS if provider == "all" else (provider,)
    results = []
    for item in selected:
        if item == "elevenlabs":
            try:
                config = (config_loader or _config)()
                result = _elevenlabs(config, http_get)
            except Exception:
                result = _base(item, "configuration_error", "无法安全读取本地配音配置；未返回配置内容。")
        elif item == "doubao":
            result = _base(item, "requires_separate_billing_credentials", "当前豆包配音 AppID/Access Token 不等于火山 OpenAPI 账务凭据。ResourcePacksStatus 与 QueryBalanceAcct 需另行配置并授权最小只读访问；本工具未读取额外凭据或调用这些接口。")
        else:
            result = _base(item, "not_applicable", "当前 Edge TTS 集成没有账户余额/额度查询接口；不代表无限额度。")
        results.append(result)
    return {"read_only": True, "checked_at": datetime.now(timezone.utc).isoformat(),
            "cached": False, "providers": results,
            "warning": "仅查询，不代表允许产生费用；额度可能因并发任务变化，未知额度不会按零或无限处理。"}
