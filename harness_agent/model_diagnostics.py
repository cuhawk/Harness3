"""LiteLLM/OpenAI-compatible model diagnostics.

This module intentionally stays outside AgentRuntime and tool-calling. Model
health checks should keep working when the active chat model or tool binding is
broken, and tests should be able to inject a fake HTTP session without touching
the network.
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests

from . import core


CONTEXT_KEYS = (
    "context_window",
    "max_context_tokens",
    "max_input_tokens",
    "max_prompt_tokens",
    "max_total_tokens",
    "input_token_limit",
    "max_tokens",
)

RATE_LIMIT_HEADERS = {
    "x-ratelimit-limit-requests": "limit_requests",
    "x-ratelimit-remaining-requests": "remaining_requests",
    "x-ratelimit-reset-requests": "reset_requests",
    "x-ratelimit-limit-tokens": "limit_tokens",
    "x-ratelimit-remaining-tokens": "remaining_tokens",
    "x-ratelimit-reset-tokens": "reset_tokens",
    "ratelimit-limit": "limit",
    "ratelimit-remaining": "remaining",
    "ratelimit-reset": "reset",
    "retry-after": "retry_after",
}


def _http_session(session: Any | None = None) -> Any:
    return session or requests


def proxy_url(path: str) -> str:
    return core.LITELLM_BASE_URL.rstrip("/") + "/" + path.lstrip("/")


def auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if core.LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {core.LITELLM_API_KEY}"
    return headers


def mask_secret(value: str) -> str:
    value = str(value or "")
    if not value:
        return ""
    if len(value) <= 8:
        return value[0] + "*" * max(0, len(value) - 2) + value[-1]
    return value[:4] + "*" * max(4, len(value) - 7) + value[-3:]


def api_key_summary() -> dict[str, Any]:
    return {
        "present": bool(core.LITELLM_API_KEY),
        "masked": mask_secret(core.LITELLM_API_KEY),
    }


def safe_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return None


def response_error(response: Any) -> str:
    payload = safe_json(response)
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        return core.trim_text(json.dumps(payload, ensure_ascii=False), 1000)
    return core.trim_text(getattr(response, "text", "") or "", 1000)


def extract_rate_limit_headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    normalized = {str(k).lower(): str(v) for k, v in dict(headers or {}).items()}
    for header, name in RATE_LIMIT_HEADERS.items():
        if header in normalized:
            out[name] = normalized[header]
    return out


def extract_model_names(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        names = []
        for item in data:
            if isinstance(item, dict):
                name = item.get("id") or item.get("model_name") or item.get("model")
                if name:
                    names.append(str(name))
            elif item:
                names.append(str(item))
        return sorted(set(names))
    if isinstance(data, dict):
        return sorted(str(k) for k in data.keys())
    return []


def list_proxy_models(session: Any | None = None, timeout_sec: int = 15) -> dict[str, Any]:
    if not core.LITELLM_API_KEY:
        return {"ok": False, "status_code": None, "models": [], "error": "LITELLM_API_KEY is not set"}
    try:
        response = _http_session(session).get(proxy_url("/models"), headers=auth_headers(), timeout=timeout_sec)
        payload = safe_json(response)
        ok = 200 <= int(response.status_code) < 300
        return {
            "ok": ok,
            "status_code": int(response.status_code),
            "models": extract_model_names(payload),
            "rate_limits": extract_rate_limit_headers(getattr(response, "headers", {})),
            "error": "" if ok else response_error(response),
        }
    except Exception as exc:
        return {"ok": False, "status_code": None, "models": [], "error": repr(exc)}


def _model_name_from_info(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("id")
        or item.get("model_name")
        or item.get("model")
        or (item.get("litellm_params") or {}).get("model")
        or ""
    )


def _find_context_window(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in CONTEXT_KEYS:
            raw = value.get(key)
            if isinstance(raw, (int, float)) and raw > 0:
                return int(raw)
            if isinstance(raw, str) and raw.isdigit():
                return int(raw)
        for child in value.values():
            found = _find_context_window(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_context_window(child)
            if found:
                return found
    return None


def fetch_model_info(session: Any | None = None, timeout_sec: int = 15) -> dict[str, Any]:
    if not core.LITELLM_API_KEY:
        return {"ok": False, "models": {}, "error": "LITELLM_API_KEY is not set"}
    try:
        response = _http_session(session).get(proxy_url("/model/info"), headers=auth_headers(), timeout=timeout_sec)
        payload = safe_json(response)
        ok = 200 <= int(response.status_code) < 300
        models: dict[str, dict[str, Any]] = {}
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            iterable = data.values()
        elif isinstance(data, list):
            iterable = data
        else:
            iterable = []
        for item in iterable:
            name = _model_name_from_info(item)
            if not name:
                continue
            models[name] = {
                "raw": item,
                "context_window": _find_context_window(item),
            }
        return {
            "ok": ok,
            "status_code": int(response.status_code),
            "models": models,
            "rate_limits": extract_rate_limit_headers(getattr(response, "headers", {})),
            "error": "" if ok else response_error(response),
        }
    except Exception as exc:
        return {"ok": False, "models": {}, "error": repr(exc)}


def estimate_text_tokens(text: Any) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    return max(1, (len(raw) + 3) // 4)


def estimate_history_tokens(history: list[Any]) -> int:
    total = 0
    for message in history or []:
        total += 4
        total += estimate_text_tokens(getattr(message, "content", ""))
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            total += estimate_text_tokens(json.dumps(tool_calls, ensure_ascii=False))
        kwargs = getattr(message, "additional_kwargs", None)
        if kwargs:
            total += estimate_text_tokens(json.dumps(kwargs, ensure_ascii=False, default=str))
    return total


def build_model_status(model_name: str, history: list[Any], model_info: dict[str, Any]) -> dict[str, Any]:
    info = model_info.get(model_name, {})
    context_window = info.get("context_window")
    used = estimate_history_tokens(history)
    remaining = int(context_window) - used if context_window else None
    return {
        "model": model_name,
        "context_window": context_window,
        "context_source": "model_info" if context_window else "unknown",
        "estimated_used_tokens": used,
        "estimated_remaining_tokens": max(0, remaining) if remaining is not None else None,
        "estimate_method": "chars_div_4_plus_message_overhead",
    }


def _fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _fmt_bool(value: Any) -> str:
    return "ok" if bool(value) else "error"


def _format_key_values(values: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in values.items()) if values else "unknown"


def _plain_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def format_model_status_table(payload_json: str) -> str:
    """Render model_status_report JSON as a human-readable plain-text table."""
    try:
        payload = json.loads(payload_json)
    except Exception:
        return payload_json
    if not isinstance(payload, dict):
        return payload_json

    proxy = payload.get("proxy") if isinstance(payload.get("proxy"), dict) else {}
    api_key = payload.get("api_key") if isinstance(payload.get("api_key"), dict) else {}
    rate_limits = proxy.get("rate_limits") if isinstance(proxy.get("rate_limits"), dict) else {}
    models_rate = rate_limits.get("models") if isinstance(rate_limits.get("models"), dict) else {}
    info_rate = rate_limits.get("model_info") if isinstance(rate_limits.get("model_info"), dict) else {}

    lines = [
        "Model Status",
        f"Proxy: {proxy.get('base_url', 'unknown')}   Models: {_fmt_bool(proxy.get('models_ok'))}   Model info: {_fmt_bool(proxy.get('model_info_ok'))}",
        f"API key: {api_key.get('masked') or 'not set'}",
    ]
    if models_rate or info_rate:
        if models_rate:
            lines.append(f"Rate limits: models {_format_key_values(models_rate)}")
        if info_rate:
            lines.append(f"Rate limits: model_info {_format_key_values(info_rate)}")

    rows = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        rows.append([
            str(item.get("model") or ""),
            _fmt_int(item.get("context_window")),
            _fmt_int(item.get("estimated_used_tokens")),
            _fmt_int(item.get("estimated_remaining_tokens")),
            str(item.get("context_source") or "unknown"),
        ])
    lines.append("")
    if rows:
        lines.append(_plain_table(["Model", "Ctx", "Used", "Left", "Source"], rows))
    else:
        lines.append("No model status rows returned.")

    errors = payload.get("errors") if isinstance(payload.get("errors"), dict) else {}
    error_lines = [f"{key}: {value}" for key, value in errors.items() if value]
    if error_lines:
        lines.extend(["", "Errors:", *error_lines])
    return "\n".join(lines)


def model_status_report(active_model: str, history: list[Any], all_models: bool = False, session: Any | None = None) -> str:
    models_result = list_proxy_models(session=session)
    info_result = fetch_model_info(session=session)
    model_names = models_result.get("models") or []
    targets = model_names if all_models else [active_model]
    statuses = [
        build_model_status(model, history, info_result.get("models", {}))
        for model in targets
        if model
    ]
    ok = bool(models_result.get("ok")) and bool(statuses)
    return core.to_json({
        "ok": ok,
        "proxy": {
            "base_url": core.LITELLM_BASE_URL,
            "models_ok": models_result.get("ok"),
            "model_info_ok": info_result.get("ok"),
            "rate_limits": {
                "models": models_result.get("rate_limits", {}),
                "model_info": info_result.get("rate_limits", {}),
            },
        },
        "api_key": api_key_summary(),
        "active_model": active_model,
        "count": len(statuses),
        "models": statuses,
        "errors": {
            "models": models_result.get("error") or "",
            "model_info": info_result.get("error") or "",
        },
    })


def smoke_test_model(model_name: str, session: Any | None = None, prompt: str = "hi", max_tokens: int = 4, timeout_sec: int = 30) -> dict[str, Any]:
    if not core.LITELLM_API_KEY:
        return {"ok": False, "model": model_name, "status_code": None, "error": "LITELLM_API_KEY is not set"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        response = _http_session(session).post(proxy_url("/chat/completions"), headers=auth_headers(), json=payload, timeout=timeout_sec)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        body = safe_json(response)
        ok = 200 <= int(response.status_code) < 300
        output = ""
        usage = {}
        if isinstance(body, dict):
            choices = body.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                output = str(message.get("content") or choices[0].get("text") or "")
            usage = body.get("usage") or {}
        return {
            "ok": ok,
            "model": model_name,
            "status_code": int(response.status_code),
            "latency_ms": latency_ms,
            "output": core.trim_text(output, 200),
            "usage": usage,
            "rate_limits": extract_rate_limit_headers(getattr(response, "headers", {})),
            "error": "" if ok else response_error(response),
        }
    except Exception as exc:
        return {"ok": False, "model": model_name, "status_code": None, "error": repr(exc)}


def model_test_report(active_model: str, all_models: bool = False, session: Any | None = None) -> str:
    targets = [active_model]
    models_result = None
    if all_models:
        models_result = list_proxy_models(session=session)
        targets = models_result.get("models") or []
    results = [smoke_test_model(model, session=session) for model in targets if model]
    return core.to_json({
        "ok": bool(results) and all(item.get("ok") for item in results),
        "proxy": {"base_url": core.LITELLM_BASE_URL},
        "api_key": api_key_summary(),
        "count": len(results),
        "results": results,
        "errors": {"models": (models_result or {}).get("error", "")},
    })
