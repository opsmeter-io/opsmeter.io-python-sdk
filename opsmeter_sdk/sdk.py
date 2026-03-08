from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional

VALID_ENVIRONMENTS = {"prod", "staging", "dev"}
VALID_DATA_MODES = {"real", "test", "demo"}


@dataclass
class _Config:
    api_key: str
    workspace_id: Optional[str]
    telemetry_base_url: str
    environment: str
    enabled: bool
    flush_interval_ms: int
    max_batch_size: int
    request_timeout_seconds: float
    dedupe_window_seconds: float
    max_retries: int
    debug: bool
    on_telemetry_result: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]]


_context_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("opsmeter_ctx", default={})
_initialized = False
_config: Optional[_Config] = None
_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
_worker_stop = threading.Event()
_worker_thread: Optional[threading.Thread] = None
_dedupe: Dict[str, float] = {}
_lock = threading.Lock()


class InitResult(dict):
    pass


def _normalize_environment(value: Optional[str]) -> str:
    normalized = (value or "prod").strip().lower()
    if normalized not in VALID_ENVIRONMENTS:
        raise ValueError("environment must be prod|staging|dev")
    return normalized


def _normalize_data_mode(value: Optional[str]) -> str:
    normalized = (value or "real").strip().lower()
    if normalized not in VALID_DATA_MODES:
        raise ValueError("data_mode must be real|test|demo")
    return normalized


def _normalize_provider(value: Optional[str]) -> str:
    normalized = (value or "openai").strip().lower()
    if normalized in {"openai", "anthropic"}:
        return normalized
    return "openai"


def _read_int(config: Dict[str, Any], key: str, fallback: int, *, min_value: int = 0) -> int:
    raw = config.get(key, None)
    value = fallback if raw is None else int(raw)
    if value < min_value:
        raise ValueError(f"{key} must be >= {min_value}")
    return value


def _read_float(config: Dict[str, Any], key: str, fallback: float, *, min_value: float = 0.0) -> float:
    raw = config.get(key, None)
    value = fallback if raw is None else float(raw)
    if value < min_value:
        raise ValueError(f"{key} must be >= {min_value}")
    return value


def _normalize_init(config: Dict[str, Any]) -> _Config:
    if not isinstance(config, dict):
        raise ValueError("init config must be a dict")

    forbidden = {"user_id", "tenant_id", "endpoint", "feature", "prompt_version", "external_request_id"}
    present = sorted(field for field in forbidden if field in config)
    if present:
        raise ValueError(f"request-level fields are not allowed in init(): {', '.join(present)}")

    enabled = bool(config.get("enabled", True))
    api_key = str(config.get("api_key", "")).strip()
    if enabled and not api_key:
        raise ValueError("api_key is required when SDK is enabled")

    callback = config.get("on_telemetry_result")
    if callback is not None and not callable(callback):
        raise ValueError("on_telemetry_result must be callable")

    return _Config(
        api_key=api_key,
        workspace_id=str(config["workspace_id"]).strip() if config.get("workspace_id") else None,
        telemetry_base_url=str(config.get("telemetry_base_url", "https://api.opsmeter.io")).rstrip("/"),
        environment=_normalize_environment(config.get("environment", "prod")),
        enabled=enabled,
        flush_interval_ms=_read_int(config, "flush_interval_ms", 1000, min_value=1),
        max_batch_size=_read_int(config, "max_batch_size", 50, min_value=1),
        request_timeout_seconds=_read_float(config, "request_timeout_seconds", 0.6, min_value=0.001),
        dedupe_window_seconds=_read_float(config, "dedupe_window_seconds", 300.0, min_value=0.001),
        max_retries=_read_int(config, "max_retries", 2, min_value=0),
        debug=bool(config.get("debug", False)),
        on_telemetry_result=callback,
    )


def _same_config(a: _Config, b: _Config) -> bool:
    return a == b


def _log_debug(*parts: Any) -> None:
    if _config and _config.debug:
        print("[opsmeter.io-sdk]", *parts)


def _emit_result(result: Dict[str, Any], payload: Dict[str, Any]) -> None:
    if not _config or not _config.on_telemetry_result:
        return

    try:
        _config.on_telemetry_result(result, payload)
    except Exception:
        # callback failures should never break runtime
        pass


def _start_worker() -> None:
    global _worker_thread

    if _worker_thread and _worker_thread.is_alive():
        return

    _worker_stop.clear()
    _worker_thread = threading.Thread(target=_worker_loop, name="opsmeter.io-sdk-worker", daemon=True)
    _worker_thread.start()


def init(**kwargs: Any) -> InitResult:
    global _initialized, _config

    normalized = _normalize_init(kwargs)

    with _lock:
        if not _initialized:
            _config = normalized
            _initialized = True
            _start_worker()
            return InitResult(did_init=True, initialized=True, warning=None)

        assert _config is not None
        if _same_config(_config, normalized):
            return InitResult(did_init=False, initialized=True, warning=None)

        return InitResult(
            did_init=False,
            initialized=True,
            warning="SDK already initialized with a different config. First init config is kept.",
        )


def get_context() -> Dict[str, Any]:
    return dict(_context_var.get())


def _normalize_context(values: Dict[str, Any]) -> Dict[str, Any]:
    metadata = values.get("metadata") if isinstance(values.get("metadata"), dict) else None
    if metadata is not None:
        metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}

    normalized: Dict[str, Any] = {
        "user_id": str(values["user_id"]) if values.get("user_id") else None,
        "tenant_id": str(values["tenant_id"]) if values.get("tenant_id") else None,
        "endpoint": str(values["endpoint"]) if values.get("endpoint") else None,
        "feature": str(values["feature"]) if values.get("feature") else None,
        "prompt_version": str(values["prompt_version"]) if values.get("prompt_version") else None,
        "external_request_id": str(values["external_request_id"]) if values.get("external_request_id") else None,
        "data_mode": _normalize_data_mode(values.get("data_mode")) if values.get("data_mode") else None,
        "metadata": metadata,
    }
    return {k: v for k, v in normalized.items() if v is not None}


@contextlib.contextmanager
def context(**kwargs: Any) -> Iterator[None]:
    parent = get_context()
    merged = {**parent, **_normalize_context(kwargs)}
    token = _context_var.set(merged)
    try:
        yield
    finally:
        _context_var.reset(token)


def _generate_external_request_id(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return f"ext_{digest}"


def _extract_usage(response: Optional[Dict[str, Any]]) -> Dict[str, int]:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _build_payload(
    *,
    provider: str,
    operation: str,
    request: Optional[Dict[str, Any]],
    response: Optional[Dict[str, Any]],
    error: Optional[BaseException],
    latency_ms: int,
    external_request_id: str,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    usage = _extract_usage(response)
    safe_provider = _normalize_provider(provider)

    return {
        "externalRequestId": external_request_id,
        "provider": safe_provider,
        "model": (response or {}).get("model") or (request or {}).get("model") or "unknown",
        "promptVersion": ctx.get("prompt_version", "unknown"),
        "endpointTag": ctx.get("feature") or ctx.get("endpoint") or "sdk.unknown",
        "inputTokens": usage["input_tokens"],
        "outputTokens": usage["output_tokens"],
        "totalTokens": usage["total_tokens"],
        "latencyMs": int(latency_ms),
        "status": "error" if error else "success",
        "errorCode": (getattr(error, "code", None) or error.__class__.__name__) if error else None,
        "userId": ctx.get("user_id"),
        "dataMode": _normalize_data_mode(ctx.get("data_mode") or "real"),
        "environment": _config.environment if _config else "prod",
        "metadata": {
            "operation": operation,
            "tenantId": ctx.get("tenant_id"),
            "endpoint": ctx.get("endpoint"),
            "feature": ctx.get("feature"),
            "providerRequestId": (response or {}).get("id"),
            "sdkLanguage": "python",
        },
    }


def _dedupe_key(payload: Dict[str, Any]) -> str:
    operation = ((payload.get("metadata") or {}).get("operation") or "unknown")
    return f"{payload.get('externalRequestId')}:{payload.get('provider')}:{operation}"


def _sweep_dedupe(now_ts: float) -> None:
    if not _config:
        return

    cutoff = now_ts - _config.dedupe_window_seconds
    for key, ts in list(_dedupe.items()):
        if ts < cutoff:
            _dedupe.pop(key, None)


def _enqueue(payload: Dict[str, Any]) -> bool:
    if not _initialized or not _config or not _config.enabled:
        return False

    now_ts = time.time()
    _sweep_dedupe(now_ts)

    key = _dedupe_key(payload)
    if key in _dedupe:
        _log_debug("dedupe dropped", key)
        return False

    _dedupe[key] = now_ts
    _queue.put({"payload": payload, "attempt": 0})
    return True


def _post_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _config or not _config.enabled:
        return {"ok": True, "status": 204, "skipped": "disabled"}

    endpoint = f"{_config.telemetry_base_url}/v1/ingest/llm-request"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-API-Key": _config.api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_config.request_timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            body = json.loads(raw) if raw else None
            return {"ok": 200 <= response.status < 300, "status": response.status, "body": body}
    except urllib.error.HTTPError as error:
        body_text = error.read().decode("utf-8") if error.fp else ""
        try:
            body = json.loads(body_text) if body_text else None
        except json.JSONDecodeError:
            body = {"raw": body_text}
        return {"ok": False, "status": error.code, "body": body}
    except Exception as error:  # swallow to keep business path safe
        return {"ok": False, "status": 0, "error": str(error)}


def _send_item(item: Dict[str, Any]) -> None:
    result = _post_payload(item["payload"])
    _emit_result(result, item["payload"])

    if result.get("ok"):
        return

    if not _config:
        return

    attempt = int(item.get("attempt", 0))
    if attempt >= _config.max_retries:
        _log_debug("drop after retries", result)
        return

    next_attempt = attempt + 1
    backoff = 0.2 * (2**next_attempt)

    def _requeue() -> None:
        _queue.put({"payload": item["payload"], "attempt": next_attempt})

    timer = threading.Timer(backoff, _requeue)
    timer.daemon = True
    timer.start()


def _worker_loop() -> None:
    while not _worker_stop.is_set():
        if not _config:
            time.sleep(0.05)
            continue

        batch = []
        try:
            item = _queue.get(timeout=_config.flush_interval_ms / 1000.0)
            batch.append(item)
            while len(batch) < _config.max_batch_size:
                batch.append(_queue.get_nowait())
        except queue.Empty:
            pass

        for item in batch:
            _send_item(item)


def flush(timeout_seconds: float = 5.0) -> None:
    start = time.time()
    while not _queue.empty() and (time.time() - start) < timeout_seconds:
        time.sleep(0.01)


def _build_capture_payload(
    *,
    provider: str,
    request: Optional[Dict[str, Any]],
    operation: str,
    external_request_id: Optional[str],
) -> Dict[str, Any]:
    ctx = get_context()
    safe_provider = _normalize_provider(provider)
    generated_ext = external_request_id or ctx.get("external_request_id") or _generate_external_request_id(
        f"{safe_provider}:{operation}:{(request or {}).get('model', 'unknown')}:{time.time()}"
    )
    return {"ctx": ctx, "external_request_id": generated_ext}


def _handle_emit(payload: Dict[str, Any], await_telemetry_response: bool) -> Dict[str, Any]:
    if await_telemetry_response:
        result = _post_payload(payload)
        _emit_result(result, payload)
        return result

    queued = _enqueue(payload)
    return {"ok": True, "status": 202 if queued else 204, "queued": queued}


def capture_openai_chat_completion(
    call: Callable[[], Any],
    *,
    request: Optional[Dict[str, Any]] = None,
    operation: str = "chat.completions.create",
    external_request_id: Optional[str] = None,
    provider: str = "openai",
) -> Any:
    if not callable(call):
        raise ValueError("call must be callable")

    start = time.time()
    capture = _build_capture_payload(
        provider=provider,
        request=request,
        operation=operation,
        external_request_id=external_request_id,
    )

    try:
        response_obj = call()
        response = response_obj.model_dump() if hasattr(response_obj, "model_dump") else response_obj
        payload = _build_payload(
            provider=provider,
            operation=operation,
            request=request,
            response=response if isinstance(response, dict) else None,
            error=None,
            latency_ms=int((time.time() - start) * 1000),
            external_request_id=capture["external_request_id"],
            ctx=capture["ctx"],
        )
        _handle_emit(payload, await_telemetry_response=False)
        return response_obj
    except BaseException as error:
        payload = _build_payload(
            provider=provider,
            operation=operation,
            request=request,
            response=None,
            error=error,
            latency_ms=int((time.time() - start) * 1000),
            external_request_id=capture["external_request_id"],
            ctx=capture["ctx"],
        )
        _handle_emit(payload, await_telemetry_response=False)
        raise


def capture_openai_chat_completion_with_result(
    call: Callable[[], Any],
    *,
    request: Optional[Dict[str, Any]] = None,
    operation: str = "chat.completions.create",
    external_request_id: Optional[str] = None,
    await_telemetry_response: bool = False,
    provider: str = "openai",
) -> Dict[str, Any]:
    if not callable(call):
        raise ValueError("call must be callable")

    start = time.time()
    capture = _build_capture_payload(
        provider=provider,
        request=request,
        operation=operation,
        external_request_id=external_request_id,
    )

    try:
        response_obj = call()
        response = response_obj.model_dump() if hasattr(response_obj, "model_dump") else response_obj
        payload = _build_payload(
            provider=provider,
            operation=operation,
            request=request,
            response=response if isinstance(response, dict) else None,
            error=None,
            latency_ms=int((time.time() - start) * 1000),
            external_request_id=capture["external_request_id"],
            ctx=capture["ctx"],
        )
        telemetry = _handle_emit(payload, await_telemetry_response=await_telemetry_response)
        return {
            "provider_response": response_obj,
            "telemetry": telemetry,
            "external_request_id": capture["external_request_id"],
            "payload": payload,
        }
    except BaseException as error:
        payload = _build_payload(
            provider=provider,
            operation=operation,
            request=request,
            response=None,
            error=error,
            latency_ms=int((time.time() - start) * 1000),
            external_request_id=capture["external_request_id"],
            ctx=capture["ctx"],
        )
        _handle_emit(payload, await_telemetry_response=await_telemetry_response)
        raise


async def capture_openai_chat_completion_async(
    call: Callable[[], Any],
    *,
    request: Optional[Dict[str, Any]] = None,
    operation: str = "chat.completions.create",
    external_request_id: Optional[str] = None,
    await_telemetry_response: bool = False,
    provider: str = "openai",
) -> Dict[str, Any]:
    if not callable(call):
        raise ValueError("call must be callable")

    start = time.time()
    capture = _build_capture_payload(
        provider=provider,
        request=request,
        operation=operation,
        external_request_id=external_request_id,
    )

    try:
        response_obj = await call()
        response = response_obj.model_dump() if hasattr(response_obj, "model_dump") else response_obj
        payload = _build_payload(
            provider=provider,
            operation=operation,
            request=request,
            response=response if isinstance(response, dict) else None,
            error=None,
            latency_ms=int((time.time() - start) * 1000),
            external_request_id=capture["external_request_id"],
            ctx=capture["ctx"],
        )

        if await_telemetry_response:
            telemetry = await asyncio.to_thread(_post_payload, payload)
            _emit_result(telemetry, payload)
        else:
            queued = _enqueue(payload)
            telemetry = {"ok": True, "status": 202 if queued else 204, "queued": queued}

        return {
            "provider_response": response_obj,
            "telemetry": telemetry,
            "external_request_id": capture["external_request_id"],
            "payload": payload,
        }
    except BaseException as error:
        payload = _build_payload(
            provider=provider,
            operation=operation,
            request=request,
            response=None,
            error=error,
            latency_ms=int((time.time() - start) * 1000),
            external_request_id=capture["external_request_id"],
            ctx=capture["ctx"],
        )
        if await_telemetry_response:
            telemetry = await asyncio.to_thread(_post_payload, payload)
            _emit_result(telemetry, payload)
        else:
            _enqueue(payload)
        raise


def patch_openai_client(client: Any, *, operation: str = "chat.completions.create") -> Dict[str, Any]:
    create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
    if not callable(create):
        raise ValueError("OpenAI client chat.completions.create function not found")

    if getattr(create, "__opsmeter_patched__", False):
        return {"patched": False, "reason": "already_patched"}

    def _patched(request: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return capture_openai_chat_completion(
            lambda: create(request, *args, **kwargs),
            request=request,
            operation=operation,
        )

    _patched.__opsmeter_patched__ = True
    client.chat.completions.create = _patched
    return {"patched": True}


def capture_anthropic_message(
    call: Callable[[], Any],
    *,
    request: Optional[Dict[str, Any]] = None,
    operation: str = "messages.create",
    external_request_id: Optional[str] = None,
) -> Any:
    return capture_openai_chat_completion(
        call,
        request=request,
        operation=operation,
        external_request_id=external_request_id,
        provider="anthropic",
    )


def capture_anthropic_message_with_result(
    call: Callable[[], Any],
    *,
    request: Optional[Dict[str, Any]] = None,
    operation: str = "messages.create",
    external_request_id: Optional[str] = None,
    await_telemetry_response: bool = False,
) -> Dict[str, Any]:
    return capture_openai_chat_completion_with_result(
        call,
        request=request,
        operation=operation,
        external_request_id=external_request_id,
        await_telemetry_response=await_telemetry_response,
        provider="anthropic",
    )


async def capture_anthropic_message_async(
    call: Callable[[], Any],
    *,
    request: Optional[Dict[str, Any]] = None,
    operation: str = "messages.create",
    external_request_id: Optional[str] = None,
    await_telemetry_response: bool = False,
) -> Dict[str, Any]:
    return await capture_openai_chat_completion_async(
        call,
        request=request,
        operation=operation,
        external_request_id=external_request_id,
        await_telemetry_response=await_telemetry_response,
        provider="anthropic",
    )


def reset_for_tests() -> None:
    global _initialized, _config, _worker_thread

    _worker_stop.set()
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=0.2)

    _worker_thread = None
    _initialized = False
    _config = None
    _worker_stop.clear()

    while not _queue.empty():
        try:
            _queue.get_nowait()
        except queue.Empty:
            break

    _dedupe.clear()
    _context_var.set({})
