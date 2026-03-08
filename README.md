# opsmeter-sdk (Preview)

Python SDK preview for Opsmeter auto-instrumentation.

## Install

```bash
pip install opsmeter-sdk
```

## Core model

- `init(...)` once at process startup (idempotent)
- request-level attribution via `context(...)`
- provider call stays direct (no proxy)
- telemetry emit is async and non-blocking by default

## Quickstart

```python
import opsmeter_sdk as opsmeter
from openai import OpenAI

opsmeter.init(
    api_key="...",
    workspace_id="ws_123",
    environment="prod",
)

client = OpenAI()

with opsmeter.context(
    user_id="u_1",
    tenant_id="tenant_a",
    endpoint="/api/chat",
    feature="assistant",
    prompt_version="v12",
):
    response = opsmeter.capture_openai_chat_completion(
        lambda: client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hello"}]),
        request={"model": "gpt-4o-mini"},
    )
```

## Show Opsmeter ingest result

```python
captured = opsmeter.capture_openai_chat_completion_with_result(
    lambda: client.chat.completions.create(**request),
    request=request,
    await_telemetry_response=True,
)

print(captured["telemetry"])  # { ok, status, body? }
```

## API

- `init(...)`
- `context(...)`
- `get_context()`
- `capture_openai_chat_completion(...)`
- `capture_openai_chat_completion_with_result(...)`
- `capture_openai_chat_completion_async(...)`
- `patch_openai_client(...)`
- `flush()`

## Tests

```bash
python3 -m py_compile opsmeter_sdk/sdk.py tests/test_sdk.py
python3 -m unittest tests/test_sdk.py -v
```
