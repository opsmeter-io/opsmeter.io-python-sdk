# opsmeter-sdk (Preview)

Python SDK preview for Opsmeter auto-instrumentation.
Package publishing: coming soon
Integration examples: [opsmeter-integration-examples](https://github.com/opsmeter/opsmeter-integration-examples)

Provider/model names should come from: [https://opsmeter.io/docs/catalog](https://opsmeter.io/docs/catalog)
Current SDK provider support: **OpenAI** and **Anthropic** only.

## Install

```bash
pip install opsmeter-sdk openai
# optional:
# pip install anthropic
```

## Core model

- `init(...)` once at process startup (idempotent)
- request-level attribution via `context(...)`
- provider call stays direct (no proxy)
- telemetry emit is async and non-blocking by default

## Telemetry usage (no options)

```python
import opsmeter_sdk as opsmeter
from openai import OpenAI

opsmeter.init(
    api_key="...",
    environment="prod",
)

client = OpenAI()

opsmeter.capture_openai_chat_completion(
    lambda: client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
)
```

## Telemetry usage (with options/context)

```python
import opsmeter_sdk as opsmeter
from openai import OpenAI
from anthropic import Anthropic

opsmeter.init(
    api_key="...",
    workspace_id="ws_123",
    environment="prod",
)

openai = OpenAI()
anthropic = Anthropic()

with opsmeter.context(
    user_id="u_1",
    tenant_id="tenant_a",
    endpoint="/api/chat",
    feature="assistant",
    prompt_version="v12",
    data_mode="real",
):
    openai_captured = opsmeter.capture_openai_chat_completion_with_result(
        lambda: openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        ),
        request={"model": "gpt-4o-mini"},
        await_telemetry_response=True,
    )

with opsmeter.context(
    user_id="u_1",
    tenant_id="tenant_a",
    endpoint="/api/support",
    feature="support",
    prompt_version="v8",
):
    anthropic_captured = opsmeter.capture_anthropic_message_with_result(
        # Provider/model names: https://opsmeter.io/docs/catalog
        lambda: anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=128,
            messages=[{"role": "user", "content": "Summarize this support ticket."}],
        ),
        request={"model": "claude-3-5-sonnet-20241022"},
        await_telemetry_response=True,
    )

print(openai_captured["telemetry"])     # { ok, status, body? }
print(anthropic_captured["telemetry"])  # { ok, status, body? }
```

## API

- `init(...)`
- `context(...)`
- `get_context()`
- `capture_openai_chat_completion(...)`
- `capture_openai_chat_completion_with_result(...)`
- `capture_openai_chat_completion_async(...)`
- `capture_anthropic_message(...)`
- `capture_anthropic_message_with_result(...)`
- `capture_anthropic_message_async(...)`
- `patch_openai_client(...)`
- `flush()`

## Tests

```bash
python3 -m py_compile opsmeter_sdk/sdk.py tests/test_sdk.py
python3 -m unittest tests/test_sdk.py -v
```
