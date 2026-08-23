from __future__ import annotations

import httpx
from conftest import CHAT_RESPONSE, client_for_handler
from helpers import captured_json, chat_payload


def test_model_override_applied_when_configured():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = client_for_handler(handler, model="qwen-model")

    payload = chat_payload()  # model: "client-model"
    r = client.post("/v1/chat/completions", json=payload)

    assert r.status_code == 200
    assert captured_json(captured[0])["model"] == "qwen-model"
    # caller's dict untouched
    assert payload["model"] == "client-model"


def test_model_override_applied_to_streaming():
    captured: list[httpx.Request] = []

    async def agen():
        yield b"data: x\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=agen(), headers={"content-type": "text/event-stream"})

    client = client_for_handler(handler, model="qwen-model")
    r = client.post("/v1/chat/completions", json=chat_payload(stream=True))
    assert r.status_code == 200
    assert captured_json(captured[0])["model"] == "qwen-model"


def test_client_model_preserved_when_no_override_configured():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = client_for_handler(handler)  # INFERENCE__MODEL unset
    payload = chat_payload(model=None)
    payload["model"] = "client-model"

    client.post("/v1/chat/completions", json=payload)

    assert captured_json(captured[0])["model"] == "client-model"


def test_original_payload_not_mutated_on_streaming_path():
    captured: list[httpx.Request] = []

    async def agen():
        yield b"data: x\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=agen(), headers={"content-type": "text/event-stream"})

    client = client_for_handler(handler, model="override-model")
    payload = chat_payload(stream=True)

    client.post("/v1/chat/completions", json=payload)

    assert payload["model"] == "client-model"
    assert captured_json(captured[0])["model"] == "override-model"


def test_arbitrary_openai_fields_forwarded():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = client_for_handler(handler)
    payload = {
        **chat_payload(),
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 128,
        "stop": ["END"],
        "seed": 42,
        "response_format": {"type": "json_object"},
        "metadata": {"session": "abc"},
        "logprobs": True,
        "parallel_tool_calls": False,
    }

    r = client.post("/v1/chat/completions", json=payload)

    assert r.status_code == 200
    assert captured_json(captured[0]) == payload


def test_malformed_json_rejected_as_invalid_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream must not be called for malformed JSON")

    client = client_for_handler(handler)
    r = client.post(
        "/v1/chat/completions",
        content=b'{"model": "m", "messages": [',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
