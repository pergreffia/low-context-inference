from __future__ import annotations

import httpx
from conftest import CHAT_RESPONSE, client_for_handler
from helpers import captured_json, chat_payload


class _TestAsyncStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"data: x\n\n"


def test_server_model_configuration_is_ignored():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = client_for_handler(handler, model="server-model")
    payload = chat_payload()  # model: "client-model"
    r = client.post("/v1/chat/completions", json=payload)

    assert r.status_code == 200
    assert captured_json(captured[0])["model"] == "client-model"
    # caller's dict untouched
    assert payload["model"] == "client-model"


def test_server_model_configuration_is_ignored_for_streaming():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=_TestAsyncStream(),
            headers={"content-type": "text/event-stream"},
        )

    client = client_for_handler(handler, model="server-model")
    payload = chat_payload(stream=True)
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert captured_json(captured[0])["model"] == "client-model"


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

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=_TestAsyncStream(),
            headers={"content-type": "text/event-stream"},
        )

    client = client_for_handler(handler, model="ignored-server-model")
    payload = chat_payload(stream=True)

    client.post("/v1/chat/completions", json=payload)

    assert payload["model"] == "client-model"
    assert captured_json(captured[0])["model"] == "client-model"


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
