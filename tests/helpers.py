from __future__ import annotations

import json
from typing import Any

import httpx


def json_response(
    status_code: int,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(status_code, json=body, headers=headers)


def raw_response(status_code: int, text: str, content_type: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=text.encode("utf-8"),
        headers={"content-type": content_type},
    )


def chat_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return payload


def error_body(response: httpx.Response) -> dict[str, Any]:
    return response.json()["error"]


def assert_openai_error_shape(body: dict[str, Any]) -> None:
    assert set(body) == {"message", "type", "param", "code"}


def captured_json(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)
