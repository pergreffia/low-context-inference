"""Deterministic function/custom tool-call lifecycle regressions (review §4).

Real-provider tool tests are model-dependent and may legitimately SKIP. This
suite is the authoritative compatibility contract against a controlled fake
provider:

    request -> function/custom definition -> assistant tool_call
            -> validation -> capture -> persistence -> replay

Covers non-streaming AND streaming reconstruction from SSE chunks, multiple
tool calls, replay after persistence. Unknown-type passthrough is covered by
tests/test_final_fixes.py::TestCustomToolCalls::test_unknown_tool_type_passthrough.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from context_proxy.main import create_app
from tests.conftest import UPSTREAM, make_settings

FUNCTION_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    },
}
CUSTOM_TOOL_DEF = {
    "type": "custom",
    "custom": {"name": "run_query", "description": "Run a query"},
}


def function_call(call_id: str = "call_f1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
    }


def custom_call(call_id: str = "call_c1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "custom",
        "custom": {"name": "run_query", "input": "SELECT 1;"},
    }


def completion_response(message: dict[str, Any], finish_reason: str = "tool_calls") -> dict:
    return {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
    }


class RecordingStore:
    def __init__(self) -> None:
        self.conversations: dict[str, list[dict[str, Any]]] = {}
        self.metadata: list[dict | None] = []

    async def ping(self):
        return None

    async def ensure_conversation(self, conversation_id):
        self.conversations.setdefault(conversation_id, [])

    async def reconcile_history(self, conversation_id, messages, metadata=None):
        bucket = self.conversations.setdefault(conversation_id, [])
        overlap = min(len(bucket), len(messages))
        for index in range(overlap):
            if bucket[index] != messages[index]:
                from context_proxy.conversation.store import HistoryDivergenceError

                raise HistoryDivergenceError(conversation_id, index)
        suffix = messages[len(bucket) :]
        if suffix:
            self.metadata.append(metadata)
        bucket.extend(suffix)
        return []

    async def get_messages(self, conversation_id):
        return list(self.conversations.get(conversation_id, []))


def buffered_client(responses: list[dict], store: RecordingStore):
    """Fake provider returning the queued completions in order."""
    queue = list(responses)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            raise AssertionError("streaming handler used for buffered test")
        return httpx.Response(200, json=queue.pop(0))

    app = create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(base_url=UPSTREAM, transport=httpx.MockTransport(handler)),
        store=store,
    )
    return app, captured


def sse_client(chunks_source, store: RecordingStore):
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)

        async def agen():
            for chunk in chunks_source():
                yield chunk

        return httpx.Response(
            200,
            content=agen(),
            headers={"content-type": "text/event-stream"},
        )

    app = create_app(
        make_settings(),
        llm_client=httpx.AsyncClient(base_url=UPSTREAM, transport=httpx.MockTransport(handler)),
        store=store,
    )
    return app, captured


CONV = "22222222-2222-2222-2222-222222222222"


def persisted(store) -> list[dict[str, Any]]:
    """Synchronously reload authoritative history from the fake store."""
    import asyncio

    return asyncio.run(store.get_messages(CONV))


def post_chat(client: TestClient, *, stream: bool = False, **extra):
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "do it"}],
        "conversation_id": CONV,
        **extra,
    }
    if stream:
        payload["stream"] = True
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            return b"".join(response.iter_bytes())
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    return response


# --------------------------------------------------- function lifecycle


class TestFunctionToolLifecycle:
    def test_non_streaming_definition_call_persisted_replayed(self):
        call = function_call()
        assistant_message = {"role": "assistant", "content": None, "tool_calls": [call]}
        responses = [
            completion_response(assistant_message),
            # second answer after the client posts the tool result
            completion_response({"role": "assistant", "content": "done"}, finish_reason="stop"),
        ]
        store = RecordingStore()
        app, captured = buffered_client(responses, store)

        with TestClient(app) as client:
            r1 = post_chat(client, tools=[FUNCTION_TOOL_DEF])
            assert r1.json()["choices"][0]["message"]["tool_calls"] == [call]

            # authoritative reload + replay with the tool result
            reloaded = persisted(store)
            assert [(m["role"], m["content"]) for m in reloaded] == [
                ("user", "do it"),
                ("assistant", None),
            ]
            assert reloaded[-1]["tool_calls"] == [call]

            replay = [
                *reloaded,
                {"role": "tool", "tool_call_id": call["id"], "content": "file contents"},
                {"role": "user", "content": "summarize"},
            ]
            r2 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": replay,
                    "conversation_id": CONV,
                    "tools": [FUNCTION_TOOL_DEF],
                },
            )
            assert r2.status_code == 200

        # provider-facing replay kept the tool-call structure EXACTLY
        sent = json.loads(captured[-1].content)
        assert sent["tools"] == [FUNCTION_TOOL_DEF]
        replayed_call = sent["messages"][1]["tool_calls"][0]
        assert replayed_call == call
        assert sent["messages"][2]["role"] == "tool"
        assert sent["messages"][2]["tool_call_id"] == call["id"]
        # first persistence recorded finish_reason tool_calls
        assert any((m or {}).get("finish_reason") == "tool_calls" for m in store.metadata)

    def test_streaming_function_tool_reconstructed_and_persisted(self):
        """Name/arguments arrive fragmented across SSE chunks."""
        chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":0,"id":"call_s1","type":"function","function":{"name":"read_"}}]},'
            b'"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":0,"function":{"name":"file","arguments":"{\\"path\\""}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":0,"function":{"arguments":": \\"a.py\\"}"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        expected_call = {
            "id": "call_s1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
        }

        store = RecordingStore()
        app, _captured = sse_client(lambda: chunks, store)
        with TestClient(app) as client:
            body = post_chat(client, stream=True, tools=[FUNCTION_TOOL_DEF])

        assert body == b"".join(chunks)                    # passthrough complete
        last = persisted(store)[-1]
        assert last["role"] == "assistant"
        assert last["tool_calls"] == [expected_call]  # exact structure

        # replay after streamed persistence keeps the reconstructed call
        captured: list[httpx.Request] = []

        def replay_handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=completion_response(
                {"role": "assistant", "content": "ok"}, finish_reason="stop"
            ))

        app2 = create_app(
            make_settings(),
            llm_client=httpx.AsyncClient(
                base_url=UPSTREAM, transport=httpx.MockTransport(replay_handler)
            ),
            store=store,
        )
        reloaded = persisted(store)
        replay = [
            *reloaded,
            {"role": "tool", "tool_call_id": "call_s1", "content": "contents"},
            {"role": "user", "content": "go on"},
        ]
        with TestClient(app2) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": replay, "conversation_id": CONV},
            )
        assert r.status_code == 200
        sent = json.loads(captured[-1].content)
        assert sent["messages"][1]["tool_calls"] == [expected_call]


# ----------------------------------------------------- custom lifecycle


class TestCustomToolLifecycle:
    def test_non_streaming_definition_call_persisted_replayed(self):
        call = custom_call()
        assistant_message = {"role": "assistant", "content": None, "tool_calls": [call]}
        responses = [
            completion_response(assistant_message),
            completion_response({"role": "assistant", "content": "ran"}, finish_reason="stop"),
        ]
        store = RecordingStore()
        app, captured = buffered_client(responses, store)

        with TestClient(app) as client:
            r1 = post_chat(client, tools=[CUSTOM_TOOL_DEF])
            assert r1.json()["choices"][0]["message"]["tool_calls"] == [call]

            reloaded = persisted(store)
            assert reloaded[-1]["tool_calls"] == [call]

            replay = [
                *reloaded,
                {"role": "tool", "tool_call_id": call["id"], "content": "1 row"},
                {"role": "user", "content": "again?"},
            ]
            r2 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": replay,
                    "conversation_id": CONV,
                    "tools": [CUSTOM_TOOL_DEF],
                },
            )
            assert r2.status_code == 200

        sent = json.loads(captured[-1].content)
        assert sent["tools"] == [CUSTOM_TOOL_DEF]           # definition untouched
        assert sent["messages"][1]["tool_calls"][0] == call  # type custom preserved

    def test_streaming_custom_tool_reconstructed_and_persisted(self):
        chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":0,"id":"cs","type":"custom","custom":{"name":"run_quer"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":0,"custom":{"name":"y","input":"SEL"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":0,"custom":{"input":"ECT 1;"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        expected_call = {
            "id": "cs",
            "type": "custom",
            "custom": {"name": "run_query", "input": "SELECT 1;"},
        }

        store = RecordingStore()
        app, _captured = sse_client(lambda: chunks, store)
        with TestClient(app) as client:
            body = post_chat(client, stream=True, tools=[CUSTOM_TOOL_DEF])

        assert body.endswith(b"data: [DONE]\n\n")
        last = persisted(store)[-1]
        assert last["tool_calls"] == [expected_call]


# ------------------------------------------------------ multi-tool calls


class TestMultipleToolCalls:
    def test_mixed_function_and_custom_calls_streaming_order_preserved(self):
        """Two interleaved indexes rebuild into ordered calls, types intact."""
        chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            # index 0: function id
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":0,"id":"mf","type":"function","function":{"name":"ls"}}]}}]}\n\n',
            # index 1: custom id
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":1,"id":"mc","type":"custom","custom":{"name":"run_q"}}]}}]}\n\n',
            # interleave argument fragments
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":1,"custom":{"input":"SE"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":0,"function":{"arguments":"{}"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":['
            b'{"index":1,"custom":{"name":"uery","input":"L"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        store = RecordingStore()
        app, _captured = sse_client(lambda: chunks, store)
        with TestClient(app) as client:
            body = post_chat(client, stream=True, tools=[FUNCTION_TOOL_DEF, CUSTOM_TOOL_DEF])

        assert body == b"".join(chunks)
        calls = persisted(store)[-1]["tool_calls"]
        assert [c["type"] for c in calls] == ["function", "custom"]   # index order
        assert calls[0] == {
            "id": "mf", "type": "function",
            "function": {"name": "ls", "arguments": "{}"},
        }
        assert calls[1] == {
            "id": "mc", "type": "custom",
            "custom": {"name": "run_query", "input": "SEL"},
        }

    def test_multiple_calls_replay_after_persistence(self):
        calls = [function_call("m1"), custom_call("m2")]
        assistant_message = {"role": "assistant", "content": None, "tool_calls": calls}
        responses = [
            completion_response(assistant_message),
            completion_response({"role": "assistant", "content": "ok"}, finish_reason="stop"),
        ]
        store = RecordingStore()
        app, captured = buffered_client(responses, store)

        with TestClient(app) as client:
            post_chat(client, tools=[FUNCTION_TOOL_DEF, CUSTOM_TOOL_DEF])
            assert persisted(store)[-1]["tool_calls"] == calls

            replay = [
                *persisted(store),
                {"role": "tool", "tool_call_id": "m1", "content": "a.py"},
                {"role": "tool", "tool_call_id": "m2", "content": "rows"},
                {"role": "user", "content": "next"},
            ]
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": replay,
                    "conversation_id": CONV,
                    "tools": [FUNCTION_TOOL_DEF, CUSTOM_TOOL_DEF],
                },
            )
            assert r.status_code == 200

        sent = json.loads(captured[-1].content)
        assert sent["messages"][1]["tool_calls"] == calls     # both, verbatim
