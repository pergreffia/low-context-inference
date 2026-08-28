from context_proxy.api.validation import validate_chat_payload
from context_proxy.context.tokens import TokenCounter


def test_chat_payload_validation_keeps_provider_extensions_opaque():
    payload = {
        "model": "provider-specific",
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "provider_extension", "anything": {"x": 1}}],
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "future",
                        "provider_payload": {"arbitrary": True},
                    }
                ],
            }
        ],
        "tools": [{"type": "future_tool", "provider": {"arbitrary": True}}],
        "n": 1,
        "provider_extension": {"arbitrary": [1, 2, 3]},
    }

    validate_chat_payload(payload)


def test_token_counter_does_not_break_on_unknown_tool_call_shapes():
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "future", "provider_payload": {"x": 1}},
            {"id": "call_2", "type": "function", "function": "opaque"},
            "provider-specific-tool-call",
        ],
    }

    tokens = TokenCounter().message(message)

    assert tokens > 0
