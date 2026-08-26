from context_proxy.conversation.reconciliation import PRUNED_TOOL_RESULT, reconcile_projection


def test_pruned_tool_result_is_accepted_for_same_tool_call():
    persisted = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "content": "large original output", "tool_call_id": "call-1"},
    ]
    incoming = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "content": PRUNED_TOOL_RESULT, "tool_call_id": "call-1"},
    ]

    assert reconcile_projection(persisted, incoming).mode == "exact"


def test_pruned_tool_result_with_different_call_id_is_rejected():
    persisted = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "content": "large original output", "tool_call_id": "call-1"},
    ]
    incoming = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "content": PRUNED_TOOL_RESULT, "tool_call_id": "call-2"},
    ]

    assert reconcile_projection(persisted, incoming).mode == "conflict"
