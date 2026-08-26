from context_proxy.conversation.reconciliation import reconcile_projection


def test_truncated_history_can_append_new_tail():
    persisted = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "B1"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "C1"},
    ]
    incoming = [
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "C1"},
        {"role": "user", "content": "new"},
    ]

    result = reconcile_projection(persisted, incoming)

    assert result.mode == "truncate_append"
    assert result.append_from == 2
