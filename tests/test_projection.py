from context_proxy.conversation.projection import is_auxiliary_projection


def test_opencode_title_projection_is_auxiliary() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Generate a title for this conversation:\n"},
        {"role": "user", "content": "Implement the feature"},
    ]

    assert is_auxiliary_projection(messages) is True


def test_title_like_single_user_message_is_not_auxiliary() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Generate a title for this conversation: please explain this code"},
    ]

    assert is_auxiliary_projection(messages) is False


def test_normal_build_projection_is_canonical() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Implement the feature"},
    ]

    assert is_auxiliary_projection(messages) is False


def test_custom_agent_content_is_not_classified_by_agent_name() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "review this implementation"},
        {"role": "assistant", "content": "I will review it"},
        {"role": "user", "content": "continue"},
    ]

    assert is_auxiliary_projection(messages) is False
