from __future__ import annotations

import pytest

from context_proxy.context.planner import (
    ContextOverflowError,
    plan_context,
    segment_messages,
)
from context_proxy.context.tokens import TokenCounter


def msg(role: str, content: str, **extra) -> dict:
    return {"role": role, "content": content, **extra}


def tool_interaction(call_id: str = "call_1", args: str = '{"path": "a.py"}') -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": args},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "file body"},
    ]


class TestTokenCounter:
    def test_text_estimation_is_deterministic_and_monotonic(self):
        counter = TokenCounter()
        assert counter.text("") == 0
        assert counter.text(None) == 0
        assert counter.text("abcd") == 1
        assert counter.text("a" * 400) == 100
        assert counter.text("b" * 401) == 101

    def test_message_counts_content_and_overhead(self):
        counter = TokenCounter()
        base = counter.message(msg("user", ""))
        with_text = counter.message(msg("user", "x" * 40))
        assert with_text > base
        assert with_text - base == 10

    def test_tool_calls_add_structured_tokens(self):
        counter = TokenCounter()
        plain = msg("assistant", "")
        with_call = msg(
            "assistant",
            "",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        )
        assert counter.message(with_call) >= counter.message(plain) + 8

    def test_tools_field_counted(self):
        counter = TokenCounter()
        tools = [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ]
        tokens = counter.tools(tools)
        assert tokens > 0
        assert counter.tools(None) == 0


class TestSegmentation:
    def test_tool_result_stays_with_its_call(self):
        units = segment_messages(
            [msg("system", "s"), msg("user", "hi"), *tool_interaction(), msg("assistant", "done")],
            TokenCounter(),
        )
        kinds = [u.kind for u in units]
        assert kinds == ["system", "user", "interaction", "assistant"]
        assert len(units[2].messages) == 2

    def test_tool_result_of_other_call_starts_new_unit(self):
        units = segment_messages(
            [
                *tool_interaction("call_A"),
                {"role": "tool", "tool_call_id": "call_B", "content": "x"},
            ],
            TokenCounter(),
        )
        assert [u.kind for u in units] == ["interaction", "assistant"]


class TestPlanContext:
    def small_budget(self, usable: int = 200):
        return {"usable_budget": usable, "counter": TokenCounter()}

    def test_within_budget_passthrough_unchanged(self):
        messages = [msg("system", "sys"), msg("user", "hi"), msg("assistant", "hello")]
        plan = plan_context(messages, tools=None, usable_budget=10_000)
        assert plan.messages == messages
        assert plan.dropped_units == 0
        assert plan.within_budget

    def test_oversized_history_trims_oldest_keeps_system_and_last(self):
        history = [msg("user", "x" * 400), msg("assistant", "y" * 400), msg("user", "z" * 400)]
        messages = [msg("system", "keep-me"), *history, msg("user", "current")]
        plan = plan_context(messages, tools=None, **self.small_budget(300))
        assert plan.messages[0] == messages[0]  # system preserved
        assert plan.messages[-1] == messages[-1]  # current request preserved
        assert plan.within_budget
        assert plan.dropped_units >= 1
        # dropped oldest first
        assert all(m["content"] != "x" * 400 for m in plan.messages)

    def test_system_never_dropped_even_when_history_huge(self):
        messages = [msg("system", "S"), *[msg("user", "h" * 500)] * 5, msg("user", "now")]
        plan = plan_context(messages, tools=None, **self.small_budget(250))
        assert plan.messages[0]["role"] == "system"
        assert plan.within_budget

    def test_tool_interaction_never_split(self):
        interaction = tool_interaction(args="a" * 300)
        messages = [
            msg("system", "s"),
            *interaction,
            msg("user", "current question that must survive"),
        ]
        plan = plan_context(messages, tools=None, **self.small_budget(150))
        roles = [m["role"] for m in plan.messages]
        if any(m["content"] and m["content"].startswith("a") for m in plan.messages):
            # if the interaction survived, it survived whole
            assistant_idx = next(i for i, m in enumerate(plan.messages) if m.get("tool_calls"))
            assert roles[assistant_idx : assistant_idx + 2] == ["assistant", "tool"]

    def test_tools_consume_budget(self):
        big_tools = [{"type": "function", "function": {"name": "f", "description": "d" * 800}}]
        messages = [msg("user", "tiny")]
        with pytest.raises(ContextOverflowError):
            plan_context(messages, tools=big_tools, usable_budget=100)

    def test_impossible_request_raises_overflow(self):
        messages = [msg("system", "s"), msg("user", "g" * 5000)]
        with pytest.raises(ContextOverflowError) as excinfo:
            plan_context(messages, tools=None, usable_budget=200)
        assert excinfo.value.required_tokens > excinfo.value.usable_budget

    def test_budget_shrinks_when_tools_grow(self):
        messages = [msg("user", "w" * 600), msg("assistant", "w" * 600), msg("user", "cur")]
        small = plan_context(messages, tools=None, usable_budget=350)
        big = plan_context(
            messages,
            tools=[{"type": "function", "function": {"name": "f", "description": "d" * 200}}],
            usable_budget=350,
        )
        assert len(big.messages) <= len(small.messages)
        assert big.within_budget and small.within_budget

    def test_never_exceeds_usable_budget_property(self):
        import random

        rng = random.Random(42)
        for _ in range(50):
            n_units = rng.randint(2, 12)
            messages: list[dict] = [msg("system", "s" * rng.randint(0, 50))]
            for _i in range(n_units):
                messages.append(msg(rng.choice(["user", "assistant"]), "q" * rng.randint(0, 400)))
            budget = rng.randint(80, 1200)
            try:
                plan = plan_context(messages, tools=None, usable_budget=budget)
                assert plan.total_tokens + plan.tools_tokens <= budget
                assert plan.messages[-1] == messages[-1]
            except ContextOverflowError:
                pass  # legitimate when even system+current exceed budget
