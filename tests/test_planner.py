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
    def test_tool_chain_stays_one_turn(self):
        units = segment_messages(
            [msg("system", "s"), msg("user", "hi"), *tool_interaction(), msg("assistant", "done")],
            TokenCounter(),
        )
        kinds = [u.kind for u in units]
        assert kinds == ["system", "turn"]
        # turn = user + assistant(tool_call) + tool(result) + assistant(final)
        assert len(units[1].messages) == 4
        roles = [m["role"] for m in units[1].messages]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_each_user_message_starts_new_turn(self):
        units = segment_messages(
            [msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2"), msg("assistant", "a2")],
            TokenCounter(),
        )
        assert [u.kind for u in units] == ["turn", "turn"]
        assert len(units[0].messages) == 2
        assert len(units[1].messages) == 2

    def test_orphan_assistant_before_any_user_is_droppable_prefill(self):
        units = segment_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_A",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_A", "content": "x"},
                msg("user", "real start"),
            ],
            TokenCounter(),
        )
        assert [u.kind for u in units] == ["prefill", "turn"]
        assert len(units[0].messages) == 2  # call + result kept together even orphaned


class TestPlanContext:
    def small_budget(self, usable: int = 200):
        return {"usable_budget": usable, "counter": TokenCounter()}

    def test_within_budget_passthrough_unchanged(self):
        messages = [msg("system", "sys"), msg("user", "hi"), msg("assistant", "hello")]
        plan = plan_context(
            history=messages,
            current_request=[], tools=None, usable_budget=10_000)
        assert plan.messages == messages
        assert plan.dropped_units == 0
        assert plan.within_budget

    def test_oversized_history_trims_oldest_keeps_system_and_last(self):
        history = [msg("user", "x" * 400), msg("assistant", "y" * 400), msg("user", "z" * 400)]
        messages = [msg("system", "keep-me"), *history, msg("user", "current")]
        plan = plan_context(
            history=messages,
            current_request=[], tools=None, **self.small_budget(300))
        assert plan.messages[0] == messages[0]  # system preserved
        assert plan.messages[-1] == messages[-1]  # current request preserved
        assert plan.within_budget
        assert plan.dropped_units >= 1
        # dropped oldest first
        assert all(m["content"] != "x" * 400 for m in plan.messages)

    def test_system_never_dropped_even_when_history_huge(self):
        messages = [msg("system", "S"), *[msg("user", "h" * 500)] * 5, msg("user", "now")]
        plan = plan_context(
            history=messages,
            current_request=[], tools=None, **self.small_budget(250))
        assert plan.messages[0]["role"] == "system"
        assert plan.within_budget

    def test_tool_interaction_never_split(self):
        interaction = tool_interaction(args="a" * 300)
        messages = [
            msg("system", "s"),
            *interaction,
            msg("user", "current question that must survive"),
        ]
        plan = plan_context(
            history=messages,
            current_request=[], tools=None, **self.small_budget(150))
        roles = [m["role"] for m in plan.messages]
        if any(m["content"] and m["content"].startswith("a") for m in plan.messages):
            # if the interaction survived, it survived whole
            assistant_idx = next(i for i, m in enumerate(plan.messages) if m.get("tool_calls"))
            assert roles[assistant_idx : assistant_idx + 2] == ["assistant", "tool"]

    def test_tools_consume_budget(self):
        big_tools = [{"type": "function", "function": {"name": "f", "description": "d" * 800}}]
        messages = [msg("user", "tiny")]
        with pytest.raises(ContextOverflowError):
            plan_context(
            history=messages,
            current_request=[], tools=big_tools, usable_budget=100)

    def test_impossible_request_raises_overflow(self):
        with pytest.raises(ContextOverflowError) as excinfo:
            plan_context(
                history=[msg("system", "s")],
                current_request=[msg("user", "g" * 5000)],
                tools=None,
                usable_budget=200,
            )
        assert excinfo.value.required_tokens > excinfo.value.usable_budget

    def test_budget_shrinks_when_tools_grow(self):
        messages = [msg("user", "w" * 600), msg("assistant", "w" * 600), msg("user", "cur")]
        small = plan_context(
            history=messages,
            current_request=[], tools=None, usable_budget=350)
        big = plan_context(
            history=messages,
            current_request=[],
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
            history: list[dict] = [msg("system", "s" * rng.randint(0, 50))]
            for _i in range(n_units):
                history.append(msg(rng.choice(["user", "assistant"]), "q" * rng.randint(0, 400)))
            current = [msg("user", "cur")]
            budget = rng.randint(80, 1200)
            try:
                plan = plan_context(
                    history=history,
                    current_request=current,
                    tools=None,
                    usable_budget=budget,
                )
                assert plan.total_tokens + plan.tools_tokens <= budget
                assert plan.messages[-1] == current[-1]
            except ContextOverflowError:
                pass  # legitimate when even system+current exceed budget


class TestInteractionUnitTrimming:
    def big(self, text: str) -> dict:
        return msg("user", text)

    def test_only_newest_interaction_fits_no_orphan_assistant(self):
        """10.7: oldest turn evicted whole; newest kept intact."""
        messages = [
            msg("system", "s"),
            self.big("q" * 800),
            msg("assistant", "a" * 400),  # old interaction
            msg("user", "u2" * 100),
            msg("assistant", "final answer"),  # newest interaction
        ]
        plan = plan_context(
            history=messages,
            current_request=[], tools=None, usable_budget=260, counter=TokenCounter())
        roles = [m["role"] for m in plan.messages]
        assert "assistant" in roles
        # no orphan assistant from the evicted interaction
        for i, m in enumerate(plan.messages):
            if m["role"] == "assistant" and m is not plan.messages[-1]:
                assert any(
                    prior["role"] == "user" for prior in plan.messages[:i]
                ), "assistant retained without its user interaction"

    def test_tool_chain_atomic_under_forced_trim(self):
        """10.8: user -> assistant(tc) -> tool -> assistant(final) -> user(next)."""
        call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        }
        messages = [
            msg("user", "u" * 500),
            call,
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
            msg("assistant", "done"),
            msg("user", "next question"),
        ]
        plan = plan_context(
            history=messages,
            current_request=[], tools=None, usable_budget=200, counter=TokenCounter())
        selected = plan.messages
        # first turn either fully present or fully absent
        has_call = any(m.get("tool_calls") for m in selected)
        has_result = any(m.get("role") == "tool" for m in selected)
        assert has_call == has_result
        if has_call:
            idx_call = next(i for i, m in enumerate(selected) if m.get("tool_calls"))
            assert selected[idx_call - 1]["role"] == "user"
            assert selected[idx_call + 1]["role"] == "tool"
            assert selected[idx_call + 2]["content"] == "done"
        else:
            assert all(m["content"] != "u" * 500 for m in selected)
        assert selected[-1] == messages[-1]  # current request preserved

    def test_current_request_never_evicted_when_history_huge(self):
        """10.9."""
        messages = [msg("user", "h" * 3000), msg("assistant", "h" * 3000), msg("user", "keep me")]
        plan = plan_context(
            history=messages,
            current_request=[], tools=None, usable_budget=120, counter=TokenCounter())
        assert plan.messages[-1]["content"] == "keep me"

    def test_pinned_reservation_shrinks_available_context(self):
        messages = [msg("user", "x" * 400), msg("assistant", "x" * 400), msg("user", "cur")]
        without = plan_context(
            history=messages,
            current_request=[], tools=None, usable_budget=350, reserved_tokens=0)
        with_res = plan_context(
            history=messages,
            current_request=[], tools=None, usable_budget=350, reserved_tokens=150)
        assert len(with_res.messages) <= len(without.messages)
        assert with_res.total_tokens + 150 <= 350


class TestTokenCounterEdgeCases:
    """10.12: invariants over content shapes, not tokenizer equivalence."""

    def setup_method(self):
        self.counter = TokenCounter()

    def test_empty_and_whitespace(self):
        assert self.counter.text("") == 0
        assert self.counter.text(None) == 0
        assert self.counter.message(msg("user", "")) >= 1

    def test_ascii_monotonic(self):
        short = self.counter.text("abc")
        long = self.counter.text("a" * 300)
        assert 0 < short < long

    def test_unicode_counted_not_crashing(self):
        tokens = self.counter.text("你好世界 🌍 émojis café")
        assert tokens > 0

    def test_code_blocks_counted(self):
        code = "def f():\n    return 'x' * 10\n"
        assert self.counter.text(code) > self.counter.text("def f():")

    def test_json_tool_arguments_counted(self):
        plain = self.counter.message(msg("assistant", ""))
        with_args = self.counter.message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"path": "src/x.py", "limit": 50}'},
                    }
                ],
            }
        )
        assert with_args > plain

    def test_multiple_messages_sum(self):
        messages = [msg("user", "a" * 40), msg("assistant", "b" * 80)]
        total = self.counter.messages(messages)
        assert total == sum(self.counter.message(m) for m in messages)

    def test_repeated_calculations_deterministic(self):
        messages = [msg("system", "s"), msg("user", "hello world"), *tool_interaction()]
        values = {self.counter.messages(list(messages)) for _ in range(100)}
        assert len(values) == 1
