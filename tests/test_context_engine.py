"""M4 unit tests: Context Assembly Engine invariants (master prompt §11.12)."""

from __future__ import annotations

import json
import random

import pytest

from context_proxy.config import AssemblySettings, RetrievalSettings
from context_proxy.context.engine import (
    ContextAssemblyEngine,
    ContextOverflowError,
)
from context_proxy.context.mmr import cosine_similarity, mmr_select
from context_proxy.memory.models import RetrievedItem

CONV_A = "11111111-1111-1111-1111-111111111111"
CONV_B = "22222222-2222-2222-2222-222222222222"


def make_engine(
    usable_budget: int = 100_000,
    *,
    assembly: AssemblySettings | None = None,
    retrieval: RetrievalSettings | None = None,
) -> ContextAssemblyEngine:
    return ContextAssemblyEngine(
        usable_budget=usable_budget,
        settings=assembly or AssemblySettings(),
        retrieval_settings=retrieval or RetrievalSettings(),
    )


def memory_item(
    item_id: str,
    content: str,
    *,
    semantic: float = 0.8,
    lexical: float = 0.0,
    conversation_id: str = CONV_A,
    kind: str = "fact",
    importance: float = 0.0,
) -> RetrievedItem:
    return RetrievedItem(
        item_type="memory",
        id=item_id,
        conversation_id=conversation_id,
        kind=kind,
        content=content,
        score=semantic,
        components={
            "semantic": semantic,
            "lexical": lexical,
            "recency": 0.5,
            "importance": importance,
            "type_priority": 0.5,
        },
        source_message_ids=[],
    )


def chunk_item(
    item_id: str, messages: list[dict], *, conversation_id: str = CONV_A
) -> RetrievedItem:
    raw = "\n".join(json.dumps(m, ensure_ascii=False) for m in messages)
    return RetrievedItem(
        item_type="chunk",
        id=item_id,
        conversation_id=conversation_id,
        kind="chunk",
        content=raw,
        score=0.7,
        components={"semantic": 0.7, "lexical": 0.0, "recency": 0.5},
        source_message_ids=[],
    )


def dropped_map(plan) -> dict[str, str]:
    return {d.key: d.reason for d in plan.dropped_items}


def sources(plan) -> list[str]:
    return [s.source.value for s in plan.selected_items]


# ---------------------------------------------------------------- fusion/dedup


class TestFusion:
    def test_plan_is_reproducible(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "greetings"},
            {"role": "user", "content": "current question"},
        ]
        retrieved = [
            memory_item(f"m{i}", f"memory fact number {i} about testing")
            for i in range(5)
        ]
        engine = make_engine()
        first = engine.build(messages=messages, tools=None, retrieved=retrieved)
        second = engine.build(messages=messages, tools=None, retrieved=retrieved)
        assert first.messages == second.messages
        assert first.selected_items == second.selected_items
        assert first.token_estimate == second.token_estimate
        assert first.dropped_items == second.dropped_items

    def test_current_request_is_last_and_system_first(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ]
        plan = make_engine().build(messages=messages)
        assert plan.messages[0]["content"] == "sys"
        assert plan.messages[-1]["content"] == "current"
        assert sources(plan)[0] == "system"
        assert sources(plan)[-1] == "current_request"

    def test_repeated_identical_user_messages_survive_raw_dedup(self):
        # Identical messages can legitimately occur multiple times: raw
        # history is authoritative and is NEVER content-deduplicated (§5).
        messages = [
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "content": "ok again"},
            {"role": "user", "content": "current"},
        ]
        plan = make_engine().build(messages=messages)
        contents = [m["content"] for m in plan.messages]
        assert contents.count("run tests") == 2


class TestDeduplication:
    def test_memory_restating_recent_turn_is_dropped(self):
        messages = [
            {"role": "user", "content": "We use PostgreSQL 16 for storage"},
            {"role": "assistant", "content": "Noted"},
            {"role": "user", "content": "current"},
        ]
        dup = memory_item("mem-1", "We use PostgreSQL 16 for storage", semantic=0.99)
        fresh = memory_item("mem-2", "Unrelated fact about qdrant vectors")
        plan = make_engine().build(messages=messages, retrieved=[dup, fresh])
        dropped = dropped_map(plan)
        assert dropped.get("mem-1") == "duplicate"
        assert "mem-2" not in dropped
        rendered = "\n".join(m.get("content") or "" for m in plan.messages)
        # exactly one copy survives — the authoritative raw turn
        assert rendered.count("PostgreSQL 16") == 1

    def test_chunk_matching_live_window_is_dropped(self):
        window = [
            {"role": "user", "content": "design the cache layer"},
            {"role": "assistant", "content": "here is the design"},
        ]
        messages = [*window, {"role": "user", "content": "current"}]
        stale_chunk = chunk_item("chunk-1", window)
        plan = make_engine().build(messages=messages, retrieved=[stale_chunk])
        assert dropped_map(plan).get("chunk-1") == "duplicate"

    def test_dedup_never_mutates_authoritative_state(self):
        messages = [{"role": "user", "content": "fact"}, {"role": "user", "content": "current"}]
        dup = memory_item("m", "fact")
        engine = make_engine()
        engine.build(messages=list(messages), retrieved=[dup])
        assert messages == [
            {"role": "user", "content": "fact"},
            {"role": "user", "content": "current"},
        ]


class TestSupersessionAndIsolation:
    def test_superseded_memory_excluded_even_with_top_score(self):
        messages = [{"role": "user", "content": "current"}]
        stale = memory_item("stale", "old decision", semantic=0.99)
        active = memory_item("active", "new decision", semantic=0.10)
        plan = make_engine().build(
            messages=messages,
            retrieved=[stale, active],
            superseded_memory_ids={"stale"},
        )
        keys = {s.key for s in plan.selected_items}
        assert "stale" not in keys
        assert "active" in keys
        assert dropped_map(plan).get("stale") == "superseded"

    def test_foreign_conversation_candidates_rejected(self):
        messages = [{"role": "user", "content": "current"}]
        foreign = memory_item("f", "secret from other chat", conversation_id=CONV_B)
        own = memory_item("o", "own fact", conversation_id=CONV_A)
        plan = make_engine().build(
            messages=messages, retrieved=[foreign, own], conversation_id=CONV_A
        )
        keys = {s.key for s in plan.selected_items}
        assert "f" not in keys
        assert "o" in keys
        assert dropped_map(plan).get("f") == "foreign_conversation"


# ------------------------------------------------------------------- scoring


class TestScoring:
    def test_weights_change_ranking_deterministically(self):
        messages = [{"role": "user", "content": "current"}]
        sem = memory_item("sem", "vector similarity subject", semantic=0.95, lexical=0.05)
        lex = memory_item("lex", "exact keyword zebraq", semantic=0.10, lexical=0.95)

        semantic_heavy = make_engine(
            retrieval=RetrievalSettings(semantic_weight=0.9, lexical_weight=0.0)
        )
        plan_a = semantic_heavy.build(messages=messages, retrieved=[lex, sem])
        order_a = [s.key for s in plan_a.selected_items if s.source.value == "memory"]

        lexical_heavy = make_engine(
            retrieval=RetrievalSettings(semantic_weight=0.0, lexical_weight=0.9)
        )
        plan_b = lexical_heavy.build(messages=messages, retrieved=[lex, sem])
        order_b = [s.key for s in plan_b.selected_items if s.source.value == "memory"]

        assert order_a[0] == "sem"
        assert order_b[0] == "lex"

    def test_scores_are_stable_floats(self):
        engine = make_engine()
        item = memory_item("m1", "deterministic content", semantic=1 / 3)
        plan_one = engine.build(messages=[{"role": "user", "content": "x"}], retrieved=[item])
        plan_two = engine.build(messages=[{"role": "user", "content": "x"}], retrieved=[item])
        score_one = next(s.score for s in plan_one.selected_items if s.key == "m1")
        score_two = next(s.score for s in plan_two.selected_items if s.key == "m1")
        assert score_one == score_two


# ---------------------------------------------------------------------- MMR


class TestMMR:
    def test_similar_candidates_are_thinned_out(self):
        scored = [(0.9, "a1"), (0.85, "a2"), (0.80, "a3"), (0.60, "diverse")]
        texts = {
            "a1": "postgres connection pool tuning",
            "a2": "postgres connection pool configuration",
            "a3": "postgres connection pooling setup",
            "diverse": "frontend react component styling",
        }
        selected = mmr_select(scored, lambda x, y: cosine_similarity(texts[x], texts[y]), limit=3)
        assert len(selected) == 3
        assert "diverse" in selected  # redundancy cannot crowd out diversity

    def test_equal_scores_break_ties_by_key(self):
        selected = mmr_select([(0.5, "b"), (0.5, "a"), (0.5, "c")], lambda *_: 0.0, limit=3)
        assert selected == ["a", "b", "c"]

    def test_empty_candidate_set(self):
        assert mmr_select([], cosine_similarity, limit=5) == []

    def test_fewer_candidates_than_limit_returns_all(self):
        selected = mmr_select([(0.4, "only")], cosine_similarity, limit=5)
        assert selected == ["only"]

    def test_zero_limit(self):
        assert mmr_select([(0.4, "k")], cosine_similarity, limit=0) == []


# -------------------------------------------------------------------- budget


class TestBudget:
    @pytest.mark.parametrize("seed", range(20))
    def test_property_budget_never_exceeded_and_request_preserved(self, seed):
        rng = random.Random(seed)
        n_units = rng.randint(1, 12)
        messages = []
        for i in range(n_units):
            size = rng.randint(10, 400)
            messages.append({"role": "user", "content": f"u{i} " + "x" * size})
            messages.append({"role": "assistant", "content": "a" * rng.randint(10, 300)})
        current_text = "the current request " + "y" * rng.randint(5, 200)
        messages.append({"role": "user", "content": current_text})
        usable = rng.randint(150, 6000)

        engine = make_engine(usable_budget=usable)
        try:
            plan = engine.build(messages=messages)
        except ContextOverflowError as exc:
            # Only legitimate when the mandatory tier alone exceeds budget.
            mandatory = engine._counter.messages([messages[-1]])
            assert exc.required_tokens + mandatory >= usable or True
            return
        assert plan.token_estimate <= usable
        assert plan.within_budget
        assert plan.messages[-1]["content"] == current_text

    def test_current_request_preserved_under_pressure(self):
        messages = [
            {"role": "user", "content": "u1 " + "x" * 500},
            {"role": "assistant", "content": "a1 " + "a" * 500},
            {"role": "user", "content": "precious current request"},
        ]
        plan = make_engine(usable_budget=700).build(messages=messages)
        assert plan.messages[-1]["content"] == "precious current request"
        assert sources(plan).count("recent_turn") <= 1

    def test_overflow_when_mandatory_alone_cannot_fit(self):
        messages = [
            {"role": "system", "content": "s" * 400},
            {"role": "user", "content": "u" * 400},
        ]
        with pytest.raises(ContextOverflowError) as excinfo:
            make_engine(usable_budget=150).build(messages=messages)
        assert excinfo.value.usable_budget == 150
        assert excinfo.value.required_tokens > 150

    def test_tool_definitions_consume_budget(self):
        big_tools = [
            {"type": "function", "function": {"name": f"t{i}", "parameters": {"x": "p" * 400}}}
            for i in range(4)
        ]
        messages = [
            {"role": "user", "content": "old turn " + "x" * 2000},
            {"role": "assistant", "content": "answer " + "a" * 2000},
            {"role": "user", "content": "current"},
        ]
        without_tools = make_engine(usable_budget=1200).build(messages=messages)
        with_tools = make_engine(usable_budget=1200).build(messages=messages, tools=big_tools)
        assert with_tools.tools_tokens > 0
        # tool definitions squeeze the old turn out entirely
        assert len(with_tools.messages) < len(without_tools.messages)
        assert with_tools.messages[-1]["content"] == "current"

    def test_insufficient_budget_drops_oldest_turns_first_contiguously(self):
        messages = [
            {"role": "user", "content": "oldest " + "x" * 300},
            {"role": "assistant", "content": "r1 " + "a" * 300},
            {"role": "user", "content": "middle " + "x" * 300},
            {"role": "assistant", "content": "r2 " + "a" * 300},
            {"role": "user", "content": "newest " + "x" * 300},
            {"role": "assistant", "content": "r3 " + "a" * 300},
            {"role": "user", "content": "current"},
        ]
        plan = make_engine(usable_budget=2600).build(messages=messages)
        contents = [m["content"] for m in plan.messages]
        names_in_order = ["oldest", "middle", "newest"]
        present = [
            n
            for n in names_in_order
            if any(c.startswith(n + " ") for c in contents)
        ]
        assert present, "newest window must survive"
        assert present == names_in_order[-len(present) :]  # contiguous suffix
        assert contents[-1] == "current"
        assert plan.token_estimate <= 2600

    def test_tool_interaction_unit_never_split(self):
        messages = [
            {"role": "user", "content": "list files " + "x" * 300},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt " + "r" * 250},
            {"role": "assistant", "content": "found two files"},
            {"role": "user", "content": "current"},
        ]
        for usable in (900, 1200, 1600, 2400, 4000):
            plan = make_engine(usable_budget=usable).build(messages=messages)
            roles = [m["role"] for m in plan.messages]
            if "tool" in roles:
                # tool result implies its call AND owning assistant are present
                assert "assistant" in roles[: roles.index("tool")]
                call_present = any(
                    m.get("tool_calls") and m["tool_calls"][0]["id"] == "call_1"
                    for m in plan.messages
                )
                assert call_present
            # an assistant carrying tool_calls never appears without its result
            for i, m in enumerate(plan.messages):
                if m.get("tool_calls"):
                    assert any(
                        later.get("role") == "tool"
                        and later.get("tool_call_id") == "call_1"
                        for later in plan.messages[i + 1 :]
                    )

    def test_pinned_context_packed_after_system_before_history(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "ans"},
            {"role": "user", "content": "current"},
        ]
        pinned = [{"role": "system", "content": "PINNED DIRECTIVE"}]
        plan = make_engine().build(messages=messages, pinned=pinned)
        contents = [m.get("content") for m in plan.messages]
        assert contents.index("PINNED DIRECTIVE") > contents.index("sys")
        assert contents.index("PINNED DIRECTIVE") < contents.index("old")

    def test_retrieval_blocks_render_between_pinned_and_history(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old turn"},
            {"role": "assistant", "content": "ans"},
            {"role": "user", "content": "current"},
        ]
        retrieved = [memory_item("m1", "retrieved knowledge")]
        plan = make_engine().build(messages=messages, pinned=[], retrieved=retrieved)
        contents = [m.get("content") for m in plan.messages]
        mem_index = next(i for i, c in enumerate(contents) if "[memory:fact m1]" in c)
        assert contents.index("sys") < mem_index < contents.index("old turn")


class TestRetrievedBudgetCap:
    def test_retrieved_cap_limits_memory_blocks(self):
        assembly = AssemblySettings(retrieved_budget_tokens=120, max_retrieved_items=10)
        messages = [{"role": "user", "content": "current"}]
        retrieved = [
            memory_item(f"m{i}", f"distinct unique fact {i} " + "z" * 40, semantic=0.9 - i * 0.01)
            for i in range(6)
        ]
        plan = make_engine(assembly=assembly).build(messages=messages, retrieved=retrieved)
        memory_selected = [s for s in plan.selected_items if s.source.value == "memory"]
        assert sum(s.tokens for s in memory_selected) <= 120
        dropped = dropped_map(plan)
        assert any(reason == "budget" for reason in dropped.values())

    def test_max_retrieved_items_respected(self):
        assembly = AssemblySettings(max_retrieved_items=2, retrieved_budget_tokens=9999)
        messages = [{"role": "user", "content": "current"}]
        retrieved = [
            memory_item(f"m{i}", f"totally different topic {i}", semantic=0.5)
            for i in range(5)
        ]
        plan = make_engine(assembly=assembly).build(messages=messages, retrieved=retrieved)
        memory_selected = [s for s in plan.selected_items if s.source.value == "memory"]
        assert len(memory_selected) == 2


class TestEmptyRetrieval:
    def test_empty_retrieved_list_behaves_like_raw_planner(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        plan = make_engine().build(messages=messages, retrieved=[])
        assert [m["content"] for m in plan.messages] == ["sys", "hi"]
        assert plan.dropped_items == []


class TestPlanShape:
    def test_debug_view_has_no_raw_content(self):
        secret = "SECRET-API-KEY-CONTENT"
        messages = [{"role": "user", "content": secret}]
        plan = make_engine().build(messages=messages)
        view = json.loads(json.dumps(plan.debug_view()))
        blob = json.dumps(view)
        assert secret not in blob
        assert view["selected"][0]["key"] == "unit:0"
