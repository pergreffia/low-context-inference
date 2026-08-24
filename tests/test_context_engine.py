"""M4 unit tests: Context Assembly Engine invariants (master prompt §11.12).

Covers the review-fix requirements: structural current-request separation,
tool-budget enforcement, retrieval cap <= remaining budget, MMR semantics,
candidate identity, atomicity/budget properties.
"""

from __future__ import annotations

import json
import random

import pytest

from context_proxy.config import AssemblySettings, RetrievalSettings
from context_proxy.context.engine import (
    ContextAssemblyEngine,
    ContextOverflowError,
    separate_current_request,
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
    start_seq: int | None = None,
    end_seq: int | None = None,
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
        start_seq=start_seq,
        end_seq=end_seq,
    )


def chunk_item(
    item_id: str,
    messages: list[dict],
    *,
    conversation_id: str = CONV_A,
    start_seq: int | None = None,
    end_seq: int | None = None,
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
        start_seq=start_seq,
        end_seq=end_seq,
    )


def dropped_map(plan) -> dict[str, str]:
    return {d.key: d.reason for d in plan.dropped_items}


def sources(plan) -> list[str]:
    return [s.source.value for s in plan.selected_items]


def assert_whole_units(plan_messages: list[dict], units: list[list[dict]]) -> None:
    """Every plan message belongs to exactly one whole original unit."""
    remaining = list(plan_messages)
    pool = [list(u) for u in units]
    while remaining:
        for unit in pool:
            if not unit:
                continue
            n = len(unit)
            if remaining[:n] == unit:
                remaining = remaining[n:]
                break
        else:
            pytest.fail(f"partial unit detected at: {remaining[:2]!r}")


# ------------------------------------------------- structural request split


class TestRequestSeparation:
    def test_history_and_current_are_disjoint(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "B"},
        ]
        history, current = separate_current_request(messages)
        assert [m["content"] for m in history] == ["sys", "A", "a"]
        assert [m["content"] for m in current] == ["B"]
        flattened = history + current
        assert len(flattened) == len(messages)

    def test_empty_payload(self):
        assert separate_current_request([]) == ([], [])

    def test_current_tool_unit_stays_with_request(self):
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
            {"role": "assistant", "content": "done"},
        ]
        _, current = separate_current_request(messages)
        assert [m["role"] for m in current] == ["user", "assistant", "tool", "assistant"]


class TestSeparateBoundaries:
    """Structural split contract, explicit (final review §6)."""

    def test_empty_input(self):
        assert separate_current_request([]) == ([], [])

    def test_system_only(self):
        # No user-initiated interaction exists: the lone system message is
        # history/system context, current request is empty.
        history, current = separate_current_request([{"role": "system", "content": "s"}])
        assert history == [{"role": "system", "content": "s"}]
        assert current == []

    def test_system_plus_user(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
        ]
        history, current = separate_current_request(messages)
        assert history == [{"role": "system", "content": "sys"}]
        assert current == [{"role": "user", "content": "question"}]

    def test_no_loss_roundtrip(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        history, current = separate_current_request(messages)
        assert history + current == messages

    def test_current_request_exactly_once_even_if_repeated_in_history(self):
        history = [
            {"role": "user", "content": "same text"},
            {"role": "assistant", "content": "old answer"},
        ]
        current = [{"role": "user", "content": "same text"}]
        plan = make_engine().build(history=history, current_request=current)
        contents = [m.get("content") for m in plan.messages]
        assert contents.count("same text") == 2  # history copy + the request
        assert sources(plan).count("current_request") == 1
        assert sources(plan)[-1] == "current_request"

    def test_current_request_never_selected_as_recent_turn(self):
        history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        current = [{"role": "user", "content": "current"}]
        plan = make_engine().build(history=history, current_request=current)
        keys = [s.key for s in plan.selected_items]
        assert keys.count("current_request") == 1
        assert all(k != f"unit:{len(history)}" for k in keys)


# ---------------------------------------------------------------- fusion


class TestFusion:
    def test_plan_is_reproducible(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "greetings"},
        ]
        current = [{"role": "user", "content": "current question"}]
        retrieved = [
            memory_item(f"m{i}", f"memory fact number {i} about testing")
            for i in range(5)
        ]
        first = make_engine().build(history=history, current_request=current, retrieved=retrieved)
        second = make_engine().build(history=history, current_request=current, retrieved=retrieved)
        assert first.messages == second.messages
        assert first.selected_items == second.selected_items
        assert first.token_estimate == second.token_estimate
        assert first.dropped_items == second.dropped_items

    def test_system_first_current_last(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
        ]
        current = [{"role": "user", "content": "current"}]
        plan = make_engine().build(history=history, current_request=current)
        assert plan.messages[0]["content"] == "sys"
        assert plan.messages[-1]["content"] == "current"
        assert sources(plan)[0] == "system"
        assert sources(plan)[-1] == "current_request"

    def test_no_current_request_is_legal(self):
        plan = make_engine().build(history=[{"role": "user", "content": "old"}], current_request=[])
        assert sources(plan) == ["recent_turn"]

    def test_repeated_identical_user_messages_survive_raw_dedup(self):
        history = [
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "content": "ok again"},
        ]
        current = [{"role": "user", "content": "again?"}]
        plan = make_engine().build(history=history, current_request=current)
        contents = [m["content"] for m in plan.messages]
        assert contents.count("run tests") == 2

    def test_candidate_identity_structural_fields(self):
        span_chunk = chunk_item(
            "ch1",
            [{"role": "user", "content": "x"}],
            start_seq=3,
            end_seq=5,
        )
        candidate = __import__(
            "context_proxy.context.candidates", fromlist=["candidate_from_retrieved"]
        ).candidate_from_retrieved(span_chunk, make_engine()._counter)
        assert candidate.metadata["start_seq"] == 3
        assert candidate.metadata["end_seq"] == 5
        assert candidate.metadata["conversation_id"] == CONV_A
        fingerprint = candidate.metadata["fingerprint"]
        assert isinstance(fingerprint, str) and len(fingerprint) == 16
        again = __import__(
            "context_proxy.context.candidates", fromlist=["candidate_from_retrieved"]
        ).candidate_from_retrieved(span_chunk, make_engine()._counter)
        assert again.metadata["fingerprint"] == fingerprint


# ---------------------------------------------------------------- dedup


class TestDeduplication:
    def test_memory_restating_recent_turn_is_dropped(self):
        history = [
            {"role": "user", "content": "We use PostgreSQL 16 for storage"},
            {"role": "assistant", "content": "Noted"},
        ]
        current = [{"role": "user", "content": "current"}]
        dup = memory_item("mem-1", "We use PostgreSQL 16 for storage", semantic=0.99)
        fresh = memory_item("mem-2", "Unrelated fact about qdrant vectors")
        plan = make_engine().build(
            history=history, current_request=current, retrieved=[dup, fresh]
        )
        dropped = dropped_map(plan)
        assert dropped.get("mem-1") == "duplicate"
        assert "mem-2" not in dropped
        rendered = "\n".join(m.get("content") or "" for m in plan.messages)
        assert rendered.count("PostgreSQL 16") == 1

    def test_chunk_matching_live_window_is_dropped(self):
        window = [
            {"role": "user", "content": "design the cache layer"},
            {"role": "assistant", "content": "here is the design"},
        ]
        stale_chunk = chunk_item("chunk-1", window, start_seq=0, end_seq=1)
        plan = make_engine().build(
            history=window, current_request=[{"role": "user", "content": "current"}],
            retrieved=[stale_chunk],
        )
        assert dropped_map(plan).get("chunk-1") == "duplicate"

    def test_two_memories_with_same_fingerprint_dedup(self):
        one = memory_item("m1", "duplicate knowledge block", semantic=0.9)
        two = memory_item("m2", "Duplicate   KNOWLEDGE block", semantic=0.8)
        plan = make_engine().build(
            history=[], current_request=[{"role": "user", "content": "q"}],
            retrieved=[one, two],
        )
        dropped = dropped_map(plan)
        assert dropped.get("m2") == "duplicate"
        keys = {s.key for s in plan.selected_items}
        assert keys == {"m1", "current_request"}

    def test_dedup_does_not_mutate_inputs(self):
        history = [{"role": "user", "content": "fact"}]
        current = [{"role": "user", "content": "current"}]
        dup = memory_item("m", "fact")
        engine = make_engine()
        engine.build(history=list(history), current_request=list(current), retrieved=[dup])
        assert history == [{"role": "user", "content": "fact"}]
        assert current == [{"role": "user", "content": "current"}]


# -------------------------------------------------- supersession / isolation


class TestSupersessionAndIsolation:
    def test_superseded_memory_excluded_even_with_top_score(self):
        stale = memory_item("stale", "old decision", semantic=0.99)
        active = memory_item("active", "new decision", semantic=0.10)
        plan = make_engine().build(
            history=[],
            current_request=[{"role": "user", "content": "current"}],
            retrieved=[stale, active],
            superseded_memory_ids={"stale"},
        )
        keys = {s.key for s in plan.selected_items}
        assert "stale" not in keys
        assert "active" in keys
        assert dropped_map(plan).get("stale") == "superseded"

    def test_foreign_conversation_candidates_rejected(self):
        foreign = memory_item("f", "secret from other chat", conversation_id=CONV_B)
        own = memory_item("o", "own fact", conversation_id=CONV_A)
        plan = make_engine().build(
            history=[],
            current_request=[{"role": "user", "content": "current"}],
            retrieved=[foreign, own],
            conversation_id=CONV_A,
        )
        keys = {s.key for s in plan.selected_items}
        assert "f" not in keys
        assert "o" in keys
        assert dropped_map(plan).get("f") == "foreign_conversation"


# ------------------------------------------------------------------- scoring


class TestScoring:
    def test_weights_change_ranking_deterministically(self):
        sem = memory_item("sem", "vector similarity subject", semantic=0.95, lexical=0.05)
        lex = memory_item("lex", "exact keyword zebraq", semantic=0.10, lexical=0.95)
        current = [{"role": "user", "content": "current"}]

        semantic_heavy = make_engine(
            retrieval=RetrievalSettings(semantic_weight=0.9, lexical_weight=0.0)
        )
        plan_a = semantic_heavy.build(
            history=[], current_request=current, retrieved=[lex, sem]
        )
        order_a = [s.key for s in plan_a.selected_items if s.source.value == "memory"]

        lexical_heavy = make_engine(
            retrieval=RetrievalSettings(semantic_weight=0.0, lexical_weight=0.9)
        )
        plan_b = lexical_heavy.build(
            history=[], current_request=current, retrieved=[lex, sem]
        )
        order_b = [s.key for s in plan_b.selected_items if s.source.value == "memory"]

        assert order_a[0] == "sem"
        assert order_b[0] == "lex"

    def test_scores_are_stable_floats(self):
        engine = make_engine()
        item = memory_item("m1", "deterministic content", semantic=1 / 3)
        current = [{"role": "user", "content": "x"}]
        plan_one = engine.build(history=[], current_request=current, retrieved=[item])
        plan_two = engine.build(history=[], current_request=current, retrieved=[item])
        score_one = next(s.score for s in plan_one.selected_items if s.key == "m1")
        score_two = next(s.score for s in plan_two.selected_items if s.key == "m1")
        assert score_one == score_two


# ---------------------------------------------------------------------- MMR


class TestMMR:
    def test_pure_relevance_lambda_one_orders_by_relevance(self):
        scored = [(0.9, "b-key"), (0.95, "a-key"), (0.5, "c-key")]
        selected = mmr_select(scored, cosine_similarity, limit=3, lam=1.0)
        assert selected == ["a-key", "b-key", "c-key"]

    def test_diverse_candidate_beats_similar_cluster(self):
        scored = [(0.9, "a1"), (0.85, "a2"), (0.80, "a3"), (0.60, "diverse")]
        texts = {
            "a1": "postgres connection pool tuning",
            "a2": "postgres connection pool configuration",
            "a3": "postgres connection pooling setup",
            "diverse": "frontend react component styling",
        }
        sim = lambda x, y: cosine_similarity(texts[x], texts[y])  # noqa: E731
        selected = mmr_select(scored, sim, limit=3, lam=0.5)
        assert len(selected) == 3
        assert "diverse" in selected

    def test_equal_scores_break_ties_by_key(self):
        selected = mmr_select([(0.5, "b"), (0.5, "a"), (0.5, "c")], cosine_similarity, limit=3)
        assert selected == ["a", "b", "c"]

    def test_identical_runs_produce_identical_order(self):
        scored = [(0.9, "k1"), (0.9, "k2"), (0.7, "k3"), (0.7, "k4")]
        runs = [mmr_select(scored, cosine_similarity, limit=4) for _ in range(10)]
        assert all(run == runs[0] for run in runs)

    def test_empty_candidate_set(self):
        assert mmr_select([], cosine_similarity, limit=5) == []

    def test_fewer_candidates_than_limit_returns_all(self):
        assert mmr_select([(0.4, "only")], cosine_similarity, limit=5) == ["only"]

    def test_zero_limit(self):
        assert mmr_select([(0.4, "k")], cosine_similarity, limit=0) == []

    def test_empty_selected_set_handled_by_formula(self):
        # With nothing selected max_similarity must be treated as 0: the first
        # pick is pure relevance, verified by lam=0 still choosing top score.
        selected = mmr_select([(0.9, "top"), (0.1, "low")], lambda *_: 1.0, limit=2, lam=0.0)
        assert selected[0] == "top"


# -------------------------------------------------------------------- budget


class TestBudget:
    @pytest.mark.parametrize("seed", range(20))
    def test_property_budget_and_request_preserved(self, seed):
        rng = random.Random(seed)
        n_units = rng.randint(1, 12)
        history = []
        for i in range(n_units):
            size = rng.randint(10, 400)
            history.append({"role": "user", "content": f"u{i} " + "x" * size})
            history.append({"role": "assistant", "content": "a" * rng.randint(10, 300)})
        current_text = "the current request " + "y" * rng.randint(5, 200)
        current = [{"role": "user", "content": current_text}]
        usable = rng.randint(150, 6000)

        engine = make_engine(usable_budget=usable)
        try:
            plan = engine.build(history=history, current_request=current)
        except ContextOverflowError:
            return  # legal only when mandatory alone exceeds budget
        assert plan.token_estimate <= usable
        assert plan.within_budget
        assert plan.messages[-1]["content"] == current_text
        assert sources(plan).count("current_request") == 1

    def test_current_request_preserved_under_pressure(self):
        history = [
            {"role": "user", "content": "u1 " + "x" * 500},
            {"role": "assistant", "content": "a1 " + "a" * 500},
        ]
        current = [{"role": "user", "content": "precious current request"}]
        plan = make_engine(usable_budget=700).build(history=history, current_request=current)
        assert plan.messages[-1]["content"] == "precious current request"
        assert sources(plan).count("recent_turn") <= 1

    def test_overflow_when_mandatory_alone_cannot_fit(self):
        history = [{"role": "system", "content": "s" * 400}]
        current = [{"role": "user", "content": "u" * 400}]
        with pytest.raises(ContextOverflowError) as excinfo:
            make_engine(usable_budget=150).build(history=history, current_request=current)
        assert excinfo.value.usable_budget == 150
        assert excinfo.value.required_tokens > 150

    def test_tool_definitions_consume_budget(self):
        big_tools = [
            {"type": "function", "function": {"name": f"t{i}", "parameters": {"x": "p" * 400}}}
            for i in range(4)
        ]
        history = [
            {"role": "user", "content": "old turn " + "x" * 2000},
            {"role": "assistant", "content": "answer " + "a" * 2000},
        ]
        current = [{"role": "user", "content": "current"}]
        without_tools = make_engine(usable_budget=1200).build(
            history=history, current_request=current
        )
        with_tools = make_engine(usable_budget=1200).build(
            history=history, current_request=current, tools=big_tools
        )
        assert with_tools.tools_tokens > 0
        # tool definitions squeeze the old turn out entirely
        assert len(with_tools.messages) < len(without_tools.messages)
        assert with_tools.messages[-1]["content"] == "current"
        assert with_tools.token_estimate <= 1200

    def test_tools_counted_in_mandatory_overflow(self):
        big_tools = [
            {"type": "function", "function": {"name": "t", "parameters": {"x": "p" * 4000}}}
        ]
        history = []
        current = [{"role": "user", "content": "u" * 200}]
        with pytest.raises(ContextOverflowError):
            make_engine(usable_budget=300).build(
                history=history, current_request=current, tools=big_tools
            )

    def test_insufficient_budget_drops_oldest_turns_first_contiguously(self):
        history = [
            {"role": "user", "content": "oldest " + "x" * 300},
            {"role": "assistant", "content": "r1 " + "a" * 300},
            {"role": "user", "content": "middle " + "x" * 300},
            {"role": "assistant", "content": "r2 " + "a" * 300},
            {"role": "user", "content": "newest " + "x" * 300},
            {"role": "assistant", "content": "r3 " + "a" * 300},
        ]
        current = [{"role": "user", "content": "current"}]
        plan = make_engine(usable_budget=2600).build(history=history, current_request=current)
        contents = [m["content"] for m in plan.messages]
        names_in_order = ["oldest", "middle", "newest"]
        present = [
            n for n in names_in_order if any(c.startswith(n + " ") for c in contents)
        ]
        assert present, "newest window must survive"
        assert present == names_in_order[-len(present) :]  # contiguous suffix
        assert contents[-1] == "current"
        assert plan.token_estimate <= 2600

    def test_retrieval_cap_limited_by_remaining_budget(self):
        assembly = AssemblySettings(retrieved_budget_tokens=4000, max_retrieved_items=10)
        retrieved = [
            memory_item(f"m{i}", f"distinct unique fact {i} " + "z" * 60, semantic=0.9 - i * 0.01)
            for i in range(6)
        ]
        history = [
            {"role": "user", "content": "h " + "x" * 2200},
            {"role": "assistant", "content": "a " + "y" * 2200},
        ]
        current = [{"role": "user", "content": "current"}]
        plan = make_engine(usable_budget=2600, assembly=assembly).build(
            history=history, current_request=current, retrieved=retrieved
        )
        retrieved_selected = [s for s in plan.selected_items if s.source.value == "memory"]
        assert sum(s.tokens for s in retrieved_selected) <= 1200

    def test_retrieved_cap_configured_maximum_respected(self):
        assembly = AssemblySettings(retrieved_budget_tokens=120, max_retrieved_items=10)
        retrieved = [
            memory_item(f"m{i}", f"distinct unique fact {i} " + "z" * 40, semantic=0.9 - i * 0.01)
            for i in range(6)
        ]
        plan = make_engine(assembly=assembly).build(
            history=[],
            current_request=[{"role": "user", "content": "current"}],
            retrieved=retrieved,
        )
        memory_selected = [s for s in plan.selected_items if s.source.value == "memory"]
        assert sum(s.tokens for s in memory_selected) <= 120
        assert any(reason == "budget" for reason in dropped_map(plan).values())

    def test_max_retrieved_items_respected(self):
        assembly = AssemblySettings(max_retrieved_items=2, retrieved_budget_tokens=9999)
        retrieved = [
            memory_item(f"m{i}", f"totally different topic {i}", semantic=0.5)
            for i in range(5)
        ]
        plan = make_engine(assembly=assembly).build(
            history=[],
            current_request=[{"role": "user", "content": "current"}],
            retrieved=retrieved,
        )
        assert len([s for s in plan.selected_items if s.source.value == "memory"]) == 2

    def test_empty_retrieval_keeps_recent_and_request_within_budget(self):
        history = [
            {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "hello"},
        ]
        current = [{"role": "user", "content": "how are you?"}]
        plan = make_engine(usable_budget=500).build(
            history=history, current_request=current, retrieved=[]
        )
        assert [m["content"] for m in plan.messages] == ["hi there", "hello", "how are you?"]
        assert plan.dropped_items == []
        assert plan.within_budget

    def test_pinned_context_packed_after_system_before_history(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "ans"},
        ]
        current = [{"role": "user", "content": "current"}]
        pinned = [{"role": "system", "content": "PINNED DIRECTIVE"}]
        plan = make_engine().build(history=history, current_request=current, pinned=pinned)
        contents = [m.get("content") for m in plan.messages]
        assert contents.index("PINNED DIRECTIVE") > contents.index("sys")
        assert contents.index("PINNED DIRECTIVE") < contents.index("old")

    def test_retrieval_blocks_render_between_pinned_and_history(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old turn"},
            {"role": "assistant", "content": "ans"},
        ]
        current = [{"role": "user", "content": "current"}]
        plan = make_engine().build(
            history=history,
            current_request=current,
            pinned=[],
            retrieved=[memory_item("m1", "retrieved knowledge")],
        )
        contents = [m.get("content") for m in plan.messages]
        mem_index = next(
            i for i, c in enumerate(contents) if "[retrieved memory:fact id=m1]" in c
        )
        assert contents.index("sys") < mem_index < contents.index("old turn")


# --------------------------------------------------------------- atomicity


class TestAtomicity:
    def _tool_unit(self, tag: str, size: int = 250) -> list[dict]:
        return [
            {"role": "user", "content": f"{tag} question " + "x" * size},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{tag}",
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call_{tag}", "content": "result " + "r" * size},
            {"role": "assistant", "content": f"{tag} final answer"},
        ]

    def test_current_tool_interaction_atomic(self):
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
        current = self._tool_unit("cur", size=30)
        for usable in (600, 900, 1400, 2400):
            plan = make_engine(usable_budget=usable).build(
                history=history, current_request=current
            )
            sources_list = sources(plan)
            # exactly one atomic CURRENT_REQUEST candidate — never split into
            # CURRENT_REQUEST + RECENT_TURN, never duplicated (final review §5)
            assert sources_list.count("current_request") == 1
            request_item = next(
                s for s in plan.selected_items if s.source.value == "current_request"
            )
            assert request_item.key == "current_request"
            assert request_item.tokens == make_engine()._counter.messages(current)
            assert_whole_units(plan.messages, [history[:2], history, current])
            roles = [m["role"] for m in plan.messages]
            if "tool" in roles:
                idx = roles.index("tool")
                assert roles[idx - 1] == "assistant"
                assert any(
                    m.get("tool_calls") and m["tool_calls"][0]["id"] == "call_cur"
                    for m in plan.messages[:idx]
                )

    @pytest.mark.parametrize("seed", range(12))
    def test_property_no_partial_units_ever(self, seed):
        rng = random.Random(seed)
        history: list[dict] = []
        all_units: list[list[dict]] = []
        for i in range(rng.randint(1, 6)):
            if rng.random() < 0.4:
                unit = self._tool_unit(f"t{i}", size=rng.randint(20, 150))
            else:
                unit = [
                    {"role": "user", "content": f"u{i} " + "x" * rng.randint(20, 150)},
                    {"role": "assistant", "content": "resp " + "a" * rng.randint(10, 120)},
                ]
            all_units.append(list(unit))
            history.extend(unit)
        system = [{"role": "system", "content": "sys"}]
        all_units.append(list(system))
        history = [*system, *history]
        current = [{"role": "user", "content": "final current"}]
        all_units.append(list(current))

        engine = make_engine(usable_budget=rng.randint(400, 5000))
        try:
            plan = engine.build(history=history, current_request=current)
        except ContextOverflowError:
            return
        assert_whole_units(plan.messages, all_units)


class TestRobustCurrentRequestDetection:
    """Current request = last USER-initiated interaction (review P3)."""

    def test_trailing_system_after_assistant(self):
        messages = [
            {"role": "system", "content": "s0"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "system", "content": "s1"},
        ]
        history, current = separate_current_request(messages)
        assert current == [{"role": "user", "content": "q"},
                           {"role": "assistant", "content": "a"}]
        assert history[0] == {"role": "system", "content": "s0"}
        assert history[-1] == {"role": "system", "content": "s1"}
        # same message multiset, systems normalized into history position
        assert len(history) + len(current) == len(messages)
        assert {json.dumps(m, sort_keys=True) for m in history + current} == {
            json.dumps(m, sort_keys=True) for m in messages
        }

    def test_tool_unit_before_trailing_user(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "r"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "final question"},
        ]
        _, current = separate_current_request(messages)
        assert current == [{"role": "user", "content": "final question"}]

