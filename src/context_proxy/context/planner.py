"""Dynamic context budgeting (master prompt §14–§16).

The planner segments the incoming message list into atomic interaction units
and, when they exceed the usable budget, drops complete oldest units while
preserving:

- system messages (priority 1);
- the current request (never dropped);
- tool-call/tool-result association (one interaction = one unit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from context_proxy.context.tokens import TokenCounter

logger = logging.getLogger(__name__)


class ContextOverflowError(Exception):
    """The request cannot fit the usable budget even after trimming history."""

    def __init__(self, required_tokens: int, usable_budget: int):
        self.required_tokens = required_tokens
        self.usable_budget = usable_budget
        super().__init__(
            f"request requires {required_tokens} tokens, "
            f"usable budget is {usable_budget}"
        )


UnitKind = Literal["system", "user", "assistant", "interaction"]


@dataclass(frozen=True)
class Unit:
    kind: UnitKind
    messages: tuple[dict[str, Any], ...]
    tokens: int

    @property
    def is_system(self) -> bool:
        return self.kind == "system"


@dataclass
class ContextPlan:
    messages: list[dict[str, Any]]
    total_tokens: int
    tools_tokens: int
    usable_budget: int
    dropped_units: int = 0
    dropped_tokens: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def within_budget(self) -> bool:
        return self.total_tokens + self.tools_tokens <= self.usable_budget


def segment_messages(messages: list[dict[str, Any]], counter: TokenCounter) -> list[Unit]:
    """Group messages into atomic units; tool results stay with their calls."""
    units: list[Unit] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            call_ids = {
                tc.get("id")
                for tc in message["tool_calls"]
                if isinstance(tc, dict) and tc.get("id")
            }
            end = index + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                if call_ids and messages[end].get("tool_call_id") not in call_ids:
                    break  # belongs to a different assistant turn
                end += 1
            chunk = tuple(messages[index:end])
            units.append(Unit("interaction", chunk, counter.messages(list(chunk))))
            index = end
            continue
        if role == "system":
            kind: UnitKind = "system"
        else:
            # user / assistant / orphan tool result -> individually droppable
            kind = "user" if role == "user" else "assistant"
        chunk = (message,)
        units.append(Unit(kind, chunk, counter.messages(list(chunk))))
        index += 1
    return units


def plan_context(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    usable_budget: int,
    counter: TokenCounter | None = None,
) -> ContextPlan:
    """Build the largest within-budget raw context from recent messages.

    Priority when trimming (master prompt §16): system prompts and the current
    request are never sacrificed; oldest non-system units are dropped first.
    Raises ContextOverflowError if no valid plan exists.
    """
    counter = counter or TokenCounter()
    tools_tokens = counter.tools(tools)
    available_for_messages = usable_budget - tools_tokens

    original = segment_messages(messages, counter)
    kept: list[Unit | None] = list(original)
    total = sum(unit.tokens for unit in original)

    if total > available_for_messages:
        # Never drop the last unit (current request) nor system units.
        droppable_positions = [i for i, u in enumerate(original[:-1]) if not u.is_system]
        for position in droppable_positions:
            if total <= available_for_messages:
                break
            total -= original[position].tokens
            kept[position] = None

    final_units = [u for u in kept if u is not None]
    final_total = sum(u.tokens for u in final_units)
    if final_total + tools_tokens > usable_budget:
        raise ContextOverflowError(final_total + tools_tokens, usable_budget)

    dropped_units = len(original) - len(final_units)
    dropped_tokens = sum(u.tokens for u in original) - final_total
    logger.info(
        "context_planned",
        extra={
            "units_total": len(original),
            "units_kept": len(final_units),
            "dropped_units": dropped_units,
            "dropped_tokens": dropped_tokens,
            "message_tokens": final_total,
            "tools_tokens": tools_tokens,
            "usable_budget": usable_budget,
        },
    )
    return ContextPlan(
        messages=[m for unit in final_units for m in unit.messages],
        total_tokens=final_total,
        tools_tokens=tools_tokens,
        usable_budget=usable_budget,
        dropped_units=dropped_units,
        dropped_tokens=dropped_tokens,
        details={
            "units_total": len(original),
            "units_kept": len(final_units),
            "system_tokens": sum(u.tokens for u in final_units if u.is_system),
        },
    )
