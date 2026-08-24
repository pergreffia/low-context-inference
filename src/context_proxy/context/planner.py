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


UnitKind = Literal["system", "turn", "prefill"]

# Roles that carry trusted instructions and share the protected tier.
INSTRUCTION_ROLES = frozenset({"system", "developer"})


def is_instruction_role(role: Any) -> bool:
    return role in INSTRUCTION_ROLES


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


def segment_messages(
    messages: list[dict[str, Any]], counter: TokenCounter | None = None
) -> list[Unit]:
    """Group messages into atomic interaction units (M2.1 §3).

    A turn starts at a user message and spans everything up to the next user
    message: user -> assistant(tool_call) -> tool(result) -> assistant(final)
    stays one indivisible unit. System/developer messages are their own
    protected units emitted immediately — they never close an open interaction,
    so a developer note landing mid-turn cannot orphan the assistant (review
    B1); instruction units are normalized into the request head by packing.
    Messages preceding the first user message (rare assistant prefill) form a
    droppable prefill unit so no assistant is ever retained without its
    interaction.
    """
    counter = counter or TokenCounter()
    units: list[Unit] = []
    prefill: list[dict[str, Any]] = []
    turn: list[dict[str, Any]] | None = None

    def close_turn() -> None:
        nonlocal turn
        if turn:
            units.append(Unit("turn", tuple(turn), counter.messages(turn)))
            turn = None

    def close_prefill() -> None:
        if prefill:
            units.append(Unit("prefill", tuple(prefill), counter.messages(prefill)))
            prefill.clear()

    for message in messages:
        role = message.get("role")
        if is_instruction_role(role):
            # Trusted instructions are emitted as their own protected unit
            # WITHOUT closing the open interaction: a developer/system note
            # landing between a user and its assistant must never orphan the
            # assistant (M0–M6 review B1). Instructions are normalized into
            # the head of the assembled request by the packing layer.
            close_prefill()
            units.append(Unit("system", (message,), counter.messages([message])))
        elif role == "user":
            close_turn()
            close_prefill()
            turn = [message]
        elif turn is not None:
            turn.append(message)
        else:
            prefill.append(message)
    close_turn()
    close_prefill()
    return units


def plan_context(
    history: list[dict[str, Any]],
    *,
    current_request: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    usable_budget: int,
    reserved_tokens: int = 0,
    counter: TokenCounter | None = None,
) -> ContextPlan:
    """Build the largest within-budget raw context from recent messages.

    The current request is STRUCTURALLY separate (shared semantics with the
    M4 engine via separate_current_request): it is mandatory whenever it fits
    and can never be dropped under budget pressure, even when instruction
    units (system/developer) trail it in the inbound payload.

    Priority when trimming (master prompt §16): system/developer instructions
    and the current request are never sacrificed; oldest non-instruction
    units are dropped whole. reserved_tokens covers future context consumers
    so the final context can never exceed usable_budget once they land.
    Raises ContextOverflowError if no valid plan exists.
    """
    counter = counter or TokenCounter()
    tools_tokens = counter.tools(tools)

    current_unit = (
        Unit("turn", tuple(current_request), counter.messages(current_request))
        if current_request
        else None
    )
    current_tokens = current_unit.tokens if current_unit else 0

    original = segment_messages(history, counter)
    mandatory = (
        sum(u.tokens for u in original if u.is_system) + tools_tokens + reserved_tokens
        + current_tokens
    )
    if mandatory > usable_budget:
        raise ContextOverflowError(mandatory, usable_budget)

    available_for_history = usable_budget - tools_tokens - reserved_tokens - current_tokens
    kept: list[Unit | None] = list(original)
    total = sum(unit.tokens for unit in original)

    if total > available_for_history:
        # Drop oldest non-instruction units whole.
        for position, unit in enumerate(original):
            if total <= available_for_history:
                break
            if unit.is_system:
                continue
            total -= unit.tokens
            kept[position] = None

    final_units = [u for u in kept if u is not None]
    if current_unit is not None:
        final_units.append(current_unit)
    final_total = sum(u.tokens for u in final_units)
    if final_total + tools_tokens > usable_budget:
        raise ContextOverflowError(final_total + tools_tokens, usable_budget)

    dropped_units = len(original) - (len(final_units) - (1 if current_unit else 0))
    dropped_tokens = sum(u.tokens for u in original) - (
        final_total - current_tokens
    )
    logger.info(
        "context_planned",
        extra={
            "units_total": len(original),
            "units_kept": len(final_units),
            "dropped_units": dropped_units,
            "dropped_tokens": max(0, dropped_tokens),
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
        dropped_tokens=max(0, dropped_tokens),
        details={
            "units_total": len(original),
            "units_kept": len(final_units),
            "system_tokens": sum(u.tokens for u in final_units if u.is_system),
            "current_request_tokens": current_tokens,
        },
    )
