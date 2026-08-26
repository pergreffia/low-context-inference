# Fix Prompt — Client History Reconciliation v2 / OpenCode Compatibility

Timestamp: 2026-08-26 16:00 CEST

## Objective

Fix the conversation history reconciliation so that Low Context Inference remains compatible with agentic clients such as OpenCode.

The current implementation compares incoming messages against persisted messages positionally and requires JSON-level equality.

This is too strict.

OpenCode maintains a durable session transcript but can build a different model-visible history through compaction, pruning, summarization, reasoning normalization, tool processing, and other transformations.

Therefore:

    persisted history != client-provided model context

is a valid and expected condition.

The goal is NOT to weaken history consistency.

The goal is to distinguish legitimate client-side history transformations from actual conversation/session inconsistencies.

---

# 1. Inspect the current implementation first

Inspect the complete current implementation before changing anything.

At minimum inspect:

- history reconciliation
- conversation/session handling
- message persistence
- message IDs / message metadata
- request parsing
- context assembly
- capture/persistence
- tool-call persistence
- existing reconciliation tests
- OpenCode compatibility tests
- current history divergence diagnostics

Do not assume the exact file paths.

Also inspect the current database schema for message identity and ordering fields.

Determine exactly which identifiers are currently available for:

- conversation
- message
- interaction/turn
- tool call
- persisted ordering

Do not introduce a new identity mechanism if an existing one can safely be reused.

---

# 2. Identify the architectural problem

The current reconciliation is conceptually equivalent to:

    for index, incoming_message in enumerate(messages):
        assert incoming_message == persisted_messages[index]

This must no longer be the primary reconciliation rule.

It breaks legitimate scenarios such as:

    full persisted history:
        A B C D E F G

    client-visible history:
        A B C [summary] G

or:

    persisted:
        assistant(reasoning_content=A)

    incoming:
        assistant(reasoning_content=B)

when the client legitimately reconstructed/normalized the model-visible representation.

It also breaks when content representation changes while preserving conversation semantics.

---

# 3. Preserve the important invariants

Do NOT remove consistency checking.

The following invariants must remain enforced:

## Conversation isolation

A request associated with conversation/session X must never be allowed to mutate or use persisted state belonging to conversation/session Y.

## Persisted history is authoritative

The persisted conversation remains the durable source of truth.

The incoming client history must NEVER overwrite the persisted transcript.

## No silent history corruption

A genuinely unrelated or incompatible incoming history must still be rejected.

## Ordering integrity

The system must still detect impossible message ordering or invalid replay.

## Tool-call integrity

Tool calls and tool results must remain correctly associated.

## Interaction atomicity

Existing interaction/turn atomicity guarantees must remain unchanged.

## Exactly-once persistence

Do not introduce duplicate persistence as a side effect of the new reconciliation algorithm.

---

# 4. Model the incoming history as a client projection

Treat the incoming messages as:

    client_projection(persisted_conversation)

rather than:

    canonical_persisted_history

This means the client is allowed to:

- compact old messages;
- summarize old messages;
- prune old messages;
- normalize message representation;
- omit messages no longer required by its current context;
- transform reasoning representation;
- transform multimodal representation where semantically equivalent;
- rebuild model-visible context.

These operations must not automatically produce a history conflict.

---

# 5. Design a reconciliation algorithm based on identity/anchors

Do NOT implement a simple:

    "ignore content differences"

solution.

Do NOT implement:

    "compare only roles"

or:

    "compare only the last message"

or:

    "accept any history with the same conversation_id"

The algorithm must establish that the incoming history still belongs to the same logical conversation.

Prefer existing persistent message identifiers or stable metadata if the current project already has them.

Determine whether OpenCode-visible message IDs can be mapped reliably to persisted messages.

If the client does not expose stable IDs for all messages, use robust anchors based on the portions of history that can be identified reliably.

Possible anchor candidates include:

- persisted message ID when available;
- tool_call_id;
- interaction/turn boundaries;
- stable canonical message fingerprints;
- ordered combinations of role + semantic content;
- known conversation checkpoints.

Do not rely exclusively on raw positional indexes.

---

# 6. Support client-side compaction

This is a mandatory scenario.

Given persisted history:

    A B C D E F G

and incoming history:

    A B C SUMMARY G

The request must NOT fail solely because D/E/F are no longer present or have been replaced by a summary.

The system must recognize that:

    A B C

and:

    G

still anchor the same conversation, where such anchors can be established safely.

The persisted D/E/F messages remain untouched.

The summary is treated as a client-side projection.

Do not persist the summary as a replacement for D/E/F unless the existing application explicitly models client summaries as separate persisted events.

---

# 7. Support client-side pruning/truncation

Also support:

    persisted:
        A B C D E F G

    incoming:
        D E F G

or:

    persisted:
        A B C D E F G

    incoming:
        E F G

when the incoming request is a legitimate projection of the same conversation.

Do not require the client to resend the entire transcript.

Do not require a full common prefix if the available anchors prove continuity safely.

---

# 8. Support reasoning normalization

Do NOT use exact equality of:

    reasoning_content

as a mandatory history identity condition.

Reasoning may be:

- normalized;
- transformed;
- omitted;
- reconstructed;
- compacted;
- represented differently by the client.

The persisted reasoning should still be retained.

However, if reasoning is the only difference between an otherwise valid conversation projection and persisted history, it must not cause a false history conflict.

Do NOT delete reasoning from persistence.

Do NOT globally ignore all assistant fields.

---

# 9. Support content representation normalization

Do not assume that:

    content = "hello"

and:

    content = [{"type":"text","text":"hello"}]

are necessarily different semantic messages.

Inspect the project's current content normalization/canonicalization utilities.

If the project already has canonical content helpers, reuse them.

If safe, use canonicalized semantic content for reconciliation.

Do not change the persisted representation merely to make reconciliation pass.

Persisted raw data must remain available according to the existing storage design.

---

# 10. Preserve real conflict detection

The following must still be rejected.

## Wrong conversation/session

Incoming history belongs to a different conversation/session.

## Impossible ordering

Incoming history contradicts the established sequence in a way that cannot be explained by compaction/pruning.

## Conflicting anchored message

An established anchor identifies message X, but incoming history contains a genuinely incompatible message at that same logical point.

Example:

    persisted:
        message_id=123
        role=user
        content="deploy production"

    incoming projection:
        same message identity/anchor
        role=user
        content="delete production"

This must remain a conflict.

## Invalid tool relationship

Tool result references a tool call that does not belong to the conversation.

## Forked history

The incoming history establishes a new branch that cannot be reconciled with the persisted conversation.

Do not silently accept these cases.

---

# 11. Do not mutate persisted history during reconciliation

Reconciliation must be read/validate only.

It must NOT:

- delete old messages;
- replace messages;
- rewrite message content;
- overwrite reasoning;
- replace tool calls;
- replace multimodal payloads;
- rewrite timestamps.

Only the normal persistence path for the current interaction may append new canonical messages.

---

# 12. Keep current-session request semantics

The existing conversation/session ID remains mandatory for state isolation.

Do not infer conversation identity solely from message content.

Do not allow two conversations with similar histories to be merged.

---

# 13. Handle the current OpenCode examples

Add regression coverage reproducing these real failures.

## Case A — reasoning difference

Persisted:

    assistant
    content = X
    reasoning_content = A

Incoming:

    assistant
    content = X
    reasoning_content = B

Expected:

    accepted as the same logical conversation projection

Persisted reasoning A must remain unchanged.

---

## Case B — content representation difference

Persisted:

    content = "hello"

Incoming:

    content = [{"type":"text","text":"hello"}]

If the project's semantic normalization considers these equivalent:

    accepted

Otherwise explicitly document the chosen canonicalization rule and test it.

---

# 14. Add compaction regression tests

At minimum:

## Test 1 — prefix + summary + suffix

Persist:

    A B C D E F G

Incoming:

    A B C SUMMARY G

Expected:

    accepted

Persisted D/E/F remain unchanged.

---

## Test 2 — truncated history

Persist:

    A B C D E F G

Incoming:

    E F G

Expected:

    accepted if continuity can be safely established.

---

## Test 3 — compacted history + new user message

Persist:

    A B C D E F G

Incoming:

    A B C SUMMARY G H

Expected:

    accepted

H is the new interaction.

---

## Test 4 — genuine fork

Persist:

    A B C D

Incoming:

    A B X D

Expected:

    rejected

---

## Test 5 — wrong conversation

Persisted conversation:

    conversation X

Incoming request:

    conversation Y

Expected:

    rejected / isolated according to existing contract.

---

## Test 6 — duplicate/replayed request

Submit the same incoming projection multiple times.

Expected:

    no duplicate persistence
    no false history conflict

---

# 15. Add tool-call regression tests

Verify that compaction/reconciliation does not break tool relationships.

Example persisted:

    assistant(tool_call_id=T1)
    tool(result_for=T1)
    assistant(content=...)

Incoming projection may omit older tool messages if they are inside a compacted region.

Expected:

    accepted if the omitted region is a legitimate client projection.

But:

    tool_result(tool_call_id=UNKNOWN)

must remain rejected.

---

# 16. Add multimodal regression tests

Verify that equivalent multimodal representations do not create false conflicts where the project's canonicalization defines them as equivalent.

At minimum test:

- text-only content;
- text + image;
- content-part representation;
- compacted history containing a multimodal turn.

Do not load large binary media unnecessarily during reconciliation.

Use existing media identifiers/registry where applicable.

---

# 17. Improve diagnostics

Keep the existing safe diagnostic fields:

    persisted_sha256
    incoming_sha256
    different_fields

Extend diagnostics where useful with a reason/category such as:

    history_projection_accepted
    history_compaction_detected
    history_truncation_detected
    history_conflict
    history_wrong_conversation

Do NOT log message contents.

Do NOT log:

- prompts;
- assistant responses;
- tool arguments;
- credentials;
- image payloads.

Diagnostics should make it clear why a request was accepted or rejected.

---

# 18. Update tests that currently assert positional equality

Search the entire test suite for assumptions equivalent to:

    incoming[i] == persisted[i]

or tests expecting any representation-level change to raise a conflict.

Update them to reflect the new reconciliation contract.

Do not simply delete those tests.

Replace them with tests for the actual invariant:

    logical conversation continuity

---

# 19. Performance requirements

Do not turn reconciliation into an O(n²) algorithm.

The common case must remain efficient.

Prefer:

- indexed message lookup;
- hashes/fingerprints;
- maps keyed by stable IDs;
- bounded anchor searches;
- database queries scoped to the relevant conversation.

Do not load an arbitrarily large conversation into Python for every request if the existing system can query only the required range.

This matters for long-running OpenCode sessions.

---

# 20. Security requirements

The new reconciliation algorithm must not create a history injection vulnerability.

An attacker must not be able to fabricate a matching anchor and gain access to another conversation.

Conversation/session identity remains authoritative.

Any client-supplied identifier must be validated against persisted state.

Do not trust client message IDs as globally unique authorization credentials.

They are reconciliation hints, not authentication.

---

# 21. Preserve capture improvements

The previous fix for assistant reasoning preservation remains valid.

Do not revert:

- reasoning_content capture;
- reasoning canonicalization;
- tool-call capture;
- divergence fingerprints.

The new reconciliation layer should operate on the canonicalized representation produced by the existing capture/persistence system.

---

# 22. Full regression suite

Run:

    ruff check .
    pytest -q

Then run the PostgreSQL integration suite.

Run the existing Docker/Compose smoke tests.

Run the OpenCode compatibility test if available.

Verify:

- conversation isolation;
- persistence;
- streaming;
- tool calls;
- multimodal;
- compaction;
- rate limiting;
- circuit breaker;
- token accounting.

---

# 23. Acceptance criteria

- [ ] Positional full-JSON equality is no longer the reconciliation mechanism.
- [ ] Conversation/session isolation remains strict.
- [ ] Persisted history remains the durable source of truth.
- [ ] Client-side compaction is accepted.
- [ ] Client-side truncation/pruning is accepted when continuity is safely established.
- [ ] Reasoning representation differences do not create false conflicts.
- [ ] Legitimate content representation differences are handled through canonicalization.
- [ ] Genuine conflicting history is still rejected.
- [ ] Wrong conversation/session is still rejected.
- [ ] Tool-call integrity remains enforced.
- [ ] Multimodal integrity remains enforced.
- [ ] Persisted history is never overwritten by client history.
- [ ] Replayed requests remain idempotent.
- [ ] Reconciliation does not become O(n²).
- [ ] No message contents are leaked into diagnostics.
- [ ] Existing M0–M6 invariants remain intact.
- [ ] ruff check . passes.
- [ ] Full test suite passes.
- [ ] PostgreSQL integration passes.
- [ ] Docker smoke tests pass.
- [ ] OpenCode compatibility scenario passes.

---

# 24. Scope boundary

This is a targeted redesign of history reconciliation only.

Do NOT:

- redesign persistence;
- redesign the context planner;
- redesign memory/retrieval;
- redesign embeddings;
- redesign rate limiting;
- redesign the circuit breaker;
- change the inference API contract unnecessarily;
- introduce distributed state;
- add external infrastructure;
- create a new session model;
- remove conversation isolation.

Do not implement a generic "accept any history" mechanism.

The goal is:

    strict logical conversation continuity

instead of:

    strict JSON equality.

---

# 25. Final report

Report:

1. current reconciliation algorithm;
2. new reconciliation algorithm;
3. identity/anchor mechanism used;
4. how compaction is detected;
5. how truncation/pruning is detected;
6. how reasoning/content normalization is handled;
7. how genuine conflicts are still detected;
8. files changed;
9. tests added;
10. tests executed and results;
11. OpenCode compatibility result;
12. any remaining limitations.

Do not claim completion unless the real OpenCode scenarios are covered by deterministic regression tests.
