# M7 — History Reconciliation v2 / OpenCode Compatibility

Timestamp: 2026-08-26 19:44 CEST

## Objective

Make Low Context Inference compatible with agentic clients such as OpenCode, whose model-visible context may legitimately differ from the durable conversation transcript because of compaction, pruning, summarization, reasoning normalization, tool processing, and other transformations.

The current positional/full-JSON reconciliation is too strict:

    persisted[i] == incoming[i]

M7 must replace that model with strict **logical conversation continuity** while preserving conversation isolation, persistence integrity, tool-call integrity, and security invariants.

Do NOT implement a generic "accept any history" mechanism.

---

## 1. Inspect before modifying

Inspect the complete current implementation and tests before changing anything.

At minimum inspect:

- history reconciliation;
- conversation/session handling;
- message persistence and schema;
- message IDs and ordering metadata;
- request parsing;
- context assembly;
- capture/persistence;
- tool-call persistence;
- current divergence diagnostics;
- reconciliation tests;
- OpenCode compatibility tests.

Determine which existing identifiers can be used for reliable anchors. Do not introduce a new identity mechanism if an existing one can safely be reused.

---

## 2. Required architectural model

Treat incoming messages as a client projection of the persisted conversation:

    client_projection(persisted_conversation)

not as the canonical persisted transcript.

The persisted conversation remains the durable source of truth.

The client may legitimately:

- compact old messages;
- summarize old messages;
- prune/truncate history;
- normalize message representation;
- omit messages no longer required by its current context;
- transform reasoning representation;
- transform semantically equivalent multimodal content;
- rebuild model-visible context.

These operations must not automatically produce `history_conflict`.

---

## 3. Preserve mandatory invariants

The following must remain strict:

### Conversation isolation

A request for conversation/session X must never access or mutate state belonging to Y.

### Persisted history is authoritative

Incoming client history must NEVER overwrite the persisted transcript.

### No silent corruption

Genuinely incompatible history must still be rejected.

### Ordering integrity

Impossible ordering and invalid replay must remain detectable.

### Tool-call integrity

Tool calls and results must remain correctly associated.

### Interaction atomicity

Existing interaction/turn atomicity guarantees must remain unchanged.

### Exactly-once persistence

Reconciliation must not introduce duplicate persistence.

---

## 4. Replace positional equality with identity/anchor reconciliation

Do NOT solve this by:

- ignoring content differences;
- comparing only roles;
- comparing only the latest message;
- accepting any history with the same conversation ID;
- disabling reconciliation.

Build a reconciliation algorithm based on reliable identity/anchors.

Prefer existing persistent message IDs or stable metadata.

Possible anchors, depending on what the current implementation supports:

- persisted message ID;
- tool_call_id;
- interaction/turn boundaries;
- stable canonical message fingerprints;
- ordered role + semantic-content anchors;
- known conversation checkpoints.

Do not rely exclusively on raw positional indexes.

Client-supplied IDs are reconciliation hints, not authentication credentials. Validate them against persisted conversation state.

---

## 5. Support client-side compaction

Mandatory scenario.

Persisted:

    A B C D E F G

Incoming:

    A B C SUMMARY G

Expected: accepted when continuity can be established safely.

D/E/F remain persisted and untouched.

The summary must not replace persisted D/E/F.

---

## 6. Support pruning/truncation

Support legitimate projections such as:

    persisted: A B C D E F G
    incoming:   D E F G

and:

    persisted: A B C D E F G
    incoming:   E F G

when reliable anchors establish continuity.

Do not require the client to resend the entire transcript.

Do not require a full common prefix when safe continuity can be established otherwise.

---

## 7. Reasoning normalization

Do not use exact `reasoning_content` equality as mandatory conversation identity.

Reasoning may be normalized, transformed, omitted, reconstructed, compacted, or represented differently by the client.

Persisted reasoning must still be retained.

If reasoning is the only difference in an otherwise valid projection, it must not produce a false conflict.

Do not delete reasoning from persistence and do not globally ignore all assistant fields.

---

## 8. Content representation normalization

Inspect and reuse existing canonical content utilities where available.

If the project considers these semantically equivalent:

    content = "hello"

and:

    content = [{"type": "text", "text": "hello"}]

reconciliation must not report a false conflict.

Do not change persisted representation merely to make reconciliation pass.

---

## 9. Genuine conflicts must remain rejected

Reject at minimum:

### Wrong conversation/session

Incoming history belongs to another conversation/session.

### Impossible ordering

The incoming history contradicts established sequence in a way that cannot be explained by projection/compaction/pruning.

### Conflicting anchor

A known anchor identifies a persisted message, but incoming history contains genuinely incompatible content at that logical point.

Example:

    persisted:
        message_id=123
        role=user
        content="deploy production"

    incoming:
        same anchor
        role=user
        content="delete production"

This remains a conflict.

### Invalid tool relationship

A tool result references a tool call that does not belong to the conversation.

### Unreconcilable fork

Incoming history represents a new branch that cannot safely be associated with the persisted conversation.

---

## 10. Reconciliation must not mutate history

Reconciliation is read/validate only.

It must NOT:

- delete old messages;
- replace messages;
- rewrite message content;
- overwrite reasoning;
- replace tool calls;
- replace multimodal payloads;
- rewrite timestamps.

Only the normal current-interaction persistence path may append canonical messages.

---

## 11. Current OpenCode regressions

Add deterministic regression coverage for the two observed failures.

### Case A — reasoning difference

Persisted:

    assistant
    content = X
    reasoning_content = A

Incoming:

    assistant
    content = X
    reasoning_content = B

Expected: accepted as the same logical conversation projection when continuity is otherwise valid.

Persisted reasoning A remains unchanged.

### Case B — content representation difference

Persisted:

    content = "hello"

Incoming:

    content = [{"type":"text","text":"hello"}]

If existing semantic canonicalization considers them equivalent, accept them. Otherwise document and test the chosen rule explicitly.

---

## 12. Required compaction tests

Add at minimum:

1. Prefix + summary + suffix

    persisted: A B C D E F G
    incoming:  A B C SUMMARY G

    → accepted
    → D/E/F remain unchanged

2. Truncated history

    persisted: A B C D E F G
    incoming:  E F G

    → accepted when continuity is safely established

3. Compacted history + new user message

    persisted: A B C D E F G
    incoming:  A B C SUMMARY G H

    → accepted
    → H is the new interaction

4. Genuine fork

    persisted: A B C D
    incoming:  A B X D

    → rejected

5. Wrong conversation

    persisted conversation X
    incoming conversation Y

    → rejected/isolated according to the existing contract

6. Duplicate/replayed request

    Submit the same incoming projection repeatedly.

    → no duplicate persistence
    → no false conflict

---

## 13. Tool-call tests

Verify compaction/reconciliation does not break tool relationships.

Example persisted history:

    assistant(tool_call_id=T1)
    tool(result_for=T1)
    assistant(content=...)

An incoming projection may omit older tool messages if they are inside a compacted region.

Expected: accepted when the projection is otherwise valid.

But:

    tool_result(tool_call_id=UNKNOWN)

must remain rejected.

Also preserve existing streaming, multiple-tool-call, replay, and custom-tool behavior.

---

## 14. Multimodal tests

Verify equivalent multimodal representations do not create false conflicts where canonicalization defines them as equivalent.

At minimum cover:

- text-only content;
- text + image;
- content-part representation;
- compacted history containing a multimodal turn.

Do not unnecessarily load large binary payloads during reconciliation. Reuse existing media identifiers/registry where applicable.

---

## 15. Diagnostics

Keep the existing safe diagnostics:

    persisted_sha256
    incoming_sha256
    different_fields

Where useful, add an explicit reason/category such as:

    history_projection_accepted
    history_compaction_detected
    history_truncation_detected
    history_conflict
    history_wrong_conversation

Never log message contents, prompts, responses, tool arguments, credentials, or image payloads.

Diagnostics should make clear why a request was accepted or rejected without leaking sensitive data.

---

## 16. Update existing tests

Search the entire test suite for assumptions equivalent to:

    incoming[i] == persisted[i]

or tests that expect any representation-level change to cause a conflict.

Update them to assert logical conversation continuity instead.

Do not simply delete old tests.

---

## 17. Performance

The common path must remain efficient.

Do not implement O(n²) reconciliation.

Prefer:

- indexed message lookup;
- hashes/fingerprints;
- maps keyed by stable IDs;
- bounded anchor searches;
- database queries scoped to the relevant conversation.

Do not load arbitrarily large histories into Python for every request when a bounded database query can be used.

---

## 18. Security

The new algorithm must not introduce history injection or cross-conversation access.

Conversation/session identity remains authoritative.

Client-supplied message IDs are not authorization credentials.

Every client-provided anchor must be validated against persisted state for the active conversation.

---

## 19. Preserve previous capture hardening

Do not revert the previous assistant reasoning preservation work.

Keep:

- reasoning_content capture;
- reasoning canonicalization;
- tool-call capture;
- divergence fingerprints.

Reconciliation should operate on the canonicalized representation produced by the current capture/persistence layer.

---

## 20. Full verification

Run:

    ruff check .
    pytest -q

Then run:

- PostgreSQL integration suite;
- Docker/Compose smoke tests;
- OpenCode compatibility test, if available.

Verify existing behavior for:

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

## 21. Acceptance criteria

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
- [ ] `ruff check .` passes.
- [ ] Full test suite passes.
- [ ] PostgreSQL integration passes.
- [ ] Docker smoke tests pass.
- [ ] OpenCode compatibility scenario passes.

---

## 22. Scope boundary

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

## 23. Final report

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
