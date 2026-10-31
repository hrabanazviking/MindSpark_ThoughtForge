# TF-031 — Error Honesty: a Failed Turn States Its Failure

*Part II, slice TF-031. Code: `src/thoughtforge/cognition/error_honesty.py`
(`GenerationFailure`, `failure_text`, `degraded_knowledge_text`,
`partial_disclosure`, `apply_partial_disclosure`), wired into
`ThoughtForgeCore` (recording at every generation swallow point;
application in `_finish_turn`), with the failure list carried on
`RuntimeTurnState.generation_failures`. Tests:
`tests/test_tf031_error_honesty.py` (28 tests). Two TF-026 pins
were superseded by this slice, with the reason recorded at each
pin.*

## 1. The gap, as found

Before TF-031 the core degraded *silently*. Every generation
failure point logged a warning and swallowed the failure:

- **Model load, candidate, and backend failures all became the
  knowledge-only answer.** `_generate_candidates` returned the
  (empty) candidate list on a load failure; a per-mode exception
  was caught and logged; `_generate_via_unified_backend` returned
  `[]` on the backend's error channel and on exceptions. An empty
  candidate list then produced `_knowledge_only_response` —
  presented exactly like a deliberate no-model answer. When the
  store was also empty, the user was told *"The forge has no
  knowledge relevant to this query"* — a sentence that blames the
  **knowledge** for what the **model** did.
- **Partial failure was invisible.** When one of the two candidate
  modes failed and the other survived, salvage composed a
  normal-looking answer from the survivor alone. Nothing on the
  answer or the record said half the generation never happened.
- **Empty text was a candidate.** An engine (or backend) returning
  empty/whitespace text had that nothing appended as a candidate
  and scored — a completion containing no answer at all.
- **Streamed turns were half-honest.** TF-026 already emitted a
  `candidate_error` event per failed mode, but the terminal
  `final` event still carried the silent knowledge-only record —
  a consumer reading only the final answer received the
  fabrication.

One behaviour was already honest and is preserved untouched:
a **deliberate no-model** core's knowledge-only answer (TF-005
measured its empty-store sentence "honest in fact"). No-model is
a mode, not a failure.

## 2. The contract

| Piece | Contract |
|---|---|
| `GenerationFailure` | One failure as data: `stage` (one of `FAILURE_STAGES` = `model_load` / `candidate` / `backend` / `repair`; anything else is a `ValueError`), `mode` ("" when not per-mode), `detail` (`"ClassName: message"` via `failure_detail`, capped at `MAX_DETAIL_CHARS = 160`). Dict round-trip exact. |
| Recording | The core appends one failure to `turn.generation_failures` at **every** swallow point: model load (plain + streamed), per-mode candidate exceptions (plain + streamed), empty completed text (plain + streamed + backend), backend error channel / exceptions / empty text (plain + streamed), and the repair pass. A failure is never only a log line again. |
| `failure_text(failures)` | The stated failure for a turn with nothing honest to show: opens with the disclosure marker, names stage/mode/detail, and states *"No model answer was generated…"*. Empty or repair-only lists raise `ValueError` — a stated failure with no failure behind it would itself be a fabrication. |
| `degraded_knowledge_text(failures, body)` | Total failure **with** retrieved knowledge: the stated failure heads the knowledge body, which is labelled *"retrieved knowledge only… not a model-written answer"*. The record keeps `mode = "knowledge_only"` — the body *is* knowledge; the heading says why it is there alone. |
| `partial_disclosure(failures)` + `apply_partial_disclosure(final, failures)` | Partial failure: appended **once** (marker-guarded) to whichever final text stands after the repair decision; names the failed mode(s) and that the answer was composed from the surviving candidate(s) only; the record's token count is recomputed and the failure is named in its enforcement notes. |
| Total failure, empty store | The final record is `mode = "failed"`, `salvage_path = "model_failed"`, `enforcement_passed = False`, notes `"model failure: …"`, text = `failure_text(...)`. The empty-store knowledge sentence never appears on a model-failed turn. |

**Deliberate rulings, stated plainly:**

- **A model never configured is not a failure.** Failures exist
  only where generation was *attempted* (an engine configured or
  loaded, or a backend present) and did not produce. A no-model
  core records nothing and its knowledge-only answers are
  byte-identical to before TF-031 (pinned by test).
- **Repair failure is recorded, not announced.** Repair is an
  optional improvement attempt over an already-composed, real
  answer; when it fails the user holds exactly that answer —
  nothing fabricated, nothing lost. Its failure rides the turn
  record (and TF-038's trace) but is excluded from every
  user-visible text (`generation_failures()` is that filter).
  Generation failures are always announced, because they change
  what the answer *is*.
- **Interruption is not failure.** A cut turn (TF-027) records no
  failure: the cut is the caller's act, already stated by the
  `interrupted` event / `TurnInterrupted`. Resume generates only
  the missing modes; their failures, if any, record then.
- **Retrieval failure stays out of this slice.** `_retrieve`'s
  empty-bundle fallback is the knowledge side's existing
  behaviour (TF-041 owns the store map); where it compounds a
  model failure, the stated failure's "no retrieved knowledge is
  available" remains true.

## 3. Measured (through real turns, stub engines/backends)

- All modes raising + empty store → `mode "failed"`, text opens
  with the marker, names `RuntimeError: engine exploded` for both
  scaffold modes, 2 failures on the turn record, token count
  recomputed over the stated text.
- Model path pointing at a missing file → exactly one
  `model_load` failure, stated.
- Engine returning whitespace → 0 candidates, 2 "empty generation"
  failures, stated failure.
- Backend error channel (`GenerationResponse.error`) → one
  `backend` failure whose detail is the error string exactly;
  backend exception → `ConnectionError: …`; backend empty text →
  stated.
- One mode dying, one surviving → the real composed answer stands
  ("eleven minutes" fixture text), disclosure appended exactly
  once, exactly one visible failure recorded.
- Streamed: a stream dying mid-way in both modes → 2
  `candidate_error` events, no `candidate_end`, terminal `final`
  is the stated failure; a half-dying stream → 1 error + 1 end,
  final carries the disclosure.
- Total failure with a retrieved record (bundle injected at the
  retrieval seam — the existence gate checks the *default* DB
  path, CORE_LOOP F2, TF-041's) → knowledge shown under the
  stated failure, labelled retrieved, `mode` stays
  `"knowledge_only"`, notes carry `model failure: …`.
- Repair raising under a low-quality composed answer →
  `repair_attempted` true, exactly one `repair` failure on the
  record, **no** marker on the answer text.
- Pre-fired interrupt token + failing engine → `TurnInterrupted`,
  zero failures recorded.

## 4. Superseded pins and the discovery

- TF-026's two all-fail pins expected the final record's mode to
  be `"knowledge_only"` — precisely the silent degradation this
  slice exists to remove. Both pins now assert `mode == "failed"`
  with the reason recorded at the pin (TF-027 set the precedent
  for updating TF-026 pins this way). TF-026's streaming docs'
  "degrades to the same final `think()` would have produced"
  remains true in substance: `think()` now produces the stated
  failure too.
- **Discovery (docket D-HON-1, owned by TF-084):** the streamed
  backend path cannot name the backend's error. The default
  `UnifiedBackend.generate_stream` yields *nothing* on an error
  response, so the error string never crosses the stream
  contract; the streamed record's detail is honestly "backend
  returned no text (error or empty)" where the plain path records
  `resp.error` exactly. The failure is stated either way — only
  the detail is thinner on the stream. Giving the stream contract
  an error channel is inference-side work, assigned in TF-016.

## 5. What this slice does not do

- No retries, no fallback models, no circuit breaking — honesty
  about failure, not recovery from it (the utils retry/circuit
  estate is TF-011's D-ORPHAN-2 verdict's to place).
- Salvage's internal refine-pass failures remain salvage's own
  (it composes from what it has by design); the core records the
  failures at *its* swallow points only.
- Trace *export* of `generation_failures` is TF-038's, like every
  other turn-state field's.
