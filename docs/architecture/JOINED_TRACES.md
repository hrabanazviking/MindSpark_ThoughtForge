# TF-084 — Joined Traces: Inference Traces Join Cognition Traces

*One exportable story per turn. The estate lives in
`src/thoughtforge/inference/joined_trace.py`
(`TRACE_RULES` as data, formats
`thoughtforge-inference-trace/1` for the inference
section and `thoughtforge-joined-trace/1` for the
joined document). The D-HON-1 fix lives in
`src/thoughtforge/inference/unified_backend.py`
(the default stream's `last_stream_response` stash)
and `src/thoughtforge/inference/turboquant_backend.py`
(the load-failure stream leg stashes likewise),
consumed by the core in
`src/thoughtforge/cognition/core.py`. The core
surface is `ThoughtForgeCore.last_turn_joined_trace()`,
recorded in `cognition/core_docs.py`'s
POST_FREEZE_ADDITIONS as a TF-034 MINOR addition
beside `last_turn_trace` (and filed in
`docs/architecture/CORE_API_FREEZE_V1.md`).*

## The plain truth first

Before this slice, a turn had two half-stories that
never met. TF-038's cognition trace exported the
turn's route, retrieval, scaffold, candidates,
fragments, failures, final answer and timings — but
nothing about the inference that produced the
candidates: which backend or engine served, with
what request parameters, how long each call took,
what the backend itself reported, or why a call
failed beyond the failure record's one line. The
inference side recorded nothing per turn at all: a
`GenerationResponse` / `GenerationResult` was
consumed by the core and discarded, and fallback
attempts lived only on `FallbackBackend.last_report`
until the next call overwrote them.

TF-038's own §7 named this slice as the consumer
that joins the two sides. This slice is that join:
one document per turn carrying the cognition trace
whole, the inference calls that turn made, and a
reconciliation derived from the two — recomputed by
the checker, never trusted as filed.

## The join

```json
{
  "format": "thoughtforge-joined-trace/1",
  "turn_id": "…",
  "cognition": { "…": "TF-038's trace, whole and unmodified" },
  "inference": { "format": "thoughtforge-inference-trace/1", "calls": ["…"] },
  "reconciliation": { "…": "derived from the two sections" }
}
```

Sections ride in story order (`JOINED_SECTIONS`:
cognition / inference / reconciliation), under one
`turn_id` across all three. The cognition trace is
judged by TF-038's own checker inside
`joined_trace_problems` — the join never re-judges
or rewrites it.

### The recorder

`InferenceRecorder` is an ordered, per-turn record
of inference calls. The cognition core owns one per
turn (created in `_run_turn`, continued across a
resume of the same turn) and records **every leaf
generation call at the call site**: unified-backend
generate / stream, local-engine generate / stream,
the repair pass, and a model-load failure
(`CALL_KINDS`: backend_generate / backend_stream /
engine_generate / engine_stream / engine_repair /
model_load). Each call ends in a named outcome
(`CALL_OUTCOMES`: served / error / empty /
exception / interrupted — TF-027's cut is stated,
never counted as a failure). `TracingBackend` is
the same recorder behind a `UnifiedBackend`
wrapper, for consumers that hold a backend outside
the core.

A knowledge-only turn records zero calls and its
generation source is stated as none — never dressed
as an inference turn.

### What a call records — sizes, never texts

TF-038 §7 ruled prompts out of the cognition trace;
this slice extends the ruling inference-side. A
call records the prompt's *size* (chars + a token
count with its basis named) and the response's
*size* (chars + tokens with their basis named) —
never the texts themselves, which live once, in the
cognition trace's candidates. A streamed call has
no per-call token report from its source, so its
recorded `tokens_generated` is the completed text's
count on the basis the core itself used for the
candidate (word count after stream) — the same
number the candidate carries, never a model count
dressed as one.

Latency is named per call, never conflated:
`latency_ms` is the recorder's wall around the leaf
call; `reported_latency_ms` is what the backend or
engine itself reported, filed beside it, neither
ever substituted for the other (a source that
reported nothing files None).

When the serving backend is a TF-073
`FallbackBackend`, its `last_report` attempts (each
member's outcome, in order) ride the call record as
`fallback_attempts`, captured before the next call
can overwrite the report — the chain's story is
part of the turn's story.

Recording is guard-wrapped in the core: a recorder
failure is logged at debug level and the turn
proceeds exactly as it would untraced. A trace is
a record, not a gate.

### The reconciliation is recomputed

The reconciliation block is derived from the two
sections — call counts, failed-call count, served
candidate-call tokens, repair tokens, total
inference latency — and `joined_trace_problems`
recomputes every field and names any disagreement,
including the two cross-section identities:

- failed inference calls == the cognition trace's
  `generation_failures` count (with exactly one
  stated exception: a repair call returning empty
  files no TF-031 failure — the core composes from
  it as-is — so the failed count excludes that one
  shape, stated in `_reconciliation`);
- served candidate-call tokens == the candidates'
  own `token_estimate` sum (repair tokens reconcile
  separately: a repair produces a final, not a
  candidate).

A tampered reconciliation is therefore caught by
name, not by suspicion.

## D-HON-1 — FIXED in scope

D-HON-1 (found by TF-031, assigned here) was the
streamed path's thinner truth: `UnifiedBackend.
generate_stream`'s default yields nothing at all
when its response carries an error, so a streamed
turn could only record "backend returned no text
(error or empty)" while the plain path recorded
`resp.error` exactly.

The fix keeps the contract and restores the detail.
The yield-nothing contract itself is unchanged
(TF-026/TF-065's pins stand); the default stream
now stashes the response it obtained on
`backend.last_stream_response`, `TurboQuantBackend`'s
load-failure stream leg stashes the same way, and
the core's streamed path reads the stash (or a
fallback chain's `last_report`) so the streamed
failure record and the joined trace carry the exact
error string, exactly as the plain path does. The
generic no-text detail now stands only for a
genuinely detail-less empty.

## The named gap — D-TRACE-1 (OPEN, TF-097)

One thing the join honestly does not record: the
refine passes `FragmentSalvage` makes through the
engine inside its own `forge()` are not
individually recorded here. The core hands the
engine over and cannot see the passes' per-call
sizes or latencies, and an aggregate invented from
the salvage result would be fabrication. TF-097 —
Part V's rescue-record slice, which the roadmap
already ties to TF-084 — owns recording them. Until
then a turn whose final came through refine passes
carries those passes unrecorded, stated in
`TRACE_RULES` (`salvage_refine_passes_are_a_named_gap`),
never silently.

## Proof grade

FIXTURE — real core turns on this VM
(knowledge-only, stub-backend plain + streamed, a
stated backend failure plain vs streamed carrying
the *same* exact error after the D-HON-1 fix,
stub-engine turns, a FallbackBackend chain whose
attempts ride the call record, and hand-built
records for the engine legs a no-model VM cannot
run). A live model's joined trace is **NOT
PROVEN**: no `llama_cpp` and no GGUF exist in the
tree, so the engine legs are proven against stub
engines shaped exactly as the real one's results;
TF-127/TF-175's live panels own that road, and
TF-086's truth table must carry the line
(`TRACE_RULES` `live_model_join_not_proven`).
