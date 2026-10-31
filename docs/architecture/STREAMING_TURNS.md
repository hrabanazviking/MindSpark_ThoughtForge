# ThoughtForge Architecture — Streaming Turns (TF-026)

Streaming turns: cognition works on streamed generation — scaffold
applied pre-stream, salvage post-stream, honestly. Built
2026-10-31 on branch `yrsa/thoughtforge-road-build`.

The event contract lives in
`src/thoughtforge/cognition/streaming.py` (`StreamEvent`,
`EVENT_KINDS`, `chunk_words`); the streamed turn is
`ThoughtForgeCore.think_stream` in `cognition/core.py`; the
sources are `TurboQuantEngine.generate_stream`
(`inference/turboquant.py`) and `UnifiedBackend.generate_stream`
(`inference/unified_backend.py`). Tests:
`tests/test_tf026_streaming.py` (18 tests). No TF-016 docket entry
owned streaming; nothing is flipped by this slice.

## 1. What was true before

- `think()` was all-or-nothing: the whole turn ran, then one
  finished `FinalResponseRecord` came back. No caller could watch
  a turn arrive, show progress, or render a candidate as it formed.
- No generation source could stream at all: Ollama and LM Studio
  were called with `"stream": False`, `TurboQuantEngine` had only
  the completed-text `generate()`, and `GenerationParams.stream`
  existed as a flag nothing honoured.
- The pipeline stage that *could* honestly stream — candidate
  generation — was welded to the stages that cannot: fragment
  scoring, salvage (which drops, reorders and regenerates text),
  enforcement (which may append a forge note) and the quality
  gate all need completed text.

## 2. The honesty contract

One sentence: **pre-stream is the whole cognition front half,
only candidate generation streams, post-stream is not streamed
at all — and simulated streaming is labelled.**

- **Pre-stream.** Routing, retrieval and scaffold building run
  before the first chunk, unchanged. The prompt a streamed
  candidate is generated from is byte-identical to the prompt
  `think()` builds for the same turn and mode (pinned by test
  against `PromptBuilder.build_candidate_prompt` under TF-023's
  contract).
- **The stream.** Per candidate, in scaffold mode order:
  `candidate_start` → `chunk`\* → `candidate_end` (full assembled
  text). A chunk's `native` flag is True only when the source
  itself yielded incrementally. When a source returns completed
  text only, the core re-chunks it for delivery with the one
  shared chunker (`chunk_words`, 8 words, deterministic,
  word-preserving) and marks every chunk `native=False`. The
  candidate record always keeps the source's original text, never
  a reconstruction from chunks.
- **Post-stream.** Salvage, enforcement and the quality gate run
  on completed candidate text, exactly as in `think()`. Their
  answer arrives as ONE terminal `final` event carrying the
  finished `FinalResponseRecord` — token-streaming the final
  would be a lie, since no token of it is final until salvage
  and enforcement have run. (Salvage's internal refine passes
  likewise use the engine's completed-text `generate()`; they
  are post-stream machinery, not part of the stream.)
- **Failure.** A candidate whose stream raises — even mid-way —
  is discarded whole (a candidate is its completed text or it is
  not a candidate), a `candidate_error` event carries the
  failure, and the turn continues with the remaining modes:
  `think()`'s per-mode failure handling, made visible. If every
  candidate fails, the turn degrades to the same knowledge-only
  final `think()` would have produced.
- **One pipeline, two views.** `think()` and `think_stream()`
  are both drivers of the single `_run_turn` generator; a
  streamed turn builds the identical `RuntimeTurnState` (TF-025),
  retained on `last_turn_state`, and the `final` event's record
  IS `turn.final_response`.

Event order is fixed: `turn_started`, then per-candidate groups,
then exactly one `final`. A knowledge-only turn (no model)
streams nothing between: `turn_started`, `final` — there is no
generation to pretend to stream.

## 3. The sources

| Source | Streaming | Native? |
|---|---|---|
| `TurboQuantEngine.generate_stream` | llama call with `stream=True`; yields each choice-delta's text in order, unstripped (`"".join(...)` is the raw completion; the core strips the assembled candidate, matching `generate()`). Params merged and token budget enforced exactly as `generate()`; a mid-stream failure raises `RuntimeError`. | yes |
| Local engine without `generate_stream` (any object with only `generate`) | core falls back to `generate()` + `chunk_words` | no — labelled |
| `UnifiedBackend.generate_stream` (default) | yields the completed `generate()` response as one chunk; yields nothing on an error response (the core then emits `candidate_error` and no candidate — the same outcome as `_generate_via_unified_backend` returning `[]`) | no — labelled |
| `UnifiedBackend` subclass overriding `generate_stream` | core detects the override (`type(backend).generate_stream is not UnifiedBackend.generate_stream`) and passes its chunks through untouched | yes |

The streamed unified candidate's `token_estimate` is the tree's
word count of its text (a streamed response carries no reliable
token total); the non-streamed unified path keeps the backend's
reported `tokens_generated`.

## 4. Measured (synthetic stubs, this slice's tests)

- Native stub (17-char deltas over a 139-char text): 10 chunks
  per candidate, per-candidate concatenation == candidate text
  == the stub's text, 2 candidates (the scaffold's two modes).
- Simulated path: candidate text and `token_estimate` identical
  to the completed-text path's; chunks ≤ 8 words each.
- Mid-stream failure on mode 1: 1 `candidate_error`, 1 surviving
  candidate with the full text, turn completes with a `final`.
- Knowledge-only: exactly `[turn_started, final]`.

## 5. Deliberately not here

- **Interruption & resume of a stream** — TF-027's slice; the
  event contract here is what it will cut into.
- **Streaming across the Part IX bindings** — TF-164 composes
  TF-026 with TF-072 there.
- **True token streaming inside the HTTP backends** (Ollama /
  LM Studio called with `"stream": False` today): a backend
  gains native streaming by overriding `generate_stream`; doing
  so for the shipped backends belongs with their Part V/IX work,
  not with the cognition contract this slice pins.
