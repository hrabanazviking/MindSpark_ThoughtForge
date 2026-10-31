# TF-036 — Determinism Mode: Seeded Runs Reproducible

*Part II, slice TF-036. Code:
`src/thoughtforge/cognition/determinism.py` (the census, the
mode, the seeded streams, the fingerprint, the runner), the
plumbing in `src/thoughtforge/cognition/core.py`
(`_mint_id` / `_next_generation_seed`), the `seed` fields on
`GenerationParams` / `GenerationRequest`
(`inference/turboquant.py`, `inference/unified_backend.py`,
the TurboQuant backend bridge), and the refine-pass seed in
`refinement/salvage.py`. Tests:
`tests/test_tf036_determinism.py` (37 tests).*

## 1. The question

The cognition layers were deterministic by accident —
router, scaffold, prompt builder, scoring, salvage and
enforcement are pure functions with no RNG anywhere — but no
run could ever be *proven* reproducible, because two things
varied freely: every record id was minted from
`uuid.uuid4()`, and generation sampling had no seed to carry
(the params/request types had no seed field at all). Tests
and benchmarks — the consumers TF-121 and TF-135 will build
on — need runs they can diff exactly. This slice gives a
conversation a determinism mode and defines, in data, what
"reproduced" means.

## 2. The mode

A conversation selects it in its core config:
`{"determinism": {"seed": 42}}` — resolved at construction
by `resolve_determinism_mode`, exposed as
`core.determinism_mode` (a MINOR addition under TF-034's
versioning rules; the frozen v1 tables are untouched). The
seed must be a non-negative int — negative, float, string
and bool seeds raise `ValueError` at construction, never a
silent default. Seed 0 is a real seed. No seed = mode off =
the pre-TF-036 behaviour byte for byte (uuid4 ids, no seed
on any generation call; both pinned by tests).

With a seed selected, the core holds two dedicated
`random.Random` streams — never the process-global module,
whose seeding would couple every conversation and thread
(TF-035) to one stream; a test proves a seeded turn leaves
the global RNG's state untouched:

- **the id stream** (`DeterministicIds`) mints every id the
  core creates — turn, candidate, fragment, response — in
  the standing shapes (`turn_` + 8 hex, etc.). Measured this
  slice: two fresh cores seeded 42 both opened their first
  turn as `turn_3881d6d8`.
- **the generation stream** draws one seed per generation
  call, carried as `seed=` on the params/request: candidate
  generation (local engine plain + streamed, unified backend
  plain + streamed), the quality-gate repair pass, and one
  draw per compose for the salvage refine passes (pass *i*
  carries draw + *i*). The TurboQuant engine forwards a set
  seed to the llama call and omits it entirely when unset,
  so unseeded calls are unchanged; `generate_drafts` derives
  per-draft seeds (seed + *i*) so a seeded draft set stays
  both reproducible and varied. Measured: two seeded-42
  cores' backends recorded the identical seed sequence,
  first draw 1349640143.

## 3. What "reproducible" means: the fingerprint

`turn_fingerprint(turn)` is the contract: ids, scaffold
(intent / depth / candidate modes / goal / fact_block),
candidates (mode, text, token estimate), kept fragments,
and the final (response id, text, mode, token count,
composite + tier, enforcement verdict + notes, salvage
path, source ids, citations). `reproducibility_problems(a,
b)` diffs two finished turns' fingerprints and names each
differing field by path (`final.text`,
`candidates[0].candidate_id`, …) — TF-135's flake reports
inherit "what drifted", not just "something did".

Excluded, deliberately and in writing: timestamps
(`started_at` / `completed_at`) and stage timings
(`*_ms`) — wall clock and measurements, not content
(TF-033 owns timings). A test mutates all of them post-run
and the comparison still comes back clean, while a tampered
final text is caught by name.

The census (`DETERMINISM_SOURCES`) rules every surface
that could vary: the pure layers deterministic; generation
sampling and record ids seeded-in-mode; timestamps/timings,
caller-owned chat-history ids (which never reach a prompt —
the TF-024 render is role + content only), and the global
RNG excluded, each with its reason. Salvage's *internal*
fragment ids are excluded with evidence: they are minted
inside `FragmentSalvage.forge()`, whose result carries no
ids, and never reach a turn record.

## 4. The runner, the reset, and the defect they caught

`run_seeded_script(core, texts)` runs a script's turns in
order and returns finals + fingerprints. By default it
first calls `reset_deterministic_state(core)`, which
re-derives both streams from the seed — and restarts the
**will ledger**. That second half is this slice's own
discovered defect, found by its probe before banking: a
streams-only reset reproduced a two-turn script's *finals*
exactly, but the re-run's scaffolds carried the previous
run's standing goal (the planning turn had set it; TF-029's
ledger is the core's one piece of cross-turn state), so the
fingerprints differed. The reset now treats a re-run as
what it is — a new conversation on the same core — and the
regression is pinned by test. On an unseeded core both the
reset and the runner raise before running anything: a
benchmark that forgot its seed finds out loudly, not from
mysterious flakes later.

Two seeded cores interleaved turn-by-turn also each
reproduce their solo script run exactly — the streams are
per-core, so concurrency (TF-035) does not disturb
reproducibility.

## 5. Proof grade and limits

FIXTURE proof, on this VM: a seed-honouring stochastic
backend and engine (both sample among phrasings with
`random.Random(seed)`), and knowledge-only cores. What is
**NOT PROVEN** here:

- a *live model's* reproducibility under a carried seed —
  that is the model's own property; the quant and
  small-model panels (TF-067 / TF-126) re-measure it where
  a real model runs. The local engine's seed forward to
  llama.cpp is plumbed and unit-pinned at the params-merge
  level, not exercised against a loaded model in this
  slice.
- unified backends honouring the request seed — the field
  is carried and the fixture honours it; making honouring
  it a contract *duty* was TF-065's backend-contract slice
  — LANDED by TF-065 (see BACKEND_CONTRACT.md): seed is a
  named contract duty, all four shipped backends forward
  the carried seed (measured in their payloads / engine
  params), and a backend that cannot must declare
  `not_supported`. A live model's reproducibility under
  the forwarded seed remains the model's own property
  (TF-067 / TF-126, above).
- streamed *native* engine reproducibility beyond the
  fixture engine's single-delta stream (the chunking and
  event contract are TF-026's, unchanged).
