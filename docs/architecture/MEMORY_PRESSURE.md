# TF-080 — Memory-Pressure Behaviour

*At the profile's ceiling the engine sheds load by
rule (context first), never crashes mute. The estate
lives in `src/thoughtforge/inference/memory_pressure.py`
(`PRESSURE_RULES` / `SHED_STEPS` as data,
`apply_pressure()` / `profile_turn_load()` /
`assess_pressure()` / `model_pressure()` /
`store_pressure()`, format
`thoughtforge-memory-pressure/1`).*

## The plain truth first

Nothing in inference or cognition responded to
memory pressure at all. The engine's one cap was a
static clip, identical at 1% memory and at 99%; no
context ask was ever checked against the profile's
context ceiling before a generation ran; and when a
backend did fail on memory, the whole turn failed
with it — announced by TF-031, but total. There was
no graduated middle: no rule tried the turn smaller
first. And no real model exists on this VM — no GGUF
in-tree, `llama_cpp` absent — so a loaded model's
real footprint under genuine pressure is
**NOT PROVEN**, stated here first rather than last.
What this slice proves is the *governor*: the bands,
the shed order, the protected pieces, and the named
refusal, judged against the live profiles and a real
no-model engine.

## Measured start

Probed live this tick:

- **Grep for pressure / shed / OOM over src finds
  exactly one pressure response in the tree**, and
  it is not the engine's: the Wikidata ETL commits
  early when psutil reports host memory over 85%
  (`etl/wikidata.py._check_memory`). That is the
  tree's sole precedent threshold, and it protects a
  database import, not a turn.
- **The engine's cap is static and pressure-blind.**
  `TurboQuantEngine._enforce_token_budget` clips
  max_tokens to `min(requested, token_budget_per_turn,
  max_response_tokens)` — the same clip whatever the
  memory state. `max_context_tokens` is used exactly
  once, as the `n_ctx` passed to llama at load; no
  prompt or context ask is checked against it.
- **A memory failure arrives whole.** Probed live: a
  backend raising `llama_decode: out of memory`
  through a real core turn becomes TF-031's degraded
  turn ("the model failed during this turn"). The
  announcement exists; the graduated shedding before
  it did not. D-PRESS-1 (TF-016).
- **The sheddable pieces existed, ungoverned.**
  TF-024 windows history under flat, profile-blind
  budgets (20 turns / 1000 word-tokens on pi_zero's
  256-token context and the server's 32768 alike);
  TF-056 caps retrieval per profile but nothing ever
  lowers those caps under pressure; the profiles'
  draft_count and max_refinement_passes are never
  reduced by anything.

## Pressure: a fraction, in bands

Pressure is a fraction of a named ceiling, judged in
four bands — **normal** below 0.75, **elevated**
[0.75, 0.90), **high** [0.90, 1.00), **at_ceiling**
at or above 1.00. The thresholds are campaign-set,
stated as such; they bracket the tree's one
precedent (the ETL's 85%) rather than copying it,
because a turn can shed in stages where an import
can only commit.

Four gauges supply fractions, each honest about what
it measures:

| gauge | numerator | ceiling | notes |
|---|---|---|---|
| context | a turn's total context ask (protected prompt + sheddable context) | the profile's `max_context_tokens` | the governor's own gauge — recomputed after shedding; the only one refusal is pronounced on |
| explicit GB | a caller-measured `used_gb` (`assess_pressure`) | the profile's declared ram+vram | the measurement is the caller's, never invented here |
| model | TF-069's accounted model size (`model_pressure`) | the same declared ceiling | **unprovable for q2_k**, exactly as TF-069 files it — fraction None, never borrowed |
| store | TF-056's per-kind live/hard ratios, max (`store_pressure`) | each kind's hard cap | an input signal only; eviction stays TF-056's lifecycle work |

## The shed ladder — context first

One ordered ladder (`SHED_STEPS`); bands take
cumulative prefixes — elevated takes step 1, high
takes steps 1–4, at_ceiling takes all six:

1. **context_halved** — the sheddable context halves.
2. **retrieval_halved** — bundle / sql / vector caps
   halve (floor 1 each).
3. **drafts_to_one** — the draft count cuts to one.
4. **refinement_off** — refinement passes cut to zero.
5. **response_halved** — the response-token cap
   halves (floor 32).
6. **context_shed_entire** — the remaining sheddable
   context is shed entirely.

A step that would change nothing on a load is not
reported as applied: on pi_zero (already one draft,
zero refinement) the at-ceiling report names exactly
`context_halved, retrieval_halved, response_halved,
context_shed_entire` — the steps that actually shed
something, in ladder order, context first and last.

**Never shed:** the current user prompt (carried on
the load as `prompt_tokens` so the gauge counts it;
no step touches it), the resident model (its fit is
TF-069 admission, not runtime), and protected store
records (TF-056's protection stands — breadth is
shed, records are not evicted by this governor).

## Never crashes mute

The governor never raises on pressure. Its outcomes
are data (`PressureReport`):

- **no_pressure** — the load is returned untouched.
- **shed** — the steps are named in the report and,
  on the context gauge, the after-fraction and
  after-band are recomputed and carried.
- **refused_named** — when even the fully shed load
  cannot fit, which on the context gauge means the
  protected prompt alone exceeds the profile's
  context ceiling, the report refuses BY NAME: the
  reason carries the profile, the prompt size, the
  ceiling, and the steps already shed. A caller
  pronounces that refusal to the user. Nothing is
  silently truncated; nothing dies with a bare
  exception. (Unknown profiles and mismatched loads
  still raise ValueError, as everywhere in this
  tree — a programming error is not pressure.)

A caller-supplied external fraction (from the GB or
model gauge) sheds by band the same way, with the
after-gauge honestly reported as None: the governor
does not recompute a measurement it did not make,
and it pronounces no refusal on one — the caller
re-measures and calls again.

## Worked rows (measured by the tests)

- desktop_cpu, prompt 100 + context 3172 (gauge
  0.799, elevated): only `context_halved` applies;
  context 3172 → 1586; status shed, after-gauge
  0.412.
- desktop_cpu, prompt 100 + context 3800 (gauge
  0.952, high): the first four steps apply; context
  → 1900, retrieval 15/15/8 → 7/7/4, drafts → 1,
  refinement → 0, response cap untouched at 400.
- pi_zero at the ceiling (prompt 20 + context 492,
  gauge 2.0): context → 0 across its two stages,
  retrieval 3/3/2 → 1/1/1, response 120 → 60, the
  prompt untouched at 20; status shed, after-gauge
  0.078 — the turn survives, smaller.
- desktop_cpu, prompt 20,000 alone: fully shed and
  still over — `refused_named`, the reason naming
  desktop_cpu, 20000, and 4096.

The maximum turn load per profile (`profile_turn_load`
— context allocation at `max_context_tokens`, which
is exactly what the engine allocates as `n_ctx` at
load; retrieval at TF-056's caps; drafts / refinement
/ response at the resolved params) sits at gauge
exactly 1.0 on every profile by construction: the
ceiling reference point the ladder is filed against,
recomputed live by the checker for all six profiles.
A real no-model `TurboQuantEngine` load's resolved
params match those fields on pi_zero and desktop_cpu.

## Docket

**D-PRESS-1 added + FIXED at fixture grade**: no
memory-pressure behaviour existed in inference or
cognition (the tree's only pressure response was the
Wikidata ETL's commit-early); the engine's token cap
was a static, pressure-blind clip; no context ask
was ever checked against the profile's context
ceiling; and a backend memory failure became a
whole-turn TF-031 degradation with no graduated
shedding attempted. This slice is the fix — the
governor estate above. Composing it into
`TurboQuantEngine.generate` / the core's turn path
is TF-153's service-shape work, in the standing
pattern of TF-073/TF-074's unconsumed estates; the
live-pressure share stands NOT PROVEN under TF-082 /
TF-175, exactly as the entry states.

## Proof grade

FIXTURE — the governor judged against the live
profile JSONs, TF-056's budgets and TF-069's
accounting, real no-model engine loads, and a
fixture store, on this VM. A loaded model's real
memory footprint and its behaviour under genuine
host memory pressure are **NOT PROVEN** (no GGUF
in-tree, `llama_cpp` absent, no resident model's RSS
ever measured here), never claimed. TF-086's truth
table must carry that line.
