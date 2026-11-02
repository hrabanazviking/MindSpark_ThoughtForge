# ThoughtForge Architecture — Small-Model Honesty Panel (TF-126)

Small-model honesty panel: the same suites across
model sizes — where the forge helps most, published
from results. Part VII's sixth slice, built on
TF-121's harness and TF-122 / TF-123 / TF-124 /
TF-125's four suites. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`benchmarks/honesty_panel.py` (`MODEL_TIERS`,
`TRANSFORMS`, `PANEL_ROWS`, `TIER_MEANS`,
`FINDING`, `PANEL_RULES`, checked by
`honesty_panel_problems()`), registered in
`benchmarks/harness.py` with a pinned baseline.
The tests are `tests/test_tf126_honesty_panel.py`.
This slice **adds and fixes D-BENCH-12** in the
measuring sense (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **Four suites, four pinned baselines, and
  nothing across sizes.** The estate stood at 13
  benchmarks / 6 fixtures / 6 baselines / 41
  metric definitions, all checkers clean. Each
  suite's pinned baseline re-measured exactly:
  persona overall 0.8453703703703704, memory
  overall 0.8055555555555557, depth OFF
  0.47222222222222227 → ON 0.8055555555555557,
  presence overall 0.7777777777777778. No module,
  panel or row anywhere had ever run any suite at
  more than one model size, and each of
  TF-122..TF-125's filings had assigned that run
  to TF-126 without the gap ever being numbered
  in the TF-016 docket (see §5).
- **No live model can run on this VM.** No GGUF
  in the tree, no `llama_cpp` (TF-078's filing,
  re-probed). So the panel follows TF-067's
  fixture-panel precedent — a published curve
  filed as literal data, recomputed live by its
  checker, over scripted fixtures through the
  real scorers: it measures **filed size-tier
  degradations** of the suites' own scripted
  corpora through the suites' **real runners**.
  A tier is a degradation profile anchored to a
  real catalogue model, never a claim that the
  model produced these answers. Live per-size
  behaviour stays NOT PROVEN (§6).
- **Every suite runner accepts an explicit
  corpus** — `run_persona_suite(identity, voice,
  pressure)`, `run_memory_suite(present, true,
  explain)`, `run_depth_suite(chain, plan,
  check)`, `run_presence_suite(coherence,
  steering, closure)` — probed live. The panel
  reuses those runners and the suites' corpora
  exactly as filed; it copies no corpus and no
  scorer.
- **The tier anchors are real.** All four anchor
  filenames exist in
  `ModelBrowser.CURATED_MODELS` (probed live),
  and `params_from_name` returns 14.0 / 8.0 / 3.0
  for the first three and **None** for the
  TinyLlama filename — its `1.1b` token is
  lower-case, which that reader does not take.
  The tiny tier's params are filed from the
  filename's own `1.1b` token, regex-verified by
  the checker, never invented.

## 2. The panel

`benchmarks/honesty_panel.py` files four size
tiers. Each tier degrades the four suites' filed
corpora by its filed transforms, applied in
order to the filed answer fields only (persona:
identity/pressure `text` + voice strings; memory:
cases' `text`; depth: `off_answer`, `working`,
`on_answer`; presence: steering/closure turns'
`assistant`).

| Tier | Anchor (live catalogue) | Params | Transforms | Drift shift |
|---|---|---|---|---|
| reference_14b | Qwen2.5-14B-Instruct-Q4_K_M.gguf | 14.0 (params_from_name) | — | 0 |
| mid_8b | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | 8.0 (params_from_name) | citation_drop | 0 |
| small_3b | Llama-3.2-3B-Instruct-Q4_K_M.gguf | 3.0 (params_from_name) | citation_drop, terse_first_sentence | 1 |
| tiny_1_1b | tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | 1.1 (filename token) | citation_drop, terse_first_sentence, generic_closer, fact_swap | 2 |

The transforms, filed as data:

| Transform | What it does |
|---|---|
| citation_drop | strips bracketed citation markers (`\[[^\]]{1,48}\]`) and standalone QID tokens (`\bQ\d{4,}\b`) |
| terse_first_sentence | keeps only up to and including the first sentence-ending `.` `!` or `?` (whole text if none) |
| generic_closer | appends ` As an AI, I cannot provide more detail.` — carrying the `GENERIC_PHRASES` member "as an ai" verbatim (checker-verified against `benchmarks.persona_consistency.GENERIC_PHRASES`) |
| fact_swap | swaps dawn→midnight, eleven→twelve, twice→once (case-insensitive, word-boundary, case preserved) — TF-067's confident-contradiction shape |

Coherence sagas are shapes, not text: their tier
degradation is the filed **drift shift** — a
tier's `drift_shift` moves a saga's `drift_onset`
earlier by that count (clamped at ≥ 2; onset None
stays None; `drift_turns` and `upto` unchanged).

**The forge repairs (the ON road), one per
suite, all from real machinery:**

| Suite | Repair |
|---|---|
| memory | citation repair — for each citation-axis case (present / true) whose degraded text carries no citation marker, re-attach the marker of its first filed record (`citation_for_record(RECORDS[record]).marker`, the real TF-048 payload), in the suite's bracketed format, appended into the final sentence before its terminal period (TF-049's in-claim discipline). The explain axis runs degraded-only on both roads: its holding rule is carrying NO citation (TF-123's cite-or-explain), so appending one would break the suite's own discipline — filed as a rule |
| persona | generic strip — remove any sentence containing a `GENERIC_PHRASES` member, over identity/pressure texts and voice strings |
| presence | will repair — replay each will saga through a fresh real `GoalLedger` exactly as `judge_will_saga` does; on any turn where the ledger's `will_line` stands and the degraded answer lacks the standing statement, prepend `Standing goal: <statement>. ` (TF-029's inject_will shape). **Coherence: NO repair exists — ON == degraded there, filed as a rule** |
| depth | ON is the suite's native scaffold road (its `on_answer`/`working`), degraded by the same tier transforms; OFF is its `off_answer` degraded |

**The measured panel** (OFF → ON, delta; exact
values are the literal `PANEL_ROWS` in the
module, recomputed live by the checker):

| Tier | Persona | Memory | Depth | Presence | Tier mean |
|---|---|---|---|---|---|
| reference_14b | 0.8454 → 0.8889 (+0.0435) | 0.8056 → 0.8333 (+0.0278) | 0.4722 → 0.8056 (+0.3333) | 0.7778 → 0.8333 (+0.0556) | 0.7252 → 0.8403 (**+0.1150**) |
| mid_8b | 0.8454 → 0.8889 (+0.0435) | 0.2778 → 0.8056 (+0.5278) | 0.4722 → 0.8056 (+0.3333) | 0.7778 → 0.8333 (+0.0556) | 0.5933 → 0.8333 (**+0.2400**) |
| small_3b | 0.8361 → 0.8796 (+0.0435) | 0.2778 → 0.8889 (+0.6111) | 0.4722 → 0.2778 (**−0.1944**) | 0.7500 → 0.8056 (+0.0556) | 0.5840 → 0.7130 (**+0.1289**) |
| tiny_1_1b | 0.6339 → 0.7870 (+0.1531) | 0.2778 → 0.3056 (+0.0278) | 0.0000 → 0.0000 (0.0000) | 0.7222 → 0.7778 (+0.0556) | 0.4085 → 0.4676 (**+0.0591**) |

**Verdict PASS** — on the filed rule only: the
reference tier's OFF scores reproduce the four
pinned suite baselines **exactly** (persona
0.8453703703703704, memory 0.8055555555555557,
depth OFF 0.47222222222222227 and depth reference
ON 0.8055555555555557, presence
0.7777777777777778) and all 16 cells are
measured. An empty panel scores 0.0 / REVIEW —
no vacuous pass.

## 3. The finding, and the honest shapes

**Where the forge helps most: the mid tier** —
mean delta +0.24004629629629637, ahead of small
(+0.1289), reference (+0.1150) and tiny
(+0.0591). That is the measured answer, and it
is not the intuitive one: at the smallest size
the filed degradation destroys content the
repairs cannot restore (below), so the forge's
recoverable margin peaks in the middle. Ties are
broken toward the smaller tier by the filed rule
(depth's reference/mid tie at +0.3333 resolves
to mid; presence's reference/mid tie at +0.0556
likewise). Per suite:

- **Persona — helped most at tiny (+0.1531).**
  The generic closer degrades every tiny answer;
  the generic strip removes exactly that sentence
  (and the corpus's own generic sentences), so
  the voice axis recovers from 0.5406 to 1.0000.
- **Memory — helped most at small (+0.6111).**
  Citation drop collapses all three tiers below
  reference to 0.2778 OFF; repair re-attaches the
  real markers. At small the ON score (0.8889)
  even exceeds the reference OFF — honestly,
  because terseness first truncates the harder
  second claims and wrong-sentence shapes, and
  repair then re-cites the surviving claim; the
  interaction is filed here, not hidden. At tiny
  the gain collapses (+0.0278): fact_swap turns
  dawn into midnight, and a re-attached citation
  cannot make a swapped fact true.
- **Depth — helped most at mid (+0.3333, tied
  with reference), and NEGATIVE at small
  (−0.1944).** The terse transform truncates each
  case's numbered `working` at its first period —
  the scaffold's own step list is destroyed, so
  the ON road falls below the (also degraded)
  OFF road. At tiny, depth scores 0.0000 on both
  roads: nothing survives to repair.
- **Presence — a flat +0.0556 at every tier.**
  The will repair restores the two dropped /
  contradicted steering sagas at every tier
  (steering 0.8333 → 1.0000); coherence has no
  repair, so its tier losses (0.75 → 0.6667 at
  small, → 0.5833 at tiny, from the drift shift)
  stand unrecovered on the ON road too, filed
  plainly.
- **Reference ON > OFF, honestly stated:** even
  undegraded, persona ON (0.8889) beats OFF
  because the generic strip also removes the
  voice corpus's six deliberately generic turns'
  generic sentences, and memory ON (0.8333) beats
  OFF because repair fixes the corpus's filed
  missing-citation case. The repairs act on the
  filed corpora exactly as filed; nothing was
  tuned to make a tier story come out.

## 4. Harness registration

In `benchmarks/harness.py`, in the open:

- a new `honesty_panel` benchmark entry (kind
  `panel`, claims axes depth + presence + will)
  bound live to
  `benchmarks.honesty_panel.run_honesty_panel`,
  with a new `honesty_panel_corpus` fixture — the
  estate now files **14 benchmarks / 7 fixtures /
  7 baselines / 54 metric definitions**;
- `METRIC_DEFINITIONS` extended with the panel's
  thirteen metrics (per tier `<tier>_off_score` /
  `<tier>_on_score` / `<tier>_delta` for the
  reference / mid / small / tiny prefixes, plus
  `total_cells`), each stating what it actually
  computes;
- the four suites' baseline notes updated in the
  open: where they said a live model's behaviour
  is NOT PROVEN (TF-126), they now file TF-126
  DONE in the measuring sense, with live
  behaviour still NOT PROVEN (TF-173);
- TF-121 + TF-123 + TF-124 + TF-125's test pins
  updated in the open for the new counts;
- a new pinned baseline, measured live on this
  VM twice, identical, latency pinned nowhere:

**`honesty_panel_synthetic_v1`** — reference OFF
0.7252 → ON 0.8403, mid OFF 0.5933 → ON 0.8333,
small OFF 0.5840 → ON 0.7130, tiny OFF 0.4085 →
ON 0.4676, 16 cells; its filed target is the
reference-reproduction truth (the reference
tier's mean OFF is pinned at the value the four
suites' baselines imply, met). (**Every earlier
pinned baseline is preserved exactly** — persona
0.7166666666666667, persona suite
0.8453703703703704, memory suite
0.8055555555555557, depth suite OFF
0.47222222222222227 → ON 0.8055555555555557,
presence suite 0.7777777777777778, the
empty-store profile baseline — never re-pinned;
`baseline_problems()` is clean over all seven.)

## 5. Docket dispositions (TF-016)

- **D-BENCH-12 — added and FIXED by this slice,
  in the measuring sense.** The defect, as probed
  in §1: the four suites had never been run
  across model sizes — each suite's pinned
  baseline measured its filed corpus once, at no
  size, and no per-size row existed, so where the
  forge helps most was measured by nothing. The
  gap was assigned to TF-126 by TF-122..TF-125's
  filed limits but was never numbered in this
  docket — the D-BENCH-9 / D-BENCH-10 /
  D-BENCH-11 pattern. This slice numbers it and
  fixes the measuring share the roadmap assigns
  to TF-126: the suites now run across four filed
  size tiers with a published panel
  (`benchmarks/honesty_panel.py`) and a pinned
  harness baseline.
- No other entry moves. TF-016 totals are
  therefore 127 entries — **35 OPEN, 1
  OPEN-ENVIRONMENT, 88 FIXED, 3
  CLOSED-BY-VERDICT** (D-BENCH-12 added + FIXED in
  the measuring sense). Counting note, in
  TF-125's manner: the docket's header-vs-row
  counting mismatches are pre-existing, are
  preserved unchanged in shape (parts now sum
  125 against 127, D-BENCH-12's owner part VII
  counted), and are noted here rather than
  silently "repaired".

## 6. Filed limits

- The tiers are **filed degradation profiles
  over scripted synthetic fixtures** (TF-017),
  anchored to catalogue models — never live runs
  of those models; `data/` was never opened.
  **Live per-size behaviour of real models is
  NOT PROVEN** — no GGUF in the tree, no
  `llama_cpp` on this VM (TF-078); TF-173 owns
  the real-hardware road.
- The transforms are the filed degradation
  shapes only: citation loss, terseness, a
  generic closer, confident fact swaps, earlier
  coherence drift. Real small models degrade in
  more ways; those are not filed and not
  claimed.
- The repairs are the filed roads only: memory's
  repair re-attaches the *first* filed record's
  marker (a case whose claim belonged to a later
  record stays wrong-record, as TF-123's t12
  shape shows on the ON road too); persona's
  strip is sentence-level over the scorer's
  phrase vocabulary; presence's repair restores
  dropped steering lines and cannot restore
  coherence; depth's scaffold cannot survive its
  working being truncated — each limit is
  visible in the measured rows of §2 / §3.
- The panel baseline pins the filed panel only.

Proof grade: FIXTURE — synthetic scripted corpora
degraded by filed transforms and scored through
the real TF-122..TF-125 suite machinery on this
VM. No live model was run; that is NOT PROVEN,
never claimed.
