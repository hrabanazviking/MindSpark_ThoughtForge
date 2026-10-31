# TF-067 — Quant Quality Fixtures

**The plain truth first: until this slice, ThoughtForge had
quant *claims* and no quant *measurements*.** TF-066 filed
every loss claim in the tree — "Q8 near full-precision
quality", "Q5 higher fidelity", "IQ2 extreme compression" —
and every one was unmeasured catalogue copy. This slice builds
the measuring estate and publishes the first curve in the
tree that is *computed from a corpus through a scorer* rather
than copied from marketing: eight prompts, the same at every
quant level, scored by the Part VII harness shape, with loss
stated against the panel's own full-precision row. The estate
lives in `src/thoughtforge/inference/quant_quality.py` (format
`thoughtforge-quant-quality-panel/1`); TF-127's live panel
consumes its runner mechanically, and TF-068 chooses
per-profile defaults from its shape.

## The levels

Exactly the levels the shipped system spans — no more, no
fewer (the checker re-derives both surfaces live): the
hardware profiles recommend `q2_k` (phone_low, pi_zero),
`q4_k_m` (pi_5, desktop_cpu), `q8_0` (desktop_gpu), `fp16`
(server_gpu); the download catalogue declares `Q4_K_M` ×9,
`Q5_K_M` ×2, `Q8_0` ×2, `IQ2_XS` ×2. Panel order, most
compressed first: `iq2_xs, q2_k, q4_k_m, q5_k_m, q8_0, fp16`.
No bits-per-weight figure is stated for any level — TF-066
ruled the tree states none, and this slice invents none.

## The prompts and the scorer

Eight synthetic prompts (TF-017's Lantern Hollow world; two
arithmetic reasoning prompts; one numbered-list instruction;
no canon anywhere), each specified as data with *required
phrase groups* and *forbidden phrases* in TF-028's anchor
shape, plus a reference answer pinned to score a clean 1.0.

The scorer is pure and follows the rule TF-028/TF-063
established: **retention** = satisfied required groups /
required groups; a **forbidden phrase** — the answer
contradicting the fact it was asked about — **zeroes the
prompt**, because a confident wrong answer is worse than a
partial one. Loss is measured against the panel's own `fp16`
row, never against an assumed 1.0: a panel run without the
reference level raises, as does an empty prompt set (no
vacuous passes — D-BENCH-4's lesson).

## The corpus

`FIXTURE_RESPONSES` holds one recorded-style response per
(prompt, level) — 48 cells, completeness pinned. The failure
shapes are the ones quantised small models are known to take:
dropped anchors (`q5_k_m` names the red lantern but not the
high river), terse partials (`q2_k`: "She writes it in
chalk."), confident contradictions (`q2_k`'s twelve-minute
ferry; `iq2_xs`'s blue lantern, diary, dawn bell), and empty
hedges (`iq2_xs` on the crossing rules). The authorship is
deliberately *not* uniform — `q2_k` answers the fare-maths
and red-lantern prompts perfectly where `q5_k_m` fumbles one
of them — so the scorer, not the corpus design, decides the
curve, and the curve comes out with a wrinkle (below).

## The published curve (measured on the fixture corpus)

| Quant | Mean score | Loss vs fp16 | Perfect / 8 | Contradiction prompts |
|---|---|---|---|---|
| iq2_xs | 0.0000 | 1.0000 | 0 | 7 |
| q2_k | 0.4375 | 0.5625 | 2 | 2 |
| q4_k_m | 0.8125 | 0.1875 | 6 | 1 |
| q5_k_m | 0.7083 | 0.2917 | 4 | 1 |
| q8_0 | 1.0000 | 0.0000 | 8 | 0 |
| fp16 | 1.0000 | 0.0000 | 8 | 0 |

`CURVE` is filed as literal data in the module and the
checker recomputes it from the corpus on every run — edit
the corpus without re-publishing and the checker fails.

**The wrinkle, filed rather than smoothed:** the curve is
not monotone. On this eight-prompt corpus `q4_k_m` (0.8125)
out-scores `q5_k_m` (0.7083), because `q5_k_m` contradicts
the oil-maths prompt outright ("22 measures remain") while
`q4_k_m` answers it cleanly. `NON_MONOTONE_STEPS` files the
step as data and the checker recomputes it. The honest
reading for TF-068 is the curve's *shape* — catastrophic
loss at iq2_xs, steep loss at q2_k, a working band from
q4_k_m up, parity between q8_0 and fp16 on this corpus — not
a ranking of neighbouring levels, which an eight-prompt
corpus cannot support. TF-127's larger live panel owns the
ranking.

## What this does and does not prove

**Proof grade: FIXTURE.** No GGUF exists in this tree,
`llama_cpp` is not installed on this machine, and no live
model was run. The numbers above are a real measurement *of
the fixture corpus through the real scorer*: they prove the
harness measures, discriminates levels, and catches
contradictions, and they give TF-068 a curve shape to reason
from. They are **not** a measurement of any real model's
quantisation loss — TF-066's `LOSS_CLAIMS` therefore stay
`measured_in_tree == False` (pinned by this slice's tests):
the catalogue's copy remains unmeasured until TF-127 runs
this same panel against live models and TF-126's honesty
panel reports across model sizes. On the fixture corpus
alone, q8_0's "near full-precision" copy happens to hold
(parity with fp16) and the spec blueprint's "noticeable
degradation, particularly in reasoning" expectation is
borne out in shape (both reasoning prompts collapse at the
two lowest levels) — recorded as fixture observations, never
as verdicts on real models.
