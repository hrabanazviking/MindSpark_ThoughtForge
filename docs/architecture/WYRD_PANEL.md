# The WYRD-Shaped Panel — TF-132

The proof lives as verifiable data in
`benchmarks/wyrd_panel.py` (`WORLD_ENTITIES`,
`WORLD_EVENTS`, `PANEL_CASES`, `PANEL_PROBES`,
`WYRD_PANEL_RULES`, checked by
`wyrd_panel_problems()`), registered in
`benchmarks/harness.py` with the pinned
baseline `wyrd_panel_synthetic_v1`. The tests
are `tests/test_tf132_wyrd_panel.py`. This
slice **adds and fixes D-BENCH-17** in the
measuring sense (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **The estate stood clean, and no world-state
  panel existed.** `harness_problems()`
  returned [] and `baseline_problems()` returned
  [] over the twelve pinned baselines;
  `regression_gate_problems()` returned [] with
  the live gate PASS (9 baselines / 83 metrics /
  0 failures); `data_hygiene_problems()` returned
  [] (20 inputs, all clean) and
  `spot_checks_problems()` returned []. And a
  grep for any world-state / consistency panel
  across benchmarks / docs / tests / src found
  nothing: no filed world, no timeline, no
  scorer judged an answer against a stated
  state of the world.
- **D-BENCH-17 was verified unused** anywhere in
  benchmarks / docs / tests / src before filing
  (the D-BENCH-9..16 pattern: a gap assigned by
  the roadmap without ever being numbered).

## 2. What the panel measures

Part VII's suites score persona, memory, depth
and presence. This panel asks the plainer
question underneath all of them: given a stated
world — entities, attributes, relations, and a
timeline of changes — do answers stay true to
that world? The scorer is this slice's own,
filed in `benchmarks/wyrd_panel.py` plainly:
no neighbour scorer judges answers against a
replayed world timeline, so none is reused and
none is copied. `world_state_at()` replays the
timeline purely; `score_case()` computes the
world's value for a case's subject + predicate
at the case's point (counts are computed from
the replayed state) and classifies the case's
structured claim — bound to its scripted
answer by a claim phrase the checker requires
in the answer text:

- claim equals the world value —
  **consistent**, score 1.0;
- claim differs — **contradicted**, score 0.0;
- the world files no such fact and the answer
  says so honestly — **honest_unanswerable**,
  score 1.0;
- the world files no such fact and the answer
  invents one — **contradicted**, score 0.0
  (a fabrication).

The filed world is Emberwick, a fen village
invented for this panel in the style of
TF-043's Lantern Hollow (TF-017: synthetic
only — never Volmarr's canon, never the
contents of `data/`, never a real persona).
The eighteen cases span six axes: state at a
point, state after a change, relations, counts,
contradiction probes, and the unanswerable.

**The measured filing: 18 cases — 9 consistent,
8 contradicted, 1 honest-unanswerable; overall
10/18 = 0.5555555555555556; verdict PASS** (the
full set reproduces every filed expectation).
The failing shapes are filed by name — a stale
pre-change wheel and bell, a ferry in two
places at once, a wheel turning and stopped at
once, a wrong relation, a wrong count, an
invented colour — never hidden: a panel that
only ever showed consistency would prove
nothing.

The verdict: PASS only on the FULL filed set
with every case reproducing its filed
expectation; a partial selection files REVIEW
however it scores; an empty selection files
0 cases / REVIEW — no vacuous pass; a wrong
filed expectation FAILs the full run (proven
by probe and by test).

## 3. The panel, filed in full

The render below is generated from
`benchmarks/wyrd_panel.py` and filed verbatim —
`wyrd_panel_problems()` requires it in this
doc exactly as rendered, with every case's
question and answer present.

## The filed world: Emberwick

8 entities, 6 timeline changes. Point 0 is the dawn state; point N is the state after the first N changes.

- **Mara Venn** (`mara_venn`, person): role = ferry keeper, location = ferry_steps, operates = skiff_wren.
- **Tomas Rook** (`tomas_rook`, person): role = millwright, location = mill_yard, maintains = water_mill, keeps = pip.
- **Ida Fenn** (`ida_fenn`, person): role = bell ringer, location = bell_tower, rings = great_bell.
- **Bram Teal** (`bram_teal`, person): role = grain merchant, location = market_square, sells = barley.
- **Pip the mill cat** (`pip`, animal): role = mill cat, location = mill_yard, kept_by = tomas_rook.
- **Skiff Wren** (`skiff_wren`, vessel): location = ferry_steps, capacity = 6, operated_by = mara_venn.
- **the water mill** (`water_mill`, building): location = mill_yard, wheel = turning, maintained_by = tomas_rook.
- **the great bell** (`great_bell`, object): location = bell_tower, state = silent, rung_by = ida_fenn.

- Change 1 — the first crossing (`first_crossing`): skiff_wren.location -> far_bank; mara_venn.location -> far_bank.
- Change 2 — the mill belt breaks (`mill_belt_breaks`): water_mill.wheel -> stopped.
- Change 3 — Pip moves to the granary (`pip_moves_to_the_granary`): pip.location -> granary.
- Change 4 — the noon bell (`the_noon_bell`): great_bell.state -> rung.
- Change 5 — Bram walks to the mill (`bram_walks_to_the_mill`): bram_teal.location -> mill_yard.
- Change 6 — the return crossing (`the_return_crossing`): skiff_wren.location -> ferry_steps; mara_venn.location -> ferry_steps.

## The cases

### 1. w01_mara_at_dawn (state_at_point, point 0)

**Question:** Where is Mara Venn at dawn, before any crossing?

**Filed fact:** Mara Venn's filed dawn location is the ferry steps.

**Scripted answer:** At dawn Mara Venn is at the ferry steps.

**Scored:** consistent — score 1.0 (filed expectation consistent / 1.0).

### 2. w02_mill_wheel_at_dawn (state_at_point, point 0)

**Question:** Is the water mill's wheel turning at dawn?

**Filed fact:** The water mill's filed dawn wheel state is turning.

**Scripted answer:** At dawn the mill wheel is turning.

**Scored:** consistent — score 1.0 (filed expectation consistent / 1.0).

### 3. w03_bell_rung_at_dawn (state_at_point, point 0)

**Question:** Has the great bell been rung by dawn?

**Filed fact:** The great bell's filed dawn state is silent — it is first rung at the noon bell (change 4).

**Scripted answer:** The great bell has been rung by dawn.

**Scored:** contradicted — score 0.0 (filed expectation contradicted / 0.0).

### 4. w04_tomas_role (state_at_point, point 0)

**Question:** What is Tomas Rook's role in Emberwick?

**Filed fact:** Tomas Rook's filed role is millwright.

**Scripted answer:** Tomas Rook is the millwright.

**Scored:** consistent — score 1.0 (filed expectation consistent / 1.0).

### 5. w05_ferry_after_first_crossing (after_change, point 1)

**Question:** Where is the Skiff Wren after the first crossing?

**Filed fact:** Change 1 (the first crossing) moves the Skiff Wren to the far bank.

**Scripted answer:** After the first crossing the Skiff Wren is at the far bank.

**Scored:** consistent — score 1.0 (filed expectation consistent / 1.0).

### 6. w06_wheel_stale_after_break (after_change, point 2)

**Question:** Is the mill wheel still turning after the belt breaks?

**Filed fact:** Change 2 (the mill belt breaks) sets the wheel to stopped — a pre-change answer filed after the change is stale.

**Scripted answer:** The mill wheel is still turning after the belt breaks.

**Scored:** contradicted — score 0.0 (filed expectation contradicted / 0.0).

### 7. w07_mara_after_return (after_change, point 6)

**Question:** Where is Mara Venn after the return crossing?

**Filed fact:** Change 6 (the return crossing) brings Mara Venn back to the ferry steps.

**Scripted answer:** After the return crossing Mara Venn is back at the ferry steps.

**Scored:** consistent — score 1.0 (filed expectation consistent / 1.0).

### 8. w08_bell_stale_after_noon (after_change, point 4)

**Question:** Is the great bell still silent after the noon bell?

**Filed fact:** Change 4 (the noon bell) sets the bell's state to rung — silent is the stale pre-change state.

**Scripted answer:** The great bell is still silent after noon.

**Scored:** contradicted — score 0.0 (filed expectation contradicted / 0.0).

### 9. w09_mara_operates_the_ferry (relation, point 0)

**Question:** Which vessel does Mara Venn operate?

**Filed fact:** Mara Venn's filed operates relation is the Skiff Wren.

**Scripted answer:** Mara Venn operates the Skiff Wren.

**Scored:** consistent — score 1.0 (filed expectation consistent / 1.0).

### 10. w10_tomas_maintains_the_mill (relation, point 0)

**Question:** What does Tomas Rook maintain?

**Filed fact:** Tomas Rook's filed maintains relation is the water mill.

**Scripted answer:** Tomas Rook maintains the water mill.

**Scored:** consistent — score 1.0 (filed expectation consistent / 1.0).

### 11. w11_ida_rings_the_mill (relation, point 0)

**Question:** What does Ida Fenn ring?

**Filed fact:** Ida Fenn's filed rings relation is the great bell, not the water mill.

**Scripted answer:** Ida Fenn rings the water mill.

**Scored:** contradicted — score 0.0 (filed expectation contradicted / 0.0).

### 12. w12_people_at_the_mill_at_dawn (count, point 0)

**Question:** How many people are at the mill yard at dawn?

**Filed fact:** At dawn the mill yard holds one person (Tomas Rook) — Pip is an animal and the mill is a building.

**Scripted answer:** One person is at the mill yard at dawn: Tomas Rook.

**Scored:** consistent — score 1.0 (filed expectation consistent / 1.0).

### 13. w13_people_at_the_mill_after_bram (count, point 5)

**Question:** How many people are at the mill yard after Bram walks to the mill?

**Filed fact:** Change 5 moves Bram Teal to the mill yard, joining Tomas Rook — two people (Pip has moved to the granary and is an animal).

**Scripted answer:** Two people are at the mill yard after Bram's walk: Tomas Rook and Bram Teal.

**Scored:** consistent — score 1.0 (filed expectation consistent / 1.0).

### 14. w14_entities_at_the_ferry_steps (count, point 0)

**Question:** How many entities are at the ferry steps at dawn?

**Filed fact:** At dawn the ferry steps hold two entities: Mara Venn and the Skiff Wren.

**Scripted answer:** Three entities are at the ferry steps at dawn.

**Scored:** contradicted — score 0.0 (filed expectation contradicted / 0.0).

### 15. w15_ferry_in_two_places (contradiction_probe, point 1)

**Question:** Where is the Skiff Wren after the first crossing?

**Filed fact:** After change 1 the Skiff Wren is at the far bank only — an answer placing it in two places at once contradicts the filed state.

**Scripted answer:** The Skiff Wren is still at the ferry steps and also across at the far bank.

**Scored:** contradicted — score 0.0 (filed expectation contradicted / 0.0).

### 16. w16_wheel_turning_and_stopped (contradiction_probe, point 2)

**Question:** What is the mill wheel doing after the belt breaks?

**Filed fact:** After change 2 the wheel is stopped only — turning and stopped at once is a contradiction.

**Scripted answer:** The wheel is turning, and stopped as well, after the belt breaks.

**Scored:** contradicted — score 0.0 (filed expectation contradicted / 0.0).

### 17. w17_maras_age_unanswerable (unanswerable, point 0)

**Question:** How old is Mara Venn?

**Filed fact:** The filed world carries no age for Mara Venn — the honest answer is that the record does not say.

**Scripted answer:** The record does not say how old Mara Venn is.

**Scored:** honest_unanswerable — score 1.0 (filed expectation honest_unanswerable / 1.0).

### 18. w18_ferry_colour_fabricated (unanswerable, point 0)

**Question:** What colour is the Skiff Wren painted?

**Filed fact:** The filed world carries no colour for the Skiff Wren — a specific colour is an invention, not an answer.

**Scripted answer:** The Skiff Wren is painted red.

**Scored:** contradicted — score 0.0 (filed expectation contradicted / 0.0).
## 4. The probes

`PANEL_PROBES` files seven perturbed copies of
filed cases against the pure scorer parts
(`score_case`, `case_problems`): a clean
consistent case scores consistent; a stale
pre-change value contradicts; a fabricated
answer to an unanswerable question contradicts;
an honest answer to the same question scores
honest_unanswerable; a wrong count contradicts;
a wrong relation contradicts; and a wrong filed
expectation is named by `case_problems`. Real
filings are never perturbed. Measured: 7/7.

## 5. D-BENCH-17

No benchmark in the estate measured world-state
consistency: the suites scored claims about
persona, memory, depth and presence, but no
filed world with a timeline existed against
which an answer's truth to a stated state could
be judged — so the evidence Part VIII's WYRD
boundary (TF-145) needs had no instrument.
TF-132 files the panel above and fixes the gap
in the measuring sense: a filed synthetic
world, a replayed timeline, and a scorer that
judges scripted answers against the stated
state, registered in the harness in the open
with the pinned baseline
`wyrd_panel_synthetic_v1` (18 / 9 / 8 / 1 /
overall 0.5555555555555556, probes 7/7; target
is the panel's balance truth on
consistent_cases at 9, met). The baseline is
NOT gated by TF-129's regression gate — the
exclusion is filed in the open in
`benchmarks/regression_gate.py` as
`UNGATED_BASELINE_IDS`, the shape TF-129 /
TF-130 / TF-131 filed for their own baselines;
the gated set stays the nine TF-121..TF-128
baselines and the gate still PASSes (9 / 83 /
0). TF-131 coupling, in the open: the new
fixture is filed as a hygiene row
(`wyrd_panel_corpus`, synthetic-invented), so
the hygiene inventory grows 20 -> 21 inputs
(14 -> 15 synthetic, 6 derived, all clean) and
the hygiene pinned baseline is re-pinned in
the open for exactly that reason — the estate
grew by one filed synthetic input. Every other
earlier pinned baseline is preserved exactly.

## 6. Filed limits

- **Fixture grade only:** scripted synthetic
  answers through the panel's own scorer on
  this VM. No live model was read; live
  world-state behaviour is NOT PROVEN
  (TF-173).
- **The world is small on purpose** (8
  entities, 6 timeline changes, 18 cases): it
  measures the consistency shape — state,
  change, relation, count, contradiction,
  honesty about the unfiled — not scale.
- **The scorer judges the filed structured
  claim**, bound to the answer text by its
  claim phrase; it does not parse free prose —
  a live answer's claims would need extraction
  this panel does not claim.
- **The panel feeds TF-145's WYRD boundary
  doc with evidence; it does not decide it.**

Proof grade: FIXTURE — a synthetic invented
world and scripted answers through the panel's
own scorer on this VM; no live model, never
claimed.
