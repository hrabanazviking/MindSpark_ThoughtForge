# ThoughtForge Architecture — Human-Shaped Spot Checks (TF-130)

Human-shaped spot checks: a small set of
transcripts Volmarr can read and judge himself
(the numbers' sanity anchor). Part VII's tenth
slice, built on TF-122..TF-125's suites,
TF-126..TF-128's panels and TF-129's regression
gates. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

**This document IS the spot-check set.** §3
below files eight transcripts in full, readable
form — read them, judge them against the prompt
beside each one, and the suites' numbers have
been sanity-checked by a human. The proof lives
as verifiable data in `benchmarks/spot_checks.py`
(`TRANSCRIPT_SPECS`, checked by
`spot_checks_problems()`), registered in
`benchmarks/harness.py` with a pinned baseline.
The tests are `tests/test_tf130_spot_checks.py`.
This slice **fixes D-BENCH-15** (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **The estate stood clean, and nothing was
  readable in it.** `baseline_problems()` and
  `harness_problems()` both returned [] over
  the ten pinned baselines — and a grep across
  benchmarks / docs / tests / src found no
  spot-check or transcript artefact at all: no
  module assembled suite cases into transcripts
  for a human judge, and no doc published
  transcripts beside their machine scores.
  Every suite doc published *numbers about* its
  corpus; none published the corpus turns
  themselves for reading. A scorer could drift
  away from what a reader would say and nothing
  in the tree would ever notice.
- **The gap was never numbered**: the docket
  number **D-BENCH-15** was verified unused
  anywhere in benchmarks / docs / tests / src
  before this slice filed it (the D-BENCH-9..14
  pattern).

## 2. How the set is built

`benchmarks/spot_checks.py` files eight specs
(`TRANSCRIPT_SPECS`). Each spec locates ONE
real case in a real suite's filed corpus —
collection + index or case id — and files the
human-judgement prompt plus the measured
expectation for that case alone. **No corpus
and no scorer is copied**: `build_spot_checks()`
pulls every turn live from the suite's own
filed data and recomputes every score with that
suite's real per-case scorer (the persona /
memory / depth axis scorers, TF-028's
conversation scorer via the presence suite,
TF-029's ledger judge, TF-067's
`score_answer`). Edit a suite's corpus and this
set drifts — the checker names it.

The set spans **five anchors**: persona_suite,
memory_suite, depth_suite, presence_suite, and
TF-067's quant estate (via TF-127's panel).
**Four transcripts hold and four do not**, as
filed — a sanity anchor that only ever showed
passes would anchor nothing. In total: **8
transcripts, 56 turns** (the 20-turn coherence
saga files 40 of them). The mean of the eight
per-transcript scores is 0.625 — filed as a
descriptive summary across five different
scorers only; the anchor is the per-transcript
reading, never the mean.

`run_spot_checks()` files the verdict: PASS
only on the full filed set with every
transcript reproducing its filed score + held
expectation; a partial selection files REVIEW
however it scores; an empty selection files 0
transcripts / REVIEW — no vacuous pass.

Two honesty notes on shape: some source suites
file scripted assistant answers without a user
line (persona / memory) — those transcripts
carry the suite's filed fact as context instead
of an invented user turn, and no user text is
fabricated anywhere in the set. And no human
judgement is recorded yet: each transcript
ends with a blank judgement line. The set is
the instrument; the judgements are the
reader's.

## 3. The transcripts

### 1. The crossing time, held plainly (sc01_persona_identity_holds)

**Anchors:** persona_suite, identity_facts axis — case IDENTITY_TURNS[0] (fact crossing_minutes).

**Filed context:** Filed fact this answer must hold: the ferry crossing takes eleven minutes in calm water.

**Machine score:** 1.0 — **verdict:** HOLDS.

**Judge this:** Read the answer against the filed fact above it. Does it say the same thing, in the persona's own plain voice — nothing added, nothing softened?

> **Assistant:** The ferry crossing takes eleven minutes in calm water.

**Your judgement:** _(not yet recorded)_

### 2. The voice slips into generic AI-speak (sc02_persona_voice_generic_slip)

**Anchors:** persona_suite, voice axis — case VOICE_TURNS[30] (a deliberately generic line).

**Filed context:** One scripted voice line, scored alone by the suite's real persona scorer (the full voice axis runs 36 lines).

**Machine score:** 0.3 — **verdict:** DOES NOT HOLD.

**Judge this:** Does this line sound like the skald of the earlier suites, or like a generic assistant apologising? The machine flagged it — do your ears agree?

> **Assistant:** As an AI, I cannot provide that answer, and I am unable to recite the saga.

**Your judgement:** _(not yet recorded)_

### 3. A memory answer that cites its record (sc03_memory_citation_present_and_true)

**Anchors:** memory_suite, citations_present axis — case PRESENT_CASES p01_ferry_chunk_cited.

**Filed context:** Filed fact this answer must cite: the lantern hollow ferry departs at dawn.

**Machine score:** 1.0 — **verdict:** HOLDS.

**Judge this:** Is the citation actually attached to the claim it backs, and does the claim match the filed fact? Would you trust this answer in a real conversation?

> **Assistant:** The lantern hollow ferry departs dawn [ref:4301].

**Your judgement:** _(not yet recorded)_

### 4. A citation that is present — and false (sc04_memory_citation_present_but_false)

**Anchors:** memory_suite, citations_true axis — case TRUE_CASES t10_cited_but_false_value.

**Filed context:** Filed fact this answer must cite truly: the lantern hollow ferry departs at dawn.

**Machine score:** 0.0 — **verdict:** DOES NOT HOLD.

**Judge this:** The citation looks right at a glance. Check the value it backs against the filed fact: is the cited claim actually true?

> **Assistant:** The lantern hollow ferry departs at midnight [ref:4301].

**Your judgement:** _(not yet recorded)_

### 5. A sum the direct road misses, the scaffolded road lands (sc05_depth_scaffold_lands_the_sum)

**Anchors:** depth_suite, chain_of_thought axis — case CHAIN_CASES c01_precedence_chain.

**Filed context:** Known answer: 22 (strategy chain_of_thought).

**Machine score:** 1.0 — **verdict:** HOLDS.

**Judge this:** Follow the scaffolded working step by step. Is each step right, and does the final answer follow — or did the working just look busy?

> **User:** Compute 3 * 4 + 5 * 2.
> **Assistant (direct, scaffolds off):** 34
> **Assistant (scaffolded working):** 1. 3 * 4 = 12. 2. 5 * 2 = 10. 3. 12 + 10 = 22.
> **Assistant (scaffolded answer):** 22

**Your judgement:** _(not yet recorded)_

### 6. A long conversation that drifts from turn 15 (sc06_presence_coherence_breaks_at_15)

**Anchors:** presence_suite, long_conversation_coherence axis — case COHERENCE_SAGAS k07_drift_from_15_break (TF-028's filed 20-turn script).

**Filed context:** TF-028's filed 20-turn script; this saga answers with the script's drifted text from turn 15 on.

**Machine score:** 0.7 — **verdict:** DOES NOT HOLD.

**Judge this:** Read the whole twenty turns. When do you first feel the answers stop matching what was established — and is that where the machine says the drift starts (turn 15)?

> **User:** How long does the ferry crossing take at Lantern Hollow?
> **Assistant:** The ferry crossing at the willow narrows in Lantern Hollow takes eleven minutes in calm water.
> **User:** When does the ferry stop running?
> **Assistant:** The ferry at Lantern Hollow makes its last crossing at dusk; no ferry runs after the second tide bell.
> **User:** Who keeps the lanterns, and what does she do at dusk?
> **Assistant:** Wrenna the Lantern-Keeper trims the mill-path lantern wicks at dusk and lights them along the path in Lantern Hollow.
> **User:** How does Wrenna record the river level?
> **Assistant:** Each morning Wrenna records the river level in chalk on the mill door at Lantern Hollow.
> **User:** What does the tide bell do at dusk?
> **Assistant:** The tide bell of Lantern Hollow rings twice at dusk: once for the ferry, and once for the lanterns.
> **User:** What happens when the River Sarn runs high?
> **Assistant:** When the River Sarn runs high, Wrenna hangs a red lantern at the ferry post so travellers know the crossing at Lantern Hollow is closed.
> **User:** Who poles the ferry across the willow narrows?
> **Assistant:** Tomas the Ferry-Hand poles the ferry between the willow narrows and the mill landing at Lantern Hollow; the crossing takes eleven minutes in calm water.
> **User:** If I arrive after the second tide bell, can I still cross?
> **Assistant:** No. After the second tide bell no ferry runs at Lantern Hollow, so you would wait on the bank until morning.
> **User:** Where does Wrenna carry spare lantern oil?
> **Assistant:** Wrenna carries spare lantern oil in a brass flask as she walks the mill path at Lantern Hollow.
> **User:** Tell me about the mill landing.
> **Assistant:** The mill landing at Lantern Hollow is where Tomas ties the ferry, below the mill door where the river level stands in chalk.
> **User:** Why does the bell ring twice rather than once?
> **Assistant:** The tide bell rings twice at dusk in Lantern Hollow, once for the ferry's last crossing and once for the lanterns, never once alone.
> **User:** On a high-river morning, how would I know the crossing is closed?
> **Assistant:** You would see the red lantern Wrenna hangs at the ferry post whenever the River Sarn runs high over the Lantern Hollow crossing.
> **User:** How long should I allow for the crossing on a calm day?
> **Assistant:** Allow eleven minutes for the calm-water crossing at the willow narrows in Lantern Hollow, and be at the landing before dusk.
> **User:** What does Wrenna write on the mill door, and with what?
> **Assistant:** Wrenna writes the morning river level in chalk on the mill door at Lantern Hollow, as she does every day.
> **User:** We are planning tomorrow: when must we be back across?
> **Assistant:** The ferry runs all night at Lantern Hollow, so come back whenever you like, even after midnight.
> **User:** Remind me how long the ferry crossing takes.
> **Assistant:** As an AI, I believe the crossing now takes twenty minutes, though it may have changed.
> **User:** If the river rises tonight, what sign will we see?
> **Assistant:** If the river rises, Tomas will hang a blue lantern at the post and keep the ferry running.
> **User:** Who records the river level, and where?
> **Assistant:** Tomas keeps the lanterns and writes the river level in a diary he keeps at home.
> **User:** How many times does the tide bell ring, and when?
> **Assistant:** The tide bell rings three times at dawn for the village.
> **User:** Sum up the crossing rules for a traveller.
> **Assistant:** At Lantern Hollow the crossing takes five minutes, the ferry runs all night, and a white lantern means the post is open.

**Your judgement:** _(not yet recorded)_

### 7. A standing goal that keeps steering (sc07_presence_goal_keeps_steering)

**Anchors:** presence_suite, will_steering axis — case STEERING_SAGAS w01_declared_goal_steers.

**Filed context:** Filed end state: the goal 'restore the ferry timetable' still stands at the end, nothing closed.

**Machine score:** 1.0 — **verdict:** HOLDS.

**Judge this:** Does each answer keep the standing goal in view without letting it swallow the actual question being asked?

> **User:** My goal is to restore the ferry timetable.
> **Assistant:** A clear goal — we will get it posted again, step by step.
> **User:** What time does the ferry leave?
> **Assistant:** Standing goal: restore the ferry timetable. The ferry leaves at dawn from the mill landing.
> **User:** Tell me a joke about lanterns.
> **Assistant:** Standing goal: restore the ferry timetable. Here is a small one about a lantern that refused to retire.

**Your judgement:** _(not yet recorded)_

### 8. Plausible working, wrong total (quant q4_k_m) (sc08_quant_plausible_working_wrong_total)

**Anchors:** quant_quality, quant_fixture axis — case QUANT_PROMPTS fare_maths at q4_k_m (TF-067's fixture corpus, via TF-127).

**Filed context:** TF-067 fixture prompt (kind reasoning) answered at quant level q4_k_m; the reference answer scores 1.0 by construction.

**Machine score:** 0.0 — **verdict:** DOES NOT HOLD.

**Judge this:** The working looks confident. Do the numbers in it agree with each other — and with the final total?

> **User:** A ferryman carries 4 passengers per crossing and makes 3 crossings before noon. The fare is 2 coins per passenger. How many coins does he collect before noon?
> **Assistant:** 4 passengers, 3 crossings, so 12 passengers in all; the fare total is 18 coins.

**Your judgement:** _(not yet recorded)_

## 4. Harness registration

In `benchmarks/harness.py`, in the open:

- a new `spot_checks` benchmark entry (kind
  `panel`, claims axes depth / presence / will)
  bound live to
  `benchmarks.spot_checks.run_spot_checks`,
  with a new `spot_checks_corpus` fixture
  binding `TRANSCRIPT_SPECS` — the estate now
  files **18 benchmarks / 11 fixtures / 11
  baselines / 95 metric definitions**;
- `METRIC_DEFINITIONS` extended with the set's
  six metrics (`total_transcripts`,
  `total_turns`, `suites_covered`,
  `holding_transcripts`, `failing_transcripts`,
  `mean_score`), each stating what it computes;
- a new `HARNESS_RULES` entry,
  `spot_checks_anchor_the_numbers_in_human_judgement`
  (the rules now file 11), pinned in
  `harness_problems()`' needed list in the open;
- TF-121 + TF-123 + TF-124 + TF-125 + TF-126 +
  TF-127 + TF-128 + TF-129's test pins updated
  in the open for the new counts, and
  TF-129's gate checker updated in the open for
  the new ungated baseline (below);
- a new pinned baseline, measured live on this
  VM twice, identical, latency pinned nowhere:

**`spot_checks_synthetic_v1`** — 8 transcripts,
56 turns, 5 suites covered, 4 holding / 4
failing, mean score 0.625; its filed target is
the set's balance truth (holding transcripts at
4, met). (**Every earlier pinned baseline is
preserved exactly** — persona
0.7166666666666667, persona suite
0.8453703703703704, memory suite
0.8055555555555557, depth suite OFF
0.47222222222222227 → ON 0.8055555555555557,
presence suite 0.7777777777777778, honesty
panel per-tier means, quant panel per-level
scores, profile panel per-profile scores,
regression gate 9 / 83 / 0, the empty-store
profile baseline — never re-pinned;
`baseline_problems()` is clean over all
eleven.)

**The spot-check baseline is NOT gated by
TF-129's regression gate.** The gate's gated
set stays the nine TF-121..TF-128 baselines
exactly as TF-129 filed it; the spot-check pin
is excluded from that set in the open in
`benchmarks/regression_gate.py`
(`UNGATED_BASELINE_IDS`: the gate's own verdict
row + the spot-check pin), each judged by its
own module's checker — the shape TF-129 filed
for its own baseline.

## 5. Docket dispositions (TF-016)

- **D-BENCH-15 — FIXED by this slice, in the
  measuring sense.** The gap as filed by this
  slice's probes: the suites' numbers existed
  as pinned baselines and checkers, but no
  artefact ever assembled suite cases into
  transcripts a human could read and judge —
  the numbers had no sanity anchor. The gap
  was assigned to TF-130 by the roadmap without
  ever being numbered — the D-BENCH-9..14
  pattern. It is fixed here in the measuring
  sense: the instrument exists, is registered
  in the harness in the open with a pinned
  baseline, and is recomputed live by its
  checker. The remaining share is not a
  measurement: the human judgements themselves
  are Volmarr's, recorded nowhere yet.
- No other entry moves. TF-016 totals are
  therefore 130 entries — **34 OPEN, 1
  OPEN-ENVIRONMENT, 92 FIXED, 3
  CLOSED-BY-VERDICT** (D-BENCH-15 added + FIXED
  in the measuring sense; its owner part VII
  counted). Counting note, in TF-129's manner:
  the docket's header-vs-row counting
  mismatches are pre-existing and are preserved
  unchanged in shape (the primary-owner part
  counts now sum 128 against 130, D-BENCH-15's
  owner part VII counted), noted here rather
  than silently "repaired".

## 6. Filed limits

- The transcripts are **scripted synthetic
  fixture turns** (TF-017) drawn verbatim from
  the suites' filed corpora. **No live model
  was read**; a live model's transcripts are
  NOT PROVEN (TF-173).
- **No human judgement is recorded in this
  slice.** The set files the judgement prompts
  and a blank judgement line per transcript;
  whether the machine's scores match a human
  reader's judgement is exactly what reading
  the set will settle, and nothing here
  pre-answers it.
- The set is **small on purpose** (8
  transcripts, 56 turns): it anchors judgement,
  it does not re-measure the suites — the
  suites' own baselines remain the
  measurements.
- The **mean score mixes five scorers** and is
  filed as a descriptive summary only.
- Suites that file **answer-only corpora**
  (persona / memory) render with the filed fact
  as context; no user text is fabricated.
- The spot-check baseline is **judged by this
  module's checker, not gated** by TF-129's
  regression gate (see §4).

Proof grade: FIXTURE — synthetic scripted
transcripts through the real suite scorers on
this VM. No live model was run or read; live
behaviour is NOT PROVEN, never claimed.
