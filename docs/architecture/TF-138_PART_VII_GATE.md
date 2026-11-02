# TF-138 — Part VII Gate: Benchmarks & the Proof Harness (Part VII Gate)

Closed 2026-11-01 on branch `yrsa/thoughtforge-road-build`.
Roadmap wording: "Part VII gate: the full panel green on the
campaign tree; the baseline book published; summary filed."

## The gate verdict

**Part VII is COMPLETE. All 17 build slices (TF-121–TF-137)
are banked with evidence in `docs/architecture/`, and the
gate proof was re-measured fresh at this gate — not quoted
from its slices:**

1. **The full panel, run live at this gate through TF-137's
   one command — GREEN.** `python3 -m
   benchmarks.harness_tool --panel full_panel` (with
   `PYTHONPATH=src`) printed, verbatim:

   ```
   harness-tool run: panel=full_panel members=18 passed=18 verdict=PASS
   ```

   All 18 member baselines passed, one member line each,
   no problem lines (log
   `~/workspace/.verify-tmp/tf138_harness_tool.log`).
   The same run through the Python surface
   (`run_harness_tool()`) filed the same verdict:
   panel `full_panel`, 18 requested / 18 rows,
   0 selection problems, PASS.
2. **Every Part VII module checker, run live at this
   gate — all clean.** Each of the eighteen
   `*_problems()` functions enumerated from the
   `benchmarks/` modules returned `[]`:
   `harness_problems()`, `baseline_problems()` (over all
   eighteen pinned baselines), `persona_suite_problems()`,
   `memory_suite_problems()`, `depth_suite_problems()`,
   `presence_suite_problems()`, `honesty_panel_problems()`,
   `quant_panel_problems()`, `profile_panel_problems()`,
   `regression_gate_problems()`, `spot_checks_problems()`,
   `data_hygiene_problems()`, `wyrd_panel_problems()`,
   `cost_panel_problems()`, `baseline_book_problems()`,
   `flake_discipline_problems()`,
   `public_claims_problems()` and
   `harness_tool_problems()` (logs
   `~/workspace/.verify-tmp/tf138_checkers.log` and
   `~/workspace/.verify-tmp/tf138_runs.log`).
3. **Every Part VII run surface, run live at this gate —
   PASS.** The measured headlines: persona suite PASS
   (108 turns, overall 0.8453703703703704); memory suite
   PASS (36 cases, overall 0.8055555555555557); depth
   suite PASS (36 cases, overall on 0.8055555555555557
   vs off 0.47222222222222227, delta
   0.3333333333333334); presence suite PASS (36 cases,
   overall 0.7777777777777778); honesty panel PASS
   (16 cells, helps most at the mid_8b tier); quant
   panel PASS (48 cells, 6 overall rows + 18 kind rows);
   profile panel PASS (30 cells, 1 refused cell —
   pi_zero / depth_on, filed); regression gate PASS
   (9 baselines / 83 metrics compared / 0 failures);
   spot checks PASS (8 transcripts / 56 turns, 4 holding
   / 4 failing as filed, mean 0.625); data hygiene PASS
   (21 inputs, all clean); WYRD panel PASS (18 cases);
   cost panel PASS (24 cells, total token budget
   766800; power filed `not_proven_not_measured` on
   every profile, as TF-133 filed); baseline book PASS
   (18 entries, doc current); flake discipline PASS
   (18 censused baselines); public claims PASS
   (21 claims / 12 matched / 9 struck, recomputed live
   in the detail probe).
4. **The baseline book is published and current.**
   `docs/architecture/BASELINE_BOOK.md` exists (18
   baselines / 146 pinned metrics / 9 gated / 9 ungated
   in its header), `baseline_book_problems()` is clean,
   `run_baseline_book()` files PASS with
   `doc_current=True`, and regenerating it by its
   command (`python3 -m benchmarks.baseline_book`,
   log `~/workspace/.verify-tmp/tf138_book_regen.log`:
   "baseline book regenerated: 18 baselines / 146
   pinned metrics") produced **no diff** — `git status`
   stayed clean, so the filed book was not stale.
5. **Targeted + full suites re-run at the gate** (see
   "Suite at the gate" below): the Part VII targeted set
   (all 17 test files, TF-121–TF-137) = **453 tests,
   453 passed**; the full suite stands exactly on
   TF-137's banking line.

The gate's fresh probes found **no defect in the
estate**: nothing was fixed at this gate, and this
slice is gate doc + docket rulings only, exactly as
TF-120's and TF-104's gates were. One environmental
incident during the gate's detail work is filed
honestly in the suite section below — a sandbox
EPERM write-block window (the TF-134–TF-137 incident
shape) that made tempfile unusable and, in that
window only, made a standalone flake-census re-run
report its tempfile-dependent measurers
`unmeasurable`; the window is an environment fault,
not an estate defect, and every number filed at this
gate was measured outside it (the checkers, runs,
targeted set and full gate all completed green
before it opened, and the flake census was
re-verified after it cleared — see "Suite at the
gate").

The proof is FIXTURE grade: pinned synthetic-fixture
baselines re-measured live through the real scorers
and panels on this VM. A live model's panel results
are **NOT PROVEN** — no live model ran at this gate
or in any Part VII slice — and no claim at this gate
says otherwise (TF-173 owns the real-hardware road).

## 2. What the gate inherits, stated as one estate

Part VII is one harness, not eighteen tools that
happen to share a folder. TF-121's `benchmarks/harness.py`
is the registry every later slice binds: benchmarks as
literal data, fixtures, scorers, pinned baselines with
tolerances, and measurers that recompute every pin live.
Around that registry the part built, in order: the four
claim suites — persona consistency (TF-122, 108 scripted
turns across identity / voice / values-under-pressure),
memory fidelity (TF-123, cite-or-explain scored
mechanically, citations present *and* true), depth
(TF-124, multi-step reasoning with scaffolds on/off, the
scaffold delta measured at 0.3333, not assumed) and
presence (TF-125, long-conversation coherence + the will
layer's follow-through over scripted sagas); the three
cross-cutting panels — small-model honesty (TF-126, the
same suites across model sizes, where the forge helps
most published from results), quant-vs-quality (TF-127,
consuming TF-067's curve mechanically, no corpus or
scorer copied) and profile panels (TF-128, suites per
hardware profile with the profile's budgets enforced
during the run, refused cells named); and the proof
machinery that keeps the whole estate honest — TF-129's
regression gate (the nine TF-121..TF-128 baselines gated,
deltas beyond measured-noise tolerance fail), TF-130's
human-shaped spot checks (8 transcripts Volmarr can read
and judge himself, the numbers' sanity anchor), TF-131's
data hygiene (21 benchmark inputs, all synthetic or
licensed/derived with provenance; canon never a
benchmark input), TF-132's WYRD-shaped panel (world-state
consistency, feeding Part VIII's boundary with evidence),
TF-133's cost panel (tokens per suite per profile;
watts filed honestly NOT PROVEN), TF-134's baseline book
(every pinned baseline + method, generated by command),
TF-135's flake discipline (every pinned baseline
census-measured 5 times; no silent re-rolls), TF-136's
public-claims audit (every performance claim in the docs
matched to a suite result or struck in the audit doc)
and TF-137's harness-as-a-tool (the panels filed as data;
one command runs the panel a slice needs and prints the
deterministic citation line this gate cites in proof 1).

Two filed shapes the gate states plainly rather than
smoothing over: the gated set is exactly nine baselines
(TF-121..TF-128's) and the other nine pinned baselines
are ungated — each judged by its own module's checker
in its own slice, and re-measured against its pin by
the tool's audit panel — not by TF-129's gate; and
every number in the part is fixture grade, measured
over scripted synthetic corpora through the real
scorers — the panels have never faced a live model,
and the part's own filings (TF-121 onward) never
claimed they had.

## 3. Gate rulings on the entries and gaps assigned to this gate

1. **The D-BENCH family the part was opened to fix —
   discharged by their slices, verified at this gate.**
   Twenty of the twenty-two D-BENCH entries stand
   **FIXED** by their Part VII slices, each re-verified
   live at this gate by its module's checker returning
   `[]` and its run filing PASS: D-BENCH-2 and
   D-BENCH-6 by TF-121; D-BENCH-3 by TF-122; D-BENCH-9
   by TF-123; D-BENCH-10 by TF-124 (measuring sense);
   D-BENCH-11 by TF-125 (measuring sense); D-BENCH-12
   by TF-126 (measuring sense); D-BENCH-13 by TF-127
   (measuring sense); D-BENCH-1 by TF-128 (measuring
   sense — TF-121 fixed its validation share, TF-128
   its application share); D-BENCH-14 by TF-129;
   D-BENCH-15 by TF-130 (measuring sense); D-BENCH-16
   by TF-131 (measuring sense); D-BENCH-17 by TF-132
   (measuring sense); D-BENCH-18 by TF-133 (measuring
   sense); D-BENCH-7 by TF-133 (consumption share);
   D-BENCH-19 by TF-134 (measuring sense); D-BENCH-20
   by TF-135; D-BENCH-21 by TF-136; D-BENCH-22 by
   TF-137. No D-BENCH entry closes at this gate itself
   — each was fixed by its own slice with evidence, and
   the gate's fresh panel + targeted re-run found no
   regression in any of them. No new docket entry is
   added by this gate: the gate found no defect (the
   EPERM window below is an environment incident, the
   same shape already filed by TF-134..TF-137, not a
   new estate defect).
2. **D-BENCH-4 (the "100+ turns" persona claim; vacuous
   empty-list PASS) — stays OPEN, exactly as TF-136
   left it.** Both filed defects stand remedied in
   substance (TF-122: `score([])` returns 0.0 / REVIEW,
   and the 108-turn persona suite exists with its pin
   re-verified PASS at this gate), and TF-136 struck
   the claim's wording in the audit doc with the reason
   filed. What stands is only what the entry itself
   files: the source docs (BUILD_PLAN_v1.md,
   MODEL_CARD.md, TASK_PHASE6_RELEASE.md) still word
   the claim without pointing at the suite — the audit
   strikes in its own doc only and, by its filed rule,
   never edits source docs. The correction of the
   source docs belongs to Part X's doc work as already
   filed — **TF-184** (the final claims audit,
   completing TF-136 for the release) and, for the
   model card, D-CARD-1's filed owner **TF-183** —
   with **TF-192** verifying at the end that the entry
   is left open with this reason and no other.
3. **D-BENCH-5 (`locustfile.py` unrunnable as shipped;
   TASK_PHASE6's stale HttpUser description) — stays
   OPEN, exactly as TF-136 left it.** TF-121 decided
   and filed the locust file's fate in the harness
   index (`excluded_by_verdict`: a manual throughput
   script, never a suite), and TF-136 struck the stale
   HttpUser description in the audit doc with the
   reason filed; the source doc itself stands unedited
   by the audit's filed rule, which is the entry's
   remaining share. Its future is the same Part X road
   as ruling 2: **TF-184**'s final claims audit, with
   **TF-192** verifying it is left open with its
   reason.
4. **D-BENCH-8 (the `PYTHONPATH=src` requirement stated
   in no usage doc) — stays OPEN, owner TF-182 as
   filed; no Part VII share.** This gate notes it only
   because the gate's own first checker invocation
   (before the probe's path fix) and the keepers'
   TF-135/TF-137 verifications met the same shape: run
   without `PYTHONPATH=src`, the benchmark estate's
   imports fail — an invocation error, not a slice
   defect, exactly as those slices filed. TF-182
   (developer docs) owns stating the requirement; no
   claim at this gate touches it.
5. **D-CARD-1 (the root MODEL_CARD compatible-models
   table drifts from the profiles/catalogue) — stays
   OPEN on its TF-183 share.** Its filed owners are
   TF-136 + TF-183; TF-136's audit struck the card's
   *performance-claim* rows (citation / enforcement /
   average-words targets, the 100+-turns wording, the
   edge-speed claim) but the compatible-models *table*
   drift this entry files was not a performance-claim
   row and stands uncorrected in the source doc, as
   the entry itself states ("the root card is not
   edited by TF-078" — and not by TF-136 either).
   **TF-183** (doc care, existing text preserved,
   standing law) makes the correction; **TF-184**
   completes the claims audit for the release. No
   Part VII work remains on it.
6. **D-PROF-5 (profiles promise models their own
   caps/ceilings exclude) — stays OPEN; its TF-127
   share was NOT discharged, as TF-127 itself filed.**
   TF-127's panel consumed TF-067's curve mechanically
   (re-verified PASS at this gate), but the live panel
   TF-067 imagined — which might calibrate q2_k — did
   not and could not run here: no live model runs on
   this VM, as TF-127 filed in the open ("the live
   panel … and TF-068's live re-rank / q2_k
   calibration triggers (D-PROF-5's line) are NOT
   discharged and NOT claimed"). The entry's rows
   stand as TF-086's truth table v2 ruled them, and
   **TF-185** (release truth) carries them, as the
   entry's owner line files. The live-hardware legs
   that could move it are Part X's (**TF-173** /
   TF-175), not this gate's.
7. **Live-model claims generally — NOT PROVEN, owned
   by Part X as already filed.** Every struck
   live-model claim in TF-136's audit (the MODEL_CARD
   edge-speed claim among them) and every panel in
   this part points at the same road: **TF-173**
   (real-hardware lab I), TF-175 (the laptop line:
   the full Part VII panel on purplelongship-class
   hardware) and TF-186 (the small-model showcase).
   Nothing at this gate re-opens, re-assigns or
   softens those filings.

**Docket (TF-016) at the gate: 137 entries — 33 OPEN /
1 OPEN-ENVIRONMENT / 100 FIXED / 3 CLOSED-BY-VERDICT**
(unchanged by this gate — no entry closes here and
none is added; the rulings above record why the
entries that stay open stay open, in the docket's own
filed terms, and TF-192 verifies at the end that no
entry is left open without a reason. The pre-existing
header-vs-row counting mismatches filed by
TF-136/TF-137 — primary-owner parts as filed sum 134
against 137 — are preserved in shape, noted here as
there, not silently repaired. The docket file itself
is not edited by this gate.)

## 4. Plain-language summary (the Volmarr-facing half of this gate)

Here is what Part VII did, in plain words:

- **The forge's claims are now numbers with receipts.**
  Before this part, the project asserted it gave small
  models depth, presence and memory. Now each claim has
  a scored suite: character consistency over 108
  scripted turns, memory that must cite a real source
  or explain itself, reasoning measured with help
  switched on and off (the help is worth a measured
  third of the score, not a guess), and long-conversation
  coherence scored over scripted sagas.
- **The same tests were run across model sizes, quants
  and hardware profiles — and the awkward results are
  published, not buried.** The profile panel names the
  one cell it refuses (the smallest Pi profile cannot
  run the depth suite's refinement), the quant panel
  shows reasoning quality stuck at half from the
  cheapest quants until the top one, and the cost panel
  files its electricity numbers as NOT PROVEN rather
  than inventing watts.
- **The whole estate now guards itself.** A regression
  gate fails any future change that moves a pinned
  score beyond measured noise; every test input is
  filed as invented or licensed (your canon was never
  used as a benchmark input); every suite was measured
  five times over to prove it gives the same answer
  every time; and every performance claim in the
  project's documents was matched to a real result or
  struck through in a public audit — 12 matched, 9
  struck, including the old "100+ turns" wording.
- **At this gate the whole panel was run again, fresh,
  by the one command built for it — 18 of 18 passed**,
  every checker came back clean, the published
  baseline book (18 baselines, 146 pinned numbers,
  each beside its method) was regenerated and matched
  the filed copy exactly, and the full test suite
  stands exactly where the last slice left it.
- **Nothing here has faced a live model.** Every number
  is fixture grade — real scoring code, scripted
  answers, on this machine. How the forge scores a
  real small model, on real small hardware, is still
  **NOT PROVEN** — that road is Part X's (TF-173,
  TF-175), and this part's honesty machinery exists
  precisely so that road's numbers, when they come,
  can be believed.

## 5. The evidence, slice by slice

| Slice | Artefact | Headline |
|---|---|---|
| TF-121 | `HARNESS_ARCHITECTURE.md` | Harness architecture — benchmarks as code with fixtures, scorers, and pinned baselines (the `benchmarks/` estate, TF-010, unified); D-BENCH-2 + D-BENCH-6 fixed |
| TF-122 | `PERSONA_CONSISTENCY_SUITE.md` | Persona consistency suite — the existing persona_consistency benchmark completed into a scored suite (identity facts, voice, values under pressure); 108 turns, overall 0.8454 PASS; D-BENCH-3 fixed |
| TF-123 | `MEMORY_FIDELITY_SUITE.md` | Memory fidelity suite — cite-or-explain scored mechanically (citations present, citations *true*); D-BENCH-9 fixed |
| TF-124 | `DEPTH_SUITE.md` | Depth suite — multi-step reasoning with scaffolds on/off, the scaffold value measured (delta 0.3333), not assumed; D-BENCH-10 fixed in the measuring sense |
| TF-125 | `PRESENCE_SUITE.md` | Presence suite — long-conversation coherence + the will layer's follow-through (TF-029) scored over scripted sagas; D-BENCH-11 fixed in the measuring sense |
| TF-126 | `HONESTY_PANEL.md` | Small-model honesty panel — the same suites across model sizes, where the forge helps most, published from results; D-BENCH-12 fixed in the measuring sense |
| TF-127 | `QUANT_PANEL.md` | Quant-vs-quality panel (consumes TF-067 mechanically); D-BENCH-13 fixed in the measuring sense; D-PROF-5's live/q2_k share expressly NOT discharged |
| TF-128 | `PROFILE_PANEL.md` | Profile panels — suites per hardware profile with the profile's budgets enforced during the run; D-BENCH-1 fixed in the measuring sense |
| TF-129 | `REGRESSION_GATES.md` | Regression gates — benchmark deltas beyond tolerance fail the campaign's slice gate (thresholds from baseline noise, measured); D-BENCH-14 fixed |
| TF-130 | `SPOT_CHECKS.md` | Human-shaped spot checks — a small set of transcripts Volmarr can read and judge himself (the numbers' sanity anchor); D-BENCH-15 fixed in the measuring sense |
| TF-131 | `BENCHMARK_DATA_HYGIENE.md` | Benchmark data hygiene — all fixtures synthetic or licensed; canon never a benchmark input (21 inputs, all clean); D-BENCH-16 fixed in the measuring sense |
| TF-132 | `WYRD_PANEL.md` | The WYRD-shaped panel — world-state consistency questions (feeds Part VIII's boundary with evidence); D-BENCH-17 fixed in the measuring sense |
| TF-133 | `COST_PANEL.md` | The cost panel — tokens + time per suite per profile, sharpness filed per budgeted token and watts filed honestly NOT PROVEN; D-BENCH-18 + D-BENCH-7's consumption share fixed |
| TF-134 | `BASELINE_BOOK.md` | The baseline book — every suite's pinned baseline + method, in-repo, regenerated by command; D-BENCH-19 fixed in the measuring sense |
| TF-135 | `FLAKE_DISCIPLINE.md` | Flake discipline — nondeterministic suites quarantined or made deterministic (TF-036), no silent re-rolls; D-BENCH-20 fixed |
| TF-136 | `PUBLIC_CLAIMS_AUDIT.md` | Public-claims audit — every performance claim in the repo's docs matched to a suite result or struck in the audit doc (21 / 12 matched / 9 struck); D-BENCH-21 fixed; D-BENCH-4 + D-BENCH-5 narrowed, kept OPEN |
| TF-137 | `HARNESS_TOOL.md` | The harness as a tool — one command runs the panel a slice needs; keepers cite its output verbatim (full panel 18/18, the citation this gate's proof 1 re-ran); D-BENCH-22 fixed |

## Road re-cut assessment

Part VII changed no Part VIII–X slice's wording or
order. Assignments recorded here per TF-018's rule,
all exactly as already filed: D-BENCH-4 and D-BENCH-5
stay OPEN for Part X's doc road (**TF-184**'s final
claims audit; D-CARD-1's correction with **TF-183**)
with **TF-192** verifying each is left open with its
reason; D-BENCH-8 stands with **TF-182**; D-PROF-5's
rows stand with **TF-185**; live-model proof stands
with **TF-173** (real-hardware lab I), **TF-175**
(the laptop line — the full Part VII panel on
purplelongship-class hardware) and **TF-186** (the
showcase); the final baseline book is **TF-194**'s
(TF-134 regenerated at the release tree). Part VIII
inherits the WYRD-shaped panel's evidence exactly as
TF-132 filed it — world-state consistency measured
over fixtures — for its boundary verdicts, starting
with TF-139's bond graph verdict; nothing in Part
VII pre-decides any Part VIII verdict.

## Suite at the gate (measured this slice, TF-014 method)

`PYTHONPATH=src python3 -m pytest tests/ -q` on this
VM, read from the on-disk gate log
(`~/workspace/.verify-tmp/tf138_gate.log`):
**3573 collected / 3571 passed / 1 skipped /
1 failed (D-ENV-1, unchanged — the phase8 dependency
check naming sentence-transformers absent on this
VM, an environment entry not a code defect)** —
exactly TF-137's banking line. The gate's targeted
re-run (all 17 Part VII test files, TF-121–TF-137)
is **453 collected / 453 passed** (log
`~/workspace/.verify-tmp/tf138_targeted.log`), and
the gate proofs above were measured fresh at this
gate: the full panel through TF-137's command (proof
1), all eighteen `*_problems()` checkers and all
sixteen `run_*()` surfaces through direct probes
(logs `~/workspace/.verify-tmp/tf138_checkers.log` /
`tf138_runs.log`), the public-claims matched/struck
split (12 / 9) recomputed in a detail probe, and the
baseline book regenerated by its command with no
diff. This slice adds one gate doc plus the TF-016
docket rulings filed in §3 and no code or tests, so
the count does not move at banking.

**Incident, filed honestly (the TF-134–TF-137
shape):** after every run above had completed green,
a sandbox EPERM write-block window opened in this
worker's sandbox: Python writes landed 0 bytes /
`PermissionError: Operation not permitted`,
`tempfile` reported no usable temporary directory,
two detail-probe runs exited 120 with 0-byte logs,
and `git status` failed on its index.lock write. In
that window only, a standalone flake-census re-run
reported 15 deterministic / 2 `unmeasurable`
(profile_panel and profile_benchmark_empty_store —
both measurers raising the tempfile
`FileNotFoundError`) / 1 `nondeterministic_unfiled`
(the regression gate's `gate_failures`, a knock-on
of the same tempfile failures inside its member
runs) — an environment fault propagating through
tempfile-dependent measurers, not an estate defect:
the same census, in `flake_discipline_problems()`
and in the probe runs measured before the window,
recomputed the filed census exactly (18
deterministic, verdict PASS), and the full targeted
+ full suites — which include TF-135's own tests —
passed green before the window opened.
The window opened after the full gate log had
completed (18:55 EST) — first symptoms in the
detail probes at ~18:56 EST, `git status` failing
by 19:03 EST — and cleared on its own by the
~19:14 EST retry, roughly eighteen minutes, with
nothing deleted anywhere and no repo file truncated
(the tree, checked when `git status` worked again,
carried only this gate doc as new). After it
cleared, the flake census was re-run standalone in
this sandbox and re-verified exactly as filed —
**18 deterministic / 0 quarantined, verdict PASS,
`flake_discipline_problems()` []** — and the full
panel was re-run through TF-137's command and
printed the same citation line verbatim a second
time. A separate, later wall is filed for
completeness: after the window cleared, fresh
`exec` writes across `~/workspace` began failing
with "Disk quota exceeded" (a shared sandbox quota;
the file tools were unaffected, and every
`tf138_*.log` this gate cites had already landed
before it) — the post-window re-verifications above
were therefore run with stdout-only output, and no
evidence this gate relies on was written after that
wall appeared.


## What opens next

Part VIII — The Wyrdforge Wing: Graduate or Retire
(TF-139–TF-152), starting with TF-139: bond graph
verdict — wyrdforge's bond graph vs the shipping
knowledge store: adopt (as a layer), merge, or
retire, decided on TF-006's census + a working spike.
