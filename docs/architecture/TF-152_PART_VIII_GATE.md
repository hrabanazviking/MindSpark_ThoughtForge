# TF-152 — Part VIII Gate: The Wyrdforge Wing — Graduate or Retire (Part VIII Gate)

Closed 2026-11-02 on branch `yrsa/thoughtforge-road-build`.
Roadmap wording: "Part VIII gate: no research code left in
limbo; every verdict executed; the boundary doc shipped."

## The gate verdict

**Part VIII is COMPLETE. All 13 build slices (TF-139–TF-151)
are banked with evidence in `docs/architecture/`, and the
gate proof was re-measured fresh at this gate — not quoted
from its slices:**

1. **Every Part VIII module checker, run live at this
   gate — all clean.** Each of the fourteen Part VIII
   `*_problems()` functions returned `[]`:
   `bond_graph_verdict_problems()`,
   `micro_rag_verdict_problems()`,
   `persona_compiler_verdict_problems()`,
   `truth_calibrator_verdict_problems()`,
   `bond_graph_problems()` (TF-143),
   `persona_compiler_problems()` (TF-143),
   `wing_retirements_problems()` (TF-144),
   `wyrd_boundary_problems()` (TF-145),
   `bond_bridge_problems()` (TF-146),
   `runmal_cards_problems()` (TF-147),
   `research_hygiene_problems()` (TF-148),
   `wing_tests_problems()` (TF-149),
   `wing_provenance_problems()` (TF-150) and
   `panel_rerun_problems()` (TF-151). TF-151's
   neighbours were re-run live in the same pass and are
   also clean: `wyrd_panel_problems()` [],
   `persona_suite_problems()` [] and
   `baseline_problems()` [] over all pinned baselines
   (log `~/workspace/.verify-tmp/tf152_checkers.log`).
2. **Every Part VIII run surface, run live at this gate —
   PASS.** The measured headlines (log
   `~/workspace/.verify-tmp/tf152_runs.log`): the four
   verdict probes each 7/7 — TF-139 **ADOPT_AS_LAYER**
   (spike re-measured: repair_debt 0.35 / safety 0.725,
   closeness 0.4375 / sacred 0.5333 / rupture 0.2083,
   excerpt 9 lines), TF-140 **RETIRE** (shipping
   precision 0.8889 / recall 0.9630 / F1 0.9111 vs the
   micro-RAG spike's 0.3333 / 0.8889 / 0.4749, as filed
   in its verdict data), TF-141 **MERGE**, TF-142
   **RETIRE**; TF-143's production probes 7/7 in each
   module (bond parity exact against TF-139's spike);
   TF-144's retirement probes 7/7 with the archive state
   re-measured live (2 retirements / 3 archived files /
   3 bannered / 0 adopted files bannered / 6 lessons —
   3 from TF-140, 3 from TF-142); TF-145's boundary
   probes 7/7 with its state re-measured (12 ownership
   rows — ThoughtForge 6 / WYRD 3 / neither 3; 5
   interface rows — 2 in production / 2 fixture grade /
   1 not built; 2 consumers; WYRD panel PASS at 18
   cases; bond excerpt 9 lines; 0 live WYRD interface
   files, 0 live-client tokens); TF-146's bridge probes
   7/7 (9 exported sections / 4 reservations, as filed);
   TF-147's card probes **8/8** (14 fields filed — 5
   mapped / 5 filed-empty / 1 derived / 2
   exporter-constant / 1 extension; 14 source records);
   TF-148's hygiene probes 7/7; TF-149's wing-tests
   probes 7/7; TF-150's provenance probes 7/7; and
   TF-151's re-run probes 7/7 with its state re-measured
   in proof 4 below.
3. **No research code left in limbo — proven live, claim
   (a).** TF-148's live state, re-measured at this gate
   (`run_research_hygiene_state()`): every code/test
   file under `research_data/src` + `research_data/tests`
   files exactly one disposition — **11 file rows:
   graduated_source 2 / archived_retired 3 /
   fenced_research 3 / fenced_standalone 2 /
   fence_import 1**; **12 baseline areas** (13 area rows
   including this campaign's own fence file),
   **96 baseline tracked files / 97 tracked files with
   the fence file**, and the fence file present live
   (`fence_file_present: true`;
   `research_data/RESEARCH_FENCE.md` on disk). Nothing
   in the wing stands unclassified.
4. **Every verdict executed — proven live, claim (b),
   with the panel re-run standing behind it.** TF-139's
   ADOPT_AS_LAYER was executed by TF-143's production
   `src/thoughtforge/knowledge/bond_graph.py` (checker
   [] in proof 1) and TF-146's bond bridge built from
   it (checker []); TF-141's MERGE was executed by
   TF-143's production
   `src/thoughtforge/cognition/persona_compiler.py`
   (checker []) and TF-147's Rúnmál cards built from
   the compiled shape (checker [], probes 8/8);
   TF-140's RETIRE and TF-142's RETIRE were executed by
   TF-144: the retirement banners stand live at the
   heads of `research_data/src/wyrdforge/services/micro_rag_pipeline.py`,
   `research_data/src/wyrdforge/services/truth_calibrator.py`
   and their shared test file
   `research_data/tests/test_micro_rag_and_truth.py`
   — re-measured live in TF-144's archive state
   (3 files present, 3 bannered) and pinned by
   `wing_retirements_problems()` []. The adopted wing
   files carry **no** banner, checker-pinned live the
   same way (archive state: adopted files present 3,
   bannered 0 — the bond-graph and persona-compiler
   wing services among them, their heads re-read at
   this gate carrying no banner marker). And TF-151's
   re-run, re-measured live at this gate, stands
   exactly as filed: **19 metrics / 19 unmoved /
   0 moved, both panel verdicts PASS before and after**
   (WYRD panel PASS, overall 0.5555555555555556;
   persona suite PASS, 108 turns, overall
   0.8453703703703704; persona consistency
   0.7166666666666667) — the verdicts landed in other
   modules, no re-run panel consumes adopted code
   (7 verdict rows, 0 consumed, proven by TF-151's
   source inspection and re-verified by its checker
   here), so the zero deltas are the honest result.
5. **The boundary doc shipped — proven live, claim
   (c).** `docs/architecture/WYRD_BOUNDARY.md` exists,
   `wyrd_boundary_problems()` is [] (proof 1), and the
   doc is current by the module's own regeneration
   method: `python3 -m thoughtforge.wyrd_boundary`
   was re-run at this gate (log
   `~/workspace/.verify-tmp/tf152_wyrd_regen.log`:
   "wyrd boundary doc regenerated") and produced **no
   tree diff** — `git status` stayed clean, so the
   filed doc was not stale. The module does have a
   regen entry (`__main__` → `main()` →
   `write_wyrd_boundary_doc()`), and its checker's
   doc-text rule pins the filed doc verbatim to the
   render in any case.
6. **The closing doc stands complete.** 
   `docs/architecture/WING_CLOSING.md` stands with
   TF-150's section 9 ("## 9. Provenance preserved
   (TF-150)") appended below TF-144's sections 1–8,
   re-read at this gate; `wing_provenance_problems()`
   [] (proof 1) covers it live through its
   closing-doc validator (TF-144's render standing as
   prefix + the TF-150 appendix, nothing else), and
   TF-150's provenance state re-measured live files
   14 history rows / 7 authorship rows (3 known /
   1 known-none / 3 UNKNOWN, nothing invented),
   3 wing commits in total, the import at 96 files /
   23,701 insertions.
7. **Targeted + full suites re-run at the gate** (see
   "Suite at the gate" below): the Part VIII targeted
   set (all 13 test files, TF-139–TF-151) = **359
   tests, 359 passed**; the full suite stands exactly
   on TF-151's banking line.

The gate's fresh probes found **no defect in the
estate**: nothing was fixed at this gate, and this
slice is gate doc + docket rulings only, exactly as
TF-138's, TF-120's and TF-104's gates were. No
incident occurred at this gate — no EPERM window, no
quota wall, nothing deleted anywhere (see "Suite at
the gate" for the one invocation note, an error in
the probe's own path setup, not an estate fault).

The proof is FIXTURE/STATIC grade: the wing's schemas,
service shells, production landings and synthetic
fixtures re-measured live on this VM. A live model's
behaviour, and any live WYRD consumer's, are **NOT
PROVEN** — no live model ran at this gate or in any
Part VIII slice — and no claim at this gate says
otherwise (TF-173 owns the real-hardware road).

## 2. What the gate inherits, stated as one estate

Part VIII is one judgement carried out in order, not
thirteen separate opinions about a research folder.
TF-006's census had found the wing's shape: a younger
prototype (`research_data/src/wyrdforge/`) whose models
package was never committed, so it could not import —
four inventions (bond graph, micro-RAG, persona
compiler, truth calibrator) standing in limbo beside
the shipping tree. The part first judged each one on
measurement, not taste: TF-139 adopted the bond graph
**as a layer** beside the shipping knowledge store
(9 comparison rows — 5 unique / 1 layer-link /
2 different / 1 overlap — decided on TF-006's census +
a working spike rebuilt from the schemas); TF-140
retired micro-RAG by measurement on TF-043's fixtures
(the shipping engine winning precision / recall / F1
by the margins in proof 2); TF-141 merged the persona
compiler in part (its compiled shape proving better on
the same synthetic ground truth, its two measured
defects filed for TF-143 to NOT import); TF-142
retired the truth calibrator by measurement on
TF-100's chaos corpus (the calibrator passing broken
outputs its own rubric should have caught, Part V's
enforcement scoring the corpus instead). Then the
part executed its own verdicts: TF-143 landed the two
adoptions as production code under the shipping tree's
discipline (the bond layer standing beside the store,
evidence naming store record ids; the compiler merged
beside TF-030, its defects not imported, the derived
indices' weights filed as an explicit equal-weights
decision); TF-144 executed the two retirements as
archived-in-place — banners at the three files' heads,
bodies byte-for-byte below, six lessons harvested into
the wing's closing doc. Around those executions the
part drew the lines the wing's existence had left
unfiled: TF-145's WYRD boundary (what ThoughtForge
owns — cognition — vs what WYRD owns — the canonical
world state — with the interface between them, for the
Second Life/Hermes consumers); TF-146's bond bridge
(the adopted layer exported as WYRD-consumable data,
built from the production graph, no bond logic
duplicated); TF-147's Rúnmál cards (the merged compiled
personae exportable in Rúnmál's card shape, fields with
no compiled source filed empty, never fabricated).
Then the part closed the wing itself: TF-148 fenced
`research_data/` — every tracked area and every
code/test file filing exactly one disposition, the
fence notice a reader landing in the tree actually
sees, the wing's own test discipline stated; TF-149
filed the wing's tests (adopted code carrying 93
shipping-grade tests; retired code's tests archived
with it; the one superseded wing test's intent named
in its shipping counterparts; the one unadopted
subject's test fenced with it, no counterpart claimed);
TF-150 preserved the wing's provenance in the closing
doc (one import commit, three wing commits in total,
authorship filed known / known-none / UNKNOWN with
nothing invented); and TF-151 re-ran the panels the
verdicts might have moved and published the deltas —
19 metrics, none moved, both verdicts PASS, the
zeroes proven honest by source inspection rather than
assumed.

Two filed shapes the gate states plainly rather than
smoothing over: the wing itself was never repaired
into importability and will not be under this roadmap
— `wyrdforge.models` stays missing by design, and
D-WING-1 closed *by verdict*, not by repair; and
every number in the part is fixture/static grade,
measured over schemas, shells and synthetic fixtures
through the real production modules — the adopted
code has never faced a live model or a live WYRD
consumer, and the part's own filings (TF-139 onward)
never claimed it had.

## 3. Gate rulings on the entries and gaps assigned to this gate

1. **D-WING-1 — the only Part VIII entry in the TF-016
   docket — stands CLOSED-BY-VERDICT by TF-149, and
   this gate verifies that closure live rather than
   re-deciding it.** The entry's whole assignment is
   discharged: verdicts TF-139–TF-142 decided (proof
   2), adoptions TF-143 landed, retirements TF-144
   executed, hygiene TF-148 fenced, tests TF-149 filed
   — its closing paragraph names exactly that executed
   disposition (adopted concepts rebuilt as shipping
   production code carrying 93 shipping-grade tests;
   retired mechanisms and their shared test file
   archived-in-place; the remaining wing test file
   fenced with its unadopted subject). Every checker
   that pins a share of that closure was re-run live
   at this gate and returned [] (proof 1), and the
   docket file's own totals were re-read at this gate
   in their filed header shape. **This gate adds no
   docket entry and closes none**: TF-150 owned
   provenance and TF-152 owns this gate, neither a
   D-WING-1 share, exactly as TF-149's closure filed.
2. **Docket (TF-016) at the gate: 137 entries —
   32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED /
   4 CLOSED-BY-VERDICT** — unchanged by this gate
   (the four CLOSED-BY-VERDICT entries stand as filed,
   D-WING-1 among them beside D-CARD-2, D-PER-2 and
   D-REC-1 in the header's parenthetical as the docket
   files them). The docket file itself is not edited
   by this gate.
3. **The gate's three claims, ruled on the fresh
   evidence in the verdict above:** (a) no research
   code left in limbo — **proven** (proof 3: 11/11
   code/test files filing exactly one disposition,
   12 baseline areas, the fence file present); (b)
   every verdict executed — **proven** (proof 4:
   both ADOPT-side verdicts landed as production code
   with their consumers built, both RETIRE verdicts
   bannered in place with the adopted files
   banner-free, all pinned by live checkers); (c) the
   boundary doc shipped — **proven** (proof 5: the
   doc exists, its checker is clean, and regeneration
   by its own command produced no diff).
4. **Live-model and live-consumer claims generally —
   NOT PROVEN, owned by Part X as already filed.**
   Whether a live system honours a bond excerpt, a
   compiled persona, a bond-bridge export or a Rúnmál
   card is the TF-173 class of question (real-hardware
   lab I), with TF-156 owning the live Rúnmál binding
   in Part IX and TF-175 / TF-186 their filed legs.
   Nothing at this gate re-opens, re-assigns or
   softens those filings.
5. **Part IX inherits the bindings road exactly as
   filed.** TF-145's boundary and TF-146's bridge are
   the Part VIII artefacts Part IX's consumers bind
   against (TF-156's Rúnmál binding consumes TF-147's
   card shape and TF-165's persona portability the
   compiler's output, each with its filed proof grade);
   nothing in Part VIII pre-decides any Part IX
   binding's live status, and TF-153's service shape
   is the road's next slice, not this gate's.

## 4. Plain-language summary (the Volmarr-facing half of this gate)

Here is what Part VIII did, in plain words:

- **The research wing in the back of the house got its
  day in court — and a verdict for every invention in
  it.** The bond graph (how people stand with each
  other: vows, hurts, repair) was worth keeping, so it
  was rebuilt properly as real shipping code standing
  beside the memory store, and a bridge now exports it
  in the shape the world-model neighbour (WYRD) can
  consume. The persona compiler's best ideas were
  merged into the shipping persona path — with its two
  measured flaws deliberately left behind — and its
  compiled people can now be exported as Rúnmál cards.
- **Two inventions lost on measurement, and were
  retired honourably, not deleted.** The wing's small
  retrieval engine measured clearly worse than the
  shipping one, and its truth checker passed broken
  answers it should have caught — so both were
  archived in place: a clear retirement banner at the
  top of each file, the original code untouched below
  it, and their good ideas (typed packets, an explicit
  token budget, the four-part truth rubric) written
  down as lessons in the wing's closing document.
- **Nothing is left in limbo.** Every file in the
  research folder now files exactly one fate —
  graduated, archived-retired, or clearly fenced as
  research with a fence notice a reader actually sees;
  its tests are accounted for the same way, and its
  history is recorded honestly (one import commit;
  what we don't know about its earlier authorship is
  filed as unknown, not guessed).
- **The boundary with WYRD is written down.** What
  ThoughtForge owns (the thinking) and what WYRD owns
  (the canonical state of the world) is filed in one
  shipped document — and at this gate that document
  was regenerated by its own command and matched the
  filed copy exactly.
- **At this gate everything was measured again,
  fresh:** all fourteen Part VIII checkers clean, every
  probe surface passing, the panels re-run standing
  exactly where TF-151 filed them (19 numbers, none
  moved — proven honest, because those panels never
  read the adopted code), and the full test suite
  standing exactly where the last slice left it.
- **Nothing here has faced a live model or a live
  WYRD consumer.** Every number is fixture grade —
  real code, synthetic fixtures, on this machine. How
  the adopted pieces behave live is still **NOT
  PROVEN** — that road is Part X's (TF-173), and Part
  IX next builds the bindings those live tests will
  drink through.

## 5. The evidence, slice by slice

| Slice | Artefact | Headline |
|---|---|---|
| TF-139 | `BOND_GRAPH_VERDICT.md` | Bond graph verdict — **ADOPT_AS_LAYER**, decided on TF-006's census + a working spike (9 comparison rows: 5 unique / 1 layer-link / 2 different / 1 overlap); re-measured at this gate: checker [], probes 7/7, spike repair_debt 0.35 / safety 0.725, indices 0.4375 / 0.5333 / 0.2083, excerpt 9 lines |
| TF-140 | `MICRO_RAG_VERDICT.md` | Micro-RAG verdict — **RETIRE** as a retrieval mechanism, by measurement on TF-043's fixtures (shipping 0.8889 / 0.9630 / 0.9111 vs spike 0.3333 / 0.8889 / 0.4749, MRR tied 1.0, as filed in its verdict data); re-measured at this gate: checker [], probes 7/7 |
| TF-141 | `PERSONA_COMPILER_VERDICT.md` | Persona compiler verdict — **MERGE** (adopt-in-part) on the same synthetic ground truth through both paths, its two defects filed for TF-143 to NOT import; re-measured at this gate: checker [], probes 7/7 |
| TF-142 | `TRUTH_CALIBRATOR_VERDICT.md` | Truth calibrator verdict — **RETIRE** as a calibration / enforcement mechanism, by measurement on TF-100's chaos corpus (38 fixtures, as filed); re-measured at this gate: checker [], probes 7/7 |
| TF-143 | `BOND_GRAPH.md` + `PERSONA_COMPILER.md` | Adoptions implemented — both ADOPT verdicts landed as production code (schema-rebuilt bond layer with the equal-weights decision filed; compiler merged beside TF-030, defects not imported); re-measured at this gate: both checkers [], both probes 7/7, bond parity exact against TF-139's spike |
| TF-144 | `WING_CLOSING.md` | Retirements executed — both RETIRE verdicts archived-in-place (banners on the two services + their shared test file, bodies byte-for-byte below, 6 lessons harvested); re-measured at this gate: checker [], probes 7/7, archive state 3 present / 3 bannered / 0 adopted bannered |
| TF-145 | `WYRD_BOUNDARY.md` | WYRD boundary doc — ThoughtForge vs the WYRD Protocol: 12 ownership rows (6 / 3 / 3), 5 interface rows (2 production / 2 fixture / 1 not built), 2 consumers; re-measured at this gate: checker [], probes 7/7, regenerated by its command with no diff |
| TF-146 | `BOND_BRIDGE.md` | The bond-graph bridge — bonds as WYRD-consumable data, built FROM TF-143's production graph (9 exported sections / 4 reservations, evidence as store record ids only); re-measured at this gate: checker [], probes 7/7 |
| TF-147 | `RUNMAL_CARDS.md` | Persona compiler → Rúnmál cards — compiled personae exportable as `chara_card_v2` cards (14 fields filed, no-source fields filed empty, never fabricated); re-measured at this gate: checker [], probes 8/8, fixture card from 14 source records |
| TF-148 | `RESEARCH_HYGIENE.md` + `research_data/RESEARCH_FENCE.md` | Research hygiene — `research_data/` clearly fenced, no ambiguous middle state (11 code/test files each filing exactly one disposition: 2 / 3 / 3 / 2 / 1; 12 baseline areas, 96 + fence = 97 tracked); re-measured at this gate: checker [], probes 7/7, live state exactly as filed |
| TF-149 | `WING_TESTS.md` | The wing's tests — adopted code carries shipping-grade tests (93 adopted shipping tests), retired code's tests archived with it (4 wing functions: 2 archived / 1 superseded / 1 fenced); D-WING-1 CLOSED-BY-VERDICT; re-measured at this gate: checker [], probes 7/7, state exactly as filed |
| TF-150 | `WING_PROVENANCE.md` (+ `WING_CLOSING.md` §9) | Provenance preserved — the wing's authorship/history recorded in the closing doc (14 history rows / 7 authorship rows: 3 known / 1 known-none / 3 UNKNOWN; 3 wing commits; import 96 files / 23,701 insertions); re-measured at this gate: checker [], probes 7/7 |
| TF-151 | `PANEL_RERUN.md` | Panel re-run — TF-132 + persona suites re-measured with the verdicts landed, deltas published (19 metrics / 19 unmoved / 0 moved, both verdicts PASS, 7 verdict rows consumed by no re-run panel); re-measured at this gate: checker [], probes 7/7, state exactly as filed |

## Road re-cut assessment

Part VIII changed no Part IX–X slice's wording or
order. Assignments recorded here per TF-018's rule,
all exactly as already filed: D-WING-1 stands
CLOSED-BY-VERDICT by TF-149 with no Part IX–X share;
live-model / live-consumer proof stands with
**TF-173** (real-hardware lab I), TF-175 (the laptop
line) and TF-186 (the showcase), and the live Rúnmál
binding with **TF-156** in Part IX; persona
portability across bindings is **TF-165**'s, consuming
TF-147's card shape as filed. Part IX inherits
TF-145's boundary and TF-146's bridge exactly as filed
— cognition on ThoughtForge's side, canonical world
state on WYRD's — starting with TF-153's service
shape; nothing in Part VIII pre-decides any Part IX
verdict.

## Suite at the gate (measured this slice, TF-014 method)

`PYTHONPATH=src python3 -m pytest tests/ -q` on this
VM, read from the on-disk gate log
(`~/workspace/.verify-tmp/tf152_gate.log`):
**3932 collected / 3930 passed / 1 skipped /
1 failed (D-ENV-1, unchanged — the phase8 dependency
check naming sentence-transformers absent on this
VM, an environment entry not a code defect)** —
exactly TF-151's banking line. The gate's targeted
re-run (all 13 Part VIII test files, TF-139–TF-151)
is **359 collected / 359 passed** (log
`~/workspace/.verify-tmp/tf152_targeted.log`), and
the gate proofs above were measured fresh at this
gate: all seventeen `*_problems()` checkers (the
fourteen Part VIII checkers plus TF-151's three
neighbours) and all fourteen Part VIII `run_*()`
probe surfaces plus their `run_*_state()` headlines
through direct probes (logs
`~/workspace/.verify-tmp/tf152_checkers.log` /
`tf152_runs.log`), TF-148's live hygiene state,
TF-144's live archive state, TF-145's boundary
regenerated by its command with no diff (log
`~/workspace/.verify-tmp/tf152_wyrd_regen.log`), and
`WING_CLOSING.md` §9 re-read in place. This slice
adds one gate doc plus the TF-016 docket rulings
filed in §3 and no code or tests, so the count does
not move at banking.

**No incident occurred at this gate** — no EPERM
write-block window, no disk-quota wall, nothing
deleted anywhere; all runs completed under the
established workspace TMPDIR
(`~/workspace/.verify-tmp/mindspark-tmp`). One
invocation note, filed for completeness in the
estate's known shape: this worker's first checker
probe was launched from a script under `/tmp`, whose
directory — not the repo root — headed `sys.path`,
so the `benchmarks/` imports inside two checkers
failed with `No module named 'benchmarks'`; the
probe was re-run with the repo root and `src` on the
path (the `python3 -m` invocation shape the estate's
own commands use), and every number filed at this
gate was measured on that corrected run — an
invocation error, not an estate defect, exactly as
TF-138's ruling 4 filed for its own gate. Walls under
this VM's contention (the full gate's among them) are
not baselines; the counts are exact.

## What opens next

Part IX — Bindings: the Ecosystem Drinks from the
Forge (TF-153–TF-170), starting with TF-153: service
shape — ThoughtForge callable as a library AND as a
local service (loopback only by default), one
contract.
