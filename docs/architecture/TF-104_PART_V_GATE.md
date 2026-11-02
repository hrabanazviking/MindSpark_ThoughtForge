# TF-104 — Part V Gate: Refinement — Salvage & Enforcement (Part V Gate)

Closed 2026-11-01 on branch `yrsa/thoughtforge-road-build`.
Roadmap wording: "Part V gate: chaos-corpus score
published; zero fabrication rescues in the adversarial
set; summary filed."

## The gate verdict

**Part V is COMPLETE. All 17 build slices (TF-087–TF-103)
are banked with evidence in `docs/architecture/`, and the
gate proofs were re-measured fresh at this gate — not
quoted from their slices:**

1. **Chaos-corpus score published — GREEN as filed.**
   TF-101's scored run over TF-100's corpus was recomputed
   live at this gate (`run_chaos()`; `chaos_run_problems()`
   and `chaos_run_live_problems()` both return `[]`, and
   the live score equals the filed score exactly):
   **29 of 38 entries land as filed (76.3%)** — rescued
   **12/12**, refused **12/21**, passed-clean **5/5**;
   stack verdicts rescued 12 / refused 4 / flagged 8 /
   clean 14; every clean control clean byte-identical
   (5/5, zero over-rescue). The 9 mismatches are all one
   shape (expected refused, stack clean) in exactly the
   3 named gaps TF-101 filed — draft-side hallucination
   4 (owner TF-088), persona voice at the policy surface
   3 (owner TF-096), genericness beyond hard phrases 2
   (owner TF-093) — ruled on in §3 below, carried forward
   in the open, never tuned away. The score is FIXTURE
   grade: the filed corpus run live through the real
   TF-093 policy surface on this VM; a live model's
   score is NOT PROVEN.
2. **Zero fabrication rescues in the adversarial set —
   GREEN at the marker tier, with its filed caveat.**
   Re-measured fresh at this gate, on all three legs
   that carry the claim:
   - TF-088/TF-089's adversarial corpus through the real
     `FragmentSalvage`: **6 of 6 marker fabrications
     rejected** (invented price, invented citation,
     negation flip, invented destinations, displaced
     time, dropped negation); marker adoption rate
     **0.0**, prevention **1.0** — TF-089's two absolute
     thresholds, inherited by this gate, both hold
     (`salvage_safety_live_problems()` and
     `salvage_metrics_live_problems()` both `[]`).
   - TF-101's chaos run: of the 7 hallucination +
     fabricated_citation entries, **fabrication rescues
     0** — no invented entry earns a stack `rescued`
     verdict.
   - TF-103's regression armour re-runs both estates'
     own live checkers today and is green
     (`regression_armour_live_problems() == []`), so
     the claim is re-proven, not remembered.
   The caveat stands exactly as TF-089 filed it and
   this gate does not soften it: **broad corruption is
   not zero.** Of the 2 adversarial refine outputs the
   corpus adopts, 1 is the common-word elaboration
   residual — facts added in plain nouns, carrying no
   marker any guard can name — so adopted-output broad
   corruption stands at **1/2**, and in the chaos run
   the 4 hallucination drafts pass *clean* (a pass,
   scored as mismatches, never dressed as a rescue or
   as enforcement success). "Zero fabrication rescues"
   is true of *rescues and marker adoptions*; it is
   not a claim that no invented text ever passes.
3. **Summary filed** — the plain-language summary in
   §4 below is this gate's Volmarr-facing half, and the
   evidence table in §5 files every slice's artefact
   and headline.

The targeted gate test set was also re-run as one set
at this gate: every Part V test file (TF-087–TF-103,
17 files) = **428 tests, 428 passed**.

## 2. What the gate inherits, stated as one estate

Part V's mechanisms are caller-invoked gates and guards
in `refinement/`, composed by exactly one policy surface
(TF-093's `enforce_policy`, seven checks in execution
order) with exactly one per-conversation disposition
layer (TF-094: strict / standard / relaxed, default
standard). Salvage itself gained three adoption guards
(TF-088 fabrication, TF-096 persona voice, TF-098's
per-profile cost cap) and one record (TF-097's rescue
provenance, which closes TF-084's D-TRACE-1 in the
joined trace). The telling layers are TF-095 (a block
is told, with reasons and the salvage path offered,
never run) and TF-102 (marks for the answers that
leave, as consumer data). Streaming salvage is placed
honestly per strategy (TF-099): exactly two mechanisms
work mid-stream, the rest are stream-end or post-hoc,
and no surface claims more. Nothing in the part routes
the cognition core through the policy surface — that
limit, filed by TF-093/TF-094/TF-095/TF-102 alike,
stands: `_compose_final` still calls the legacy
`EnforcementGate` directly, and consumer wiring is
Part IX's (TF-153–TF-170), not this gate's.

## 3. Gate rulings on the entries and gaps assigned to this gate

1. **D-SALV-1 (the roadmap's named rescue strategies
   did not exist in salvage.py) — ruled in four of its
   five shares; entry stays OPEN on the fifth.**
   Truncation + format (TF-090), loops (TF-091) and
   drift (TF-092) are FIXED at gate grade as caller-
   invoked gates (the TF-090 shape: salvage has no
   signal those rescues need, so they live beside it,
   and TF-087's catalogue + TF-089's metrics stand
   unmoved — exactly as those slices filed). The
   measurement share (TF-089) is discharged: rescue
   2/7, rambling 1/2 by selection, thresholds filed
   from the measurements. The one remaining share is
   **rambling compression in general**: a lone
   rambling draft is compressed **0/1** — no general
   mechanism exists, TF-091 cuts only its loop-shaped
   subset, and none is claimed. The entry stays OPEN
   with that single reason; **TF-185** (release truth)
   must carry lone-rambling compression as NOT PROVEN,
   and **TF-192** verifies at the end that the entry
   is left open with this reason and no other.
2. **D-SALV-2's residual (common-word elaboration) —
   ruled: the marker tier is closed, the residual is
   not.** TF-088's fix stands FIXED; this gate's fresh
   re-measurement (proof 2) holds its absolutes
   (marker adoption 0.0, prevention 1.0). The residual
   class — invention in common words with no marker —
   is the same phenomenon the chaos run scores as the
   draft-side hallucination gap (ruling 4a): it is a
   judgement no current surface makes. It is carried
   forward, not closed: **TF-142** (the truth
   calibrator verdict, measured on TF-100's chaos
   corpus) is its next measurement, and **TF-185**
   must carry broad adopted corruption at its measured
   1/2 until a measurement moves it.
3. **D-SALV-3's residual (flattening that trips no
   avoid entry) — ruled likewise.** TF-096's fix
   stands FIXED (selection filtered to persona-
   consistent drafts when one is at hand; refine
   adoption gated beside TF-088's). The residual is
   carried where TF-096 already filed it: **TF-122**
   (the persona consistency suite scores voice
   fidelity beyond the avoid subset), with
   **TF-151**'s panel re-run as its re-measurement.
4. **TF-101's three named gaps (not docket entries —
   scored gaps, inherited by this gate as filed):**
   - **(a) Draft-side hallucination not judged by the
     policy surface (4 entries, owner TF-088).**
     Stands OPEN. No policy check judges a draft's
     invented facts beyond TF-049's citation matching,
     and TF-088's guard is refine-adoption, not a
     draft gate. Next measurement: **TF-142**, whose
     roadmap wording already measures a calibration
     candidate against Part V's enforcement on this
     corpus. Until a draft-side mechanism lands and
     is measured, no consumer may cite Part V as
     hallucination-proof.
   - **(b) Persona voice not a policy-surface check
     (3 entries, owner TF-096).** Stands OPEN.
     `enforce_policy` takes no persona signal; the
     diagnostic (TF-096's own `check_voice`) names a
     new avoid violation in all 3 entries the stack
     passes clean. Next measurements: **TF-122** and
     **TF-151**, as in ruling 3.
   - **(c) Genericness checks hard phrases only
     (2 entries, owner TF-093).** Stands OPEN. Fluent
     filler and a restated question trip no hard
     phrase and pass the five-word floor. No later
     roadmap slice names this mechanism, so the gap
     keeps its owner surface (TF-093) and is assigned
     to **TF-185** (the release truth table must carry
     it) and **TF-192** (final audit verifies it is
     left open with its reason).
   TF-103's armour pins all three gaps' exact shape
   (9 mismatches: 4 / 3 / 2), so any future fix — or
   regression — turns that checker red until the
   registry updates in the open.
5. **The invented-reference caveat (TF-101's
   ENTRY_NOTES) — carried verbatim.** The
   `chaos_citation_invented_reference` entry counts
   matched per TF-100's filed scoring rule, but its
   flag rides an incidental term match (`hollow` /
   `lantern` against the lantern-oil chunk); the
   invented almanac is named by no check. It is not
   reference verification, and no claim at this gate
   treats it as such.

**Docket (TF-016) at the gate: 90 entries — 43 OPEN /
1 OPEN-ENVIRONMENT / 43 FIXED / 3 CLOSED-BY-VERDICT**
(unchanged by this gate — no entry closes here; the
Part V entries fixed by their own slices stand, and
D-SALV-1 stays OPEN on its one ruled share).
TF-192 verifies at the end that no entry is left
open without a reason.

## 4. Plain-language summary (the Volmarr-facing half of this gate)

Here is what Part V did, in plain words:

- **Broken answers now get rescued by rule instead of
  luck.** At the start of Part V, the "salvage" code
  mostly picked the least-bad draft and returned even
  truncated, looping, or broken-JSON answers word for
  word. Now each kind of breakage has its own repair:
  truncations are cut back to the last whole sentence
  (never finished by invention), loops are cut to one
  telling, answers wandering into the wrong language
  are pulled back by deletion only, and structured
  answers are repaired only by removing or re-closing
  what is there — never by making up values.
- **The forge is forbidden from inventing.** A repair
  step that adds a new price, a new source, a flipped
  meaning, or a moved time is refused before its
  score is even looked at — measured again at this
  gate: every one of the six planted fabrications was
  refused, and none of the invented answers in the
  chaos test was "rescued" into acceptance. The honest
  edge: invention written in plain, ordinary words,
  with no number or name to catch, can still slip
  past — that is measured (1 of 2 adopted test
  repairs), named, and carried forward openly below,
  not hidden.
- **All the checks now speak with one voice, and a
  blocked answer says so.** Seven separate checks
  became one policy surface with one verdict, a
  strictness a conversation can choose, and a plain
  refusal message that says an answer was blocked,
  why, and what could fix it. Answers that do leave
  can carry honest marks — cited, rescued, flagged —
  as data for whatever app shows them.
- **The chaos test score is published, gaps and all:
  29 of 38.** Every repair demanded was delivered,
  every clean answer passed untouched, and the 9
  misses are exactly three known blind spots — the
  checker cannot yet judge an invented fact stated
  in plain words, cannot judge a character's voice,
  and cannot judge fluent filler that answers
  nothing. Each blind spot has a named owner and a
  named future measurement, and a registry (the
  regression armour) now re-runs every historical
  bad answer's fixture on every test run, so a fix
  that quietly breaks again fails loudly.
- **Nothing here has faced a live model.** Every
  number is fixture grade — real code, planted
  broken answers, on this machine. A live small
  model's rescue and corruption rates are still
  **NOT PROVEN**, and Part VII's panels own that road.

## 5. The evidence, slice by slice

| Slice | Artefact | Headline |
|---|---|---|
| TF-087 | `SALVAGE_MAPPED.md` | Salvage's five real strategies catalogued; the five missing rescues measured (D-SALV-1 filed) |
| TF-088 | `SALVAGE_SAFETY.md` | The repair/fabrication line as data; the guard wired into refine adoption before scoring; D-SALV-2 fixed |
| TF-089 | `SALVAGE_METRICS.md` | Rescue 2/7, marker corruption 0, broad corruption 1/2 — thresholds filed from the measurements (gate proof 2's absolutes) |
| TF-090 | `FORMAT_ENFORCEMENT.md` | Format + truncation gate: repair by removal/re-delimiting only, or refusal by name; D-SALV-1 shares fixed at gate grade |
| TF-091 | `LOOP_RESCUE.md` | Degenerate loops detected exact-normalised and cut by deletion only; the 20x loop rescues to one telling |
| TF-092 | `DRIFT_RESCUE.md` | Off-language sentences cut by deletion only, never translated; majority/whole drift flagged verbatim |
| TF-093 | `ENFORCEMENT_POLICY_SURFACE.md` | One policy surface, seven checks, citation reconciled per-claim with records; regeneration ruled not wired |
| TF-094 | `POLICY_CONFIGURATION.md` | Strict / standard / relaxed disposition per conversation; default standard proven TF-093-exact |
| TF-095 | `REFUSAL_HONESTY.md` | A block is told — that, why (verbatim + plain reason), and the salvage path offered, never run |
| TF-096 | `PERSONA_VOICE_PRESERVATION.md` | Voice guard beside TF-088's: selection filtered, refine adoption gated; D-SALV-3 fixed |
| TF-097 | `RESCUE_PROVENANCE.md` | Every refine pass recorded as it runs; the record rides into the joined trace; D-TRACE-1 fixed |
| TF-098 | `COST_DISCIPLINE.md` | Refine passes capped per profile at the salvage site itself; cost totalled in engine calls; D-COST-1 + D-PROF-4 fixed |
| TF-099 | `STREAMING_SALVAGE.md` | Per-strategy placements filed; exactly two mid-stream mechanisms built, the rest honestly post-hoc |
| TF-100 | `CHAOS_CORPUS.md` | 38 real-shaped failures in 13 families as one versioned corpus — the part's proving ground |
| TF-101 | `CHAOS_RUN.md` | The full stack scored over the corpus: 29/38, zero fabrication rescues, 9 mismatches in 3 named gaps (gate proof 1) |
| TF-102 | `QUALITY_MARKS.md` | Cited / rescued / flagged / warning marks as consumer data; a passed check alone earns no cited mark |
| TF-103 | `REGRESSION_ARMOUR.md` | 25 historical bad outputs registered and re-proven (20 covered, 3 gap_open pinned, 2 not_proven) |

## Road re-cut assessment

Part V changed no Part VI–X slice's wording or
order. Assignments recorded here per TF-018's rule:
D-SALV-1's remaining share (general rambling
compression) carries to **TF-185** (release truth)
with **TF-192** verifying; D-SALV-2's common-word
residual and TF-101 gap (a) carry to **TF-142** for
their next measurement; D-SALV-3's residual and
TF-101 gap (b) carry to **TF-122** + **TF-151** as
already filed; TF-101 gap (c) keeps owner TF-093
with **TF-185**/**TF-192** carrying it. Part VI
inherits a refinement estate whose policy surface,
verdicts, and filed limits are frozen as data.

## Suite at the gate (measured this slice, TF-014 method)

`PYTHONPATH=src python3 -m pytest tests/ -q` on this
VM: **2798 collected / 2796 passed / 1 skipped /
1 failed (D-ENV-1, unchanged — the phase8 dependency
check naming sentence-transformers absent on this
VM, an environment entry not a code defect)** —
exactly TF-103's banking line. The gate's targeted
re-run (all 17 Part V test files, TF-087–TF-103) is
428 / 428 passed, and the gate proofs above were
re-measured by direct runs of `run_chaos()`,
`chaos_run_live_problems()`, `salvage_safety_live_problems()`,
`salvage_metrics_live_problems()` and
`regression_armour_live_problems()` at this gate,
all returning `[]` / the filed score. This slice
adds one gate doc plus the TF-016 docket rulings
and no code or tests, so the count does not move
at banking.

## What opens next

Part VI — The Ingestion Forge (ETL) (TF-105–TF-120),
starting with TF-105: ETL mapped — sources → schema
→ embeddings → subset → store, drawn from code; the
pipeline's contracts per stage.
