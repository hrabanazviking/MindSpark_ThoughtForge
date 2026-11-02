# Panel Re-run — TF-132 + Persona Suites Re-measured With the Verdicts Landed (TF-151)

GENERATED from the filed data in `src/thoughtforge/panel_rerun.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.panel_rerun`. The checker (`panel_rerun_problems()`) re-measures every value live through the harness and fails on any drift.

Format: `thoughtforge-panel-rerun/1`
Panels re-run: 3; metrics re-measured: 19 (unmoved 19 / moved 0); verdicts in force: 7 (consumed by a re-run panel: 0)
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built, from the runners only: TF-132's WYRD panel PASSes at 18 cases / 9 consistent / 8 contradicted / 1 honest-unanswerable, overall 0.5555555555555556; TF-122's persona suite PASSes at 108 turns, overall 0.8453703703703704 (identity 0.8333333333333334 / voice 0.8694444444444445 / values 0.8333333333333334); the persona-consistency scorer underneath the voice axis measures 0.7166666666666667 over 6 turns. Every value equals its pinned harness baseline exactly. Source inspection proves why: none of the three measurers imports the bond graph, the persona compiler, the bond bridge or the Runmal cards — the Part VIII verdicts landed in other modules, and these panels do not read them. The deltas are therefore honestly zero, published below rather than assumed.

## 2. The verdicts in force at the re-run

| Slice | Verdict | Consumed by a re-run panel? | Why |
|---|---|---|---|
| TF-139 | ADOPT_AS_LAYER | no | no re-run panel measures bonds; the WYRD panel judges world-state claims and the persona suites judge persona turns — neither imports the bond layer |
| TF-140 | RETIRE | no | a retirement adopts no code; no re-run panel ever used the micro-RAG retriever |
| TF-141 | MERGE | no | the persona suites bind TF-030's PersonaExpectation and the phrase scorer directly; neither imports the compiled expectation TF-143 landed |
| TF-142 | RETIRE | no | a retirement adopts no code; no re-run panel ever used the truth calibrator |
| TF-143 | ADOPTIONS_IMPLEMENTED | no | TF-143 landed the two adoptions as production modules beside the panels; the panels' imports did not change, so their measurements cannot move |
| TF-146 | BRIDGE_BUILT | no | the bridge exports bond data for WYRD consumers; no re-run panel consumes bond exports |
| TF-147 | CARDS_BUILT | no | the cards export compiled personae; the persona suites score scripted turns, not card exports |

## 3. The deltas, published

| Panel | Metric | Pinned | Re-measured | Delta | Outcome |
|---|---|---|---|---|---|
| wyrd_panel | total_cases | 18 | 18 | 0 | unmoved |
| wyrd_panel | consistent_cases | 9 | 9 | 0 | unmoved |
| wyrd_panel | contradicted_cases | 8 | 8 | 0 | unmoved |
| wyrd_panel | honest_unanswerable_cases | 1 | 1 | 0 | unmoved |
| wyrd_panel | overall_score | 0.5555555555555556 | 0.5555555555555556 | 0.0 | unmoved |
| wyrd_panel | probes_passed | 7 | 7 | 0 | unmoved |
| wyrd_panel | probes_total | 7 | 7 | 0 | unmoved |
| persona_suite | identity_score | 0.8333333333333334 | 0.8333333333333334 | 0.0 | unmoved |
| persona_suite | voice_score | 0.8694444444444445 | 0.8694444444444445 | 0.0 | unmoved |
| persona_suite | values_score | 0.8333333333333334 | 0.8333333333333334 | 0.0 | unmoved |
| persona_suite | overall_score | 0.8453703703703704 | 0.8453703703703704 | 0.0 | unmoved |
| persona_suite | total_turns | 108 | 108 | 0 | unmoved |
| persona_consistency | consistency_score | 0.7166666666666667 | 0.7166666666666667 | 0.0 | unmoved |
| persona_consistency | total_turns | 6 | 6 | 0 | unmoved |
| persona_consistency | flagged_turns | 2 | 2 | 0 | unmoved |
| persona_consistency | generic_phrase_hits | 7 | 7 | 0 | unmoved |
| persona_consistency | norse_tone_hits | 10 | 10 | 0 | unmoved |
| persona_consistency | citation_turns | 1 | 1 | 0 | unmoved |
| persona_consistency | skald_quality_hits | 5 | 5 | 0 | unmoved |

Headline: 19 metrics re-measured, 19 unmoved, 0 moved. Both panel verdicts stand PASS before and after. The verdicts landed changed no re-run measurement — proven live, not assumed — because no re-run measurer consumes the adopted code.

## 4. Harness coupling

None, filed in the open on the TF-138 gate precedent: a re-measurement slice adds no harness baseline and re-pins none, so the baseline book (18 baselines / 146 pinned metrics), the flake census (18 deterministic) and the public-claims audit do not move, and `baseline_problems()` stays clean over the existing pins this re-run compares against.

## 5. Docket

None. TF-151 has no share of any TF-016 entry and its re-runs found no defect. TF-016 totals stand unchanged: 137 entries — 32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4 CLOSED-BY-VERDICT.

## 6. The probes

- `clean_rerun_passes`: the filed rerun + verdict + delta rows pass their validators
- `moved_delta_filed_unmoved_is_named`: a delta row whose remeasured value moved but which files unmoved is named
- `wrong_delta_arithmetic_is_named`: a delta row whose filed delta is not remeasured - pinned is named
- `panel_claiming_consumption_is_named`: a rerun row claiming it consumes adopted code is named — no re-run panel does
- `verdict_claiming_consumption_is_named`: a verdict row claiming a re-run panel consumes it is named
- `empty_row_is_named`: empty rerun / verdict / delta rows are named, never crashed on
- `doc_drift_is_named`: a doc text that drifts from the filed render is named

## 7. Filed limits

- Fixture/static grade only: every value above is a pinned synthetic-fixture baseline re-measured live on this VM through the harness's own measurers. No live model was read; live persona or world-state behaviour is NOT PROVEN (TF-173 class), never claimed.
- The zero deltas are the finding, not a failure: the three re-run measurers consume no Part VIII adopted code (checked live against their sources), so the verdicts could not move them. Any future panel that binds the bond layer or the compiled persona must be re-run as its own slice and file its own delta.
- No harness baseline was added or re-pinned by this slice (the TF-138 gate precedent for a re-measurement slice): the comparison stands against the existing pins, and the baseline book / flake census / public-claims filings do not move.
- No docket change: TF-151 has no TF-016 share and found no defect; TF-016 totals stand unchanged (137 — 32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4 CLOSED-BY-VERDICT). TF-152 owns the Part VIII gate.
- No canon or persona content is filed: only the estate's synthetic fixture scores, counts and verdict names.
