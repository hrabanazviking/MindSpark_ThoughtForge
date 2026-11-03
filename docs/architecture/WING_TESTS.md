# The Wing's Tests — Adopted Code Ships Tested, Retired Code's Tests Archived (TF-149)

GENERATED from the filed data in `src/thoughtforge/wing_tests.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.wing_tests`. The checker (`wing_tests_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-wing-tests/1`
Wing test functions filed: 4 in 2 files (archived with retired code 2 / superseded by shipping tests 1 / fenced with unadopted subject 1)
Adopted shipping tests: 93 across 4 adopted production modules
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built: the wing carries exactly 2 test files with exactly 4 test functions (counted by AST). `test_micro_rag_and_truth.py` was already archived-in-place by TF-144 — it opens with the retirement banner; `test_memory_and_persona.py` carries no banner. TF-148 files the first archived_retired and the second fenced_research; this slice contradicts neither filing — it adds and removes no banner, moves and deletes no wing file, and does not resurrect `wyrdforge.models` (still absent by design, so the wing tests still cannot run). The adopted production estate carries 93 shipping tests in the root `tests/` tree (TF-143's file 41, TF-146's 26, TF-147's 26, counted by AST), and the root pyproject's testpaths = ["tests"] means the main suite never collects the wing tests. All six neighbouring checkers stood clean live.

## 2. Every wing test function, filed

| Wing test | Subject | Subject verdict | TF-149 disposition | Counterpart / evidence |
|---|---|---|---|---|
| `test_micro_rag_assembles_packet` (`research_data/tests/test_micro_rag_and_truth.py`) | wyrdforge micro-RAG pipeline (assemble()) | RETIRE by TF-140 (as a retrieval mechanism) | archived_with_retired_code | verdict evidence: `tests/test_tf140_micro_rag_verdict.py` (29 tests) |
| `test_truth_calibrator_flags_unsupported_exclusivity` (`research_data/tests/test_micro_rag_and_truth.py`) | wyrdforge truth calibrator (evaluate()) | RETIRE by TF-142 (as a calibration / enforcement mechanism) | archived_with_retired_code | verdict evidence: `tests/test_tf142_truth_calibrator_verdict.py` (33 tests) |
| `test_persona_compiler_builds_packet` (`research_data/tests/test_memory_and_persona.py`) | wyrdforge persona compiler (compile()) | MERGE by TF-141 (adopt-in-part), landed in TF-143 as production code | superseded_by_shipping_tests | shipping tests: `test_main_counts_move_exactly_as_filed`, `test_inject_persona_delivers_base_and_compiled` |
| `test_memory_store_search_and_promote` (`research_data/tests/test_memory_and_persona.py`) | wing memory store (InMemoryRecordStore) + runtime demo seed | never adopted — no verdict slice adopted it; fenced research per TF-148 | fenced_with_unadopted_subject | none — no counterpart claimed |

- `test_micro_rag_assembles_packet`: no shipping counterpart is claimed: the subject mechanism was retired, not adopted; its proof is the verdict evidence named here.
- `test_truth_calibrator_flags_unsupported_exclusivity`: no shipping counterpart is claimed: the subject mechanism was retired, not adopted; its proof is the verdict evidence named here.
- `test_persona_compiler_builds_packet`: the wing test's intent — a compile builds a packet carrying identity + guidance — is carried in production shape by TF-143's shipping tests named here (the compiled counts pinned exactly, and inject_persona delivering base + compiled facets); the wing original stays fenced in place, unbannered, exactly as TF-148 filed it.
- `test_memory_store_search_and_promote`: NO shipping counterpart is claimed: the shipping memory estate is a different shape and no equivalence is filed; the test stays in the wing suite under TF-148's fence with its unadopted subject.

## 3. The adopted estate's shipping tests

| Production module | Adopting slice | Shipping test file | Tests | Grade evidence |
|---|---|---|---|---|
| `src/thoughtforge/knowledge/bond_graph.py` | TF-143 | `tests/test_tf143_adoptions.py` | 13 (TestBondGraphLanding 13) | `bond_graph_problems()` clean live |
| `src/thoughtforge/cognition/persona_compiler.py` | TF-143 | `tests/test_tf143_adoptions.py` | 28 (TestPersonaCompilerLanding 17 + TestMergedDelivery 5 + TestCrossCutting 6) | `persona_compiler_problems()` clean live |
| `src/thoughtforge/knowledge/bond_bridge.py` | TF-146 | `tests/test_tf146_bond_bridge.py` | 26 (TestBridgeFiled 4 + TestExportShape 6 + TestRoundTrip 3 + TestReservations 3 + TestBoundaryAndDocket 5 + TestProbesAndDoc 5) | `bond_bridge_problems()` clean live |
| `src/thoughtforge/cognition/runmal_cards.py` | TF-147 | `tests/test_tf147_runmal_cards.py` | 26 (TestCardsFiled 4 + TestMapping 7 + TestFiledEmptyAndExtension 4 + TestRoundTrip 2 + TestCouplingAndDocket 4 + TestProbesAndDoc 5) | `runmal_cards_problems()` clean live |

Counting rules, filed in the rows themselves: `bond_graph` — this row files exactly its own landing class in TF-143's shared adoption file; `persona_compiler` — 17 landing + 5 merged-delivery + 6 cross-cutting; the cross-cutting class judges both TF-143 landings together and is counted exactly once, under this row, so the two TF-143 rows sum to the file's 41 with no double-counting; `bond_bridge` — the whole file judges this module only; `runmal_cards` — the whole file judges this module only. The rows sum to 93, recomputed live by AST in the checker (the TF-143 file's 41 tests are attributed across its two rows with no class counted twice).

## 4. Shipping-grade criteria

- `lives_under_tests_and_runs_in_the_full_gate`: the shipping tests live under the root tests/ tree, which the root pyproject's testpaths = ["tests"] collects, so they run in the campaign's full gate — Verified: the checker re-reads the root testpaths live and files every adopted test file under tests/.
- `module_checker_clean`: the adopted module's own *_problems() checker is clean live — Verified: the checker calls every adopted module's checker (and both Part VIII neighbours) live.
- `probes_pass`: the adopted module's filed probes all pass — Verified: each adopted module's checker pins its own probes, so a clean checker carries this leg.
- `doc_render_pinned_verbatim`: the adopted module's architecture doc is filed verbatim as its render — Verified: each adopted module's checker pins its own doc against its render, so a clean checker carries this leg.
- `deterministic`: the shipping tests are deterministic — the same tree measures the same counts and verdicts on a re-run — Verified: the counts in this filing are AST counts recomputed live by the checker; the adopted modules' own determinism tests stand in their files.
- `synthetic_fixtures_only`: the shipping tests run on synthetic fixtures only (TF-017) — no canon / character data — Verified: filed as the estate's standing rule; the adopted modules' own fixture tests pin it in their files.

## 5. Docket

TF-149 closes D-WING-1 in the open in `docs/audit/TF-016_BUG_AND_GAP_DOCKET.md` as **CLOSED-BY-VERDICT** — the wing-tests share was the last share every prior narrowing named. The verdict: the wing was never repaired into importability and will not be under this roadmap; its every part received an executed disposition (adopted concepts rebuilt as shipping production code carrying 93 shipping-grade tests; retired mechanisms and their shared test file archived-in-place by TF-144; the remaining wing test file fenced with its unadopted subject per TF-148, its persona test's intent superseded by TF-143's shipping tests). TF-016 totals move in the open: 137 entries — 31 OPEN / 1 OPEN-ENVIRONMENT / 101 FIXED / 4 CLOSED-BY-VERDICT. TF-150 / TF-152 own provenance and the Part VIII gate, not this entry.

## 6. The probes

- `clean_estate_passes`: the filed wing rows + adopted rows + grade rows pass their validators
- `retired_test_not_archived_is_named`: a retired subject's test filed under a non-archived disposition is named
- `superseded_test_without_a_counterpart_is_named`: a superseded row with its shipping counterpart names emptied is named
- `adopted_row_test_count_wrong_is_named`: an adopted row whose filed count disagrees with its class counts is named
- `unknown_disposition_is_named`: a wing row filing a disposition outside the filed vocabulary is named
- `unfiled_wing_function_is_named`: a wing test function present in the tree but unfiled (or filed but missing) is named
- `doc_drift_is_named`: a doc text that drifts from the filed render is named

## 7. Filed limits

- Fixture/static grade only: this slice files the wing tests' dispositions and re-measures the adopted estate's shipping tests by AST; the wing tests themselves still cannot run — the wing still cannot import by design, exactly as TF-148 filed it.
- No wing file was moved, deleted or bannered by this slice, and no README.md anywhere was touched; TF-150 owns the wing's provenance and TF-152 the Part VIII gate.
- The fenced memory-store test claims NO shipping counterpart: the shipping memory estate is a different shape, and no equivalence is filed.
- No live model was read; live behaviour is NOT PROVEN (TF-173 class), never claimed.
