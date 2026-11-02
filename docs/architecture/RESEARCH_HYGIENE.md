# Research Hygiene — research_data/ Fenced (TF-148)

GENERATED from the filed data in `src/thoughtforge/research_hygiene.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.research_hygiene` (the command also regenerates `research_data/RESEARCH_FENCE.md`). The checker (`research_hygiene_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-research-hygiene/1`
Tracked census: 96 baseline files in 12 areas, plus the fence file = 97 tracked files once banked
Code/test files filed by disposition: 11 (graduated 2 / archived 3 / fenced research 3 / fenced standalone 2 / fence import 1)
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built: `git ls-files research_data` files 96 tracked files in twelve areas. This clone is sparse — config, examples, scripts and wyrd_runtime are tracked but not on disk here (12 files), so this doc counts disk areas for the sparse-present areas and judges the sparse-excluded areas only through a pure validator fed a tracked-file list. The wing still cannot import: the models package does not exist and the package init re-exports ten names from it (that init's import lines stand unaltered — a neighbouring verdict checker reads them live). Two services had graduated in TF-143, two services and their shared test file were archived-retired in TF-144, and five code or test files had never been given any disposition — the ambiguous middle this slice ends.

## 2. Every area, filed

| Area | Files | Disposition | Rule |
|---|---|---|---|
| research_data/*.md (top-level design and spec markdown) | 58 | fenced_reference | design and spec provenance, read as reference; not shipping documentation and not code |
| research_data/src/wyrdforge/schemas/*.json | 14 | fenced_evidence | verdict evidence, kept unmarked per TF-144; the shapes execute nothing and the verdict spikes are checked against them live |
| research_data/src/wyrdforge/services/*.py | 5 | split_by_file | split by FILE_ROWS: each service files its own graduated, archived or fenced disposition |
| research_data/wyrd_runtime/*.md | 5 | fenced_reference | runtime packet provenance, read as reference; tracked but not on disk in this sparse clone, so judged only through the tracked-file validator |
| research_data/config/*.yaml | 4 | fenced_examples | example configuration for the research wing only; tracked but not on disk in this sparse clone, so judged only through the tracked-file validator |
| research_data/examples/*.yaml | 2 | fenced_examples | example data for the research wing only; tracked but not on disk in this sparse clone, so judged only through the tracked-file validator |
| research_data/src/wyrdforge/security/*.py | 2 | split_by_file | split by FILE_ROWS: both guards file fenced_standalone |
| research_data/tests/*.py | 2 | split_by_file | split by FILE_ROWS and judged under the wing's own test discipline; TF-149 owns their final disposition |
| research_data/src/wyrdforge/runtime/*.py | 1 | split_by_file | split by FILE_ROWS: the demo seed files fenced_research |
| research_data/scripts/*.py | 1 | fenced_tooling | generate_json_schemas.py is a research-side generator, not the shipping build; tracked but not on disk in this sparse clone, so judged only through the tracked-file validator |
| research_data/src/wyrdforge/__init__.py | 1 | split_by_file | split by FILE_ROWS: the package init files fence_import, the broken-import fence state |
| research_data/pyproject.toml | 1 | fence_marker | the fence marker itself: the wing is its own package (wyrdforge 0.1.0) with its own pythonpath and testpaths — its own test discipline |
| research_data/RESEARCH_FENCE.md | 1 | fence_marker | this slice's fence notice, filed verbatim as render_research_fence_markdown(); the tracked census is the 96 baseline files plus this file |

## 3. Every code and test file, filed

| File | Disposition | Rule |
|---|---|---|
| `research_data/src/wyrdforge/services/bond_graph_service.py` | graduated_source | graduated by TF-143: the production counterpart in the shipping tree is the code that ships; this wing shell stays as the adopted source and carries no retirement banner (TF-144) |
| `research_data/src/wyrdforge/services/persona_compiler.py` | graduated_source | graduated by TF-143: the production counterpart in the shipping tree is the code that ships; this wing shell stays as the merged source and carries no retirement banner (TF-144) |
| `research_data/src/wyrdforge/services/micro_rag_pipeline.py` | archived_retired | retired by TF-140's verdict and archived-in-place by TF-144 under its filed banner; verdict evidence, not to be imported or revived without a new roadmap word |
| `research_data/src/wyrdforge/services/truth_calibrator.py` | archived_retired | retired by TF-142's verdict and archived-in-place by TF-144 under its filed banner; verdict evidence, not to be imported or revived without a new roadmap word |
| `research_data/src/wyrdforge/services/memory_store.py` | fenced_research | fenced research: never adopted and never retired; it imports the missing models package, cannot run, and is not shipping code |
| `research_data/src/wyrdforge/security/permission_guard.py` | fenced_standalone | fenced research, standalone leg: it imports nothing from the missing models package and is the wing code that can be imported on its own — still research, not adopted into the shipping tree |
| `research_data/src/wyrdforge/security/prompt_injection_guard.py` | fenced_standalone | fenced research, standalone leg: it imports nothing from the missing models package and is the wing code that can be imported on its own — still research, not adopted into the shipping tree |
| `research_data/src/wyrdforge/runtime/demo_seed.py` | fenced_research | fenced research: a demo seed for the wing's own tests; it imports the missing models package, cannot run, and is not shipping code |
| `research_data/src/wyrdforge/__init__.py` | fence_import | the filed broken-import state: the init re-exports ten names from the missing models package, TF-139's checker requires those import lines to remain, and TF-148 does not alter them — the fence files the breakage, it does not hide it |
| `research_data/tests/test_memory_and_persona.py` | fenced_research | fenced research under the wing's own test discipline: it imports the missing models package and errors at collection there; TF-149 owns the wing tests' final disposition, which this slice does not discharge |
| `research_data/tests/test_micro_rag_and_truth.py` | archived_retired | archived-in-place with the retired code by TF-144 under its filed banner; verdict evidence; TF-149 owns the wing tests' final disposition |

Graduated counterparts, checked live: `research_data/src/wyrdforge/services/bond_graph_service.py` -> `src/thoughtforge/knowledge/bond_graph.py`; `research_data/src/wyrdforge/services/persona_compiler.py` -> `src/thoughtforge/cognition/persona_compiler.py`. Archived files carry their TF-144 banner at their head, checked live; graduated files carry none, checked live.

## 4. Its own test discipline

- `main_suite` (the campaign suite at the repo root): the root pyproject files testpaths = ["tests"] only, so the main suite never collects the wing's tests under research_data/tests — State: in force, checked live against both pyprojects.
- `wing_suite` (the wing's own suite inside research_data/): the wing's own pyproject files its own package, pythonpath = ["src"] and testpaths = ["tests"]; its 2 test files carry 4 test functions — State: collection error, filed not hidden: the missing models package makes the wing suite error at collection (2 errors, 0 run, TF-006).
- `graduated_tests` (tests for graduated code): adopted code is judged by shipping tests under the root tests/ tree (TF-143's landings carry theirs), never by the wing's own tests — State: in force; TF-149 owns the wing-test final disposition, explicitly not discharged here.
- `retired_tests` (tests for retired code): retired code's tests are archived-in-place with it under the TF-144 banner, standing as verdict evidence — State: executed by TF-144 for the shared micro/truth wing test file.

## 5. The fence rules

- `no_shipping_imports_the_wing`: no shipping file under src/thoughtforge imports the wing package — judged on real import statements, parsed; prose, docstrings and string data mentioning the wing are not imports
- `no_research_file_is_shipping`: no file under research_data/ is shipping code: graduated ideas ship only as their production counterparts in the shipping tree
- `revival_needs_a_new_roadmap_word`: reviving, importing or building on any fenced or archived file needs a new roadmap word
- `schemas_stay_evidence`: the 14 schemas stay unmarked as verdict evidence; they specify shapes and execute nothing
- `readmes_never_touched`: no README.md anywhere is touched by the fence — the fence notice is its own new file
- `models_package_not_resurrected`: the missing models package is not resurrected by this slice; if it ever appears, this filing must be updated and the checker names that

## 6. Docket

TF-148 discharges the research-hygiene share of D-WING-1 (narrowed in the open in `docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). D-WING-1 itself stays OPEN for TF-149's wing-tests share — TF-148 is not the last Part VIII slice on the entry. No new docket entry is added and TF-016 totals do not move (137 entries — 33 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 3 CLOSED-BY-VERDICT).

## 7. The probes

- `clean_estate_passes`: the filed areas + files + discipline + a complete tracked list pass their validators
- `unclassified_tracked_file_is_named`: a tracked list carrying an extra unclassified path is named
- `wrong_area_count_is_named`: a tracked list missing one schema file files the wrong schemas count and is named
- `unknown_disposition_is_named`: a file row filing a disposition outside the filed vocabulary is named
- `graduated_row_missing_counterpart_is_named`: a graduated file row with its production counterpart emptied is named
- `archived_row_missing_banner_is_named`: an archived file row with its banner marker emptied is named
- `fence_doc_drift_is_named`: a fence text that drifts from the filed render is named

## 8. Filed limits

- Fence/static grade only: this slice files dispositions and fences the tree; it adopts no research code, retires nothing new, and resurrects no package — the wing still cannot import, exactly as filed.
- TF-149 owns the wing tests' final disposition and TF-150 the wing's provenance; neither share is discharged here.
- The sparse-excluded areas are judged through the tracked-file list only, never assumed to exist on disk in this clone.
- No README.md anywhere was touched; no canon data was touched; no live model was read — live behaviour is NOT PROVEN (TF-173 class), never claimed.
