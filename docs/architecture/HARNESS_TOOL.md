# The Harness as a Tool — TF-137

GENERATED from the filed data in `benchmarks/harness_tool.py` — the filed doc IS the render: regenerate with `python3 -m benchmarks.harness_tool`. The checker (`harness_tool_problems()`) binds every panel against the live registries and fails on any drift from the filed rows.

Format: `thoughtforge-harness-tool/1`
Panels filed: 3
Panel `full_panel` members: 18
Panel `gate_panel` members: 9
Panel `audit_panel` members: 9

One command runs the panel a slice needs, and keepers cite its output verbatim. A member runs by re-measuring its pinned baseline through the harness's own measurer and comparing it with its pin — no scorer or runner is copied here.

## Commands

- List the panels: `python3 -m benchmarks.harness_tool --list-panels`
- Run one panel: `python3 -m benchmarks.harness_tool --panel gate_panel`
- Run individual baselines: `python3 -m benchmarks.harness_tool --baseline persona_suite_synthetic_v1` (repeat `--baseline` to select more)
- Regenerate this doc: `python3 -m benchmarks.harness_tool`

Every run prints its citation line first — `harness-tool run: panel=<panel> members=<run> passed=<passed> verdict=<verdict>` — then one member line per baseline and one line per problem. No wall time ever stands in the output, so a cited line reproduces verbatim.

## The panels

### Panel `full_panel`

- Purpose: the whole estate: the panel TF-138's Part VII gate runs, and any slice whose change could move any pinned baseline — every pinned baseline in the harness, this tool's own included; PASS lives on this selection only
- Members: 18
  - `persona_consistency_synthetic_v1`
  - `persona_suite_synthetic_v1`
  - `memory_suite_synthetic_v1`
  - `depth_suite_synthetic_v1`
  - `presence_suite_synthetic_v1`
  - `honesty_panel_synthetic_v1`
  - `quant_panel_synthetic_v1`
  - `profile_panel_synthetic_v1`
  - `regression_gate_synthetic_v1`
  - `spot_checks_synthetic_v1`
  - `data_hygiene_synthetic_v1`
  - `wyrd_panel_synthetic_v1`
  - `cost_panel_synthetic_v1`
  - `baseline_book_synthetic_v1`
  - `flake_discipline_synthetic_v1`
  - `public_claims_synthetic_v1`
  - `harness_tool_synthetic_v1`
  - `profile_benchmark_empty_store_v1`

### Panel `gate_panel`

- Purpose: a slice that touched a gated suite or panel (TF-121..TF-128): the nine baselines TF-129's regression gate gates, run here per member against their pins (the gate itself, with its noise-derived tolerances, stays benchmarks/regression_gate.py)
- Members: 9
  - `persona_consistency_synthetic_v1`
  - `persona_suite_synthetic_v1`
  - `memory_suite_synthetic_v1`
  - `depth_suite_synthetic_v1`
  - `presence_suite_synthetic_v1`
  - `honesty_panel_synthetic_v1`
  - `quant_panel_synthetic_v1`
  - `profile_panel_synthetic_v1`
  - `profile_benchmark_empty_store_v1`

### Panel `audit_panel`

- Purpose: a slice that touched a checker-judged filing — the gate's verdict row, the spot checks, the hygiene inventory, the WYRD panel, the cost panel, the baseline book, the flake discipline, the public-claims audit or this tool: the ungated baselines, each judged by its own module's checker in its own slice and re-measured against its pin here
- Members: 9
  - `regression_gate_synthetic_v1`
  - `spot_checks_synthetic_v1`
  - `data_hygiene_synthetic_v1`
  - `wyrd_panel_synthetic_v1`
  - `cost_panel_synthetic_v1`
  - `baseline_book_synthetic_v1`
  - `flake_discipline_synthetic_v1`
  - `public_claims_synthetic_v1`
  - `harness_tool_synthetic_v1`

## The rules

- `panels_are_literal_data_bound_to_the_real_estate`: PANEL_ROWS files each panel as literal data — its id, its purpose (which slice need it serves) and its member baselines — and every member is a pinned baseline in benchmarks/harness.py whose runner is the real run_* function the BENCHMARKS index binds; no scorer or runner is copied into this module
- `one_command_runs_the_panel_a_slice_needs`: python3 -m benchmarks.harness_tool is the one command: --list-panels lists the panels, --panel <id> runs one panel, and --baseline <id> (repeatable) runs an individual selection; with no selection argument it regenerates the filed doc
- `citation_lines_are_deterministic_and_verbatim`: every run prints deterministic lines a keeper cites verbatim — one citation line (panel, members run, members passed, verdict), one member line per baseline, one line per problem — and no wall time ever stands in the output: timing is excluded-by-rule (TF-121), so a cited line reproduces exactly
- `a_member_passes_by_reproducing_its_pin`: a member runs by re-measuring its baseline through the harness's own measurer and comparing every metric with its pin within its filed tolerance — baseline_problems()'s judgement, per member; a member that cannot re-measure files ran=False with its reason, never a silent skip
- `pass_needs_the_full_panel_all_passing`: the verdict is PASS only on a full panel selection (the full panel's member list, exactly) with every member running and passing; a partial selection — a smaller panel or an individual selection — files REVIEW however it scores
- `empty_and_unknown_selections_are_named_review`: an empty selection, an unknown panel and an unknown baseline are each named and file REVIEW — never a vacuous pass, never a crash
- `probes_prove_the_validator_can_fail`: TOOL_PROBES perturbs copies of filed rows against the pure validators (an unknown member, a duplicate member, an empty panel, a panel missing its purpose, a member reported passing that did not run, a citation line that does not match the run) plus the clean shape — real filings are never perturbed
- `the_harness_tool_baseline_is_not_gated_by_tf129`: TF-129's regression gate gates the nine TF-121..TF-128 pinned baselines exactly as filed; this slice's own baseline (harness_tool_synthetic_v1) is judged by this module's checker, not gated by the gate — the shape TF-129..TF-136 filed for their own baselines, with the exclusion filed in the open in regression_gate.py
- `no_new_benchmark_input_was_added`: the tool runs only already-filed baselines through their already-filed measurers — it adds no benchmark input, so it files no new harness fixture and no hygiene row, and TF-131's inventory and its pinned baseline do not move (filed in the open, TF-133's coupling shape)

## Filed limits

- The tool runs the harness's pinned FIXTURE baselines only — scripted synthetic suites through the real scorers on this VM. No live model is run; live behaviour is NOT PROVEN (TF-173).
- The audit panel's members are each also judged by their own module's checker in their own slices; this tool re-measures them against their pins and does not re-run those checkers.
- Run output files no timing: walls under contention are not baselines (TF-121).

Proof grade: FIXTURE — pinned synthetic-fixture baselines re-measured live on this VM; no live model, never claimed.
