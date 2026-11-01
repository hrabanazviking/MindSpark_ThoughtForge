# Public-Claims Audit — TF-136

GENERATED from the filed data in `benchmarks/public_claims.py` — the filed doc IS the render: regenerate with `python3 -m benchmarks.public_claims`. The checker (`public_claims_problems()`) recomputes every matched value live through the harness measurers and fails on any drift from the filed rows.

Format: `thoughtforge-public-claims/1`
Claims censused: 21
Matched claims: 12
Struck claims: 9
Distinct source docs: 12

Every performance claim in the repo's docs that purports to state a suite / harness result or a public performance target is censused here: matched to a suite result (baseline + metric + the value, recomputed live) or struck with its reason filed. Struck claims stay in their source docs — the audit strikes them in this ledger only, and never edits a source doc.

## Matched claims (suite result behind each)

| Claim | Source | Baseline | Metric | Filed value |
|---|---|---|---|---|
| `persona_suite_overall_score` | `docs/architecture/PERSONA_CONSISTENCY_SUITE.md` | `persona_suite_synthetic_v1` | `overall_score` | 0.8453703703703704 |
| `persona_suite_turn_count` | `docs/architecture/PERSONA_CONSISTENCY_SUITE.md` | `persona_suite_synthetic_v1` | `total_turns` | 108 |
| `memory_suite_overall_score` | `docs/architecture/MEMORY_FIDELITY_SUITE.md` | `memory_suite_synthetic_v1` | `overall_score` | 0.8055555555555557 |
| `depth_suite_overall_on_score` | `docs/architecture/DEPTH_SUITE.md` | `depth_suite_synthetic_v1` | `overall_on_score` | 0.8055555555555557 |
| `depth_suite_delta` | `docs/architecture/DEPTH_SUITE.md` | `depth_suite_synthetic_v1` | `delta` | 0.3333333333333334 |
| `presence_suite_overall_score` | `docs/architecture/PRESENCE_SUITE.md` | `presence_suite_synthetic_v1` | `overall_score` | 0.7777777777777778 |
| `quant_panel_q4_k_m_score` | `docs/architecture/QUANT_PANEL.md` | `quant_panel_synthetic_v1` | `q4_k_m_score` | 0.8125 |
| `quant_panel_q5_k_m_score` | `docs/architecture/QUANT_PANEL.md` | `quant_panel_synthetic_v1` | `q5_k_m_score` | 0.7083333333333334 |
| `wyrd_panel_overall_score` | `docs/architecture/WYRD_PANEL.md` | `wyrd_panel_synthetic_v1` | `overall_score` | 0.5555555555555556 |
| `cost_panel_total_token_budget` | `docs/architecture/COST_PANEL.md` | `cost_panel_synthetic_v1` | `total_token_budget` | 766800 |
| `baseline_book_baseline_count` | `docs/architecture/BASELINE_BOOK.md` | `baseline_book_synthetic_v1` | `total_baselines` | 17 |
| `flake_discipline_deterministic_count` | `docs/architecture/FLAKE_DISCIPLINE.md` | `flake_discipline_synthetic_v1` | `deterministic_baselines` | 17 |

## Struck claims (reason filed for each)

### Struck: `model_card_citation_accuracy_target`

- Source: `MODEL_CARD.md`
- Claim: MODEL_CARD target table: 'Citation accuracy ≥ 85%'
- Reason: filed as a target only, never as a result: the only measured citation_accuracy in the estate is the empty-store baseline's 0.0 (profile_benchmark_empty_store_v1, target 0.85 filed unmet) — no suite result supports the claim as achieved

### Struck: `model_card_enforcement_pass_target`

- Source: `MODEL_CARD.md`
- Claim: MODEL_CARD target table: 'Enforcement pass rate ≥ 90%'
- Reason: filed as a target only: the only measured enforcement_pass_rate is the empty-store baseline's 0.0 (target 0.90 filed unmet) — no suite result supports it as achieved

### Struck: `model_card_avg_response_words_target`

- Source: `MODEL_CARD.md`
- Claim: MODEL_CARD target table: 'Avg response words ≥ 30'
- Reason: filed as a target only: the measured avg_response_words is 9.0 on the empty-store baseline (target 30.0 filed unmet, D-BENCH-2 / TF-121) — no suite result supports it as achieved

### Struck: `model_card_persona_100_plus_turns`

- Source: `MODEL_CARD.md`
- Claim: MODEL_CARD: 'Persona consistency ≥ 0.75 — PersonaConsistencyScorer across 100+ turns'
- Reason: D-BENCH-4: no PersonaConsistencyScorer run over 100+ turns exists — the scorer's own pin is 6 turns at 0.7167 (below 0.75); the 108-turn run that does exist is TF-122's persona SUITE, a different instrument, and this doc does not point at it

### Struck: `build_plan_persona_100_plus_turns`

- Source: `BUILD_PLAN_v1.md`
- Claim: BUILD_PLAN_v1: 'Personality stable across 100+ turns on any hardware tier' (and its 100+ turn test milestone)
- Reason: D-BENCH-4: the wording points at no run; no per-hardware-tier persona run exists (live per-profile behaviour is NOT PROVEN, TF-173) — the suite's 108-turn fixture run is not a hardware-tier result

### Struck: `build_plan_citation_accuracy_85`

- Source: `BUILD_PLAN_v1.md`
- Claim: BUILD_PLAN_v1: 'Benchmarks per hardware profile: citation accuracy (>85%)'
- Reason: no per-profile citation-accuracy result exists: TF-128's profile panel files suite scores and bench average words per profile, no citation accuracy — the claim has no suite result behind it

### Struck: `task_phase6_persona_100_plus_turns`

- Source: `TASK_PHASE6_RELEASE.md`
- Claim: TASK_PHASE6: PersonaConsistencyScorer 'Designed to validate the Skald persona across 100+ turns'
- Reason: D-BENCH-4: the same standing wording — no such scorer run exists (see model_card_persona_100_plus_turns)

### Struck: `task_phase6_locust_httpuser`

- Source: `TASK_PHASE6_RELEASE.md`
- Claim: TASK_PHASE6: 'ThoughtForgeUser(HttpUser) — Locust load test user'
- Reason: D-BENCH-5: stale description — the locust file is no-HTTP by its own header, unrunnable without the dev extra, and TF-121 excluded it from the harness panel by verdict as a manual throughput script

### Struck: `model_card_edge_inference_speed`

- Source: `MODEL_CARD.md`
- Claim: MODEL_CARD: 'Pi Zero and phone_low profiles are slow (minutes per response without model optimization)'
- Reason: a live-model speed claim with no live run behind it: no live model runs on this VM (TF-173) and latency is never a pinned baseline (TF-121) — no suite result supports any per-response time

## The rules

- `every_public_performance_claim_is_censused`: every performance claim in the repo's docs (docs/**/*.md plus the top-level *.md docs) that purports to state a suite / harness result or a public performance target files one CLAIM_ROWS row: its source doc, a marker standing verbatim in that doc, the claim, and its disposition
- `matched_means_a_live_verified_suite_value`: a claim is matched only with a harness baseline id, a metric, and the filed value — and the checker recomputes that metric live through the harness's own measurer, so a drifted value is named, never carried
- `struck_means_a_filed_reason_never_a_deletion`: a claim no suite result supports — a target filed as if a result, a live-model claim with no live run, a stale number or description — is struck WITH its reason filed in the audit doc; struck claims are never silently deleted and their source docs are never edited by this slice (README.md is read for claims, never edited at all)
- `targets_are_not_results`: a target a doc files as a target stays a target: it files matched only when a suite result actually meets it under the instrument the claim names — a target table is not evidence
- `probes_prove_the_validator_can_fail`: CLAIM_PROBES perturbs copies of filed rows against the pure validator (a matched claim with a wrong value, a struck claim that actually matches, a claim missing its source, a matched claim missing its metric) plus the clean shapes — real filings are never perturbed
- `pass_needs_the_full_census_classified`: the verdict is PASS only on the full census with every claim classified matched-with-live-verified-value or reasoned-struck; a partial selection files REVIEW however it scores, and an empty selection files 0 claims / REVIEW
- `the_public_claims_baseline_is_not_gated_by_tf129`: TF-129's regression gate gates the nine TF-121..TF-128 pinned baselines exactly as filed; this slice's own baseline (public_claims_synthetic_v1) is judged by this module's checker, not gated by the gate — the shape TF-129..TF-135 filed for their own baselines, with the exclusion filed in the open in regression_gate.py
- `no_new_benchmark_input_was_added`: the audit consumes only already-filed docs and pins — it adds no benchmark input, so it files no new harness fixture and no hygiene row, and TF-131's inventory and its pinned baseline do not move (filed in the open, TF-133's coupling shape)

## Filed limits

- The census covers performance claims that purport to state a suite / harness result or a public performance target. Per-slice device filings owned and checked by their own slice modules are not re-judged here; vague capability prose files no claim.
- A matched claim is matched to a pinned FIXTURE baseline — scripted synthetic suites through the real scorers on this VM. No live model stands behind any matched value; live behaviour is NOT PROVEN (TF-173).
- A struck claim is struck from the ledger of claims the repo's suites support; the audit does not edit its source doc.

Proof grade: FIXTURE — pinned synthetic-fixture baselines re-measured live on this VM; no live model, never claimed.
