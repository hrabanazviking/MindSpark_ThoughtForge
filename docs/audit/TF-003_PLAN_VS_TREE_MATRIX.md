# TF-003 — Plan-vs-Tree Matrix (Part I Audit)

Measured in-tree 2026-10-30, branch `yrsa/thoughtforge-road-build`.
BUILD_PLAN_v1 (BP) marks every phase "NOT STARTED" / "Current Phase:
Phase 0 (not yet started)". That status line is stale against the
tree in every row below. Verdicts: DONE-in-fact (code + tests exist),
PARTIAL (some deliverables real, some absent/unproven),
ABSENT, SUPERSEDED (roadmap §6 / campaign law replaces the item).

## BUILD_PLAN_v1 phases 0–6 (the plan defines 0–6, not 0–8)

| BP phase | Plan claim | Tree reality (file evidence) | Verdict |
|---|---|---|---|
| BP-Ph0 Foundation | NOT STARTED | Skeleton `src/thoughtforge/` (44 files), `docs/specs/` 27 spec files, `pyproject.toml` + `setup.py` + `requirements.txt`, 6 hardware-profile JSONs in `hardware_profiles/`, `CONTRIBUTING.md`, `.gitignore`, CI at `.github/workflows/ci.yml` | DONE-in-fact — with one defect: ci.yml auto-triggers on push/PR, violating Volmarr's manual-only-workflows law; fixed in the commit following this doc |
| BP-Ph1 Knowledge | NOT STARTED | `knowledge/models.py` (18 record types — plan said 14; tree grew past the plan), `knowledge/forge.py` hybrid retrieval, `knowledge/lifecycle.py` with all 4 prune modes (light/routine/heavy/emergency, lines 34–51), `knowledge/scoring.py` formulae, `etl/wikidata.py`, `etl/sources.py` (DBpedia/ConceptNet/GeoNames/Reference), `etl/embeddings.py`, `forge_memory.py` CLI | DONE-in-fact as code; PARTIAL as proof — the ETL modules have no direct tests (TF-002) and a full-dump build was not re-run this campaign (TF-005 tests the CLI on a synthetic store) |
| BP-Ph2 TurboQuant | NOT STARTED | `inference/turboquant.py` TurboQuantEngine, `inference/profiles.py` detection/selection for the 6 tiers, `inference/backends.py` CUDA/ROCm/Metal/Vulkan/DirectML/CPU probes, unified backend + ollama/lmstudio/hf backends | DONE-in-fact as code; PARTIAL as proof — no live model generation re-proven this campaign yet (Part IV owns that) |
| BP-Ph3 Cognition | NOT STARTED | `cognition/router.py` InputRouter, `cognition/scaffold.py` ScaffoldBuilder, `cognition/prompt_builder.py`, `cognition/core.py` ThoughtForgeCore.think() loop with inline citation gate, `configs/personality_core.yaml` | DONE-in-fact |
| BP-Ph4 Salvage | NOT STARTED | `refinement/salvage.py` FragmentSalvage (multi-draft scoring/reassembly), `refinement/enforcement.py` EnforcementGate, `run_thoughtforge.py` interactive CLI, core.py citation/scoring helpers | DONE-in-fact |
| BP-Ph5 Deployment | NOT STARTED | Native install scripts for linux/mac/pi/termux/windows in `scripts/`; `inference/onnx_export.py`; subset builder `etl/subset.py`. Docker artefacts exist (Dockerfile, docker-compose.yml) but campaign law forbids Docker use. iOS/MLC appears only as runtime strings in `hardware_profiles/phone_low.json` — no code | PARTIAL + SUPERSEDED-in-part (Docker road superseded by the no-Docker law; iOS/MLC ABSENT as code) |
| BP-Ph6 Testing/Release | NOT STARTED | Test suite real: 620 items collected by the runner (TF-004), incl. adversarial + integration files; `locustfile.py`; `benchmarks/persona_consistency.py` scorer + `benchmark_profiles.py`; `mkdocs.yml`; CHANGELOG claims [1.2.0] 2026-04-01 | PARTIAL — testing/benchmark/docs DONE-in-fact; "v1.0 GitHub release + HF model cards + public demo" NOT PROVEN from the tree (release truth is Part X, TF-185) |

## TASK_PHASE docs (elder task files)

Present in tree root: TASK_PHASE0_SETUP, 3_COGNITION, 4_SALVAGE,
5_DEPLOYMENT, 6_RELEASE, 7_SETUP_WIZARD, 8_ROBUSTNESS.
**TASK_PHASE1 and TASK_PHASE2 docs are ABSENT from the tree** (git
ls-tree confirms) — their work is nevertheless in the tree
(knowledge/ and inference/ packages, phase1/phase2 test files).

| Task doc | Its own baseline claim | Tree check | Verdict |
|---|---|---|---|
| TASK_PHASE0 | skeleton pending | skeleton + all listed artefacts exist | DONE-in-fact; its pending checklist is stale |
| TASK_PHASE3 | "cognition/ and refinement/ packages are empty stubs" | cognition/ = 65,896 B of real code; refinement/ likewise | Claim SUPERSEDED by the tree (the doc predates the work it commissioned) |
| TASK_PHASE4 | "refinement/__init__.py stub only", Phase 0–3 complete, 191 tests | refinement/ fully implemented; runner now collects 620 | DONE-in-fact; counts stale |
| TASK_PHASE5 | deliverables table ✅ incl. Dockerfile/compose | files exist; Docker items SUPERSEDED by law | DONE-in-fact except Docker road |
| TASK_PHASE6 | release phase | suite + benchmarks + mkdocs exist; release artefacts NOT PROVEN | PARTIAL |
| TASK_PHASE7 | setup wizard + multi-backend + chat, depends Phase 6 ✅ | `setup_thoughtforge.py` (26,765 B), ollama/lmstudio/hf/unified backends, `cognition/chat_history.py`, phase7 test files (83 test fns) all present | DONE-in-fact as code |
| TASK_PHASE8 | robustness: health/self-heal/perf/db-integrity ✅ table | all four modules exist at the named paths; phase8 test files present; CHANGELOG 1.2.0 entry describes exactly these | DONE-in-fact — and its dependency health check is the source of TF-004's one environment failure |

## Headline

The plan is not a plan for future work; it is a stale label on
finished work. The true open frontier is exactly what the roadmap's
Parts II–X name: proof, hardening, and honest per-profile truth —
not building Phases 0–6 from scratch. BUILD_PLAN_v1 itself is not
edited (TF-015 records its supersession in this audit line).
