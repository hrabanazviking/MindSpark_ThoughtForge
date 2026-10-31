# TF-016 — The Bug & Gap Docket (Part I Audit)

Compiled 2026-10-30 on branch `yrsa/thoughtforge-road-build`.
Method: every defect code (`D-*`) grepped out of TF-002…TF-015 and
re-read in its source doc, plus every uncoded finding in those docs
given a number here in the same family style, plus the one defect
the campaign itself found and fixed (D-CI-1). Nothing in Part I is
left unnumbered; nothing here is fixed by this slice — each entry
is assigned to the slice that fixes it, and an entry closes only
when that slice banks with evidence. TF-192 (roadmap audit)
verifies at the end that no entry is left open without a verdict.

**Totals: 46 entries — 42 OPEN, 1 OPEN-ENVIRONMENT, 2 FIXED
(D-CI-1 in Part I; D-DUP-1 by TF-025 in Part II), 1
CLOSED-BY-VERDICT.** Primary-owner parts: Part I 1 ·
Part II 7 · Part III 5 · Part IV 7 · Part V 1 · Part VI 6 ·
Part VII 7 · Part VIII 1 · Part IX 2 · Part X 9.

## The docket

| ID | Defect (evidence) | Fixing slice(s) | Part | Status |
|---|---|---|---|---|
| D-ENV-1 | sentence-transformers absent on this VM → the suite's 1 failure + the doctor's 1 issue; same root at TF-004, TF-005 (×2, incl. forge_memory's pre-install click failure family), TF-013, TF-014 | Install the full declared set on the test machine (the truthful road — it is a core dependency in pyproject), or a required/optional split in the dependency check; final baseline book must state machine + stack: TF-194 | X | OPEN-ENVIRONMENT |
| D-TEST-1 | `test_yaml_present` asserts the whole dependency set, not YAML — stale-test naming smell (TF-004) | TF-187 (regression armour review) or any slice touching phase8 tests | X | OPEN |
| D-CI-1 | `.github/workflows/ci.yml` auto-triggered on push/PR, violating Volmarr's manual-only-workflows law (TF-003) | Fixed by the campaign itself: commit `2b1201c` (workflow_dispatch only), banked locally, not pushed | I | **FIXED** |
| D-COV-1 | `knowledge/forge.py` — the retrieval heart — has no direct test import; coverage is indirect via core (TF-002) | TF-043 (retrieval precision fixtures exercise it directly) | III | OPEN |
| D-COV-2 | ETL `sources.py`, `wikidata.py`, `embeddings.py` have no direct test imports (TF-002) | TF-107 / TF-112 (Part VI owns, per TF-002) | VI | OPEN |
| D-COV-3 | `utils/paths.py` and `utils/logging_setup.py` have no direct test imports (TF-002) | TF-038 (logging_setup) + TF-069 (paths, via profile loading) | II | OPEN |
| D-WING-1 | The research wing cannot import: the entire `wyrdforge.models` subpackage was never committed; its 2 test files error at collection, 0 run (TF-006) | Verdicts TF-139–TF-142; any ADOPT budgets rebuilding the models from the 14 JSON schemas (TF-143); tests TF-149 | VIII | OPEN |
| D-LIC-1 | The project's own licence contradicts itself: CC BY 4.0 (permits commercial use) vs a "non-commercial" classifier; no LICENSE file in the repo; a content licence applied to code (TF-008) | TF-190 — with Volmarr's word; the licence choice is his | X | OPEN |
| D-LIC-2 | Upstream data-source licences unverified and recorded nowhere in-tree: ConceptNet / GeoNames / DBpedia / Wikidata terms, the D&D 5e SRD in the library, and the all-MiniLM-L6-v2 model-card licence (TF-008 finding 7, TF-007 finding 4) | TF-106 (source registry records each licence from upstream terms); embedding-model licence TF-053; closes TF-190 | VI | OPEN |
| D-DEP-1 | `requests` is an undeclared hard dependency: top-level, unguarded import in 4 inference backend modules; works only transitively (TF-008) | TF-065 (declares it, or the backends go lazy — contract slice chooses) | IV | OPEN |
| D-DEP-2 | Broken console scripts: `[project.scripts]` points at `thoughtforge.cli:main` and `thoughtforge.etl.cli:main`; no `cli.py` exists in either package — `pip install` succeeds, both commands crash on first run (TF-008) | TF-153 (service/library shape owns the packaged entry points) | IX | OPEN |
| D-DEP-3 | Declared-but-unused dependency weight: sqlalchemy, rich, platformdirs in core; bitsandbytes, auto-gptq, accelerate in extras — promised to every installer, imported by no code (TF-008) | TF-069 / TF-110 weigh removal (install weight is a feature cost on phone_low) | IV | OPEN |
| D-DEP-4 | No pins, no lockfile: every requirement is a bare lower bound, so "the dependency set" is a moving target (TF-008). Interim: TF-014 froze the tested-version record | TF-185 (release shaping decides the lockfile) | X | OPEN |
| D-DEP-5 | Further undeclared imports riding transitively or declared nowhere: transformers, huggingface_hub (lazy), onnxruntime, optimum, psutil (guarded) (TF-008) | TF-065 (backend contract declares or guards each) | IV | OPEN |
| D-CFG-1 | `default.yaml` is mostly a mirage: 27 of 34 keys SHADOWED by identical hard-coded constants, 2 DEAD; its retrieval weights are also name-drifted (`tone_match`/`user_preference` vs code's `tone_similarity`/`preference_relevance`), so naive wiring would drop 2 of 5 weights (TF-009) | TF-032 (live keys only, per TF-009's exact list) + TF-044 (scoring weights made tunable) | II | OPEN |
| D-CFG-2 | Personality expressive fields DORMANT: traits, speech_style, behavior_rules and all 5 tuning weights are parsed and carried but steer nothing — only `avoid[:3]` + `record_id` are consumed (TF-009) | TF-030 (persona consistency hooks) + TF-141 (persona compiler verdict) | II | OPEN — **narrowed by TF-030**: traits / speech_style / behavior_rules now steer every prompt via the persona steering block (`cognition/persona.py`, `inject_persona`), and `avoid` is checkable through the hook; the remaining dormancy is exactly the 5 tuning weights — carried on `PersonaExpectation` for scorers, given no steering effect by TF-030's deliberate ruling (numbers are not prose; the compiler question is TF-141's) |
| D-CFG-3 | `user_config.yaml` mixes 10 LIVE backend keys with 5 DEAD (profile / knowledge_db_built / chat_*); the factory's fallback key `openai_model` is read and written by the wizard but absent from the shipped file (TF-009) | TF-065 (backend contract) + TF-153 (service shape) | IV | OPEN |
| D-CFG-4 | Credential-shaped keys (`openai_api_key`, `hf_token`) live in a plain YAML file — empty in the shipped copy, so no secret is in the tree, but the shape is the defect (TF-009) | TF-160 (Kista custody) | IX | OPEN |
| D-CFG-5 | No single config system: three loaders (cached `load_config()`, hand-parsing in unified_backend + validators/health/self-heal, the memory store's own personality load) never meet; no code path sees a merged view (TF-009 finding 1) | TF-032 + TF-153 | II | OPEN |
| D-BENCH-1 | `benchmark_profiles`: `profile_id` is never validated (a bogus profile exits 0) and never applied — `ThoughtForgeCore` has no profile parameter; every profile measures the same ephemeral empty-store core (TF-010). Child of D-PROF-3 | TF-121 (harness) + TF-128 (profile panels) | VII | OPEN |
| D-BENCH-2 | Metric name drift in `benchmark_profiles`: `token_efficiency` and `citation_accuracy` do not compute what their names/docstrings say; MODEL_CARD's "avg response words ≥ 30" target is checked nowhere (TF-010) | TF-121 | VII | OPEN |
| D-BENCH-3 | Persona scorer: the `weights` argument is built into a dict and never read — callers are silently ignored; the docstring formula is not the implementation (skald bonus undocumented) (TF-010) | TF-122 (persona consistency suite) | VII | OPEN |
| D-BENCH-4 | The "100+ turns" persona claim exists in 3 docs with no fixture or recorded run anywhere; an empty response list scores a vacuous 1.0 PASS (TF-010) | TF-122 + TF-136 (public-claims audit strikes or proves the claim) | VII | OPEN |
| D-BENCH-5 | `locustfile.py` is unrunnable as shipped (locust is a dev extra, not installed; its 3 tests are structural only) and TASK_PHASE6's HttpUser description of it is stale (TF-010) | TF-121 (harness decides its fate) + TF-136 (stale claim) | VII | OPEN |
| D-BENCH-6 | Benchmark estate decay bundle (TF-010): `quality_tiers` declared and never populated (always `{}`); warm-up turns are drawn from the measured set; the p95 index picks ~p90 at n=10 (printed below the mean); nothing is persisted — no baseline exists to regress against; the persona CLI cannot reach the module's own `generate_report()` | TF-121 (harness architecture starts from zero, per TF-010) | VII | OPEN |
| D-BENCH-7 | Two unconnected latency systems: `utils/perf.py`'s tracker is wired into the core, but no benchmark consumes it — `ProfileBenchmark` times turns with its own `perf_counter` calls (TF-010) | TF-121 + TF-133 (cost panel) | VII | OPEN |
| D-BENCH-8 | The `PYTHONPATH=src` requirement (package not pip-installed) is stated in no usage doc — without it the benchmark run-tests fail 3× with ModuleNotFoundError (TF-010 method note) | TF-182 (developer docs) | X | OPEN |
| D-DEAD-1 | `utils/validators.py`: 5 of its 6 public functions are never called anywhere (`sanitise_system_prompt`, `validate_config`, `validate_model_path`, `validate_api_key`, `validate_jsonl_line`) — a validation layer that guards nothing (TF-011) | Wire-or-retire verdict per function: TF-032 (config surface) / TF-094 (policy configuration) | II | OPEN |
| D-DEAD-2 | Scalar `cosine_similarity` (etl/embeddings.py:84) is dead while its batch sibling is live (TF-011) | TF-053 (embedding discipline) | III | OPEN |
| D-DEAD-3 | `utils/paths.py:get_embeddings_dir` has never been called (TF-011) | TF-053 / TF-110 era | III | OPEN |
| D-ORPHAN-1 | `knowledge/lifecycle.py` is test-only: MemoryLifecycle is never wired into the live engine — no fact ever ages, decays, or is pruned at runtime (TF-011) | TF-045 (lifecycle wiring) | III | OPEN |
| D-ORPHAN-2 | `utils/retry.py` is test-only: CircuitBreaker/retry protect no production backend call (TF-011) | TF-073 (backend health & fallback) | IV | OPEN |
| D-ORPHAN-3 | Four more test-only modules, split owners (TF-011): `etl/db_integrity.py`, `etl/subset.py`, `inference/model_browser.py`, `inference/onnx_export.py` — see the reassignment note below | TF-167 / TF-110 / TF-078 / TF-070 respectively | IV | OPEN |
| D-DUP-1 | Three copied private helpers, drift-prone: `_now_iso` ×3 (one byte-identical pair), `_split_sentences` ×2, `_build_messages` ×2 (TF-011) | TF-025-era hygiene (one shared utils home) | II | **FIXED by TF-025**: implementations now live once in `utils/common.py` (`now_iso` / `split_sentences` / `build_messages`); old private names are aliases, identity-pinned by tests/test_tf025_turn_state.py |
| D-DUP-2 | `MindSpark_ThoughtForge/` shadow directory holds md5-identical copies of root RULES.AI.md / PHILOSOPHY.md — copies that can silently diverge (TF-011; its README was not read or touched, per law) | TF-182 (Part X docs verdict) | X | OPEN |
| D-PROF-1 | Every profile's entire `memory` + `retrieval` + `deployment` sections, `max_params_b`, and `recommended_models` are read by no code — the per-profile budget claims are unenforced on all six profiles (TF-012) | TF-056 (memory budgets) + TF-069 (profile runner) + TF-110 (subset craft) | III | OPEN |
| D-PROF-2 | Selector floors ≠ the JSONs' claimed floors on all six rows (TF-012): pi_5 handed to 2 GB devices, desktop_cpu to 6 GB, desktop_gpu to 6 GB VRAM, server_gpu selected with RAM never checked; phone_low is also the x86 low-RAM fallback and pi_zero a phone landing. Sub-items: the selector class docstring's rule order is not the code's order; failed RAM detection assumes 4 GB; the `n_gpu_layers` ladder exists only in the resolver, stated in no JSON; BUILD_PLAN_v1's pi_zero size (1B) disagrees with its JSON (0.5B) | TF-069 (profile runner); truth table v2 TF-086 | IV | OPEN |
| D-PROF-3 | The cognition core is profile-blind: flat retrieval top_k=10, max_tokens=512, salvage 2 passes, history 20 turns/2000 chars, mode temperatures 0.50–0.80 on every profile (TF-012). Parent of D-BENCH-1 | TF-032 + TF-033 (latency budgets) + TF-037 (small-model floor) | II | OPEN |
| D-PROF-4 | Resolved `max_refinement_passes` is computed and read by no code; salvage runs a flat 2 passes even on pi_zero, whose JSON promises 0 (TF-012) | TF-098 (refinement cost discipline) | V | OPEN |
| D-DATA-1 | Stale count in code: sources.py + ReferenceDataETL docstrings say "40 built-in files"; the library holds 215 root files + 17 SRD chapters (TF-007) | TF-119 (ETL docs; Part VI owns per TF-007) | VI | OPEN |
| D-DATA-2 | Data duplication: 37 duplicate blob hashes, 44 extra copies, 28,109,539 B redundant — incl. `norse_paganism_1000_training_pairsv2.jsonl` being the *same blob* as v1 in both locations, and two differently-named protocol files that are one blob (TF-007). Nothing was moved or deleted by the census; canon handling stays Volmarr's word | TF-109 (dedup & merge rules) | VI | OPEN |
| D-DATA-3 | No provenance/licence file inside `data/`; provenance for external-source sets lives only in ETL code comments (TF-007 finding 4) | TF-106 (source registry) + TF-190 | VI | OPEN |
| D-DATA-4 | Most of `data/` is not ingestible by ReferenceDataETL: persona imagery (225 jpg / 152.9 MB) and the lone .csv/.py/.rst fall outside its supported set — ingestion must skip by rule, not by accident (TF-007 finding 1) | TF-118 (reference library as a source) + TF-058 (binding) | VI | OPEN |
| D-REC-1 | TASK_PHASE1 and TASK_PHASE2 docs are absent from the tree (git ls-tree, TF-003) — a gap in the record, not in the software; their work exists as code | None — TF-015 verdict: no reconstruction commissioned (inventing them retroactively would falsify the record); TF-182 notes the gap for new readers | X | **CLOSED-BY-VERDICT** |
| D-REL-1 | BP Phase 6's release claims are NOT PROVEN from the tree: no v1.0 GitHub release, HF model cards, or public demo is evidenced (TF-003) | TF-185 (release shaping) | X | OPEN |
| D-DOC-1 | BUILD_PLAN_v1's spec pointers drift: of its 14 named spec files, 8 match exactly, 2 differ by case, 4 stand under drifted names — all subjects present, the plan's filenames unreliable (TF-015). No elder spec is renamed by the verdict | TF-182 (developer docs carry the correct pointers) | X | OPEN |

## Notes the table cannot hold

- **D-ORPHAN-3 reassignment (a catch of this docket).** TF-011
  assigned `etl/db_integrity.py`'s wiring to TF-013 — but TF-013
  has already banked, and it extended the doctor's *counting*, not
  its integrity checking: verified this slice, neither
  `forge_doctor.py` nor `forge_memory.py` references
  `DBIntegrityChecker` / `db_integrity` anywhere. The assignment
  was missed, so it moves to the next doctor slice, **TF-167**
  (the doctor, service edition). The other three keep TF-011's
  owners: subset → TF-110, model_browser → TF-078, onnx_export →
  TF-070.
- **D-ENV-1 is an environment entry, not a code defect.** The
  health checker told the truth about this machine. It stays on
  the docket so the campaign's one failure is never silently
  normalised; it closes by installation on a named machine or by
  the dependency-check split — never by weakening the test
  (TF-004's ruling stands).
- **Doctor gaps are not docket entries.** TF-013 listed five
  headlines `forge_doctor --audit` deliberately does not emit
  (suite counts, data totals, config classification, benchmark
  scores, profile truth) because each needs a run, a trace, or a
  verdict rather than a count. Each already has an owner
  (TF-014 — banked; TF-007, TF-009, TF-010, TF-012 stand as the
  records; TF-032 and Part VII consume them). They are gaps in
  one instrument's output, not defects in the tree.
- **Closed by TF-015, not defects:** BUILD_PLAN_v1's stale status
  lines, counts (14 vs 18 record types), and superseded numbers
  in CHANGELOG / TASK_PHASE8 are verdicts of record in TF-003 /
  TF-015 against the TF-014 zero line. The elder docs keep their
  bytes; no docket entry is needed to fix a document the campaign
  has already ruled historical.
- **The closure rule (binding on Parts II–X):** a slice that
  owns a docket entry names it in its banked doc, shows the fix
  or the verdict, and the entry's status flips there. Any entry
  still OPEN at TF-192 must carry a written reason — "nothing
  found, nothing unfixed" is this docket's whole point.

Suite at banking: see CAMPAIGN_STATE.md (TF-014 zero line held —
this slice adds one audit doc and no code or tests).
