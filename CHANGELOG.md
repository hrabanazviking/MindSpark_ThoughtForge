# Changelog

All notable changes to MindSpark: ThoughtForge are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [1.0.0] — 2026-03-31

### Added

**Phase 0 — Foundation**
- Full package structure: `src/thoughtforge/{knowledge,inference,etl,cognition,refinement,utils}/`
- `pyproject.toml`, `requirements.txt`, `setup.py` with all optional extras
- Six hardware profile JSONs: `phone_low`, `pi_zero`, `pi_5`, `desktop_cpu`, `desktop_gpu`, `server_gpu`
- GitHub Actions CI: lint (ruff + mypy) + test matrix (Ubuntu/Windows/macOS × Python 3.10–3.12)
- `configs/default.yaml`, `CONTRIBUTING.md`, `.gitignore`

**Phase 1 — Memory Forge + Sovereign RAG**
- All 14 data structure types (`PersonalityCoreRecord`, `UserPreferenceRecord`, `UserFactRecord`,
  `EpisodicMemoryRecord`, `ResponsePatternRecord`, `ActiveThreadStateRecord`, `InputSketch`,
  `MemoryActivationBundle`, `CognitionScaffold`, `CandidateRecord`, `FragmentRecord`,
  `FinalResponseRecord`, `WritebackRecord`, `RuntimeTurnState`)
- `MemoryForge` — hybrid SQL+vector retrieval with activation scoring and bundle assembly
- `MemoryStore` — file-based persistent store (YAML, JSONL, JSON)
- `MemoryLifecycle` — 4-mode pruning: light, routine, heavy, emergency
- ETL pipelines: Wikidata (streaming ijson), DBpedia, ConceptNet, GeoNames, 40 built-in reference files
- `EmbeddingStore` — sentence-transformer embeddings (all-MiniLM-L6-v2) via sqlite-vss
- `forge_memory.py` Click CLI: `init / wikidata / conceptnet / geonames / dbpedia / reference / embeddings / status / all`

**Phase 2 — TurboQuant Universal Inference Engine**
- `TurboQuantEngine` — llama-cpp-python wrapper with strict token budget enforcement, multi-draft generation
- `BackendDetector` — auto-detects CUDA, ROCm, Vulkan, Metal, CPU with priority ordering
- `HardwareProfileLoader` — loads profile JSON, auto-detects hardware tier
- 6 hardware profiles fully specified (RAM, VRAM, quantization, token budgets, draft counts)

**Phase 3 — Cognition Scaffolds + Orchestration**
- `InputRouter` — intent classification (8 categories), tone detection, retrieval path derivation
- `ScaffoldBuilder` — table-driven `CognitionScaffold` assembly (goal, tone, focus, avoid, depth, fact_block)
- `PromptBuilder` — mode-specific candidate prompts, refine prompts, repair prompts
- `ThoughtForgeCore.think()` — 8-step mandatory pipeline: retrieve → score → scaffold → generate → salvage → enforce → write back → return
- `configs/personality_core.yaml` — Skald persona (calm, direct, cite-or-explain)
- Heuristic scoring: keyword_overlap, genericness_penalty, specificity_score, length_score (no judge model)

**Phase 4 — Fragment Salvage + Refinement**
- `FragmentSalvage` — multi-pass draft scoring (length 45% + citation 55%), sentence-level extraction, up to 2 refine passes
- `EnforcementGate` — citation integrity, length (≥5 words), genericness checks; soft-fail `[Forge:]` notes
- `ThoughtForgeCore` — wired `FragmentSalvage` + `EnforcementGate` into `_compose_final()`
- `run_thoughtforge.py` — interactive REPL (`Forge>`) + single-query CLI with argparse
- `FinalResponseRecord.enforcement_passed` / `.enforcement_notes` fields

**Phase 5 — Edge + Cross-Platform Deployment**
- `OnnxExporter` — exports sentence-transformer models to ONNX (optimum → torch fallback, int8 quantization)
- `ONNXEmbedder` — drop-in onnxruntime encoder with mean-pool + L2 normalization
- `EdgeSubsetBuilder` — builds reduced SQLite knowledge DB for edge profiles (50K–200K entities)
- `Dockerfile` — multi-stage Python 3.11-slim, `--build-arg PROFILE`, healthcheck
- `docker-compose.yml` — desktop, GPU, Pi, phone named services
- `scripts/install_linux.sh` — Debian/Ubuntu/Arch/Fedora with auto-detection
- `scripts/install_mac.sh` — Homebrew + Apple Silicon Metal flag
- `scripts/install_windows.ps1` — PowerShell + Vulkan flag
- `scripts/install_termux.sh` — Termux/Android, phone_low profile, ARM build
- `scripts/install_pi.sh` — Pi Zero/5 auto-detect via `/proc/meminfo`, Vulkan VideoCore VII

**Phase 6 — Testing, Benchmarking, Release**
- `ProfileBenchmark` — per-profile metrics: citation accuracy, latency (avg/median/p95), token efficiency, enforcement pass rate
- `PersonaConsistencyScorer` — phrase-level Skald persona validation; generic penalty + Norse tone bonus + citation bonus
- `locustfile.py` — Locust load test in no-HTTP mode (`ThoughtForgeUser`)
- Integration test suite: end-to-end `think()`, multi-call stability, knowledge-only mode, `FinalResponseRecord` completeness
- Adversarial test suite: empty input, whitespace, very long query, SQL injection string, Unicode/emoji, repeated queries
- MkDocs documentation site with Material theme: index, quickstart, hardware profiles, API reference
- `CHANGELOG.md`, `MODEL_CARD.md`

### Test Coverage

- 433 tests passing across 8 test modules
- Platform: Windows 11, Python 3.10.11 (CI: Ubuntu/Windows/macOS × Python 3.10–3.12)

### Known Limitations

- ONNX export requires `optimum` or `torch` (not in default install)
- `EdgeSubsetBuilder` requires a populated full DB to subset from
- Locust load tests run in Python-native mode — HTTP deployment wrapper not included in v1.0
- sqlite-vss vector search is optional — falls back to SQL-only retrieval if not installed
- Wikidata full dump ETL requires ~100 GB free disk + several hours of processing time

---

## [0.1.0] — 2026-03-31

Initial development builds (Phases 0–5). See commit log for details.
Not released publicly.

---

[1.0.0]: https://github.com/hrabanazviking/MindSpark_ThoughtForge/releases/tag/v1.0.0
[0.1.0]: https://github.com/hrabanazviking/MindSpark_ThoughtForge/commits/development
