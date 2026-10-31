# TF-002 — Module Census (Part I Audit)

Measured in-tree 2026-10-30 on branch `yrsa/thoughtforge-road-build`
(base `development` @ d267148 + TF-001 filing e7d21aa). Sizes are file
bytes; surface = top-level classes/functions from AST parse; "tests"
means a test file imports the module directly (inline imports count).

## Package summary

| Package | Files | Bytes | Role (from package docstrings) |
|---|---|---|---|
| cognition | 6 | 65,896 | Scaffolds, intent routing, orchestration loop |
| etl | 7 | 82,379 | Streaming ingestion pipelines for knowledge sources |
| inference | 11 | 90,656 | TurboQuant engine + unified backend abstraction |
| knowledge | 6 | 70,680 | Memory Forge + Sovereign RAG |
| refinement | 3 | 19,872 | Fragment salvage and citation enforcement |
| utils | 10 | 70,092 | Logging, config, health, retry, self-heal, validation |
| **src total** | **44** | **399,944** | (package rows include their `__init__` files; the root `thoughtforge/__init__.py`, 369 B, is in the total only) | |

Root instruments (not in src/): `run_thoughtforge.py` (10,840 B),
`forge_memory.py` (10,123 B), `forge_doctor.py` (6,114 B),
`setup_thoughtforge.py` (26,765 B).

## Module-by-module

### cognition/
| Module | Bytes | Public surface | Direct tests |
|---|---|---|---|
| core.py | 28,943 | ThoughtForgeCore + 14 module helpers (scoring, citation gate, sentence split) | yes (phase3, phase7_chat) |
| router.py | 11,511 | InputRouter | yes |
| scaffold.py | 9,901 | ScaffoldBuilder | yes |
| prompt_builder.py | 9,489 | PromptBuilder | yes |
| chat_history.py | 5,414 | ChatMessage, ChatHistory | yes |
| __init__.py | 638 | package docstring only | n/a |

### inference/
| Module | Bytes | Public surface | Direct tests |
|---|---|---|---|
| turboquant.py | 17,848 | GenerationParams, GenerationResult, DraftSet, TurboQuantEngine, create_engine | yes |
| profiles.py | 15,282 | HardwareSnapshot, HardwareDetector, ProfileMatch, ProfileSelector, ResolvedGenerationParams | yes |
| onnx_export.py | 13,285 | ExportResult, OnnxExporter, ONNXEmbedder | yes (inline, phase5) |
| model_browser.py | 10,269 | ModelEntry, ModelBrowser | yes |
| backends.py | 9,493 | BackendInfo, detect/select/build_llama_kwargs + 6 probes | yes |
| hf_backend.py | 6,746 | HuggingFaceBackend | yes |
| turboquant_backend.py | 4,970 | TurboQuantBackend | via unified_backend tests only |
| ollama_backend.py | 4,682 | OllamaBackend | yes |
| lmstudio_backend.py | 3,966 | LMStudioBackend | yes |
| unified_backend.py | 3,644 | GenerationRequest, GenerationResponse, UnifiedBackend, load_backend_from_config | yes |
| __init__.py | 471 | — | n/a |

### knowledge/
| Module | Bytes | Public surface | Direct tests |
|---|---|---|---|
| models.py | 17,834 | 18 record/dataclass types (plan says "14 data structure types" — tree has more; see TF-003) | yes |
| forge.py | 16,320 | KnowledgeForge (hybrid SQL+vector retrieval) | **NO direct test import** — reached only via cognition/core.py; gap logged for TF-016 docket |
| lifecycle.py | 15,738 | MemoryLifecycle | yes |
| store.py | 11,805 | MemoryStore | yes |
| scoring.py | 8,812 | RetrievalDimensions, Prune/Survival/Writeback components + 8 scoring functions | yes |
| __init__.py | 171 | — | n/a |

### etl/
| Module | Bytes | Public surface | Direct tests |
|---|---|---|---|
| sources.py | 21,912 | ConceptNetETL, GeoNamesETL, DBpediaETL, ReferenceDataETL | **no direct test import found** |
| wikidata.py | 14,156 | WikidataETL | **no direct test import found** |
| subset.py | 13,903 | SubsetResult, EdgeSubsetBuilder | yes (inline, phase5) |
| db_integrity.py | 12,888 | IntegrityReport, RepairResult, DBIntegrityChecker | yes |
| embeddings.py | 10,757 | encode/vector-search function set (11 functions) | **no direct test import found** |
| schema.py | 8,571 | get_connection, initialize_schema, rebuild_fts_indexes, get_entity_count | yes |
| __init__.py | 192 | — | n/a |

### refinement/
| Module | Bytes | Public surface | Direct tests |
|---|---|---|---|
| salvage.py | 14,809 | ScoredDraft, SalvageResult, FragmentSalvage | yes |
| enforcement.py | 4,615 | EnforcementResult, EnforcementGate | yes |
| __init__.py | 448 | — | n/a |

### utils/
| Module | Bytes | Public surface | Direct tests |
|---|---|---|---|
| health.py | 17,255 | HealthResult, HealthChecker | yes |
| self_heal.py | 16,476 | SelfHealer, atomic_write | yes |
| retry.py | 8,092 | CircuitState, CircuitBreaker, CircuitOpenError, retry + backoff strategies | yes |
| errors.py | 7,554 | 20-class typed exception hierarchy | yes |
| perf.py | 7,309 | PerfEvent, EventStats, PerfSummary, PerfTracker | yes |
| validators.py | 7,238 | sanitise/validate function set | partial (sanitise_query only) |
| config.py | 2,385 | load_config, load/list hardware profiles | yes |
| paths.py | 2,080 | 9 path-resolution functions | **no direct test import found** |
| logging_setup.py | 1,640 | setup_logging | **no direct test import found** |
| __init__.py | 63 | — | n/a |

## Test estate (presence)

13 test files, 585 `def test_` functions by grep; the runner collects
620 items (parametrisation/class expansion accounts for the difference —
runner count governs, TF-004). Per file: phase1_knowledge 37,
phase2_inference 59, phase3_cognition 75, phase4_refinement 47,
phase5_deployment 74, phase6_adversarial 34, phase6_integration 32,
phase6_release 54, phase7_backends 43, phase7_chat 40,
phase8_errors 35, phase8_health 38, phase8_perf 17.

## Census findings carried forward
1. `knowledge/forge.py` — the retrieval heart — has no direct test
   import; its coverage is indirect through core. → TF-016 docket.
2. ETL ingestion modules (sources, wikidata, embeddings) likewise
   lack direct tests. → TF-016 docket, Part VI owns the fix.
3. utils/paths.py and utils/logging_setup.py untested directly.
4. No module in src/ is empty or a stub: every elder "stub" claim in
   the TASK docs is stale (see TF-003).
