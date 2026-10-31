# TF-011 — Dead & Duplicate Code Census (Part I Audit)

Measured 2026-10-30 on branch `yrsa/thoughtforge-road-build`.
Method: AST import graph + AST definition census over all 44
`src/thoughtforge/` files, with the consumer corpus = src +
tests + benchmarks + root scripts + the research wing
(`research_data/src/`) — a name counts as *used* if it appears
anywhere in that corpus beyond its own definition. Denominators:
189 top-level definitions in src (91 functions, 98 classes).

## 1. Dead modules: none — but six modules are test-only orphans

Every one of the 44 src modules is imported by at least one other
file; there is no wholly dead module. The sharper truth is the
split between *production* importers (src, root scripts,
benchmarks) and *test-only* existence:

| Module | Only importer(s) | What that means in production |
|---|---|---|
| `knowledge/lifecycle.py` | tests/test_phase1_knowledge.py | **MemoryLifecycle is never wired into the live engine** — core/store never import it, so no fact ever ages, decays, or is pruned at runtime. Part III's lifecycle is a tested island. → D-ORPHAN-1 |
| `utils/retry.py` | tests/test_phase8_errors.py | `CircuitBreaker` / `retry` protect nothing: no backend call in production goes through them. → D-ORPHAN-2 |
| `etl/db_integrity.py` | tests/test_phase8_health.py | `DBIntegrityChecker` is never invoked by forge_doctor or forge_memory — store integrity is checked only in tests. → D-ORPHAN-3 |
| `etl/subset.py` | tests/test_phase5_deployment.py (inline) | `EdgeSubsetBuilder` is never run by any ingestion path; profile-sized subsets are not built in production. → D-ORPHAN-3 |
| `inference/model_browser.py` | tests/test_phase7_backends.py | `ModelBrowser` has no production caller. → D-ORPHAN-3 |
| `inference/onnx_export.py` | tests/test_phase5_deployment.py (inline) | The ONNX export path is exercised only by tests. → D-ORPHAN-3 |

These are not dead code to delete — they are *unwired* code, each
already assigned a wiring verdict in its owning part (lifecycle →
TF-045, retry → TF-073, db_integrity → TF-013, subset → TF-110,
onnx_export → TF-070, model_browser → TF-078's model cards).

## 2. Truly dead functions: 7 of 189 top-level definitions

Each of these occurs exactly once in the entire consumer corpus —
at its own `def` line (grep-verified). No caller in production,
tests, benchmarks, root scripts, or the research wing:

| Function | File:line | Note |
|---|---|---|
| `cosine_similarity` | etl/embeddings.py:84 | The scalar version is dead while its sibling `batch_cosine_similarity` is live (called at embeddings.py:260, :312 by the vector searches). → D-DEAD-2 |
| `get_embeddings_dir` | utils/paths.py:38 | paths.py is imported by 16 files and its other getters are live; this one getter has never been called. → D-DEAD-3 |
| `sanitise_system_prompt` | utils/validators.py:75 | Never called — system prompts pass through no sanitiser. → D-DEAD-1 |
| `validate_config` | utils/validators.py:83 | Never called — configs are loaded (TF-009) but never validated by this, their own validator. → D-DEAD-1 |
| `validate_model_path` | utils/validators.py:136 | Mentioned in the module docstring (line 5), never called anywhere. → D-DEAD-1 |
| `validate_api_key` | utils/validators.py:168 | Never called. → D-DEAD-1 |
| `validate_jsonl_line` | utils/validators.py:194 | Never called — no JSONL ingestion path uses it. → D-DEAD-1 |

**validators.py is the concentration:** of its 6 public functions
only `sanitise_query` is live (core + tests). The other 5 are a
validation layer that exists on paper and guards nothing — the
fix is a wire-or-retire verdict per function in Parts II/V
(TF-032 config surface, TF-094 policy configuration), not a
drive-by deletion. Docket D-DEAD-1 for TF-016.

Public-but-internal-only definitions (used inside their own file,
never imported elsewhere) are **not** dead and are recorded only
to close the question: embeddings' `encode_texts` / `to_blob` /
`from_blob` / `batch_cosine_similarity` (the module's working
interior), db_integrity's `RepairResult`, perf's `PerfEvent`.

## 3. Duplicate code: three helper copies, no wholesale duplication

- **No within-file redefinitions:** zero top-level names defined
  twice in one file, zero duplicate method names within any class
  (AST-verified across all 44 files).
- **Exactly one byte-identical function-body pair** in the whole
  tree by AST-dump hash: `_now_iso` in cognition/chat_history.py:15
  and knowledge/models.py:30 (the same 72-char one-liner).
- **Near-duplicate private helpers** (same logic, trivially
  varied copies — the real duplication pattern here):
  - `_now_iso` ×3 — chat_history.py:15, models.py:30 (identical),
    plus cognition/core.py:707 (same logic, local-import variant).
  - `_split_sentences` ×2 — cognition/core.py:682 and
    refinement/salvage.py:372; identical body, docstring differs
    by three words ("for fragment extraction").
  - `_build_messages` ×2 — inference/ollama_backend.py:133 and
    inference/lmstudio_backend.py:116; identical logic, one copy
    carries the docstring.
  All are private, zero-risk today, and drift-prone tomorrow:
  one shared home (a utils text/time helper) is the TF-025-era
  hygiene fix. → D-DUP-1.
- **Duplicate documents:** `MindSpark_ThoughtForge/` at repo root
  holds byte-identical copies of root `RULES.AI.md` and
  `PHILOSOPHY.md` (md5-verified identical pairs) — a shadow copy
  of the project's own rule/philosophy docs that can silently
  diverge. (The directory also holds a README.md; per standing
  law it was not read or touched — existence noted only.)
  → D-DUP-2, verdict in Part X docs work (TF-182).

## 4. Stub & TODO archaeology: the tree is cleaner than its docs

- **No stub code:** zero pass-only function bodies and zero
  `raise NotImplementedError` in src (AST-verified). TF-002's
  "no module is a stub" now extends to function level.
- **Zero TODO/FIXME/XXX/HACK markers in any Python file** —
  src, tests, benchmarks, and root scripts, grep-verified (0
  hits). The code carries no inline debt markers at all.
- **`TODO.md` (root, 4 lines) is the only TODO artefact**, and
  its items are pre-build session instructions, every one
  overtaken in fact: "investigate all the files" (Part I is that
  investigation), "write a staged multi-step plan and discuss it
  with Volmarr" (BUILD_PLAN_v1, then this roadmap), "add Sovereign
  Local RAG" (built; censused in TF-002/TF-003). It is a museum
  piece, not a backlog.
- The only other "TODO" hits in the repo's markdown are prose in
  `RULES.AI.md` ("Read the TODO.md file every new session", a
  rule *about* TODOs) and this roadmap's own TF-011 line.
- The "stub" mentions in TASK_PHASE3_COGNITION.md:24 and
  TASK_PHASE4_SALVAGE.md:24 are checked-off `[x]` checklist items
  describing the *starting* state those phases began from —
  historical, not current claims (consistent with TF-003).

## 5. Verdict for the road

- Nothing here blocks Part II: no dead module, no stub, no
  duplicated engine. The debt is concentrated and named:
  one unwired lifecycle (Part III's to wire), one unused
  protection layer (retry, Part IV's), four more test-only
  modules with wiring verdicts already roaded, 7 dead functions
  (5 of them one module's paper validation layer), and 3 copied
  private helpers.
- Docket for TF-016: D-DEAD-1 (validators ×5 — wire-or-retire in
  TF-032/TF-094), D-DEAD-2 (scalar cosine_similarity — TF-053),
  D-DEAD-3 (get_embeddings_dir — TF-053/TF-110 era), D-ORPHAN-1
  (lifecycle unwired — TF-045), D-ORPHAN-2 (retry unwired —
  TF-073), D-ORPHAN-3 (db_integrity / subset / model_browser /
  onnx_export test-only — TF-013 / TF-110 / TF-078 / TF-070),
  D-DUP-1 (helper copies ×3 names — TF-025), D-DUP-2 (shadow
  docs directory — TF-182).

Suite at banking: 618 passed, 1 skipped, 1 known env failure
(D-ENV-1) under the TF-004 method — baseline held, no new failures.
