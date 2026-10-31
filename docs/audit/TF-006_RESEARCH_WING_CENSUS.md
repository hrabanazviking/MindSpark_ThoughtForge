# TF-006 — Research Wing Census: research_data/src/wyrdforge (Part I Audit)

Measured in-tree 2026-10-30 (git ls-tree against HEAD + checked-out
files). `research_data/` holds 96 files in total: ~50 design/research
markdown documents (architecture, memory patterns, persona theory,
MicroRAG / PersonaCompiler / BondGraph / TruthCalibration specs and
eval sets), a pyproject.toml, and the `src/wyrdforge/` code wing.

## The code wing as committed

| Area | Files | Bytes / contents |
|---|---|---|
| services/ | 5 | bond_graph_service.py 2,797 · memory_store.py 2,729 · micro_rag_pipeline.py 4,198 · persona_compiler.py 4,568 · truth_calibrator.py 3,157 (17,449 B total) |
| security/ | 2 | permission_guard.py 843 · prompt_injection_guard.py 365 |
| runtime/ | 1 | demo_seed.py 1,679 |
| schemas/ | 14 | JSON schemas: bond_edge, canonical_fact_record, contradiction_record, episode_summary_record, eval_case, eval_result, hurt, micro_context_packet, observation_record, persona_packet, policy_record, retrieval_item, symbolic_trace_record, vow |
| __init__.py | 1 | re-exports 10 names from `wyrdforge.models.*` |
| tests (research_data/tests/) | 2 | test_memory_and_persona.py (2 test fns, 1,139 B), test_micro_rag_and_truth.py (2 test fns, 2,173 B) |

## The headline finding — the wing cannot import

**The entire `wyrdforge.models` subpackage is missing from the tree.**
`git ls-tree -r HEAD research_data/src/wyrdforge` lists 23 files and
no `models/` entry. Yet:
- `wyrdforge/__init__.py` imports MemoryRecord, ObservationRecord,
  CanonicalFactRecord, BondEdge, Vow, Hurt, PersonaPacket,
  MicroContextPacket, EvalCase, EvalResult — all from `.models.*`.
- Every service imports from `wyrdforge.models.*` (bond, memory,
  common, micro_rag, persona, evals), as does runtime/demo_seed.py.

Runner proof (TF-004 method, inside research_data/):
`PYTHONPATH=src python3 -m pytest tests/ -q` →
**2 collection errors, 0 tests run** —
`ModuleNotFoundError: No module named 'wyrdforge.models'`.

So the roadmap §2 description ("a research prototype… with its own
tests") is true about intent and false about runnable state: as
committed, the wing is a set of service shells + JSON schemas whose
data-model layer was never committed (single upload commit 6ba2df7
in the wing's history). This is a REAL defect class, not environment:
→ TF-016 docket as D-WING-1.

## Overlap / uniqueness vs the shipping tree (for Part VIII)

| Wyrdforge invention | Shipping-tree counterpart | Uniqueness |
|---|---|---|
| bond graph (services/bond_graph_service.py + bond_edge/vow/hurt schemas) | none — knowledge/ has no bond/relationship graph | UNIQUE concept; no shipping equivalent |
| micro-RAG (micro_rag_pipeline.py + micro_context_packet schema) | knowledge/forge.py hybrid retrieval | OVERLAP in purpose, different shape (packet-based, small-model-first) |
| persona compiler (persona_compiler.py + persona_packet schema) | configs/personality_core.yaml + cognition prompt/scaffold path | OVERLAP in purpose, compiler shape is unique |
| truth calibrator (truth_calibrator.py + eval_case/eval_result schemas) | refinement/enforcement.py EnforcementGate | OVERLAP in purpose; eval-scoring shape is unique |
| security guards (permission + prompt-injection) | utils/validators.py sanitisation | PARTIAL overlap, much smaller |
| schemas (14 JSON schemas) | knowledge/models.py dataclasses (18 types) | The schemas are the wing's most complete artefact — they fully specify the missing models' shapes |

## Verdict input for Part VIII
- Nothing in the wing is adoptable *as code* today: it does not run.
- What IS salvageable as evidence: the 14 JSON schemas + the design
  docs + the service shells' interfaces. A TF-139–142 verdict that
  says ADOPT must budget rebuilding `wyrdforge.models` from the
  schemas first; a RETIRE verdict must still harvest the bond-graph
  concept, which has no shipping counterpart.
- The wing's 4 tests cannot pass or fail — they error at collection.
  Their state is recorded here so TF-149/TF-151 measure from truth.
