# TF-015 — Audit Verdicts on the Elder Docs (Part I Audit)

Measured in-tree 2026-10-30, branch `yrsa/thoughtforge-road-build`.
This document records the verdicts. **No elder doc is edited by
this slice** — BUILD_PLAN_v1 and the TASK_PHASE files keep their
bytes exactly; their supersession lives here and in the roadmap,
which is where a reader arriving after 2026-10-30 must look first.

## The headline verdict

**BUILD_PLAN_v1 is SUPERSEDED** — as a *plan*. It is retained,
unedited, as the project's founding vision document and historical
record. The governing plan from 2026-10-30 forward is
`docs/MINDSPARK_THOUGHTFORGE_ROADMAP_2026-10-30.md` (196 slices,
TF-001–TF-196, Parts I–X), commissioned by Volmarr 2026-10-30.

Why superseded, in one paragraph of evidence: the plan's own
status lines ("Current Phase: Phase 0 (not yet started)"; every
phase "NOT STARTED") are stale in every row (TF-003: Phases 0–4
DONE-in-fact, 5–6 PARTIAL). Its counts are stale (it promises 14
data structure types; `knowledge/models.py` carries 18 — the tree
grew past the plan). One of its roads is closed by standing law
(Phase 5's Docker images — Volmarr's no-Docker law; the artefacts
exist in-tree but are never used by this campaign). And its
release claims are precisely what was never proven from the tree
(TF-003/TF-010; release truth is owned by Part X, TF-185/TF-196).

## The concordance — the vision is preserved, not discarded

Roadmap §1 carries the plan's vision forward verbatim in substance.
Nothing in this verdict retires an idea; it retires a *schedule*.

| BUILD_PLAN_v1 element | Where it lives now |
|---|---|
| "It is not a model. It is the forge that makes any model sharper." | Roadmap §1, quoted as the project's soul |
| Pillar 1 — Sovereign Local RAG | Roadmap Pillar 1; built/proven in Part III (TF-041–TF-064) + Part VI (TF-105–TF-120) |
| Pillar 2 — TurboQuant Inference | Roadmap Pillar 2; Part IV (TF-065–TF-086) |
| Pillar 3 — Cognition Scaffolds | Roadmap Pillar 3; Part II (TF-019–TF-040) |
| Pillar 4 — Fragment Salvage | Roadmap Pillar 4; Part V (TF-087–TF-104) |
| Pillar 5 — Memory-Enforced Cognition (cite-or-explain) | Roadmap Pillar 5; TF-048/TF-049 + TF-093, gated at TF-064 |
| 6 hardware profiles `phone_low` → `server_gpu` | Roadmap §1 hardware truth; measured truth in TF-012, re-issued TF-086 |
| BP Phase 0 — Foundation | DONE-in-fact (TF-003); no successor slices needed |
| BP Phase 1 — Knowledge | Code DONE-in-fact; proof owned by Parts III/VI |
| BP Phase 2 — TurboQuant | Code DONE-in-fact; proof owned by Part IV |
| BP Phase 3 — Cognition | Code DONE-in-fact; proof/hardening owned by Part II |
| BP Phase 4 — Salvage | Code DONE-in-fact; proof owned by Part V |
| BP Phase 5 — Deployment | PARTIAL; Docker road SUPERSEDED by law; native roads owned by Parts IV/X (TF-082, TF-173–TF-175) |
| BP Phase 6 — Testing/Release | Suite DONE-in-fact (TF-014 zero line); release truth owned by Part X |
| "No internet dependency at runtime" (sovereignty) | Campaign law, roadmap §3; proven at TF-117/TF-172 |
| Full-dump Wikidata ambition | Not re-proven this campaign; ETL hardening owned by TF-107; no claim carried forward until measured |

## Verdicts on the TASK_PHASE docs (elder task files)

All seven present files are **HISTORICAL — superseded as task
drivers, retained as the record of how the tree was built**. Their
internal status lines are not corrected in place; the corrections
of record are TF-003's matrix and this doc. TASK_PHASE1 and
TASK_PHASE2 are **ABSENT from the tree** (git ls-tree, TF-003) —
their work exists as code; the missing docs are a gap in the
record, not in the software, and no reconstruction is commissioned
(inventing them retroactively would falsify the record).

| Doc | Its own claim | Verdict of record |
|---|---|---|
| TASK_PHASE0_SETUP | "READY TO START… no code written yet" | STALE — foundation DONE-in-fact (TF-003) |
| TASK_PHASE3_COGNITION | "COMPLETE — commit 7fb8a60"; baseline claims cognition/ + refinement/ are "empty stubs" | COMPLETE claim holds in fact (packages are real code); its baseline stub claim is SUPERSEDED by the very work it commissioned |
| TASK_PHASE4_SALVAGE | "IN PROGRESS"; "refinement/__init__.py stub only" | STALE — refinement is fully implemented (TF-003); the phase is DONE-in-fact |
| TASK_PHASE5_DEPLOYMENT | "IN PROGRESS"; deliverables ✅ incl. Dockerfile/compose | DONE-in-fact except the Docker road, SUPERSEDED by the no-Docker law |
| TASK_PHASE6_RELEASE | "IN PROGRESS (FINAL)" | PARTIAL — suite/benchmarks/docs real; v1.0 release + HF cards NOT PROVEN (Part X owns) |
| TASK_PHASE7_SETUP_WIZARD | "COMPLETE ✅" | DONE-in-fact as code (TF-003: wizard, backends, chat all present) |
| TASK_PHASE8_ROBUSTNESS | "COMPLETE — 620 tests passing, tagged v1.2.0" | DONE-in-fact as code; its "620 passing" count is SUPERSEDED as a number by the TF-014 freeze (627 collected / 625 passed / 1 skipped / 1 environment failure, from the runner) — the count grew because the tree kept being true, not because Phase 8 was wrong |

## Verdict on the spec library (`docs/specs/`, 27 files)

**RETAINED AS DESIGN REFERENCE — not governing plan.** The specs
are the elder design sources the code was built from; where a spec
and the shipped code disagree, the code + the Part I audit are the
truth of record and the roadmap decides what changes. One concrete
drift, measured this slice: BUILD_PLAN_v1's "Key Spec Documents"
table names 14 files — 8 exist in `docs/specs/` under the exact
name given, 2 differ only by case (`ThoughtForge_Complete_Module_
Library.md`, `Production_Ready_Implementation_Package.md`), and 4
stand under drifted names (the plan's `The_TurboQuant_Cognition_
Blueprint.md` → `TurboQuant_Cognition_Blueprint.md`;
`Detailed_Wikidata_ETL_Pipeline.md` → `Wikidata_ETL_Pipeline.md`;
`MindSpark_Master_Game_Plan.md` → `Master_Game_Plan_Roadmap.md`;
`Sovereign_Local_RAG_Architecture.md` → the subject is carried by
`Sovereign_RAG_Technical_Overview.md` / `Sovereign_RAG_Brainstorming.md`).
All 14 subjects are present; the plan's filenames are not reliable
pointers. No spec file is renamed by this verdict — renaming elder
sources to match a stale table would break every other reference.

## Verdicts on the remaining elder prose

- `CHANGELOG.md` — a release record, not a plan; its Phase-8 count
  is superseded as a number by TF-014 (above). Historical entries
  stand as written.
- `TODO.md` — 4-line museum piece (TF-011): every item overtaken in
  fact (the staged build plan it asks for exists twice over —
  BUILD_PLAN_v1 and now the roadmap; the Sovereign Local RAG
  planning it asks for is the shipped knowledge layer). Verdict:
  HISTORICAL; its asks are answered, not pending.
- `docs/api.md`, `docs/quickstart.md`, `docs/hardware_profiles.md`,
  `docs/index.md` — user-facing elder docs; not plans, so not
  superseded by this verdict. Their factual claims are audited
  later where the roadmap assigns it (hardware truth TF-012/TF-086;
  public-claims audit TF-136/TF-184; user docs TF-179–TF-182).

## What a new reader should do (the point of this doc)

1. Read the roadmap (`docs/MINDSPARK_THOUGHTFORGE_ROADMAP_2026-10-30.md`) — it governs.
2. Read BUILD_PLAN_v1 for the founding vision and design decisions — and nothing else; its statuses, counts, and file pointers are stale where this doc and TF-003 say so.
3. Read the TASK_PHASE files and specs as history and design sources.
4. Trust numbers only from the TF-014 zero line and later runner output.
