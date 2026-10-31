# TF-018 — Part I Gate: the Audit Stands, the Road Holds (Part I Gate)

Closed 2026-10-30 on branch `yrsa/thoughtforge-road-build`.
Roadmap wording: "Part I gate: audit doc published in-repo;
Volmarr-facing summary in plain language; the road below re-cut
if the facts demanded it (changes recorded here, with reasons)."

## The gate verdict

**Part I is COMPLETE. All 17 audit slices (TF-001–TF-017) are
banked with evidence in `docs/audit/`, the suite stands exactly
on its frozen zero line plus the slices' own added tests, and
no fact found in Part I demands a re-cut of Parts II–X.** The
one correction Part I made to its own assignments (the
db_integrity wiring owner, TF-013 → TF-167) is recorded in
TF-016, not here, because TF-016 caught it first.

## Plain-language summary (the Volmarr-facing half of this gate)

Here is what the audit found, in plain words:

- **The forge is real, and further along than its own plan
  says.** The old plan claimed the work had barely started.
  In fact the thinking engine, the memory store, the ingestion
  pipeline, the output-repair tools, and the hardware profiles
  all exist as working, tested code. The plan was stale; the
  tree is the truth, and now the tree has been measured.
- **The tests are healthy.** 637 tests are collected; 635 pass,
  1 is skipped by design, and exactly 1 fails — not because the
  code is broken, but because one heavy optional library
  (sentence-transformers) is deliberately not installed on this
  machine. That single failure is named, tracked (D-ENV-1), and
  is the same one we started with. Nothing new broke anywhere
  in Part I.
- **Every defect found is on a numbered list with an owner.**
  The docket (TF-016) holds 46 entries. None is vague and none
  is orphaned: each names the exact later slice that fixes it.
  One defect — the build workflow firing automatically, against
  the house rule that workflows run only when called by hand —
  was already fixed during the audit itself.
- **The honest gaps, named plainly:** the hardware profiles
  promise budgets (memory limits, retrieval sizes) that no code
  currently enforces — the engine runs the same flat settings on
  a phone-class profile as on a server. The settings files are
  mostly decoration: most keys in them are ignored or shadowed
  by hard-coded values. The benchmarks are an estate, not a
  proof system — they run, but they measure an empty store and
  prove almost nothing yet. And the young research wing inside
  the repo (wyrdforge) cannot even be imported: a whole
  subpackage of it was never committed. Each of these has its
  fixing slices already roaded in Parts II–VIII.
- **Your private material stayed private.** The data census
  counted files and sizes only; no canon content was copied
  anywhere, and the new test fixtures are invented stand-ins
  (a made-up village, Lantern Hollow) guarded by tests that
  fail if any test ever reaches for the real data.
- **One thing will need your word later, not now:** the
  project's own licence contradicts itself (TF-008 / D-LIC-1).
  Choosing the licence is yours; it is roaded to TF-190 in
  Part X, and nothing in Parts II–IX waits on it.

## The evidence, slice by slice

| Slice | Artefact | Headline |
|---|---|---|
| TF-001 | Campaign opening | Sparse clone, branch, roadmap filed in `docs/` and pushed (filing only, `e7d21aa`, ls-remote verified) |
| TF-002 | `TF-002_MODULE_CENSUS.md` | 44 src files / 399,944 B; coverage gaps named |
| TF-003 | `TF-003_PLAN_VS_TREE_MATRIX.md` | Every BP phase DONE-in-fact or PARTIAL; every plan status line stale |
| TF-004 | `TF-004_BASELINE_TESTS.md` | Base baseline: 620 collected / 618 passed / 1 skipped / 1 env failure |
| TF-005 | `TF-005_BASELINE_ENTRY_POINTS.md` | All 3 root instruments run and honest |
| TF-006 | `TF-006_RESEARCH_WING_CENSUS.md` | wyrdforge wing cannot import — `wyrdforge.models` never committed (D-WING-1) |
| TF-007 | `TF-007_DATA_CENSUS.md` | data/ = 484 files / 242,173,121 B, metadata only; duplication + provenance gaps named |
| TF-008 | `TF-008_DEPENDENCY_LICENCE_AUDIT.md` | All deps permissive; project's own licence self-contradicts (D-LIC-1); `requests` undeclared (D-DEP-1) |
| TF-009 | `TF-009_CONFIG_CENSUS.md` | 64 config leaf keys traced: default.yaml mostly shadowed (D-CFG-1); persona fields dormant (D-CFG-2) |
| TF-010 | `TF-010_BENCHMARK_ESTATE.md` | 2 modules, 0 suites; profile benchmark measures an empty store (D-BENCH-1) |
| TF-011 | `TF-011_DEAD_DUPLICATE_CENSUS.md` | No dead modules; 6 test-only modules, 7 dead functions, each with a wiring owner |
| TF-012 | `TF-012_HARDWARE_PROFILE_TRUTH_TABLE.md` | Profiles govern only TurboQuant; the cognition core is profile-blind (D-PROF-3) |
| TF-013 | `TF-013_DOCTORS_REPORT.md` + code | `forge_doctor --audit` emits Part I headlines live; +7 tests |
| TF-014 | `TF-014_BASELINE_FREEZE.md` + `scripts/rebaseline.sh` | Zero line frozen: 627 / 625 / 1 / 1 at tree e35a3ab |
| TF-015 | `TF-015_ELDER_DOC_VERDICTS.md` | BUILD_PLAN_v1 SUPERSEDED as plan (bytes untouched); TASK_PHASE docs historical |
| TF-016 | `TF-016_BUG_AND_GAP_DOCKET.md` | 46 entries: 43 OPEN, 1 OPEN-ENVIRONMENT, 1 FIXED, 1 CLOSED-BY-VERDICT — every one owned |
| TF-017 | `TF-017_FIXTURE_POLICY.md` + `tests/fixtures/` | Synthetic stand-ins + guard tests; canon name purged from a phase8 test; +10 tests |

## Road re-cut assessment (the demanded check, done explicitly)

Each Part I surprise was tested against the roadmap's Parts
II–X as written:

1. **wyrdforge cannot import (D-WING-1).** Part VIII already
   roads verdicts before adoptions (TF-139–TF-142), budgets
   adoption work at TF-143, and test discipline at TF-149;
   TF-016 records that any ADOPT verdict must budget rebuilding
   the models from the wing's 14 JSON schemas. **No re-cut.**
2. **Profiles are unenforced and the core is profile-blind
   (D-PROF-1…4).** Already the stated work of TF-032/TF-033/
   TF-037 (Part II), TF-056 (Part III), TF-069/TF-086 (Part IV),
   TF-098 (Part V). **No re-cut.**
3. **Config mirage (D-CFG-1…5).** TF-009's exact live-key list
   was recorded *for* TF-032, and TF-044/TF-065/TF-153/TF-160
   own the rest, as roaded. **No re-cut.**
4. **Benchmark estate is scripts, not suites (D-BENCH-1…8).**
   Part VII already starts the harness from zero (TF-121) and
   roads every named defect. **No re-cut.**
5. **Broken console scripts (D-DEP-2), undeclared deps
   (D-DEP-1/5), dead weight (D-DEP-3).** Owned by TF-065 and
   TF-153 as roaded. **No re-cut.**
6. **Licence self-contradiction (D-LIC-1) and unverified
   upstream licences (D-LIC-2).** Owned by TF-106/TF-053 and
   closed at TF-190, which is explicitly gated on Volmarr's
   word. **No re-cut; one future decision for Volmarr, named
   in the summary above.**
7. **Data duplication and non-ingestible canon (D-DATA-1…4).**
   Owned by TF-109/TF-118/TF-058/TF-119 as roaded; canon itself
   is never moved or deleted by the campaign. **No re-cut.**
8. **D-ENV-1 (the one suite failure).** An environment fact,
   not a code defect; roaded to the final baseline book
   (TF-194). **No re-cut.**

**Conclusion: the road below stands exactly as carved.** Part I
changed the campaign's knowledge, not its order — which is
what Part I was for.

## Suite at the gate (measured this slice, TF-014 method)

`PYTHONPATH=src python3 -m pytest tests/ -q` on this VM:
**637 collected / 635 passed / 1 skipped / 1 failed
(D-ENV-1, unchanged)** — exactly the TF-014 zero line
(627 / 625 / 1 / 1) plus TF-013's +7 and TF-017's +10 tests,
all passing. This slice adds one gate doc and no code or
tests, so the count does not move at banking.

## What opens next

Part II — The Cognition Core (TF-019–TF-040), starting with
TF-019: the core loop mapped from the code and published as
the architecture doc.
