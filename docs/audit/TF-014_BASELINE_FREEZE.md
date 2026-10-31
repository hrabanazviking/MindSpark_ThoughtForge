# TF-014 — Baseline Freeze: the Campaign's Zero Line (Part I Audit)

Frozen 2026-10-30 on branch `yrsa/thoughtforge-road-build` @
e35a3ab (TF-013 banked). This is the pinned zero line every later
slice is measured against. Re-measure identically at any time with
`sh scripts/rebaseline.sh` (added by this slice), or by hand with
the method below.

## The method (identical for every re-measurement)

- Machine: this VM (7.7 GB RAM, x86_64 — auto-selects the
  `desktop_cpu` hardware profile, TF-012). Label: **VM profile**.
- Interpreter: Python 3.12.3 (`/usr/bin/python3`), pytest 9.1.1.
- The package is NOT pip-installed; `PYTHONPATH=src` is the method.
- Suite command: `PYTHONPATH=src python3 -m pytest tests/ -q`
  (pyproject supplies `testpaths=["tests"]`).
- Entry points run from the repo root as
  `PYTHONPATH=src python3 <entry>`; forge_memory only ever against a
  synthetic store under /tmp — never a real or canon store.

## The frozen numbers

### Suite (TF-004 baseline → TF-014 freeze)

| | TF-004 (base tree) | TF-014 freeze (this tree) |
|---|---|---|
| Collected | 620 | **627** |
| Passed | 618 | **625** |
| Skipped | 1 | **1** |
| Failed | 1 (environment) | **1 (environment, same test, same cause)** |
| Warnings | 6 | 6 |

The +7 collected / +7 passed delta is exactly TF-013's new test file
(`tests/test_tf013_doctor_audit.py`, 7 tests) — no other drift.
The one failure is unchanged: `tests/test_phase8_health.py::
TestCheckDependencies::test_yaml_present`, failing because
sentence-transformers is not installed on this VM (docket D-ENV-1).
The one skip is unchanged (tests/test_phase5_deployment.py).

### Entry points (TF-005 → TF-014, re-run this slice)

| Entry point | Frozen result |
|---|---|
| `forge_doctor.py` | Runs; exit 1 solely for D-ENV-1 ("Missing required packages: sentence-transformers"); knowledge.db absent reported degraded ("empty knowledge base — OK"). With `--audit` it now also emits the TF-013 headlines (44 files / 399,944 B) |
| `run_thoughtforge.py --help` | Exit 0, full argparse surface |
| `run_thoughtforge.py "What is ThoughtForge?"` (no model, no DB) | Exit 0, knowledge-only: "The forge has no knowledge relevant to this query." Citations none, Confidence 0.408 [weak], Enforcement REVIEW, Tokens 9 — byte-identical verdicts to TF-005 |
| `forge_memory.py --help` | Exit 0 (click present, user-local install from TF-005 stands) |
| `forge_memory.py --db /tmp/…/synthetic.db init` + `status` | Init succeeds (0.2 MB); status reports zero counts across all 8 tables — identical to TF-005 |

### Environment note (part of the freeze, not a footnote)

Present: numpy, pyyaml, platformdirs, pytest + the TF-005 light set
(click, tqdm, rich, ijson, sqlalchemy, user-local). Absent by
deliberate choice: sentence-transformers and its torch stack
(D-ENV-1). A re-measurement on a machine WITH that stack installed
should see 626 passed / 0 failed at this tree — and must say which
machine and stack it used, or the numbers do not compare.

## The green rule (binding on every later slice)

A slice banks "full suite green" when, measured by this method:
**no new failure** beyond the frozen D-ENV-1 failure, **no lost
pass**, and the collected count moves only by the slice's own added
tests. Any other movement is a regression the slice must fix or
name before banking — never leave a discovered bug unfixed.
