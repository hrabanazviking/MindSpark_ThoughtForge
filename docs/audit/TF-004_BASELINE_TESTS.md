# TF-004 — Fresh Baseline I: Test Suite (Part I Audit)

Run 2026-10-30 on the untouched base tree (branch
`yrsa/thoughtforge-road-build` @ e7d21aa, i.e. development @ d267148 +
the TF-001 filing commit only — no campaign code changes yet).

## Method (reproduce identically)
- Interpreter: Python 3.12.3 (`/usr/bin/python3`), pytest 9.1.1.
- Command: `PYTHONPATH=src python3 -m pytest tests/ -q`
  (pyproject.toml supplies `testpaths=["tests"]`, addopts `-v --tb=short`;
  the package was not pip-installed — PYTHONPATH=src is the method).
- Duration: 132.38 s.

## Runner counts (the only counts that govern)
- **Collected: 620**
- **Passed: 618**
- **Skipped: 1** (in tests/test_phase5_deployment.py, per progress output)
- **Failed: 1**
- Warnings: 6 (PytestRemovedIn10Warning — class-scoped fixture defined
  as instance method, in tests/test_phase6_release.py; deprecation
  only, no test affected)

Per-file progress was clean dots for every file except
test_phase8_health.py (one F) and test_phase5_deployment.py (one s).

## The one failure — classified ENVIRONMENT, not code
`tests/test_phase8_health.py::TestCheckDependencies::test_yaml_present`
- The test asserts the Dependencies health check is `ok` or `degraded`.
- The checker returned status `fail` with message:
  "Missing required packages: sentence-transformers" and fix hint
  "Run: pip install sentence-transformers".
- Ground truth on this VM at run time: `pip list` shows no
  sentence-transformers (and its heavy torch stack) installed; PyYAML,
  numpy, platformdirs, pytest are present.
- Verdict: the health checker is working correctly — it truthfully
  reported a genuinely missing optional/heavy dependency of this
  machine. The test's name (test_yaml_present) does not match what it
  asserts (the whole dependency set); that naming mismatch is a
  stale-test smell recorded for TF-016, not a failure cause.
- Disposition: NOT fixed by installing the heavy stack in this tick
  (multi-GB torch pull on the VM, against machine-load etiquette) and
  NOT fixed by weakening the test. Recorded in the TF-016 docket as
  D-ENV-1 with the honest fix paths: (a) install the full dependency
  set on the test machine, or (b) a later slice makes the dependency
  check's required/optional split match pyproject (sentence-transformers
  is a core dependency in pyproject, so (a) is the truthful road for a
  full baseline; a VM-profile baseline without it must be labelled).

## Baseline statement
Zero line for the campaign: **618 passed / 1 skipped / 1 failed
(environment) / 620 collected** on this VM profile. Any later slice's
"full suite green" claim is measured against this line plus the
environment note: a slice is green when no NEW failure appears and
the known environment failure is either unchanged or resolved by
installing the dependency set.
