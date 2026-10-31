# TF-013 — The Doctor's Report (Part I Audit)

Measured 2026-10-30 on branch `yrsa/thoughtforge-road-build`, this VM.
Roadmap wording: "forge_doctor extended (or its gaps listed) to emit
the audit's headline numbers by itself." This slice does both: the
doctor now emits the headlines that can be counted truthfully from
the tree, and the rest are listed here as named gaps with owners.

## What the doctor now emits — `forge_doctor.py --audit`

New `--audit` flag (works in text and `--json` modes; JSON key
`audit_headlines`), backed by `audit_headlines()` in forge_doctor.py
and covered by `tests/test_tf013_doctor_audit.py` (7 tests, including
an independent recount of the source tree). Live output this run:

| Headline | Doctor's live value | Audit source it reproduces |
|---|---|---|
| Source Python files under `src/thoughtforge` | 44 | TF-002 census (44) ✓ |
| Source bytes | 399,944 | TF-002 census (399,944) ✓ exact |
| Test files under `tests/` | 14 (13 at TF-004 baseline + this slice's new file) | TF-004 estate |
| Hardware profiles | 6 — desktop_cpu, desktop_gpu, phone_low, pi_5, pi_zero, server_gpu | TF-012 truth table |
| Config files | default.yaml, personality_core.yaml, user_config.yaml | TF-009 census |
| Benchmark modules | benchmark_profiles.py, persona_consistency.py | TF-010 estate (2 modules) ✓ |
| Audit docs filed under `docs/audit` | 11 before this doc, 12 with it | Part I itself |
| Knowledge DB present | no (sparse campaign tree; empty-base truth, matches the doctor's own degraded check) | TF-005 |

Unchanged behaviour without `--audit`: output and exit codes are
identical to before (proven by the no-flag test). The doctor's health
verdict this run is unchanged from TF-005: 1 issue — the missing
sentence-transformers package (docket D-ENV-1, environment, not code).

## The gaps — headlines the doctor deliberately does NOT emit

| Headline | Why not | Owner |
|---|---|---|
| Test collected / passed / skipped counts (620 / 618 / 1) | Counting tests without running them would be a quoted number, not a measured one; the doctor is a quick non-destructive check and must not run the suite | TF-014 baseline freeze pins the runner method instead |
| Data census totals (484 files / 242,173,121 B, TF-007) | This campaign tree is a sparse checkout that excludes most of `data/` by design; a local count would under-report and contradict TF-007 while looking authoritative | TF-007 stands as the census; doctor emits only knowledge-DB presence |
| Config live/dead classification (64 keys: live / shadowed / dead, TF-009) | Requires tracing code paths per key, not counting files | TF-009 stands; TF-032 consumes its live-key list |
| Benchmark scores (persona 0.883 sample PASS, profile 0.0% metrics, TF-010) | Scores exist only by running the benchmarks | TF-010 stands; Part VII harness owns scores |
| Hardware-profile applied-vs-claimed truth (TF-012) | A verdict from code reading, not a countable headline | TF-012 stands |

Rule recorded for later slices: the doctor counts what a tree can
prove by counting. Anything that needs a run, a trace, or a verdict
belongs to the runner, the audit, or the harness — never to a number
the doctor invents by quoting a doc.
