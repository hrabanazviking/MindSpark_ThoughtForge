# TF-017 — Fixture Policy: Synthetic Canon Stand-Ins (Part I Audit)

Measured/built 2026-10-30 on branch `yrsa/thoughtforge-road-build`,
this VM. Roadmap wording: "Fixture policy: synthetic canon stand-ins
created for tests so no test ever needs real data."

## The policy (binding on every later slice)

1. **No test ever needs real data.** Canon — the repo's `data/` tree,
   its persona material, and its `knowledge_reference` library —
   is never read, opened, copied, or quoted by a test. Metadata
   censuses (TF-007) are audit work, not test input.
2. **Stand-ins live under `tests/fixtures/` only**, are invented
   whole, and carry the `SYNTHETIC_FIXTURE` marker (or the
   `"synthetic_fixture": true` field in structured files). A fixture
   must never live under `data/` or masquerade as canon.
3. **Stand-ins mirror the real shapes** — persona core YAML, reference
   documents in the ETL's supported formats (.md/.jsonl/.yaml),
   memory records through the real record classes — so code paths
   are exercised honestly, against known content with known right
   answers.
4. **The guards are tests, not intentions** (`tests/test_tf017_fixture_policy.py`):
   the estate is scanned every run for repo-root-anchored access to
   `data/`, for the canon path helpers (`get_data_dir()`,
   `get_knowledge_reference_dir()`, `get_knowledge_db_path()`), for
   `knowledge_reference` / `pictures_of_` references, and for canon
   persona names used as test data.
5. Not canon access, by verdict: asserting the string `data/` inside
   a `.dockerignore`-content test (TF estate, pre-existing), writing
   the word `data` as tmp-file content, and creating an *empty* tmp
   directory named `data` to probe a presence check (TF-013's doctor
   test). None of these read the repo's canon tree; the guard
   patterns are shaped to the read, not the word.

Public-domain mythology terms in generic tone tests (e.g. the
phase-6 adversarial set) are not canon data and are untouched by
this policy. Canon means Volmarr's datasets and persona material.

## What was created

The invented world: **Lantern Hollow** on the River Sarn, kept by
**Wrenna the Lantern-Keeper** — a place, person, and customs that
exist nowhere but these fixtures.

| Stand-in | Shape it proves |
|---|---|
| `tests/fixtures/synthetic_personality_core.yaml` | Personality core YAML in `configs/personality_core.yaml`'s exact field shape; loads via `PersonalityCoreRecord.from_dict` |
| `tests/fixtures/synthetic_reference/synthetic_lantern_hollow.md` | Reference markdown for `ReferenceDataETL` |
| `tests/fixtures/synthetic_reference/synthetic_facts.jsonl` | Reference JSONL, each line flagged `synthetic_fixture: true` |
| `tests/fixtures/synthetic_reference/synthetic_places.yaml` | Reference YAML |
| `tests/fixtures/synthetic.py` | Loader/builders: `load_synthetic_personality()`, `synthetic_reference_text()`, `build_synthetic_store(dir)` (persona + preference + fact + episode, all synthetic), `SYNTHETIC_TURNS` (4 invented Q/A turns) |

Proven by `tests/test_tf017_fixture_policy.py` (10 tests): the
persona loads through the real shape; the reference directory is
ingested by the real `ReferenceDataETL` into a tmp DB (3 files,
≥3 chunks, 0 skipped); the synthetic store round-trips through the
real `MemoryStore` with the known right answers (ferry crossing =
11 minutes); the guards hold over the whole test estate.

## Discovered defect, fixed in scope

- `tests/test_phase8_health.py::test_valid_yaml_is_ok` used the
  canon persona name ("Sigrid", with an age) as sample YAML content
  — exactly the habit this policy exists to end. Replaced with the
  synthetic stand-in's name (Wrenna). The guard test now fails any
  future return of that name as test data. Behaviour of the health
  test is unchanged (valid YAML in a tmp dir → ok).

## Suite at banking

Full suite measured with the TF-014 method (`PYTHONPATH=src
python3 -m pytest tests/ -q`, run as two file-halves whose union is
all 15 test files): **637 collected / 635 passed / 1 skipped /
1 failed (D-ENV-1, unchanged)** — exactly the TF-014 zero line
(627 / 625 / 1 / 1) plus this slice's +10 tests, all passing.
