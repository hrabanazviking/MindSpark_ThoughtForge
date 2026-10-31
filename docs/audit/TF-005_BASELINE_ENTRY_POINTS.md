# TF-005 — Fresh Baseline II: Entry Points Actually Run (Part I Audit)

Run 2026-10-30, branch `yrsa/thoughtforge-road-build`, method
`PYTHONPATH=src python3 <entry>` from the repo root, this VM.

## Declared-dependency truth at first contact
pyproject core dependencies vs this VM, at first run:
PRESENT — numpy, pyyaml, platformdirs.
MISSING — sqlalchemy, sentence-transformers, ijson, tqdm, click, rich.
The light missing set (click, tqdm, rich, ijson, sqlalchemy) was then
installed user-local (`pip install --user --break-system-packages`,
pure/light wheels only) so the entry points could be exercised;
sentence-transformers (heavy torch stack) was deliberately NOT
installed — it remains the TF-004 environment gap, docket D-ENV-1.

## Results

| Entry point | Invocation | Result |
|---|---|---|
| `forge_doctor.py` | `python3 forge_doctor.py` | RUNS, exit 0. Reports: knowledge.db not found at `data/knowledge.db` ("empty knowledge base — OK", hint to run `forge_memory.py reference`); Dependencies ✗ "Missing required packages: sentence-transformers". 1 issue found. Truthful on both counts |
| `run_thoughtforge.py` | `--help` | RUNS — full argparse surface (query, --model, --profile, --retrieval, --chat, --history, --system, --backend, --debug) |
| `run_thoughtforge.py` | single query `"What is ThoughtForge?"`, no model, no knowledge DB | RUNS, exit 0, knowledge-only mode. Output: "The forge has no knowledge relevant to this query." Citations: none. Confidence 0.408 [weak]. Enforcement: REVIEW (no knowledge available). Tokens 9. **This is the TF-031 error-honesty behaviour already working in fact: no fabricated answer over an empty store** |
| `forge_memory.py` | `--help` BEFORE light-dep install | FAILED to start: `ModuleNotFoundError: No module named 'click'` (environment, not code — click is a declared pyproject dependency) |
| `forge_memory.py` | `--help` AFTER install | RUNS — commands: all, conceptnet, dbpedia, embeddings, geonames, init, reference, status, wikidata; `--db` option |
| `forge_memory.py` | `--db /tmp/tf005/synthetic.db init` then `status` | OPERATES on a synthetic store: schema initialized successfully (188 KB db); status reports size 0.2 MB and zero counts across entities/statements/labels/embeddings/conceptnet_edges/geonames_places/dbpedia_entities/reference_chunks — correct for a fresh synthetic store. No canon data touched (synthetic path under /tmp only) |

## Verdicts
- All three root instruments are real, working code — none is a stub.
- Every failure seen at this slice traces to this machine's missing
  declared dependencies, none to the entry points' own code.
- Docket carry-overs: D-ENV-1 (sentence-transformers absent → the one
  TF-004 failure + the doctor's one issue; same root, two sightings).
