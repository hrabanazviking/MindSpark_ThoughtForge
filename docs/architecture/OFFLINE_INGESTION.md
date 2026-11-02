# ThoughtForge Architecture — Offline Ingestion (TF-117)

Offline ingestion: everything except the fetch itself
works with the network cut (sovereignty, proven).
Part VI's thirteenth slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`src/thoughtforge/etl/offline_ingestion.py`
(`FETCH_STEPS`, `OFFLINE_STEPS`, `OFFLINE_RULES`,
`NetworkCut`, `run_offline_pipeline()`, checked by
`offline_problems()` and `offline_live_problems()`);
the embedding loader in `etl/embeddings.py` gained its
local-files-only guard in the open, and TF-105's living
map gained its `offline` orchestration entry in the
open. The tests are `tests/test_tf117_offline_ingestion.py`.
This slice **adds and fixes D-OFFLINE-1, D-OFFLINE-2
and D-OFFLINE-3**.

## 1. Measured start

Probed live this tick, before any change:

- **Offline was an assumption, never a proof
  (D-OFFLINE-1).** The five builders take local file
  paths, so the pipeline *looked* offline — but no
  module, test or checker had ever run ingestion with
  the network cut. No socket was blocked, no attempt
  recorded. Sovereignty was a reading of signatures,
  not a measurement.
- **The embedding load was an unguarded fetch
  (D-OFFLINE-2).** `etl/embeddings._get_model`
  constructed `SentenceTransformer(name)` with no
  `local_files_only` and no use of TF-053's cache
  proof. On a machine with the package but no cached
  model, the embedding step of an otherwise local
  build (`forge_memory all` step 6, the `embeddings`
  command, TF-111's default encoder) silently became
  a Hugging Face fetch — and offline it failed with a
  hub/network error, not a named local verdict.
- **The fetch boundary was undefined (D-OFFLINE-3).**
  Nothing filed which operations ARE the fetch:
  source downloads existed only as docstring URLs in
  `etl/sources.py` (no downloader exists in the
  tree), model download existed only as D-OFFLINE-2's
  hidden side effect, and package installation was
  named nowhere as a separate act. "Except the fetch
  itself" named a thing nobody had drawn.

The probe that grounded the fix: with socket connect,
`create_connection`, `getaddrinfo` and urllib
`urlopen` all hard-blocked, TF-105's synthetic
pipeline (all five builders, cached-fake embeddings,
subset) completed with the filed counts — the
builders really are local; only the loader needed
the guard.

## 2. The boundary and the cut

- **The fetch boundary, as data.** `FETCH_STEPS`
  files exactly three fetches, all outside ingestion:
  downloading a source file (by hand; builders take
  local paths, docstrings carry the URLs, no
  downloader exists in `etl/`), downloading an
  embedding model into a local cache (an explicit act
  before ingestion), and installing packages (pip, at
  setup). No ingestion code performs any of them.
  `OFFLINE_STEPS` files every other Part VI operation
  — schema, the five builders (bound to TF-105's map
  by id), cached/injected embeddings, subset, FTS
  rebuild, and the dry-run / metrics / canon / scan
  consumers — with its module.
- **The cut.** `NetworkCut` patches socket
  connect/connect_ex, `create_connection`,
  `getaddrinfo`/`gethostbyname` and
  `urllib.request.urlopen` to raise
  `OfflineAttemptError` (an `OSError`) and record
  each attempt with its target. Creating a socket
  object is not an attempt; connecting is. Leaving
  the cut restores every patched name exactly —
  including on exception and under nesting.
- **The run.** `run_offline_pipeline(root)` runs
  TF-105's whole synthetic pipeline plus a TF-116
  `scan_store` over the resulting store, all inside
  one cut, and returns the report (format
  `thoughtforge-offline-ingestion/1`): pipeline
  counts verbatim, scan counts, `network_attempts`,
  and `offline` — true only when attempts is empty.
  An attempt is filed in the report, never hidden
  and never converted into a bare exception.
- **The loader fix.** `_get_model` loads with
  `local_files_only=True`, falling back to a plain
  construction only when the constructor rejects
  that argument (test fakes do; the real constructor
  accepts it). An uncached real model now fails
  locally instead of fetching. TF-053's
  `model_cache_status` remains how a machine proves
  the model cached before it ingests.

## 3. Filed limits

- **Fixture grade, process-local cut.** The proof is
  synthetic inputs through the real builders under a
  socket block in this process. It does not pull the
  machine's real network cable, and a library reaching
  the network without sockets/urllib is outside its
  reach (none is in the ingestion path; the static
  checker pins the etl estate's imports — the only
  socket/urllib importer is this module, which imports
  them only to block them).
- **Real-model encode NOT PROVEN here.** The package
  is absent on this VM (D-ENV-1). The embedding leg is
  proven in its cached-fake and injected-encoder
  shapes; what this slice proves about the real leg
  is that the loader can no longer turn it into a
  fetch.
- **No CLI flag.** `forge_memory.py` gains no
  `--offline` flag; its steps are the builders proven
  here, and TF-119 owns CLI wiring of this report.

## 4. Docket

- **D-OFFLINE-1 — added + FIXED.** Offline was
  assumed, never proven (§1): the whole pipeline now
  runs under a recording `NetworkCut` with zero
  attempts, in the live checker and the tests.
- **D-OFFLINE-2 — added + FIXED.** The embedding
  loader was an unguarded fetch (§1): `_get_model`
  is local-files-only, so an uncached model fails
  locally and model download is the filed fetch step.
- **D-OFFLINE-3 — added + FIXED.** The fetch boundary
  was undefined (§1): `FETCH_STEPS` / `OFFLINE_STEPS`
  file it as data, bound to TF-105's builders, with
  TF-105's map updated in the open.

TF-016 totals after this slice: 121 entries —
42 OPEN / 1 OPEN-ENVIRONMENT / 75 FIXED /
3 CLOSED-BY-VERDICT.

**Proof grade: FIXTURE** — synthetic sources through
the real builders under a hard socket block on this
VM. A real dump ingested with the machine's network
physically cut is NOT PROVEN, never claimed. TF-118
owns the reference library as a source; TF-119 owns
the ETL docs + new-source recipe; TF-120 is Part VI's
gate.
