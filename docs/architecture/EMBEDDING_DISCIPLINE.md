# TF-053 — Embedding Discipline: Models per Profile, Offline Availability Proven

*Part III, slice TF-053. Code:
`src/thoughtforge/etl/embedding_discipline.py` (the model
registry, the canonical name fold, the per-profile
assignment, the availability report, the stored-estate
check, and `embedding_problems()` re-deriving every claim
below from the live tree), wired into
`src/thoughtforge/etl/embeddings.py` (normalisation at the
loader, both builders, and both searches; the dead scalar
cosine retired) and `src/thoughtforge/knowledge/forge.py`
(model name normalised at construction; the vector legs
degrade to SQL instead of raising; the offline claim
corrected). Tests: `tests/test_tf053_embedding_discipline.py`.*

## 1. The question and the measured starting state

The roadmap asks: which embedding model serves which
hardware profile — and can each profile actually run it
*offline*, proven rather than assumed? Measured before
this slice, by direct probe:

- **No profile named an embedding model at all.** The six
  hardware profiles carry LLM recommendations and vector
  top-k values, but no embedding assignment. Every profile
  silently inherited the one global default —
  `all-MiniLM-L6-v2` (~91 MB plus the torch stack
  sentence-transformers rides on) — from
  `etl/embeddings.DEFAULT_MODEL`, the forge's constructor
  default, and `configs/default.yaml` alike. A pi_zero
  with 512 MB of RAM was assigned, in effect, the same
  embedding stack as a server.
- **The model's name was split in two.** The ETL, the
  forge, and the config spell it `all-MiniLM-L6-v2`;
  `inference/onnx_export` spells it
  `sentence-transformers/all-MiniLM-L6-v2`. Both resolve
  to the same hub model, but the loader's cache is keyed
  by the string it is handed and the knowledge DB's
  `model_name` column is filtered by exact match —
  embeddings built under one spelling are invisible to a
  search under the other.
- **Offline availability was assumed, never proven.**
  Nothing checked the package or the local cache before
  `SentenceTransformer(model_name)` was constructed. On
  an uncached machine that constructor reaches for the
  network; on a machine without the package it raises —
  while the forge's class docstring claimed "Fully
  offline — no network calls" over both facts.
- **The measured consequence on this VM:** the package
  is absent (docket D-ENV-1) and the model is in no local
  cache (the Hugging Face cache holds an unrelated TTS
  model, nothing else). `KnowledgeForge.retrieve(
  path="hybrid")` — the core's default path — **raised
  `RuntimeError` out of the vector leg and took the
  working SQL leg down with it**. Only the core's
  turn-level catch stood between that and a failed turn,
  and it still lost every SQL result the turn should
  have had.
- **The dimension was an untied constant.**
  `EMBEDDING_DIM = 384` is the one model's dimension;
  nothing tied it to the model, so a different model
  would have produced mismatched blobs unchecked.
- **The model's licence was recorded nowhere** (docket
  D-LIC-2's embedding share, TF-008 finding 7).

## 2. The registry

`EMBEDDING_MODELS` holds every embedding model the tree
names — exactly one today:

| field | value |
|---|---|
| canonical name | `all-MiniLM-L6-v2` |
| hub id | `sentence-transformers/all-MiniLM-L6-v2` |
| dimension | 384 (== `EMBEDDING_DIM`, pinned) |
| parameters | 22.7 M |
| size | ~91 MB (fp32 artefact) |
| licence | **Apache-2.0**, per the upstream model card at huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (recorded here for TF-008 finding 7; TF-190's final pass re-verifies) |

`canonical_model_name()` folds every spelling of a
registered model — its hub id, or any other org-prefixed
spelling — to the canonical name. Existing local paths
and unregistered names pass through unchanged:
registration is the registry's deliberate act, not a
guess. The fold is wired into the ETL loader (`_get_model`,
so the cache can no longer split), both embedding
builders (so rows are *stored* canonically), both vector
searches (so the exact-match filter always compares
canonical to canonical), and the forge's constructor.

Registering a second model is now a stated act: registry
entry (dimension + licence included), a per-profile
assignment, and the stored-estate check below extended
by the registry itself — never a string change at a
call site.

## 3. The assignment, per profile

`PROFILE_EMBEDDINGS`, as data:

| profile | model | runtime | batch | why |
|---|---|---|---|---|
| server_gpu | all-MiniLM-L6-v2 | sentence_transformers | 128 | 64 GB RAM; the estate's largest batches |
| desktop_gpu | all-MiniLM-L6-v2 | sentence_transformers | 128 | 16 GB RAM; same stack as server |
| desktop_cpu | all-MiniLM-L6-v2 | sentence_transformers | 64 | 8 GB shared with a CPU LLM; batches halved |
| pi_5 | all-MiniLM-L6-v2 | sentence_transformers | 32 | 4 GB on ARM; fits, batches step down |
| phone_low | all-MiniLM-L6-v2 | **onnx** | 16 | 2 GB cannot carry the torch stack beside a phone LLM; the profile's own deployment section already declares `onnx_export: true`, so the same model is assigned through the ONNX runtime (TF-070 owns the export path's truth) |
| pi_zero | **none** | none | 0 | 512 MB: no embedding model is honestly assignable. Vector retrieval is *not assigned*; the profile runs SQL-only **by rule, not by accident** |

Every profile's fallback is the one stated rule,
`EMBEDDING_FALLBACK = "sql"`: when the assigned runtime
is unavailable, retrieval degrades to the SQL leg —
announced in the log, never a raised error, never a
silent empty.

The batch sizes are the discipline's assignment; the
builders still take their `batch_size` arguments from
their callers (TF-111 wires ingestion to this table).

## 4. The offline proof

`availability_report(profile_id)` returns an
`AvailabilityReport` whose status is one of
`AVAILABILITY_STATUSES`:

- `available_offline` — claimed only when the facts hold;
- `unavailable_package_missing` — the runtime's package
  is not importable on this machine;
- `unavailable_model_not_cached` — package present, but
  the model is in no local cache (a first load would
  reach for the network, so offline is NOT proven);
- `unavailable_onnx_artefact_missing` — no exported
  model in the embeddings directory;
- `not_assigned` — pi_zero's ruling.

For a sentence-transformers profile, both facts are
checked: the package via `importlib.util.find_spec`,
and the cache by filesystem inspection of the Hugging
Face hub layout (`models--<org>--<name>/snapshots/…`)
and the legacy sentence-transformers home layout, across
`HF_HUB_CACHE` / `HF_HOME` / `SENTENCE_TRANSFORMERS_HOME`
and the default roots. A cached model means a snapshot
holding `config.json` and a weights file. For the ONNX
profile, the artefact is checked in the embeddings
directory — `utils/paths.get_embeddings_dir()`, docket
D-DEAD-3's never-called helper, now the named home this
report checks (and `data/embeddings/` is already the
estate's gitignored artefact home). The report never
loads a model and never touches the network, in any
branch; a verdict is for *this* machine at call time and
does not travel.

**This VM's verdicts today, from the live checker:**
every sentence-transformers profile —
`unavailable_package_missing` (D-ENV-1 stands OPEN; the
package is still not installed — this slice makes the
system behave truthfully in its absence and prove the
absence per profile; it does not install the torch
stack). phone_low — `unavailable_onnx_artefact_missing`
(no export has landed one). pi_zero — `not_assigned`,
by rule.

## 5. The forge degrades

`KnowledgeForge.vector_retrieve` now wraps each vector
leg separately: a failing entity or reference search
logs a warning naming the exception and contributes
nothing; the other leg and the whole SQL path stand.
Measured after the fix on TF-043's fixture corpus with
the entity leg forced to raise: hybrid retrieval
returns its SQL records, no exception. On this VM the
live behaviour matches: `vector_retrieve` returns `[]`
and hybrid `retrieve` completes instead of raising the
RuntimeError it raised before this slice.

The class docstring's "Fully offline — no network
calls" claim is corrected to the truth in place: the
SQL leg is fully local; the vector leg's offline
availability is *proven by the report*, never assumed
by the forge.

## 6. The stored estate, checked

`stored_embeddings_problems(db_path, model=…)` reads a
knowledge DB's `embeddings` and `reference_embeddings`
tables and names every row that breaks the discipline:
filed under an alias spelling instead of the canonical
name, or carrying a blob whose length is not the
registered dimension × 4 bytes (float32). Rows filed
under other models are left to their own registry
entries. TF-111 (embedding at ingestion, checksummed)
consumes this check.

## 7. Docket movement

- **D-DEAD-2 — FIXED by retirement.** The scalar
  `cosine_similarity` had no caller anywhere (TF-011)
  while its batch sibling is the live path; it is
  removed from `etl/embeddings.py`, and the checker
  exercises the batch function's identity case.
- **D-DEAD-3 — FIXED by wiring.** `get_embeddings_dir()`
  is now called: it is the ONNX artefact home the
  availability report checks.
- **D-LIC-2 — narrowed.** The embedding-model share is
  recorded (§2: Apache-2.0, model card). The
  data-source licences (ConceptNet / GeoNames / DBpedia
  / Wikidata, the SRD) remain TF-106's, closing at
  TF-190.
- **D-EMB-1 — found and FIXED by this slice.** Hybrid /
  vector retrieval raised `RuntimeError` when the
  embedding backend was unavailable, losing the working
  SQL leg with it; the forge's docstring claimed full
  offline operation over the top. Fixed: per-leg
  degrade + the corrected claim (§5).
- **D-EMB-2 — found and FIXED by this slice.** The
  model's two spellings split the loader cache and the
  DB's exact-match `model_name` filter, so embeddings
  built under one spelling were invisible to searches
  under the other. Fixed: the canonical fold wired
  through loader, builders, searches, and forge (§2).
- **D-ENV-1 — stands OPEN-ENVIRONMENT**, unchanged in
  substance: the package is still absent on this VM.
  What changed is that its absence is now a proved,
  per-profile verdict with a graceful degrade, not a
  crash wearing an offline claim.

## 8. Boundaries

- **Vector/hybrid retrieval quality is NOT PROVEN** —
  it has never been measurable on this VM (D-ENV-1),
  and this slice does not change that; TF-064's gate
  and Part VII's panels own quality claims.
- **phone_low's ONNX leg is NOT PROVEN** — the
  assignment is made and its absence is proved
  truthfully; TF-070 owns the export path and TF-173 /
  TF-128 own on-device proof.
- **Availability verdicts are per-machine.** This VM's
  verdicts are recorded above; Mimir and
  purplelongship prove their own by running the same
  report when their turns come (TF-082 / TF-174).
- No profile JSON was edited, no config value changed,
  and no shipped embedding behaviour changed for a
  machine where the package and cache are present —
  the fold maps the spellings the tree already used to
  the name the DB already stored by default.
