# TF-120 — Part VI Gate: The Ingestion Forge (ETL) (Part VI Gate)

Closed 2026-11-01 on branch `yrsa/thoughtforge-road-build`.
Roadmap wording: "Part VI gate: a full source → store →
retrieval round trip on fixtures, with provenance intact
at every hop".

## The gate verdict

**Part VI is COMPLETE. All 15 build slices (TF-105–TF-119)
are banked with evidence in `docs/architecture/`, and the
gate proof was re-measured fresh at this gate — not quoted
from its slices:**

1. **A full source → store → retrieval round trip, run
   live at this gate, with provenance intact at every
   hop — GREEN.** A fresh probe (synthetic fixtures only,
   TF-017's invented world — one synthetic town, "Lantern
   Town", carried by four sources; `data/` never opened)
   ran the real pipeline end to end on this VM:
   - **Registry (TF-106).** `check_source_use(...,
     ingest_private)` returned each source's filed
     provenance + licence: GeoNames `CC BY 4.0`, Wikidata
     `CC0 1.0 (structured data…)`, DBpedia `CC BY-SA 3.0 +
     GNU Free Documentation License (dual, from release
     3.4 onward)`; the reference library passed private
     ingestion as `quarantined_unverified` with licence
     `None`, and its redistribution was refused by name
     (`QuarantinedSourceError`: "private local ingestion
     only, with no licence claim for the whole").
   - **Dry-run (TF-112).** A 3-row GeoNames fixture
     (one valid row, one renamed duplicate, one row with
     latitude 999.0) dry-ran into scratch only: stats
     `{'inserted': 2, 'skipped': 1}`; the TF-108 gate
     ledger travelled in the report (refused 1,
     `by_reason {'out_of_range': 1}`); the TF-109 merge
     ledger travelled too (1 collision,
     `conflict_overwrite`); the report carried the
     registry entry (`CC BY 4.0`,
     `licensed_with_obligations`); the embeddings leg was
     filed `not_run` (no encoder injected — a dry run
     loads no model on its own word); and the named real
     store was **proven untouched** (`target_untouched
     True`, hash + counts agreeing).
   - **Real builders into a fresh store.** Wikidata
     `{'entities': 1, 'statements': 1, 'labels': 1,
     'skipped': 0}` with its schema-gate ledger clean
     (seen 1, refused 0) and poison ledger clean (seen 1,
     refused 0, quarantined 0); GeoNames
     `{'inserted': 1, 'skipped': 0}`; DBpedia
     `{'inserted': 1, 'skipped': 0}`; the reference
     builder `{'files': 1, 'chunks': 1, 'skipped': 0}`
     with merge / poison / library / update ledgers all
     carried and clean. Store counts after the builds:
     entities 1, statements 1, labels 1, geonames_places
     1, dbpedia_entities 1, reference_chunks 1.
   - **Stored rows inspected in the DB.** Every landed
     row still names its source: the entity row
     `(Q120001, 'Lantern Town', source 'wikidata')`; the
     statement `(Q120001, P31, Q120900, 'wikidata')`; the
     place `(120001, 'Lantern Town', 1.5, 2.5,
     'geonames')`; the DBpedia row
     `('http://example.org/Lantern_Town', 'dbpedia')`;
     the reference chunk `(chunk_id 1,
     source_file 'lantern_care.md', source 'reference')`;
     and the ingestion log holds exactly its one filed
     row, `('wikidata', 'complete')`.
   - **Embeddings (TF-111).** The driver ran over the
     landed store with an injected fake encoder (the
     filed shape — sentence-transformers is absent on
     this VM, D-ENV-1): plan model `all-MiniLM-L6-v2`
     from the builders' defaults, `entities_built 1`,
     `reference_built 1`, 2 batches, `stored_problems
     []`, estate digest
     `e78a4a0b61871358a573ca129b9f469d4d032d3557ac02881ae340a7e98a3e87`;
     both stored rows carry the plan's `model_name`, and
     `row_checksums` covers both tables (entity
     `Q120001`, reference chunk `1`).
   - **Retrieval (KnowledgeForge + TF-048).**
     `sql_retrieve("Lantern Town lantern")` returned
     exactly 3 hits, one per retrievable source —
     wikidata, reference, geonames — and
     `retrieve(..., path="sql")` returned all 3 as a
     bundle whose TF-048 citation payloads name their
     sources: the entity `marker 'Q120001', source
     'wikidata', locator 'Q120001'`; the place
     `marker 'geonames:120001', source 'geonames',
     locator '120001'`; the reference chunk
     `marker 'ref:1', source 'reference', locator
     'lantern_care.md'` — the locator IS the chunk's
     source file, so the landed fact still points back
     at the file it came from. The retained retrieval
     report: considered 3, fired all 3, `screened_out
     []` (TF-116's screen passed everything the builders
     landed).
   - **Finished-store scan (TF-116).** `scan_store` over
     the completed store (read-only): `scanned_rows 6`,
     `flagged []`, `by_verdict {}` — nothing the round
     trip landed is adversarial by TF-062's classes.
   The probe found **no defect**: nothing was fixed at
   this gate, and this slice is gate doc + docket
   rulings only, exactly as TF-104's gate was.
2. **Targeted + full suites re-run at the gate** (see
   "Suite at the gate" below): the Part VI targeted set
   (all 15 test files, TF-105–TF-119) = **322 tests,
   322 passed**; the full suite stands exactly on
   TF-119's banking line.

The proof is FIXTURE grade: synthetic sources through
the real registry, builders, store, embedding driver
and retrieval on this VM. A real dump's round trip
(wikidata / geonames / dbpedia / the reference library
at scale) is **NOT PROVEN**, and a real model's
embedding leg is NOT PROVEN on this VM (D-ENV-1) —
neither is claimed anywhere at this gate.

## 2. What the gate inherits, stated as one estate

Part VI's pipeline is five builders (TF-105's map:
wikidata, conceptnet, geonames, dbpedia, reference)
writing one 10-table + 3-FTS store, with every stage's
contract filed as verifiable data. Around the builders
stand the part's guards, each a ledger the build
carries: TF-106's registry (provenance + licence per
source, quarantine by rule), TF-108's schema gate
(malformed records refused whole, with reasons),
TF-109's merge rules (duplicates never rewritten,
conflicts ledgered before resolution), TF-114's
lifecycle (re-runs supersede and retire by presented
set), TF-116's poison screen (at landing and again at
retrieval), and TF-118's library gate (the reference
road skips by rule and chunks each distinct content
once). TF-107 hardened the one long-haul builder
(checkpointed, resumable, FULL posture); TF-110 sizes
subsets by measured bytes against live profile
budgets; TF-111 embeds per profile, checksummed and
resumable; TF-112 test-ingests into scratch with the
registry consulted first and the target proven
untouched; TF-113 gives the owner's own materials the
same pipeline behind a private-store affirmation;
TF-115 reports every run (rate, refusal rate, growth,
all ledgers); TF-117 proves the whole road runs with
the network cut; TF-119 files the docs index and the
new-source recipe whose first step is the registry
filing. The one orchestrator remains `forge_memory.py`
's `all` (seven steps since TF-119) — there is still
no in-package pipeline runner, as TF-105 filed.

Two filed limits the gate states plainly rather than
smoothing over: KnowledgeForge has **no DBpedia
retrieval leg** (its legs are entities, reference
chunks, conceptnet edges and geonames places), so a
DBpedia fact's provenance is proven at the store hop
and its round trip ends there — the probe's DBpedia
row is inspected in the DB, not retrieved; and the
builders themselves still do not consult the registry
at build time (D-ETL-2's filed shape, closed by its
two consumer shares — TF-112's dry-run and TF-119's
recipe — not by builder wiring, and not claimed
otherwise).

## 3. Gate rulings on the entries and gaps assigned to this gate

1. **D-LIC-2 (upstream data-source licences unverified)
   — the Part VI share stands discharged; the entry
   stays OPEN on its TF-190 shares.** TF-106 recorded
   every data-source licence from upstream terms and
   TF-053 recorded the embedding model's; this gate
   re-verified the registry leg live (proof 1, hop 1):
   each `check_source_use` returned the filed
   provenance + licence, and the quarantined reference
   library still states no licence and refuses
   redistribution. What remains is exactly what the
   entry and the registry entries themselves name:
   **TF-190**'s final licence pass re-verifies the SRD
   in-tree copy against SRD 5.1's CC BY 4.0, the
   DBpedia entry against the actual export used, and
   the model card. No Part VI work remains on it.
2. **D-DATA-2 (duplication in `data/`) — stays OPEN
   on its canon-files share alone.** Its store-level
   share stands FIXED by TF-109 (the same fact can no
   longer duplicate inside the store — the gate's
   dry-run merge ledger shows the machinery live: the
   renamed duplicate filed as a ledgered
   `conflict_overwrite`, never a silent second row).
   The duplicate files inside `data/` stand untouched:
   they are Volmarr's canon, canon handling is his
   word, and this gate — like every Part VI slice —
   never opened `data/`. **TF-192** verifies at the
   end that the entry is left open with this reason
   and no other.
3. **D-EMB-4 (two models' rows cannot coexist) —
   stays OPEN, owner TF-177.** The gate's embedding
   leg ran exactly one model (`all-MiniLM-L6-v2`), as
   TF-111 filed: per-model selection (D-EMB-3) makes
   each build honest, but the tables' primary key is
   the row id alone, so a second model's build would
   replace the first's rows. The composite-key
   migration is TF-177's upgrade path; no claim at
   this gate touches multi-model coexistence.
4. **No other Part VI entry remains open.** Every
   other entry the part owns was fixed by its own
   slice with evidence (TF-105 D-SUBSET-1; TF-106
   D-DATA-3; TF-107 D-WIKI-1/2/3; TF-108 D-GATE-1/2/3;
   TF-109 D-MERGE-1/2/3/4; TF-110 D-SUBSET-2/3/4 and
   D-PROF-1's last share; TF-111 D-EMB-3; TF-112
   D-COV-2; TF-113 D-CANON-1; TF-114 D-UPDATE-1/2/3;
   TF-115 D-MET-1/2/3; TF-116 D-POISON-1/2/3; TF-117
   D-OFFLINE-1/2/3; TF-118 D-DATA-4, D-REF-2, D-REF-3;
   TF-119 D-ETL-1, D-ETL-2, D-DATA-1), and the gate's
   fresh probe + targeted re-run found no regression
   in any of them. No new docket entry is added by
   this gate: the probe found no defect.

**Docket (TF-016) at the gate: 123 entries — 38 OPEN /
1 OPEN-ENVIRONMENT / 81 FIXED / 3 CLOSED-BY-VERDICT**
(unchanged by this gate — no entry closes here; the
three rulings above record why the entries that stay
open stay open, and TF-192 verifies at the end that
no entry is left open without a reason).

## 4. Plain-language summary (the Volmarr-facing half of this gate)

Here is what Part VI did, in plain words:

- **Every source now carries its papers.** Before this
  part, nothing in the tree recorded where the
  knowledge came from or under what licence. Now each
  source is filed with its origin and its terms —
  Wikidata free for all, GeoNames free with credit,
  ConceptNet and DBpedia free with credit and
  share-alike — and the built-in reference library,
  whose licence nobody can honestly state as a whole,
  is quarantined by rule: it may be used privately on
  this machine, never handed on, and no licence is
  ever claimed for it.
- **Bad records are refused at the door, out loud.**
  A malformed record — a place plotted at latitude
  999, a broken line, a poisoned text — is refused
  whole, with the reason written in a ledger the build
  carries, instead of crashing the build or slipping
  in silently. At this gate a planted bad row was
  refused for exactly its reason (`out_of_range`), in
  the open.
- **Nothing duplicates quietly, and re-running a
  source updates instead of piling up.** The same fact
  arriving twice is recognised; a changed fact replaces
  the old one with the change written down; a fact that
  disappears from a re-presented file is retired, not
  left stale. Cross-source overlaps are named as
  findings, never silently merged — each source keeps
  its own rows and its own name on them.
- **The whole journey was walked again at this gate,
  and the name-tags survived it.** One invented town
  was carried by four different sources through the
  registry check, a test-ingestion, the real builders,
  the store, the embedding step, and back out through
  search — and at every stop it still said where it
  came from, ending with search results whose citation
  tags name the source and, for the reference fact,
  the very file it came from.
- **Nothing here has faced a real dump.** Every number
  is fixture grade — real code, invented records, on
  this machine. Ingesting a real Wikidata or GeoNames
  dump end to end, and embedding with a real model,
  are still **NOT PROVEN**, and the duplicate files in
  the canon folder were never touched — that decision
  stays yours.

## 5. The evidence, slice by slice

| Slice | Artefact | Headline |
|---|---|---|
| TF-105 | `ETL_MAPPED.md` | The pipeline mapped from code (5 stages, 5 builders, contracts as data); D-SUBSET-1 fixed (limited subsets keep their chunks) |
| TF-106 | `SOURCE_REGISTRY.md` | Every source filed with provenance + licence from upstream terms; the unlicensed-in-fact library quarantined by rule |
| TF-107 | `WIKIDATA_PIPELINE.md` | The long-haul builder hardened: checkpointed commits, fingerprint resume, FULL posture; D-WIKI-1/2/3 fixed |
| TF-108 | `SCHEMA_VALIDATION_GATE.md` | Every external record judged before writing; refusals whole, reasoned, counted, ledgered; D-GATE-1/2/3 fixed |
| TF-109 | `DEDUP_AND_MERGE.md` | Per-table identities + merge ledger: duplicates never rewritten, conflicts ledgered, overlaps named never merged; D-MERGE-1/2/3/4 fixed |
| TF-110 | `SUBSET_CRAFT.md` | Subsets built fresh, measured in bytes, trimmed to the profile's live budget; D-SUBSET-2/3/4 fixed, D-PROF-1 closed |
| TF-111 | `EMBEDDING_AT_INGESTION.md` | Embedding per profile (TF-053), checksummed, resumable by fingerprint; D-EMB-3 fixed, D-EMB-4 filed to TF-177 |
| TF-112 | `INGESTION_DRY_RUN.md` | Test-ingest into scratch with the full report (gate + merge ledgers, target proven untouched); D-COV-2 fixed |
| TF-113 | `CANON_INGESTION.md` | The owner's materials ride the same pipeline behind a private-store affirmation + provenance manifest; D-CANON-1 fixed |
| TF-114 | `UPDATE_INGESTION.md` | Re-runs supersede/retire by presented set instead of duplicating; D-UPDATE-1/2/3 fixed |
| TF-115 | `INGESTION_METRICS.md` | Per-run reporting: records/sec, refusal rate, store growth, every ledger carried; D-MET-1/2/3 fixed |
| TF-116 | `POISONED_SOURCE_FIXTURES.md` | Adversarial source text refused/quarantined at landing and screened again at retrieval; D-POISON-1/2/3 fixed |
| TF-117 | `OFFLINE_INGESTION.md` | The whole pipeline proven under a recording network cut; the fetch boundary filed as exactly three steps; D-OFFLINE-1/2/3 fixed |
| TF-118 | `REFERENCE_LIBRARY_AS_SOURCE.md` | The reference library reproducible + checksummed: manifest, verify, two-build agreement; D-DATA-4, D-REF-2, D-REF-3 fixed |
| TF-119 | `ETL_DOCS_AND_NEW_SOURCE_RECIPE.md` | Docs index + 12-step new-source recipe filed as data, registry filing first; D-ETL-1, D-ETL-2, D-DATA-1 fixed |

## Road re-cut assessment

Part VI changed no Part VII–X slice's wording or
order. Assignments recorded here per TF-018's rule:
D-LIC-2's remaining shares stand with **TF-190** as
already filed; D-DATA-2's canon-files share stays
open for Volmarr's word with **TF-192** verifying it
is left open with its reason; D-EMB-4 stands with
**TF-177** as already filed. Part VII inherits an
ingestion estate whose sources, licences, ledgers
and filed limits are frozen as data — its harness
(TF-121) measures the cognition claims against
stores this part can build, dry-run, and prove.

## Suite at the gate (measured this slice, TF-014 method)

`PYTHONPATH=src python3 -m pytest tests/ -q` on this
VM, read from the on-disk gate log
(`~/workspace/.verify-tmp/tf120_gate.log`):
**3120 collected / 3118 passed / 1 skipped /
1 failed (D-ENV-1, unchanged — the phase8 dependency
check naming sentence-transformers absent on this
VM, an environment entry not a code defect)** —
exactly TF-119's banking line. The gate's targeted
re-run (all 15 Part VI test files, TF-105–TF-119) is
322 / 322 passed, and the gate proof above was
measured by a fresh live probe at this gate (the
TF-106 registry gate, a TF-112 GeoNames dry-run, the
real wikidata / geonames / dbpedia / reference
builders into a fresh store, direct DB inspection of
the landed rows, TF-111's driver with an injected
encoder, KnowledgeForge `sql_retrieve` + `retrieve`,
and TF-116's `scan_store`), every hop's provenance
asserted in the probe itself, which exited clean.
This slice adds one gate doc plus the TF-016 docket
rulings and no code or tests, so the count does not
move at banking.

## What opens next

Part VII — Benchmarks & the Proof Harness
(TF-121–TF-138), starting with TF-121: harness
architecture — benchmarks as code with fixtures,
scorers, and pinned baselines (the `benchmarks/`
estate, TF-010, unified).
