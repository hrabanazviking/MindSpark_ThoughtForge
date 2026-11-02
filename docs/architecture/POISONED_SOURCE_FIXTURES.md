# ThoughtForge Architecture — Poisoned-Source Fixtures (TF-116)

Poisoned-source fixtures: adversarial source data
refused / quarantined, kin to TF-062. Part VI's
twelfth slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The gate lives as verifiable data in
`src/thoughtforge/etl/poisoned_source.py`
(`SOURCE_TEXT_FIELDS`, `POISON_RULES`,
`POISON_FIXTURES`, `BENIGN_FIXTURES`, `PoisonLedger`,
`scan_store()`, checked by
`poisoned_source_problems()` and
`poisoned_source_live_problems()`); every builder
holds its ledger as `poison_gate`, TF-105's living
map gained its `poison` orchestration entry in the
open, and `KnowledgeForge` screens at retrieval.
The tests are `tests/test_tf116_poisoned_source.py`.
This slice **adds and fixes D-POISON-1, D-POISON-2
and D-POISON-3**.

## 1. Measured start

Probed live this tick, before any change, with one
synthetic adversarial string (TF-062's own fixture
shape) treated as data:

- **Nothing screened source content (D-POISON-1).**
  TF-108 judges a record's shape; TF-106 judges a
  source's licence; neither reads what the text
  says. The adversarial string landed verbatim,
  `skipped: 0`, through every text-carrying builder:
  a GeoNames place name, a DBpedia label and
  abstract, a wikidata label / description / aliases
  (in both the entities and labels tables), and a
  reference chunk's content.
- **Landed poison was retrievable (D-POISON-2).**
  That reference chunk was returned by
  `KnowledgeForge.sql_retrieve` for a query over its
  own words — TF-062's proved ride into a prompt,
  through the knowledge store. TF-062 had filed this
  boundary in the open as its stated remainder:
  "the knowledge DB estate is not screened … Part VI
  owns that boundary."
- **No disposition existed (D-POISON-3).** No
  ledger, scan or quarantine shape named poison in a
  knowledge store. A row planted directly in the
  store file (plain SQLite — a real vector) was
  invisible to every Part VI ledger, and no one
  could ask a finished store what in it is
  adversarial.

## 2. The gate

- **TF-062's classes, reused never copied.** The
  screen is `knowledge/adversarial.screen_text` over
  TF-062's nine threat classes: six refuse classes
  (direct instructions to a model) and three
  quarantine classes (directive-shaped text with a
  possible legitimate twin). Verdicts are TF-062's
  words — pass / quarantine / refuse, strongest
  finding governs — matched on folded text, so case
  tricks do not slip a pattern.
- **Only landed fields are screened.**
  `SOURCE_TEXT_FIELDS` files, per source, exactly
  the text a builder writes: wikidata label /
  description / aliases / statement text / labels;
  conceptnet concepts + relation (as landed text,
  after URI-to-text); geonames name + ascii name;
  dbpedia label / abstract / types; reference title
  / content / tags. Text no column carries (GeoNames
  alternate names) is not screened and cannot refuse
  a record.
- **Landing, per builder.** After TF-108's shape
  gate and before insertion, each builder screens:
  wikidata per entity (a non-pass entity lands no
  entity, label or statement row and does not
  present its subject to TF-114), conceptnet per
  edge (both the CSV and reference-edge paths),
  geonames per place, dbpedia per entity, reference
  per chunk (a held-out chunk's index is not
  presented, so TF-114 retires any stored copy on a
  re-run; a file whose chunks are all held out lands
  nothing). TF-113's canon path rides the reference
  builder and inherits the gate.
- **Refuse drops; quarantine holds out.** A refuse
  record is dropped whole. A quarantine record is
  held OUT of the store and named in the ledger —
  not landed with a tag. The difference from TF-062
  is filed, not hidden: a personal-memory record has
  an operator (its owner) to decide later, so TF-062
  tags in place; a source record has no operator of
  its own, so the hold is the quarantine. Both count
  in the builder's `skipped`; the builder's fresh
  `poison_gate` ledger (TF-108's shape) names every
  hold with source, position, ref, verdict, classes
  and findings — nothing is silently dropped.
- **Retrieval screens what landing missed.**
  `KnowledgeForge.sql_retrieve` and
  `vector_retrieve` screen every result's text and
  exclude non-pass records; `retrieve()`'s retained
  report names them in `screened_out` (record,
  verdict, classes), and
  `forge.last_poison_screened_out` carries a leg's
  own exclusions. A benign store's results are
  untouched.
- **The scan.** `scan_store(db_path)` opens a store
  read-only (a missing store files `exists: False`
  and is never created), screens every filed text
  column of the seven knowledge tables, and names
  every flagged row by table + key with verdict and
  classes, with per-table and per-verdict counts.
- **The fixtures.** `POISON_FIXTURES` files ten
  adversarial shapes (at least one per source; both
  verdict legs on wikidata, geonames and reference)
  and `BENIGN_FIXTURES` five lookalikes that must
  pass — a record *about* prompt injection, ordinary
  instruction prose, "you are now entering the
  boathouse". The static checker re-judges every
  fixture against the real screen on every run; the
  live checker runs poisoned + clean inputs through
  the real builders and the real retrieval.

## 3. Filed limits

- **English-only patterns.** TF-062's rows are
  English-only; a Spanish or German injection in
  source data is NOT PROVEN caught. The shape
  (classes + verdicts + wiring) is
  language-independent; the pattern rows are not.
- **Shape, not intent.** An abstract that quotes an
  injection phrase verbatim carries the shape and
  is held/refused. Part V's enforcement over answers
  is the defence in depth behind the gate.
- **Subset copies and embedding builds do not
  re-screen.** A subset of a screened store inherits
  its screening; a planted row is caught by the scan
  and at retrieval, not at copy or embed time —
  retrieval is where embedded text reaches a prompt,
  and both vector legs are screened there.
- **No CLI or metrics wiring yet.** `forge_memory.py`
  still prints stats only, and TF-115's per-run
  report does not carry the poison ledger — TF-119
  owns the recipe / wiring share. No schema table
  was added; the ledger is data on the builder.

## 4. Docket

- **D-POISON-1 — added + FIXED.** No content
  screening in ingestion (§1): every builder now
  screens landed text by TF-062's classes before
  insertion, refuse dropped / quarantine held out,
  ledgered per record.
- **D-POISON-2 — added + FIXED.** Landed poison was
  retrieved into bundles (§1): both KnowledgeForge
  legs screen results and `retrieve()` names
  exclusions in `screened_out`.
- **D-POISON-3 — added + FIXED.** No disposition
  existed (§1): the per-build `poison_gate` ledger
  names every hold, and `scan_store` names poison
  already inside a finished store, read-only.

TF-016 totals after this slice: 118 entries —
42 OPEN / 1 OPEN-ENVIRONMENT / 72 FIXED /
3 CLOSED-BY-VERDICT.

**Proof grade: FIXTURE** — synthetic adversarial +
benign sources through the real builders and the
real retrieval on this VM. No real dump has been
screened, and screening at dump scale is NOT
PROVEN, never claimed. TF-117 owns offline
ingestion; TF-118 owns the reference library as a
source; TF-119 owns the ETL docs + new-source recipe
(including any CLI / metrics wiring of this
ledger); TF-120 is Part VI's gate.
