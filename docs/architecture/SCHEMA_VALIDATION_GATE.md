# ThoughtForge Architecture — Schema Validation at the Gate (TF-108)

Schema validation at the gate: malformed records refused
with reasons, counted, never silently dropped. Part VI's
fourth slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The gate lives as verifiable data in
`src/thoughtforge/etl/schema_gate.py` (`RECORD_SCHEMAS`,
`GATE_RULES`, `validate_record()`, `SchemaGate`, checked
by `gate_problems()` and `gate_live_problems()`); the
enforcement is in `src/thoughtforge/etl/wikidata.py` and
`src/thoughtforge/etl/sources.py`; the tests are
`tests/test_tf108_schema_gate.py`. This slice **adds and
fixes D-GATE-1, D-GATE-2 and D-GATE-3**. TF-105's living
map was updated in the open (the four gated builders'
notes).

## 1. Measured start

Probed live this tick, before any change, with synthetic
malformed records through the real builders:

- **Malformed records crashed whole builds (D-GATE-1).**
  A DBpedia line holding a JSON number (`42`) or list
  died with `AttributeError` on `.get`; a DBpedia label
  of `123` died with `TypeError` when sliced; a
  ConceptNet metadata weight of `"abc"` died with
  `ValueError` (the parse around it caught only
  `JSONDecodeError` / `TypeError`); a wikidata entity
  whose `labels` was a string, whose `claims` was a list
  or a string, whose claim entries were strings, whose
  `aliases` was a string, or whose `id` was an integer
  died with `AttributeError` / `TypeError` mid-stream.
  In every case the build aborted — records already read
  were neither landed nor counted as refused.
- **Out-of-schema values were silently accepted
  (D-GATE-2).** A GeoNames row with latitude 999.0 /
  longitude -999.0 was inserted as filed, and so was a
  row with an empty name. A ConceptNet line whose
  metadata was not JSON at all was inserted with its
  weight silently defaulted to 1.0; lines with empty
  concepts or an empty relation were counted
  "inserted"; a ConceptNet weight of -5 was accepted.
  A wikidata label whose value was the integer 123 was
  stored as the text `"123"`.
- **Refusals carried no reasons (D-GATE-3).** Every
  builder counted a bare `skipped` integer: a short row,
  an unparseable row and a filtered row were
  indistinguishable, and no reason was recorded
  anywhere. ConceptNet's `build_from_reference` was
  worse — non-dict items, field-less items and
  non-array files were dropped by a plain `continue`,
  counted nowhere at all.

## 2. The gate

One gate, held by each builder as `self.schema_gate`
(a fresh `SchemaGate` per build call; the ledger
describes the most recent build):

- **Judgement before insertion.** Every external record
  is presented to the gate before any part of it is
  written. `validate_record(source, record)` returns a
  list of reasons (`code:field: detail` — so counts
  group by code); an empty list means the record may
  pass. A refused record contributes no row to any
  table, and the build continues with the next record:
  **no malformed record raises out of a build** (the
  pre-fix crashes of §1 are now refusals), and **no
  out-of-schema value is repaired into acceptance** —
  unparseable metadata does not default a weight,
  out-of-range coordinates do not land, a non-string
  label does not coerce.
- **Counted, with reasons.** A refusal is recorded in
  the ledger with its source, its position in the run
  (per-source, from 1), an optional caller reference
  (an id the record itself carries — a QID, a uri, a
  geonames id — never the record's content), and every
  reason. Refusals also count in the builder's existing
  `skipped` total, so the stats shapes TF-105 filed do
  not change; the ledger adds the breakdown by source
  and by reason code. TF-115 owns reporting refusal
  rates per run.
- **Parse-level refusals** (a line that never became a
  record — too few fields, invalid JSON) enter the
  ledger through `refuse()` with the builder's own
  named reason; `refuse()` itself raises on an empty
  reason list, so a record can never be dropped
  unnamed.
- **Filters are not refusals.** A schema-valid record
  excluded by a builder's stated filter (ConceptNet's
  language filter, GeoNames' `min_population` /
  `feature_classes`, wikidata's excluded instance_of
  types and `instance_of_filter`, deprecated-rank
  claims) still counts in `skipped` but never enters
  the ledger. The gate judges shape; filters judge
  selection.
- **A blank line is not a record.** In line-delimited
  sources an empty line is file padding: neither seen
  by the gate nor counted. Every non-blank line ends
  landed, filter-skipped, or refused with reasons.
- **An unknown source is refused by name.**
  `validate_record` raises `ValueError` for an
  unregistered source id — a new source files its
  schema here before its builder gates on it (TF-119's
  recipe owns that step; D-ETL-2's registry gate is a
  different gate, over sources rather than records).

## 3. The schemas

| Source | Record | The checks that refuse it |
|---|---|---|
| wikidata | one dump entity object | `id` present, a string, matching `^Q[1-9][0-9]*$`; `labels` / `descriptions` objects of `{value: string}`; `aliases` object of lists of `{value: string}`; `sitelinks` an object; `claims` an object of lists of objects |
| conceptnet | one 5-field assertions line + its parsed metadata | uri / relation non-empty; both concept uris non-empty with a non-empty final segment; metadata parses as a JSON object; a carried weight is a finite, non-negative number — never a string, never a bool |
| geonames | one 19-field places row, raw | `geonames_id` a positive integer; `name` non-empty; latitude empty or in [-90, 90]; longitude empty or in [-180, 180]; population empty or a non-negative integer; elevation empty or an integer |
| dbpedia | one JSON line | the value is an object; `uri` present, a string, non-empty; `label` / `abstract` strings when carried; `types` a string or a list of strings |
| conceptnet_reference | one object from a flat reference array | `concept1` / `relation` / `concept2` all non-empty strings (the builder's subject/predicate aliases resolve before the gate) |

## 4. Filed limits

- **The reference builder is not gated here.** Its
  records are chunks the builder generates itself from
  files, not external records; its ingestion side
  (reproducible + checksummed, the read/write halves of
  D-DATA-4) is TF-118's road. The gate covers the four
  external record sources plus the conceptnet
  reference-edge path, whose records *are* external
  data in file form.
- **Statement-level shapes inside a schema-valid
  wikidata entity** remain the builder's parse: a claim
  whose snak carries an unknown datatype, a `novalue` /
  `somevalue` snak, or a deprecated rank produces no
  statement under the builder's filed semantics — the
  entity itself is valid, and TF-109 / TF-114 own what
  overlapping and changed facts mean.
- **The ledger is in memory, per build.** It rides the
  builder instance for the caller (and TF-112's dry-run
  report is its natural first consumer); no schema
  table was added for it, so TF-105's filed schema
  facts stand.

## 5. Docket

- **D-GATE-1 — added + FIXED.** Malformed records
  crashed whole builds (§1): every gated builder now
  judges before it parses, and the live checker + tests
  run the §1 crash shapes through all four builders —
  no exception, valid records landed, malformed ones
  refused.
- **D-GATE-2 — added + FIXED.** Out-of-schema values
  silently accepted (§1): out-of-range coordinates, an
  empty name, unparseable metadata, a negative or
  string weight, empty concepts and a non-string label
  value are all refused whole; the live checker proves
  the GeoNames store holds only the valid row.
- **D-GATE-3 — added + FIXED.** Refusals counted with
  no reasons, and one path not counted at all (§1):
  every refusal now lands in the builder's gate ledger
  with source, position and reasons, broken down by
  source and reason code — including
  `build_from_reference`, whose non-dict items,
  field-less items and non-array files are refused by
  name there.

**Proof grade: FIXTURE** — synthetic malformed records
through the real builders on this VM. No real dump
(wikidata / conceptnet / geonames / dbpedia at scale)
has been ingested by this campaign, and a live source's
malformation *rate* is **NOT PROVEN**, never claimed.
TF-109 owns dedup & merge for overlapping sources;
TF-112's dry-run is the ledger's first reporting
consumer; TF-115 owns per-run refusal-rate reporting.
