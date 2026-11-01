# ThoughtForge Architecture — Source Registry (TF-106)

Source registry: every source carries provenance + licence
(TF-008's audit consumed); unlicensed-in-fact sources
quarantined by rule. Part VI's second slice. Built
2026-11-01 on branch `yrsa/thoughtforge-road-build`.

The registry lives as verifiable data in
`src/thoughtforge/etl/source_registry.py` (`SOURCES`,
`REGISTRY_RULES`, checked by `registry_problems()` and
`registry_live_problems()`; the gate is
`check_source_use()`, the proposal judge is
`classify_candidate()`); the tests are
`tests/test_tf106_source_registry.py`. This slice **fixes
the TF-106 shares of D-LIC-2 and D-DATA-3** and **files
D-ETL-2** (§5).

## 1. Measured start

TF-008 finding 7, filed 2026-10-30 and still true at this
tick's probes: data-source licences were "a separate,
unverified ledger" — the ETL ingests ConceptNet, GeoNames,
DBpedia and Wikidata, the library ships the D&D 5e SRD,
and the upstream share-alike and attribution terms were
verified nowhere and recorded nowhere in-tree. TF-007
finding 4 is the same wound from the other side: no
provenance/licence file inside `data/`; provenance for the
external sets lived only in ETL code comments (the
builders' docstring download notes in `etl/sources.py` and
`etl/wikidata.py`). Nothing in the tree could answer the
question "under what terms may this source leave the
machine?" — and nothing refused a source that had no
answer.

## 2. The registry

| Source | Builder | Provenance | Licence (upstream terms) | Status |
|---|---|---|---|---|
| Wikidata | WikidataETL | Wikimedia Foundation / community structured data | CC0 1.0 (structured namespaces) | licensed |
| ConceptNet 5 | ConceptNetETL | Commonsense Computing Initiative (OMCS, Wiktionary, DBpedia, WordNet…) | CC BY-SA 4.0 (complete data) | licensed, obligations: attribution + share-alike |
| GeoNames | GeoNamesETL | GeoNames, compiled from official public sources | CC BY 4.0 | licensed, obligations: attribution |
| DBpedia | DBpediaETL | DBpedia project, extracted from Wikipedia | CC BY-SA 3.0 + GFDL (dual, from 3.4) | licensed, obligations: attribution + share-alike |
| Reference library | ReferenceDataETL | Mixed (TF-007 census: owner's authored sets, public-domain classics in unrecorded editions, topical sets, installed files) | **none verified for the whole** | **QUARANTINED** |
| D&D 5e SRD (sub-source) | ReferenceDataETL, `dnd-5e-srd-markdown/` only | Wizards of the Coast | CC BY 4.0 (SRD 5.1, 2023-01-27) | licensed, obligations: attribution |

Terms were read from the upstream pages on 2026-11-01 and
each entry names where: Wikidata:Licensing (structured
data CC0; other namespaces' text is CC BY-SA 4.0 and is
not what the builder ingests); ConceptNet's data credits
(the complete data CC BY-SA 4.0; provenance-tracked
subsets extractable under CC BY 4.0 alone — the builder
ingests the complete assertions file, so BY-SA is filed);
the GeoNames About page and dump readme (CC BY 4.0);
DBpedia's dataset licence note (dual CC BY-SA 3.0 + GFDL
from release 3.4, GFDL only before); WotC's SRD 5.1
CC BY 4.0 release and prescribed attribution.

Two entries carry honest caveats in their notes, not
smoothed: DBpedia's builder consumes a JSON export rather
than a numbered release dump, so the dual licence is
filed with its pre-3.4 leg named; and the in-tree SRD
markdown copy carries its own legal chapter (00legal) —
the filed licence is SRD 5.1's CC BY 4.0, and TF-190's
final licence pass re-verifies the copy against it.

## 3. The quarantine rule

A source with no licence verified in fact is
**quarantined by rule**: it may be ingested into a private
local store only, it may never be redistributed, and no
licence claim may be made for it. In the registry that
means a quarantined entry's `licence` is `None` by
construction — the registry never prints a licence it did
not verify — and the gate enforces the disposition:

- `check_source_use(id, "ingest_private")` returns the
  entry for every registered source, quarantined included
  (private local ingestion is what quarantine allows).
- `check_source_use(id, "redistribute")` returns licensed
  entries **with their obligations**, and raises
  `QuarantinedSourceError` for a quarantined one.
- An **unregistered** id raises `ValueError` by name: a
  new source is quarantined by default until it is filed
  here with provenance and a verified licence.
- `classify_candidate()` applies the same rule to a
  proposed source before filing: missing provenance or an
  unverified licence quarantines it, with the missing
  pieces named; a complete proposal is "admissible for
  filing" — filing is still a recorded act, the classifier
  grants nothing by itself.

The reference library is the estate's one quarantined
source, and quarantine fits its facts twice over: its
licence is unverified for the whole, and it is the
owner's canon — private by his law regardless. A file
family leaves quarantine only by being filed as its own
entry on its own verified terms, as the SRD is; the SRD's
entry does not lift the library's quarantine, and the
library's quarantine governs any whole-library claim.

## 4. What this slice does not do

No builder's behaviour changes. **The builders do not yet
call the gate** — nothing in `etl/` consults the registry
at build time, so the quarantine is enforced at the
registry surface, not yet inside a build. That wiring is
filed as D-ETL-2 (§5). TF-107 owns wikidata hardening;
TF-108 schema validation at the gate (a different gate —
record shape, not licence); TF-112's dry-run is the first
consumer that must consult this registry; TF-119's
new-source recipe must file a source here before
documenting it; TF-190 owns the final licence pass,
including the SRD-copy and DBpedia-export re-verifications
named above.

## 5. Docket

- **D-LIC-2 — FIXED (TF-106 share).** The data-source
  licences are now recorded in-tree from upstream terms
  (this registry); the embedding-model share was already
  recorded by TF-053. TF-190's final pass remains an
  audit duty, named in the entries it must re-verify.
- **D-DATA-3 — FIXED (TF-106 share).** Provenance for the
  external sets is recorded in the registry (code + this
  doc), not only in code comments. No file was placed
  inside `data/` itself — that directory is the owner's
  canon and the campaign does not write into it; the
  registry in `src/` is the record.
- **D-ETL-2 — filed OPEN, owner TF-112 + TF-119.** The
  builders ingest without consulting the registry: a
  quarantined or unregistered source has no build-time
  gate today. The registry's static checker pins the
  filed shape (no builder module imports the registry),
  so wiring a builder turns it red until the entry is
  updated in the open.

**Proof grade: FIXTURE** — the registry is checked
against the live TF-105 map and the gate/classifier are
exercised on the filed entries and synthetic candidates,
on this VM. No source was fetched or ingested at scale
for this slice, and no legal advice is claimed: entries
record upstream terms as published, with TF-190 owning
the final re-verification.
