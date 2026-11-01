"""
Source registry, as data (TF-106).

Every source ThoughtForge ingests carries provenance and a
licence, recorded here from its upstream terms — the ledger
TF-008 finding 7 and TF-007 finding 4 found missing (data-
source licences were "a separate, unverified ledger",
recorded nowhere in-tree; provenance lived only in ETL code
comments, D-LIC-2 / D-DATA-3):

  * ``SOURCES`` — one entry per TF-105 builder, plus one
    sub-source entry for the D&D 5e SRD inside the reference
    library: provenance (who made it, from what), the licence
    exactly as its upstream terms state it, the obligations
    that licence carries, and the upstream page the terms
    were read from (verified 2026-11-01).
  * ``REGISTRY_RULES`` — the registry's law as data. The
    load-bearing rule is the quarantine rule: a source with
    no licence verified in fact is QUARANTINED — it may be
    ingested into a private local store only, it may never
    be redistributed, and no licence claim may be made for
    it. Quarantine is a disposition, not a deletion.
  * ``check_source_use()`` — the gate: licensed sources pass
    ingestion and (subject to their filed obligations)
    redistribution; a quarantined source raises
    ``QuarantinedSourceError`` on redistribution and passes
    private-local ingestion with its quarantine stated; an
    unregistered source raises ``ValueError`` by name — a
    new source is quarantined by default until it is filed
    here with provenance and a verified licence.
  * ``classify_candidate()`` — the same rule applied to a
    proposed new source before it is filed: missing
    provenance or an unverified licence quarantines it, with
    the missing pieces named.

The measured truth this registry records about the shipped
estate:

  * Four external sources are licensed in fact: Wikidata
    structured data under CC0 1.0; ConceptNet 5's complete
    data under CC BY-SA 4.0 (provenance-tracked subsets may
    be extracted under CC BY 4.0 alone, per its data
    credits — the assertions builder ingests the complete
    data, so BY-SA is the filed licence); GeoNames under
    CC BY 4.0; DBpedia dual-licensed CC BY-SA 3.0 + GFDL
    from release 3.4 onward.
  * The built-in reference library has NO single licence in
    fact: TF-007's census shows a mixed library — the
    owner's own authored sets, public-domain-classic texts
    in unverified editions, third-party material, and one
    separately licensed sub-source (the SRD). It is filed
    QUARANTINED as a whole: private local ingestion only
    (which is also the canon law for that library), no
    redistribution, no licence claim for the whole.
  * The SRD sub-source is licensed in fact on its own:
    SRD 5.1 was released by Wizards of the Coast under
    CC BY 4.0 (2023-01-27); the in-tree markdown copy's own
    legal chapter is the copy's terms and TF-190's final
    licence pass re-verifies it against the filed entry.

Filed limit, stated plainly: the builders do not yet call
``check_source_use()`` — nothing in the BUILDERS consults
this registry at build time, so the quarantine is enforced
at the registry/gate surface, not yet inside a build. That
wiring is docketed as D-ETL-2 (OPEN for TF-119's share;
TF-112's dry-run — etl/ingestion_dry_run.py, a consumer,
not a builder — now consults the registry first, as
filed, and discharges D-ETL-2's TF-112 share; TF-119's
new-source recipe must file a source here before
documenting it). No builder's behaviour changes in this
slice.

The human-readable write-up is
``docs/architecture/SOURCE_REGISTRY.md``. Proof grade:
FIXTURE — the registry and gate are checked against the
live TF-105 map and exercised on synthetic candidates on
this VM; no source was fetched or ingested at scale, and
no legal advice is claimed — the entries record upstream
terms as published, and TF-190 owns the final pass.
"""

from __future__ import annotations

from typing import Any

REGISTRY_FORMAT = "thoughtforge-source-registry/1"
REGISTRY_DOC = "docs/architecture/SOURCE_REGISTRY.md"

STATUS_LICENSED = "licensed"
STATUS_OBLIGATIONS = "licensed_with_obligations"
STATUS_QUARANTINED = "quarantined_unverified"

USE_INGEST_PRIVATE = "ingest_private"
USE_REDISTRIBUTE = "redistribute"
USES = (USE_INGEST_PRIVATE, USE_REDISTRIBUTE)


class QuarantinedSourceError(ValueError):
    """A quarantined source was asked to do what quarantine forbids."""


# ── The registry ────────────────────────────────────────────────────────────
#
# "builder_id" binds each entry to TF-105's SOURCE_BUILDERS.
# The SRD is a sub-source: it rides the reference builder but
# only for files under its path marker. A quarantined entry's
# licence is None by construction — quarantine means no
# licence was verified in fact, so none may be stated.

SOURCES: tuple[dict[str, Any], ...] = (
    {"id": "wikidata", "builder_id": "wikidata", "sub_path": None,
     "name": "Wikidata",
     "provenance": "Wikimedia Foundation / Wikidata community: the structured-data knowledge base (entities, statements, labels) the WikidataETL streams from the JSON dump",
     "licence": "CC0 1.0 (structured data in the main / Property / Lexeme / EntitySchema namespaces)",
     "status": STATUS_LICENSED,
     "obligations": (),
     "upstream_terms": "Wikidata:Licensing (wikidata.org), read 2026-11-01: all structured data is released into the public domain under Creative Commons Zero; text in other namespaces is CC BY-SA 4.0 and is not what this builder ingests",
     "notes": "CC0 carries no attribution or share-alike duty; the filed scope is the structured dump only"},
    {"id": "conceptnet", "builder_id": "conceptnet", "sub_path": None,
     "name": "ConceptNet 5",
     "provenance": "Commonsense Computing Initiative (Luminoso / MIT Media Lab lineage): commonsense assertions compiled from Open Mind Common Sense, Wiktionary, DBpedia, WordNet and other credited sources, ingested by ConceptNetETL from the assertions CSV",
     "licence": "CC BY-SA 4.0 (the complete data)",
     "status": STATUS_OBLIGATIONS,
     "obligations": ("attribution", "share_alike"),
     "upstream_terms": "ConceptNet 5 DATA-CREDITS / copying-and-sharing terms, read 2026-11-01: the complete data is available under CC BY-SA 4.0; because provenance is tracked per assertion, a subset may be extracted under CC BY 4.0 alone",
     "notes": "the builder ingests the complete assertions file, so BY-SA is the filed licence; a provenance-filtered subset extraction could claim BY alone, but no such extraction exists in the tree"},
    {"id": "geonames", "builder_id": "geonames", "sub_path": None,
     "name": "GeoNames",
     "provenance": "GeoNames (geonames.org): the geographical database compiled from official public sources (NGA GEOnet Names Server, national mapping agencies and others), ingested by GeoNamesETL from the tab-delimited dump",
     "licence": "CC BY 4.0",
     "status": STATUS_OBLIGATIONS,
     "obligations": ("attribution",),
     "upstream_terms": "GeoNames About page (geonames.org), read 2026-11-01: the geographical database is available for download free of charge under a Creative Commons Attribution licence; the dump's own readme states CC BY 4.0",
     "notes": "individual upstream feeds inside GeoNames carry their own credits on the About page; the database as downloaded is CC BY 4.0"},
    {"id": "dbpedia", "builder_id": "dbpedia", "sub_path": None,
     "name": "DBpedia",
     "provenance": "DBpedia project (Leipzig / Mannheim / OpenLink lineage): structured content extracted from Wikipedia, ingested by DBpediaETL from NDJSON abstracts",
     "licence": "CC BY-SA 3.0 + GNU Free Documentation License (dual, from release 3.4 onward)",
     "status": STATUS_OBLIGATIONS,
     "obligations": ("attribution", "share_alike"),
     "upstream_terms": "DBpedia dataset licence note (downloads.dbpedia.org data-set pages), read 2026-11-01: DBpedia is distributed under the same licensing terms as Wikipedia; from release 3.4 the data is dual-licensed CC BY-SA 3.0 and GFDL (releases up to 3.3 were GFDL only)",
     "notes": "the builder's input is a JSON export rather than a numbered release dump, so the entry files the dual licence and names the pre-3.4 GFDL-only leg; TF-190 re-verifies against the actual export used"},
    {"id": "reference_library", "builder_id": "reference", "sub_path": None,
     "name": "Built-in reference library (data/knowledge_reference/)",
     "provenance": "Mixed, per TF-007's census: the owner's own authored sets and Sigrid knowledge-build outputs, public-domain-classic texts in editions whose licence is not recorded in-tree, topical authored sets, and externally installed files the library was designed to receive — no single origin",
     "licence": None,
     "status": STATUS_QUARANTINED,
     "obligations": ("no_redistribution", "no_licence_claim_for_the_whole"),
     "upstream_terms": "No upstream terms exist for the library as a whole — that absence is the finding (TF-007 finding 4, TF-008 finding 7)",
     "notes": "QUARANTINED as a whole: private local ingestion only (the library is also the owner's canon, private by his law); a file family may leave quarantine only by being filed as its own entry with verified terms, as the SRD is below"},
    {"id": "dnd_5e_srd", "builder_id": "reference",
     "sub_path": "dnd-5e-srd-markdown/",
     "name": "D&D 5e System Reference Document (in-library copy)",
     "provenance": "Wizards of the Coast: the SRD chapters held inside the reference library under dnd-5e-srd-markdown/ (TF-007: 17 files, chapters 00legal-16npcs)",
     "licence": "CC BY 4.0 (SRD 5.1, released 2023-01-27)",
     "status": STATUS_OBLIGATIONS,
     "obligations": ("attribution",),
     "upstream_terms": "Wizards of the Coast's SRD 5.1 release under CC BY 4.0 (2023) and the SRD legal text (CC-BY-4.0, attribution statement prescribed), read 2026-11-01",
     "notes": "the in-tree markdown copy predates or parallels that release and carries its own legal chapter (00legal); the filed licence is SRD 5.1's CC BY 4.0 and TF-190's final pass re-verifies the copy against it — until then the SRD rides its parent builder under the parent entry's quarantine for any whole-library claim"},
)

REGISTRY_RULES: tuple[dict[str, Any], ...] = (
    {"id": "every_builder_has_exactly_one_entry",
     "rule": "every id in TF-105's SOURCE_BUILDERS is the builder_id of exactly one primary (non-sub-source) registry entry"},
    {"id": "every_entry_carries_provenance_and_terms",
     "rule": "no entry exists without provenance and a named upstream-terms source — a registry row is evidence, not a label"},
    {"id": "unlicensed_in_fact_is_quarantined",
     "rule": "a source whose licence is not verified in fact is quarantined: private local ingestion only, no redistribution, no licence claim"},
    {"id": "quarantine_states_no_licence",
     "rule": "a quarantined entry's licence is None — the registry never prints a licence it did not verify"},
    {"id": "an_unregistered_source_is_quarantined_by_default",
     "rule": "check_source_use raises by name for an unregistered id; a new source enters only by being filed here with provenance + verified licence (classify_candidate applies the same rule to proposals)"},
    {"id": "obligations_ride_the_licence",
     "rule": "attribution / share-alike duties are filed per entry and returned by the gate, so a redistributing consumer sees the duties with the permission"},
    {"id": "a_sub_source_may_leave_its_parents_quarantine_only_on_its_own_terms",
     "rule": "a sub-source entry (the SRD) carries its own verified licence for its own path; it does not lift the parent library's quarantine, and the parent's quarantine governs any whole-library claim"},
    {"id": "the_registry_is_the_record_not_the_fetch",
     "rule": "entries record upstream terms as published and name where they were read; no entry asserts a licence for data this campaign has not examined (TF-190 owns the final re-verification)"},
)


def get_source(source_id: str) -> dict[str, Any]:
    """The registry entry for ``source_id`` (raises by name)."""
    for entry in SOURCES:
        if entry["id"] == source_id:
            return entry
    raise ValueError(
        f"unregistered source {source_id!r}: quarantined by default "
        "until filed in the source registry with provenance and a "
        "verified licence (TF-106)")


def classify_candidate(name: str, provenance: str | None,
                       licence: str | None) -> dict[str, Any]:
    """Judge a proposed new source by the quarantine rule.

    Returns the verdict as data: quarantined with the missing
    pieces named, or admissible for filing (which still means
    a human files the entry — this function grants nothing).
    """
    missing: list[str] = []
    if not provenance or not provenance.strip():
        missing.append("provenance")
    if not licence or not licence.strip():
        missing.append("licence")
    if missing:
        return {"name": name, "status": STATUS_QUARANTINED,
                "missing": tuple(missing),
                "reason": "no source is admitted on an unverified "
                          "licence: " + " and ".join(missing) +
                          " not established"}
    return {"name": name, "status": "admissible_for_filing",
            "missing": (),
            "reason": "provenance and licence both stated; the "
                      "source may be filed in SOURCES with its "
                      "upstream terms named"}


def check_source_use(source_id: str, use: str) -> dict[str, Any]:
    """The gate: may ``source_id`` be used for ``use``?

    Returns the entry (with its obligations) when the use is
    allowed. Raises ``ValueError`` for an unknown use or an
    unregistered source, and ``QuarantinedSourceError`` when
    a quarantined source is asked to be redistributed.
    """
    if use not in USES:
        raise ValueError(
            f"unknown source use {use!r}: expected one of {USES}")
    entry = get_source(source_id)
    if entry["status"] == STATUS_QUARANTINED and use == USE_REDISTRIBUTE:
        raise QuarantinedSourceError(
            f"source {source_id!r} is quarantined (no licence "
            "verified in fact): redistribution is forbidden; "
            "private local ingestion only, with no licence "
            "claim for the whole")
    return entry


# ── Checkers ────────────────────────────────────────────────────────────────

def registry_problems() -> list[str]:
    """Static truth: the registry against the live TF-105 map."""
    from pathlib import Path

    from thoughtforge.etl import sources as sources_module
    from thoughtforge.etl import wikidata as wikidata_module
    from thoughtforge.etl.etl_map import SOURCE_BUILDERS

    problems: list[str] = []
    # D-ETL-2's filed shape: no builder consults this registry
    # yet. Wiring one turns this checker red until the docket
    # entry and the module docstring are updated in the open.
    for module in (sources_module, wikidata_module):
        text = Path(module.__file__).read_text(encoding="utf-8")
        if "source_registry" in text:
            problems.append(
                f"D-ETL-2 changed shape: {module.__name__} now "
                "consults the source registry — update the "
                "docket entry and this slice's filed limit")
    rule_ids = [r["id"] for r in REGISTRY_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate REGISTRY_RULES ids")
    entry_ids = [e["id"] for e in SOURCES]
    if len(entry_ids) != len(set(entry_ids)):
        problems.append("duplicate SOURCES ids")

    builder_ids = [b["id"] for b in SOURCE_BUILDERS]
    primary = [e for e in SOURCES if e["sub_path"] is None]
    for builder_id in builder_ids:
        matches = [e for e in primary if e["builder_id"] == builder_id]
        if len(matches) != 1:
            problems.append(
                f"builder {builder_id!r} has {len(matches)} primary "
                "registry entries (exactly one required)")
    for entry in SOURCES:
        if entry["builder_id"] not in builder_ids:
            problems.append(
                f"entry {entry['id']!r} binds unknown builder "
                f"{entry['builder_id']!r}")
        for field in ("name", "provenance", "upstream_terms", "notes"):
            if not entry[field]:
                problems.append(f"entry {entry['id']!r} has no {field}")
        if entry["status"] == STATUS_QUARANTINED:
            if entry["licence"] is not None:
                problems.append(
                    f"quarantined entry {entry['id']!r} states a "
                    "licence — quarantine means none was verified")
            if "no_redistribution" not in entry["obligations"]:
                problems.append(
                    f"quarantined entry {entry['id']!r} does not "
                    "carry the no_redistribution obligation")
        else:
            if not entry["licence"]:
                problems.append(
                    f"licensed entry {entry['id']!r} states no licence")
            if entry["status"] not in (STATUS_LICENSED,
                                       STATUS_OBLIGATIONS):
                problems.append(
                    f"entry {entry['id']!r} has unknown status "
                    f"{entry['status']!r}")
        if entry["sub_path"] is not None and not entry["sub_path"].endswith("/"):
            problems.append(
                f"sub-source {entry['id']!r} path marker does not "
                "end with '/'")
    return problems


def registry_live_problems() -> list[str]:
    """Behavioural truth: the gate and the classifier, exercised."""
    problems: list[str] = []

    for entry in SOURCES:
        try:
            got = check_source_use(entry["id"], USE_INGEST_PRIVATE)
        except Exception as exc:  # a refused private ingest is a finding
            problems.append(
                f"private ingestion refused for {entry['id']!r}: {exc!r}")
            continue
        if got["id"] != entry["id"]:
            problems.append(f"gate returned {got['id']!r} for "
                            f"{entry['id']!r}")
        if entry["status"] == STATUS_QUARANTINED:
            try:
                check_source_use(entry["id"], USE_REDISTRIBUTE)
                problems.append(
                    f"quarantined source {entry['id']!r} was allowed "
                    "to redistribute")
            except QuarantinedSourceError:
                pass
        else:
            try:
                check_source_use(entry["id"], USE_REDISTRIBUTE)
            except Exception as exc:
                problems.append(
                    f"licensed source {entry['id']!r} refused "
                    f"redistribution: {exc!r}")

    try:
        check_source_use("no_such_source_tf106", USE_INGEST_PRIVATE)
        problems.append("an unregistered source was admitted")
    except ValueError:
        pass
    try:
        check_source_use("wikidata", "publish_everywhere")
        problems.append("an unknown use was admitted")
    except ValueError:
        pass

    bare = classify_candidate("bare_source", None, None)
    if bare["status"] != STATUS_QUARANTINED or \
            bare["missing"] != ("provenance", "licence"):
        problems.append(f"a bare candidate was not quarantined: {bare}")
    half = classify_candidate("half_source", "someone's scrape", None)
    if half["status"] != STATUS_QUARANTINED or \
            half["missing"] != ("licence",):
        problems.append(f"a licence-less candidate was not "
                        f"quarantined: {half}")
    full = classify_candidate("full_source", "a named publisher",
                              "CC BY 4.0")
    if full["status"] != "admissible_for_filing" or full["missing"] != ():
        problems.append(f"a complete candidate was quarantined: {full}")
    return problems


def registry_as_data() -> dict[str, Any]:
    """The whole registry as plain data, for consumers and the doctor."""
    return {
        "format": REGISTRY_FORMAT,
        "doc": REGISTRY_DOC,
        "sources": [dict(s) for s in SOURCES],
        "rules": [dict(r) for r in REGISTRY_RULES],
        "uses": list(USES),
    }
