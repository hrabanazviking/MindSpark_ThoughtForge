"""
Dedup & merge rules for overlapping sources, as data (TF-109).

TF-105 mapped the ingestion forge, TF-106 registered its
sources, TF-107 made the wikidata pipeline resumable and
statement-idempotent, TF-108 gated malformed records.
None of them said what happens when two records claim
the same row — or the same real-world thing. TF-109's
probes measured the starting state through the real
builders:

  * **A re-run silently ERRORED complementary data
    (D-MERGE-1).** A wikidata entity built with a
    description, aliases and popularity 0.02, then built
    again (new checkpoint, ``resume=False``) with a
    changed label and NO description in the dump, came
    back with ``description_en`` NULL, ``aliases_en``
    NULL and ``popularity_score`` 0.0: the flush's
    ``INSERT OR REPLACE`` replaced the whole row with the
    thinner one — fields the second dump did not carry
    were erased, not merged, and nothing recorded it.
  * **ConceptNet had no semantic identity (D-MERGE-2).**
    Two assertion lines asserting the same edge
    (lantern / IsA / light) under different assertion
    uris (``/a/1`` weight 2.0, ``/a/2`` weight 5.0)
    landed as TWO rows — the table's only identity was
    the uri, so the same fact from an overlapping source
    duplicated. And the same uri re-asserted with a
    STRONGER weight (9.0) was dropped by
    ``INSERT OR IGNORE``: the store kept 2.0, the
    stronger assertion was lost, and the stats still
    counted all three lines "inserted" (3 counted,
    2 rows).
  * **Every other collision was a silent overwrite
    (D-MERGE-3).** A GeoNames place re-built with a new
    name and population, and a DBpedia entity re-built
    with a new abstract, replaced the stored rows whole
    (``INSERT OR REPLACE``) with no record anywhere of
    the old value, the new value, or that a collision
    happened at all — TF-050's rule for the profile
    store (*surfaced, scored, resolved by the stated
    rule, never silently overwritten*) had no counterpart
    on the ingestion side. Labels and reference chunks
    replaced the same silent way.
  * **Cross-source overlap was invisible (D-MERGE-4).**
    The same synthetic town ingested from wikidata
    (entity Q1, "Lantern Town"), GeoNames (place 109001,
    "Lantern Town") and DBpedia (uri .../Lantern_Town,
    "Lantern Town") sat in three tables as three
    unlinked rows: no shared key, no link, and no
    function in the tree could name the overlap — a
    reader (or TF-110's subset craft, or TF-112's
    dry-run) had no way to know three sources were
    talking about one thing.

The rules, filed here as data and enforced in the
builders:

  * ``TABLE_IDENTITIES`` — per table, the key that makes
    a row *the same row* and the value fields a merge
    judges. Statements' identity is the whole stored
    tuple (TF-107's idempotence, now filed as a merge
    rule); ConceptNet edges carry TWO identities — the
    uri (provenance) and the semantic key
    (concept1, relation, concept2, language) — and dedup
    judges the semantic one.
  * ``decide_merge(existing, incoming, value_fields)`` —
    the pure judgement over one collision, consuming
    TF-050's contradiction rule in the form the
    ingestion side can honour:

      - no field differs -> ``duplicate``: the row is
        NOT rewritten;
      - fields differ only where one side is empty ->
        ``merged_complementary``: the merged row is the
        union — an empty incoming field NEVER erases a
        stored value, and a stored gap is filled;
      - some field carries two different non-empty
        values -> ``conflict_overwrite``: the conflict
        is recorded in the builder's ``MergeLedger``
        with the field, the stored value and the
        incoming value (truncated, never whole texts)
        BEFORE the incoming value wins. Within one
        source's own table the incoming row is that
        source's newest statement of itself, so it
        wins — but the overwrite is surfaced, never
        silent. TF-114 owns replacing this default with
        lifecycle update rules, and must update this
        module in the open when it does.

  * ConceptNet edges resolve by **strength** instead,
    because two provenances assert the same edge there
    (the assertion file and ``build_from_reference``):
    the edge's own weight is its strength (TF-050's
    scoring consumed in miniature) — the stronger
    assertion stands (``conflict_stronger_wins`` /
    ``conflict_first_stands``), an exact tie keeps the
    first row, and every resolution is ledgered.
  * ``MergeLedger`` — the ledger each builder holds,
    fresh per build, beside its TF-108 ``SchemaGate``:
    every COLLISION (duplicate / merged / conflict) is
    recorded with table, key, outcome and per-field
    detail; new rows are not collisions and are the
    stats' job, exactly as the gate ledger records only
    refusals. Builder stats shapes do not change
    (TF-105's filed shapes stand) — the ledger is the
    truth about what the writes did, the stats remain
    the count of records accepted, and the filed limit
    below names the gap where those disagree.
  * ``find_overlaps(db_path)`` — cross-source overlap
    as a FINDING, never a merge: rows in different
    tables are grouped by normalised name (entities'
    labels, DBpedia labels, GeoNames names, ConceptNet
    concept texts); a group with members from at least
    two tables is an overlap group, returned as data
    with every member's table, key and source. Dedup
    NEVER merges or deletes across tables — the sources
    keep their provenance (TF-106's registry), and a
    cross-source value conflict is for a reader to see,
    not for ingestion to settle silently. TF-110 /
    TF-112 are the filed first consumers.

Docket (TF-016): D-MERGE-1 … D-MERGE-4 are added and
FIXED by this slice. D-DATA-2 (duplicate blobs inside
``data/``) is narrowed: its store-level share — the
same fact duplicated inside the knowledge store — is
fixed here; the duplicate FILES in ``data/`` are
Volmarr's canon, nothing is moved or deleted by this
campaign, and that share stands OPEN for his word.

The human-readable write-up is
``docs/architecture/DEDUP_AND_MERGE.md``. Proof grade:
FIXTURE — synthetic overlapping records through the
real builders on this VM; no real dump's overlap rate
has been measured and none is claimed.
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

MERGE_FORMAT = "thoughtforge-dedup-merge/1"
MERGE_DOC = "docs/architecture/DEDUP_AND_MERGE.md"

#: Field values in a ledger conflict entry are truncated
#: to this many characters: the conflict is surfaced
#: (which field, from what, to what) without the ledger
#: becoming a second copy of the store's texts.
LEDGER_VALUE_LIMIT = 120

# ── Identities ──────────────────────────────────────────────────────────────
#
# One entry per merged table: the key fields that make a
# row the same row, and the value fields a merge judges.
# Columns a flush writes that are neither (a constant
# ``source``) ride the write and are never judged.

TABLE_IDENTITIES: tuple[dict[str, Any], ...] = (
    {"table": "entities", "builder": "WikidataETL",
     "key": ("qid",),
     "value_fields": ("label_en", "description_en", "aliases_en",
                      "instance_of", "popularity_score"),
     "identity": "the Wikidata QID"},
    {"table": "labels", "builder": "WikidataETL",
     "key": ("qid", "language"),
     "value_fields": ("label",),
     "identity": "the QID in one language"},
    {"table": "statements", "builder": "WikidataETL",
     "key": ("subject_qid", "property_pid", "object_value",
             "object_type", "qualifiers", "refs"),
     "value_fields": (),
     "identity": "the whole stored tuple — a statement has no "
                 "value field apart from its identity (TF-107)"},
    {"table": "conceptnet_edges", "builder": "ConceptNetETL",
     "key": ("uri",),
     "semantic_key": ("concept1", "relation", "concept2", "language"),
     "value_fields": ("weight",),
     "identity": "the assertion uri as provenance; the semantic "
                 "key (concept1, relation, concept2, language) as "
                 "the fact — dedup judges the semantic key"},
    {"table": "geonames_places", "builder": "GeoNamesETL",
     "key": ("geonames_id",),
     "value_fields": ("name", "ascii_name", "latitude", "longitude",
                      "feature_class", "feature_code", "country_code",
                      "population", "elevation", "timezone",
                      "admin1_code", "admin2_code"),
     "identity": "the GeoNames id"},
    {"table": "dbpedia_entities", "builder": "DBpediaETL",
     "key": ("uri",),
     "value_fields": ("label_en", "abstract_en", "types"),
     "identity": "the DBpedia uri"},
    {"table": "reference_chunks", "builder": "ReferenceDataETL",
     "key": ("source_file", "chunk_index"),
     "value_fields": ("title", "content", "tags", "word_count"),
     "identity": "the source file + chunk position within it"},
)

#: Collision outcomes the ledger records.
MERGE_OUTCOMES: tuple[str, ...] = (
    "duplicate", "merged_complementary", "conflict_overwrite",
    "conflict_stronger_wins", "conflict_first_stands",
)


# ── The rules ───────────────────────────────────────────────────────────────

MERGE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "identity_is_the_filed_key",
     "rule": "two rows are the same row exactly when their TABLE_IDENTITIES key agrees; statements' key is the whole stored tuple, and a ConceptNet edge is judged by its semantic key, not its assertion uri alone"},
    {"id": "exact_duplicate_is_not_rewritten",
     "rule": "a collision in which no value field differs is a duplicate: the stored row is not rewritten, and the decision is recorded in the builder's MergeLedger"},
    {"id": "complementary_fields_merge_never_erase",
     "rule": "a collision differing only where one side is empty (None or '') merges to the union of both rows: an empty incoming field never erases a stored value, and a stored gap is filled from the incoming row (D-MERGE-1)"},
    {"id": "conflicts_are_surfaced_never_silent",
     "rule": "a field carrying two different non-empty values is a conflict in TF-050's sense: it is recorded in the MergeLedger with the field name and both values (truncated to LEDGER_VALUE_LIMIT) before any resolution writes — no overwrite on the ingestion side is silent any more (D-MERGE-3)"},
    {"id": "within_a_table_the_incoming_row_wins",
     "rule": "within one source's own table, a recorded conflict resolves to the incoming row — it is that source's newest statement of itself. TF-114 owns replacing this default with lifecycle update rules and must update this module in the open"},
    {"id": "conceptnet_edges_resolve_by_strength",
     "rule": "a ConceptNet semantic-key collision resolves by the edges' own weights, TF-050's strength rule in miniature: the stronger assertion stands and the weaker is not written; an exact tie keeps the first row; the surviving row keeps the stronger assertion's uri and weight (D-MERGE-2)"},
    {"id": "cross_source_overlap_is_a_finding_not_a_merge",
     "rule": "rows in different tables are never merged, rewritten or deleted by dedup: find_overlaps groups them by normalised name and returns the group as data, provenance intact (TF-106); settling a cross-source disagreement is a reader's judgement, not ingestion's (D-MERGE-4)"},
    {"id": "overlap_names_are_normalised_not_altered",
     "rule": "overlap matching compares a derived key only — casefolded, underscores/hyphens folded to spaces, punctuation stripped, whitespace collapsed; stored texts are never modified to make them match"},
    {"id": "the_ledger_records_collisions_only",
     "rule": "a builder's MergeLedger is fresh per build and records every collision with table, key, outcome and per-field detail; a row with no stored counterpart is not a collision and is the builder stats' job, as the TF-108 gate ledger records only refusals"},
    {"id": "stats_shapes_do_not_change",
     "rule": "builder stats keep TF-105's filed shapes and keep counting records accepted; where an accepted record's write was a duplicate or a lost conflict, the store's rows and the MergeLedger — not the stats — are the truth, and TF-115 owns reporting the difference per run"},
)


# ── The pure judgement ──────────────────────────────────────────────────────

def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _ledger_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > LEDGER_VALUE_LIMIT:
        return value[:LEDGER_VALUE_LIMIT] + "…"
    return value


def decide_merge(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    value_fields: tuple[str, ...] | list[str],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """
    Judge one collision. Returns ``(outcome, merged, detail)``:

      * ``duplicate`` — no value field differs; ``merged``
        is the existing row and must not be rewritten.
      * ``merged_complementary`` — differences are only
        empty-vs-filled; ``merged`` is the union (a filled
        field is never erased by an empty one).
      * ``conflict_overwrite`` — at least one field
        carries two different non-empty values; ``merged``
        takes the incoming value on conflicted fields and
        the union elsewhere. ``detail`` names every
        differing field with both values and how it
        resolved (``filled`` / ``kept_stored_empty_incoming``
        / ``incoming_wins``).
    """
    merged = dict(existing)
    detail: list[dict[str, Any]] = []
    conflict = False
    differs = False
    for field_name in value_fields:
        old = existing.get(field_name)
        new = incoming.get(field_name)
        if old == new:
            continue
        differs = True
        if _is_empty(old) and not _is_empty(new):
            merged[field_name] = new
            detail.append({"field": field_name, "stored": _ledger_value(old),
                           "incoming": _ledger_value(new),
                           "resolution": "filled"})
        elif _is_empty(new) and not _is_empty(old):
            detail.append({"field": field_name, "stored": _ledger_value(old),
                           "incoming": _ledger_value(new),
                           "resolution": "kept_stored_empty_incoming"})
        else:
            conflict = True
            merged[field_name] = new
            detail.append({"field": field_name, "stored": _ledger_value(old),
                           "incoming": _ledger_value(new),
                           "resolution": "incoming_wins"})
    if not differs:
        return "duplicate", dict(existing), []
    if conflict:
        return "conflict_overwrite", merged, detail
    return "merged_complementary", merged, detail


# ── The ledger ──────────────────────────────────────────────────────────────

class MergeLedger:
    """The collision ledger a builder holds for one run."""

    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []

    def record(self, table: str, key: dict[str, Any], outcome: str,
               detail: list[dict[str, Any]] | None = None) -> None:
        if outcome not in MERGE_OUTCOMES:
            raise ValueError(
                f"unknown merge outcome {outcome!r}: the ledger "
                f"records {list(MERGE_OUTCOMES)}")
        self.decisions.append({
            "table": table, "key": dict(key), "outcome": outcome,
            "detail": [dict(d) for d in (detail or [])],
        })

    @property
    def collisions(self) -> int:
        return len(self.decisions)

    def by_outcome(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.decisions:
            counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1
        return counts

    def by_table(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.decisions:
            counts[entry["table"]] = counts.get(entry["table"], 0) + 1
        return counts

    def as_data(self) -> dict[str, Any]:
        return {
            "format": MERGE_FORMAT,
            "collisions": self.collisions,
            "by_outcome": self.by_outcome(),
            "by_table": self.by_table(),
            "decisions": [dict(d) for d in self.decisions],
        }


# ── Applying a merge ────────────────────────────────────────────────────────

def _identity_for(table: str) -> dict[str, Any]:
    for entry in TABLE_IDENTITIES:
        if entry["table"] == table:
            return entry
    raise ValueError(
        f"no filed identity for table {table!r}: file it in "
        "TABLE_IDENTITIES before merging into it")


def apply_row_merge(
    conn: sqlite3.Connection,
    ledger: MergeLedger,
    table: str,
    incoming: dict[str, Any],
) -> str:
    """
    Write one row through the merge rules and return the
    outcome. ``incoming`` carries every column to write;
    columns outside the filed key + value fields (a
    constant ``source``) ride the write unjudged. A
    ``duplicate`` writes nothing; any other outcome
    writes the merged row exactly once.
    """
    identity = _identity_for(table)
    key_fields = tuple(identity["key"])
    value_fields = tuple(identity["value_fields"])
    where = " AND ".join(f"{name} = ?" for name in key_fields)
    row = conn.execute(
        f"SELECT * FROM {table} WHERE {where}",
        tuple(incoming.get(name) for name in key_fields),
    ).fetchone()
    if row is None:
        columns = list(incoming)
        conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES "
            f"({', '.join('?' * len(columns))})",
            tuple(incoming[name] for name in columns),
        )
        return "new"
    existing = dict(row)
    key = {name: incoming.get(name) for name in key_fields}
    outcome, merged, detail = decide_merge(existing, incoming, value_fields)
    ledger.record(table, key, outcome, detail)
    if outcome == "duplicate":
        return outcome
    for name, value in incoming.items():
        if name not in merged:
            merged[name] = value
    columns = [name for name in incoming if name in merged]
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES "
        f"({', '.join('?' * len(columns))})",
        tuple(merged[name] for name in columns),
    )
    return outcome


def merge_conceptnet_edge(
    conn: sqlite3.Connection,
    ledger: MergeLedger,
    row: dict[str, Any],
) -> str:
    """
    Write one ConceptNet edge through the semantic dedup
    and return the outcome. The edge is looked up by uri
    first, then by semantic key; a collision resolves by
    weight (the stronger assertion stands, a tie keeps
    the first row) and is ledgered either way.
    """
    existing_row = conn.execute(
        "SELECT * FROM conceptnet_edges WHERE uri = ?",
        (row["uri"],),
    ).fetchone()
    if existing_row is None:
        existing_row = conn.execute(
            "SELECT * FROM conceptnet_edges WHERE concept1 = ? "
            "AND relation = ? AND concept2 = ? AND language = ?",
            (row["concept1"], row["relation"], row["concept2"],
             row["language"]),
        ).fetchone()
    if existing_row is None:
        conn.execute(
            "INSERT INTO conceptnet_edges "
            "(uri, concept1, relation, concept2, weight, language) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row["uri"], row["concept1"], row["relation"],
             row["concept2"], row["weight"], row["language"]),
        )
        return "new"
    existing = dict(existing_row)
    key = {"concept1": row["concept1"], "relation": row["relation"],
           "concept2": row["concept2"], "language": row["language"]}
    if existing["uri"] == row["uri"] \
            and float(existing["weight"]) == float(row["weight"]):
        ledger.record("conceptnet_edges", key, "duplicate", [])
        return "duplicate"
    detail = [
        {"field": "weight", "stored": existing["weight"],
         "incoming": row["weight"], "resolution": ""},
        {"field": "uri", "stored": existing["uri"],
         "incoming": row["uri"], "resolution": ""},
    ]
    if float(row["weight"]) > float(existing["weight"]):
        for entry in detail:
            entry["resolution"] = "incoming_wins"
        ledger.record("conceptnet_edges", key,
                      "conflict_stronger_wins", detail)
        conn.execute(
            "UPDATE conceptnet_edges SET uri = ?, weight = ? "
            "WHERE edge_id = ?",
            (row["uri"], row["weight"], existing["edge_id"]),
        )
        return "conflict_stronger_wins"
    for entry in detail:
        entry["resolution"] = "kept_stored"
    ledger.record("conceptnet_edges", key, "conflict_first_stands", detail)
    return "conflict_first_stands"


# ── Cross-source overlap ────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def normalise_name(value: Any) -> str:
    """The derived overlap key for a name/label: casefolded,
    underscores + hyphens folded to spaces, punctuation
    stripped, whitespace collapsed. Derived only — the
    stored text is never altered."""
    text = str(value or "").casefold().replace("_", " ").replace("-", " ")
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def find_overlaps(db_path: Path | str) -> list[dict[str, Any]]:
    """
    Group rows across the store's name-carrying tables by
    normalised name; return every group whose members
    come from at least two different tables, as data:
    ``{name, members: [{table, key, source, label}]}``
    sorted by name, members sorted by (table, key). A
    finding only — nothing is merged, rewritten or
    deleted (MERGE_RULES cross_source_overlap…).
    """
    from thoughtforge.etl.schema import get_connection

    members_by_name: dict[str, list[dict[str, Any]]] = {}

    def add(name: Any, member: dict[str, Any]) -> None:
        key = normalise_name(name)
        if key:
            members_by_name.setdefault(key, []).append(member)

    conn = get_connection(Path(db_path), read_only=True)
    try:
        for qid, label, source in conn.execute(
                "SELECT qid, label_en, source FROM entities"):
            add(label, {"table": "entities", "key": qid,
                        "source": source, "label": label})
        for uri, label, source in conn.execute(
                "SELECT uri, label_en, source FROM dbpedia_entities"):
            add(label, {"table": "dbpedia_entities", "key": uri,
                        "source": source, "label": label})
        for gid, name, source in conn.execute(
                "SELECT geonames_id, name, source FROM geonames_places"):
            add(name, {"table": "geonames_places", "key": gid,
                       "source": source, "label": name})
        seen_concepts: set[tuple[str, str]] = set()
        for concept, source in conn.execute(
                "SELECT concept1, source FROM conceptnet_edges"):
            if (concept, source) in seen_concepts:
                continue
            seen_concepts.add((concept, source))
            add(concept, {"table": "conceptnet_edges", "key": concept,
                          "source": source, "label": concept})
        for concept, source in conn.execute(
                "SELECT concept2, source FROM conceptnet_edges"):
            if (concept, source) in seen_concepts:
                continue
            seen_concepts.add((concept, source))
            add(concept, {"table": "conceptnet_edges", "key": concept,
                          "source": source, "label": concept})
    finally:
        conn.close()

    groups: list[dict[str, Any]] = []
    for name in sorted(members_by_name):
        members = sorted(
            members_by_name[name],
            key=lambda m: (m["table"], str(m["key"])))
        if len({m["table"] for m in members}) >= 2:
            groups.append({"name": name, "members": members})
    return groups


def overlap_as_data(db_path: Path | str) -> dict[str, Any]:
    groups = find_overlaps(db_path)
    return {
        "format": MERGE_FORMAT,
        "doc": MERGE_DOC,
        "groups": groups,
        "group_count": len(groups),
        "member_count": sum(len(g["members"]) for g in groups),
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def merge_problems() -> list[str]:
    """Static truth: the filed rules against the live builders."""
    from thoughtforge.etl import sources as sources_module
    from thoughtforge.etl import wikidata as wikidata_module

    problems: list[str] = []
    rule_ids = [r["id"] for r in MERGE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate MERGE_RULES ids")
    tables = [e["table"] for e in TABLE_IDENTITIES]
    if tables != ["entities", "labels", "statements", "conceptnet_edges",
                  "geonames_places", "dbpedia_entities",
                  "reference_chunks"]:
        problems.append(f"TABLE_IDENTITIES tables drifted: {tables}")
    for entry in TABLE_IDENTITIES:
        if not entry["key"] or not entry["identity"]:
            problems.append(f"{entry['table']}: identity not filed")

    sources_src = Path(sources_module.__file__).read_text(encoding="utf-8")
    wikidata_src = Path(wikidata_module.__file__).read_text(encoding="utf-8")
    for src, why in ((sources_src, "sources.py builders must merge"),
                     (wikidata_src, "the wikidata builder must merge")):
        if "from thoughtforge.etl.dedup_merge import" not in src:
            problems.append(f"a builder lost its dedup_merge import: {why}")
        if "merge_ledger" not in src:
            problems.append(f"a builder carries no merge_ledger: {why}")
    for cls_name in ("ConceptNetETL", "GeoNamesETL", "DBpediaETL",
                     "ReferenceDataETL"):
        cls = getattr(sources_module, cls_name)
        if not hasattr(cls(), "merge_ledger"):
            problems.append(f"{cls_name} carries no merge_ledger")
    if not hasattr(wikidata_module.WikidataETL(), "merge_ledger"):
        problems.append("WikidataETL carries no merge_ledger")
    if "merge_conceptnet_edge" not in sources_src:
        problems.append("ConceptNetETL flush lost its semantic merge")
    if "_flush_statements" in wikidata_src \
            and "existing" not in wikidata_src[
                wikidata_src.find("def _flush_statements"):]:
        problems.append("TF-107 statement idempotence guard vanished")
    return problems


def _synthetic_entity(qid: str, label: str, desc: str | None) -> dict[str, Any]:
    return {
        "id": qid,
        "labels": {"en": {"value": label}},
        "descriptions": {"en": {"value": desc}} if desc else {},
        "aliases": {"en": [{"value": "the lit town"}]} if desc else
        {"en": []},
        "sitelinks": {"enwiki": {}, "dewiki": {}},
        "claims": {"P31": [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": "Q109900"}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        }]},
    }


def merge_live_problems() -> list[str]:
    """Behavioural truth: the probed failures, re-run live."""
    import json as _json

    from thoughtforge.etl.sources import (
        ConceptNetETL,
        DBpediaETL,
        GeoNamesETL,
    )
    from thoughtforge.etl.wikidata import WikidataETL

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # D-MERGE-1: a thinner re-run must not erase, and a
        # conflicting label must be ledgered, not silent.
        db_path = root / "w.db"
        dump = root / "dump.json"
        dump.write_text(_json.dumps([_synthetic_entity(
            "Q1", "Lantern Town", "A synthetic town")]), encoding="utf-8")
        WikidataETL(db_path).build(dump_path=dump, resume=False)
        dump.write_text(_json.dumps([_synthetic_entity(
            "Q1", "Lantern Town Renamed", None)]), encoding="utf-8")
        etl = WikidataETL(db_path)
        etl.build(dump_path=dump, resume=False)
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT label_en, description_en, aliases_en "
                "FROM entities WHERE qid = 'Q1'").fetchone()
        finally:
            conn.close()
        if row != ("Lantern Town Renamed", "A synthetic town",
                   "the lit town"):
            problems.append(
                f"D-MERGE-1 reopened live: merged entity row is {row}")
        if etl.merge_ledger.by_outcome().get("conflict_overwrite", 0) < 1:
            problems.append(
                "D-MERGE-3 reopened live: the label overwrite was "
                f"not ledgered: {etl.merge_ledger.as_data()}")

        # D-MERGE-2: semantic duplicates collapse, the
        # stronger assertion stands.
        csv_path = root / "a.csv"
        csv_path.write_text("\n".join([
            "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 2.0}",
            "/a/2\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 5.0}",
            "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 9.0}",
        ]) + "\n", encoding="utf-8")
        cetl = ConceptNetETL(root / "c.db")
        cetl.build(csv_path=csv_path)
        conn = sqlite3.connect(str(root / "c.db"))
        try:
            edges = conn.execute(
                "SELECT uri, weight FROM conceptnet_edges").fetchall()
        finally:
            conn.close()
        if edges != [("/a/1", 9.0)]:
            problems.append(
                f"D-MERGE-2 reopened live: conceptnet rows are {edges}")

        # D-MERGE-4: the same town from three sources is
        # one overlap group naming all three tables. (Q1's
        # label was renamed by the D-MERGE-1 leg above, so
        # a fresh Q2 carries the shared name here.)
        dump.write_text(_json.dumps([_synthetic_entity(
            "Q2", "Lantern Town", "A synthetic town")]), encoding="utf-8")
        WikidataETL(db_path).build(dump_path=dump, resume=False)
        base = ["109001", "Lantern Town", "Lantern Town", "", "1.5",
                "2.5", "P", "PPL", "XX", "", "A1", "A2", "", "", "100",
                "10", "10", "UTC", "2026-01-01"]
        txt_path = root / "p.txt"
        txt_path.write_text("\t".join(base) + "\n", encoding="utf-8")
        GeoNamesETL(db_path).build(txt_path=txt_path)
        json_path = root / "a.json"
        json_path.write_text(_json.dumps({
            "uri": "http://example.org/Lantern_Town",
            "label": "Lantern Town", "abstract": "A synthetic town.",
            "type": "Place"}) + "\n", encoding="utf-8")
        DBpediaETL(db_path).build(json_path=json_path)
        groups = find_overlaps(db_path)
        town = [g for g in groups if g["name"] == "lantern town"]
        if len(town) != 1 or {m["table"] for m in town[0]["members"]} != {
                "entities", "geonames_places", "dbpedia_entities"}:
            problems.append(
                f"D-MERGE-4 reopened live: overlap groups are {groups}")
    return problems


def merge_as_data() -> dict[str, Any]:
    """The rules as plain data, for consumers and the doctor."""
    return {
        "format": MERGE_FORMAT,
        "doc": MERGE_DOC,
        "outcomes": list(MERGE_OUTCOMES),
        "rules": [dict(r) for r in MERGE_RULES],
        "identities": [
            {"table": e["table"], "builder": e["builder"],
             "key": list(e["key"]),
             "value_fields": list(e["value_fields"]),
             **({"semantic_key": list(e["semantic_key"])}
                if "semantic_key" in e else {}),
             "identity": e["identity"]}
            for e in TABLE_IDENTITIES
        ],
    }


__all__ = [
    "LEDGER_VALUE_LIMIT",
    "MERGE_DOC",
    "MERGE_FORMAT",
    "MERGE_OUTCOMES",
    "MERGE_RULES",
    "TABLE_IDENTITIES",
    "MergeLedger",
    "apply_row_merge",
    "decide_merge",
    "find_overlaps",
    "merge_as_data",
    "merge_conceptnet_edge",
    "merge_live_problems",
    "merge_problems",
    "normalise_name",
    "overlap_as_data",
]
