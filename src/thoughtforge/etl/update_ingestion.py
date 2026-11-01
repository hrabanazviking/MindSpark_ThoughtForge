"""
Update ingestion — lifecycle rules for re-running a
source, as data (TF-114).

The roadmap's rule: *re-running a source updates facts
by the lifecycle rules instead of duplicating them.*

TF-109 gave overlapping sources dedup & merge rules,
but its within-source conflict default was filed as a
placeholder — "the incoming row wins", with TF-114
named as the owner of replacing it with lifecycle
update rules. And three behaviours were measured live
on this tree before this slice built anything
(synthetic fixtures through the real builders):

  * **Changed wikidata statements APPENDED beside the
    old fact (D-UPDATE-1).** Q1 built with P31 -> Q111,
    then re-built (``resume=False``) with P31 -> Q999:
    the store held BOTH tuples live, an unflagged
    contradiction in TF-050's sense. A P17 claim the
    re-run no longer carried stayed too — statements
    only ever grew. TF-107's idempotence covers the
    *duplicate* line (the exact tuple is not inserted
    twice); nothing owned the *changed* line.
  * **Chunks that disappeared from a re-presented
    reference file stayed stale (D-UPDATE-2).** A file
    built with two chunks, then edited down to one
    changed chunk, kept the vanished second chunk in
    the store — still searchable through the FTS index
    TF-058 rebuilt — beside the updated first chunk. A
    file deleted outright kept every chunk too.
  * **There was no lifecycle statement and no update
    record (D-UPDATE-3).** Row tables (entities,
    labels, GeoNames, DBpedia, reference chunks)
    resolved a changed fact by TF-109's unqualified
    default, and the only record was the merge ledger's
    conflict entry; nothing anywhere stated what a
    re-run *should* do with an unchanged fact, a
    changed fact, or a fact that disappears, and no
    ledger named a supersede or a retirement as such.

The lifecycle rules, filed here as data
(``UPDATE_RULES``) and justified from this repo's own
precedents — TF-050's contradiction rule (*surfaced,
scored, resolved by the stated rule, never silently
overwritten*; a decisive new claim *supersedes* the
stored one, which is archived out of the live store;
a duplicate is reinforced, not copied) and TF-045's
lifecycle (*retirement is physical; fact retirement
is an operator's act, never a side effect*):

  * **Unchanged — duplicate, not rewritten.** TF-109's
    rule stands unchanged.
  * **Changed within the same source — supersede.**
    The incoming value stands in place of the stored
    one; the stored value is preserved in the ledgers
    (TF-109's per-field conflict detail, and this
    module's ``UpdateLedger`` where the update is
    structural) before the write. That is TF-050's
    supersede in the form this store can honour: the
    ETL tables carry no status column and no archive
    table exists (TF-105's schema facts stand, no
    schema table is added), so the superseded value
    leaves the live row and its record is the ledger.
    TF-109's ``dedup_merge.py`` is updated in the open
    to state this rule in place of its placeholder.
  * **Statements supersede by presented set.** A
    wikidata entity the re-run presents is that
    source's newest complete statement of the entity:
    its presented statement tuples are judged whole —
    a stored tuple for the subject that the re-run
    does not present is removed from the live store
    and ledgered: ``statement_superseded`` when the
    re-run presents another tuple for the same
    property (the fact changed), ``statement_retired``
    when it does not (the fact disappeared from a
    presented entity). Multi-valued facts survive:
    every tuple the re-run presents stands. Only
    subjects the re-run actually presents are judged —
    checkpoint-resumed runs (TF-107) present only the
    items they stream.
  * **A disappeared chunk of a presented file is
    retired.** The reference builder judges per file
    it successfully reads: stored chunks of that file
    whose index the new presentation does not produce
    are deleted (with any reference-embedding row,
    which keys on the chunk) and ledgered
    ``chunk_retired``; TF-058's FTS rebuild at the end
    of the build drops them from search too.
  * **A unit absent from the re-run is KEPT.** An
    entity, place, uri or file the re-run does not
    present at all is left untouched, however long it
    has been absent. The builders accept partial
    inputs by design — a limit, a filter, a country
    subset, one file of a dump — so absence cannot be
    told from partiality, and deleting on absence
    would let a partial build destroy the store.
    Whole-unit retirement stays an operator's act
    (TF-045), never a re-run's side effect. Measured
    above: Q2, the GeoNames place "Beta", DBpedia's
    B and the deleted file's chunks all stay — now by
    stated rule, not by accident.
  * **Cross-source rows are never updated by a
    re-run.** TF-109 stands: dedup never merges across
    tables, and a re-run of one source touches only
    that source's own rows (statements are judged
    under ``source = 'wikidata'``; chunks under their
    own source file).

  * ``UpdateLedger`` — the ledger the wikidata and
    reference builders now hold fresh per build beside
    their TF-108 gate and TF-109 merge ledgers,
    recording every supersede/retirement this module
    performs, as data for TF-115's per-run reporting.
  * The retirement itself is performed by
    ``retire_unpresented_statements`` (wikidata,
    driven by the builder's connection-local presented
    sets — TEMP tables only, no schema change) and
    ``retire_unpresented_chunks`` (reference), both
    called by the builders in the open.

Docket (TF-016): D-UPDATE-1, D-UPDATE-2 and D-UPDATE-3
are added and FIXED by this slice.

The human-readable write-up is
``docs/architecture/UPDATE_INGESTION.md``. Proof
grade: FIXTURE — synthetic re-runs through the real
builders on this VM; no real dump has been re-run and
no at-scale update rate is claimed.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from thoughtforge.etl.dedup_merge import LEDGER_VALUE_LIMIT

UPDATE_FORMAT = "thoughtforge-update-ingestion/1"
UPDATE_DOC = "docs/architecture/UPDATE_INGESTION.md"

#: Connection-local (TEMP) tables the wikidata builder
#: fills with what a run presented, so retirement can be
#: judged in SQL at the end of the build without holding
#: a dump in memory and without any schema change.
PRESENTED_SUBJECTS_TABLE = "tf114_presented_subjects"
PRESENTED_STATEMENTS_TABLE = "tf114_presented_statements"

#: Update outcomes the UpdateLedger records.
UPDATE_OUTCOMES: tuple[str, ...] = (
    "statement_superseded", "statement_retired", "chunk_retired",
)


# ── The rules ───────────────────────────────────────────────────────────────

UPDATE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "unchanged_fact_is_a_duplicate_not_rewritten",
     "rule": "a fact the re-run presents unchanged is TF-109's duplicate: the stored row is not rewritten and the decision rides the builder's MergeLedger"},
    {"id": "changed_fact_within_a_source_supersedes",
     "rule": "a fact the re-run presents changed, within its own source's table, supersedes: the incoming value stands in place of the stored one (TF-050's new_supersedes_stored), and the stored value is recorded in the ledgers before the write — never a silent overwrite; TF-109's dedup_merge states this rule in place of its old incoming-wins placeholder"},
    {"id": "complementary_change_merges_never_erases",
     "rule": "a re-run carrying a thinner record still merges complementarily (TF-109): an empty incoming field never erases a stored value and a stored gap is filled — supersede judges only fields carrying two different non-empty values"},
    {"id": "statements_supersede_by_presented_set",
     "rule": "for each wikidata subject a re-run presents, the presented statement tuples are the subject's complete newest statement set: a stored tuple (source wikidata) the re-run does not present leaves the live store and is ledgered statement_superseded when another tuple for the same property is presented, statement_retired when none is; every presented tuple stands, so multi-valued facts are never collapsed"},
    {"id": "only_presented_subjects_are_judged",
     "rule": "statement retirement judges only subjects the run actually presented (accepted by the gate and the filters and streamed by this run): a checkpoint-resumed run (TF-107) presents only the items it streams, and a completed-checkpoint return presents nothing and retires nothing"},
    {"id": "disappeared_chunk_of_a_presented_file_is_retired",
     "rule": "for each reference file a re-run successfully reads, stored chunks of that file whose chunk_index the new presentation does not produce are deleted from the live store — with any reference_embeddings row keyed on them — and ledgered chunk_retired; the build's FTS rebuild (TF-058) drops them from search in the same build"},
    {"id": "a_unit_absent_from_the_rerun_is_kept",
     "rule": "a unit the re-run does not present at all (an entity, place, uri, or whole file) is kept untouched: the builders accept partial inputs by design (limit, filters, subsets), so absence cannot be told from partiality, and whole-unit retirement is an operator's act (TF-045), never a re-run's side effect"},
    {"id": "cross_source_rows_are_never_updated_by_a_rerun",
     "rule": "a re-run touches only its own source's rows (TF-109: cross-source overlap is a finding, never a merge): statement retirement is judged under source = 'wikidata' and chunk retirement under the chunk's own source_file, so no other source's rows can be retired by this source's re-run"},
    {"id": "retirement_is_physical_and_ledgered",
     "rule": "a retired or superseded statement/chunk leaves the live store physically (TF-045: retirement is physical; the ETL schema carries no status column and no archive table is added — TF-105's schema facts stand) and its full stored tuple is preserved in the UpdateLedger, truncated to LEDGER_VALUE_LIMIT like TF-109's conflict detail"},
    {"id": "the_update_ledger_is_fresh_per_build",
     "rule": "each wikidata / reference build holds a fresh UpdateLedger beside its TF-108 gate and TF-109 merge ledgers, recording every supersede and retirement that build performed, as data; TF-115 owns reporting it per run"},
)


# ── The ledger ──────────────────────────────────────────────────────────────

def _ledger_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > LEDGER_VALUE_LIMIT:
        return value[:LEDGER_VALUE_LIMIT] + "…"
    return value


class UpdateLedger:
    """The update ledger a builder holds for one run."""

    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []

    def record(self, table: str, key: dict[str, Any], outcome: str,
               detail: list[dict[str, Any]] | None = None) -> None:
        if outcome not in UPDATE_OUTCOMES:
            raise ValueError(
                f"unknown update outcome {outcome!r}: the ledger "
                f"records {list(UPDATE_OUTCOMES)}")
        self.decisions.append({
            "table": table, "key": dict(key), "outcome": outcome,
            "detail": [dict(d) for d in (detail or [])],
        })

    @property
    def updates(self) -> int:
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
            "format": UPDATE_FORMAT,
            "updates": self.updates,
            "by_outcome": self.by_outcome(),
            "by_table": self.by_table(),
            "decisions": [dict(d) for d in self.decisions],
        }


# ── Statement retirement (wikidata) ─────────────────────────────────────────

def prepare_presented_tables(conn: sqlite3.Connection) -> None:
    """Create the connection-local presented-set tables."""
    conn.execute(
        f"CREATE TEMP TABLE IF NOT EXISTS {PRESENTED_SUBJECTS_TABLE} "
        "(qid TEXT PRIMARY KEY)")
    conn.execute(
        f"CREATE TEMP TABLE IF NOT EXISTS {PRESENTED_STATEMENTS_TABLE} "
        "(subject_qid TEXT, property_pid TEXT, object_value TEXT, "
        "object_type TEXT, qualifiers TEXT, refs TEXT)")


def note_presented_statements(
    conn: sqlite3.Connection, batch: list[tuple[Any, ...]],
) -> None:
    """Record one flushed batch's tuples as presented.

    Called by the wikidata builder's statement flush;
    a no-op when the presented tables do not exist (a
    direct flush outside a build presents nothing to
    retire against, so it must not create the tables
    and thereby arm a retirement).
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_temp_master WHERE type = 'table' "
        "AND name = ?", (PRESENTED_STATEMENTS_TABLE,)).fetchone()
    if not exists or not batch:
        return
    conn.executemany(
        f"INSERT INTO {PRESENTED_STATEMENTS_TABLE} VALUES "
        "(?, ?, ?, ?, ?, ?)", [tuple(row) for row in batch])


def retire_unpresented_statements(
    conn: sqlite3.Connection, ledger: UpdateLedger,
) -> dict[str, int]:
    """
    Retire stored wikidata statements a run did not
    re-present for the subjects it presented. Returns
    the counts by outcome. Must run inside the build,
    after the final flush and before the final commit:
    a failed build rolls the retirement back with the
    rest of the run's uncommitted work.
    """
    rows = conn.execute(
        "SELECT s.statement_id, s.subject_qid, s.property_pid, "
        "s.object_value, s.object_type, s.qualifiers, s.refs "
        "FROM statements s "
        f"JOIN {PRESENTED_SUBJECTS_TABLE} p ON p.qid = s.subject_qid "
        "WHERE s.source = 'wikidata' AND NOT EXISTS ("
        f"SELECT 1 FROM {PRESENTED_STATEMENTS_TABLE} t WHERE "
        "t.subject_qid IS s.subject_qid AND "
        "t.property_pid IS s.property_pid AND "
        "t.object_value IS s.object_value AND "
        "t.object_type IS s.object_type AND "
        "t.qualifiers IS s.qualifiers AND t.refs IS s.refs)"
    ).fetchall()
    counts = {"statement_superseded": 0, "statement_retired": 0}
    for statement_id, subject, pid, obj, otype, quals, refs in rows:
        superseding = conn.execute(
            f"SELECT object_value FROM {PRESENTED_STATEMENTS_TABLE} "
            "WHERE subject_qid = ? AND property_pid = ? LIMIT 1",
            (subject, pid)).fetchone()
        key = {"subject_qid": subject, "property_pid": pid,
               "object_value": obj, "object_type": otype,
               "qualifiers": quals, "refs": refs}
        detail = [{"field": "object_value",
                   "stored": _ledger_value(obj),
                   "incoming": _ledger_value(superseding[0])
                   if superseding else None,
                   "resolution": "superseded_by_presented_value"
                   if superseding else "not_represented"}]
        outcome = ("statement_superseded" if superseding
                   else "statement_retired")
        ledger.record("statements", key, outcome, detail)
        conn.execute("DELETE FROM statements WHERE statement_id = ?",
                     (statement_id,))
        counts[outcome] += 1
    return counts


# ── Chunk retirement (reference) ────────────────────────────────────────────

def retire_unpresented_chunks(
    conn: sqlite3.Connection, ledger: UpdateLedger,
    source_file: str, presented_indexes: set[int],
) -> int:
    """
    Retire one presented file's stored chunks whose
    index the new presentation does not produce,
    together with any reference-embedding rows keyed
    on them. Returns the number retired.
    """
    rows = conn.execute(
        "SELECT chunk_id, chunk_index, title FROM reference_chunks "
        "WHERE source_file = ?", (source_file,)).fetchall()
    retired = 0
    for chunk_id, chunk_index, title in rows:
        if chunk_index in presented_indexes:
            continue
        ledger.record(
            "reference_chunks",
            {"source_file": source_file, "chunk_index": chunk_index},
            "chunk_retired",
            [{"field": "title", "stored": _ledger_value(title),
              "incoming": None, "resolution": "not_represented"}])
        conn.execute(
            "DELETE FROM reference_embeddings WHERE chunk_id = ?",
            (chunk_id,))
        conn.execute(
            "DELETE FROM reference_chunks WHERE chunk_id = ?",
            (chunk_id,))
        retired += 1
    return retired


# ── Checkers ────────────────────────────────────────────────────────────────

def update_problems() -> list[str]:
    """Static truth: the filed rules against the live tree."""
    from thoughtforge.etl import sources as sources_module
    from thoughtforge.etl import wikidata as wikidata_module

    problems: list[str] = []
    rule_ids = [r["id"] for r in UPDATE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate UPDATE_RULES ids")
    for needed in ("unchanged_fact_is_a_duplicate_not_rewritten",
                   "changed_fact_within_a_source_supersedes",
                   "statements_supersede_by_presented_set",
                   "disappeared_chunk_of_a_presented_file_is_retired",
                   "a_unit_absent_from_the_rerun_is_kept",
                   "cross_source_rows_are_never_updated_by_a_rerun"):
        if needed not in rule_ids:
            problems.append(f"UPDATE_RULES lost {needed}")

    # TF-109's placeholder must have been replaced in
    # the open by this slice's supersede rule.
    from thoughtforge.etl.dedup_merge import MERGE_RULES

    conflict_rule = [r for r in MERGE_RULES
                     if r["id"] == "within_a_table_the_incoming_row_wins"]
    if len(conflict_rule) != 1:
        problems.append("dedup_merge lost its within-table conflict rule")
    else:
        text = conflict_rule[0]["rule"]
        if "TF-114" not in text or "supersede" not in text:
            problems.append(
                "dedup_merge's within-table rule was not updated in "
                "the open to TF-114's lifecycle supersede rule")
        if "owns replacing this default" in text:
            problems.append(
                "dedup_merge still carries TF-109's TF-114 placeholder")

    wikidata_src = Path(wikidata_module.__file__).read_text(
        encoding="utf-8")
    for marker in ("update_ledger", "prepare_presented_tables",
                   "retire_unpresented_statements",
                   "note_presented_statements"):
        if marker not in wikidata_src:
            problems.append(f"wikidata.py lost its {marker} wiring")
    sources_src = Path(sources_module.__file__).read_text(encoding="utf-8")
    for marker in ("update_ledger", "retire_unpresented_chunks"):
        if marker not in sources_src:
            problems.append(f"sources.py lost its {marker} wiring")
    if not hasattr(wikidata_module.WikidataETL(), "update_ledger"):
        problems.append("WikidataETL carries no update_ledger")
    if not hasattr(sources_module.ReferenceDataETL(), "update_ledger"):
        problems.append("ReferenceDataETL carries no update_ledger")

    from thoughtforge.etl.etl_map import ORCHESTRATION

    if "update_ingestion" not in ORCHESTRATION.get("update", ""):
        problems.append("TF-105's map lost its update orchestration entry")
    doc = Path(__file__).resolve().parents[3] / UPDATE_DOC
    if not doc.exists():
        problems.append(f"the write-up {UPDATE_DOC} is missing")
    return problems


def _synthetic_entity(
    qid: str, label: str, claims: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "id": qid,
        "labels": {"en": {"value": label}},
        "descriptions": {"en": {"value": "A synthetic town"}},
        "aliases": {"en": [{"value": "the lit town"}]},
        "sitelinks": {"enwiki": {}, "dewiki": {}},
        "claims": {pid: [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": obj}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        } for obj in objs] for pid, objs in claims.items()},
    }


def update_live_problems() -> list[str]:
    """Behavioural truth: the probed failures, re-run live."""
    import json as _json

    from thoughtforge.etl.sources import ReferenceDataETL
    from thoughtforge.etl.wikidata import WikidataETL

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # D-UPDATE-1: a changed statement supersedes, a
        # removed claim retires, an absent entity is kept.
        db_path = root / "w.db"
        dump = root / "dump.json"
        dump.write_text(_json.dumps([
            _synthetic_entity("Q1", "Lantern Town",
                              {"P31": ["Q111"], "P17": ["Q900"]}),
            _synthetic_entity("Q2", "Other Town", {"P31": ["Q222"]}),
        ]), encoding="utf-8")
        WikidataETL(db_path).build(dump_path=dump, resume=False)
        dump.write_text(_json.dumps([
            _synthetic_entity("Q1", "Lantern Town", {"P31": ["Q999"]}),
        ]), encoding="utf-8")
        etl = WikidataETL(db_path)
        etl.build(dump_path=dump, resume=False)
        conn = sqlite3.connect(str(db_path))
        try:
            statements = conn.execute(
                "SELECT subject_qid, property_pid, object_value "
                "FROM statements ORDER BY 1, 2").fetchall()
        finally:
            conn.close()
        if statements != [("Q1", "P31", "Q999"), ("Q2", "P31", "Q222")]:
            problems.append(
                f"D-UPDATE-1 reopened live: statements are {statements}")
        outcomes = etl.update_ledger.by_outcome()
        if outcomes != {"statement_superseded": 1,
                        "statement_retired": 1}:
            problems.append(
                f"D-UPDATE-1 reopened live: update ledger is {outcomes}")

        # D-UPDATE-2: a disappeared chunk of a presented
        # file retires; a deleted file's chunks are kept.
        ref_dir = root / "ref"
        ref_dir.mkdir()
        doc = ref_dir / "doc.md"
        doc.write_text("# Part One\n\nAlpha text here.\n\n"
                       "# Part Two\n\nBeta text here.\n",
                       encoding="utf-8")
        rdb = root / "r.db"
        ReferenceDataETL(rdb).build(ref_dir=ref_dir)
        doc.write_text("# Part One\n\nAlpha text changed.\n",
                       encoding="utf-8")
        retl = ReferenceDataETL(rdb)
        retl.build(ref_dir=ref_dir)
        conn = sqlite3.connect(str(rdb))
        try:
            chunks = conn.execute(
                "SELECT source_file, chunk_index FROM "
                "reference_chunks ORDER BY 1, 2").fetchall()
        finally:
            conn.close()
        if chunks != [("doc.md", 0)]:
            problems.append(
                f"D-UPDATE-2 reopened live: chunks are {chunks}")
        if retl.update_ledger.by_outcome() != {"chunk_retired": 1}:
            problems.append(
                "D-UPDATE-2 reopened live: update ledger is "
                f"{retl.update_ledger.as_data()}")
        doc.unlink()
        (ref_dir / "new.md").write_text("# New\n\nNew text.\n",
                                        encoding="utf-8")
        ReferenceDataETL(rdb).build(ref_dir=ref_dir)
        conn = sqlite3.connect(str(rdb))
        try:
            chunks = conn.execute(
                "SELECT source_file, chunk_index FROM "
                "reference_chunks ORDER BY 1, 2").fetchall()
        finally:
            conn.close()
        if chunks != [("doc.md", 0), ("new.md", 0)]:
            problems.append(
                "absent-file rule reopened live: chunks are "
                f"{chunks}")
    return problems


def update_as_data() -> dict[str, Any]:
    """The update rules as plain data, for consumers."""
    return {
        "format": UPDATE_FORMAT,
        "doc": UPDATE_DOC,
        "outcomes": list(UPDATE_OUTCOMES),
        "rules": [dict(r) for r in UPDATE_RULES],
    }


__all__ = [
    "PRESENTED_STATEMENTS_TABLE",
    "PRESENTED_SUBJECTS_TABLE",
    "UPDATE_DOC",
    "UPDATE_FORMAT",
    "UPDATE_OUTCOMES",
    "UPDATE_RULES",
    "UpdateLedger",
    "note_presented_statements",
    "prepare_presented_tables",
    "retire_unpresented_chunks",
    "retire_unpresented_statements",
    "update_as_data",
    "update_live_problems",
    "update_problems",
]
