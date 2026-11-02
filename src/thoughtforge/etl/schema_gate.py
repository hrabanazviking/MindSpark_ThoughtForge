"""
Schema validation at the gate, as data (TF-108).

TF-105 mapped the ingestion forge's five builders; TF-106
gave every source provenance + licence; TF-107 hardened
the wikidata pipeline. None of them validated the RECORDS
themselves. TF-108's probes measured what the builders
did with malformed records before this slice:

  * a malformed record could CRASH the whole build
    (D-GATE-1): a DBpedia line holding a JSON number or
    list died with ``AttributeError`` on ``.get``; a
    DBpedia label of the wrong type died with
    ``TypeError``; a ConceptNet metadata weight of
    ``"abc"`` died with ``ValueError`` (only
    ``JSONDecodeError`` / ``TypeError`` were caught around
    it); a wikidata entity whose ``labels`` was a string,
    whose ``claims`` was a list, whose ``id`` was an
    integer, or whose claim entries were strings died
    with ``AttributeError`` / ``TypeError`` mid-stream —
    in every case the build aborted and the records
    already read were neither landed nor counted as
    refused;
  * out-of-schema values were silently ACCEPTED
    (D-GATE-2): a GeoNames row with latitude 999.0 and
    longitude -999.0, and a row with an empty name, were
    inserted as filed; a ConceptNet line whose metadata
    was not JSON at all was inserted with the weight
    silently defaulted to 1.0; lines with empty concepts
    or an empty relation were counted "inserted"; a
    wikidata label whose value was the integer 123 was
    stored as the text ``"123"``;
  * every refusal that did exist was a bare number
    (D-GATE-3): each builder counted ``skipped`` with no
    reason recorded anywhere — a short row, an
    unparseable row and a filtered row were
    indistinguishable, and ConceptNet's
    ``build_from_reference`` dropped non-dict items and
    field-less items with a plain ``continue`` and no
    count at all.

The gate, filed here as data and enforced in the
builders:

  * ``RECORD_SCHEMAS`` — per source, the fields a record
    must carry, their kinds, and the checks on them.
    The gate covers the four external record sources
    (wikidata, conceptnet, geonames, dbpedia) plus the
    conceptnet reference-edge path. The reference
    builder's records are chunks the builder generates
    itself from files, not external records; its
    ingestion side is TF-118's road, filed as a limit
    in the doc.
  * ``validate_record(source, record)`` — the pure
    judgement: a list of named reasons, empty when the
    record may pass. Reason strings are
    ``code:field: detail`` so counts group by code.
  * ``SchemaGate`` — the counting gate the builders
    hold: every record a builder presents is seen
    exactly once; a refused record is recorded with its
    source, its position in the run, an optional caller
    reference (an id the record itself carries — never
    the record's content), and its reasons. Parse-level
    refusals (a line that never became a record) enter
    through ``refuse()`` with the builder's own named
    reason. Filters are NOT refusals: a schema-valid
    record a builder's stated filter excludes (language,
    population, feature class, instance_of, excluded
    types) still counts in the builder's ``skipped`` but
    never enters the ledger — the gate judges shape,
    filters judge selection.
  * The checkers: ``gate_problems()`` judges the filed
    schemas + rules against the live builders (the gate
    must be wired in fact), and ``gate_live_problems()``
    runs malformed + valid synthetic fixtures through
    the real builders — no crash, valid records land,
    malformed records are refused with reasons and
    counted, and the ledger agrees with the stats.

Docket (TF-016): D-GATE-1 (malformed records crash the
build), D-GATE-2 (out-of-schema values silently
accepted) and D-GATE-3 (refusals counted with no reasons;
one path not counted at all) are added and FIXED by
this slice.

The human-readable write-up is
``docs/architecture/SCHEMA_VALIDATION_GATE.md``. Proof
grade: FIXTURE — synthetic malformed records through
the real builders on this VM; no real dump has been
ingested by this campaign and none is claimed.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

GATE_FORMAT = "thoughtforge-schema-gate/1"
GATE_DOC = "docs/architecture/SCHEMA_VALIDATION_GATE.md"

# The external record sources the gate covers, in TF-105's
# builder order, plus the conceptnet reference-edge path.
GATED_SOURCES: tuple[str, ...] = (
    "wikidata", "conceptnet", "geonames", "dbpedia",
    "conceptnet_reference",
)

_QID_RE = re.compile(r"^Q[1-9][0-9]*$")


# ── The rules ───────────────────────────────────────────────────────────────

GATE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "validation_before_insertion",
     "rule": "every external record is judged by validate_record (or refused by name at parse level) before any part of it is written; a record the gate refuses contributes no row to any table"},
    {"id": "refused_with_reasons",
     "rule": "a refused record is recorded in the builder's SchemaGate ledger with its source, its position in the run, and every reason validate_record named — never a bare count alone"},
    {"id": "refused_is_counted",
     "rule": "refusals count in the builder's existing skipped total (the stats shapes TF-105 filed do not change) AND in the gate ledger's refused total, broken down by source and by reason code; TF-115 owns reporting the rate per run"},
    {"id": "malformed_never_crashes",
     "rule": "no malformed record raises out of a build: wrong types, unparseable numbers, unparseable JSON and out-of-range values are refusals with reasons, and the build continues with the next record"},
    {"id": "malformed_never_silently_accepted",
     "rule": "no out-of-schema value is repaired into acceptance: unparseable metadata does not default a weight, out-of-range coordinates do not land, a non-string label does not coerce — the record is refused whole"},
    {"id": "filters_are_not_refusals",
     "rule": "a schema-valid record excluded by a builder's stated filter (language, min_population, feature_classes, instance_of filter, excluded instance_of types, deprecated rank) is a filter skip: it counts in skipped, never in the gate ledger"},
    {"id": "a_blank_line_is_not_a_record",
     "rule": "an empty line in a line-delimited source is file padding, not a record: it is neither seen by the gate nor counted — every non-blank line is one or the other (landed, filter-skipped, or refused with reasons)"},
    {"id": "the_ledger_carries_no_record_content",
     "rule": "a ledger entry carries the source, the run position, the record's own id where it carries one (a QID, a uri, a geonames id), and its reasons — never the record's text, so the ledger is safe to log and file"},
    {"id": "an_unknown_source_is_refused_by_name",
     "rule": "validate_record and the gate raise ValueError naming an unregistered source id — a new source files its schema here before its builder gates on it (TF-119's recipe owns that step)"},
)


# ── The schemas ─────────────────────────────────────────────────────────────
#
# One entry per gated source: the record shape validate_record
# judges. "fields" names each checked field, whether it is
# required, and the check in plain words (the code enforces
# exactly these).

RECORD_SCHEMAS: tuple[dict[str, Any], ...] = (
    {"source": "wikidata", "builder": "WikidataETL",
     "record": "one entity object from the JSON dump stream",
     "fields": (
         {"name": "id", "required": True,
          "check": "a string matching ^Q[1-9][0-9]*$"},
         {"name": "labels", "required": False,
          "check": "an object whose every value is an object with a string 'value'"},
         {"name": "descriptions", "required": False,
          "check": "an object whose every value is an object with a string 'value'"},
         {"name": "aliases", "required": False,
          "check": "an object whose every value is a list of objects with a string 'value'"},
         {"name": "sitelinks", "required": False,
          "check": "an object"},
         {"name": "claims", "required": False,
          "check": "an object whose every value is a list of objects"},
     )},
    {"source": "conceptnet", "builder": "ConceptNetETL",
     "record": "one tab-delimited assertions line, split into its five fields and judged with its parsed metadata",
     "fields": (
         {"name": "uri", "required": True, "check": "a non-empty string"},
         {"name": "relation", "required": True, "check": "a non-empty string"},
         {"name": "concept1", "required": True,
          "check": "a non-empty concept uri whose final segment (the concept's text) is non-empty"},
         {"name": "concept2", "required": True,
          "check": "a non-empty concept uri whose final segment (the concept's text) is non-empty"},
         {"name": "metadata", "required": True,
          "check": "a string that parses as a JSON object"},
         {"name": "weight", "required": False,
          "check": "when the metadata carries one: a finite, non-negative number (never a string, never a bool)"},
     )},
    {"source": "geonames", "builder": "GeoNamesETL",
     "record": "one tab-delimited places row, judged on its raw fields before parsing",
     "fields": (
         {"name": "geonames_id", "required": True,
          "check": "a positive integer"},
         {"name": "name", "required": True, "check": "a non-empty string"},
         {"name": "latitude", "required": False,
          "check": "empty, or a number in [-90, 90]"},
         {"name": "longitude", "required": False,
          "check": "empty, or a number in [-180, 180]"},
         {"name": "population", "required": False,
          "check": "empty, or a non-negative integer"},
         {"name": "elevation", "required": False,
          "check": "empty, or an integer"},
     )},
    {"source": "dbpedia", "builder": "DBpediaETL",
     "record": "one newline-delimited JSON value",
     "fields": (
         {"name": "record", "required": True,
          "check": "a JSON object"},
         {"name": "uri", "required": True, "check": "a non-empty string"},
         {"name": "label", "required": False, "check": "a string"},
         {"name": "abstract", "required": False, "check": "a string"},
         {"name": "types", "required": False,
          "check": "a string, or a list of strings"},
     )},
    {"source": "conceptnet_reference", "builder": "ConceptNetETL.build_from_reference",
     "record": "one object from a flat reference JSON array",
     "fields": (
         {"name": "concept1", "required": True, "check": "a non-empty string"},
         {"name": "relation", "required": True, "check": "a non-empty string"},
         {"name": "concept2", "required": True, "check": "a non-empty string"},
     )},
)


# ── The judgement ───────────────────────────────────────────────────────────

def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_text_map(record: dict[str, Any], field: str,
                    reasons: list[str]) -> None:
    """labels/descriptions shape: {lang: {"value": str}}."""
    value = record.get(field)
    if value is None:
        return
    if not isinstance(value, dict):
        reasons.append(
            f"wrong_type:{field}: expected object, got {_type_name(value)}")
        return
    for lang, entry in value.items():
        if not isinstance(entry, dict) \
                or not isinstance(entry.get("value"), str):
            reasons.append(
                f"wrong_type:{field}.{lang}.value: expected string")


def _validate_wikidata(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return [f"not_a_record:record: expected object, "
                f"got {_type_name(record)}"]
    reasons: list[str] = []
    qid = record.get("id")
    if qid is None:
        reasons.append("missing_required:id: the entity carries no id")
    elif not isinstance(qid, str):
        reasons.append(
            f"wrong_type:id: expected string, got {_type_name(qid)}")
    elif not _QID_RE.match(qid):
        reasons.append(
            f"bad_format:id: {qid!r} is not a Q-id (Q followed by digits)")
    _check_text_map(record, "labels", reasons)
    _check_text_map(record, "descriptions", reasons)
    aliases = record.get("aliases")
    if aliases is not None:
        if not isinstance(aliases, dict):
            reasons.append(
                f"wrong_type:aliases: expected object, "
                f"got {_type_name(aliases)}")
        else:
            for lang, entries in aliases.items():
                if not isinstance(entries, list) or any(
                        not isinstance(e, dict)
                        or not isinstance(e.get("value"), str)
                        for e in entries):
                    reasons.append(
                        f"wrong_type:aliases.{lang}: expected a list "
                        "of objects with a string 'value'")
    sitelinks = record.get("sitelinks")
    if sitelinks is not None and not isinstance(sitelinks, dict):
        reasons.append(
            f"wrong_type:sitelinks: expected object, "
            f"got {_type_name(sitelinks)}")
    claims = record.get("claims")
    if claims is not None:
        if not isinstance(claims, dict):
            reasons.append(
                f"wrong_type:claims: expected object, "
                f"got {_type_name(claims)}")
        else:
            for pid, entries in claims.items():
                if not isinstance(entries, list) or any(
                        not isinstance(e, dict) for e in entries):
                    reasons.append(
                        f"wrong_type:claims.{pid}: expected a list "
                        "of objects")
    return reasons


def _validate_conceptnet(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return [f"not_a_record:record: expected object, "
                f"got {_type_name(record)}"]
    reasons: list[str] = []
    for field in ("uri", "relation"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"empty:{field}: a non-empty string is required")
    for field in ("concept1", "concept2"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"empty:{field}: a non-empty string is required")
        elif not value.rstrip("/").rsplit("/", 1)[-1].strip():
            reasons.append(
                f"empty:{field}: the concept uri names no concept text")
    metadata = record.get("metadata_parsed")
    if not isinstance(metadata, dict):
        reasons.append(
            "unparseable:metadata: the metadata field is not a "
            "JSON object")
    elif "weight" in metadata:
        weight = metadata["weight"]
        if isinstance(weight, bool) \
                or not isinstance(weight, (int, float)):
            reasons.append(
                f"wrong_type:weight: expected number, "
                f"got {_type_name(weight)}")
        elif not math.isfinite(float(weight)) or float(weight) < 0:
            reasons.append(
                f"out_of_range:weight: {weight!r} is not a finite, "
                "non-negative weight")
    return reasons


def _int_field(record: dict[str, Any], field: str, reasons: list[str],
               *, required: bool, minimum: int | None) -> None:
    raw = record.get(field)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        if required:
            reasons.append(f"missing_required:{field}: no value carried")
        return
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        reasons.append(f"unparseable:{field}: {raw!r} is not an integer")
        return
    if minimum is not None and value < minimum:
        reasons.append(
            f"out_of_range:{field}: {value} is below {minimum}")


def _float_field(record: dict[str, Any], field: str, reasons: list[str],
                 *, low: float, high: float) -> None:
    raw = record.get(field)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        reasons.append(f"unparseable:{field}: {raw!r} is not a number")
        return
    if not math.isfinite(value) or value < low or value > high:
        reasons.append(
            f"out_of_range:{field}: {value} is outside [{low}, {high}]")


def _validate_geonames(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return [f"not_a_record:record: expected object, "
                f"got {_type_name(record)}"]
    reasons: list[str] = []
    _int_field(record, "geonames_id", reasons, required=True, minimum=1)
    name = record.get("name")
    if not isinstance(name, str) or not name.strip():
        reasons.append("empty:name: a non-empty string is required")
    _float_field(record, "latitude", reasons, low=-90.0, high=90.0)
    _float_field(record, "longitude", reasons, low=-180.0, high=180.0)
    _int_field(record, "population", reasons, required=False, minimum=0)
    _int_field(record, "elevation", reasons, required=False, minimum=None)
    return reasons


def _validate_dbpedia(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return [f"not_a_record:record: expected object, "
                f"got {_type_name(record)}"]
    reasons: list[str] = []
    uri = record.get("uri") or record.get("@id")
    if uri is None:
        reasons.append("missing_required:uri: the record carries no uri")
    elif not isinstance(uri, str):
        reasons.append(
            f"wrong_type:uri: expected string, got {_type_name(uri)}")
    elif not uri.strip():
        reasons.append("empty:uri: a non-empty string is required")
    for field, keys in (("label", ("label", "name")),
                        ("abstract", ("abstract", "comment"))):
        for key in keys:
            if key in record and record[key] is not None \
                    and not isinstance(record[key], str):
                reasons.append(
                    f"wrong_type:{field}: expected string, "
                    f"got {_type_name(record[key])}")
                break
    types = record.get("type", record.get("@type"))
    if types is not None and not isinstance(types, str) and not (
            isinstance(types, list)
            and all(isinstance(t, str) for t in types)):
        reasons.append(
            "wrong_type:types: expected a string or a list of strings, "
            f"got {_type_name(types)}")
    return reasons


def _validate_conceptnet_reference(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return [f"not_a_record:record: expected object, "
                f"got {_type_name(record)}"]
    reasons: list[str] = []
    for field in ("concept1", "relation", "concept2"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"empty:{field}: a non-empty string is required")
    return reasons


_VALIDATORS = {
    "wikidata": _validate_wikidata,
    "conceptnet": _validate_conceptnet,
    "geonames": _validate_geonames,
    "dbpedia": _validate_dbpedia,
    "conceptnet_reference": _validate_conceptnet_reference,
}


def validate_record(source: str, record: Any) -> list[str]:
    """The gate's judgement on one record: reasons, [] = may pass.

    Raises ``ValueError`` naming an unregistered source —
    a source files its schema here before it gates.
    """
    validator = _VALIDATORS.get(source)
    if validator is None:
        raise ValueError(
            f"unknown gated source {source!r}: the gate covers "
            f"{sorted(_VALIDATORS)} — file the schema in "
            "RECORD_SCHEMAS before gating a new source")
    return validator(record)


# ── The counting gate ───────────────────────────────────────────────────────

class SchemaGate:
    """The ledger a builder holds for one run.

    ``check`` judges a record and records the refusal when
    the record may not pass; ``refuse`` records a
    parse-level refusal the builder names itself (a line
    that never became a record). Positions are per-source
    counts of records presented, starting at 1.
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}
        self.refusals: list[dict[str, Any]] = []

    def _position(self, source: str) -> int:
        self._seen[source] = self._seen.get(source, 0) + 1
        return self._seen[source]

    def check(self, source: str, record: Any,
              ref: str | None = None) -> list[str]:
        """Judge one record; record + return its reasons."""
        reasons = validate_record(source, record)
        position = self._position(source)
        if reasons:
            self.refusals.append({
                "source": source, "position": position,
                "ref": ref, "reasons": list(reasons),
            })
        return reasons

    def refuse(self, source: str, reasons: list[str],
               ref: str | None = None) -> list[str]:
        """Record a parse-level refusal named by the builder."""
        if source not in _VALIDATORS:
            raise ValueError(
                f"unknown gated source {source!r}: the gate covers "
                f"{sorted(_VALIDATORS)}")
        if not reasons:
            raise ValueError(
                "a refusal must carry at least one reason — a "
                "record is never dropped unnamed (D-GATE-3)")
        position = self._position(source)
        self.refusals.append({
            "source": source, "position": position,
            "ref": ref, "reasons": list(reasons),
        })
        return list(reasons)

    @property
    def seen(self) -> int:
        return sum(self._seen.values())

    @property
    def refused(self) -> int:
        return len(self.refusals)

    def by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.refusals:
            for reason in entry["reasons"]:
                code = reason.split(":", 1)[0]
                counts[code] = counts.get(code, 0) + 1
        return counts

    def by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.refusals:
            counts[entry["source"]] = counts.get(entry["source"], 0) + 1
        return counts

    def as_data(self) -> dict[str, Any]:
        return {
            "format": GATE_FORMAT,
            "seen": self.seen,
            "seen_by_source": dict(self._seen),
            "refused": self.refused,
            "by_reason": self.by_reason(),
            "by_source": self.by_source(),
            "refusals": [dict(r) for r in self.refusals],
        }


# ── Checkers ────────────────────────────────────────────────────────────────

def gate_problems() -> list[str]:
    """Static truth: the filed schemas + rules against the live tree."""
    from thoughtforge.etl import sources as sources_module
    from thoughtforge.etl import wikidata as wikidata_module

    problems: list[str] = []
    rule_ids = [r["id"] for r in GATE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate GATE_RULES ids")
    schema_sources = [s["source"] for s in RECORD_SCHEMAS]
    if schema_sources != list(GATED_SOURCES):
        problems.append(f"RECORD_SCHEMAS sources drifted: {schema_sources}")
    if set(_VALIDATORS) != set(GATED_SOURCES):
        problems.append("a gated source has no validator (or the reverse)")
    for schema in RECORD_SCHEMAS:
        names = [f["name"] for f in schema["fields"]]
        if len(names) != len(set(names)) or not names:
            problems.append(f"{schema['source']}: field names not unique")

    sources_src = Path(sources_module.__file__).read_text(encoding="utf-8")
    wikidata_src = Path(wikidata_module.__file__).read_text(encoding="utf-8")
    if "from thoughtforge.etl.schema_gate import" not in sources_src:
        problems.append("sources.py lost its schema_gate import: the "
                        "conceptnet/geonames/dbpedia builders must gate")
    if "from thoughtforge.etl.schema_gate import" not in wikidata_src:
        problems.append("wikidata.py lost its schema_gate import: the "
                        "wikidata builder must gate")
    for marker, why in (
            ('self.schema_gate.check("conceptnet"', "ConceptNetETL.build"),
            ('self.schema_gate.check("geonames"', "GeoNamesETL.build"),
            ('"dbpedia", item', "DBpediaETL.build"),
            ('"conceptnet_reference", {',
             "ConceptNetETL.build_from_reference"),
            ):
        if marker not in sources_src:
            problems.append(f"{why} lost its gate check: {marker!r}")
    if '"wikidata", item' not in wikidata_src:
        problems.append("WikidataETL.build lost its gate check")
    for cls_name in ("ConceptNetETL", "GeoNamesETL", "DBpediaETL"):
        cls = getattr(sources_module, cls_name)
        if not hasattr(cls(), "schema_gate"):
            problems.append(f"{cls_name} carries no schema_gate ledger")
    if not hasattr(wikidata_module.WikidataETL(), "schema_gate"):
        problems.append("WikidataETL carries no schema_gate ledger")
    return problems


def _synthetic_entity(qid: str) -> dict[str, Any]:
    return {
        "id": qid,
        "labels": {"en": {"value": f"Lantern {qid}"}},
        "descriptions": {"en": {"value": "a synthetic entity"}},
        "aliases": {"en": []},
        "sitelinks": {},
        "claims": {"P31": [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": "Q108900"}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        }]},
    }


def gate_live_problems() -> list[str]:
    """Behavioural truth: malformed fixtures through the real builders."""
    from thoughtforge.etl.sources import (
        ConceptNetETL,
        DBpediaETL,
        GeoNamesETL,
    )
    from thoughtforge.etl.wikidata import WikidataETL

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Wikidata: 2 valid, 3 malformed (bad id type, labels
        # as string, claims as list) — no crash, refusals filed.
        dump = root / "dump.json"
        dump.write_text(json.dumps([
            _synthetic_entity("Q1"), _synthetic_entity("Q2"),
            {"id": 123, "labels": {}, "claims": {}},
            {"id": "Q3", "labels": "oops", "claims": {}},
            {"id": "Q4", "labels": {}, "claims": []},
        ]), encoding="utf-8")
        etl = WikidataETL(root / "w.db")
        try:
            stats = etl.build(dump_path=dump)
        except Exception as exc:  # D-GATE-1: a crash is a finding
            return [f"D-GATE-1 reopened live: wikidata build "
                    f"crashed on malformed records: {exc!r}"]
        if stats != {"entities": 2, "statements": 2, "labels": 2,
                     "skipped": 3}:
            problems.append(f"wikidata gated stats drifted: {stats}")
        if etl.schema_gate.refused != 3 \
                or etl.schema_gate.by_source() != {"wikidata": 3}:
            problems.append(
                f"wikidata ledger drifted: {etl.schema_gate.as_data()}")

        # ConceptNet: 1 valid, 4 malformed (short line, bad
        # metadata, bad weight, empty concept), 1 filter skip.
        csv_path = root / "assertions.csv"
        csv_path.write_text("\n".join([
            "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 2.0}",
            "short",
            "/a/2\t/r/IsA\t/c/en/badmeta\t/c/en/light\t{not json",
            "/a/3\t/r/IsA\t/c/en/badweight\t/c/en/light\t{\"weight\": \"abc\"}",
            "/a/4\t/r/IsA\t\t/c/en/light\t{\"weight\": 1.0}",
            "/a/5\t/r/IsA\t/c/fr/lanterne\t/c/fr/lumiere\t{\"weight\": 1.0}",
        ]) + "\n", encoding="utf-8")
        cetl = ConceptNetETL(root / "c.db")
        try:
            cstats = cetl.build(csv_path=csv_path)
        except Exception as exc:
            return problems + [f"D-GATE-1 reopened live: conceptnet "
                               f"build crashed: {exc!r}"]
        if cstats != {"inserted": 1, "skipped": 5}:
            problems.append(f"conceptnet gated stats drifted: {cstats}")
        if cetl.schema_gate.refused != 4:
            problems.append(
                f"conceptnet ledger refused "
                f"{cetl.schema_gate.refused} != 4 (the French line is "
                "a filter skip, not a refusal)")

        # GeoNames: 1 valid, 3 malformed (out-of-range
        # coordinates, empty name, short row).
        base = ["108001", "Lantern Town", "Lantern Town", "", "1.5",
                "2.5", "P", "PPL", "XX", "", "A1", "A2", "", "", "100",
                "10", "10", "UTC", "2026-01-01"]
        bad = list(base); bad[0] = "108002"; bad[4] = "999.0"
        noname = list(base); noname[0] = "108003"; noname[1] = ""
        txt_path = root / "places.txt"
        txt_path.write_text("\n".join([
            "\t".join(base), "\t".join(bad), "\t".join(noname),
            "short\trow",
        ]) + "\n", encoding="utf-8")
        getl = GeoNamesETL(root / "g.db")
        try:
            gstats = getl.build(txt_path=txt_path)
        except Exception as exc:
            return problems + [f"D-GATE-1 reopened live: geonames "
                               f"build crashed: {exc!r}"]
        if gstats != {"inserted": 1, "skipped": 3}:
            problems.append(f"geonames gated stats drifted: {gstats}")
        if getl.schema_gate.refused != 3:
            problems.append(
                f"geonames ledger refused {getl.schema_gate.refused} != 3")
        conn = sqlite3.connect(str(root / "g.db"))
        try:
            rows = conn.execute(
                "SELECT geonames_id FROM geonames_places").fetchall()
        finally:
            conn.close()
        if rows != [(108001,)]:
            problems.append(f"D-GATE-2 reopened live: geonames rows "
                            f"landed: {rows}")

        # DBpedia: 1 valid, 3 malformed (non-object, bad
        # label type, unparseable line), 1 missing-uri refusal.
        json_path = root / "abstracts.json"
        json_path.write_text("\n".join([
            json.dumps({"uri": "http://example.org/Lantern",
                        "label": "Lantern", "abstract": "A light.",
                        "type": "Device"}),
            "42",
            json.dumps({"uri": "http://example.org/Bad",
                        "label": 123}),
            "not json",
            json.dumps({"label": "No uri on this line"}),
        ]) + "\n", encoding="utf-8")
        detl = DBpediaETL(root / "d.db")
        try:
            dstats = detl.build(json_path=json_path)
        except Exception as exc:
            return problems + [f"D-GATE-1 reopened live: dbpedia "
                               f"build crashed: {exc!r}"]
        if dstats != {"inserted": 1, "skipped": 4}:
            problems.append(f"dbpedia gated stats drifted: {dstats}")
        if detl.schema_gate.refused != 4:
            problems.append(
                f"dbpedia ledger refused {detl.schema_gate.refused} != 4")
    return problems


def gate_as_data() -> dict[str, Any]:
    """The gate as plain data, for consumers and the doctor."""
    return {
        "format": GATE_FORMAT,
        "doc": GATE_DOC,
        "sources": list(GATED_SOURCES),
        "rules": [dict(r) for r in GATE_RULES],
        "schemas": [
            {"source": s["source"], "builder": s["builder"],
             "record": s["record"],
             "fields": [dict(f) for f in s["fields"]]}
            for s in RECORD_SCHEMAS
        ],
    }
