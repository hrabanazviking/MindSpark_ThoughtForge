"""
Ingestion metrics — per-run reporting, as data (TF-115).

The roadmap's rule: *ingestion metrics: records/sec,
refusal rate, store growth per source — reported per
run.*

Before this slice, none of that could be reported —
probed live on this tree before anything was built:

  * **No metrics existed anywhere (D-MET-1).** No
    module, no ``forge_memory.py`` command and no test
    named ingestion metrics; every command in the tree
    prints a builder's stats dict and discards the
    rest — the TF-108 gate ledger, the TF-109 merge
    ledger and the TF-114 update ledger live in memory
    on the builder instance and are thrown away by
    every caller except TF-112's scratch dry-run (which
    reports gate + merge for one source, but no timing,
    no refusal *rate*, no update ledger and no growth).
    TF-108, TF-109 and TF-114 each filed "TF-115 owns
    reporting it per run"; nothing owned it in fact.
  * **No build records its own elapsed time
    (D-MET-2).** A finished build's outputs — stats,
    gate ledger, merge ledger — carry no time field at
    all (probed: the key sets of all three were listed;
    none carries time / elapsed / seconds). The only
    rate in the tree is the wikidata builder's
    checkpoint *log line* (entities/sec, logged, never
    returned). A records/sec figure therefore cannot be
    computed after the fact from a build's outputs: it
    must be measured around the build, or not claimed.
  * **Accepted is not landed, and nothing named the
    difference (D-MET-3).** Builder stats count records
    *accepted*: probed live, an identical GeoNames
    re-run reported ``inserted: 1`` again while the
    store stayed at 1 row (the merge ledger alone knew
    the record was a duplicate). Store growth — rows
    and bytes actually landed — is derivable only by
    measuring the store before and after, which no
    caller did (TF-112 measures a *target's* untouched
    state and a scratch's counts, never a real run's
    growth).

What this slice builds, filed as data in
``METRICS_RULES`` and enforced here:

  * ``measure_ingestion()`` — ONE source, ingested by
    its REAL builder into the named store, timed by a
    caller-supplied clock (``time.perf_counter`` by
    default; a fake clock in tests makes the rate
    deterministic) and snapshotted before/after, returning
    an ``IngestionMetrics`` report (``.to_dict()`` is
    its plain-data shape, format
    ``thoughtforge-ingestion-metrics/1``).
  * **Records/sec is measured, never derived.** The
    elapsed covers the build call only (the after
    snapshot is taken outside the timed span). When
    the elapsed is zero or unmeasured the rate is filed
    ``None`` — never a division dressed as a fact.
  * **The refusal rate is the gate's own fraction**:
    TF-108 ``refused / seen`` for the run, with the
    by-reason / by-source breakdowns carried whole.
    Filter skips (language, population, feature class)
    are NOT refusals: the report files them apart as
    ``other_skips`` (stats skipped minus gate refused).
    The ungated reference builder files ``gate is
    None`` and a ``None`` rate, with the reason — never
    a fabricated zero.
  * **Store growth is a delta of snapshots** (the
    TF-110 measurement shape): per-table row deltas
    over TF-105's counted tables + byte delta (db +
    WAL + SHM, named apart in each snapshot). Growth
    may be negative — a TF-114 update run retires rows
    — and is filed as measured, never clamped.
  * **Accepted vs landed is filed, not smoothed.**
    The report carries the stats verbatim, the output
    rows the stats claim, the primary table's growth,
    and their difference (``accepted_vs_primary_growth``,
    judged on the primary count: entities / inserted /
    chunks per source) — duplicates, merged conflicts
    and retirements live in that difference, and the
    merge + update ledgers carried whole say which.
  * **Every ledger rides the report as data**: gate
    (where the builder is gated), merge, and update
    (wikidata + reference; the other builders file
    ``None`` with the TF-114 note that their updates
    ride the merge ledger).
  * ``summarise_runs()`` — per-source aggregation over
    per-run reports (a multi-source run's "per source"
    view, TF-105's ``all`` shape): runs, presented,
    refused + refusal rate over summed gate counts,
    summed elapsed with the rate recomputed from the
    sums (never an average of rates), summed growth.

Filed limits, stated plainly:

  * A failed build files no report: the builder's
    exception propagates unchanged and the partial
    measurements are discarded (for wikidata, TF-107's
    checkpoint is the failure's resume truth).
  * On a resumed or completed-checkpoint wikidata run
    the stats are TF-107's cumulative filed stats for
    the whole dump ask, so presented and the rate
    describe the ask, not only this call's streaming —
    the report files stats verbatim and never adjusts
    them (a filed rule below, not a hidden caveat).
  * The report is data returned to the caller: no
    report file is written, no schema table is added
    (TF-105's schema facts stand), and the CLI gains no
    command in this slice — ``forge_memory.py``'s
    stats-only printing stands as filed in TF-105
    (TF-119 owns the recipe / wiring share).

Docket (TF-016): D-MET-1, D-MET-2 and D-MET-3 are
added and FIXED by this slice.

The human-readable write-up is
``docs/architecture/INGESTION_METRICS.md``. Proof
grade: FIXTURE — synthetic sources through the real
builders on this VM, rates proven under a fake clock;
no real dump has been measured at scale, and no
at-scale rate is claimed.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

METRICS_FORMAT = "thoughtforge-ingestion-metrics/1"
METRICS_DOC = "docs/architecture/INGESTION_METRICS.md"


# ── The per-source shapes ───────────────────────────────────────────────────
#
# One entry per TF-105 builder (bound to TF-112's
# BUILDER_SOURCES ids — the checker pins the binding):
# which stats key counts accepted records, which counts
# the primary output rows, and which store table is the
# source's primary table (TF-105's filed targets).

SOURCE_METRIC_SHAPES: tuple[dict[str, Any], ...] = (
    {"builder_id": "wikidata", "unit": "entities",
     "accepted_key": "entities", "primary_key": "entities",
     "output_keys": ("entities", "statements", "labels"),
     "primary_table": "entities",
     "target_tables": ("entities", "statements", "labels")},
    {"builder_id": "conceptnet", "unit": "edges",
     "accepted_key": "inserted", "primary_key": "inserted",
     "output_keys": ("inserted",),
     "primary_table": "conceptnet_edges",
     "target_tables": ("conceptnet_edges",)},
    {"builder_id": "geonames", "unit": "places",
     "accepted_key": "inserted", "primary_key": "inserted",
     "output_keys": ("inserted",),
     "primary_table": "geonames_places",
     "target_tables": ("geonames_places",)},
    {"builder_id": "dbpedia", "unit": "entities",
     "accepted_key": "inserted", "primary_key": "inserted",
     "output_keys": ("inserted",),
     "primary_table": "dbpedia_entities",
     "target_tables": ("dbpedia_entities",)},
    {"builder_id": "reference", "unit": "files",
     "accepted_key": "files", "primary_key": "chunks",
     "output_keys": ("chunks",),
     "primary_table": "reference_chunks",
     "target_tables": ("reference_chunks",)},
)


def _shape(source: str) -> dict[str, Any]:
    for entry in SOURCE_METRIC_SHAPES:
        if source == entry["builder_id"]:
            return entry
    raise ValueError(
        f"unregistered source {source!r}: ingestion metrics cover "
        f"{[e['builder_id'] for e in SOURCE_METRIC_SHAPES]} — a new "
        "source is filed in TF-105's map and TF-106's registry "
        "before it is measured (TF-119's recipe)")


# ── The rules ───────────────────────────────────────────────────────────────

METRICS_RULES: tuple[dict[str, Any], ...] = (
    {"id": "metrics_are_measured_never_derived_from_stats_alone",
     "rule": "a run's rate comes from a clock measured around the real build and its growth from store snapshots taken before and after; builder stats alone carry neither time nor landed rows (D-MET-2, D-MET-3), so no metric is ever computed from stats alone"},
    {"id": "records_per_second_is_presented_over_build_elapsed",
     "rule": "records_per_second = records presented (accepted + skipped, in the builder's TF-105 stats shapes) divided by the elapsed seconds of the build call only — the after snapshot is measured outside the timed span"},
    {"id": "an_unmeasured_or_zero_elapsed_files_no_rate",
     "rule": "when the elapsed is zero, negative or unmeasured the rate is filed None, never a division by a non-time and never an invented figure"},
    {"id": "refusal_rate_is_gate_refused_over_gate_seen",
     "rule": "the refusal rate is TF-108's refused / seen for this run, carried with its by-reason and by-source breakdowns; stated filter skips (language, population, feature class) are not refusals and are filed apart as other_skips = stats skipped - gate refused"},
    {"id": "an_ungated_builder_files_no_rate",
     "rule": "the reference builder carries no schema gate (TF-108's filed limit): its report files gate=None and refusal_rate=None with the reason, never a fabricated empty ledger or a zero rate"},
    {"id": "store_growth_is_a_delta_of_snapshots",
     "rule": "growth is after minus before over TF-105's counted tables and over the store's bytes (db + WAL + SHM, named apart in each snapshot); growth may be negative on a TF-114 update run and is filed as measured, never clamped to zero"},
    {"id": "accepted_is_not_landed_and_the_difference_is_filed",
     "rule": "stats count records a builder accepted; landed new rows are the primary table's growth; the report files accepted_vs_primary_growth (the primary count minus the primary table's growth) — duplicates, merged conflicts and retirements live in that difference, and the merge + update ledgers carried whole say which"},
    {"id": "every_ledger_rides_the_report_as_data",
     "rule": "the report carries the build's TF-108 gate ledger (where gated), TF-109 merge ledger and TF-114 update ledger (wikidata + reference) whole as data; a builder with no update ledger files None with the TF-114 note that its updates ride its merge ledger"},
    {"id": "a_failed_build_files_no_report",
     "rule": "a builder failure propagates as the builder raised it and the partial measurements are discarded — a metrics wrapper never converts a crash into a report with invented numbers (for wikidata, TF-107's checkpoint is the failure's resume truth)"},
    {"id": "one_report_per_run_per_source_totals_only_by_aggregation",
     "rule": "measure_ingestion runs ONE builder once into the named store and reports that run; multi-source figures exist only as summarise_runs' aggregation over per-run reports, whose rate is recomputed from the summed presented and elapsed, never averaged from rates"},
    {"id": "wikidata_resumed_stats_are_cumulative",
     "rule": "on a resumed or completed-checkpoint wikidata run the stats are TF-107's cumulative filed stats for the whole dump ask, so presented and the rate describe the ask, not only this call's streaming; the report files the stats verbatim and never adjusts them"},
    {"id": "no_report_file_and_no_schema_change",
     "rule": "the report is data returned to the caller; measuring writes nothing of its own (snapshots only read) and adds no schema table — TF-105's filed schema facts stand"},
)


# ── Store snapshots ─────────────────────────────────────────────────────────

def store_snapshot(db_path: Path | str) -> dict[str, Any]:
    """Measure a store without creating or writing it.

    A missing store files ``exists=False``, zero bytes
    and ``counts=None`` (never an empty file conjured
    by the measuring — a plain sqlite connect would
    create one).
    """
    from thoughtforge.etl.schema import get_entity_count

    path = Path(db_path)
    file_bytes = {
        label: (candidate.stat().st_size if candidate.exists() else 0)
        for label, candidate in (
            ("db", path),
            ("wal", Path(str(path) + "-wal")),
            ("shm", Path(str(path) + "-shm")),
        )
    }
    counts: dict[str, int] | None = None
    if path.is_file():
        counts = get_entity_count(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": sum(file_bytes.values()),
        "file_bytes": file_bytes,
        "counts": counts,
    }


def _growth(before: dict[str, Any],
            after: dict[str, Any]) -> dict[str, Any]:
    before_counts = before.get("counts") or {}
    after_counts = after.get("counts") or {}
    counts: dict[str, int | None] = {}
    for table, after_n in after_counts.items():
        if after_n is None or after_n < 0:
            counts[table] = None
        else:
            counts[table] = after_n - int(before_counts.get(table, 0) or 0)
    return {
        "counts": counts,
        "total_rows": sum(n for n in counts.values() if n is not None),
        "bytes": int(after.get("bytes", 0)) - int(before.get("bytes", 0)),
    }


# ── The report ──────────────────────────────────────────────────────────────

@dataclass
class IngestionMetrics:
    source: str
    input_path: str
    db_path: str
    stats: dict[str, Any]
    records: dict[str, Any]
    elapsed_seconds: float | None
    records_per_second: float | None
    refusals: dict[str, Any]
    merge: dict[str, Any]
    update: dict[str, Any] | None
    update_note: str | None
    store_before: dict[str, Any]
    store_after: dict[str, Any]
    growth: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return metrics_report_as_data(self)


def metrics_report_as_data(report: IngestionMetrics) -> dict[str, Any]:
    """One run's report as plain data (the filed format)."""
    return {
        "format": METRICS_FORMAT,
        "doc": METRICS_DOC,
        "source": report.source,
        "input_path": report.input_path,
        "db_path": report.db_path,
        "stats": dict(report.stats),
        "records": dict(report.records),
        "elapsed_seconds": report.elapsed_seconds,
        "records_per_second": report.records_per_second,
        "refusals": dict(report.refusals),
        "merge": report.merge,
        "update": report.update,
        "update_note": report.update_note,
        "store_before": report.store_before,
        "store_after": report.store_after,
        "growth": report.growth,
    }


def build_run_metrics(
    source: str,
    stats: dict[str, Any],
    gate: dict[str, Any] | None,
    merge: dict[str, Any],
    update: dict[str, Any] | None,
    update_note: str | None,
    before: dict[str, Any],
    after: dict[str, Any],
    elapsed_seconds: float | None,
    input_path: str = "",
    db_path: str = "",
) -> IngestionMetrics:
    """Compute one run's metrics from its measured parts.

    Pure computation — the wrapper below measures the
    parts (clock, snapshots, ledgers); tests and other
    wrappers may file parts they measured themselves.
    Nothing here invents a part that was not measured.
    """
    shape = _shape(source)
    accepted = int(stats.get(shape["accepted_key"], 0))
    skipped = int(stats.get("skipped", 0))
    presented = accepted + skipped
    output_rows = sum(int(stats.get(k, 0)) for k in shape["output_keys"])
    primary_count = int(stats.get(shape["primary_key"], 0))

    if elapsed_seconds is not None and elapsed_seconds > 0:
        rate: float | None = presented / elapsed_seconds
        elapsed_filed: float | None = float(elapsed_seconds)
    else:
        rate = None
        elapsed_filed = (float(elapsed_seconds)
                         if elapsed_seconds is not None else None)

    growth = _growth(before, after)
    primary_growth = growth["counts"].get(shape["primary_table"])

    if gate is not None:
        seen = int(gate.get("seen", 0))
        refused = int(gate.get("refused", 0))
        refusals = {
            "gate_present": True,
            "seen": seen,
            "refused": refused,
            "refusal_rate": (refused / seen) if seen > 0 else None,
            "by_reason": dict(gate.get("by_reason", {})),
            "by_source": dict(gate.get("by_source", {})),
            "other_skips": skipped - refused,
            "note": None,
        }
    else:
        refusals = {
            "gate_present": False,
            "seen": None,
            "refused": None,
            "refusal_rate": None,
            "by_reason": {},
            "by_source": {},
            "other_skips": None,
            "note": ("this builder carries no schema gate "
                     "(TF-108's filed limit: the reference "
                     "builder's records are builder-generated "
                     "chunks, not external records)"),
        }

    growth = {
        **growth,
        "primary_table": shape["primary_table"],
        "primary_table_rows": primary_growth,
        "accepted_vs_primary_growth": (
            primary_count - primary_growth
            if primary_growth is not None else None),
    }

    return IngestionMetrics(
        source=source,
        input_path=input_path,
        db_path=db_path,
        stats=dict(stats),
        records={
            "unit": shape["unit"],
            "accepted": accepted,
            "skipped": skipped,
            "presented": presented,
            "output_rows": output_rows,
        },
        elapsed_seconds=elapsed_filed,
        records_per_second=rate,
        refusals=refusals,
        merge=merge,
        update=update,
        update_note=update_note,
        store_before=before,
        store_after=after,
        growth=growth,
    )


# ── The driver ──────────────────────────────────────────────────────────────

def _build(source: str, db_path: Path, input_path: Path,
           builder_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Construct the real builder on the named store and run it.

    The binding (class + input keyword) is TF-112's
    BUILDER_SOURCES map, reused — one binding, two
    consumers. Unlike the dry run, no resume override is
    imposed: a measured real run uses the builder's own
    defaults (wikidata resumes, per TF-107).
    """
    from thoughtforge.etl.ingestion_dry_run import BUILDER_SOURCES

    entry = next(e for e in BUILDER_SOURCES
                 if e["builder_id"] == source)
    if source == "wikidata":
        from thoughtforge.etl.wikidata import WikidataETL

        builder = WikidataETL(db_path)
    else:
        from thoughtforge.etl import sources as sources_module

        builder = getattr(sources_module, entry["class"])(db_path)
    stats = builder.build(
        **{entry["input_kwarg"]: input_path}, **builder_kwargs)
    return builder, stats


def measure_ingestion(
    source: str,
    input_path: Path | str,
    db_path: Path | str,
    builder_kwargs: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> IngestionMetrics:
    """Ingest one source into a store; return its per-run metrics.

    ``source`` is a TF-105 builder id (``wikidata`` /
    ``conceptnet`` / ``geonames`` / ``dbpedia`` /
    ``reference``). The store is snapshotted before and
    after, the build is timed by ``clock`` (the build
    call only), and the build's gate / merge / update
    ledgers ride the report whole. A builder failure
    propagates unchanged and files no report.
    """
    _shape(source)  # an unregistered source raises before anything runs
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"metrics input not found for source {source!r}: "
            f"{input_path}")
    db_path = Path(db_path)

    before = store_snapshot(db_path)
    started = clock()
    builder, stats = _build(source, db_path, input_path,
                            builder_kwargs or {})
    elapsed = clock() - started
    after = store_snapshot(db_path)

    if hasattr(builder, "schema_gate"):
        gate: dict[str, Any] | None = builder.schema_gate.as_data()
    else:
        gate = None
    merge = (builder.merge_ledger.as_data()
             if hasattr(builder, "merge_ledger") else {
                 "format": "thoughtforge-dedup-merge/1",
                 "collisions": 0, "by_outcome": {}, "by_table": {},
                 "decisions": []})
    if hasattr(builder, "update_ledger"):
        update: dict[str, Any] | None = builder.update_ledger.as_data()
        update_note = None
    else:
        update = None
        update_note = ("this builder carries no update ledger: its "
                       "changed rows ride the merge ledger as "
                       "ledgered supersedes (TF-114's filed limit)")

    return build_run_metrics(
        source, stats, gate, merge, update, update_note,
        before, after, elapsed,
        input_path=str(input_path), db_path=str(db_path))


# ── Aggregation ─────────────────────────────────────────────────────────────

def summarise_runs(
    reports: list[IngestionMetrics | dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-run reports per source (+ totals).

    Rates are recomputed from the summed presented and
    summed elapsed, and refusal rates from the summed
    gate counts — never averaged from per-run rates.
    Runs whose elapsed was unmeasured contribute their
    records but no time (filed in ``unmeasured_runs``).
    """
    per_source: dict[str, dict[str, Any]] = {}
    for item in reports:
        data = (item.to_dict() if isinstance(item, IngestionMetrics)
                else dict(item))
        bucket = per_source.setdefault(data["source"], {
            "runs": 0, "presented": 0, "accepted": 0, "skipped": 0,
            "output_rows": 0, "gate_seen": 0, "gate_refused": 0,
            "elapsed_seconds": 0.0, "unmeasured_runs": 0,
            "growth_rows": 0, "growth_bytes": 0,
            "accepted_vs_primary_growth": 0,
        })
        bucket["runs"] += 1
        records = data["records"]
        bucket["presented"] += int(records["presented"])
        bucket["accepted"] += int(records["accepted"])
        bucket["skipped"] += int(records["skipped"])
        bucket["output_rows"] += int(records["output_rows"])
        refusals = data["refusals"]
        if refusals.get("gate_present"):
            bucket["gate_seen"] += int(refusals["seen"] or 0)
            bucket["gate_refused"] += int(refusals["refused"] or 0)
        if data.get("elapsed_seconds") is not None:
            bucket["elapsed_seconds"] += float(data["elapsed_seconds"])
        else:
            bucket["unmeasured_runs"] += 1
        growth = data["growth"]
        bucket["growth_rows"] += int(growth["total_rows"])
        bucket["growth_bytes"] += int(growth["bytes"])
        diff = growth.get("accepted_vs_primary_growth")
        if diff is not None:
            bucket["accepted_vs_primary_growth"] += int(diff)

    def _finish(bucket: dict[str, Any]) -> dict[str, Any]:
        elapsed = bucket["elapsed_seconds"]
        return {
            **bucket,
            "records_per_second": (
                bucket["presented"] / elapsed if elapsed > 0 else None),
            "refusal_rate": (
                bucket["gate_refused"] / bucket["gate_seen"]
                if bucket["gate_seen"] > 0 else None),
        }

    sources = {name: _finish(bucket)
               for name, bucket in sorted(per_source.items())}
    totals_bucket = {
        "runs": sum(b["runs"] for b in per_source.values()),
        "presented": sum(b["presented"] for b in per_source.values()),
        "accepted": sum(b["accepted"] for b in per_source.values()),
        "skipped": sum(b["skipped"] for b in per_source.values()),
        "output_rows": sum(b["output_rows"] for b in per_source.values()),
        "gate_seen": sum(b["gate_seen"] for b in per_source.values()),
        "gate_refused": sum(b["gate_refused"]
                            for b in per_source.values()),
        "elapsed_seconds": sum(b["elapsed_seconds"]
                               for b in per_source.values()),
        "unmeasured_runs": sum(b["unmeasured_runs"]
                               for b in per_source.values()),
        "growth_rows": sum(b["growth_rows"] for b in per_source.values()),
        "growth_bytes": sum(b["growth_bytes"]
                            for b in per_source.values()),
        "accepted_vs_primary_growth": sum(
            b["accepted_vs_primary_growth"] for b in per_source.values()),
    }
    return {
        "format": METRICS_FORMAT,
        "doc": METRICS_DOC,
        "sources": sources,
        "totals": _finish(totals_bucket),
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def metrics_problems() -> list[str]:
    """Static truth: the filed shapes + rules against the live tree."""
    from thoughtforge.etl.etl_map import SOURCE_BUILDERS
    from thoughtforge.etl.ingestion_dry_run import BUILDER_SOURCES

    problems: list[str] = []
    rule_ids = [r["id"] for r in METRICS_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate METRICS_RULES ids")
    for needed in ("metrics_are_measured_never_derived_from_stats_alone",
                   "refusal_rate_is_gate_refused_over_gate_seen",
                   "store_growth_is_a_delta_of_snapshots",
                   "accepted_is_not_landed_and_the_difference_is_filed",
                   "a_failed_build_files_no_report"):
        if needed not in rule_ids:
            problems.append(f"METRICS_RULES lost {needed}")

    builder_ids = [b["id"] for b in SOURCE_BUILDERS]
    shaped = [e["builder_id"] for e in SOURCE_METRIC_SHAPES]
    if shaped != builder_ids:
        problems.append(
            f"SOURCE_METRIC_SHAPES {shaped} != TF-105 builders "
            f"{builder_ids}")
    dry_ids = [e["builder_id"] for e in BUILDER_SOURCES]
    if shaped != dry_ids:
        problems.append(
            f"SOURCE_METRIC_SHAPES {shaped} != TF-112 bindings "
            f"{dry_ids}")
    filed_tables = {b["id"]: tuple(b["tables"]) for b in SOURCE_BUILDERS}
    for entry in SOURCE_METRIC_SHAPES:
        if entry["target_tables"] != filed_tables[entry["builder_id"]]:
            problems.append(
                f"{entry['builder_id']}: target tables drifted from "
                "TF-105's filed tables")
        if entry["primary_table"] not in entry["target_tables"]:
            problems.append(
                f"{entry['builder_id']}: primary table is not a target")

    from thoughtforge.etl.etl_map import ORCHESTRATION

    if "ingestion_metrics" not in ORCHESTRATION.get("metrics", ""):
        problems.append("TF-105's map lost its metrics orchestration "
                        "entry")
    doc = Path(__file__).resolve().parents[3] / METRICS_DOC
    if not doc.exists():
        problems.append(f"the write-up {METRICS_DOC} is missing")
    return problems


def _geonames_inputs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = ["115001", "Lantern Town", "Lantern Town", "", "1.5",
            "2.5", "P", "PPL", "XX", "", "A1", "A2", "", "", "100",
            "10", "10", "UTC", "2026-01-01"]
    changed = list(base); changed[1] = "Lantern Town Renamed"
    bad = list(base); bad[0] = "115002"; bad[4] = "999.0"
    txt_path = root / "places.txt"
    txt_path.write_text("\n".join(
        ["\t".join(base), "\t".join(changed), "\t".join(bad)]) + "\n",
        encoding="utf-8")
    return txt_path


def metrics_live_problems() -> list[str]:
    """Behavioural truth: measured runs through the real builders."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # A missing store snapshots without being created.
        ghost = root / "ghost.db"
        snap = store_snapshot(ghost)
        if snap["exists"] or snap["counts"] is not None or ghost.exists():
            problems.append(
                f"a missing store was created by measuring: {snap}")

        # GeoNames: 1 new + 1 conflict + 1 gate refusal in
        # one measured run under a fake clock (2.0 s).
        ticks = iter([10.0, 12.0])
        db = root / "g.db"
        report = measure_ingestion(
            "geonames", _geonames_inputs(root / "in"), db,
            clock=lambda: next(ticks))
        if report.stats != {"inserted": 2, "skipped": 1}:
            problems.append(f"geonames stats drifted: {report.stats}")
        if report.records != {"unit": "places", "accepted": 2,
                              "skipped": 1, "presented": 3,
                              "output_rows": 2}:
            problems.append(f"geonames records drifted: {report.records}")
        if report.elapsed_seconds != 2.0 or report.records_per_second != 1.5:
            problems.append(
                "geonames rate drifted: "
                f"{report.elapsed_seconds} / {report.records_per_second}")
        if report.refusals["refused"] != 1 or \
                report.refusals["by_reason"] != {"out_of_range": 1} or \
                abs(report.refusals["refusal_rate"] - (1 / 3)) > 1e-9 or \
                report.refusals["other_skips"] != 0:
            problems.append(f"geonames refusals drifted: {report.refusals}")
        if report.merge["collisions"] != 1 or \
                report.merge["by_outcome"] != {"conflict_overwrite": 1}:
            problems.append(f"geonames merge drifted: {report.merge}")
        if report.update is not None or not report.update_note:
            problems.append("geonames filed an update ledger it lacks")
        if report.growth["counts"].get("geonames_places") != 1 or \
                report.growth["primary_table_rows"] != 1 or \
                report.growth["accepted_vs_primary_growth"] != 1:
            problems.append(f"geonames growth drifted: {report.growth}")
        if report.store_before["exists"] or \
                report.store_after["counts"]["geonames_places"] != 1:
            problems.append("geonames snapshots drifted")

        # An identical re-run: accepted again, zero growth,
        # the difference files the duplicate (D-MET-3).
        ticks = iter([20.0, 21.0])
        rerun = measure_ingestion(
            "geonames", _geonames_inputs(root / "in"), db,
            clock=lambda: next(ticks))
        # (the changed row now conflicts back; the bad row
        # refuses again — presented 3, store still 1 row)
        if rerun.growth["counts"].get("geonames_places") != 0 or \
                rerun.growth["accepted_vs_primary_growth"] != 2:
            problems.append(f"re-run growth drifted: {rerun.growth}")

        # The ungated reference builder files no rate.
        ref_dir = root / "ref"
        ref_dir.mkdir()
        (ref_dir / "lantern_care.md").write_text(
            "# Lantern Care\n\nKeep the lantern trimmed.\n",
            encoding="utf-8")
        ticks = iter([30.0, 31.0])
        ref = measure_ingestion("reference", ref_dir, root / "r.db",
                                 clock=lambda: next(ticks))
        if ref.refusals["gate_present"] or \
                ref.refusals["refusal_rate"] is not None:
            problems.append(f"reference filed a gate rate: {ref.refusals}")
        if ref.update is None or ref.update["updates"] != 0:
            problems.append(f"reference update ledger drifted: {ref.update}")
        if ref.growth["counts"].get("reference_chunks") != 1:
            problems.append(f"reference growth drifted: {ref.growth}")

        # Aggregation recomputes the rate from the sums:
        # (3 + 3) presented over (2.0 + 1.0) s = 2.0 /s.
        summary = summarise_runs([report, rerun])
        geo = summary["sources"]["geonames"]
        if geo["runs"] != 2 or geo["presented"] != 6 or \
                geo["records_per_second"] != 2.0 or \
                geo["growth_rows"] != 1:
            problems.append(f"aggregation drifted: {geo}")

        # A failed build files no report.
        try:
            measure_ingestion("geonames", root / "missing.txt", db)
            problems.append("a missing input was measured")
        except FileNotFoundError:
            pass
    return problems


def metrics_as_data() -> dict[str, Any]:
    """The metrics shapes + rules as plain data, for consumers."""
    return {
        "format": METRICS_FORMAT,
        "doc": METRICS_DOC,
        "sources": [dict(e) for e in SOURCE_METRIC_SHAPES],
        "rules": [dict(r) for r in METRICS_RULES],
    }


__all__ = [
    "METRICS_DOC",
    "METRICS_FORMAT",
    "METRICS_RULES",
    "SOURCE_METRIC_SHAPES",
    "IngestionMetrics",
    "build_run_metrics",
    "measure_ingestion",
    "metrics_as_data",
    "metrics_live_problems",
    "metrics_problems",
    "metrics_report_as_data",
    "store_snapshot",
    "summarise_runs",
]
