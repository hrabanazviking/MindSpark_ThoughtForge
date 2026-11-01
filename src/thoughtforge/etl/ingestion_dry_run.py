"""
Ingestion dry-run, as data (TF-112).

The roadmap's rule: *a source can be test-ingested into a
scratch store with a full report before touching a real
store.*

Before this slice, that was impossible in fact — probed
live on this tree before anything was built:

  * **No dry-run existed anywhere.** No module, no
    ``forge_memory.py`` command and no test named a dry
    run (grep over ``src/``, ``tests/`` and the root
    instrument: the only hits were the filed promises in
    TF-106 / TF-107 / TF-108 / TF-109 / TF-111 that
    TF-112 would be a consumer). The only way to find
    out what a source file would do was to build it
    straight into a real store.
  * **The registry was consulted nowhere at build
    time** (D-ETL-2, TF-106): neither builder module
    nor ``forge_memory.py`` mentions ``source_registry``
    or ``check_source_use`` — a quarantined source, or
    an unregistered one, had no gate on the road from a
    file to a store.
  * **A build's full story was thrown away.** A
    builder's stats are the only thing the CLI prints;
    the TF-108 schema-gate ledger and the TF-109 merge
    ledger live in memory on the builder instance and
    are discarded by every command in the tree (live
    probe, GeoNames: a valid row + a row with latitude
    999.0 built ``{'inserted': 1, 'skipped': 1}`` —
    the refusal's reason, ``out_of_range:latitude``,
    existed only in the discarded ledger), and
    TF-109's cross-source overlaps had no reporting
    consumer at all.

What this slice builds, filed as data in
``DRY_RUN_RULES`` and enforced here:

  * ``dry_run_ingestion()`` — one source, test-ingested
    by its REAL builder into a scratch store that is
    never the real store, returning a ``DryRunReport``
    (``report_as_data()`` is its plain-data shape,
    format ``thoughtforge-ingestion-dry-run/1``).
  * **The registry is consulted first** (D-ETL-2's
    TF-112 share): the source's primary TF-106 entry
    is judged by ``check_source_use(..., ingest_private)``
    BEFORE any scratch file is created and before any
    builder runs — an unregistered source raises by
    name with nothing created; a quarantined source
    (the reference library as a whole) passes private
    ingestion with its quarantine, licence (None) and
    obligations stated in the report, exactly as the
    registry files them. The builders themselves still
    consult nothing — TF-119 owns the recipe share of
    D-ETL-2, and TF-106's static checker keeps pinning
    that builder-side shape.
  * **The real store is never touched — and that is
    proven, not asserted.** The scratch path must not
    resolve to the named target store (that ask raises
    ``DryRunTargetError`` before anything runs). When
    a target is named, the report carries its
    before/after state: existence, byte size, SHA-256
    (when it is a file) and row counts (when it is a
    store) must agree exactly, or ``target_untouched``
    is False in the open. When no target is named, the
    report says so, and the only database written is
    the scratch.
  * **The full report.** Builder stats in TF-105's
    filed shapes; the scratch store's row counts from
    ``get_entity_count``; the builder's TF-108 gate
    ledger as data (refusals with reasons, by source
    and by reason code — the ledger's first reporting
    consumer; the reference builder is not gated, and
    the report files that as ``gate is None`` with its
    reason, never a fabricated empty ledger); the
    builder's TF-109 merge ledger as data (collisions
    by outcome and by table — its first reporting
    consumer); TF-109's ``find_overlaps`` over the
    scratch estate; and the scratch file's measured
    byte size.
  * **Embeddings, only when an encoder is supplied.**
    TF-111 filed this dry-run as the first consumer of
    its injected encoder: when the caller passes an
    ``encoder``, TF-111's driver runs over the scratch
    estate and the report carries its counts and
    estate digest; with no encoder, the report files
    the leg as not run — no real model is ever loaded
    by a dry run on its own word.
  * **Scratch discipline.** By default the scratch
    lives in a fresh temporary directory and is
    removed (database, WAL/SHM sidecars and builder
    checkpoint sidecars) before the report returns;
    ``keep_scratch=True`` keeps it at the reported
    path for inspection, and sequential dry runs into
    one kept scratch compose only when the caller
    passes ``append=True`` explicitly — a non-append
    run always starts from no scratch file. A failed
    build re-raises unchanged and still removes an
    unkept scratch: a dry run that failed leaves no
    half store behind it pretending to be a result.

Filed limits, stated plainly:

  * A dry run judges ONE source per call (sequential
    calls into a kept scratch compose an estate when a
    caller wants cross-source overlaps; a single-source
    scratch can only name overlaps its own tables
    create, which is usually none — filed, not hidden).
  * The dry run runs the builder with the caller's
    builder keyword arguments; the wikidata leg runs
    with ``resume=False`` unless the caller overrides
    it, because a scratch is a fresh ask by definition.
  * Gate and merge ledgers report the ONE build of
    this call, even on an appended scratch — they are
    per-build ledgers by TF-108/TF-109's own filing.
  * The report is data in memory / returned to the
    caller; no report file is written anywhere by
    this module, and no schema table is added
    (TF-105's filed schema facts stand).

Docket (TF-016): D-ETL-2 is narrowed by this slice —
its TF-112 share (the dry-run consults the registry
first) is FIXED here; the entry stays OPEN for
TF-119's new-source-recipe share (the builders
themselves consult nothing, as TF-106's checker
pins). D-COV-2 is FIXED by this slice: its remaining
share was TF-112's dry-run depth, and the dry-run's
live checker + tests now exercise all five builders
and both embedding legs directly, in depth, on
synthetic inputs.

The human-readable write-up is
``docs/architecture/INGESTION_DRY_RUN.md``. Proof
grade: FIXTURE — synthetic sources through the real
builders into scratch stores on this VM; no real
dump has been dry-run at scale, and none is claimed.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DRY_RUN_FORMAT = "thoughtforge-ingestion-dry-run/1"
DRY_RUN_DOC = "docs/architecture/INGESTION_DRY_RUN.md"

#: The scratch database's file name inside a scratch dir.
SCRATCH_DB_NAME = "dry-run-scratch.db"


class DryRunTargetError(ValueError):
    """The scratch would be (or would touch) the real store."""


# ── The source map ──────────────────────────────────────────────────────────
#
# One entry per TF-105 builder: the builder class, the
# keyword its build() takes for its input, and the
# TF-106 primary registry entry that must be consulted
# before it runs. The reference builder's registry id
# differs from its builder id (reference_library binds
# builder "reference" as its primary entry).

BUILDER_SOURCES: tuple[dict[str, Any], ...] = (
    {"builder_id": "wikidata", "registry_id": "wikidata",
     "class": "WikidataETL", "module": "etl/wikidata.py",
     "input_kwarg": "dump_path", "input_kind": "file"},
    {"builder_id": "conceptnet", "registry_id": "conceptnet",
     "class": "ConceptNetETL", "module": "etl/sources.py",
     "input_kwarg": "csv_path", "input_kind": "file"},
    {"builder_id": "geonames", "registry_id": "geonames",
     "class": "GeoNamesETL", "module": "etl/sources.py",
     "input_kwarg": "txt_path", "input_kind": "file"},
    {"builder_id": "dbpedia", "registry_id": "dbpedia",
     "class": "DBpediaETL", "module": "etl/sources.py",
     "input_kwarg": "json_path", "input_kind": "file"},
    {"builder_id": "reference", "registry_id": "reference_library",
     "class": "ReferenceDataETL", "module": "etl/sources.py",
     "input_kwarg": "ref_dir", "input_kind": "directory"},
)


def _source_entry(source: str) -> dict[str, Any]:
    """Resolve a builder id or its registry id to the map entry."""
    for entry in BUILDER_SOURCES:
        if source in (entry["builder_id"], entry["registry_id"]):
            return entry
    raise ValueError(
        f"unregistered source {source!r}: the dry run covers "
        f"{[e['builder_id'] for e in BUILDER_SOURCES]} — a new "
        "source is filed in TF-105's map and TF-106's registry "
        "before it is dry-run (TF-119's recipe)")


# ── The rules ───────────────────────────────────────────────────────────────

DRY_RUN_RULES: tuple[dict[str, Any], ...] = (
    {"id": "the_registry_is_consulted_first",
     "rule": "check_source_use(registry_id, ingest_private) runs before any scratch file is created and before any builder runs; an unregistered source raises by name with nothing created, and a quarantined source passes private ingestion with its quarantine stated in the report (D-ETL-2, TF-112 share)"},
    {"id": "a_dry_run_writes_only_the_scratch",
     "rule": "the builder is constructed with the scratch path and nothing else: no builder ever receives the real store's path in a dry run"},
    {"id": "the_scratch_is_never_the_real_store",
     "rule": "a scratch path that resolves to the named target store raises DryRunTargetError before anything runs; a target and a scratch in one directory remain different files"},
    {"id": "the_target_is_proven_untouched",
     "rule": "when a target store is named, its existence, byte size, SHA-256 and row counts are measured before and after the run and must agree exactly for the report to carry target_untouched=True; a target that did not exist must still not exist"},
    {"id": "a_missing_input_raises_before_the_scratch",
     "rule": "an input file/directory that does not exist raises FileNotFoundError naming it before any scratch file is created — a dry run of nothing files no report"},
    {"id": "the_report_is_the_full_story",
     "rule": "the report carries the builder stats (TF-105 shapes), the scratch store counts, the TF-108 gate ledger as data where the builder is gated, the TF-109 merge ledger as data, TF-109's overlaps over the scratch estate, and the scratch file's measured bytes — nothing a builder knew is discarded"},
    {"id": "an_ungated_builder_files_no_gate",
     "rule": "the reference builder carries no schema gate (TF-108's filed limit): its report files gate=None with the reason, never an empty ledger dressed as a judgement"},
    {"id": "embeddings_only_with_an_injected_encoder",
     "rule": "the embedding leg runs only when the caller injects an encoder (TF-111's filed consumer), through TF-111's driver over the scratch estate; with no encoder the leg is filed not-run and no model is loaded"},
    {"id": "scratch_is_removed_unless_kept",
     "rule": "an unkept scratch (database + WAL/SHM + builder checkpoint sidecars) is removed before the report returns, including after a failed build; a kept scratch stays at the reported path and is the caller's to remove"},
    {"id": "composition_is_explicit",
     "rule": "a non-append run starts from no scratch file; sequential runs compose one scratch estate only when append=True is passed explicitly, and each report still files only its own build's ledgers"},
    {"id": "a_failed_build_reraises_unchanged",
     "rule": "a builder failure propagates as the builder raised it — a dry run never converts a crash into a report with invented stats"},
)


# ── Target measurement ──────────────────────────────────────────────────────

def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure_target(path: Path) -> dict[str, Any]:
    from thoughtforge.etl.schema import get_entity_count

    measured: dict[str, Any] = {
        "path": str(path), "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _file_sha256(path), "counts": None,
    }
    if path.is_file():
        try:
            measured["counts"] = get_entity_count(path)
        except Exception:
            measured["counts"] = None
    return measured


def _targets_agree(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return all(before[key] == after[key]
               for key in ("exists", "size_bytes", "sha256", "counts"))


def _scratch_sidecars(scratch_db: Path) -> list[Path]:
    return [Path(str(scratch_db) + suffix) for suffix in (
        "-wal", "-shm", ".wikidata-checkpoint.json",
        ".embedding-checkpoint.json")]


def _remove_scratch(scratch_db: Path) -> None:
    for candidate in [scratch_db, *_scratch_sidecars(scratch_db)]:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


# ── The report ──────────────────────────────────────────────────────────────

@dataclass
class DryRunReport:
    source: str
    registry_id: str
    registry: dict[str, Any]
    input_path: str
    stats: dict[str, Any]
    store_counts: dict[str, int]
    gate: dict[str, Any] | None
    gate_note: str | None
    merge: dict[str, Any]
    overlaps: dict[str, Any]
    embeddings: dict[str, Any]
    scratch_path: str
    scratch_size_bytes: int
    scratch_kept: bool
    appended: bool
    target: dict[str, Any] | None
    target_untouched: bool | None

    def to_dict(self) -> dict[str, Any]:
        return report_as_data(self)


def report_as_data(report: DryRunReport) -> dict[str, Any]:
    """The report as plain data (the filed format)."""
    return {
        "format": DRY_RUN_FORMAT,
        "doc": DRY_RUN_DOC,
        "source": report.source,
        "registry_id": report.registry_id,
        "registry": dict(report.registry),
        "input_path": report.input_path,
        "stats": dict(report.stats),
        "store_counts": dict(report.store_counts),
        "gate": report.gate,
        "gate_note": report.gate_note,
        "merge": report.merge,
        "overlaps": report.overlaps,
        "embeddings": dict(report.embeddings),
        "scratch_path": report.scratch_path,
        "scratch_size_bytes": report.scratch_size_bytes,
        "scratch_kept": report.scratch_kept,
        "appended": report.appended,
        "target": report.target,
        "target_untouched": report.target_untouched,
    }


# ── The driver ──────────────────────────────────────────────────────────────

def _build(builder_entry: dict[str, Any], scratch_db: Path,
           input_path: Path, builder_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Construct the real builder on the scratch and run it."""
    kwargs = dict(builder_kwargs)
    if builder_entry["builder_id"] == "wikidata":
        from thoughtforge.etl.wikidata import WikidataETL

        kwargs.setdefault("resume", False)
        builder = WikidataETL(scratch_db)
        stats = builder.build(**{builder_entry["input_kwarg"]: input_path},
                              **kwargs)
        return builder, stats
    from thoughtforge.etl import sources as sources_module

    builder = getattr(sources_module, builder_entry["class"])(scratch_db)
    stats = builder.build(**{builder_entry["input_kwarg"]: input_path},
                          **kwargs)
    return builder, stats


def dry_run_ingestion(
    source: str,
    input_path: Path | str,
    scratch_dir: Path | str | None = None,
    target_db_path: Path | str | None = None,
    keep_scratch: bool = False,
    append: bool = False,
    builder_kwargs: dict[str, Any] | None = None,
    encoder: Callable[..., Any] | None = None,
) -> DryRunReport:
    """Test-ingest one source into a scratch store; file the full report.

    ``source`` is a TF-105 builder id (``wikidata`` /
    ``conceptnet`` / ``geonames`` / ``dbpedia`` /
    ``reference``) or, for the reference builder, its
    registry id ``reference_library``. ``input_path`` is
    the source file (a directory for ``reference``).
    See the module docstring for the filed rules.
    """
    from thoughtforge.etl.dedup_merge import overlap_as_data
    from thoughtforge.etl.schema import get_entity_count
    from thoughtforge.etl.source_registry import (
        USE_INGEST_PRIVATE,
        check_source_use,
    )

    entry = _source_entry(source)

    # Rule: the registry is consulted FIRST — before the
    # input is even stat'ed, and before any scratch exists.
    registry_entry = check_source_use(entry["registry_id"],
                                      USE_INGEST_PRIVATE)
    registry = {
        "id": registry_entry["id"],
        "name": registry_entry["name"],
        "provenance": registry_entry["provenance"],
        "licence": registry_entry["licence"],
        "status": registry_entry["status"],
        "obligations": list(registry_entry["obligations"]),
        "use_checked": USE_INGEST_PRIVATE,
    }

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"dry-run input not found for source "
            f"{entry['builder_id']!r}: {input_path}")
    if entry["input_kind"] == "file" and not input_path.is_file():
        raise FileNotFoundError(
            f"dry-run input for source {entry['builder_id']!r} "
            f"is not a file: {input_path}")
    if entry["input_kind"] == "directory" and not input_path.is_dir():
        raise FileNotFoundError(
            f"dry-run input for source {entry['builder_id']!r} "
            f"is not a directory: {input_path}")

    target = Path(target_db_path) if target_db_path is not None else None

    created_scratch_root = False
    if scratch_dir is None:
        scratch_root = Path(tempfile.mkdtemp(
            prefix="thoughtforge-dry-run-"))
        created_scratch_root = True
    else:
        scratch_root = Path(scratch_dir)
        scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_db = scratch_root / SCRATCH_DB_NAME

    try:
        if target is not None and scratch_db.resolve() == target.resolve():
            raise DryRunTargetError(
                f"the dry-run scratch {scratch_db} IS the target "
                f"store {target} — a dry run never writes the "
                "real store; choose another scratch_dir")
        target_before = _measure_target(target) if target is not None else None

        appended = bool(append and scratch_db.exists())
        if scratch_db.exists() and not append:
            _remove_scratch(scratch_db)

        builder, stats = _build(entry, scratch_db, input_path,
                                builder_kwargs or {})

        store_counts = get_entity_count(scratch_db)
        if hasattr(builder, "schema_gate"):
            gate: dict[str, Any] | None = builder.schema_gate.as_data()
            gate_note = None
        else:
            gate = None
            gate_note = ("this builder carries no schema gate "
                         "(TF-108's filed limit: the reference "
                         "builder's records are builder-generated "
                         "chunks, not external records)")
        merge = (builder.merge_ledger.as_data()
                 if hasattr(builder, "merge_ledger") else {
                     "format": "thoughtforge-dedup-merge/1",
                     "collisions": 0, "by_outcome": {}, "by_table": {},
                     "decisions": []})
        overlaps = overlap_as_data(scratch_db)

        embeddings: dict[str, Any]
        if encoder is not None:
            from thoughtforge.etl.embedding_ingestion import (
                build_embeddings_at_ingestion,
            )

            result = build_embeddings_at_ingestion(
                scratch_db, encoder=encoder)
            embeddings = {"status": "run", **result.to_dict()}
            # Counts move when embeddings land; re-measure.
            store_counts = get_entity_count(scratch_db)
        else:
            embeddings = {
                "status": "not_run",
                "reason": "no encoder was injected — a dry run "
                          "loads no model on its own word "
                          "(TF-111's injected-encoder leg)",
            }

        scratch_size = (scratch_db.stat().st_size
                        if scratch_db.is_file() else 0)
        target_after = _measure_target(target) if target is not None else None
        untouched = (_targets_agree(target_before, target_after)
                     if target is not None else None)
        target_report = None
        if target is not None:
            target_report = {"before": target_before, "after": target_after}

        return DryRunReport(
            source=entry["builder_id"],
            registry_id=entry["registry_id"],
            registry=registry,
            input_path=str(input_path),
            stats=dict(stats),
            store_counts=store_counts,
            gate=gate, gate_note=gate_note,
            merge=merge, overlaps=overlaps,
            embeddings=embeddings,
            scratch_path=str(scratch_db),
            scratch_size_bytes=scratch_size,
            scratch_kept=keep_scratch,
            appended=appended,
            target=target_report,
            target_untouched=untouched,
        )
    finally:
        if not keep_scratch:
            _remove_scratch(scratch_db)
            if created_scratch_root:
                try:
                    scratch_root.rmdir()
                except OSError:
                    pass


# ── Checkers ────────────────────────────────────────────────────────────────

def dry_run_problems() -> list[str]:
    """Static truth: the filed map + rules against the live tree."""
    from thoughtforge.etl.etl_map import SOURCE_BUILDERS
    from thoughtforge.etl.source_registry import SOURCES

    problems: list[str] = []
    rule_ids = [r["id"] for r in DRY_RUN_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate DRY_RUN_RULES ids")
    for needed in ("the_registry_is_consulted_first",
                   "the_scratch_is_never_the_real_store",
                   "the_target_is_proven_untouched",
                   "the_report_is_the_full_story",
                   "embeddings_only_with_an_injected_encoder"):
        if needed not in rule_ids:
            problems.append(f"DRY_RUN_RULES lost {needed}")

    builder_ids = [b["id"] for b in SOURCE_BUILDERS]
    mapped = [e["builder_id"] for e in BUILDER_SOURCES]
    if mapped != builder_ids:
        problems.append(
            f"BUILDER_SOURCES {mapped} != TF-105 builders {builder_ids}")
    primary = {e["builder_id"]: e for e in SOURCES if e["sub_path"] is None}
    for entry in BUILDER_SOURCES:
        registry_entry = primary.get(entry["builder_id"])
        if registry_entry is None or \
                registry_entry["id"] != entry["registry_id"]:
            problems.append(
                f"{entry['builder_id']}: registry binding drifted "
                f"(map says {entry['registry_id']!r})")
        module_path = Path(__file__).resolve().parent / (
            "wikidata.py" if entry["builder_id"] == "wikidata"
            else "sources.py")
        if entry["class"] not in module_path.read_text(encoding="utf-8"):
            problems.append(
                f"{entry['class']} missing from {module_path.name}")
    doc = Path(__file__).resolve().parents[3] / DRY_RUN_DOC
    if not doc.exists():
        problems.append(f"the write-up {DRY_RUN_DOC} is missing")
    return problems


def _synthetic_inputs(root: Path) -> dict[str, Path]:
    """Tiny synthetic inputs, one per builder (TF-017)."""
    import json

    root.mkdir(parents=True, exist_ok=True)
    entity = {
        "id": "Q112001",
        "labels": {"en": {"value": "Lantern Town"}},
        "descriptions": {"en": {"value": "A synthetic town"}},
        "aliases": {"en": [{"value": "the lit town"}]},
        "sitelinks": {"enwiki": {}},
        "claims": {"P31": [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": "Q112900"}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        }]},
    }
    dump = root / "dump.json"
    dump.write_text(json.dumps([entity]), encoding="utf-8")

    csv_path = root / "assertions.csv"
    csv_path.write_text(
        "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 2.0}\n",
        encoding="utf-8")

    base = ["112001", "Lantern Town", "Lantern Town", "", "1.5",
            "2.5", "P", "PPL", "XX", "", "A1", "A2", "", "", "100",
            "10", "10", "UTC", "2026-01-01"]
    changed = list(base); changed[1] = "Lantern Town Renamed"
    bad = list(base); bad[0] = "112002"; bad[4] = "999.0"
    txt_path = root / "places.txt"
    txt_path.write_text("\n".join(
        ["\t".join(base), "\t".join(changed), "\t".join(bad)]) + "\n",
        encoding="utf-8")
    single_path = root / "places_single.txt"
    single_path.write_text("\t".join(base) + "\n", encoding="utf-8")

    json_path = root / "abstracts.json"
    json_path.write_text(json.dumps({
        "uri": "http://example.org/Lantern_Town",
        "label": "Lantern Town", "abstract": "A synthetic town.",
        "type": "Place"}) + "\n", encoding="utf-8")

    ref_dir = root / "ref"
    ref_dir.mkdir(exist_ok=True)
    (ref_dir / "lantern_care.md").write_text(
        "# Lantern Care\n\nKeep the lantern trimmed.\n",
        encoding="utf-8")
    return {"wikidata": dump, "conceptnet": csv_path,
            "geonames": txt_path, "geonames_single": single_path,
            "dbpedia": json_path, "reference": ref_dir}


def dry_run_live_problems(base: Path | None = None) -> list[str]:
    """Behavioural truth: every builder dry-run, end to end."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(base) if base is not None else Path(tmp) / "live"
        inputs = _synthetic_inputs(root / "inputs")

        # A real store with one row, to prove untouched.
        from thoughtforge.etl.schema import get_connection, initialize_schema

        real_db = root / "real.db"
        initialize_schema(real_db)
        conn = get_connection(real_db)
        try:
            conn.execute(
                "INSERT INTO entities (qid, label_en) VALUES (?, ?)",
                ("Q999", "sentinel"))
            conn.commit()
        finally:
            conn.close()

        # GeoNames: 1 landed + 1 conflict collision + 1 gate
        # refusal, the full story in one report, target proven.
        report = dry_run_ingestion(
            "geonames", inputs["geonames"], target_db_path=real_db)
        if report.stats != {"inserted": 2, "skipped": 1}:
            problems.append(f"geonames dry-run stats drifted: {report.stats}")
        if report.store_counts.get("geonames_places") != 1:
            problems.append("geonames scratch counts drifted: "
                            f"{report.store_counts}")
        if report.gate is None or report.gate["refused"] != 1 \
                or report.gate["by_reason"] != {"out_of_range": 1}:
            problems.append(f"geonames gate report drifted: {report.gate}")
        if report.merge["collisions"] != 1 or \
                report.merge["by_outcome"] != {"conflict_overwrite": 1}:
            problems.append(f"geonames merge report drifted: {report.merge}")
        if report.target_untouched is not True:
            problems.append("the named real store was touched")
        if report.scratch_kept or Path(report.scratch_path).exists():
            problems.append("an unkept scratch survived its report")
        if report.registry["licence"] != "CC BY 4.0":
            problems.append("geonames registry entry not in the report")

        # Every other builder dry-runs through the same road.
        expected = {
            "wikidata": ({"entities": 1, "statements": 1, "labels": 1,
                          "skipped": 0}, "entities", 1),
            "conceptnet": ({"inserted": 1, "skipped": 0},
                           "conceptnet_edges", 1),
            "dbpedia": ({"inserted": 1, "skipped": 0},
                        "dbpedia_entities", 1),
            "reference": ({"files": 1, "chunks": 1, "skipped": 0},
                          "reference_chunks", 1),
        }
        for source, (stats, table, count) in expected.items():
            one = dry_run_ingestion(source, inputs[source],
                                    target_db_path=real_db)
            if one.stats != stats:
                problems.append(f"{source} dry-run stats drifted: {one.stats}")
            if one.store_counts.get(table) != count:
                problems.append(f"{source} scratch counts drifted: "
                                f"{one.store_counts}")
            if one.target_untouched is not True:
                problems.append(f"{source} dry run touched the real store")
        reference_report = dry_run_ingestion("reference_library",
                                             inputs["reference"])
        if reference_report.source != "reference":
            problems.append("the registry id did not resolve to reference")
        if reference_report.registry["status"] != "quarantined_unverified" \
                or reference_report.registry["licence"] is not None:
            problems.append("the quarantined reference library was not "
                            "filed quarantined in the report")
        if reference_report.gate is not None or not reference_report.gate_note:
            problems.append("the ungated reference builder filed a gate")

        # An unregistered source raises before creating anything.
        scratch_dir = root / "never-created"
        try:
            dry_run_ingestion("no_such_source_tf112", inputs["geonames"],
                              scratch_dir=scratch_dir)
            problems.append("an unregistered source was dry-run")
        except ValueError as exc:
            if "no_such_source_tf112" not in str(exc):
                problems.append(f"unregistered refusal unnamed: {exc}")
        if scratch_dir.exists():
            problems.append("an unregistered source created a scratch")

        # The scratch may never be the target.
        guard_dir = root / "guard"
        guard_dir.mkdir(exist_ok=True)
        try:
            dry_run_ingestion(
                "geonames", inputs["geonames"],
                scratch_dir=guard_dir,
                target_db_path=guard_dir / SCRATCH_DB_NAME)
            problems.append("a scratch equal to the target was allowed")
        except DryRunTargetError:
            pass
        if (guard_dir / SCRATCH_DB_NAME).exists():
            problems.append("the refused target run created its scratch")

        # Composition: two kept, appended runs name the overlap.
        # (The single-row GeoNames input keeps the shared name —
        # the three-row fixture above renames its place by merge.)
        shared = root / "shared"
        dry_run_ingestion("wikidata", inputs["wikidata"],
                          scratch_dir=shared, keep_scratch=True)
        composed = dry_run_ingestion(
            "geonames", inputs["geonames_single"], scratch_dir=shared,
            keep_scratch=True, append=True)
        if not composed.appended:
            problems.append("an append run did not know it appended")
        names = [g["name"] for g in composed.overlaps["groups"]]
        if "lantern town" not in names:
            problems.append(
                f"composed scratch overlaps miss lantern town: {names}")
        _remove_scratch(shared / SCRATCH_DB_NAME)

        # The injected-encoder leg (TF-111's filed consumer).
        def fake_encoder(texts, model_name):
            import hashlib

            import numpy as np

            vectors = []
            for text in texts:
                seed = int(hashlib.sha256(
                    text.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                vectors.append(rng.random(384, dtype=np.float32))
            return np.array(vectors, dtype=np.float32)

        embedded = dry_run_ingestion(
            "wikidata", inputs["wikidata"], encoder=fake_encoder)
        if embedded.embeddings.get("status") != "run" \
                or embedded.embeddings.get("entities_built") != 1:
            problems.append(
                f"the injected-encoder leg drifted: {embedded.embeddings}")
        if embedded.store_counts.get("embeddings") != 1:
            problems.append("embedded scratch counts drifted: "
                            f"{embedded.store_counts}")
    return problems


def dry_run_as_data() -> dict[str, Any]:
    """The dry-run's map as plain data, for consumers and the doctor."""
    return {
        "format": DRY_RUN_FORMAT,
        "doc": DRY_RUN_DOC,
        "sources": [dict(e) for e in BUILDER_SOURCES],
        "rules": [dict(r) for r in DRY_RUN_RULES],
    }
