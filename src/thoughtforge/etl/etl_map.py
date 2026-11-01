"""
ETL pipeline map, as data (TF-105).

Part VI opens by mapping the ingestion forge exactly as the
code stands — sources -> schema -> embeddings -> subset ->
store, drawn from the code, with the contract of every stage
kept as *verifiable data* rather than prose alone (the pattern
TF-041 set for the store and TF-087 for salvage):

  * ``PIPELINE_STAGES`` — the five stages in the roadmap's
    order, each with its module(s), what it consumes, what it
    produces, and the tables it owns.
  * ``SOURCE_BUILDERS`` — the five source builders of the
    sources stage: input shape, target table, write mode,
    batch default, stats shape, and their differing durability
    and logging behaviour, all checked against the live
    classes by ``etl_map_problems()``.
  * ``SCHEMA_FACTS`` / ``EMBEDDING_FACTS`` / ``SUBSET_FACTS``
    — the schema, embeddings, and subset stages' filed
    truths, re-derived from the live modules on every check.
  * ``ORCHESTRATION`` — how a full build actually runs today:
    there is no orchestrator inside this package. The one
    pipeline runner is the root instrument ``forge_memory.py``
    (its ``all`` command), and its step order is filed here
    from that file's own source.
  * ``ETL_RULES`` — the pipeline's contracts stated as data.
    ``etl_map_problems()`` re-checks the static ones against
    the live tree; ``etl_map_live_problems()`` runs a whole
    synthetic pipeline (every builder on tiny synthetic
    inputs, embeddings through a fake model, a limited subset
    build) in a temporary directory and checks the
    behavioural ones end to end.

Two defects this map carries as findings, not decorations:

  * **D-SUBSET-1 — FIXED by this slice.** The subset stage's
    reference-chunk copy ordered by ``retrieval_score``, a
    column the schema has never carried: every limited build
    (every edge profile) raised, swallowed the error into its
    notes, copied ZERO reference chunks, and still reported
    success — probed live before the fix (pi_zero build of a
    store holding one chunk: copied 0, success True). The
    fix orders by ``chunk_id``; the live checker below proves
    a limited subset now keeps its chunks.
  * **D-ETL-1 — filed OPEN, owner TF-119.** The orchestrator's
    ``--dbpedia-json`` option is accepted and never used: the
    ``all`` command's steps ingest reference, wikidata,
    conceptnet, and geonames data, then embeddings — a user
    who passes a DBpedia file is told nothing and gets no
    DBpedia rows. The static checker pins the gap's exact
    shape (the option exists; no DBpedia step exists), so
    wiring the step turns the checker red until this map is
    updated in the open.

Filed limits the map states plainly:

  * Only the wikidata builder writes the ``ingestion_log``;
    the other four leave no run record in the store.
  * Bulk loads run with ``synchronous = OFF`` (conceptnet,
    geonames, dbpedia) except the reference builder (the
    schema's NORMAL) and the embeddings builders (FULL,
    TF-054's hardening) — one pipeline, three durability
    postures, mapped not harmonised (TF-111 owns any further
    change). TF-107 moved the wikidata builder out of the
    OFF leg to FULL: its checkpoint commits must survive
    power loss, so the map's wikidata row now files FULL.
  * The subset copies exactly three tables' rows (entities,
    statements, reference chunks). Labels, both embedding
    tables, conceptnet edges, geonames places, dbpedia
    entries, and the ingestion log are schema-cloned but
    row-empty in a subset — a subset store has no vector
    search until embeddings are rebuilt against it.
  * The subset's schema clone copies table DDL only: the
    schema's 15 named indexes are not recreated (only the
    autoindexes a primary key or UNIQUE constraint implies).

The human-readable write-up is
``docs/architecture/ETL_MAPPED.md``. Proof grade: FIXTURE —
the live legs run synthetic inputs through the real builders
on this VM; no real dump (wikidata / conceptnet / geonames /
dbpedia at scale) has been ingested by this campaign, and
embedding generation is proven through a fake model only
(sentence-transformers is absent here, D-ENV-1) — a real
at-scale ingestion is NOT PROVEN, never claimed.
"""

from __future__ import annotations

import inspect
import json
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ETL_FORMAT = "thoughtforge-etl-map/1"
ETL_DOC = "docs/architecture/ETL_MAPPED.md"

_REPO_ROOT = Path(__file__).resolve().parents[3]

# ── The five stages (roadmap order: sources -> schema -> ───────────────────
# embeddings -> subset -> store). "position_in_build_all" records where
# the stage actually runs in the one orchestrator that exists: the
# schema is initialised FIRST there (step 1), not second — the roadmap
# order names the data flow, the orchestrator order is filed beside it
# so the difference is never smoothed over.

PIPELINE_STAGES: tuple[dict[str, Any], ...] = (
    {"id": "sources", "order": 1,
     "modules": ("etl/wikidata.py", "etl/sources.py"),
     "consumes": "external source files + the built-in reference directory",
     "produces": "rows in the builder's target table(s)",
     "position_in_build_all": "steps 2-5 (reference always; the rest only when their file is passed)",
     "contract": "every builder initialises the schema itself, raises FileNotFoundError on a missing input, and returns its own stats dict"},
    {"id": "schema", "order": 2,
     "modules": ("etl/schema.py",),
     "consumes": "a database path (created if absent)",
     "produces": "the 10-table + 3-FTS store shape, idempotently",
     "position_in_build_all": "step 1 (before every source)",
     "contract": "initialize_schema is idempotent (IF NOT EXISTS throughout); connections carry the filed pragma sets"},
    {"id": "embeddings", "order": 3,
     "modules": ("etl/embeddings.py", "etl/embedding_discipline.py", "etl/vector_store.py"),
     "consumes": "entities + reference chunks already in the store",
     "produces": "validated embedding BLOBs in embeddings + reference_embeddings",
     "position_in_build_all": "step 6 (with the FTS rebuild), skippable by flag",
     "contract": "only rows missing an embedding under the model being built are built (per-model since TF-111, D-EMB-3); every vector is dimension/finiteness-validated before it is written (TF-054); TF-111's driver (etl/embedding_ingestion.py) runs this stage per profile (TF-053's assignment), resumably, with per-row SHA-256 checksums verified on read-back and a sidecar checkpoint"},
    {"id": "subset", "order": 4,
     "modules": ("etl/subset.py", "etl/subset_craft.py"),
     "consumes": "a populated store + a hardware profile id",
     "produces": "a smaller store file for that profile",
     "position_in_build_all": "not a step — a standalone manual build, no caller in the orchestrator",
     "contract": "top-N entities by popularity (qid tie-break) with their statements and a proportional share of reference chunks; TF-110's craft builds fresh, measures the file's bytes, and trims the limit until the measured size fits the profile's byte budget (or fails naming the measured size); success and per-stage notes ride the SubsetResult"},
    {"id": "store", "order": 5,
     "modules": ("etl/schema.py", "knowledge/forge.py"),
     "consumes": "everything the earlier stages wrote",
     "produces": "the sovereign knowledge database, read at turn time",
     "position_in_build_all": "the target of every step (one file, the knowledge DB)",
     "contract": "one SQLite file in WAL mode; row counts readable via get_entity_count; retrieval is TF-041's map, not this one"},
)

# ── The sources stage: five builders ────────────────────────────────────────
#
# "write_mode" is the INSERT flavour of the builder's flush;
# "synchronous" is the posture its build runs under (the schema
# connection default is NORMAL; embeddings harden to FULL).

SOURCE_BUILDERS: tuple[dict[str, Any], ...] = (
    {"id": "wikidata", "class": "WikidataETL", "module": "etl/wikidata.py",
     "input": "Wikidata JSON dump (.json or .json.gz), streamed via ijson",
     "tables": ("entities", "statements", "labels"),
     "write_mode": "entities/labels merged per filed identity (TF-109: duplicates not rewritten, complementary fields union, conflicts ledgered with the incoming row winning); statements INSERT skipping rows already stored (TF-107 idempotence, skips ledgered by TF-109)",
     "batch_default": 2000, "stats_keys": ("entities", "statements", "labels", "skipped"),
     "logs_ingestion": True, "synchronous": "FULL (TF-107; was OFF during the load)",
     "notes": "TF-107 hardened: checkpointed commit + atomic sidecar checkpoint every 1000 stream items (etl/wikidata_pipeline.py), resumable by dump+config fingerprint, optional max_items_per_second rate cap; checks host memory every 10000 entities and commits early above 85% (psutil optional); excludes 6 instance_of types; deprecated-rank claims skipped; TF-108: every entity is judged by the schema gate (etl/schema_gate.py) before any field is read — malformed entities are refused with reasons in the builder's schema_gate ledger and counted in skipped, never a crash or a coerced row; TF-109: entity/label writes merge per etl/dedup_merge.py and collisions ride the builder's merge_ledger"},
    {"id": "conceptnet", "class": "ConceptNetETL", "module": "etl/sources.py",
     "input": "ConceptNet assertions CSV (.csv or .csv.gz), tab-delimited, 5+ columns",
     "tables": ("conceptnet_edges",),
     "write_mode": "semantic dedup (TF-109: one row per (concept1, relation, concept2, language); a collision resolves by weight, the stronger assertion stands; the uri remains UNIQUE provenance)",
     "batch_default": 5000, "stats_keys": ("inserted", "skipped"),
     "logs_ingestion": False, "synchronous": "OFF",
     "notes": "language filter defaults to {en} and must match BOTH concepts (a filter over valid records, not a refusal); weight read from the metadata JSON, defaulting to 1.0 when the metadata carries none; build_from_reference also loads flat {concept1, relation, concept2} JSON files from the reference directory into the same table; TF-108: every line is judged by the schema gate (etl/schema_gate.py) before insertion — short lines, unparseable metadata, non-numeric weights and empty concepts/relations are refused with reasons in the builder's schema_gate ledger and counted in skipped; build_from_reference items gate the same way; TF-109: edge writes dedup by semantic key in etl/dedup_merge.py (assertions and reference edges are two provenances for the same fact) and collisions ride the builder's merge_ledger"},
    {"id": "geonames", "class": "GeoNamesETL", "module": "etl/sources.py",
     "input": "GeoNames tab-delimited text, 19 columns per row",
     "tables": ("geonames_places",),
     "write_mode": "merged per filed identity (TF-109: duplicates not rewritten, complementary fields union, conflicts ledgered with the incoming row winning)",
     "batch_default": 5000, "stats_keys": ("inserted", "skipped"),
     "logs_ingestion": False, "synchronous": "OFF",
     "notes": "rows below min_population and rows outside feature_classes are filter-skipped over valid records; TF-108: the schema gate (etl/schema_gate.py) judges the raw fields first — rows under 19 columns, unparseable numbers, out-of-range coordinates and empty names are refused with reasons in the builder's schema_gate ledger; every non-landed row is counted skipped, never written; TF-109: place writes merge per etl/dedup_merge.py and collisions ride the builder's merge_ledger"},
    {"id": "dbpedia", "class": "DBpediaETL", "module": "etl/sources.py",
     "input": "newline-delimited JSON, one {uri, label, abstract, type} object per line",
     "tables": ("dbpedia_entities",),
     "write_mode": "merged per filed identity (TF-109: duplicates not rewritten, complementary fields union, conflicts ledgered with the incoming row winning)",
     "batch_default": 2000, "stats_keys": ("inserted", "skipped"),
     "logs_ingestion": False, "synchronous": "OFF",
     "notes": "label truncated to 512 chars, abstract to 2000, types to 512; TF-108: every line is judged by the schema gate (etl/schema_gate.py) — unparseable JSON, non-object values, wrong-typed fields and a missing uri are refused with reasons in the builder's schema_gate ledger and counted in skipped (a blank line is not a record and is not counted); NOT wired into the orchestrator's all command (D-ETL-1); TF-109: entity writes merge per etl/dedup_merge.py and collisions ride the builder's merge_ledger"},
    {"id": "reference", "class": "ReferenceDataETL", "module": "etl/sources.py",
     "input": "the built-in reference directory: .md / .json / .jsonl / .yaml / .yml / .txt files, found recursively",
     "tables": ("reference_chunks",),
     "write_mode": "merged per filed identity on (source_file, chunk_index) (TF-109: duplicates not rewritten, complementary fields union, conflicts ledgered with the incoming row winning)",
     "batch_default": 1000, "stats_keys": ("files", "chunks", "skipped"),
     "logs_ingestion": False, "synchronous": "the schema default (NORMAL)",
     "notes": "chunk target 1000 chars with 100 overlap; markdown chunks by heading first; unsupported suffixes are passed over UNCOUNTED (only empty-chunk files and read errors count as skipped); build ends by rebuilding the FTS indexes (TF-058, D-REF-1); TF-109: chunk writes merge per etl/dedup_merge.py and collisions ride the builder's merge_ledger"},
)

SCHEMA_FACTS: dict[str, Any] = {
    "tables": ("entities", "statements", "labels", "embeddings",
               "conceptnet_edges", "geonames_places", "dbpedia_entities",
               "reference_chunks", "reference_embeddings", "ingestion_log"),
    "fts_tables": ("entities_fts", "reference_fts", "conceptnet_fts"),
    "named_indexes": 15,
    "counted_by_get_entity_count": ("entities", "statements", "labels",
                                    "embeddings", "conceptnet_edges",
                                    "geonames_places", "dbpedia_entities",
                                    "reference_chunks"),
    "idempotent": True,
    "journal_mode": "WAL",
}

EMBEDDING_FACTS: dict[str, Any] = {
    "default_model": "all-MiniLM-L6-v2",
    "dimension": 384,
    "entity_batch_default": 128,
    "reference_batch_default": 64,
    "entity_text": "label_en: description_en (label alone when no description)",
    "reference_text": "title: content[:500] (content[:500] alone when no title)",
    "only_missing": True,
    "only_missing_scope": "per model since TF-111 (D-EMB-3): both builders join on model_name as well as the row id, so a row stored under one model no longer blocks building it under another",
    "ingestion_driver": "TF-111 (etl/embedding_ingestion.py): model + batch sizes from TF-053's per-profile assignment (no profile = the builders' defaults above), per-row SHA-256 checksums read back after each committed batch, a sidecar checkpoint (<db>.embedding-checkpoint.json) written only after verified commits, resume by fingerprint, a completed checkpoint returns its filed result, and completion requires TF-053's stored_embeddings_problems to be empty for the run's model",
    "search": "brute-force cosine over the stored BLOBs; entity search top_k 10 / min 0.3, reference search top_k 5 / min 0.3",
    "model_missing_behaviour": "RuntimeError naming the pip install (sentence-transformers absent on this VM, D-ENV-1)",
}

SUBSET_FACTS: dict[str, Any] = {
    "limits": {"pi_zero": 50000, "phone_low": 80000, "pi_5": 200000,
               "desktop_cpu": 1000000, "desktop_gpu": 2000000,
               "server_gpu": 0},
    "limit_override": "a profile JSON's deployment/memory subset_entity_limit wins when present; no shipped profile carries the key, so the table above is what runs today",
    "copies_rows": ("entities", "statements", "reference_chunks"),
    "does_not_copy_rows": ("labels", "embeddings", "conceptnet_edges",
                           "geonames_places", "dbpedia_entities",
                           "reference_embeddings", "ingestion_log"),
    "reference_chunk_cap": "entity_limit // 10 when limited (TF-110 removed the 1000-row floor, D-SUBSET-2); all chunks when unlimited; order is chunk_id after D-SUBSET-1",
    "schema_clone": "table DDL only (FTS virtuals included); the 15 named indexes are not recreated",
    "failure_shape": "a failed copy returns SubsetResult(success=False) with the error in notes; a skipped stage rides notes with success still True",
    "byte_budget": "TF-110 (etl/subset_craft.py): pi_zero's budget is its declared deployment.knowledge_subset_max_gb; every other profile's is derived at 0.25 x declared RAM (campaign rule, budget_source names the road); the craft loop rebuilds fresh and trims by measured overshoot until the file's measured bytes fit, or reports success=False",
    "fresh_builds": "TF-110 (D-SUBSET-3): every subset build starts from no file and refuses the source itself as its output; selection ties break by qid (D-SUBSET-4)",
}

ORCHESTRATION: dict[str, Any] = {
    "in_package_orchestrator": False,
    "runner": "forge_memory.py (root instrument), command `all`",
    "steps": ("1 schema (always)", "2 reference (always)",
              "3 wikidata (only with --wikidata-dump)",
              "4 conceptnet (only with --conceptnet-csv)",
              "5 geonames (only with --geonames-txt)",
              "6 embeddings + FTS rebuild (unless --skip-embeddings)"),
    "gaps": ("--dbpedia-json is accepted by `all` and never ingested (D-ETL-1, OPEN, owner TF-119)",
             "the subset stage is not a step of `all` at all — it runs only when invoked directly",
             "per-source commands exist beside `all` for wikidata / conceptnet / geonames / dbpedia / reference / embeddings"),
    "dry_run": "TF-112 (etl/ingestion_dry_run.py): a single-source scratch runner beside `all`, not a step of it — one real builder test-ingested into a scratch store (never the target, proven untouched by hash + counts), the TF-106 registry consulted first, and a full report (stats, store counts, TF-108 gate ledger, TF-109 merge ledger + overlaps, TF-111 embeddings when an encoder is injected); `all`'s filed shape above stands unchanged",
}

ETL_RULES: tuple[dict[str, Any], ...] = (
    {"id": "stages_are_five_in_roadmap_order",
     "rule": "the pipeline is sources -> schema -> embeddings -> subset -> store, and every stage's contract is filed in this map"},
    {"id": "every_builder_initialises_the_schema_itself",
     "rule": "no builder assumes the store exists: each build() calls initialize_schema first"},
    {"id": "missing_input_raises_by_name",
     "rule": "every builder raises FileNotFoundError on a missing input file/directory before writing anything"},
    {"id": "only_wikidata_writes_the_ingestion_log",
     "rule": "of the five builders only WikidataETL records its run in ingestion_log (start + finish, complete or failed)"},
    {"id": "embeddings_are_built_only_for_rows_missing_one",
     "rule": "both embedding builders select with a LEFT JOIN ... IS NULL, so a re-run builds nothing twice — since TF-111 (D-EMB-3) the join is per model, so 'missing' means missing under the model being built"},
    {"id": "the_subset_copies_three_tables_and_no_more",
     "rule": "a subset carries entities, their statements, and reference chunks; the other seven tables are schema-cloned and row-empty"},
    {"id": "a_limited_subset_keeps_its_reference_chunks",
     "rule": "D-SUBSET-1 fixed: a limited (edge-profile) subset copies its capped share of reference chunks instead of silently copying none"},
    {"id": "there_is_no_orchestrator_in_the_package",
     "rule": "src/thoughtforge/etl exposes builders, never a pipeline runner; the order lives in forge_memory.py's all command"},
    {"id": "build_all_never_ingests_dbpedia",
     "rule": "D-ETL-1 (OPEN, TF-119): the all command accepts --dbpedia-json and has no DBpedia step; this rule pins the gap until the step is wired"},
)


# ── Synthetic fixtures (TF-017: synthetic stand-ins only) ───────────────────
#
# One tiny input per builder, shaped like its real source and small
# enough to run in milliseconds. The live checker and the TF-105
# tests share them through run_synthetic_pipeline().

_WIKIDATA_ENTITY: dict[str, Any] = {
    "id": "Q105001",
    "labels": {"en": {"value": "Lantern Hollow"}},
    "descriptions": {"en": {"value": "A synthetic hollow lit by lanterns"}},
    "aliases": {"en": [{"value": "the lit hollow"}]},
    "sitelinks": {"enwiki": {}, "dewiki": {}},
    "claims": {
        "P31": [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": "Q105900"}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        }],
    },
}

_CONCEPTNET_LINES = (
    "/a/[/r/IsA/,/c/en/lantern/,/c/en/light/]\t/r/IsA\t/c/en/lantern"
    "\t/c/en/light\t{\"weight\": 2.0}",
    "/a/[/r/IsA/,/c/fr/lanterne/,/c/fr/lumiere/]\t/r/IsA\t/c/fr/lanterne"
    "\t/c/fr/lumiere\t{\"weight\": 1.0}",
)

_GEONAMES_LINE = "\t".join((
    "105001", "Lantern Town", "Lantern Town", "", "1.5", "2.5", "P",
    "PPL", "XX", "", "A1", "A2", "", "", "100", "10", "10", "UTC",
    "2026-01-01",
))

_DBPEDIA_LINES = (
    json.dumps({"uri": "http://example.org/Lantern", "label": "Lantern",
                "abstract": "A portable synthetic light.",
                "type": "Device"}),
    json.dumps({"label": "No uri on this line"}),
)

_REFERENCE_MD = "# Lantern Care\n\nKeep the lantern trimmed and filled.\n"


class _FakeEmbeddingModel:
    """Deterministic stand-in for a sentence-transformer.

    Registered in the embeddings module's cache under the
    canonical model name by the live checker, so the REAL
    builders run (selection, validation, writes) with no
    model download and no sentence-transformers install.
    """

    def encode(self, texts: Any, batch_size: int = 128,
               show_progress_bar: bool = False,
               normalize_embeddings: bool = True,
               convert_to_numpy: bool = True) -> np.ndarray:
        rows = []
        for text in texts:
            seed = sum(str(text).encode("utf-8")) % (2 ** 32)
            vec = np.random.default_rng(seed).random(
                EMBEDDING_FACTS["dimension"], dtype=np.float64,
            ).astype(np.float32)
            norm = float(np.linalg.norm(vec)) or 1.0
            rows.append(vec / norm)
        return np.stack(rows).astype(np.float32)


def run_synthetic_pipeline(root: Path) -> dict[str, Any]:
    """Run every stage on synthetic inputs under ``root``.

    Returns the measured counts (builder stats, store counts,
    embedding builds, subset result) the live checker and the
    tests judge. The fake model is installed into the real
    embeddings cache for the embedding legs and removed after.
    """
    from thoughtforge.etl import embeddings as embeddings_module
    from thoughtforge.etl.embeddings import (
        build_entity_embeddings,
        build_reference_embeddings,
    )
    from thoughtforge.etl.schema import get_entity_count, initialize_schema
    from thoughtforge.etl.sources import (
        ConceptNetETL,
        DBpediaETL,
        GeoNamesETL,
        ReferenceDataETL,
    )
    from thoughtforge.etl.subset import EdgeSubsetBuilder
    from thoughtforge.etl.wikidata import WikidataETL

    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "synthetic.db"
    initialize_schema(db_path)

    dump = root / "dump.json"
    dump.write_text(json.dumps([_WIKIDATA_ENTITY]), encoding="utf-8")
    wikidata_stats = WikidataETL(db_path).build(dump_path=dump)

    csv_path = root / "assertions.csv"
    csv_path.write_text("\n".join(_CONCEPTNET_LINES) + "\n", encoding="utf-8")
    conceptnet_stats = ConceptNetETL(db_path).build(csv_path=csv_path)

    txt_path = root / "places.txt"
    txt_path.write_text(_GEONAMES_LINE + "\n", encoding="utf-8")
    geonames_stats = GeoNamesETL(db_path).build(txt_path=txt_path)

    json_path = root / "abstracts.json"
    json_path.write_text("\n".join(_DBPEDIA_LINES) + "\n", encoding="utf-8")
    dbpedia_stats = DBpediaETL(db_path).build(json_path=json_path)

    ref_dir = root / "ref"
    ref_dir.mkdir()
    (ref_dir / "lantern_care.md").write_text(_REFERENCE_MD, encoding="utf-8")
    (ref_dir / "lantern_photo.jpg").write_bytes(b"\xff\xd8\xff\xe0synthetic")
    reference_stats = ReferenceDataETL(db_path).build(ref_dir=ref_dir)

    canonical = embeddings_module.canonical_model_name(
        embeddings_module.DEFAULT_MODEL)
    embeddings_module._MODEL_CACHE[canonical] = _FakeEmbeddingModel()
    try:
        entity_embeddings = build_entity_embeddings(db_path=db_path)
        reference_embeddings = build_reference_embeddings(db_path=db_path)
        entity_embeddings_rerun = build_entity_embeddings(db_path=db_path)
    finally:
        embeddings_module._MODEL_CACHE.pop(canonical, None)

    subset_path = root / "subset.db"
    subset = EdgeSubsetBuilder().build(
        source_db=db_path, output_path=subset_path, profile_id="pi_zero")

    conn = sqlite3.connect(str(subset_path))
    try:
        subset_counts = {
            table: conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("entities", "statements", "reference_chunks",
                          "embeddings", "labels")
        }
        subset_indexes = sorted(
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name LIKE 'idx_%'"))
    finally:
        conn.close()

    conn = sqlite3.connect(str(db_path))
    try:
        log_rows = conn.execute(
            "SELECT source, status FROM ingestion_log").fetchall()
    finally:
        conn.close()

    return {
        "wikidata": wikidata_stats,
        "conceptnet": conceptnet_stats,
        "geonames": geonames_stats,
        "dbpedia": dbpedia_stats,
        "reference": reference_stats,
        "entity_embeddings": entity_embeddings,
        "reference_embeddings": reference_embeddings,
        "entity_embeddings_rerun": entity_embeddings_rerun,
        "store_counts": get_entity_count(db_path),
        "ingestion_log": [tuple(row) for row in log_rows],
        "subset": {
            "entities": subset.entities_copied,
            "statements": subset.statements_copied,
            "reference_chunks": subset.reference_chunks_copied,
            "success": subset.success,
            "notes": list(subset.notes),
        },
        "subset_counts": subset_counts,
        "subset_named_indexes": subset_indexes,
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def etl_map_problems() -> list[str]:
    """Static truth: the filed map against the live tree."""
    from thoughtforge.etl import embeddings as embeddings_module
    from thoughtforge.etl import schema as schema_module
    from thoughtforge.etl import sources as sources_module
    from thoughtforge.etl import subset as subset_module
    from thoughtforge.etl import wikidata as wikidata_module

    problems: list[str] = []

    stage_ids = [s["id"] for s in PIPELINE_STAGES]
    if stage_ids != ["sources", "schema", "embeddings", "subset", "store"]:
        problems.append(f"PIPELINE_STAGES order drifted: {stage_ids}")
    if [s["order"] for s in PIPELINE_STAGES] != [1, 2, 3, 4, 5]:
        problems.append("PIPELINE_STAGES order fields are not 1..5")
    rule_ids = [r["id"] for r in ETL_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate ETL_RULES ids")

    modules = {"etl/wikidata.py": wikidata_module,
               "etl/sources.py": sources_module}
    for builder in SOURCE_BUILDERS:
        cls = getattr(modules[builder["module"]], builder["class"], None)
        if cls is None:
            problems.append(f"{builder['class']} missing from "
                            f"{builder['module']}")
            continue
        params = inspect.signature(cls.build).parameters
        if "batch_size" in params:
            if params["batch_size"].default != builder["batch_default"]:
                problems.append(
                    f"{builder['class']}.build batch default drifted: "
                    f"{params['batch_size'].default} != "
                    f"{builder['batch_default']}")
        elif builder["id"] != "reference":
            problems.append(f"{builder['class']}.build lost its batch_size "
                            "parameter")
        # The reference builder's filed "batch" is its chunk target,
        # checked against the module constants just below.
        build_src = inspect.getsource(cls.build)
        if "initialize_schema" not in build_src:
            problems.append(f"{builder['class']}.build no longer "
                            "initialises the schema itself")
        if "FileNotFoundError" not in build_src:
            problems.append(f"{builder['class']}.build no longer raises "
                            "FileNotFoundError on a missing input")
    ref_params = inspect.signature(
        sources_module.ReferenceDataETL.build).parameters
    if (ref_params["chunk_size"].default != sources_module.CHUNK_SIZE_CHARS
            or ref_params["overlap"].default
            != sources_module.CHUNK_OVERLAP_CHARS):
        problems.append("reference chunk defaults drifted from the "
                        "sources module constants")
    if (sources_module.CHUNK_SIZE_CHARS, sources_module.CHUNK_OVERLAP_CHARS) \
            != (1000, 100):
        problems.append("reference chunking constants drifted from "
                        "1000/100")

    tables_in_ddl = set(re.findall(
        r"CREATE TABLE IF NOT EXISTS (\w+)", schema_module._TABLES))
    if tables_in_ddl != set(SCHEMA_FACTS["tables"]):
        problems.append(f"schema tables drifted: {sorted(tables_in_ddl)}")
    fts_in_ddl = set(re.findall(
        r"CREATE VIRTUAL TABLE IF NOT EXISTS (\w+)", schema_module._FTS5))
    if fts_in_ddl != set(SCHEMA_FACTS["fts_tables"]):
        problems.append(f"schema FTS tables drifted: {sorted(fts_in_ddl)}")
    if schema_module._INDEXES.count("CREATE INDEX") \
            != SCHEMA_FACTS["named_indexes"]:
        problems.append("schema named-index count drifted from 15")
    counted_src = inspect.getsource(schema_module.get_entity_count)
    for table in SCHEMA_FACTS["counted_by_get_entity_count"]:
        if f'"{table}"' not in counted_src:
            problems.append(f"get_entity_count no longer counts {table}")

    if embeddings_module.DEFAULT_MODEL != EMBEDDING_FACTS["default_model"]:
        problems.append("embeddings DEFAULT_MODEL drifted")
    if embeddings_module.EMBEDDING_DIM != EMBEDDING_FACTS["dimension"]:
        problems.append("embeddings EMBEDDING_DIM drifted from 384")
    for fn_name, filed in (("build_entity_embeddings", 128),
                           ("build_reference_embeddings", 64)):
        fn = getattr(embeddings_module, fn_name)
        if inspect.signature(fn).parameters["batch_size"].default != filed:
            problems.append(f"{fn_name} batch default drifted from {filed}")
        if "IS NULL" not in inspect.getsource(fn):
            problems.append(f"{fn_name} lost its only-missing selection")

    if subset_module._PROFILE_ENTITY_LIMITS != SUBSET_FACTS["limits"]:
        problems.append("subset profile limits drifted from the filed table")
    subset_src = Path(subset_module.__file__).read_text(encoding="utf-8")
    if "ORDER BY retrieval_score DESC" in subset_src:
        problems.append("D-SUBSET-1 reopened: subset.py orders by the "
                        "nonexistent retrieval_score column again")
    if "ORDER BY chunk_id" not in subset_src:
        problems.append("subset reference copy lost its chunk_id order "
                        "(D-SUBSET-1 fix)")

    forge_memory = (_REPO_ROOT / "forge_memory.py").read_text(
        encoding="utf-8")
    build_all_src = forge_memory.split("def build_all", 1)[1].split(
        "\ndef ", 1)[0]
    step_marks = ["initialize_schema", "ReferenceDataETL", "WikidataETL",
                  "ConceptNetETL", "GeoNamesETL", "build_entity_embeddings"]
    positions = [build_all_src.find(mark) for mark in step_marks]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        problems.append("forge_memory build_all step order drifted from "
                        "ORCHESTRATION")
    if "--dbpedia-json" not in forge_memory:
        problems.append("the --dbpedia-json option vanished; D-ETL-1's "
                        "filed shape changed")
    if "DBpediaETL" in build_all_src:
        problems.append("D-ETL-1 changed shape: build_all now wires "
                        "DBpediaETL — update ORCHESTRATION and this rule "
                        "in the open")

    wikidata_src = Path(wikidata_module.__file__).read_text(encoding="utf-8")
    if "_start_log" not in wikidata_src or "ingestion_log" not in wikidata_src:
        problems.append("wikidata lost its ingestion_log wiring")
    sources_src = Path(sources_module.__file__).read_text(encoding="utf-8")
    if "ingestion_log" in sources_src:
        problems.append("a sources.py builder started writing the "
                        "ingestion_log; the only-wikidata rule changed")
    return problems


def etl_map_live_problems() -> list[str]:
    """Behavioural truth: a whole synthetic pipeline, run live."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = run_synthetic_pipeline(Path(tmp))
        except Exception as exc:  # a crashed stage is a finding
            return [f"synthetic pipeline raised: {exc!r}"]

    expected_stats = {
        "wikidata": {"entities": 1, "statements": 1, "labels": 1,
                     "skipped": 0},
        "conceptnet": {"inserted": 1, "skipped": 1},
        "geonames": {"inserted": 1, "skipped": 0},
        "dbpedia": {"inserted": 1, "skipped": 1},
        "reference": {"files": 1, "chunks": 1, "skipped": 0},
    }
    for stage, expected in expected_stats.items():
        if result[stage] != expected:
            problems.append(f"{stage} stats {result[stage]} != {expected}")
    if result["entity_embeddings"] != 1:
        problems.append("entity embedding build != 1")
    if result["reference_embeddings"] != 1:
        problems.append("reference embedding build != 1")
    if result["entity_embeddings_rerun"] != 0:
        problems.append("embedding re-run built rows it already had "
                        "(only-missing rule broken)")
    counts = result["store_counts"]
    for table in ("entities", "statements", "labels", "embeddings",
                  "conceptnet_edges", "geonames_places",
                  "dbpedia_entities", "reference_chunks"):
        if counts.get(table) != 1:
            problems.append(f"store count {table} = {counts.get(table)} "
                            "!= 1 after the pipeline")
    if result["ingestion_log"] != [("wikidata", "complete")]:
        problems.append(f"ingestion_log {result['ingestion_log']} != "
                        "the single wikidata completion row")
    subset = result["subset"]
    if subset != {"entities": 1, "statements": 1, "reference_chunks": 1,
                  "success": True,
                  "notes": subset["notes"]} or any(
                      "reference_chunks skipped" in note
                      for note in subset["notes"]):
        problems.append(f"limited subset result drifted: {subset}")
    if not any(note.startswith("FTS rebuilt") for note in subset["notes"]):
        problems.append("subset did not rebuild its FTS indexes")
    if result["subset_counts"]["embeddings"] != 0:
        problems.append("subset carried embedding rows it must not copy")
    if result["subset_counts"]["labels"] != 0:
        problems.append("subset carried label rows it must not copy")
    if result["subset_named_indexes"] != []:
        problems.append("subset recreated named indexes the filed map "
                        f"says it does not: {result['subset_named_indexes']}")
    return problems


def etl_as_data() -> dict[str, Any]:
    """The whole map as plain data, for consumers and the doctor."""
    return {
        "format": ETL_FORMAT,
        "doc": ETL_DOC,
        "stages": [dict(s) for s in PIPELINE_STAGES],
        "builders": [dict(b) for b in SOURCE_BUILDERS],
        "schema": dict(SCHEMA_FACTS),
        "embeddings": dict(EMBEDDING_FACTS),
        "subset": dict(SUBSET_FACTS),
        "orchestration": dict(ORCHESTRATION),
        "rules": [dict(r) for r in ETL_RULES],
    }
