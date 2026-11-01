"""
ETL documentation index + new-source recipe, as data
(TF-119).

The roadmap's rule: *ETL docs + a new-source recipe.*

The measured starting state, probed live this tick on
this tree before anything was built:

  * **D-ETL-1 stood open exactly as TF-105 filed it.**
    ``forge_memory.py``'s ``all`` command accepted
    ``--dbpedia-json`` and never ingested it: its source
    (read by this slice before the fix) ran schema,
    reference, wikidata, conceptnet and geonames steps
    numbered /6, then embeddings — ``DBpediaETL`` appeared
    nowhere in ``build_all``. A user passing a DBpedia
    file was told nothing and got no rows.
  * **D-DATA-1 stood open exactly as TF-007 filed it.**
    ``etl/sources.py`` line 8 and the ``ReferenceDataETL``
    docstring both said "40 built-in files"; TF-007's
    measured census is 215 root files + 17 SRD chapters.
  * **No ETL documentation index and no new-source
    recipe existed (D-ETL-2's recipe share).** Part VI
    produced fourteen architecture docs (TF-105..TF-118)
    filed nowhere as a set, and a new source's filings —
    registry, schema gate, dedup identity, update
    behaviour, poison screening, builder contract, CLI
    wiring, dry-run proof, metrics reporting, offline
    boundary, its own doc — lived only as the separate
    habits of fourteen slices. Nothing ordered them, and
    nothing required a source to be filed in TF-106's
    registry *before* it was documented or built.

What this slice builds, filed as data here:

  * **D-ETL-1 FIXED, in ``forge_memory.py`` and in
    TF-105's map in the open.** ``build_all`` runs seven
    steps: 1 schema, 2 reference, 3 wikidata,
    4 conceptnet, 5 geonames, 6 dbpedia (only when
    ``--dbpedia-json`` is passed, else a skipped
    message), 7 embeddings + FTS unless
    ``--skip-embeddings``. The click option and the
    per-source ``dbpedia`` command are unchanged.
    TF-105's map (``etl/etl_map.py`` +
    ``docs/architecture/ETL_MAPPED.md`` §6) was updated
    in the open and its static checker now REQUIRES
    DBpediaETL in ``build_all`` in the filed order.
  * **D-DATA-1 FIXED, in ``etl/sources.py``.** Both
    stale counts now state TF-007's measured truth:
    the built-in knowledge reference files — 215 root
    files + 17 SRD chapters. No bare "40 files" claim
    remains in that module (the static checker below
    pins its absence).
  * ``DOC_INDEX`` — every Part VI architecture doc,
    TF-105..TF-119, each with its slice id, its file
    path and a one-line subject.
  * ``RECIPE_STEPS`` — the ordered new-source recipe:
    twelve steps, each bound to the real module /
    attribute it files into, checked live by
    ``etl_docs_problems()`` (import + getattr, plus the
    root instrument's file where a step binds there).
    Step 1 files the source in TF-106's registry
    (``check_source_use`` gate) BEFORE documenting or
    building it — that discharges D-ETL-2's TF-119
    share.
  * ``recipe_application()`` + the worked synthetic
    example — a tiny fictional source, the "lantern
    gazetteer", judged against the live registries:
    it is filed nowhere, so the application names its
    missing filings (registry first). The example adds
    NO real builder and NO real registry entry; it
    proves the recipe's checklist catches an unfiled
    source.

D-ETL-2 disposition, stated plainly: TF-119's share
is FIXED — the recipe files a source in the registry
before documenting it, and TF-112 already discharged
its dry-run share. The builders themselves still
consult nothing at build time; that standing shape
(TF-106's filed limit, pinned by TF-106's checker) is
NOT claimed fixed here.

Filed limits, stated plainly:

  * No new CLI commands are added by this slice for
    the TF-115 / TF-116 / TF-117 / TF-118 drivers (nor
    for this module): the recipe is the wiring contract
    a future source follows; it is not itself a CLI.
  * The doc index files the Part VI architecture docs
    only (TF-105..TF-119). Earlier docs are other
    parts' filings.
  * The example is a checklist judgement over the live
    registries as data — it builds nothing, writes
    nothing, and files no real entry anywhere.

Docket (TF-016): D-ETL-1 and D-DATA-1 are FIXED by
this slice; D-ETL-2 is FIXED with both shares now
discharged (TF-112 + TF-119), its builders-consult-
nothing shape standing as filed.

The human-readable write-up is
``docs/architecture/ETL_DOCS_AND_NEW_SOURCE_RECIPE.md``.
Proof grade: FIXTURE — synthetic inputs through the
real builders / CLI on this VM; no real dump was
ingested at scale, never claimed.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

ETL_DOCS_FORMAT = "thoughtforge-etl-docs/1"
ETL_DOCS_DOC = "docs/architecture/ETL_DOCS_AND_NEW_SOURCE_RECIPE.md"

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ── The documentation index ─────────────────────────────────────────────────
#
# Every Part VI architecture doc, in slice order. "file"
# is the repo-relative path the static checker verifies
# exists on disk.

DOC_INDEX: tuple[dict[str, Any], ...] = (
    {"slice": "TF-105", "file": "docs/architecture/ETL_MAPPED.md",
     "subject": "ETL mapped: the five stages, five builders, schema / embedding / subset facts and the orchestration, as data"},
    {"slice": "TF-106", "file": "docs/architecture/SOURCE_REGISTRY.md",
     "subject": "Source registry: every source carries provenance + licence; unlicensed-in-fact sources quarantined by rule"},
    {"slice": "TF-107", "file": "docs/architecture/WIKIDATA_PIPELINE.md",
     "subject": "Wikidata pipeline hardened: rate-respecting, resumable, checkpointed"},
    {"slice": "TF-108", "file": "docs/architecture/SCHEMA_VALIDATION_GATE.md",
     "subject": "Schema validation at the gate: malformed records refused with reasons, counted, never silently dropped"},
    {"slice": "TF-109", "file": "docs/architecture/DEDUP_AND_MERGE.md",
     "subject": "Dedup & merge rules for overlapping sources, per filed table identity"},
    {"slice": "TF-110", "file": "docs/architecture/SUBSET_CRAFT.md",
     "subject": "Subset craft: profile-sized subsets built by rule and measured against a byte budget"},
    {"slice": "TF-111", "file": "docs/architecture/EMBEDDING_AT_INGESTION.md",
     "subject": "Embedding at ingestion: model per profile, resumable, checksummed"},
    {"slice": "TF-112", "file": "docs/architecture/INGESTION_DRY_RUN.md",
     "subject": "Ingestion dry-run: a source test-ingested into a scratch store with a full report"},
    {"slice": "TF-113", "file": "docs/architecture/CANON_INGESTION.md",
     "subject": "Canon ingestion path: the owner's own materials ingested into private stores only"},
    {"slice": "TF-114", "file": "docs/architecture/UPDATE_INGESTION.md",
     "subject": "Update ingestion: re-running a source updates facts by the lifecycle rules"},
    {"slice": "TF-115", "file": "docs/architecture/INGESTION_METRICS.md",
     "subject": "Ingestion metrics: records/sec, refusal rate and store growth reported per run"},
    {"slice": "TF-116", "file": "docs/architecture/POISONED_SOURCE_FIXTURES.md",
     "subject": "Poisoned-source fixtures: adversarial source data refused / quarantined at landing and at retrieval"},
    {"slice": "TF-117", "file": "docs/architecture/OFFLINE_INGESTION.md",
     "subject": "Offline ingestion: everything except the fetch itself works with the network cut, proven"},
    {"slice": "TF-118", "file": "docs/architecture/REFERENCE_LIBRARY_AS_SOURCE.md",
     "subject": "The reference library as a source: its ingestion reproducible + checksummed"},
    {"slice": "TF-119", "file": "docs/architecture/ETL_DOCS_AND_NEW_SOURCE_RECIPE.md",
     "subject": "ETL docs + a new-source recipe: this index, the ordered recipe, D-ETL-1 and D-DATA-1 fixed"},
)


# ── The new-source recipe ───────────────────────────────────────────────────
#
# The ordered recipe. Each step binds the real thing it
# files: "bind_module" + "bind_attribute" are imported
# and getattr-checked by etl_docs_problems(); a step
# that also binds a file (the root instrument) carries
# "bind_file" + "bind_file_markers" checked in its text.

RECIPE_STEPS: tuple[dict[str, Any], ...] = (
    {"id": "file_in_registry_first", "order": 1,
     "step": "file the source in TF-106's registry (etl/source_registry.py SOURCES, judged by check_source_use) with provenance + a verified licence BEFORE documenting or building it — an unfiled source is quarantined by default (D-ETL-2, TF-119 share)",
     "bind_module": "thoughtforge.etl.source_registry",
     "bind_attribute": "check_source_use",
     "bind_file": "src/thoughtforge/etl/source_registry.py",
     "bind_file_markers": ("SOURCES", "check_source_use")},
    {"id": "file_record_schema_in_gate", "order": 2,
     "step": "file the record schema in TF-108's schema gate (etl/schema_gate.py RECORD_SCHEMAS): the fields a record must carry, their kinds and their checks, so malformed records are refused with reasons",
     "bind_module": "thoughtforge.etl.schema_gate",
     "bind_attribute": "RECORD_SCHEMAS",
     "bind_file": "src/thoughtforge/etl/schema_gate.py",
     "bind_file_markers": ("RECORD_SCHEMAS",)},
    {"id": "name_target_table_and_identity", "order": 3,
     "step": "name the target table and file its identity in TF-109's dedup/merge filing (etl/dedup_merge.py TABLE_IDENTITIES): the key that makes a row the same row, and the value fields a merge judges",
     "bind_module": "thoughtforge.etl.dedup_merge",
     "bind_attribute": "TABLE_IDENTITIES",
     "bind_file": "src/thoughtforge/etl/dedup_merge.py",
     "bind_file_markers": ("TABLE_IDENTITIES",)},
    {"id": "state_update_behaviour", "order": 4,
     "step": "state the source's update behaviour per TF-114's lifecycle rules (etl/update_ingestion.py UPDATE_RULES): unchanged = duplicate, changed = supersede with the stored value ledgered, disappeared-from-a-presented-unit = retired, absent unit = kept",
     "bind_module": "thoughtforge.etl.update_ingestion",
     "bind_attribute": "UPDATE_RULES",
     "bind_file": "src/thoughtforge/etl/update_ingestion.py",
     "bind_file_markers": ("UPDATE_RULES",)},
    {"id": "name_landed_text_fields_for_poison", "order": 5,
     "step": "name the landed text fields for TF-116's poison screening (etl/poisoned_source.py SOURCE_TEXT_FIELDS): exactly the text the builder writes to the store, screened after the gate and before insertion",
     "bind_module": "thoughtforge.etl.poisoned_source",
     "bind_attribute": "SOURCE_TEXT_FIELDS",
     "bind_file": "src/thoughtforge/etl/poisoned_source.py",
     "bind_file_markers": ("SOURCE_TEXT_FIELDS",)},
    {"id": "builder_contract_per_tf105", "order": 6,
     "step": "build the builder to TF-105's contract (etl/etl_map.py SOURCE_BUILDERS): it initialises the schema itself, raises FileNotFoundError on a missing input, returns its own stats dict, and holds fresh gate / merge / update / poison ledgers per build; file the builder in TF-105's map in the open",
     "bind_module": "thoughtforge.etl.etl_map",
     "bind_attribute": "SOURCE_BUILDERS",
     "bind_file": "src/thoughtforge/etl/etl_map.py",
     "bind_file_markers": ("SOURCE_BUILDERS",)},
    {"id": "wire_forge_memory_command_and_all_step", "order": 7,
     "step": "wire a forge_memory per-source command for the builder, and an `all` step in build_all if the source belongs in full builds (numbered in the filed order, with a skipped message when its file is not passed — the TF-119 D-ETL-1 shape)",
     "bind_module": "thoughtforge.etl.etl_map",
     "bind_attribute": "ORCHESTRATION",
     "bind_file": "forge_memory.py",
     "bind_file_markers": ("def build_all", "DBpediaETL")},
    {"id": "prove_by_dry_run", "order": 8,
     "step": "prove the builder by TF-112's dry-run (etl/ingestion_dry_run.py dry_run_ingestion) into a scratch store — never the target — with the registry consulted first and the full report filed",
     "bind_module": "thoughtforge.etl.ingestion_dry_run",
     "bind_attribute": "dry_run_ingestion",
     "bind_file": "src/thoughtforge/etl/ingestion_dry_run.py",
     "bind_file_markers": ("dry_run_ingestion",)},
    {"id": "report_by_metrics", "order": 9,
     "step": "report the source by TF-115's metrics (etl/ingestion_metrics.py measure_ingestion): records/sec, the refusal rate, and store growth per source, per run",
     "bind_module": "thoughtforge.etl.ingestion_metrics",
     "bind_attribute": "measure_ingestion",
     "bind_file": "src/thoughtforge/etl/ingestion_metrics.py",
     "bind_file_markers": ("measure_ingestion",)},
    {"id": "file_offline_fetch_boundary", "order": 10,
     "step": "file the source's offline / fetch boundary per TF-117 (etl/offline_ingestion.py FETCH_STEPS + OFFLINE_STEPS): downloading its file is a fetch outside ingestion; everything after the files are here is an offline step",
     "bind_module": "thoughtforge.etl.offline_ingestion",
     "bind_attribute": "FETCH_STEPS",
     "bind_file": "src/thoughtforge/etl/offline_ingestion.py",
     "bind_file_markers": ("FETCH_STEPS", "OFFLINE_STEPS")},
    {"id": "write_doc_and_register_in_index", "order": 11,
     "step": "write the source's architecture doc in the sibling style and register it in this module's DOC_INDEX (etl/etl_docs.py) with its slice id, file path and one-line subject",
     "bind_module": "thoughtforge.etl.etl_docs",
     "bind_attribute": "DOC_INDEX",
     "bind_file": "docs/architecture/ETL_DOCS_AND_NEW_SOURCE_RECIPE.md",
     "bind_file_markers": ("new-source recipe",)},
    {"id": "synthetic_fixtures_only", "order": 12,
     "step": "prove everything with synthetic fixtures only (TF-017): tiny stand-in inputs in the shape of the real source, as TF-105's run_synthetic_pipeline does — canon is never a fixture and never a test input",
     "bind_module": "thoughtforge.etl.etl_map",
     "bind_attribute": "run_synthetic_pipeline",
     "bind_file": "src/thoughtforge/etl/etl_map.py",
     "bind_file_markers": ("run_synthetic_pipeline",)},
)

RECIPE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "registry_before_documentation_or_building",
     "rule": "a new source is filed in TF-106's registry (provenance + verified licence, check_source_use gate) before it is documented or built — documentation never precedes the registry filing (D-ETL-2, TF-119 share)"},
    {"id": "every_step_files_into_a_real_thing",
     "rule": "every recipe step names the module / attribute / file it files into, and etl_docs_problems() re-checks each binding live (import + getattr; the root instrument by its file) — a step bound to nothing is not a step"},
    {"id": "the_recipe_order_is_the_filed_order",
     "rule": "RECIPE_STEPS run in their filed order 1..12: registry, schema, identity, update behaviour, poison fields, builder, CLI wiring, dry-run proof, metrics, offline boundary, doc + index, synthetic fixtures"},
    {"id": "build_all_ingests_dbpedia_when_passed",
     "rule": "D-ETL-1 FIXED: forge_memory build_all ingests DBpedia as step 6 of 7 when --dbpedia-json is passed, and reports the step skipped when it is not — an accepted option is never silently unused again"},
    {"id": "the_reference_count_is_the_measured_census",
     "rule": "D-DATA-1 FIXED: the built-in reference library is described by TF-007's measured census — 215 root files + 17 SRD chapters — never by a bare stale count"},
    {"id": "builders_consulting_nothing_is_a_stated_shape",
     "rule": "the builders themselves still consult no registry at build time (TF-106's filed limit, pinned by TF-106's checker): the recipe governs filing order, and that standing shape is stated, never claimed fixed"},
    {"id": "the_index_files_part_vi_docs_only",
     "rule": "DOC_INDEX files every Part VI architecture doc TF-105..TF-119 with slice id, path and subject, and nothing else — a Part VI doc missing from the index turns the checker red"},
    {"id": "no_new_cli_commands_for_the_drivers",
     "rule": "this slice adds no CLI commands for the TF-115 / TF-116 / TF-117 / TF-118 drivers or for this module — the recipe is the wiring contract a future source follows"},
    {"id": "canon_is_never_a_fixture",
     "rule": "all proof fixtures are synthetic (TF-017): the owner's canon and the real built-in library are never test inputs, and the worked example files no real entry anywhere"},
)


# ── The worked synthetic example (TF-017) ───────────────────────────────────
#
# A tiny fictional source, filed NOWHERE. The application
# below judges it against the live registries and names
# every missing filing — the registry filing first.

EXAMPLE_SOURCE: dict[str, Any] = {
    "id": "lantern_gazetteer",
    "name": "Lantern Gazetteer",
    "target_table": "lantern_places",
    "doc_file": "docs/architecture/LANTERN_GAZETTEER.md",
}


def recipe_application(source_id: str, target_table: str,
                       doc_file: str | None = None) -> dict[str, Any]:
    """Judge one (hypothetical) source against the recipe.

    Checks the live filings the recipe's data steps name —
    the TF-106 registry, TF-108's RECORD_SCHEMAS, TF-109's
    TABLE_IDENTITIES, TF-116's SOURCE_TEXT_FIELDS and this
    index — and returns the application as data: the steps
    present, the steps missing (in recipe order), and one
    problem string per missing filing. Builds nothing,
    writes nothing, files nothing.
    """
    from thoughtforge.etl.dedup_merge import TABLE_IDENTITIES
    from thoughtforge.etl.poisoned_source import SOURCE_TEXT_FIELDS
    from thoughtforge.etl.schema_gate import RECORD_SCHEMAS
    from thoughtforge.etl.source_registry import SOURCES

    checks: dict[str, bool] = {
        "file_in_registry_first": any(
            entry["id"] == source_id for entry in SOURCES),
        "file_record_schema_in_gate": any(
            schema["source"] == source_id for schema in RECORD_SCHEMAS),
        "name_target_table_and_identity": any(
            identity["table"] == target_table
            for identity in TABLE_IDENTITIES),
        "name_landed_text_fields_for_poison": source_id in SOURCE_TEXT_FIELDS,
        "write_doc_and_register_in_index": bool(doc_file) and any(
            entry["file"] == doc_file for entry in DOC_INDEX),
    }
    labels = {
        "file_in_registry_first":
            "no TF-106 registry filing (etl/source_registry.py SOURCES / "
            "check_source_use) — the source is unfiled and quarantined "
            "by default",
        "file_record_schema_in_gate":
            "no TF-108 record schema (etl/schema_gate.py RECORD_SCHEMAS)",
        "name_target_table_and_identity":
            "no TF-109 target-table identity (etl/dedup_merge.py "
            "TABLE_IDENTITIES) for the named target table",
        "name_landed_text_fields_for_poison":
            "no TF-116 landed text fields (etl/poisoned_source.py "
            "SOURCE_TEXT_FIELDS)",
        "write_doc_and_register_in_index":
            "no doc registered in DOC_INDEX for the named doc file",
    }
    present = [step_id for step_id, ok in checks.items() if ok]
    missing = [s["id"] for s in RECIPE_STEPS
               if s["id"] in checks and not checks[s["id"]]]
    problems = [f"{source_id}: step {step_id!r} missing — {labels[step_id]}"
                for step_id in missing]
    return {
        "format": ETL_DOCS_FORMAT,
        "source": source_id,
        "target_table": target_table,
        "doc_file": doc_file,
        "checked_steps": [s["id"] for s in RECIPE_STEPS if s["id"] in checks],
        "present": present,
        "missing": missing,
        "problems": problems,
        "complete": not missing,
    }


def example_recipe_application() -> dict[str, Any]:
    """The worked example: the fictional lantern gazetteer
    filed through the recipe's data steps.

    It is filed nowhere in fact, so its application is
    incomplete and its problems name the registry step
    first — the recipe's checklist catching an unfiled
    source, exactly as filed.
    """
    return recipe_application(
        EXAMPLE_SOURCE["id"], EXAMPLE_SOURCE["target_table"],
        EXAMPLE_SOURCE["doc_file"])


# ── Checkers ────────────────────────────────────────────────────────────────

def _build_all_source() -> str:
    text = (_REPO_ROOT / "forge_memory.py").read_text(encoding="utf-8")
    return text.split("def build_all", 1)[1].split("\ndef ", 1)[0]


def etl_docs_problems() -> list[str]:
    """Static truth: the index + recipe against the live tree."""
    problems: list[str] = []

    rule_ids = [r["id"] for r in RECIPE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate RECIPE_RULES ids")
    for needed in ("registry_before_documentation_or_building",
                   "every_step_files_into_a_real_thing",
                   "build_all_ingests_dbpedia_when_passed",
                   "the_reference_count_is_the_measured_census"):
        if needed not in rule_ids:
            problems.append(f"RECIPE_RULES lost {needed}")

    # The index: unique slices, filed order, paths on disk.
    slices = [e["slice"] for e in DOC_INDEX]
    if len(slices) != len(set(slices)):
        problems.append("duplicate DOC_INDEX slice ids")
    expected_slices = [f"TF-{n}" for n in range(105, 120)]
    if slices != expected_slices:
        problems.append(f"DOC_INDEX slices {slices} != {expected_slices}")
    for entry in DOC_INDEX:
        if not entry["subject"]:
            problems.append(f"DOC_INDEX {entry['slice']} has no subject")
        if not (_REPO_ROOT / entry["file"]).exists():
            problems.append(
                f"DOC_INDEX {entry['slice']} path missing on disk: "
                f"{entry['file']}")
    # Completeness against the live module docs: every
    # Part VI module's filed DOC constant is in the index.
    for module_name in (
            "etl_map", "source_registry", "wikidata_pipeline",
            "schema_gate", "dedup_merge", "subset_craft",
            "embedding_ingestion", "ingestion_dry_run",
            "canon_ingestion", "update_ingestion",
            "ingestion_metrics", "poisoned_source",
            "offline_ingestion", "reference_ingestion"):
        try:
            module = importlib.import_module(
                f"thoughtforge.etl.{module_name}")
        except Exception as exc:  # an unimportable neighbour is a finding
            problems.append(f"cannot import etl/{module_name}: {exc!r}")
            continue
        doc_value = None
        for name, value in vars(module).items():
            if name.endswith("_DOC") and isinstance(value, str) \
                    and value.startswith("docs/architecture/"):
                doc_value = value
                break
        if doc_value is not None and doc_value not in {
                e["file"] for e in DOC_INDEX}:
            problems.append(
                f"etl/{module_name} files {doc_value}, absent from "
                "DOC_INDEX")

    # The recipe: filed order, unique ids, live bindings.
    orders = [s["order"] for s in RECIPE_STEPS]
    if orders != list(range(1, len(RECIPE_STEPS) + 1)):
        problems.append(f"RECIPE_STEPS order is not the filed order: {orders}")
    step_ids = [s["id"] for s in RECIPE_STEPS]
    if len(step_ids) != len(set(step_ids)):
        problems.append("duplicate RECIPE_STEPS ids")
    if len(RECIPE_STEPS) < 10:
        problems.append("RECIPE_STEPS has fewer than 10 steps")
    for step in RECIPE_STEPS:
        try:
            module = importlib.import_module(step["bind_module"])
        except Exception as exc:
            problems.append(
                f"recipe step {step['id']}: cannot import "
                f"{step['bind_module']}: {exc!r}")
            continue
        if not hasattr(module, step["bind_attribute"]):
            problems.append(
                f"recipe step {step['id']}: {step['bind_module']} has "
                f"no {step['bind_attribute']}")
        bind_file = step.get("bind_file")
        if bind_file:
            path = _REPO_ROOT / bind_file
            if not path.exists():
                problems.append(
                    f"recipe step {step['id']}: bound file missing: "
                    f"{bind_file}")
            else:
                text = path.read_text(encoding="utf-8")
                for marker in step.get("bind_file_markers", ()):
                    if marker not in text:
                        problems.append(
                            f"recipe step {step['id']}: bound file "
                            f"{bind_file} lost marker {marker!r}")

    # D-ETL-1: build_all wires DBpediaETL in the filed order.
    # (forge_memory is read / AST-parsed, never imported here:
    # importing the root instrument configures logging, the
    # TF-057 side effect — its functions are judged by file.)
    try:
        build_all_src = _build_all_source()
    except Exception as exc:
        problems.append(f"cannot read forge_memory build_all: {exc!r}")
        build_all_src = ""
    step_marks = ["initialize_schema", "ReferenceDataETL", "WikidataETL",
                  "ConceptNetETL", "GeoNamesETL", "DBpediaETL",
                  "build_entity_embeddings"]
    positions = [build_all_src.find(mark) for mark in step_marks]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        problems.append(
            "forge_memory build_all does not wire DBpediaETL between "
            "GeoNamesETL and build_entity_embeddings (D-ETL-1 reopened)")
    for marker in ("Step 6/7", "Step 7/7", "--dbpedia-json"):
        forge_text = (_REPO_ROOT / "forge_memory.py").read_text(
            encoding="utf-8")
        if marker not in forge_text:
            problems.append(f"forge_memory lost its {marker!r} filing")
            break
    # The build_all function must exist as a function (AST).
    tree = ast.parse((_REPO_ROOT / "forge_memory.py").read_text(
        encoding="utf-8"))
    if not any(isinstance(node, ast.FunctionDef) and node.name == "build_all"
               for node in tree.body):
        problems.append("forge_memory has no build_all function")

    # TF-105's map must file this slice in the open.
    from thoughtforge.etl.etl_map import ORCHESTRATION, ETL_RULES as MAP_RULES

    if "etl_docs" not in ORCHESTRATION.get("etl_docs", ""):
        problems.append("TF-105's map lost its etl_docs entry")
    if "build_all_ingests_dbpedia_when_passed" not in {
            r["id"] for r in MAP_RULES}:
        problems.append("TF-105's map lost the D-ETL-1-fixed rule")

    # D-DATA-1: the stale count appears nowhere in sources.py.
    sources_text = (
        _REPO_ROOT / "src/thoughtforge/etl/sources.py").read_text(
            encoding="utf-8")
    for stale in ("40 built-in", "40 files"):
        if stale in sources_text:
            problems.append(
                f"sources.py still carries the stale count {stale!r} "
                "(D-DATA-1 reopened)")
    if "215 root files + 17 SRD chapters" not in sources_text:
        problems.append("sources.py lost TF-007's measured census")

    # The example must catch its unfiled source, registry first.
    example = example_recipe_application()
    if example["complete"] or "file_in_registry_first" not in \
            example["missing"]:
        problems.append(
            f"the lantern gazetteer example did not catch its missing "
            f"registry filing: {example}")
    if EXAMPLE_SOURCE["id"] in {
            e["file"] for e in DOC_INDEX}:
        problems.append("the example source leaked into DOC_INDEX")

    doc = _REPO_ROOT / ETL_DOCS_DOC
    if not doc.exists():
        problems.append(f"the write-up {ETL_DOCS_DOC} is missing")
    return problems


def etl_docs_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers and the doctor."""
    return {
        "format": ETL_DOCS_FORMAT,
        "doc": ETL_DOCS_DOC,
        "doc_index": [dict(e) for e in DOC_INDEX],
        "recipe_steps": [
            {**dict(s),
             "bind_file_markers": list(s.get("bind_file_markers", ()))}
            for s in RECIPE_STEPS],
        "rules": [dict(r) for r in RECIPE_RULES],
        "example_source": dict(EXAMPLE_SOURCE),
        "example_application": example_recipe_application(),
    }


__all__ = [
    "DOC_INDEX",
    "ETL_DOCS_DOC",
    "ETL_DOCS_FORMAT",
    "EXAMPLE_SOURCE",
    "RECIPE_RULES",
    "RECIPE_STEPS",
    "etl_docs_as_data",
    "etl_docs_problems",
    "example_recipe_application",
    "recipe_application",
]
