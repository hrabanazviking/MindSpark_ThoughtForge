"""
Canon ingestion path, as data (TF-113).

The roadmap's rule: *Volmarr's own materials ingested by
the same pipeline (private stores only; fixtures
synthetic).*

Before this slice, that path did not exist in fact —
probed live on this tree before anything was built:

  * **No canon path existed anywhere.** No module, no
    ``forge_memory.py`` command and no test named one
    (grep over ``src/``, ``tests/`` and the root
    instruments: no canon ingestion of any kind). The
    owner's own materials could enter a store only
    through the generic ``ReferenceDataETL`` — the
    built-in library's builder — indistinguishably from
    that library, with no record that the material was
    the owner's canon at all.
  * **Nothing guarded the store.** The generic builder
    writes into any ``db_path`` it is handed: no caller
    had to affirm the target was a private local store,
    nothing forbade the store sitting inside the canon
    directory itself, and nothing in the resulting store
    or beside it said the chunks were private canon
    (live probe: a synthetic canon directory built into
    an arbitrary DB; the rows' ``source`` is the schema
    default ``reference`` and no manifest exists).
  * **The TF-106 registry has no canon entry** — probing
    ``check_source_use("owner_canon", ...)`` raises
    unregistered-by-name. That is correct for a registry
    of *external* sources, but it left the owner's own
    materials with no gate and no provenance record
    anywhere: canon provenance lived only in campaign
    prose. (D-CANON-1, found by this slice's probes and
    FIXED here.)
  * **Skips and duplicates were silent on this road
    too** (D-DATA-4's ingestion shape): the same probe's
    stats were ``{'files': 2, 'chunks': 2, 'skipped': 0}``
    for a directory holding one unique readable file,
    one byte-identical copy of it, one .jpg and one
    .csv — the image and the CSV were passed over
    uncounted and unnamed, and the duplicate was
    chunked a second time with nothing recording it.

What this slice builds, filed as data in
``CANON_RULES`` and enforced here:

  * ``ingest_canon()`` — the owner's materials ingested
    by THE SAME pipeline: the real
    ``ReferenceDataETL.build()`` with its own chunking,
    merge ledger (TF-109) and FTS rebuild (TF-058,
    D-REF-1). No second chunker exists in this module.
  * **Private stores only, affirmed explicitly.** The
    caller must pass ``private_store=True`` — an
    affirmation that the target is a private local
    store — before anything is read or written; without
    it ``CanonPrivacyError`` raises and nothing is
    created. The store may never sit inside the canon
    directory (``CanonTargetError``). ``check_canon_use``
    is the use gate: private ingestion and private
    retrieval pass with the provenance record;
    redistribution and publication raise.
  * **Canon is read-only input, proven.** A TF-058
    ``ReferenceLibrary`` inventory (metadata only —
    path, size, SHA-256, class, duplicate keeper; never
    content) is taken before and after the build and
    must agree exactly, or the report files
    ``source_untouched=False`` in the open.
  * **The provenance manifest.** Beside the store the
    driver writes ``<db>.canon-manifest.json``
    atomically (tmp + os.replace, the TF-107/TF-111
    posture): the filed provenance (owner-authored,
    private, no redistribution, no licence claim made
    by the campaign), the metadata-only inventory, the
    builder's stats, the store counts, the merge
    ledger, the named skips and duplicates, and the
    embedding leg's state — content-hashed
    (``manifest_sha256`` over the canonical JSON) so a
    later reader can verify the record was not edited.
    No file content is carried: canon is described,
    never copied (TF-058's rule).
  * **Skips and duplicates are named, not silent.**
    Every non-readable file is named in the report and
    manifest under ``skipped_by_rule`` with its TF-058
    class (image / other); every byte-duplicate
    readable file is named with its keeper. The
    builder still chunks a duplicate's content (its
    (source_file, chunk_index) identity is per-file) —
    that ingestion-side dedup is TF-118's filed share
    of D-DATA-4 for the reference road generally, and
    this slice does not smuggle it in; what changes
    here is that on the canon road the duplication is
    *recorded*, never invisible.
  * **Embeddings only through an injected encoder**
    (TF-111's driver, as TF-112 filed): with no encoder
    the leg is filed not-run and no model is loaded.

Filed limits, stated plainly:

  * Canon is not a TF-106 registry entry: the registry
    records *external* sources' upstream terms, and
    the owner's own materials have no upstream — their
    provenance and privacy record lives here instead.
    TF-119's recipe and TF-190's final pass inherit
    this filing unchanged.
  * The privacy affirmation is an explicit caller
    declaration, not a filesystem proof: this module
    cannot prove a path is private, so it requires the
    declaration, refuses the store-in-canon shape, and
    files the obligations in the manifest where any
    consumer of the store can read them.
  * One canon directory per call; a second call into
    the same store appends through the builder's own
    merge rules and rewrites the manifest for that
    call only — manifests are per-call records.

Docket (TF-016): D-CANON-1 is added and FIXED by this
slice — owner's materials had no ingestion path with
a private-store guard or a provenance record; both
now exist here, proven on synthetic fixtures.

The human-readable write-up is
``docs/architecture/CANON_INGESTION.md``. Proof
grade: FIXTURE — synthetic canon stand-ins through
the real builder on this VM; the real canon tree is
not in this sparse checkout and was never opened,
never claimed.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CANON_FORMAT = "thoughtforge-canon-ingestion/1"
CANON_DOC = "docs/architecture/CANON_INGESTION.md"
CANON_SOURCE_ID = "owner_canon"
MANIFEST_SUFFIX = ".canon-manifest.json"

USE_INGEST_PRIVATE = "ingest_private"
USE_RETRIEVE_PRIVATE = "retrieve_private"
USE_REDISTRIBUTE = "redistribute"
USE_PUBLISH = "publish"
CANON_USES = (USE_INGEST_PRIVATE, USE_RETRIEVE_PRIVATE,
              USE_REDISTRIBUTE, USE_PUBLISH)
_PRIVATE_USES = (USE_INGEST_PRIVATE, USE_RETRIEVE_PRIVATE)

#: The provenance record for the owner's own materials.
#: There is no upstream and no licence to verify — the
#: material is the owner's, and the campaign claims no
#: licence over it; it files the privacy obligations.
CANON_PROVENANCE: dict[str, Any] = {
    "id": CANON_SOURCE_ID,
    "name": "Volmarr's own materials (canon)",
    "provenance": "authored by the store's owner (Volmarr Wyrd) — "
                  "his own writings and materials, held on his own "
                  "machines; no external source, no upstream terms",
    "licence": None,
    "licence_note": "no licence is stated or claimed by the campaign "
                    "for the owner's own materials — they are his, "
                    "private by his law",
    "privacy": "private_local_only",
    "obligations": ("no_redistribution", "no_publication",
                    "private_stores_only"),
}


class CanonPrivacyError(ValueError):
    """Canon was asked to go somewhere private law forbids."""


class CanonTargetError(ValueError):
    """The target store would sit inside the canon itself."""


def check_canon_use(use: str) -> dict[str, Any]:
    """The use gate: which uses of canon are allowed?

    Private ingestion and private retrieval return the
    provenance record. Redistribution and publication
    raise ``CanonPrivacyError``; an unknown use raises
    ``ValueError`` by name.
    """
    if use not in CANON_USES:
        raise ValueError(
            f"unknown canon use {use!r}: expected one of {CANON_USES}")
    if use not in _PRIVATE_USES:
        raise CanonPrivacyError(
            f"canon may not be used for {use!r}: the owner's own "
            "materials are private — private local ingestion and "
            "retrieval only, no redistribution, no publication")
    return dict(CANON_PROVENANCE)


# ── The rules ───────────────────────────────────────────────────────────────

CANON_RULES: tuple[dict[str, Any], ...] = (
    {"id": "the_same_pipeline",
     "rule": "canon is ingested by the real ReferenceDataETL.build() — its chunking, its TF-109 merge ledger, its TF-058 FTS rebuild; this module builds no second chunker"},
    {"id": "private_stores_only_affirmed",
     "rule": "ingest_canon requires private_store=True explicitly before anything is read or written; without the affirmation CanonPrivacyError raises and no store is created"},
    {"id": "the_store_is_never_inside_canon",
     "rule": "a db_path resolving inside the canon directory raises CanonTargetError before anything runs — the pipeline never writes into the material it reads"},
    {"id": "canon_is_read_only_input_proven",
     "rule": "a TF-058 metadata-only inventory (path, size, SHA-256, class, duplicate keeper — never content) is taken before and after the build and must agree exactly for the report to carry source_untouched=True"},
    {"id": "the_provenance_manifest_travels_with_the_store",
     "rule": "a content-hashed manifest (<db>.canon-manifest.json, written atomically) files the provenance, privacy obligations, metadata-only inventory, stats, store counts, merge ledger, named skips and duplicates beside the store; it carries no file content"},
    {"id": "skips_and_duplicates_are_named",
     "rule": "every non-readable file is named under skipped_by_rule with its TF-058 class and every byte-duplicate readable file is named with its keeper — on the canon road a skip or a duplicate is a record, never a silence (the builder's per-file chunking of duplicates stands; ingestion-side dedup is TF-118's share)"},
    {"id": "redistribution_and_publication_raise",
     "rule": "check_canon_use passes private ingestion/retrieval with the provenance record and raises CanonPrivacyError for redistribute and publish; an unknown use raises by name"},
    {"id": "a_missing_canon_directory_raises_before_the_store",
     "rule": "a canon directory that does not exist raises FileNotFoundError naming it before any store file is created — canon that is absent files no store and no manifest"},
    {"id": "embeddings_only_with_an_injected_encoder",
     "rule": "the embedding leg runs only when the caller injects an encoder, through TF-111's driver over the canon store; with no encoder the leg is filed not-run and no model is loaded"},
)


# ── The report ──────────────────────────────────────────────────────────────

@dataclass
class CanonReport:
    source_id: str
    provenance: dict[str, Any]
    canon_dir: str
    db_path: str
    stats: dict[str, Any]
    store_counts: dict[str, int]
    merge: dict[str, Any]
    inventory: dict[str, Any]
    skipped_by_rule: list[dict[str, Any]]
    duplicates: list[dict[str, Any]]
    embeddings: dict[str, Any]
    source_untouched: bool
    manifest_path: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return report_as_data(self)


def report_as_data(report: CanonReport) -> dict[str, Any]:
    """The report as plain data (the filed format)."""
    return {
        "format": CANON_FORMAT,
        "doc": CANON_DOC,
        "source_id": report.source_id,
        "provenance": dict(report.provenance),
        "canon_dir": report.canon_dir,
        "db_path": report.db_path,
        "stats": dict(report.stats),
        "store_counts": dict(report.store_counts),
        "merge": report.merge,
        "inventory": report.inventory,
        "skipped_by_rule": [dict(s) for s in report.skipped_by_rule],
        "duplicates": [dict(d) for d in report.duplicates],
        "embeddings": dict(report.embeddings),
        "source_untouched": report.source_untouched,
        "manifest_path": report.manifest_path,
        "manifest_sha256": report.manifest_sha256,
    }


def _manifest_digest(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_manifest(db_path: Path, manifest: dict[str, Any]) -> str:
    """Write the manifest atomically; return its digest."""
    digest = _manifest_digest(manifest)
    payload = {**manifest, "manifest_sha256": digest}
    target = Path(str(db_path) + MANIFEST_SUFFIX)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(target.parent),
        prefix=target.name + ".", suffix=".tmp", delete=False)
    try:
        json.dump(payload, handle, indent=2, sort_keys=True,
                  ensure_ascii=False)
        handle.write("\n")
        handle.close()
        Path(handle.name).replace(target)
    except BaseException:
        try:
            Path(handle.name).unlink()
        except OSError:
            pass
        raise
    return digest


# ── The driver ──────────────────────────────────────────────────────────────

def ingest_canon(
    canon_dir: Path | str,
    db_path: Path | str,
    *,
    private_store: bool = False,
    encoder: Callable[..., Any] | None = None,
) -> CanonReport:
    """Ingest the owner's materials into a private store.

    ``canon_dir`` is a directory of the owner's own
    materials; ``db_path`` is the private local store to
    build into. ``private_store=True`` must be passed
    explicitly. See the module docstring for the rules.
    """
    from thoughtforge.etl.schema import get_entity_count
    from thoughtforge.etl.sources import ReferenceDataETL
    from thoughtforge.knowledge.reference_library import ReferenceLibrary

    # Rule: the use is judged and the private-store
    # affirmation is required FIRST — before the canon
    # directory is even stat'ed and before any file exists.
    provenance = check_canon_use(USE_INGEST_PRIVATE)
    if not private_store:
        raise CanonPrivacyError(
            "canon ingestion requires private_store=True: the "
            "caller must affirm the target is a private local "
            "store — canon never enters a store on an unstated "
            "footing, and nothing was written")

    canon_dir = Path(canon_dir)
    if not canon_dir.is_dir():
        raise FileNotFoundError(
            f"canon directory not found: {canon_dir}")

    db_path = Path(db_path)
    if db_path.resolve().is_relative_to(canon_dir.resolve()):
        raise CanonTargetError(
            f"the target store {db_path} sits inside the canon "
            f"directory {canon_dir} — the pipeline never writes "
            "into the material it reads; choose a store outside it")

    library = ReferenceLibrary(ref_dir=canon_dir)
    inventory_before = library.inventory()

    builder = ReferenceDataETL(db_path)
    stats = builder.build(ref_dir=canon_dir)

    store_counts = get_entity_count(db_path)
    merge = builder.merge_ledger.as_data()

    embeddings: dict[str, Any]
    if encoder is not None:
        from thoughtforge.etl.embedding_ingestion import (
            build_embeddings_at_ingestion,
        )

        result = build_embeddings_at_ingestion(db_path, encoder=encoder)
        embeddings = {"status": "run", **result.to_dict()}
        store_counts = get_entity_count(db_path)
    else:
        embeddings = {
            "status": "not_run",
            "reason": "no encoder was injected — canon ingestion "
                      "loads no model on its own word "
                      "(TF-111's injected-encoder leg)",
        }

    inventory_after = library.inventory()
    source_untouched = inventory_before == inventory_after

    files = inventory_after["files"]
    skipped_by_rule = [
        {"file": f["relative_path"], "file_class": f["file_class"],
         "extension": f["extension"]}
        for f in files if f["file_class"] != "readable"
    ]
    duplicates = [
        {"file": f["relative_path"], "duplicate_of": f["duplicate_of"],
         "sha256": f["sha256"]}
        for f in files if f["duplicate_of"]
    ]
    inventory_report = {
        "total_files": inventory_after["total_files"],
        "total_bytes": inventory_after["total_bytes"],
        "files_by_class": inventory_after["files_by_class"],
        "bytes_by_class": inventory_after["bytes_by_class"],
        "files": files,
    }

    manifest: dict[str, Any] = {
        "format": CANON_FORMAT,
        "doc": CANON_DOC,
        "source_id": CANON_SOURCE_ID,
        "provenance": provenance,
        "canon_dir": str(canon_dir),
        "db_path": str(db_path),
        "stats": dict(stats),
        "store_counts": store_counts,
        "merge": merge,
        "inventory": inventory_report,
        "skipped_by_rule": skipped_by_rule,
        "duplicates": duplicates,
        "embeddings": embeddings,
        "source_untouched": source_untouched,
    }
    digest = _write_manifest(db_path, manifest)

    return CanonReport(
        source_id=CANON_SOURCE_ID,
        provenance=provenance,
        canon_dir=str(canon_dir),
        db_path=str(db_path),
        stats=dict(stats),
        store_counts=store_counts,
        merge=merge,
        inventory=inventory_report,
        skipped_by_rule=skipped_by_rule,
        duplicates=duplicates,
        embeddings=embeddings,
        source_untouched=source_untouched,
        manifest_path=str(db_path) + MANIFEST_SUFFIX,
        manifest_sha256=digest,
    )


# ── Checkers ────────────────────────────────────────────────────────────────

def canon_problems() -> list[str]:
    """Static truth: the filed rules against the live tree."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in CANON_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate CANON_RULES ids")
    for needed in ("the_same_pipeline", "private_stores_only_affirmed",
                   "the_store_is_never_inside_canon",
                   "the_provenance_manifest_travels_with_the_store",
                   "skips_and_duplicates_are_named",
                   "redistribution_and_publication_raise"):
        if needed not in rule_ids:
            problems.append(f"CANON_RULES lost {needed}")
    if CANON_PROVENANCE["licence"] is not None:
        problems.append("canon provenance states a licence — none "
                        "may be claimed for the owner's materials")
    if CANON_PROVENANCE["privacy"] != "private_local_only":
        problems.append("canon provenance lost its private_local_only "
                        "privacy")
    source = Path(__file__).read_text(encoding="utf-8")
    for marker in ("ReferenceDataETL", "ReferenceLibrary",
                   "build_embeddings_at_ingestion"):
        if marker not in source:
            problems.append(f"canon_ingestion.py lost its {marker} leg")
    from thoughtforge.etl.etl_map import ORCHESTRATION

    if "canon_ingestion" not in ORCHESTRATION.get("canon", ""):
        problems.append("TF-105's map lost its canon orchestration entry")
    doc = Path(__file__).resolve().parents[3] / CANON_DOC
    if not doc.exists():
        problems.append(f"the write-up {CANON_DOC} is missing")
    return problems


def _synthetic_canon(root: Path) -> Path:
    """A tiny synthetic canon stand-in (TF-017)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "lantern_lore.md").write_text(
        "# Lantern Lore\n\nThe keeper trims the lantern at dusk.\n",
        encoding="utf-8")
    (root / "copy.md").write_bytes(
        (root / "lantern_lore.md").read_bytes())
    (root / "empty.md").write_text("", encoding="utf-8")
    (root / "portrait.jpg").write_bytes(b"\xff\xd8x")
    (root / "notes.csv").write_text("a,b\n", encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def canon_live_problems(base: Path | None = None) -> list[str]:
    """Behavioural truth: the canon road, end to end."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(base) if base is not None else Path(tmp) / "live"
        canon = _synthetic_canon(root / "canon")
        before = _snapshot(canon)
        db = root / "private.db"

        # No affirmation, no store.
        try:
            ingest_canon(canon, db)
            problems.append("canon was ingested without private_store=True")
        except CanonPrivacyError:
            pass
        if db.exists():
            problems.append("a refused ingestion created its store")

        # The store may never sit inside canon.
        try:
            ingest_canon(canon, canon / "store.db", private_store=True)
            problems.append("a store inside canon was allowed")
        except CanonTargetError:
            pass

        # A missing canon directory raises before the store.
        try:
            ingest_canon(root / "absent-canon", db, private_store=True)
            problems.append("a missing canon directory was ingested")
        except FileNotFoundError as exc:
            if "absent-canon" not in str(exc):
                problems.append(f"missing-canon refusal unnamed: {exc}")
        if db.exists():
            problems.append("a missing-canon refusal created its store")

        # The gate: private uses pass, public uses raise.
        if check_canon_use(USE_RETRIEVE_PRIVATE)["id"] != CANON_SOURCE_ID:
            problems.append("the canon gate lost its provenance record")
        for use in (USE_REDISTRIBUTE, USE_PUBLISH):
            try:
                check_canon_use(use)
                problems.append(f"canon use {use!r} was allowed")
            except CanonPrivacyError:
                pass

        # The happy road through the real builder.
        report = ingest_canon(canon, db, private_store=True)
        if report.stats != {"files": 2, "chunks": 2, "skipped": 1}:
            problems.append(f"canon stats drifted: {report.stats}")
        if report.store_counts.get("reference_chunks") != 2:
            problems.append(f"canon store counts drifted: "
                            f"{report.store_counts}")
        if report.source_untouched is not True or _snapshot(canon) != before:
            problems.append("the canon source was touched by ingestion")
        skipped = {s["file"]: s["file_class"]
                   for s in report.skipped_by_rule}
        if skipped != {"portrait.jpg": "image", "notes.csv": "other"}:
            problems.append(f"canon skips drifted: {skipped}")
        if len(report.duplicates) != 1 or \
                report.duplicates[0]["file"] != "lantern_lore.md" or \
                report.duplicates[0]["duplicate_of"] != "copy.md":
            problems.append(f"canon duplicates drifted: {report.duplicates}")
        if report.embeddings.get("status") != "not_run":
            problems.append("the embedding leg ran without an encoder")

        # The manifest: present, metadata-only, hash true.
        manifest_path = Path(report.manifest_path)
        if not manifest_path.is_file():
            problems.append("the canon manifest was not written")
        else:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8"))
            filed_digest = payload.pop("manifest_sha256")
            if filed_digest != report.manifest_sha256 or \
                    _manifest_digest(payload) != filed_digest:
                problems.append("the canon manifest's hash does not "
                                "verify its contents")
            if any("content" in row
                   for row in payload["inventory"]["files"]):
                problems.append("the manifest carries file content "
                                "(canon rule broken)")

        # Retrieval: the ingested canon is searchable in its
        # private store (the TF-058 FTS road).
        from thoughtforge.knowledge.forge import KnowledgeForge

        rows = KnowledgeForge(db_path=db).sql_retrieve("lantern keeper")
        if not rows or rows[0]["source_file"] != "lantern_lore.md":
            problems.append(f"canon retrieval drifted: {rows[:1]}")

        # The injected-encoder leg (TF-111).
        def fake_encoder(texts, model_name):
            import numpy as np

            vectors = []
            for text in texts:
                seed = int(hashlib.sha256(
                    text.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                vectors.append(rng.random(384, dtype=np.float32))
            return np.array(vectors, dtype=np.float32)

        embedded_db = root / "embedded.db"
        embedded = ingest_canon(canon, embedded_db, private_store=True,
                                encoder=fake_encoder)
        if embedded.embeddings.get("status") != "run" or \
                embedded.embeddings.get("reference_built") != 2:
            problems.append(f"the canon embedding leg drifted: "
                            f"{embedded.embeddings}")
        from thoughtforge.etl.schema import get_connection

        conn = get_connection(embedded_db)
        try:
            ref_emb = conn.execute(
                "SELECT COUNT(*) FROM reference_embeddings").fetchone()[0]
        finally:
            conn.close()
        if ref_emb != 2:
            problems.append(f"embedded canon rows drifted: {ref_emb}")
    return problems


def canon_as_data() -> dict[str, Any]:
    """The canon path as plain data, for consumers and the doctor."""
    return {
        "format": CANON_FORMAT,
        "doc": CANON_DOC,
        "provenance": dict(CANON_PROVENANCE),
        "uses": list(CANON_USES),
        "rules": [dict(r) for r in CANON_RULES],
    }
