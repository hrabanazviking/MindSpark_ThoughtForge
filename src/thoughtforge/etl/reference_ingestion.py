"""
The reference library as a source, as data (TF-118).

The roadmap's rule: *the reference library as a source
(TF-058's write side): its ingestion reproducible +
checksummed.*

TF-058 built the READ side (``knowledge/reference_library.py``:
classified, hashed, deduped, read-only). This slice is the
WRITE side. Measured starting state, probed live on this
tree before anything was built:

  * **The ingestion side of D-DATA-4 stood open exactly as
    TF-058 filed it.** One unique readable file plus a
    byte-identical copy, a .jpg, a .csv and an empty .md
    built to ``{'files': 2, 'chunks': 2, 'skipped': 1}``:
    the duplicate was chunked a second time under its own
    name (the store held the same content twice) and the
    .jpg/.csv were passed over by a bare ``continue`` —
    uncounted (the one skip was the empty file) and named
    nowhere. TF-113 filed the same shape for the canon
    road and named this slice as the owner of the fix.
  * **No checksum existed anywhere on this road
    (D-REF-2, found by this slice's probes).** ``sources.py``
    carried no ``sha256`` at all; no per-file hash, no
    per-chunk checksum, no store digest and no manifest
    was written beside the store. A stored chunk edited
    in place (probed: one ``UPDATE`` setting its content
    to ``TAMPERED``) was detectable by no tool in the
    tree — and a plain re-run silently overwrote it
    through the merge rules, leaving no record that the
    store had ever drifted from its source.
  * **Reproducibility held in fact but was proven
    nowhere (D-REF-3, found by this slice's probes).**
    Two fresh builds of the same probe library produced
    equal chunk rows — but no test, checker, digest or
    manifest had ever compared two builds, so the claim
    rested on reading the builder's sort order, not on
    evidence a consumer could re-check.

What this slice builds:

  * **The builder itself skips by rule and chunks once
    (D-DATA-4 FIXED).** ``ReferenceDataETL.build()`` now
    holds a fresh ``library_gate`` (``LibraryLedger``)
    per build: every unsupported file is classified
    (TF-058's classes), counted in ``skipped`` and named
    in the ledger; every readable file is SHA-256'd
    before chunking and a byte-duplicate of an earlier
    (sorted-order) file is named in the ledger with its
    keeper, counted in ``skipped``, and never chunked —
    and its stale stored chunks from a pre-fix build are
    retired in the same run through TF-114's retirement,
    ledgered in the builder's ``update_ledger``.
  * ``ingest_reference_library()`` — the library ingested
    by the real builder, filed as a report + a manifest:
    a TF-058 metadata-only inventory before and after
    (the source proven untouched), per-file SHA-256 +
    status (ingested / duplicate / skipped_by_rule /
    no_chunks), per-chunk checksums, the builder's four
    ledgers whole (library, merge, update, poison), a
    library digest over the filed file hashes and a
    store digest over the landed chunk rows. The
    manifest (``<db>.reference-manifest.json``, written
    atomically, content-hashed like TF-113's) carries
    hashes and counts only — no file or chunk content.
  * ``store_reference_digest()`` + ``verify_reference_store()``
    — the checksummed claim made checkable: verification
    re-derives the manifest's own hash, every landed
    chunk's checksum and the store digest, and the
    source files' current hashes, and names every drift
    (tampered / missing / extra chunk, changed / missing
    / added source file) instead of overwriting it.
  * ``check_reproducibility()`` — two independent builds
    of one library into two stores, compared on stats,
    chunk checksums and store digest: reproducible only
    when all three agree.

Filed limits, stated plainly:

  * The digest is over the ``reference_chunks`` rows, in
    (source_file, chunk_index) order with each row's
    fields as stored — other tables (embeddings,
    ingestion_log) are a store's other stages, not this
    source's ingestion claim.
  * A re-run through the plain builder still repairs
    drift silently (TF-109/TF-114 own re-run semantics);
    what this slice adds is that drift is *detectable
    and nameable before* any re-run, through the
    manifest + verify road.
  * Embeddings run only through TF-111's driver with an
    injected encoder (as TF-112/TF-113 filed); with no
    encoder the leg is filed not-run.
  * No CLI wiring: ``forge_memory.py`` gains no command
    here — TF-119 owns CLI/docs wiring.

Docket (TF-016): D-DATA-4 is FIXED by this slice (its
TF-058 read share already stood fixed); D-REF-2 and
D-REF-3 are added and FIXED by this slice.

The human-readable write-up is
``docs/architecture/REFERENCE_LIBRARY_AS_SOURCE.md``.
Proof grade: FIXTURE — synthetic libraries through the
real builder on this VM; the real built-in library is
not in this sparse checkout and was never opened,
never claimed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REFERENCE_FORMAT = "thoughtforge-reference-ingestion/1"
REFERENCE_DOC = "docs/architecture/REFERENCE_LIBRARY_AS_SOURCE.md"
LEDGER_FORMAT = "thoughtforge-reference-library-ledger/1"
VERIFY_FORMAT = "thoughtforge-reference-verification/1"
MANIFEST_SUFFIX = ".reference-manifest.json"

#: The fields a chunk checksum covers, in filed order —
#: exactly the columns the reference builder lands.
CHUNK_FIELDS: tuple[str, ...] = (
    "source_file", "chunk_index", "title", "content", "tags",
    "word_count",
)

#: Image extensions, pinned equal to TF-058's set by the
#: static checker (the read side's classification is the
#: one classification; this copy exists so the ETL never
#: imports the knowledge package at builder time).
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
)

FILE_STATUSES: tuple[str, ...] = (
    "ingested", "duplicate", "skipped_by_rule", "no_chunks",
)


def classify_reference_extension(extension: str) -> str:
    """TF-058's class name for one extension (lowercased):
    ``image`` for imagery, ``other`` for anything else
    outside the readable set."""
    return "image" if extension.lower() in IMAGE_EXTENSIONS else "other"


def file_sha256(path: Path) -> str:
    """The SHA-256 of one file's bytes, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_checksum(row: dict[str, Any]) -> str:
    """The SHA-256 of one landed chunk row's filed fields."""
    canonical = json.dumps(
        {field: row[field] for field in CHUNK_FIELDS},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _digest_rows(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── The builder's library ledger ────────────────────────────────────────────


class LibraryLedger:
    """The ledger ReferenceDataETL holds fresh per build.

    Records ONLY the two by-rule outcomes the build's
    bare stats used to hide: files skipped because their
    suffix is outside the supported set, and readable
    files skipped because their bytes duplicate an
    earlier file's. Chunk-level outcomes stay with the
    ledgers that own them (TF-109 merge, TF-114 update,
    TF-116 poison).
    """

    def __init__(self) -> None:
        self.skipped_by_rule: list[dict[str, Any]] = []
        self.duplicates: list[dict[str, Any]] = []

    def note_skip(self, file: str, extension: str) -> None:
        self.skipped_by_rule.append({
            "file": file,
            "extension": extension,
            "file_class": classify_reference_extension(extension),
        })

    def note_duplicate(self, file: str, duplicate_of: str,
                       sha256: str) -> None:
        self.duplicates.append({
            "file": file, "duplicate_of": duplicate_of,
            "sha256": sha256,
        })

    @property
    def skipped(self) -> int:
        """Files this ledger accounts for in stats' skipped."""
        return len(self.skipped_by_rule) + len(self.duplicates)

    def as_data(self) -> dict[str, Any]:
        return {
            "format": LEDGER_FORMAT,
            "skipped_by_rule": [dict(s) for s in self.skipped_by_rule],
            "duplicates": [dict(d) for d in self.duplicates],
            "skipped_files": self.skipped,
        }


# ── The rules ───────────────────────────────────────────────────────────────

REFERENCE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "ingestion_is_the_real_builder",
     "rule": "the library is ingested by the real ReferenceDataETL.build() — its chunking, its TF-109 merge, TF-114 update and TF-116 poison ledgers; this module builds no second chunker"},
    {"id": "skip_by_rule_counted_and_named",
     "rule": "every file outside the supported set is skipped by rule: classified (image / other, TF-058's classes), counted in the build's skipped, and named in the builder's library_gate ledger — never a bare continue (D-DATA-4)"},
    {"id": "duplicates_chunked_once",
     "rule": "a readable file whose SHA-256 equals an earlier sorted-order readable file's is a duplicate: named in the library_gate ledger with its keeper, counted in skipped, and never chunked — the store holds each distinct content once (D-DATA-4)"},
    {"id": "a_duplicate_rerun_retires_stale_chunks",
     "rule": "when a re-run finds a file is now a duplicate (or a pre-fix store still holds its chunks), those stored chunks are retired in the same run through TF-114's retirement and ledgered in the update ledger"},
    {"id": "every_file_and_chunk_checksummed",
     "rule": "the manifest files every source file's SHA-256 (TF-058's metadata-only inventory) and every landed chunk's checksum over its filed fields — a claim about the store can be re-checked without trusting the store (D-REF-2)"},
    {"id": "the_store_digest_names_the_ingestion",
     "rule": "store_reference_digest hashes the landed chunk rows in (source_file, chunk_index) order; two stores with the same digest hold the same ingestion of the library, whatever else their other stages did"},
    {"id": "reproducible_means_three_agreements",
     "rule": "ingestion is reproducible only when two independent builds of the same library agree on stats, on every chunk checksum, and on the store digest — proven by check_reproducibility, never asserted from the builder's sort order (D-REF-3)"},
    {"id": "the_manifest_travels_with_the_store",
     "rule": "a content-hashed manifest (<db>.reference-manifest.json, written atomically) files the inventory, per-file statuses, chunk checksums, digests, stats, store counts and the builder's ledgers beside the store; it carries no file or chunk content"},
    {"id": "verification_names_drift_never_repairs_it",
     "rule": "verify_reference_store re-derives the manifest hash, the chunk checksums, the store digest and the source files' current hashes, and names every tampered / missing / extra chunk and changed / missing / added file; it writes nothing and repairs nothing"},
    {"id": "the_source_is_read_only_input_proven",
     "rule": "a TF-058 metadata-only inventory is taken before and after the build and must agree exactly for the report to carry source_untouched=True"},
    {"id": "embeddings_only_with_an_injected_encoder",
     "rule": "the embedding leg runs only when the caller injects an encoder, through TF-111's driver; with no encoder the leg is filed not-run and no model is loaded"},
)


# ── Reading a store's reference rows (read-only) ────────────────────────────


def store_reference_rows(db_path: Path | str) -> list[dict[str, Any]]:
    """Every landed reference chunk row, in filed order."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"reference store not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        fetched = conn.execute(
            "SELECT source_file, chunk_index, title, content, tags,"
            " word_count FROM reference_chunks"
            " ORDER BY source_file, chunk_index").fetchall()
    finally:
        conn.close()
    return [dict(zip(CHUNK_FIELDS, row)) for row in fetched]


def chunk_records(db_path: Path | str) -> list[dict[str, Any]]:
    """The landed chunks as checksum records (no content)."""
    return [{
        "chunk_id": f"{row['source_file']}#{row['chunk_index']}",
        "source_file": row["source_file"],
        "chunk_index": row["chunk_index"],
        "sha256": chunk_checksum(row),
    } for row in store_reference_rows(db_path)]


def store_reference_digest(db_path: Path | str) -> str:
    """The digest naming one store's reference ingestion."""
    return _digest_rows(chunk_records(db_path))


def library_digest(files: list[dict[str, Any]]) -> str:
    """The digest naming one library inventory: every
    file's path + SHA-256 + class, in inventory order."""
    return _digest_rows([
        {"relative_path": f["relative_path"], "sha256": f["sha256"],
         "file_class": f["file_class"]}
        for f in files
    ])


# ── The report ──────────────────────────────────────────────────────────────

@dataclass
class ReferenceReport:
    ref_dir: str
    db_path: str
    stats: dict[str, Any]
    store_counts: dict[str, int]
    files: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    library_digest: str
    store_digest: str
    library_gate: dict[str, Any]
    merge: dict[str, Any]
    update: dict[str, Any]
    poison: dict[str, Any]
    embeddings: dict[str, Any]
    source_untouched: bool
    manifest_path: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return report_as_data(self)


def report_as_data(report: ReferenceReport) -> dict[str, Any]:
    """The report as plain data (the filed format)."""
    return {
        "format": REFERENCE_FORMAT,
        "doc": REFERENCE_DOC,
        "ref_dir": report.ref_dir,
        "db_path": report.db_path,
        "stats": dict(report.stats),
        "store_counts": dict(report.store_counts),
        "files": [dict(f) for f in report.files],
        "chunks": [dict(c) for c in report.chunks],
        "library_digest": report.library_digest,
        "store_digest": report.store_digest,
        "library_gate": report.library_gate,
        "merge": report.merge,
        "update": report.update,
        "poison": report.poison,
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

def ingest_reference_library(
    ref_dir: Path | str,
    db_path: Path | str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    encoder: Callable[..., Any] | None = None,
) -> ReferenceReport:
    """Ingest one reference library into one store, with
    the checksummed manifest filed beside the store.

    ``ref_dir`` is the library directory; ``db_path`` is
    the store to build into. A missing library raises
    ``FileNotFoundError`` before any store is created.
    See the module docstring for the rules.
    """
    from thoughtforge.etl.schema import get_entity_count
    from thoughtforge.etl.sources import (
        CHUNK_OVERLAP_CHARS,
        CHUNK_SIZE_CHARS,
        ReferenceDataETL,
    )
    from thoughtforge.knowledge.reference_library import ReferenceLibrary

    ref_dir = Path(ref_dir)
    if not ref_dir.is_dir():
        raise FileNotFoundError(
            f"reference library directory not found: {ref_dir}")
    db_path = Path(db_path)

    library = ReferenceLibrary(ref_dir=ref_dir)
    inventory_before = library.inventory()

    builder = ReferenceDataETL(db_path)
    build_kwargs: dict[str, Any] = {"ref_dir": ref_dir}
    if chunk_size is not None:
        build_kwargs["chunk_size"] = chunk_size
    if overlap is not None:
        build_kwargs["overlap"] = overlap
    stats = builder.build(**build_kwargs)

    store_counts = get_entity_count(db_path)

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
            "reason": "no encoder was injected — reference "
                      "ingestion loads no model on its own word "
                      "(TF-111's injected-encoder leg)",
        }

    inventory_after = library.inventory()
    source_untouched = inventory_before == inventory_after

    chunks = chunk_records(db_path)
    landed_by_file: dict[str, int] = {}
    for chunk in chunks:
        landed_by_file[chunk["source_file"]] = \
            landed_by_file.get(chunk["source_file"], 0) + 1

    files: list[dict[str, Any]] = []
    for row in inventory_after["files"]:
        landed = landed_by_file.get(row["relative_path"], 0)
        if row["duplicate_of"]:
            status = "duplicate"
        elif row["file_class"] != "readable":
            status = "skipped_by_rule"
        elif landed:
            status = "ingested"
        else:
            status = "no_chunks"
        files.append({
            "relative_path": row["relative_path"],
            "extension": row["extension"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
            "file_class": row["file_class"],
            "duplicate_of": row["duplicate_of"],
            "status": status,
            "chunks_landed": landed,
        })

    manifest: dict[str, Any] = {
        "format": REFERENCE_FORMAT,
        "doc": REFERENCE_DOC,
        "ref_dir": str(ref_dir),
        "db_path": str(db_path),
        "chunking": {
            "chunk_size": CHUNK_SIZE_CHARS if chunk_size is None
                          else chunk_size,
            "overlap": CHUNK_OVERLAP_CHARS if overlap is None
                       else overlap,
        },
        "stats": dict(stats),
        "store_counts": store_counts,
        "files": files,
        "chunks": chunks,
        "library_digest": library_digest(inventory_after["files"]),
        "store_digest": store_reference_digest(db_path),
        "library_gate": builder.library_gate.as_data(),
        "merge": builder.merge_ledger.as_data(),
        "update": builder.update_ledger.as_data(),
        "poison": builder.poison_gate.as_data(),
        "embeddings": embeddings,
        "source_untouched": source_untouched,
    }
    digest = _write_manifest(db_path, manifest)

    return ReferenceReport(
        ref_dir=str(ref_dir),
        db_path=str(db_path),
        stats=dict(stats),
        store_counts=store_counts,
        files=files,
        chunks=chunks,
        library_digest=manifest["library_digest"],
        store_digest=manifest["store_digest"],
        library_gate=manifest["library_gate"],
        merge=manifest["merge"],
        update=manifest["update"],
        poison=manifest["poison"],
        embeddings=embeddings,
        source_untouched=source_untouched,
        manifest_path=str(db_path) + MANIFEST_SUFFIX,
        manifest_sha256=digest,
    )


# ── Verification + reproducibility ──────────────────────────────────────────

def verify_reference_store(
    db_path: Path | str,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Re-derive every checksummed claim; name every drift.

    Read-only. ``ok`` is True only when the manifest's
    own hash verifies, the store's chunks match the
    manifest's checksums and digest exactly, and the
    source library on disk still hashes to the filed
    per-file records.
    """
    db_path = Path(db_path)
    manifest_file = Path(manifest_path) if manifest_path is not None \
        else Path(str(db_path) + MANIFEST_SUFFIX)
    report: dict[str, Any] = {
        "format": VERIFY_FORMAT,
        "doc": REFERENCE_DOC,
        "db_path": str(db_path),
        "manifest_path": str(manifest_file),
        "ok": False,
        "problems": [],
    }
    if not manifest_file.is_file():
        report["problems"].append(f"manifest not found: {manifest_file}")
        return report
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    filed_digest = payload.pop("manifest_sha256", None)
    report["manifest_ok"] = (
        filed_digest is not None
        and _manifest_digest(payload) == filed_digest)
    if not report["manifest_ok"]:
        report["problems"].append(
            "the manifest's content hash does not verify its contents")

    # The store leg.
    chunk_problems: list[dict[str, Any]] = []
    try:
        current = chunk_records(db_path)
    except FileNotFoundError:
        report["problems"].append(f"store not found: {db_path}")
        current = []
    filed = {c["chunk_id"]: c for c in payload.get("chunks", [])}
    present = {c["chunk_id"]: c for c in current}
    for chunk_id, record in filed.items():
        if chunk_id not in present:
            chunk_problems.append(
                {"chunk_id": chunk_id, "drift": "missing"})
        elif present[chunk_id]["sha256"] != record["sha256"]:
            chunk_problems.append(
                {"chunk_id": chunk_id, "drift": "checksum_mismatch"})
    for chunk_id in present:
        if chunk_id not in filed:
            chunk_problems.append(
                {"chunk_id": chunk_id, "drift": "extra"})
    chunk_problems.sort(key=lambda p: p["chunk_id"])
    current_digest = store_reference_digest(db_path) if db_path.is_file() \
        else None
    report["chunks_checked"] = len(filed)
    report["chunk_problems"] = chunk_problems
    report["store_digest_matches"] = (
        current_digest == payload.get("store_digest")
        and not chunk_problems)
    if chunk_problems:
        report["problems"].append(
            f"{len(chunk_problems)} chunk(s) drifted from the manifest")
    elif current_digest != payload.get("store_digest"):
        report["problems"].append("the store digest does not match")

    # The source leg: the library as it stands now.
    source: dict[str, Any] = {"status": "checked", "problems": []}
    ref_dir = Path(payload.get("ref_dir", ""))
    if not ref_dir.is_dir():
        source["status"] = "source_missing"
        source["problems"].append(
            f"the filed source library is absent: {ref_dir}")
    else:
        from thoughtforge.knowledge.reference_library import (
            ReferenceLibrary,
        )

        now = {f["relative_path"]: f
               for f in ReferenceLibrary(ref_dir=ref_dir).inventory()["files"]}
        filed_files = {f["relative_path"]: f
                       for f in payload.get("files", [])}
        for name, record in filed_files.items():
            if name not in now:
                source["problems"].append(
                    {"file": name, "drift": "missing"})
            elif now[name]["sha256"] != record["sha256"]:
                source["problems"].append(
                    {"file": name, "drift": "changed"})
        for name in now:
            if name not in filed_files:
                source["problems"].append(
                    {"file": name, "drift": "added"})
    report["source"] = source
    if source["problems"]:
        report["problems"].append(
            "the source library drifted from the manifest")

    report["ok"] = bool(
        report["manifest_ok"] and report["store_digest_matches"]
        and not source["problems"])
    return report


def check_reproducibility(
    ref_dir: Path | str,
    first_db: Path | str,
    second_db: Path | str,
) -> dict[str, Any]:
    """Build one library into two stores and compare.

    Reproducible only when stats, every chunk checksum
    and the store digest all agree. Both stores (and
    their manifests) are left in place for the caller.
    """
    first = ingest_reference_library(ref_dir, first_db)
    second = ingest_reference_library(ref_dir, second_db)
    agree = (
        first.stats == second.stats
        and first.chunks == second.chunks
        and first.store_digest == second.store_digest)
    return {
        "format": REFERENCE_FORMAT,
        "reproducible": agree,
        "stats": first.stats,
        "stats_agree": first.stats == second.stats,
        "chunks_agree": first.chunks == second.chunks,
        "store_digest": first.store_digest,
        "digests_agree": first.store_digest == second.store_digest,
        "chunks": len(first.chunks),
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def reference_ingestion_problems() -> list[str]:
    """Static truth: the filed rules against the live tree."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in REFERENCE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate REFERENCE_RULES ids")
    for needed in ("skip_by_rule_counted_and_named",
                   "duplicates_chunked_once",
                   "every_file_and_chunk_checksummed",
                   "reproducible_means_three_agreements",
                   "verification_names_drift_never_repairs_it"):
        if needed not in rule_ids:
            problems.append(f"REFERENCE_RULES lost {needed}")
    if tuple(FILE_STATUSES) != (
            "ingested", "duplicate", "skipped_by_rule", "no_chunks"):
        problems.append("FILE_STATUSES drifted")

    source = Path(__file__).read_text(encoding="utf-8")
    for marker in ("ReferenceDataETL", "ReferenceLibrary",
                   "build_embeddings_at_ingestion"):
        if marker not in source:
            problems.append(
                f"reference_ingestion.py lost its {marker} leg")

    sources_src = (Path(__file__).parent / "sources.py").read_text(
        encoding="utf-8")
    for marker in ("library_gate", "LibraryLedger", "file_sha256",
                   "note_duplicate", "note_skip"):
        if marker not in sources_src:
            problems.append(
                f"ReferenceDataETL lost its TF-118 {marker} wiring "
                "(D-DATA-4 reopened)")

    # The classification is TF-058's — pinned, not copied
    # in spirit: the image set here must equal the read
    # side's filed set.
    from thoughtforge.knowledge.reference_library import (
        IMAGE_EXTENSIONS as TF058_IMAGE_EXTENSIONS,
    )

    if set(IMAGE_EXTENSIONS) != set(TF058_IMAGE_EXTENSIONS):
        problems.append("IMAGE_EXTENSIONS drifted from TF-058's set")

    from thoughtforge.etl.etl_map import ORCHESTRATION

    if "reference_ingestion" not in ORCHESTRATION.get(
            "reference_library", ""):
        problems.append(
            "TF-105's map lost its reference_library entry")
    doc = Path(__file__).resolve().parents[3] / REFERENCE_DOC
    if not doc.exists():
        problems.append(f"the write-up {REFERENCE_DOC} is missing")
    return problems


def _synthetic_library(root: Path) -> Path:
    """A tiny synthetic library (TF-017): two unique
    readable files, a byte-duplicate, an image, a CSV
    and an empty file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "a_ferries.md").write_text(
        "# Ferries\n\nThe ferry crosses the river at dawn.\n",
        encoding="utf-8")
    (root / "bells.txt").write_text(
        "The tide bell rings twice at dusk.\n", encoding="utf-8")
    (root / "zz_copy.md").write_bytes(
        (root / "a_ferries.md").read_bytes())
    (root / "photo.jpg").write_bytes(b"\xff\xd8x")
    (root / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "empty.md").write_text("", encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def reference_ingestion_live_problems(
        base: Path | None = None) -> list[str]:
    """Behavioural truth: the reference road, end to end."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(base) if base is not None else Path(tmp) / "live"
        lib = _synthetic_library(root / "lib")
        before = _snapshot(lib)
        db = root / "store.db"

        # The happy road: deduped, skipped by rule, filed.
        report = ingest_reference_library(lib, db)
        if report.stats != {"files": 2, "chunks": 2, "skipped": 4}:
            problems.append(f"reference stats drifted: {report.stats}")
        if report.store_counts.get("reference_chunks") != 2:
            problems.append(
                f"reference store counts drifted: {report.store_counts}")
        statuses = {f["relative_path"]: f["status"]
                    for f in report.files}
        if statuses != {
                "a_ferries.md": "ingested", "bells.txt": "ingested",
                "zz_copy.md": "duplicate",
                "photo.jpg": "skipped_by_rule",
                "table.csv": "skipped_by_rule",
                "empty.md": "no_chunks"}:
            problems.append(f"file statuses drifted: {statuses}")
        gate = report.library_gate
        if [s["file"] for s in gate["skipped_by_rule"]] != [
                "photo.jpg", "table.csv"] or \
                gate["duplicates"] != [{
                    "file": "zz_copy.md",
                    "duplicate_of": "a_ferries.md",
                    "sha256": file_sha256(lib / "zz_copy.md")}]:
            problems.append(f"library gate drifted: {gate}")
        if report.source_untouched is not True or \
                _snapshot(lib) != before:
            problems.append("the library was touched by ingestion")
        if report.embeddings.get("status") != "not_run":
            problems.append("the embedding leg ran without an encoder")

        # The manifest: present, content-free, hash true.
        manifest_path = Path(report.manifest_path)
        if not manifest_path.is_file():
            problems.append("the reference manifest was not written")
        else:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8"))
            filed_digest = payload.pop("manifest_sha256")
            if filed_digest != report.manifest_sha256 or \
                    _manifest_digest(payload) != filed_digest:
                problems.append(
                    "the manifest's hash does not verify its contents")
            if any("content" in chunk for chunk in payload["chunks"]):
                problems.append("the manifest carries chunk content")

        # Verification on the untouched store: all true.
        verdict = verify_reference_store(db)
        if verdict["ok"] is not True:
            problems.append(f"a clean store did not verify: {verdict}")

        # Reproducibility: a second store, the same digest.
        repro = check_reproducibility(lib, root / "r1.db", root / "r2.db")
        if repro["reproducible"] is not True:
            problems.append(f"ingestion not reproducible: {repro}")

        # A re-run into the same store leaves the digest.
        rerun = ingest_reference_library(lib, db)
        if rerun.store_digest != report.store_digest:
            problems.append("a re-run changed the store digest")

        # Tampering is named, never silently repaired.
        import sqlite3 as _sqlite3

        conn = _sqlite3.connect(str(db))
        try:
            conn.execute(
                "UPDATE reference_chunks SET content='TAMPERED'"
                " WHERE source_file='a_ferries.md'")
            conn.commit()
        finally:
            conn.close()
        verdict = verify_reference_store(db)
        if verdict["ok"] is not False or verdict["chunk_problems"] != [
                {"chunk_id": "a_ferries.md#0",
                 "drift": "checksum_mismatch"}]:
            problems.append(
                f"tampering was not named exactly: {verdict}")

        # A changed source file is named too (fresh store).
        lib2 = _synthetic_library(root / "lib2")
        db2 = root / "store2.db"
        ingest_reference_library(lib2, db2)
        (lib2 / "bells.txt").write_text(
            "The tide bell rings three times.\n", encoding="utf-8")
        verdict = verify_reference_store(db2)
        if verdict["ok"] is not False or verdict["source"][
                "problems"] != [{"file": "bells.txt",
                                  "drift": "changed"}]:
            problems.append(
                f"a changed source was not named: {verdict['source']}")

        # A missing library raises before any store exists.
        missing_db = root / "missing.db"
        try:
            ingest_reference_library(root / "absent-lib", missing_db)
            problems.append("a missing library was ingested")
        except FileNotFoundError as exc:
            if "absent-lib" not in str(exc):
                problems.append(f"missing-library refusal unnamed: {exc}")
        if missing_db.exists():
            problems.append("a missing-library refusal created its store")

        # Retrieval: the ingested library is searchable,
        # under the keeper's name only (TF-058's road).
        from thoughtforge.knowledge.forge import KnowledgeForge

        rows = KnowledgeForge(db_path=db).sql_retrieve("ferry")
        # (db was tampered above for the verify probe; the
        # FTS index still carries the landed text.)
        if not rows or rows[0]["source_file"] != "a_ferries.md":
            problems.append(f"reference retrieval drifted: {rows[:1]}")
    return problems


def reference_ingestion_as_data() -> dict[str, Any]:
    """The reference-ingestion filing as plain data."""
    return {
        "format": REFERENCE_FORMAT,
        "doc": REFERENCE_DOC,
        "chunk_fields": list(CHUNK_FIELDS),
        "file_statuses": list(FILE_STATUSES),
        "rules": [dict(r) for r in REFERENCE_RULES],
    }


__all__ = [
    "CHUNK_FIELDS",
    "FILE_STATUSES",
    "IMAGE_EXTENSIONS",
    "LEDGER_FORMAT",
    "MANIFEST_SUFFIX",
    "REFERENCE_DOC",
    "REFERENCE_FORMAT",
    "REFERENCE_RULES",
    "VERIFY_FORMAT",
    "LibraryLedger",
    "ReferenceReport",
    "check_reproducibility",
    "chunk_checksum",
    "chunk_records",
    "classify_reference_extension",
    "file_sha256",
    "ingest_reference_library",
    "library_digest",
    "reference_ingestion_as_data",
    "reference_ingestion_live_problems",
    "reference_ingestion_problems",
    "report_as_data",
    "store_reference_digest",
    "store_reference_rows",
    "verify_reference_store",
]
