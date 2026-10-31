"""
Reference-library binding — the library as a store source (TF-058).

The roadmap's rule: *the knowledge_reference library readable
as a store source (read-only, canon-respecting).*

The measured starting state, before this slice (probed this
tick, keeper 08:32, on the TF-017 synthetic library):

  * **The library was readable only through the knowledge
    DB — and only after two separate write steps.** No
    module could read, list, or search the files directly:
    the one reader was ``ReferenceDataETL.build()``, a
    *write* pipeline that chunks files into the
    ``reference_chunks`` table. ``KnowledgeForge`` reads
    that table, never the files. Probed: with the synthetic
    library on disk and an empty (schema-only) DB, the
    forge returned 0 records — the library, present, was
    invisible.
  * **Even the DB road was broken for the reference
    command's own output (D-REF-1, found + FIXED in this
    slice).** ``reference_chunks`` is searched through an
    external-content FTS5 table, which only a
    ``rebuild`` repopulates — and neither
    ``ReferenceDataETL.build()`` nor the
    ``forge_memory reference`` command ever rebuilt it
    (only the *embeddings* step did). Probed: after a
    plain ETL build of the synthetic library (3 files,
    6 chunks), ``KnowledgeForge.sql_retrieve`` returned
    0 rows for every query tried. The fix: ``build()``
    now rebuilds the FTS indexes before returning, so an
    ingested library is the searchable library.
  * **Skipping was by accident, not by rule (D-DATA-4).**
    The ETL's supported set {.md, .json, .jsonl, .yaml,
    .yml, .txt} silently ``continue``d past everything
    else: probed, a .jpg and a .csv added to the library
    changed no counter (stats ``skipped`` stayed 0) and
    were named nowhere. TF-007 counted the real cost:
    225 persona images / 152.9 MB live *inside* the real
    library and were skipped by that same silence.
  * **Duplicates were double-counted (TF-007 finding 3).**
    Probed: a byte-identical copy of the synthetic .md
    under a second name ingested as a second full set of
    chunks (4 files / 10 chunks for what is 3 files / 6
    chunks of content) — the DB's uniqueness is
    (source_file, chunk_index), so identical content in
    two files is two records. TF-109 owns dedup for
    ingestion/merge generally; this slice's read side
    dedups by file hash, the rule TF-007 named for
    TF-058/TF-118.

What this slice builds — ``ReferenceLibrary``, a direct,
read-only source over one library directory:

  * ``FILE_CLASSES`` / ``SUPPORTED_EXTENSIONS`` — the
    classification *as data*: ``readable`` is exactly the
    ETL's supported text set (the checker pins the two
    sets equal, so the binding can never drift from what
    ingestion would read); ``image`` and ``other`` are
    skipped *by rule* — classified, counted, and named
    in every report, never silent.
  * ``inventory()`` — metadata only: per file, its
    relative path, extension, byte size, SHA-256, class,
    and (for readable files) its duplicate keeper. No
    file content is read into the report beyond hashing,
    and no content is copied anywhere: canon is described,
    never moved (TF-007's census discipline, as a live
    API). A missing directory is a verdict in the report
    (``exists: False``), not an exception — this VM's own
    sparse checkout carries no library, and that absence
    must read as absence, not as a crash.
  * ``chunks()`` — the library chunked by the ETL's *own*
    ``_load_and_chunk`` (the binding instantiates the ETL
    class only to borrow that pure method; it never opens
    its DB path), so a chunk here is byte-for-byte the
    chunk ingestion would land. Readable files that are
    byte-duplicates of an earlier (sorted-order) file are
    skipped as duplicates, named in ``chunk_report()``;
    readable files yielding no chunks (empty, unreadable)
    are counted there too. Chunk identity is
    deterministic: ``<source_file>#<chunk_index>``.
  * ``search()`` — a deterministic term-overlap search
    returning a ``MemoryActivationBundle`` in the forge's
    own reference shape: each hit is an ``ActivatedRecord``
    of type ``reference_chunk`` whose raw dict carries
    exactly the forge SQL pass's fields, so TF-048 derives
    its citation payload by construction — marker
    ``ref:<source_file>#<chunk_index>``, locator the
    source file (TF-042: the file *is* a reference chunk's
    provenance). Scoring is stated, not hidden: the share
    of distinct query terms (the forge's key-term
    extraction) present in the chunk's title + content +
    tags; ties break by (source_file, chunk_index), so a
    search is reproducible run to run. No match is an
    empty bundle, never a fabricated hit.
  * **Read-only, proven.** The binding opens files only
    for reading, holds no DB handle, and writes nothing —
    pinned byte-for-byte (file set + contents) across
    inventory / chunks / search in the checker and tests.

Stated boundaries:

  * This is the READ side. Writing the library into the
    knowledge DB remains the ETL's (TF-118 owns its
    reproducible, checksummed ingestion; D-REF-1's fix
    above is the one write-side change this slice makes,
    because it was the binding's own road that was broken).
    Subset sizing for small profiles is TF-110's.
  * Search quality is a stated term-overlap rule, not the
    forge's bm25/vector ranking — precision of THIS source
    against a known-answer corpus is measured by its
    fixtures here only for identity (the right chunk is
    returned, cited, and reproducible); ranking research
    belongs to Part VII.
  * Proof grade is FIXTURE (the TF-017 synthetic library +
    synthetic temp libraries on this VM). The binding over
    the real canon library is NOT PROVEN — the real
    library is not in this sparse checkout, and no canon
    file was opened by this slice — never claimed.

The human-readable write-up is
``docs/architecture/REFERENCE_LIBRARY_BINDING.md``.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thoughtforge.etl.sources import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, ReferenceDataETL
from thoughtforge.knowledge.models import ActivatedRecord, MemoryActivationBundle
from thoughtforge.utils.paths import get_knowledge_reference_dir

# ── The classification, as data ──────────────────────────────────────────────

#: Exactly the text extensions the reference ETL reads —
#: the binding's readable set, pinned equal to the ETL's
#: by ``reference_library_problems()``.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml",
)

#: Image extensions: present in the real library in bulk
#: (TF-007: 225 .jpg persona images), never readable text.
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
)

#: One row per file class. ``readable`` is the only class
#: the binding chunks or searches; the others exist so a
#: skip is a named rule with a count, never a silence.
FILE_CLASSES: tuple[dict[str, Any], ...] = (
    {"name": "readable", "extensions": SUPPORTED_EXTENSIONS,
     "rule": "chunked and searchable through the binding"},
    {"name": "image", "extensions": IMAGE_EXTENSIONS,
     "rule": "skipped by rule — imagery is not knowledge text (TF-007)"},
    {"name": "other", "extensions": (),
     "rule": "skipped by rule — outside the ETL's supported text set"},
)

#: The binding's rules, as data.
BINDING_RULES: tuple[dict[str, Any], ...] = (
    {"name": "read_only",
     "rule": "the binding only reads: no file in the library is "
             "created, modified, or deleted, and no database is opened"},
    {"name": "canon_respecting",
     "rule": "inventory is metadata only (path, size, hash, class); "
             "content leaves the library only as cited search hits "
             "carrying their source file (TF-048)"},
    {"name": "skip_by_rule",
     "rule": "every non-readable file is classified and counted "
             "(image / other) — D-DATA-4's read-side share"},
    {"name": "dedup_by_hash",
     "rule": "readable files with an identical SHA-256 to an earlier "
             "file (sorted order) are duplicates: inventoried, named, "
             "and never chunked twice"},
    {"name": "chunking_is_the_etls",
     "rule": "chunks come from ReferenceDataETL._load_and_chunk with "
             "the ETL's own chunk size / overlap — the binding's "
             "chunk is ingestion's chunk"},
    {"name": "missing_is_a_verdict",
     "rule": "a missing library directory reports exists=False and "
             "searches empty; it never raises"},
)


def classify_extension(extension: str) -> str:
    """The FILE_CLASSES name for one extension (lowercased)."""
    ext = extension.lower()
    for row in FILE_CLASSES:
        if ext in row["extensions"]:
            return str(row["name"])
    return "other" if ext not in IMAGE_EXTENSIONS else "image"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ── Files and chunks ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReferenceFile:
    """One library file, metadata only — never its content."""

    relative_path: str
    extension: str
    size_bytes: int
    sha256: str
    file_class: str             # a FILE_CLASSES name
    duplicate_of: str = ""      # keeper's relative path, "" if unique/kept

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "file_class": self.file_class,
            "duplicate_of": self.duplicate_of,
        }


@dataclass(frozen=True)
class ReferenceChunk:
    """One chunk of a readable file, in the ETL's own chunking."""

    chunk_id: str               # "<source_file>#<chunk_index>"
    source_file: str
    chunk_index: int
    title: str
    content: str
    tags: str
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "word_count": self.word_count,
        }

    def raw_record(self, score: float) -> dict[str, Any]:
        """The forge SQL reference pass's raw shape, so the
        citation derivation (TF-048) sees a familiar record."""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "title": self.title,
            "content": self.content[:500],
            "tags": self.tags,
            "fts_score": score,
            "source": "reference",
            "record_type": "reference_chunk",
        }


# ── The binding ──────────────────────────────────────────────────────────────


class ReferenceLibrary:
    """A read-only store source over one library directory."""

    def __init__(self, ref_dir: Path | None = None) -> None:
        self.ref_dir = Path(ref_dir) if ref_dir is not None \
            else get_knowledge_reference_dir()

    @property
    def exists(self) -> bool:
        return self.ref_dir.is_dir()

    # -- inventory -----------------------------------------------------------

    def files(self) -> list[ReferenceFile]:
        """Every file under the library, sorted by relative
        path, classified and hashed. Duplicate detection runs
        over readable files only (an image duplicated is a
        census fact for TF-007/TF-109, not a chunking risk)."""
        if not self.exists:
            return []
        keepers: dict[str, str] = {}
        out: list[ReferenceFile] = []
        for path in sorted(self.ref_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(self.ref_dir))
            sha = _sha256(path)
            file_class = classify_extension(path.suffix)
            duplicate_of = ""
            if file_class == "readable":
                if sha in keepers:
                    duplicate_of = keepers[sha]
                else:
                    keepers[sha] = rel
            out.append(ReferenceFile(
                relative_path=rel,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                sha256=sha,
                file_class=file_class,
                duplicate_of=duplicate_of,
            ))
        return out

    def inventory(self) -> dict[str, Any]:
        """One JSON-safe, metadata-only report on the library."""
        files = self.files()
        by_class: dict[str, int] = {}
        bytes_by_class: dict[str, int] = {}
        for f in files:
            by_class[f.file_class] = by_class.get(f.file_class, 0) + 1
            bytes_by_class[f.file_class] = \
                bytes_by_class.get(f.file_class, 0) + f.size_bytes
        duplicates = [
            {"file": f.relative_path, "duplicate_of": f.duplicate_of,
             "sha256": f.sha256}
            for f in files if f.duplicate_of
        ]
        return {
            "root": str(self.ref_dir),
            "exists": self.exists,
            "total_files": len(files),
            "total_bytes": sum(f.size_bytes for f in files),
            "files_by_class": by_class,
            "bytes_by_class": bytes_by_class,
            "duplicates": duplicates,
            "files": [f.to_dict() for f in files],
        }

    # -- chunks --------------------------------------------------------------

    def _kept_files(self) -> list[ReferenceFile]:
        return [f for f in self.files()
                if f.file_class == "readable" and not f.duplicate_of]

    def chunks(self) -> list[ReferenceChunk]:
        """Every chunk of every kept readable file, in
        (source_file, chunk_index) order, chunked by the
        ETL's own method at its own size/overlap."""
        # The ETL is instantiated ONLY to borrow its pure
        # _load_and_chunk; build() is never called, its
        # db_path never opened (the placeholder is a name,
        # not a file this code touches).
        etl = ReferenceDataETL(db_path=Path("reference-binding-unused.db"))
        out: list[ReferenceChunk] = []
        for f in self._kept_files():
            path = self.ref_dir / f.relative_path
            try:
                loaded = etl._load_and_chunk(
                    path, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
            except (OSError, UnicodeDecodeError):
                loaded = []
            for idx, (title, content, tags) in enumerate(loaded):
                out.append(ReferenceChunk(
                    chunk_id=f"{f.relative_path}#{idx}",
                    source_file=f.relative_path,
                    chunk_index=idx,
                    title=title,
                    content=content,
                    tags=tags,
                    word_count=len(content.split()),
                ))
        return out

    def chunk_report(self) -> dict[str, Any]:
        """How the library chunks, with every skip named:
        by class, by duplication, by yielding no chunks."""
        files = self.files()
        chunks = self.chunks()
        chunked_files = {c.source_file for c in chunks}
        kept = self._kept_files()
        return {
            "chunks": len(chunks),
            "files_chunked": len(chunked_files),
            "files_readable": sum(
                1 for f in files if f.file_class == "readable"),
            "files_duplicate": sum(1 for f in files if f.duplicate_of),
            "files_skipped_image": sum(
                1 for f in files if f.file_class == "image"),
            "files_skipped_other": sum(
                1 for f in files if f.file_class == "other"),
            "files_no_chunks": sorted(
                f.relative_path for f in kept
                if f.relative_path not in chunked_files),
        }

    def get_chunk(self, chunk_id: str) -> ReferenceChunk | None:
        """One chunk by its deterministic id, or None."""
        for chunk in self.chunks():
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    # -- search --------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 10
    ) -> MemoryActivationBundle:
        """Deterministic term-overlap search as a bundle.

        Score = the share of the query's distinct key terms
        present in the chunk's title + content + tags
        (tokenised whole-word, folded — TF-060). Only chunks
        matching at least one term are returned, best first,
        ties by (source_file, chunk_index); the bundle's
        bookkeeping matches the forge's (total == records,
        confidence == mean score)."""
        from thoughtforge.knowledge.forge import KnowledgeForge

        terms = KnowledgeForge._extract_key_terms(query, max_terms=8)
        if not terms or top_k <= 0 or not self.exists:
            return MemoryActivationBundle(
                activated_records=[], total_retrieved=0,
                retrieval_confidence=0.0)
        # TF-060: matching is multilingual — folded tokens
        # (casefold, spaceless-script bigrams) with the
        # diacritic-stripped form auxiliary, as in the
        # personal-store searches. English scores are
        # unchanged (a folded English token is its
        # lowercase, and 1-of-2 still scores exactly 0.5).
        from thoughtforge.knowledge.multilingual import (
            match_tokens,
            terms_matching,
        )

        scored: list[tuple[float, ReferenceChunk]] = []
        for chunk in self.chunks():
            haystack = match_tokens(
                f"{chunk.title} {chunk.content} {chunk.tags}")
            matched = terms_matching(terms, haystack)
            if matched:
                scored.append((len(matched) / len(terms), chunk))
        scored.sort(key=lambda item: (
            -item[0], item[1].source_file, item[1].chunk_index))
        activated: list[ActivatedRecord] = []
        for score, chunk in scored[:top_k]:
            cue = KnowledgeForge._build_cue(chunk.raw_record(score))
            activated.append(ActivatedRecord(
                record_id=chunk.chunk_id,
                record_type="reference_chunk",
                score=round(score, 4),
                cue=cue,
                raw=chunk.raw_record(score),
            ))
        confidence = round(
            sum(r.score for r in activated) / len(activated), 4
        ) if activated else 0.0
        return MemoryActivationBundle(
            activated_records=activated,
            total_retrieved=len(activated),
            retrieval_confidence=confidence,
        )


# ── The live checker ─────────────────────────────────────────────────────────


def _library_snapshot(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        return {}
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def reference_library_problems() -> list[str]:
    """Re-derive TF-058's claims from the live tree; [] = holds.

    Pins: the readable set equals the ETL's supported set;
    the ETL build carries the D-REF-1 FTS rebuild; chunking
    through the binding is byte-identical to the ETL's own
    per-file chunking on the synthetic library; inventory is
    metadata-only; duplicates and non-readable files are
    skipped by named rule; search returns the right chunk
    with its TF-048 citation and is reproducible; every leg
    is read-only byte-for-byte; a missing library is a
    verdict, not an exception."""
    problems: list[str] = []

    # The readable set is the ETL's supported set.
    build_src = inspect.getsource(ReferenceDataETL.build)
    etl_set = set(re.findall(r'"(\.\w+)"', build_src)) & {
        ".md", ".json", ".jsonl", ".yaml", ".yml", ".txt",
        ".csv", ".rst", ".py",
    }
    if set(SUPPORTED_EXTENSIONS) != etl_set:
        problems.append(
            f"readable set {sorted(SUPPORTED_EXTENSIONS)} != "
            f"the ETL's supported set {sorted(etl_set)}")
    if [r["name"] for r in FILE_CLASSES] != ["readable", "image", "other"]:
        problems.append("FILE_CLASSES drifted from readable/image/other")
    if classify_extension(".jpg") != "image" \
            or classify_extension(".md") != "readable" \
            or classify_extension(".csv") != "other":
        problems.append("classify_extension misclassifies a pinned case")

    # D-REF-1: the ETL build rebuilds the FTS indexes.
    if "rebuild_fts_indexes" not in build_src:
        problems.append("D-REF-1 reopened: ReferenceDataETL.build "
                        "no longer rebuilds the FTS indexes")

    # The synthetic library, through the binding.
    fixtures = Path(__file__).parents[3] / "tests" / "fixtures" \
        / "synthetic_reference"
    if not fixtures.is_dir():
        problems.append("synthetic reference fixtures not found")
        return problems
    before = _library_snapshot(fixtures)
    library = ReferenceLibrary(ref_dir=fixtures)
    inventory = library.inventory()
    if not inventory["exists"] or inventory["total_files"] != 3:
        problems.append(f"synthetic inventory wrong: {inventory['total_files']}")
    if inventory["files_by_class"] != {"readable": 3}:
        problems.append(f"synthetic classes wrong: {inventory['files_by_class']}")
    if any("content" in row for row in inventory["files"]):
        problems.append("inventory carries file content (canon rule broken)")
    chunks = library.chunks()
    if len(chunks) != 6:
        problems.append(f"synthetic chunks {len(chunks)} != the ETL's 6")
    etl = ReferenceDataETL(db_path=Path("reference-binding-unused.db"))
    for f in library._kept_files():
        expected = etl._load_and_chunk(
            fixtures / f.relative_path, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        got = [c for c in chunks if c.source_file == f.relative_path]
        if [(c.title, c.content, c.tags) for c in got] != expected:
            problems.append(f"chunking drifted from the ETL for {f.relative_path}")
    hits = library.search("ferry crossing")
    if not hits.activated_records:
        problems.append("search for 'ferry crossing' returned nothing")
    else:
        top = hits.activated_records[0]
        if top.citation is None or not top.citation.marker.startswith("ref:"):
            problems.append(f"search hit citation wrong: {top.citation}")
        elif top.citation.locator != top.raw["source_file"]:
            problems.append("search hit locator is not its source file")
    again = library.search("ferry crossing")
    if [r.record_id for r in again.activated_records] != \
            [r.record_id for r in hits.activated_records]:
        problems.append("search is not reproducible")
    if library.search("zzqx unwritten").activated_records:
        problems.append("a no-match search fabricated a hit")
    if _library_snapshot(fixtures) != before:
        problems.append("the binding wrote to the library (read-only broken)")

    # Duplicates + skips by rule, on a synthetic temp library.
    with tempfile.TemporaryDirectory(prefix="tf058-") as tmp:
        root = Path(tmp)
        (root / "a.md").write_text("# Ferry\n\nthe ferry crosses.\n",
                                   encoding="utf-8")
        (root / "b.md").write_bytes((root / "a.md").read_bytes())
        (root / "pic.jpg").write_bytes(b"\xff\xd8x")
        (root / "table.csv").write_text("a,b\n", encoding="utf-8")
        temp_lib = ReferenceLibrary(ref_dir=root)
        report = temp_lib.chunk_report()
        if report["files_duplicate"] != 1 \
                or report["files_skipped_image"] != 1 \
                or report["files_skipped_other"] != 1:
            problems.append(f"temp library skip/dedup counts wrong: {report}")
        if len(temp_lib.chunks()) != 1:
            problems.append("duplicate content was chunked twice")

    # A missing library is a verdict.
    missing = ReferenceLibrary(ref_dir=Path("/nonexistent-tf058-lib"))
    if missing.inventory()["exists"] is not False \
            or missing.chunks() != [] \
            or missing.search("ferry").total_retrieved != 0:
        problems.append("a missing library is not a clean verdict")
    return problems


__all__ = [
    "BINDING_RULES",
    "FILE_CLASSES",
    "IMAGE_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "ReferenceChunk",
    "ReferenceFile",
    "ReferenceLibrary",
    "classify_extension",
    "reference_library_problems",
]
