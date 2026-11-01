"""
Alternative knowledge source ingestion pipelines for ThoughtForge.

Sources:
  - ConceptNet 5       — commonsense relations (CSV)
  - GeoNames           — geographic entities (tab-delimited)
  - DBpedia            — Wikipedia-extracted entities (TTL/JSON)
  - Reference Data     — built-in .md / .json / .jsonl / .yaml files (40 files)

TF-108 (docs/architecture/SCHEMA_VALIDATION_GATE.md; the
gate as data lives in etl/schema_gate.py): every record
the ConceptNet, GeoNames and DBpedia builders read is
judged by the schema gate before insertion — a malformed
record is refused whole with reasons recorded in the
builder's ``schema_gate`` ledger and counted in
``skipped``, never a crash and never a silently accepted
or silently dropped row. Stated filters (language,
population, feature class) remain filter skips over
schema-valid records, not refusals.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from tqdm import tqdm

from thoughtforge.etl.schema import get_connection, initialize_schema, rebuild_fts_indexes
from thoughtforge.etl.schema_gate import SchemaGate
from thoughtforge.utils.paths import get_knowledge_db_path, get_knowledge_reference_dir

logger = logging.getLogger(__name__)

CHUNK_SIZE_CHARS = 1000   # target chunk size for reference documents
CHUNK_OVERLAP_CHARS = 100  # overlap between chunks


# ── ConceptNet 5 ───────────────────────────────────────────────────────────────

class ConceptNetETL:
    """
    Ingests ConceptNet 5 assertions CSV into the conceptnet_edges table.

    Download: https://conceptnet.io/downloads/
    File:     conceptnet-assertions-5.7.0.csv.gz  (~1.3GB compressed)

    CSV format: uri | relation | concept1 | concept2 | metadata_json
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_knowledge_db_path()
        # TF-108: the schema gate's ledger for the most recent build.
        self.schema_gate = SchemaGate()

    def build(
        self,
        csv_path: Path | str,
        languages: list[str] | None = None,
        batch_size: int = 5000,
        limit: int | None = None,
    ) -> dict[str, int]:
        """
        Stream the ConceptNet assertions CSV into the knowledge DB.
        languages: if set, only include edges where both concepts match these language codes.
        """
        csv_path = Path(csv_path)
        languages = set(languages) if languages else {"en"}
        self.schema_gate = SchemaGate()

        if not csv_path.exists():
            raise FileNotFoundError(f"ConceptNet CSV not found: {csv_path}")

        initialize_schema(self.db_path)
        conn = get_connection(self.db_path)
        conn.execute("PRAGMA synchronous = OFF")

        stats = {"inserted": 0, "skipped": 0}
        batch: list[tuple[Any, ...]] = []

        open_fn: Any
        if csv_path.suffix == ".gz":
            import gzip
            open_fn = gzip.open
        else:
            open_fn = open

        try:
            with open_fn(str(csv_path), "rt", encoding="utf-8") as f:
                for line in tqdm(f, desc="ConceptNet ETL", unit=" edges", mininterval=5.0):
                    if limit and stats["inserted"] >= limit:
                        break
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 5:
                        # TF-108 (D-GATE-3): a short line is a
                        # refusal with a reason, not a bare skip.
                        self.schema_gate.refuse("conceptnet", [
                            "too_few_fields:record: expected at least "
                            f"5 tab-delimited fields, got {len(parts)}"])
                        stats["skipped"] += 1
                        continue

                    uri, relation, concept1, concept2, meta_str = parts[:5]

                    # TF-108 gate (D-GATE-1/2): the record is
                    # judged before anything is parsed into a
                    # row — unparseable metadata no longer
                    # defaults a weight, a non-numeric weight
                    # no longer crashes the build, and empty
                    # concepts/relations are refused whole.
                    try:
                        meta_parsed: Any = json.loads(meta_str)
                    except json.JSONDecodeError:
                        meta_parsed = None
                    reasons = self.schema_gate.check("conceptnet", {
                        "uri": uri, "relation": relation,
                        "concept1": concept1, "concept2": concept2,
                        "metadata_parsed": meta_parsed,
                    }, ref=uri)
                    if reasons:
                        logger.warning(
                            "ConceptNet record refused at the gate "
                            "(%s): %s", uri, "; ".join(reasons))
                        stats["skipped"] += 1
                        continue
                    meta = meta_parsed

                    # Language filter — a stated filter over
                    # schema-valid records, never a gate refusal.
                    c1_lang = self._extract_lang(concept1)
                    c2_lang = self._extract_lang(concept2)
                    if not (c1_lang in languages and c2_lang in languages):
                        stats["skipped"] += 1
                        continue

                    # The gate proved the weight numeric when
                    # present; the default is the filed 1.0.
                    weight = float(meta.get("weight", 1.0))

                    rel_name = relation.split("/")[-1] if "/" in relation else relation
                    c1_name = self._uri_to_text(concept1)
                    c2_name = self._uri_to_text(concept2)

                    batch.append((uri, c1_name, rel_name, c2_name, weight, c1_lang))
                    stats["inserted"] += 1

                    if len(batch) >= batch_size:
                        self._flush(conn, batch)
                        batch.clear()

            if batch:
                self._flush(conn, batch)

            conn.commit()

        finally:
            conn.close()

        logger.info("ConceptNet ETL complete: %s", stats)
        return stats

    def build_from_reference(self, db_path: Path | None = None) -> int:
        """
        Load any .json ConceptNet-style files found in data/knowledge_reference/.
        These are flat arrays of {concept1, relation, concept2} objects.
        """
        db_path = db_path or self.db_path
        ref_dir = get_knowledge_reference_dir()
        total = 0
        self.schema_gate = SchemaGate()

        conn = get_connection(db_path)
        try:
            for f in sorted(ref_dir.glob("*.json")):
                try:
                    with f.open("r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    if not isinstance(data, list):
                        # TF-108 (D-GATE-3): a non-array file is
                        # refused by name, not silently passed.
                        self.schema_gate.refuse("conceptnet_reference", [
                            "not_a_record:file: expected a JSON array "
                            "of edge objects"], ref=f.name)
                        continue
                    batch = []
                    for item in data:
                        if not isinstance(item, dict):
                            self.schema_gate.refuse(
                                "conceptnet_reference", [
                                    "not_a_record:record: expected "
                                    "object, got "
                                    f"{type(item).__name__}"])
                            continue
                        c1 = item.get("concept1") or item.get("subject") or item.get("start", "")
                        rel = item.get("relation") or item.get("rel") or item.get("predicate", "")
                        c2 = item.get("concept2") or item.get("object") or item.get("end", "")
                        # TF-108 gate: field-less items are
                        # refused with reasons, never dropped
                        # by a bare continue (D-GATE-3).
                        reasons = self.schema_gate.check(
                            "conceptnet_reference", {
                                "concept1": c1 if isinstance(c1, str)
                                else "",
                                "relation": rel if isinstance(rel, str)
                                else "",
                                "concept2": c2 if isinstance(c2, str)
                                else "",
                            })
                        if reasons:
                            continue
                        if c1 and rel and c2:
                            uri = f"ref/{f.stem}/{c1}/{rel}/{c2}"[:512]
                            batch.append((uri, str(c1), str(rel), str(c2), 1.0, "en"))
                    if batch:
                        self._flush(conn, batch)
                        total += len(batch)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Skipping %s: %s", f.name, e)

            conn.commit()
        finally:
            conn.close()

        logger.info("Reference JSON concept edges loaded: %d", total)
        return total

    @staticmethod
    def _extract_lang(uri: str) -> str:
        parts = uri.split("/")
        return parts[2] if len(parts) >= 3 else "en"

    @staticmethod
    def _uri_to_text(uri: str) -> str:
        parts = uri.split("/")
        text = parts[-1] if parts else uri
        return text.replace("_", " ")

    @staticmethod
    def _flush(conn: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
        conn.executemany(
            "INSERT OR IGNORE INTO conceptnet_edges "
            "(uri, concept1, relation, concept2, weight, language) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )


# ── GeoNames ───────────────────────────────────────────────────────────────────

class GeoNamesETL:
    """
    Ingests GeoNames allCountries.txt (or a country subset) into geonames_places.

    Download: https://download.geonames.org/export/dump/
    File:     allCountries.zip  (~1.5GB uncompressed)

    Tab-delimited: geonames_id, name, ascii_name, alternateNames, lat, lon,
                   feature_class, feature_code, country_code, cc2,
                   admin1, admin2, admin3, admin4, population, elevation,
                   dem, timezone, modification_date
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_knowledge_db_path()
        # TF-108: the schema gate's ledger for the most recent build.
        self.schema_gate = SchemaGate()

    def build(
        self,
        txt_path: Path | str,
        batch_size: int = 5000,
        min_population: int = 0,
        feature_classes: set[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, int]:
        """
        Stream the GeoNames .txt file into the knowledge DB.
        feature_classes: e.g. {'P', 'A'} for populated places and admin areas.
        """
        txt_path = Path(txt_path)
        self.schema_gate = SchemaGate()
        if not txt_path.exists():
            raise FileNotFoundError(f"GeoNames file not found: {txt_path}")

        initialize_schema(self.db_path)
        conn = get_connection(self.db_path)
        conn.execute("PRAGMA synchronous = OFF")

        stats = {"inserted": 0, "skipped": 0}
        batch: list[tuple[Any, ...]] = []

        try:
            with txt_path.open("r", encoding="utf-8") as f:
                for line in tqdm(f, desc="GeoNames ETL", unit=" places", mininterval=5.0):
                    if limit and stats["inserted"] >= limit:
                        break
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 19:
                        # TF-108 (D-GATE-3): refused with a reason.
                        self.schema_gate.refuse("geonames", [
                            "too_few_fields:record: expected 19 "
                            f"tab-delimited fields, got {len(parts)}"])
                        stats["skipped"] += 1
                        continue

                    # TF-108 gate (D-GATE-2): the raw fields
                    # are judged before parsing — out-of-range
                    # coordinates, an empty name and
                    # unparseable numbers are refused whole,
                    # never landed and never a crash.
                    reasons = self.schema_gate.check("geonames", {
                        "geonames_id": parts[0], "name": parts[1],
                        "latitude": parts[4], "longitude": parts[5],
                        "population": parts[14], "elevation": parts[15],
                    }, ref=parts[0])
                    if reasons:
                        logger.warning(
                            "GeoNames record refused at the gate "
                            "(%s): %s", parts[0], "; ".join(reasons))
                        stats["skipped"] += 1
                        continue

                    try:
                        gid = int(parts[0])
                        name = parts[1]
                        ascii_name = parts[2]
                        lat = float(parts[4]) if parts[4] else None
                        lon = float(parts[5]) if parts[5] else None
                        feat_class = parts[6]
                        feat_code = parts[7]
                        country = parts[8]
                        admin1 = parts[10]
                        admin2 = parts[11]
                        population = int(parts[14]) if parts[14] else 0
                        elevation = int(parts[15]) if parts[15] else None
                        timezone = parts[17]
                    except (ValueError, IndexError):
                        stats["skipped"] += 1
                        continue

                    if population < min_population:
                        stats["skipped"] += 1
                        continue
                    if feature_classes and feat_class not in feature_classes:
                        stats["skipped"] += 1
                        continue

                    batch.append((
                        gid, name, ascii_name, lat, lon, feat_class, feat_code,
                        country, population, elevation, timezone, admin1, admin2,
                    ))
                    stats["inserted"] += 1

                    if len(batch) >= batch_size:
                        self._flush(conn, batch)
                        batch.clear()

            if batch:
                self._flush(conn, batch)

            conn.commit()

        finally:
            conn.close()

        logger.info("GeoNames ETL complete: %s", stats)
        return stats

    @staticmethod
    def _flush(conn: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
        conn.executemany(
            "INSERT OR REPLACE INTO geonames_places "
            "(geonames_id, name, ascii_name, latitude, longitude, feature_class, feature_code, "
            "country_code, population, elevation, timezone, admin1_code, admin2_code) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )


# ── DBpedia ────────────────────────────────────────────────────────────────────

class DBpediaETL:
    """
    Ingests DBpedia JSON abstracts dump into dbpedia_entities.

    Download: https://downloads.dbpedia.org/repo/dbpedia/text/
    File:     long-abstracts_lang=en.ttl.bz2 or JSON export

    Supports simple line-by-line JSON: {"uri": "...", "label": "...", "abstract": "..."}
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_knowledge_db_path()
        # TF-108: the schema gate's ledger for the most recent build.
        self.schema_gate = SchemaGate()

    def build(
        self,
        json_path: Path | str,
        batch_size: int = 2000,
        limit: int | None = None,
    ) -> dict[str, int]:
        """Ingest a newline-delimited JSON file of DBpedia abstracts."""
        json_path = Path(json_path)
        self.schema_gate = SchemaGate()
        if not json_path.exists():
            raise FileNotFoundError(f"DBpedia file not found: {json_path}")

        initialize_schema(self.db_path)
        conn = get_connection(self.db_path)
        conn.execute("PRAGMA synchronous = OFF")

        stats = {"inserted": 0, "skipped": 0}
        batch: list[tuple[str, str, str, str]] = []

        try:
            with json_path.open("r", encoding="utf-8") as f:
                for line in tqdm(f, desc="DBpedia ETL", unit=" entities", mininterval=5.0):
                    if limit and stats["inserted"] >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        # TF-108 (D-GATE-3): refused with a reason.
                        self.schema_gate.refuse("dbpedia", [
                            "unparseable:record: the line is not "
                            "valid JSON"])
                        stats["skipped"] += 1
                        continue

                    # TF-108 gate (D-GATE-1): a JSON value that
                    # is not an object, or an object whose
                    # fields carry the wrong types, is refused
                    # with reasons — it no longer crashes the
                    # build on .get / slicing.
                    reasons = self.schema_gate.check(
                        "dbpedia", item,
                        ref=(item.get("uri") or item.get("@id"))
                        if isinstance(item, dict) else None)
                    if reasons:
                        logger.warning(
                            "DBpedia record refused at the gate: %s",
                            "; ".join(reasons))
                        stats["skipped"] += 1
                        continue

                    uri = item.get("uri") or item.get("@id", "")
                    label = item.get("label") or item.get("name", "")
                    abstract = item.get("abstract") or item.get("comment", "")
                    types = item.get("type") or item.get("@type", "")
                    if isinstance(types, list):
                        types = ",".join(types)

                    if not uri:
                        stats["skipped"] += 1
                        continue

                    batch.append((uri, label[:512], abstract[:2000], str(types)[:512]))
                    stats["inserted"] += 1

                    if len(batch) >= batch_size:
                        self._flush(conn, batch)
                        batch.clear()

            if batch:
                self._flush(conn, batch)

            conn.commit()

        finally:
            conn.close()

        logger.info("DBpedia ETL complete: %s", stats)
        return stats

    @staticmethod
    def _flush(conn: sqlite3.Connection, batch: list[tuple[str, str, str, str]]) -> None:
        conn.executemany(
            "INSERT OR REPLACE INTO dbpedia_entities (uri, label_en, abstract_en, types) "
            "VALUES (?, ?, ?, ?)",
            batch,
        )


# ── Built-in Reference Data ────────────────────────────────────────────────────

class ReferenceDataETL:
    """
    Ingests the 40 built-in knowledge reference files from data/knowledge_reference/
    into the reference_chunks table with chunked text for retrieval.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_knowledge_db_path()

    def build(
        self,
        ref_dir: Path | None = None,
        chunk_size: int = CHUNK_SIZE_CHARS,
        overlap: int = CHUNK_OVERLAP_CHARS,
    ) -> dict[str, int]:
        """
        Chunk and ingest all reference documents from the knowledge_reference directory.
        Supports .md, .json, .jsonl, .yaml files.
        """
        ref_dir = ref_dir or get_knowledge_reference_dir()

        if not ref_dir.exists():
            raise FileNotFoundError(f"Reference data directory not found: {ref_dir}")

        initialize_schema(self.db_path)
        conn = get_connection(self.db_path)

        stats = {"files": 0, "chunks": 0, "skipped": 0}
        supported = {".md", ".json", ".jsonl", ".yaml", ".yml", ".txt"}

        try:
            files = sorted(ref_dir.rglob("*"))
            for path in tqdm(files, desc="Reference ETL", unit=" files"):
                if not path.is_file() or path.suffix.lower() not in supported:
                    continue
                try:
                    chunks = self._load_and_chunk(path, chunk_size, overlap)
                    if not chunks:
                        stats["skipped"] += 1
                        continue
                    batch = []
                    for idx, (title, content, tags) in enumerate(chunks):
                        batch.append((
                            str(path.relative_to(ref_dir)),
                            idx,
                            title,
                            content,
                            tags,
                            len(content.split()),
                        ))
                    conn.executemany(
                        "INSERT OR REPLACE INTO reference_chunks "
                        "(source_file, chunk_index, title, content, tags, word_count) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    conn.commit()
                    stats["chunks"] += len(batch)
                    stats["files"] += 1
                except (OSError, UnicodeDecodeError) as e:
                    logger.warning("Skipping %s: %s", path.name, e)
                    stats["skipped"] += 1

        finally:
            conn.close()

        # TF-058 (D-REF-1, fixed): reference_chunks is searched
        # through an external-content FTS5 table, which a plain
        # INSERT does not populate — only a rebuild does. Before
        # this call, a library ingested by this build (or by the
        # forge_memory `reference` command, which never rebuilt
        # either) was present in the DB but invisible to
        # KnowledgeForge's SQL search. Rebuild before returning
        # so an ingested library is a searchable library.
        rebuild_fts_indexes(self.db_path)

        logger.info("Reference ETL complete: %s", stats)
        return stats

    def _load_and_chunk(
        self,
        path: Path,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[str, str, str]]:
        """Load a file and return list of (title, content, tags) chunks."""
        suffix = path.suffix.lower()
        title = path.stem.replace("_", " ").replace("-", " ").title()
        tags = self._extract_tags_from_filename(path.stem)

        raw_text: str
        if suffix == ".md":
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            return self._chunk_markdown(raw_text, title, tags, chunk_size, overlap)

        elif suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                raw_text = self._json_to_text(data)
            except json.JSONDecodeError:
                raw_text = path.read_text(encoding="utf-8", errors="replace")
            return self._chunk_text(raw_text, title, tags, chunk_size, overlap)

        elif suffix == ".jsonl":
            lines = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    lines.append(self._json_to_text(obj))
                except json.JSONDecodeError:
                    lines.append(line)
            raw_text = "\n\n".join(lines)
            return self._chunk_text(raw_text, title, tags, chunk_size, overlap)

        elif suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
                raw_text = self._json_to_text(data)
            except Exception:
                raw_text = path.read_text(encoding="utf-8", errors="replace")
            return self._chunk_text(raw_text, title, tags, chunk_size, overlap)

        else:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            return self._chunk_text(raw_text, title, tags, chunk_size, overlap)

    @staticmethod
    def _chunk_markdown(
        text: str,
        default_title: str,
        tags: str,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[str, str, str]]:
        """Split markdown by headings, then by size if needed."""
        sections: list[tuple[str, str]] = []
        current_title = default_title
        current_lines: list[str] = []

        for line in text.splitlines():
            heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading_match:
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = heading_match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))

        results: list[tuple[str, str, str]] = []
        for sec_title, sec_content in sections:
            if len(sec_content) <= chunk_size:
                if sec_content.strip():
                    results.append((sec_title, sec_content, tags))
            else:
                for i in range(0, len(sec_content), chunk_size - overlap):
                    chunk = sec_content[i : i + chunk_size]
                    if chunk.strip():
                        results.append((sec_title, chunk, tags))

        return results

    @staticmethod
    def _chunk_text(
        text: str,
        title: str,
        tags: str,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[str, str, str]]:
        """Split plain text into overlapping chunks."""
        if not text.strip():
            return []
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size].strip()
            if chunk:
                chunks.append((title, chunk, tags))
        return chunks

    @staticmethod
    def _json_to_text(data: Any, depth: int = 0) -> str:
        """Flatten JSON/YAML data into a readable text string."""
        if depth > 5:
            return str(data)[:200]
        if isinstance(data, str):
            return data
        if isinstance(data, (int, float, bool)):
            return str(data)
        if isinstance(data, dict):
            parts = []
            for k, v in data.items():
                val_text = ReferenceDataETL._json_to_text(v, depth + 1)
                parts.append(f"{k}: {val_text}")
            return "\n".join(parts)
        if isinstance(data, list):
            return "\n".join(
                ReferenceDataETL._json_to_text(item, depth + 1) for item in data[:50]
            )
        return str(data)[:500]

    @staticmethod
    def _extract_tags_from_filename(stem: str) -> str:
        """Generate tags from filename by splitting on underscores/hyphens."""
        words = re.split(r"[_\-\s]+", stem.lower())
        return ",".join(w for w in words if len(w) > 2)[:256]
