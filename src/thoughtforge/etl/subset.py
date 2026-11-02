"""
subset.py — Edge knowledge subset builder for ThoughtForge.

Builds a reduced SQLite knowledge database suitable for memory-constrained
edge devices (phones, Pi Zero, Pi 5). Copies the top-N entities by
popularity_score from a full knowledge DB, along with their associated
statements and reference chunks.

Usage:
    builder = EdgeSubsetBuilder()
    result = builder.build(
        source_db=Path("/data/thoughtforge.db"),
        output_path=Path("/data/thoughtforge_pi_zero.db"),
        profile_id="pi_zero",
    )
    print(f"Subset: {result.entities_copied} entities → {result.output_path}")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thoughtforge.utils.paths import get_knowledge_db_path

logger = logging.getLogger(__name__)

# ── Profile entity limits ─────────────────────────────────────────────────────
# Default entity caps per hardware profile (can be overridden at build time).
# These are conservative — the full DB may have millions of entities.

_PROFILE_ENTITY_LIMITS: dict[str, int] = {
    "pi_zero":    50_000,
    "phone_low":  80_000,
    "pi_5":      200_000,
    "desktop_cpu": 1_000_000,
    "desktop_gpu": 2_000_000,
    "server_gpu":  0,           # 0 = no limit (full DB)
}

_HARDWARE_PROFILES_DIR = Path(__file__).parents[4] / "hardware_profiles"


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class SubsetResult:
    """Result of an EdgeSubsetBuilder.build() call."""
    profile_id: str
    source_db: Path
    output_path: Path
    entities_copied: int
    statements_copied: int
    reference_chunks_copied: int
    entity_limit: int
    success: bool
    notes: list[str] = field(default_factory=list)

    @property
    def size_mb(self) -> float:
        """Output DB size in MB (0.0 if file doesn't exist)."""
        try:
            return self.output_path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0


# ── EdgeSubsetBuilder ─────────────────────────────────────────────────────────

class EdgeSubsetBuilder:
    """
    Builds a reduced knowledge DB for edge hardware profiles.

    The subset preserves the full ThoughtForge schema but limits entities
    to the top-N by popularity_score (a proxy for retrieval utility).
    Associated statements and reference chunks for kept entities are included.

    Typical reductions: NOT STATED — the builder's earlier
    docstring claimed per-profile output sizes (~50MB /
    ~80MB / ~200MB) that were never measured anywhere
    (D-SUBSET-2, TF-110). A subset's size is a measured
    file fact, not an entity-count guess: TF-110's
    ``subset_craft`` measures every build and trims it
    against the profile's byte budget.
    """

    def build(
        self,
        source_db: Path | str | None = None,
        output_path: Path | str | None = None,
        profile_id: str = "pi_zero",
        max_entities: int | None = None,
    ) -> SubsetResult:
        """
        Build a reduced knowledge DB subset.

        Args:
            source_db:    Path to full knowledge DB (default: auto-detected).
            output_path:  Where to write the subset DB (default: alongside source).
            profile_id:   Hardware profile to target (determines entity limit).
            max_entities: Override entity limit. 0 = no limit (copies everything).

        Returns:
            SubsetResult with copy statistics and output path.

        Raises:
            FileNotFoundError: If source_db does not exist.
            ValueError:        If profile_id is not recognized.
        """
        source_db = Path(source_db) if source_db else get_knowledge_db_path()

        if not source_db.exists():
            raise FileNotFoundError(f"Source DB not found: {source_db}")

        if profile_id not in _PROFILE_ENTITY_LIMITS:
            valid = ", ".join(sorted(_PROFILE_ENTITY_LIMITS))
            raise ValueError(
                f"Unknown profile_id '{profile_id}'. Valid profiles: {valid}"
            )

        entity_limit = max_entities if max_entities is not None else self._load_entity_limit(profile_id)

        if output_path is None:
            stem = source_db.stem
            output_path = source_db.parent / f"{stem}_{profile_id}.db"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # TF-110 (D-SUBSET-3): a build is a NEW artefact.
        # The builder used to connect to an existing output
        # file and INSERT OR IGNORE into it, so a re-build
        # with a smaller limit reported the smaller count
        # while the file still held the earlier, larger
        # subset (probed: re-build at limit 2 onto a limit-5
        # output reported 2, file held 5 entities). Refuse
        # to build over the source itself, and start every
        # build from no file.
        if output_path.resolve() == source_db.resolve():
            raise ValueError(
                "subset output must not be the source DB itself: "
                f"{output_path}"
            )
        for stale in (output_path,
                      Path(str(output_path) + "-wal"),
                      Path(str(output_path) + "-shm")):
            if stale.exists():
                stale.unlink()

        logger.info(
            "EdgeSubsetBuilder: building %s subset → %s (limit=%s)",
            profile_id, output_path, entity_limit if entity_limit else "unlimited",
        )

        result = self._copy_subset(
            source_db=source_db,
            output_path=output_path,
            profile_id=profile_id,
            entity_limit=entity_limit,
        )

        logger.info(
            "EdgeSubsetBuilder: done — %d entities, %d statements, %d ref chunks (%.1f MB)",
            result.entities_copied,
            result.statements_copied,
            result.reference_chunks_copied,
            result.size_mb,
        )

        return result

    # ── Internal copy logic ────────────────────────────────────────────────────

    def _copy_subset(
        self,
        source_db: Path,
        output_path: Path,
        profile_id: str,
        entity_limit: int,
    ) -> SubsetResult:
        """Perform the actual SQLite copy using the ATTACH technique."""
        notes: list[str] = []

        # Open source (read-only) and dest (new)
        src = sqlite3.connect(str(source_db))
        dst = sqlite3.connect(str(output_path))

        try:
            src.row_factory = sqlite3.Row

            # Clone schema from source
            self._clone_schema(src, dst)

            # Select top-N entities by popularity_score
            entities_copied = self._copy_entities(src, dst, entity_limit)

            # Copy statements referencing kept entities
            statements_copied = self._copy_statements(src, dst)

            # Copy reference chunks (all — they don't depend on entity selection)
            ref_copied = self._copy_reference_chunks(src, dst, entity_limit, notes)

            dst.commit()

            # Rebuild FTS if present
            self._rebuild_fts(dst, notes)

            dst.commit()

            # TF-110: compact before anyone measures this
            # file — a subset's reported size must be the
            # artefact's real size, not its build-time
            # high-water mark.
            try:
                dst.execute("VACUUM")
            except sqlite3.OperationalError as exc:
                notes.append(f"vacuum skipped: {exc}")

            return SubsetResult(
                profile_id=profile_id,
                source_db=source_db,
                output_path=output_path,
                entities_copied=entities_copied,
                statements_copied=statements_copied,
                reference_chunks_copied=ref_copied,
                entity_limit=entity_limit,
                success=True,
                notes=notes,
            )

        except Exception as exc:
            logger.error("EdgeSubsetBuilder: copy failed: %s", exc)
            notes.append(f"error: {exc}")
            return SubsetResult(
                profile_id=profile_id,
                source_db=source_db,
                output_path=output_path,
                entities_copied=0,
                statements_copied=0,
                reference_chunks_copied=0,
                entity_limit=entity_limit,
                success=False,
                notes=notes,
            )
        finally:
            src.close()
            dst.close()

    def _clone_schema(self, src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
        """Copy table schema (CREATE TABLE statements) from source to dest."""
        cur = src.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY rootpage"
        )
        for (ddl,) in cur.fetchall():
            if ddl:
                try:
                    dst.execute(ddl)
                except sqlite3.OperationalError:
                    pass    # table already exists

    def _copy_entities(
        self,
        src: sqlite3.Connection,
        dst: sqlite3.Connection,
        entity_limit: int,
    ) -> int:
        """Copy top-N entities by popularity_score into dest DB."""
        # TF-110 (D-SUBSET-4): ties in popularity_score
        # break by qid, so the same source always yields
        # the same subset — the old ORDER BY popularity
        # alone left tied rows to SQLite's unspecified
        # scan order.
        if entity_limit > 0:
            query = (
                "SELECT * FROM entities "
                "ORDER BY popularity_score DESC, qid ASC "
                f"LIMIT {entity_limit}"
            )
        else:
            query = ("SELECT * FROM entities "
                     "ORDER BY popularity_score DESC, qid ASC")

        try:
            rows = src.execute(query).fetchall()
        except sqlite3.OperationalError:
            logger.debug("entities table not present in source DB — skipping")
            return 0

        if not rows:
            return 0

        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
        cols = [d[1] for d in src.execute("PRAGMA table_info(entities)").fetchall()]
        placeholders = ", ".join("?" * len(cols))
        dst.executemany(
            f"INSERT OR IGNORE INTO entities ({', '.join(cols)}) VALUES ({placeholders})",
            [tuple(row) for row in rows],
        )
        logger.debug("Copied %d entities", len(rows))
        return len(rows)

    def _copy_statements(
        self,
        src: sqlite3.Connection,
        dst: sqlite3.Connection,
    ) -> int:
        """Copy statements where subject_qid is in dest entities."""
        try:
            # Get kept QIDs
            kept_qids = {row[0] for row in dst.execute("SELECT qid FROM entities")}
            if not kept_qids:
                return 0

            # Fetch matching statements from source
            query = "SELECT * FROM statements WHERE subject_qid IN ({})".format(
                ", ".join("?" * len(kept_qids))
            )
            try:
                rows = src.execute(query, list(kept_qids)).fetchall()
            except sqlite3.OperationalError:
                return 0

            if not rows:
                return 0

            cols = [d[1] for d in src.execute("PRAGMA table_info(statements)").fetchall()]
            placeholders = ", ".join("?" * len(cols))
            dst.executemany(
                f"INSERT OR IGNORE INTO statements ({', '.join(cols)}) VALUES ({placeholders})",
                [tuple(row) for row in rows],
            )
            logger.debug("Copied %d statements", len(rows))
            return len(rows)

        except Exception as e:
            logger.debug("statements copy skipped: %s", e)
            return 0

    def _copy_reference_chunks(
        self,
        src: sqlite3.Connection,
        dst: sqlite3.Connection,
        entity_limit: int,
        notes: list[str],
    ) -> int:
        """Copy reference_chunks table (proportionally capped for edge profiles)."""
        try:
            # Cap reference chunks proportionally: same ratio as entity limit
            # TF-110 (D-SUBSET-2): the cap used to floor at
            # 1000 (max(1000, limit // 10)), so a build
            # limited to 5 entities still copied up to 1000
            # chunks — the largest text in the store rode
            # along unjudged (probed: 5 entities, all 50
            # chunks, 1.24 MB). The proportion is the rule,
            # with no floor.
            chunk_limit = 0
            if entity_limit > 0:
                chunk_limit = entity_limit // 10

            if chunk_limit > 0:
                # TF-105 (D-SUBSET-1, fixed): this query used to
                # ORDER BY retrieval_score — a column the schema
                # has never carried — so every limited build raised
                # OperationalError, landed in the except below, and
                # copied ZERO reference chunks while still reporting
                # success. There is no per-chunk value score in the
                # schema to order by; chunk_id (insertion order) is
                # the honest deterministic order that stands.
                rows = src.execute(
                    "SELECT * FROM reference_chunks "
                    "ORDER BY chunk_id "
                    f"LIMIT {chunk_limit}"
                ).fetchall()
            else:
                rows = src.execute("SELECT * FROM reference_chunks").fetchall()

            if not rows:
                return 0

            cols = [d[1] for d in src.execute("PRAGMA table_info(reference_chunks)").fetchall()]
            placeholders = ", ".join("?" * len(cols))
            dst.executemany(
                f"INSERT OR IGNORE INTO reference_chunks ({', '.join(cols)}) VALUES ({placeholders})",
                [tuple(row) for row in rows],
            )
            return len(rows)

        except sqlite3.OperationalError as e:
            notes.append(f"reference_chunks skipped: {e}")
            return 0

    def _rebuild_fts(self, dst: sqlite3.Connection, notes: list[str]) -> None:
        """Rebuild FTS5 virtual tables if present."""
        try:
            fts_tables = dst.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'"
            ).fetchall()
            for (tname,) in fts_tables:
                try:
                    dst.execute(f"INSERT INTO {tname}({tname}) VALUES('rebuild')")
                    notes.append(f"FTS rebuilt: {tname}")
                except Exception:
                    pass
        except Exception as e:
            notes.append(f"FTS rebuild skipped: {e}")

    # ── Profile loading ────────────────────────────────────────────────────────

    def _load_entity_limit(self, profile_id: str) -> int:
        """
        Load entity limit from hardware profile JSON, falling back to hardcoded table.
        """
        profile_path = _HARDWARE_PROFILES_DIR / f"{profile_id}.json"
        if profile_path.exists():
            try:
                with open(profile_path) as f:
                    data: dict[str, Any] = json.load(f)
                limit = (
                    data.get("deployment", {}).get("subset_entity_limit")
                    or data.get("memory", {}).get("subset_entity_limit")
                )
                if limit is not None:
                    return int(limit)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        return _PROFILE_ENTITY_LIMITS.get(profile_id, 100_000)

    @staticmethod
    def supported_profiles() -> list[str]:
        """Return list of recognized profile IDs."""
        return sorted(_PROFILE_ENTITY_LIMITS.keys())
