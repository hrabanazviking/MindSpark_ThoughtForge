"""
TF-110 — Subset craft: profile-sized subsets built by
rule + measured.

The craft lives as verifiable data in
``src/thoughtforge/etl/subset_craft.py``
(``SUBSET_RULES``, ``budget_for_profile``,
``measure_subset``, ``build_profile_subset``, checked
by ``subset_craft_problems()`` and
``subset_craft_live_problems()``); the fixes are
enforced in ``src/thoughtforge/etl/subset.py``; the
write-up is ``docs/architecture/SUBSET_CRAFT.md``.
D-SUBSET-2 (no byte budget; the chunk cap's 1000-row
floor; unmeasured docstring sizes), D-SUBSET-3 (a
re-build was not the subset it claimed) and
D-SUBSET-4 (selection order undefined on ties) were
found by this slice's probes and are FIXED here;
D-PROF-1's deployment/subset share is discharged.

Pinned here:

  * budgets resolve live for every shipped profile —
    pi_zero's declared knowledge_subset_max_gb wins,
    every other profile's budget is the filed derived
    road (0.25 x declared RAM) with its source named,
  * measurement is the file's own: bytes with sidecars
    apart, per-table counts, dbstat table bytes,
  * through the REAL builder: a crafted phone_low
    subset fits a small byte budget by measurement
    after trimming attempts, its file holds exactly
    the rows the result reports, a re-build at a
    smaller limit holds exactly the smaller subset,
    ties break by qid, the chunk share has no floor,
    building over the source is refused,
  * an unlimited profile is measured, never trimmed,
    and an impossible budget fails honestly,
  * both checkers are clean.

All fixtures are synthetic (TF-017): invented records
in a temp dir, never a real dump or canon.

Proof grade: FIXTURE (synthetic stores through the
real builder on this VM). A real dump's subset size
and a real phone's storage are NOT PROVEN.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from thoughtforge.etl.subset import EdgeSubsetBuilder
from thoughtforge.etl.subset_craft import (
    MEASURED_TABLES,
    RAM_BUDGET_FRACTION,
    SUBSET_RULES,
    budget_for_profile,
    build_profile_subset,
    measure_subset,
    subset_craft_live_problems,
    subset_craft_problems,
)
from thoughtforge.utils.config import list_hardware_profiles

BIG = "x" * 20000


def _source(tmp_path: Path, entities: int = 30, chunks: int = 50) -> Path:
    from thoughtforge.etl.schema import initialize_schema

    src = tmp_path / "full.db"
    initialize_schema(src)
    conn = sqlite3.connect(str(src))
    try:
        for i in range(entities):
            conn.execute(
                "INSERT INTO entities VALUES (?,?,?,?,?,?,?)",
                (f"Q{i:02d}", f"entity_{i}", BIG, "", "item",
                 float(entities - i), "wikidata"),
            )
            conn.execute(
                "INSERT INTO statements (subject_qid, property_pid,"
                " object_value, object_type) VALUES (?,?,?,?)",
                (f"Q{i:02d}", "P1", "v", "t"),
            )
        for j in range(chunks):
            conn.execute(
                "INSERT INTO reference_chunks (source_file, chunk_index,"
                " title, content, tags, word_count) VALUES (?,?,?,?,?,?)",
                (f"f{j}.md", 0, "t", BIG, "", 1),
            )
        conn.commit()
    finally:
        conn.close()
    return src


def _count(db: Path, table: str) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


class TestBudgets:
    def test_rules_are_unique_data(self) -> None:
        ids = [r["id"] for r in SUBSET_RULES]
        assert len(ids) == len(set(ids))
        assert "budget_is_bytes_not_entities" in ids
        assert "unfittable_is_a_failure_not_a_claim" in ids

    def test_every_profile_resolves(self) -> None:
        for pid in list_hardware_profiles():
            budget = budget_for_profile(pid)
            assert budget.profile_id == pid
            assert budget.budget_bytes > 0
            assert budget.budget_source

    def test_pi_zero_declared_budget_wins(self) -> None:
        budget = budget_for_profile("pi_zero")
        assert budget.declared_budget_bytes == int(0.5 * 1024 ** 3)
        assert budget.budget_bytes == budget.declared_budget_bytes
        assert budget.budget_source.startswith("declared:")

    def test_phone_low_budget_is_the_derived_road(self) -> None:
        budget = budget_for_profile("phone_low")
        assert budget.declared_budget_bytes is None
        assert budget.budget_bytes == int(2 * 1024 ** 3 * RAM_BUDGET_FRACTION)
        assert budget.budget_source.startswith("derived:")

    def test_entity_limits_ride_the_builder_table(self) -> None:
        from thoughtforge.etl.subset import _PROFILE_ENTITY_LIMITS

        for pid, limit in _PROFILE_ENTITY_LIMITS.items():
            assert budget_for_profile(pid).entity_limit == limit

    def test_unknown_profile_raises_by_name(self) -> None:
        with pytest.raises(ValueError, match="neptune_ufo"):
            budget_for_profile("neptune_ufo")


class TestBuilderFixes:
    def test_ties_break_by_qid(self, tmp_path: Path) -> None:
        from thoughtforge.etl.schema import initialize_schema

        src = tmp_path / "ties.db"
        initialize_schema(src)
        conn = sqlite3.connect(str(src))
        try:
            for qid in ("Q9", "Q1", "Q5"):
                conn.execute(
                    "INSERT INTO entities VALUES (?,?,?,?,?,?,?)",
                    (qid, qid, "", "", "item", 7.0, "wikidata"),
                )
            conn.commit()
        finally:
            conn.close()
        out = tmp_path / "ties_out.db"
        EdgeSubsetBuilder().build(
            source_db=src, output_path=out,
            profile_id="pi_zero", max_entities=2,
        )
        conn = sqlite3.connect(str(out))
        try:
            kept = [r[0] for r in conn.execute("SELECT qid FROM entities")]
        finally:
            conn.close()
        assert sorted(kept) == ["Q1", "Q5"]

    def test_chunk_share_has_no_floor(self, tmp_path: Path) -> None:
        src = _source(tmp_path, entities=60, chunks=50)
        out = tmp_path / "sub.db"
        result = EdgeSubsetBuilder().build(
            source_db=src, output_path=out,
            profile_id="phone_low", max_entities=50,
        )
        assert result.entities_copied == 50
        assert result.reference_chunks_copied == 5  # 50 // 10, no floor

    def test_rebuild_holds_exactly_the_new_subset(
        self, tmp_path: Path,
    ) -> None:
        src = _source(tmp_path)
        out = tmp_path / "sub.db"
        builder = EdgeSubsetBuilder()
        builder.build(source_db=src, output_path=out,
                      profile_id="phone_low", max_entities=5)
        assert _count(out, "entities") == 5
        result = builder.build(source_db=src, output_path=out,
                               profile_id="phone_low", max_entities=2)
        assert result.entities_copied == 2
        assert _count(out, "entities") == 2

    def test_building_over_the_source_is_refused(
        self, tmp_path: Path,
    ) -> None:
        src = _source(tmp_path)
        with pytest.raises(ValueError, match="must not be the source"):
            EdgeSubsetBuilder().build(
                source_db=src, output_path=src, profile_id="phone_low",
            )


class TestMeasure:
    def test_measure_reports_the_file_itself(self, tmp_path: Path) -> None:
        src = _source(tmp_path, entities=3, chunks=2)
        out = tmp_path / "sub.db"
        EdgeSubsetBuilder().build(
            source_db=src, output_path=out,
            profile_id="phone_low", max_entities=3,
        )
        measurement = measure_subset(out)
        assert measurement["bytes"] == out.stat().st_size
        assert measurement["file_bytes"]["db"] == measurement["bytes"]
        assert measurement["counts"]["entities"] == 3
        assert measurement["counts"]["statements"] == 3
        assert set(measurement["counts"]) == set(MEASURED_TABLES)
        assert measurement["table_bytes"]["entities"] is not None
        assert measurement["table_bytes"]["entities"] > 0


class TestCraft:
    def test_craft_trims_until_the_measured_size_fits(
        self, tmp_path: Path,
    ) -> None:
        src = _source(tmp_path)
        out = tmp_path / "craft.db"
        result = build_profile_subset(
            src, out, "phone_low", max_bytes=200_000,
        )
        assert result.success is True
        assert result.fits_budget is True
        assert result.measured_bytes is not None
        assert result.measured_bytes <= 200_000
        assert len(result.attempts) >= 2
        assert result.attempts[0]["measured_bytes"] > 200_000
        assert result.attempts[-1]["measured_bytes"] == result.measured_bytes
        assert _count(out, "entities") == result.entities_copied
        assert result.budget["budget_source"].startswith("override:")

    def test_craft_at_the_profiles_own_budget(self, tmp_path: Path) -> None:
        src = _source(tmp_path, entities=3, chunks=2)
        out = tmp_path / "craft.db"
        result = build_profile_subset(src, out, "phone_low")
        assert result.success is True
        assert result.fits_budget is True
        assert len(result.attempts) == 1
        assert result.budget["budget_bytes"] == int(
            2 * 1024 ** 3 * RAM_BUDGET_FRACTION
        )

    def test_unlimited_profile_is_measured_never_trimmed(
        self, tmp_path: Path,
    ) -> None:
        src = _source(tmp_path, entities=4, chunks=3)
        out = tmp_path / "craft.db"
        result = build_profile_subset(src, out, "server_gpu")
        assert result.success is True
        assert result.entities_copied == 4
        assert len(result.attempts) == 1
        assert result.measured_bytes is not None and result.measured_bytes > 0

    def test_impossible_budget_fails_honestly(self, tmp_path: Path) -> None:
        src = _source(tmp_path)
        result = build_profile_subset(
            src, tmp_path / "craft.db", "phone_low", max_bytes=1024,
        )
        assert result.success is False
        assert result.fits_budget is False
        assert result.entities_copied == 0
        assert any("no entity trim can craft" in n for n in result.notes)


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert subset_craft_problems() == []

    def test_live_checker_clean(self, tmp_path: Path) -> None:
        assert subset_craft_live_problems(tmp_path / "live") == []
