"""TF-059 — freshness: time-aware retrieval (recent facts
weighted per the stated rule) for time-sensitive domains.

Pins knowledge/freshness.py against the live tree: the
domains as data and their classification (tags, category,
fact keys; most-perishable governs), the clock (updated_at
-> created_at -> last_used_at; unparseable reads as the
lifecycle's 30-day default), decayed-not-binary scores at
exactly 0.5 at one half-life, verdict bands (timeless is
never stale at any age), the weighting rule (sensitive
records at recency 0.40 separate by > 0.3; timeless keep
their shipped row), search_store over a synthetic store
(fresh first, stale excluded by default and named, stale
returned flagged on request, bundle bookkeeping sound,
store byte-untouched), the freshness report, and the
D-FRESH-1 forge wiring (a timestamped pair separates; a
timestamp-less pair keeps the proxy arithmetic exactly).
All clocks are explicit — no test depends on the wall
clock except where it says so.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from thoughtforge.knowledge.freshness import (
    CLOCK_FIELDS,
    FRESHNESS_AGE_DEFAULT_DAYS,
    FRESHNESS_RULES,
    TIME_SENSITIVE_DOMAINS,
    TIME_SENSITIVE_WEIGHTS,
    assess_freshness,
    classify_domain,
    freshness_problems,
    freshness_report,
    freshness_score,
    freshness_verdict,
    rank_records,
    recency_from_timestamps,
    record_age_days,
    search_store,
    weights_for,
)
from thoughtforge.knowledge.models import (
    EpisodicMemoryRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.scoring import (
    HALF_LIFE_DAYS,
    RETRIEVAL_WEIGHTS_BY_TYPE,
    compute_recency_score,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.knowledge.store_map import bundle_problems

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)


def _stamp(age_days: float) -> str:
    return (NOW - timedelta(days=age_days)).isoformat()


def _fact(rid: str, age_days: float, **kw) -> UserFactRecord:
    stamp = _stamp(age_days)
    return UserFactRecord(
        record_id=rid, created_at=stamp, updated_at=stamp,
        last_used_at=stamp, **kw)


def _store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.append_profile_record(_fact(
        "f_fresh", 0.0, tags=["schedule"],
        summary="the ferry departs at dawn"))
    store.append_profile_record(_fact(
        "f_month", 30.0, tags=["schedule"],
        summary="the ferry departs at dawn"))
    store.append_profile_record(_fact(
        "f_plain", 365.0, tags=["ferry"],
        summary="the ferry crosses the river"))
    store.append_episode(EpisodicMemoryRecord(
        record_id="ep_ferry", tags=["ferry"],
        summary="the ferry lantern was lit",
        created_at=_stamp(1.0), updated_at=_stamp(1.0),
        last_used_at=_stamp(1.0)))
    return store


# ── The domains, as data ─────────────────────────────────────────────────────


class TestDomains:
    def test_problems_empty_on_live_tree(self) -> None:
        assert freshness_problems() == []

    def test_domains_and_rules_as_data(self) -> None:
        assert [r["name"] for r in TIME_SENSITIVE_DOMAINS] == [
            "schedule", "location", "status", "price"]
        for row in TIME_SENSITIVE_DOMAINS:
            assert row["half_life_days"] < row["stale_after_days"]
            assert row["markers"]
        assert [r["name"] for r in FRESHNESS_RULES] == [
            "domain_governs", "timeless_never_stale",
            "decayed_not_binary", "stale_never_silent",
            "clock_is_the_write", "read_only"]

    def test_sensitive_weights(self) -> None:
        assert math.isclose(sum(TIME_SENSITIVE_WEIGHTS.values()), 1.0)
        assert TIME_SENSITIVE_WEIGHTS["recency"] == 0.40

    def test_classify_by_tag_category_and_fact_key(self) -> None:
        assert classify_domain(_fact(
            "a", 0.0, tags=["Schedule"])) == "schedule"
        assert classify_domain(_fact(
            "b", 0.0, tags=["opening-hours"])) == "schedule"
        assert classify_domain(_fact(
            "c", 0.0, tags=["ferry"], fact_value={"fare": 3})) == "price"
        pref = UserPreferenceRecord(
            record_id="p", category="location", summary="based at the mill",
            created_at=_stamp(0.0), updated_at=_stamp(0.0),
            last_used_at=_stamp(0.0), last_confirmed_at=_stamp(0.0))
        assert classify_domain(pref) == "location"
        assert classify_domain(_fact("d", 0.0, tags=["ferry"])) == ""

    def test_most_perishable_domain_governs(self) -> None:
        record = _fact("x", 0.0, tags=["schedule"],
                       fact_value={"fare": 3})
        assessment = assess_freshness(record, now=NOW)
        assert assessment.domain == "schedule"
        assert assessment.half_life_days == 2.0


# ── Clock and score ──────────────────────────────────────────────────────────


class TestClockAndScore:
    def test_clock_field_order(self) -> None:
        assert CLOCK_FIELDS == ("updated_at", "created_at", "last_used_at")
        record = _fact("x", 5.0, tags=[])
        record.created_at = _stamp(50.0)
        assert record_age_days(record, now=NOW) == pytest.approx(5.0)

    def test_unparseable_clock_reads_as_default(self) -> None:
        record = _fact("x", 0.0, tags=[])
        record.updated_at = record.created_at = record.last_used_at = "junk"
        assert record_age_days(record, now=NOW) == FRESHNESS_AGE_DEFAULT_DAYS
        from thoughtforge.knowledge.lifecycle import AGE_PARSE_DEFAULT_DAYS
        assert FRESHNESS_AGE_DEFAULT_DAYS == AGE_PARSE_DEFAULT_DAYS

    def test_future_clock_ages_zero(self) -> None:
        record = _fact("x", -3.0, tags=[])
        assert record_age_days(record, now=NOW) == 0.0

    def test_score_at_one_half_life_is_exactly_half(self) -> None:
        assert freshness_score(
            _fact("s", 2.0, tags=["schedule"]), now=NOW) == pytest.approx(0.5)
        assert freshness_score(
            _fact("p", 30.0, fact_value={"fare": 1}), now=NOW) == pytest.approx(0.5)

    def test_timeless_score_is_the_type_decay(self) -> None:
        record = _fact("t", 90.0, tags=["ferry"])
        assert freshness_score(record, now=NOW) == pytest.approx(
            compute_recency_score(90.0, "user_fact"))
        assert assess_freshness(record, now=NOW).half_life_days == \
            HALF_LIFE_DAYS["user_fact"]


# ── Verdicts ─────────────────────────────────────────────────────────────────


class TestVerdicts:
    def test_fresh_and_ageing_bands(self) -> None:
        assert freshness_verdict(
            _fact("a", 0.0, tags=["schedule"]), now=NOW) == "fresh"
        assert freshness_verdict(
            _fact("b", 3.0, tags=["schedule"]), now=NOW) == "ageing"

    def test_stale_is_sensitive_only(self) -> None:
        assert freshness_verdict(
            _fact("a", 30.0, tags=["schedule"]), now=NOW) == "stale"
        assert freshness_verdict(
            _fact("b", 100.0, fact_value={"fare": 1}), now=NOW) == "stale"

    def test_timeless_is_never_stale(self) -> None:
        assert freshness_verdict(
            _fact("a", 3650.0, tags=["ferry"]), now=NOW) == "timeless"


# ── The weighting rule ───────────────────────────────────────────────────────


class TestRanking:
    def test_weights_for(self) -> None:
        assert weights_for(_fact("s", 0.0, tags=["schedule"])) == \
            TIME_SENSITIVE_WEIGHTS
        plain = _fact("t", 0.0, tags=["ferry"])
        assert weights_for(plain) == RETRIEVAL_WEIGHTS_BY_TYPE["user_fact"]
        assert weights_for(plain) is not RETRIEVAL_WEIGHTS_BY_TYPE["user_fact"]

    def test_fresh_schedule_outranks_month_old_by_the_recency_share(
            self) -> None:
        fresh = _fact("f_fresh", 0.0, tags=["schedule"],
                      summary="the ferry departs at dawn")
        old = _fact("f_old", 30.0, tags=["schedule"],
                    summary="the ferry departs at dawn")
        ranked = rank_records(
            [old, fresh], {"f_fresh": 1.0, "f_old": 1.0}, now=NOW)
        assert [a.record_id for a, _ in ranked] == ["f_fresh", "f_old"]
        assert ranked[0][1] - ranked[1][1] > 0.3

    def test_timeless_recency_still_orders_but_softly(self) -> None:
        new = _fact("t_new", 0.0, tags=["ferry"], summary="x")
        old = _fact("t_old", 180.0, tags=["ferry"], summary="x")
        ranked = rank_records(
            [old, new], {"t_new": 1.0, "t_old": 1.0}, now=NOW)
        assert [a.record_id for a, _ in ranked] == ["t_new", "t_old"]
        assert 0.0 < ranked[0][1] - ranked[1][1] < 0.1

    def test_ranking_is_reproducible(self) -> None:
        records = [_fact(f"r{i}", float(i), tags=["schedule"], summary="x")
                   for i in range(5)]
        rel = {f"r{i}": 1.0 for i in range(5)}
        first = [a.record_id for a, _ in rank_records(records, rel, now=NOW)]
        second = [a.record_id for a, _ in rank_records(records, rel, now=NOW)]
        assert first == second == ["r0", "r1", "r2", "r3", "r4"]


# ── search_store ─────────────────────────────────────────────────────────────


class TestSearchStore:
    def test_fresh_first_stale_excluded_and_named(
            self, tmp_path: Path) -> None:
        bundle, report = search_store(_store(tmp_path), "ferry", now=NOW)
        ids = [r.record_id for r in bundle.activated_records]
        assert ids[0] == "f_fresh"
        assert "f_month" not in ids
        assert report["stale_excluded"] == ["f_month"]
        assert bundle_problems(bundle, top_k=10, min_score=0.0) == []
        for record in bundle.activated_records:
            assert record.raw["freshness"]["verdict"] in (
                "fresh", "ageing", "timeless")

    def test_include_stale_returns_it_flagged(self, tmp_path: Path) -> None:
        bundle, report = search_store(
            _store(tmp_path), "ferry", now=NOW, include_stale=True)
        rows = {r.record_id: r for r in bundle.activated_records}
        assert rows["f_month"].raw["freshness"]["verdict"] == "stale"
        assert rows["f_month"].raw["freshness"]["domain"] == "schedule"
        assert report["stale_excluded"] == []

    def test_no_match_and_empty_query(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert search_store(store, "zzqx unwritten", now=NOW)[0] \
            .total_retrieved == 0
        assert search_store(store, "", now=NOW)[0].total_retrieved == 0

    def test_top_k_caps(self, tmp_path: Path) -> None:
        bundle, _ = search_store(_store(tmp_path), "ferry", now=NOW, top_k=1)
        assert bundle.total_retrieved == 1
        assert bundle.activated_records[0].record_id == "f_fresh"

    def test_store_is_byte_untouched(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mem = tmp_path / "mem"
        before = {p.name: p.read_bytes()
                  for p in sorted(mem.iterdir()) if p.is_file()}
        search_store(store, "ferry", now=NOW)
        search_store(store, "ferry", now=NOW, include_stale=True)
        freshness_report(store, now=NOW)
        after = {p.name: p.read_bytes()
                 for p in sorted(mem.iterdir()) if p.is_file()}
        assert after == before


# ── freshness_report ─────────────────────────────────────────────────────────


class TestReport:
    def test_report_counts_and_names(self, tmp_path: Path) -> None:
        report = freshness_report(_store(tmp_path), now=NOW)
        assert report["total"] == 4
        assert report["by_domain"] == {"schedule": 2}
        assert report["by_verdict"]["stale"] == 1
        assert report["by_verdict"]["timeless"] == 2
        assert report["stale"] == ["f_month"]
        rows = {r["record_id"]: r for r in report["time_sensitive"]}
        assert rows["f_fresh"]["verdict"] == "fresh"
        assert rows["f_month"]["age_days"] == pytest.approx(30.0)

    def test_report_over_a_record_list(self) -> None:
        report = freshness_report(
            [_fact("a", 0.0, tags=["schedule"])], now=NOW)
        assert report["total"] == 1
        assert report["by_verdict"] == {"fresh": 1}


# ── D-FRESH-1: the forge wiring ──────────────────────────────────────────────


class TestForgeWiring:
    def _forge(self, tmp_path: Path):
        from thoughtforge.knowledge.forge import KnowledgeForge
        return KnowledgeForge(db_path=tmp_path / "k.db")

    def test_timestamped_pair_separates(self, tmp_path: Path) -> None:
        forge = self._forge(tmp_path)
        base = {"label_en": "ferry", "description_en": "x",
                "popularity_score": 0.5, "fts_score": 10.0,
                "source": "wikidata", "record_type": "entity"}
        old = dict(base, qid="Q_old", updated_at="2020-01-01T00:00:00+00:00")
        new = dict(base, qid="Q_new", updated_at=NOW.isoformat())
        scored = forge._score_records([old, new], "ferry")
        by_id = {r["qid"]: r["_composite_score"] for r in scored}
        assert by_id["Q_new"] > by_id["Q_old"]

    def test_timestamp_less_pair_keeps_proxy_equality(
            self, tmp_path: Path) -> None:
        forge = self._forge(tmp_path)
        base = {"label_en": "ferry", "description_en": "x",
                "popularity_score": 0.5, "fts_score": 10.0,
                "source": "wikidata", "record_type": "entity"}
        scored = forge._score_records(
            [dict(base, qid="Q_a"), dict(base, qid="Q_b")], "ferry")
        assert scored[0]["_composite_score"] == scored[1]["_composite_score"]
        assert scored[0]["_composite_score"] == pytest.approx(0.57)

    def test_recency_from_timestamps(self) -> None:
        assert recency_from_timestamps({"qid": "Q1"}) is None
        assert recency_from_timestamps(
            {"updated_at": NOW.isoformat()}, now=NOW) == pytest.approx(1.0)
        old = recency_from_timestamps(
            {"created_at": "2020-01-01T00:00:00+00:00"}, now=NOW)
        assert old is not None and old < 0.001
        assert recency_from_timestamps({"updated_at": "junk"}) is None
