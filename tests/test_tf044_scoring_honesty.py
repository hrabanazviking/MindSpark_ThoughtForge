"""
TF-044 — Scoring honesty: the scoring module's weights made
explicit + tunable per profile; no magic constants unlisted.

The census + checker live in
``src/thoughtforge/knowledge/scoring_honesty.py``
(``SCORING_CENSUS`` / ``EXTERNAL_SCORING`` /
``STRUCTURAL_LITERALS`` / ``FORGE_FUSION_NAMES`` /
``CONFIG_DRIFT_KEYS`` /
``PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT`` /
``PROFILE_BASELINE_IMPACT``, ``census()``,
``weights_for_profile()``, ``weight_table_problems()``,
``scoring_problems()``, ``geonames_composite()``); the
write-up is ``docs/architecture/SCORING_HONESTY.md``.

Pinned here:

  * every scoring constant in the knowledge layer is
    censused and the checker re-proves the census live
    (``scoring_problems() == []``): tables sum to 1.0,
    thresholds at their pinned values, the config file's
    drift exactly the recorded drift,
  * the constants TF-044 named carry exactly the values the
    inline literals carried (writeback tables, genericness
    0.15/0.6/0.4, recency fallback 30.0, the forge fusion
    constants), and the scoring functions compute identical
    results at their defaults — TF-043's pinned baseline
    re-measured through a default forge is the regression
    proof,
  * no unnamed literal can hide: the AST scan's structural
    set is exactly the stated one, and the forge's fusion
    functions reference the named constants,
  * per-profile tuning resolves and validates for every
    shipped profile: zero-shift profiles return the shipped
    rows untouched (and never mutate them), the shifted
    profiles move exactly the stated weight, sums stay 1.0,
    unknown profiles/record types raise naming valid options,
  * a forge built with a profile's weights measures exactly
    the recorded ``PROFILE_BASELINE_IMPACT`` on TF-043's
    fixtures — including the honest part: the shifted
    profiles measure precision 0.8519, below the 0.8889
    baseline, a recorded cost and not an improvement,
  * TF-043's GeoNames finding is re-derived from the named
    constants: ``geonames_composite()`` == 0.285 exactly,
    under MIN_MEMORY_SCORE by construction,
  * the models.py composites censused in EXTERNAL_SCORING
    still sum to 1.0 with a −0.10 genericness share.

Proof grade: FIXTURE (TF-043's synthetic corpus, SQL path,
this VM) + live-tree proofs. Live-corpus tuning NOT PROVEN.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from thoughtforge.knowledge import scoring
from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.retrieval_precision import (
    BASELINE,
    FIXTURE_QUERIES,
    build_fixture_db,
    evaluate_retrieval,
    score_query,
)
from thoughtforge.knowledge.scoring_honesty import (
    CONFIG_DRIFT_KEYS,
    EXTERNAL_SCORING,
    FORGE_FUSION_NAMES,
    GEONAMES_COMPOSITE,
    PROFILE_BASELINE_IMPACT,
    PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT,
    RETRIEVAL_DIMENSIONS,
    SCORING_CENSUS,
    STRUCTURAL_LITERALS,
    census,
    geonames_composite,
    scoring_problems,
    weight_table_problems,
    weights_for_profile,
)
from thoughtforge.utils.config import list_hardware_profiles

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "architecture" / "SCORING_HONESTY.md"


# ── The census and the checker ───────────────────────────────────────────────

def test_doc_exists_and_names_the_slice():
    text = DOC.read_text()
    assert "TF-044" in text
    assert "scoring_honesty.py" in text


def test_checker_is_clean_on_the_live_tree():
    assert scoring_problems() == []


def test_census_covers_every_scoring_module_constant():
    censused = {name for name, _kind, _role in SCORING_CENSUS}
    # Every public constant-shaped name in scoring.py is listed.
    for name in dir(scoring):
        if name.isupper():
            assert name in censused, f"{name} is not in SCORING_CENSUS"
    # And every censused name exists with a stated role.
    for name, _kind, role in SCORING_CENSUS:
        assert hasattr(scoring, name)
        assert role


def test_external_scoring_is_listed_with_owners():
    names = {name for name, _kind, _owner in EXTERNAL_SCORING}
    assert any("CandidateScores" in n for n in names)
    assert any("salvage" in n for n in names)
    assert any("core._genericness_penalty" in n for n in names)
    assert any("lifecycle" in n for n in names)


def test_census_rows_carry_live_values():
    rows = {row["name"]: row for row in census()}
    assert rows["MIN_MEMORY_SCORE"]["value"] == 0.45
    assert math.isclose(rows["PRUNE_SCORE_WEIGHTS"]["sum"], 1.0)
    assert math.isclose(
        rows["RETRIEVAL_WEIGHTS_BY_TYPE"]["rows"]["user_fact"], 1.0
    )


def test_structural_literals_are_only_definitions():
    assert set(STRUCTURAL_LITERALS) == {0, 0.0, 0.5, 1, 1.0}
    assert "decay" in STRUCTURAL_LITERALS[0.5]


def test_forge_fusion_names_cover_the_three_functions():
    assert set(FORGE_FUSION_NAMES) == {
        "_score_records", "sql_retrieve", "vector_retrieve"
    }


def test_config_drift_is_exactly_the_two_recorded_keys():
    assert CONFIG_DRIFT_KEYS == {
        "tone_match": "tone_similarity",
        "user_preference": "preference_relevance",
    }


# ── The newly named constants carry the shipped values ──────────────────────

def test_writeback_tables_carry_the_inline_values():
    assert scoring.PREFERENCE_WRITEBACK_WEIGHTS == {
        "durability": 0.30, "novelty": 0.20, "confidence": 0.25,
        "importance": 0.10, "future_relevance": 0.15,
    }
    assert scoring.EPISODE_WRITEBACK_WEIGHTS == {
        "novelty": 0.20, "confidence": 0.15, "importance": 0.25,
        "future_relevance": 0.20, "thread_relevance": 0.20,
    }
    assert scoring.PATTERN_LEARNING_WEIGHTS == {
        "final_response_quality": 0.45, "pattern_reusability": 0.30,
        "future_relevance": 0.25,
    }


def test_genericness_and_recency_constants_carry_the_inline_values():
    assert scoring.GENERICNESS_PHRASE_PENALTY == 0.15
    assert scoring.GENERICNESS_PHRASE_PENALTY_CAP == 0.6
    assert scoring.GENERICNESS_REPETITION_WEIGHT == 0.4
    assert scoring.DEFAULT_HALF_LIFE_DAYS == 30.0
    # The fallback still applies to an unlisted record type.
    assert math.isclose(
        scoring.compute_recency_score(30.0, "no_such_type"), 0.5
    )


def test_fusion_constants_carry_the_inline_values():
    assert scoring.FTS_SCORE_DIVISOR == 20.0
    assert scoring.SOURCE_RECENCY_PROXIES == {
        "reference": 0.9, "wikidata": 0.7, "conceptnet": 0.8,
        "geonames": 0.75, "dbpedia": 0.65,
    }
    assert scoring.DEFAULT_SOURCE_RECENCY == 0.7
    assert scoring.NEUTRAL_TONE_SIMILARITY == 0.5
    assert scoring.NEUTRAL_PREFERENCE_RELEVANCE == 0.5
    assert scoring.IMPORTANCE_WEIGHT_DEFAULT == 0.5
    assert scoring.GEONAMES_FIXED_FTS_SCORE == 0.5
    assert scoring.VECTOR_MIN_SCORE == 0.3


def test_scoring_functions_unchanged_at_defaults():
    dims = scoring.RetrievalDimensions(
        semantic_similarity=1.0, tone_similarity=1.0,
        preference_relevance=1.0, recency=1.0, importance=1.0,
        thread_boost=1.0, pattern_quality=1.0, priority=1.0,
    )
    assert math.isclose(scoring.score_retrieval(dims, "user_fact"), 1.0)
    wb = scoring.WritebackComponents(
        durability=1.0, novelty=1.0, confidence=1.0, importance=1.0,
        future_relevance=1.0, thread_relevance=1.0,
        final_response_quality=1.0, pattern_reusability=1.0,
    )
    assert math.isclose(scoring.compute_preference_writeback_score(wb), 1.0)
    assert math.isclose(scoring.compute_episode_writeback_score(wb), 1.0)
    assert math.isclose(scoring.compute_pattern_learning_score(wb), 1.0)
    # A generic phrase hit costs exactly the named penalty share.
    assert math.isclose(
        scoring.compute_genericness_penalty("it is what it is"),
        min(0.15 + (1.0 - 3 / 5) * 0.4, 1.0),
    )


def test_scoring_functions_accept_an_override_table():
    dims = scoring.RetrievalDimensions(semantic_similarity=1.0)
    custom = {dim: 0.0 for dim in RETRIEVAL_DIMENSIONS}
    custom["semantic_similarity"] = 1.0
    assert scoring.score_retrieval(dims, "user_fact", weights=custom) == 1.0
    assert scoring.score_retrieval(dims, "user_fact") == 0.40


# ── Per-profile tuning ───────────────────────────────────────────────────────

def test_every_shipped_profile_has_a_recorded_shift():
    assert set(PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT) == set(
        list_hardware_profiles()
    )


def test_zero_shift_profiles_return_the_shipped_rows():
    for profile_id, shift in PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT.items():
        if shift:
            continue
        resolved = weights_for_profile(profile_id)
        for rtype, row in scoring.RETRIEVAL_WEIGHTS_BY_TYPE.items():
            assert resolved[rtype] == row
        assert resolved["base"]["semantic_similarity"] == 0.35


def test_shifted_profiles_move_exactly_the_stated_weight():
    for profile_id in ("phone_low", "pi_zero"):
        shift = PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT[profile_id]
        row = weights_for_profile(profile_id, "user_fact")
        shipped = scoring.RETRIEVAL_WEIGHTS_BY_TYPE["user_fact"]
        assert math.isclose(
            row["semantic_similarity"],
            shipped["semantic_similarity"] - shift,
        )
        assert math.isclose(row["importance"], shipped["importance"] + shift)
        assert math.isclose(sum(row.values()), 1.0)
        # Nothing else in the row moved.
        for key in shipped:
            if key not in ("semantic_similarity", "importance"):
                assert row[key] == shipped[key]


def test_resolution_never_mutates_the_shipped_tables():
    before = {
        rtype: dict(row)
        for rtype, row in scoring.RETRIEVAL_WEIGHTS_BY_TYPE.items()
    }
    weights_for_profile("pi_zero")
    weights_for_profile("phone_low", "user_fact")["importance"] = 99.0
    assert scoring.RETRIEVAL_WEIGHTS_BY_TYPE == before


def test_unknown_profile_and_record_type_raise_naming_options():
    with pytest.raises(ValueError, match="unknown hardware profile"):
        weights_for_profile("bogus_profile")
    with pytest.raises(ValueError, match="unknown record type"):
        weights_for_profile("desktop_cpu", "bogus_type")


def test_weight_table_problems_catches_bad_tables():
    good = weights_for_profile("desktop_cpu", "user_fact")
    assert weight_table_problems(good, dimensions=RETRIEVAL_DIMENSIONS) == []
    bad_sum = dict(good, importance=0.99)
    assert any("sums to" in p for p in weight_table_problems(
        bad_sum, dimensions=RETRIEVAL_DIMENSIONS))
    missing = {k: v for k, v in good.items() if k != "priority"}
    assert any("keys" in p for p in weight_table_problems(
        missing, dimensions=RETRIEVAL_DIMENSIONS))
    negative = dict(good, recency=-0.1, importance=good["importance"] + 0.1)
    assert any("negative" in p for p in weight_table_problems(
        negative, dimensions=RETRIEVAL_DIMENSIONS))


# ── The measured impact, re-measured ─────────────────────────────────────────

def _profile_report(db_path, profile_id):
    forge = KnowledgeForge(
        db_path=db_path,
        retrieval_weights=weights_for_profile(profile_id, "user_fact"),
    )
    per_query = []
    for row in FIXTURE_QUERIES:
        bundle = forge.retrieve(
            row["query"], path="sql", top_k=10, min_score=scoring.MIN_MEMORY_SCORE
        )
        scored = score_query(
            [r.record_id for r in bundle.activated_records], row["relevant"]
        )
        per_query.append((row["kind"], scored))
    answered = [s for kind, s in per_query if kind != "negative"]
    return {
        "mean_precision": sum(s["precision"] for _k, s in per_query) / len(per_query),
        "mean_recall": sum(s["recall"] for _k, s in per_query) / len(per_query),
        "mean_f1": sum(s["f1"] for _k, s in per_query) / len(per_query),
        "mean_reciprocal_rank": (
            sum(s["reciprocal_rank"] for s in answered) / len(answered)
        ),
    }


def test_profile_impact_matches_the_recorded_table(tmp_path):
    db_path = build_fixture_db(tmp_path / "fixture.db")
    for profile_id, pinned in PROFILE_BASELINE_IMPACT.items():
        measured = _profile_report(db_path, profile_id)
        for metric, value in pinned.items():
            assert round(measured[metric], 4) == value, (
                f"{profile_id}.{metric}: measured {measured[metric]:.4f} "
                f"!= recorded {value}"
            )


def test_zero_shift_profiles_reproduce_tf043_baseline(tmp_path):
    db_path = build_fixture_db(tmp_path / "fixture.db")
    # The default forge (no weights) still measures TF-043's baseline.
    report = evaluate_retrieval(db_path)
    for metric, pinned in BASELINE["default"].items():
        assert round(report["aggregate"][metric], 4) == pinned
    # And a zero-shift profile's forge measures the same numbers.
    measured = _profile_report(db_path, "desktop_cpu")
    for metric, pinned in BASELINE["default"].items():
        assert round(measured[metric], 4) == pinned


def test_shifted_profiles_measure_their_recorded_cost(tmp_path):
    db_path = build_fixture_db(tmp_path / "fixture.db")
    for profile_id in ("phone_low", "pi_zero"):
        measured = _profile_report(db_path, profile_id)
        # The recorded cost is real: precision below baseline,
        # recall and MRR untouched. Not an improvement — pinned
        # as measured so a future retune must update the record.
        assert round(measured["mean_precision"], 4) == 0.8519
        assert measured["mean_precision"] < BASELINE["default"]["mean_precision"]
        assert round(measured["mean_recall"], 4) == BASELINE["default"]["mean_recall"]


# ── TF-043's findings as arithmetic ──────────────────────────────────────────

def test_geonames_composite_rederived_from_named_constants():
    assert math.isclose(geonames_composite(), GEONAMES_COMPOSITE, abs_tol=1e-9)
    assert GEONAMES_COMPOSITE == 0.285
    assert geonames_composite() < scoring.MIN_MEMORY_SCORE


def test_default_forge_has_no_retrieval_weights():
    forge = KnowledgeForge(db_path=Path("/nonexistent/tf044.db"))
    assert forge.retrieval_weights is None
