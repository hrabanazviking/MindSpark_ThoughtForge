"""
TF-098 — Cost discipline: refinement passes
capped per profile (a second small-model call is
a budget, not a habit).

The estate lives in
``src/thoughtforge/refinement/cost_discipline.py``
(``COST_RULES`` / ``COST_FIXTURES`` /
``COST_RECORD_FIXTURES`` as data,
``resolve_refine_budget`` / ``profile_refine_cap``
/ ``refine_cost``, the checkers
``cost_discipline_problems`` /
``cost_discipline_live_problems``, format
``thoughtforge-cost-discipline/1``), with the cap
enforced inside ``FragmentSalvage.forge``
(``profile_id`` keyword) and the core passing its
selected profile id through ``_compose_final``.
Write-up: ``docs/architecture/COST_DISCIPLINE.md``.

Measured start: a direct ``forge(max_passes=5)``
made 5 engine calls and returned ``refine_pass_5``
— beyond every shipped profile cap (largest 3) —
because salvage knew no profile existed and the
core's resolved max_passes was the only
enforcement in the tree (D-COST-1, found by this
slice, FIXED here); and refine cost was counted
nowhere (TF-097 records each pass; nothing
totalled them).

Pinned here:

  * the filed estate: both checkers clean, the
    doc present, proof grades stated, caps read
    live from the profile JSONs for all six
    shipped profiles,
  * every cap fixture lands exactly as filed —
    a request of 5 under each profile makes
    exactly the cap's engine calls (pi_zero: 0),
  * a cap is a ceiling not a target: a caller
    below the cap gets its own count; no profile
    means no invented cap (the boundary fixture),
  * the resolver's error legs raise by name
    (unknown profile, negative, non-integer),
  * the cost accounting over TF-097's records:
    attempted / completed / adopted / wasted /
    failed, reported sums apart from wall, an
    unsound record refused,
  * the core leg: a core built on phone_low
    passes that profile id into forge (the
    resolved cap's number unchanged).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.refinement.cost_discipline import (
    COST_DOC,
    COST_FIXTURES,
    COST_FORMAT,
    COST_RECORD_FIXTURES,
    COST_RULES,
    DEFAULT_REFINE_CAP,
    STUB_DURATION_MS,
    cost_discipline_as_data,
    cost_discipline_live_problems,
    cost_discipline_problems,
    profile_refine_cap,
    profile_refine_caps,
    refine_cost,
    resolve_refine_budget,
    run_cost_fixture,
)
from thoughtforge.refinement.rescue_provenance import (
    run_rescue_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_FIXTURE = {f["id"]: f for f in COST_FIXTURES}

_EXPECTED_CAPS = {
    "pi_zero": 0, "phone_low": 1, "pi_5": 2,
    "desktop_cpu": 2, "desktop_gpu": 3, "server_gpu": 3,
}


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert cost_discipline_problems() == []
        assert cost_discipline_live_problems() == []

    def test_format_doc_and_proof(self):
        data = cost_discipline_as_data()
        assert data["format"] == COST_FORMAT
        assert data["cost_doc"] == COST_DOC
        assert (REPO_ROOT / COST_DOC).exists()
        assert data["proof"]["grade"] == "fixture"
        assert data["proof"]["d_cost_1"] == "fixed"
        assert data["proof"]["live_model_refine_cost"] == \
            "not_proven"
        json.dumps(data)

    def test_rule_counts(self):
        ids = [r["id"] for r in COST_RULES]
        assert len(ids) == len(set(ids)) == 10

    def test_fixture_counts(self):
        assert len(COST_FIXTURES) == 9
        assert len(COST_RECORD_FIXTURES) == 4
        profiled = [f for f in COST_FIXTURES if f["profile_id"]]
        assert len(profiled) == 7  # six profiles + below-cap


# ── The caps ───────────────────────────────────────────────────


class TestCaps:
    def test_caps_read_live_match_the_profiles(self):
        assert profile_refine_caps() == _EXPECTED_CAPS
        for pid, cap in _EXPECTED_CAPS.items():
            assert profile_refine_cap(pid) == cap

    def test_every_profile_fixture_lands_on_its_cap(self):
        for pid in _EXPECTED_CAPS:
            fixture = _FIXTURE[f"cap_{pid}"]
            result, engine, budget = run_cost_fixture(fixture["id"])
            assert budget.profile_cap == _EXPECTED_CAPS[pid]
            assert engine.calls == _EXPECTED_CAPS[pid]
            assert result.passes_used == _EXPECTED_CAPS[pid]
            assert result.salvage_path == fixture["expected_path"]

    def test_zero_means_zero(self):
        result, engine, _ = run_cost_fixture("cap_pi_zero")
        assert engine.calls == 0
        assert result.salvage_path == "best_draft"
        assert result.rescue_record["refine_attempted"] is True
        assert result.rescue_record["refine_passes"] == []

    def test_cap_is_a_ceiling_not_a_target(self):
        result, engine, budget = run_cost_fixture("caller_below_cap")
        assert budget.requested_passes == 1
        assert budget.profile_cap == 3
        assert budget.effective_passes == 1
        assert budget.governed_by == "caller_request"
        assert engine.calls == 1

    def test_profile_governs_when_request_exceeds_cap(self):
        budget = resolve_refine_budget(5, profile_id="phone_low")
        assert budget.effective_passes == 1
        assert budget.governed_by == "profile"

    def test_standing_default_no_profile(self):
        result, engine, budget = run_cost_fixture(
            "standing_default_no_profile")
        assert budget.requested_passes == DEFAULT_REFINE_CAP == 2
        assert budget.profile_cap is None
        assert budget.governed_by == "standing_default"
        assert engine.calls == 2

    def test_no_profile_no_invented_cap(self):
        # The boundary, pinned: a no-profile request
        # of 5 stands at 5, exactly as before TF-098.
        result, engine, budget = run_cost_fixture(
            "caller_request_no_profile")
        assert budget.effective_passes == 5
        assert engine.calls == 5
        assert result.salvage_path == "refine_pass_5"


# ── The resolver's error legs ──────────────────────────────────


class TestResolverErrors:
    def test_unknown_profile_raises_by_name(self):
        with pytest.raises(ValueError) as excinfo:
            profile_refine_cap("bogus_profile")
        assert "bogus_profile" in str(excinfo.value)
        assert "phone_low" in str(excinfo.value)

    def test_negative_request_raises_by_name(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_refine_budget(-1)
        assert "-1" in str(excinfo.value)

    def test_non_integer_request_raises_by_name(self):
        for bad in (1.5, "two", True):
            with pytest.raises(ValueError):
                resolve_refine_budget(bad)

    def test_forge_with_unknown_profile_raises(self):
        from thoughtforge.knowledge.models import (
            ActivatedRecord,
            CandidateRecord,
            CandidateScores,
            MemoryActivationBundle,
        )
        from thoughtforge.refinement.salvage import FragmentSalvage

        candidates = [CandidateRecord(
            candidate_id="c0", mode="practical",
            text="The ferry leaves at dawn.",
            token_estimate=5, scores=CandidateScores())]
        bundle = MemoryActivationBundle(
            activated_records=[ActivatedRecord(
                "Q123", "entity", 0.8, "cue",
                raw={"qid": "Q123"})],
            total_retrieved=1, retrieval_confidence=0.8)
        with pytest.raises(ValueError) as excinfo:
            FragmentSalvage().forge(
                candidates, bundle, engine=object(),
                profile_id="bogus_profile")
        assert "bogus_profile" in str(excinfo.value)


# ── The cost accounting ────────────────────────────────────────


class TestRefineCost:
    def test_record_fixtures_total_as_filed(self):
        for fixture in COST_RECORD_FIXTURES:
            record = run_rescue_fixture(
                fixture["rescue_fixture"]).rescue_record
            cost = refine_cost(record).to_dict()
            for key, want in fixture["expected"].items():
                assert cost[key] == want, fixture["id"]

    def test_wasted_spend_is_named(self):
        record = run_rescue_fixture(
            "refine_adopt_then_fabricate").rescue_record
        cost = refine_cost(record)
        assert cost.passes_not_adopted == 1
        assert cost.reported_ms == 2 * 3.5
        assert cost.wall_ms >= 0

    def test_failed_attempt_costs_an_attempt_not_a_pass(self):
        record = run_rescue_fixture(
            "refine_engine_failure_falls_back").rescue_record
        cost = refine_cost(record)
        assert cost.passes_attempted == 1
        assert cost.passes_completed == 0
        assert cost.passes_failed == 1
        assert cost.reported_ms is None

    def test_cap_fixture_cost_matches_engine_calls(self):
        result, engine, _ = run_cost_fixture("cap_desktop_gpu")
        cost = refine_cost(result.rescue_record)
        assert cost.passes_completed == engine.calls == 3
        assert cost.reported_ms == 3 * STUB_DURATION_MS
        assert cost.reported_passes == 3

    def test_unsound_record_is_never_totalled(self):
        with pytest.raises(ValueError):
            refine_cost({"format": "not-a-record"})
        record = run_rescue_fixture("refine_adopted").rescue_record
        record["final"]["passes_used"] = 99
        with pytest.raises(ValueError):
            refine_cost(record)


# ── The core leg ───────────────────────────────────────────────

_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning.")


class _StubEngine:
    def generate(self, prompt, params=None):
        return SimpleNamespace(
            text=_TEXT, tokens_generated=len(_TEXT.split()),
            tokens_prompt=42, total_tokens=70, duration_ms=12,
            tokens_per_second=100.0, stopped_reason="eos",
            model_path="/models/test-q4.gguf",
            profile_id="desktop_cpu")


class TestCoreLeg:
    def test_core_passes_its_profile_id_and_cap_holds(self):
        tmp = tempfile.mkdtemp()
        core = ThoughtForgeCore(
            memory_dir=Path(tmp) / "memory",
            db_path=Path(tmp) / "nonexistent.db",
            model_path=None,
            config={"conversation": {"profile_id": "phone_low"}})
        assert core._conversation_config.profile_id == "phone_low"
        assert core._conversation_config.refinement_max_passes() == 1
        core._engine = _StubEngine()
        core.think("What time does the ferry leave?")
        trace = core.last_turn_trace()
        assert trace is not None
        record = trace["final"]["rescue_record"]
        assert record is not None
        # phone_low's cap is 1: at most one refine pass ran.
        assert len(record["refine_passes"]) <= 1
        cost = refine_cost(record)
        assert cost.passes_completed <= 1
