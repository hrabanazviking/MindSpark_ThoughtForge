"""
TF-097 — Provenance of rescue: a salvaged answer
carries its rescue record into the trace (TF-084).

The estate lives in
``src/thoughtforge/refinement/rescue_provenance.py``
(``RESCUE_RULES`` / ``RESCUE_FIXTURES`` as data,
``rescue_record_problems`` /
``rescue_record_result_problems``, the checkers
``rescue_provenance_problems`` /
``rescue_provenance_live_problems``, formats
``thoughtforge-rescue-provenance/1`` and
``thoughtforge-rescue-record/1``), with the record
built inside ``FragmentSalvage.forge()``, carried
on ``SalvageResult`` and the final response
record, and exported in the TF-038 trace's final
section — so TF-084's joined trace carries it.
Write-up: ``docs/architecture/RESCUE_PROVENANCE.md``.

Measured start: TF-096's adopted and
voice-rejected refine fixtures both returned
``refine_pass_1`` / ``passes_used == 1`` —
indistinguishable — and salvage's refine passes
were recorded nowhere in any trace (D-TRACE-1,
filed OPEN by TF-084, FIXED here).

Pinned here:

  * the filed estate: both checkers clean, the
    doc present, proof grade stated, the wiring
    anchors in salvage.py / core / traces,
  * every filed fixture's record lands as filed —
    path, final_source, final_pass, per-pass
    outcomes — and judges clean against its own
    result,
  * attempted ≠ adopted: the two-pass fixture's
    path says pass 2 while the record says the
    text came from pass 1,
  * a failed attempt is provenance: the engine-
    failure fixture falls back to best_draft with
    passes_used 0 and the failed pass recorded,
    uncounted, with its error,
  * rejection reasons are the guards' own lists
    (TF-088's ``new_number:50``, TF-096's
    ``vague handwaving``), latencies named never
    conflated (the stub's reported 3.5 ms beside
    the measured wall),
  * the checker catches tampering by name,
  * the trace leg: a real core turn's TF-038
    trace and TF-084 joined trace both carry the
    record and judge clean whole; a knowledge-only
    final carries None, stated,
  * TF-084's named-gap rule now records the
    closure (same id, D-TRACE-1 FIXED by TF-097).
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.traces import trace_problems
from thoughtforge.inference.joined_trace import (
    TRACE_RULES,
    joined_trace_problems,
)
from thoughtforge.refinement.rescue_provenance import (
    FINAL_SOURCES,
    PASS_OUTCOMES,
    RESCUE_DOC,
    RESCUE_FIXTURES,
    RESCUE_FORMAT,
    RESCUE_RECORD_FORMAT,
    RESCUE_RULES,
    STUB_DURATION_MS,
    rescue_provenance_as_data,
    rescue_provenance_live_problems,
    rescue_provenance_problems,
    rescue_record_problems,
    rescue_record_result_problems,
    run_rescue_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_FIXTURE = {f["id"]: f for f in RESCUE_FIXTURES}


def _record(fixture_id: str):
    result = run_rescue_fixture(fixture_id)
    assert result.rescue_record is not None
    return result, result.rescue_record


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert rescue_provenance_problems() == []
        assert rescue_provenance_live_problems() == []

    def test_format_doc_and_proof(self):
        data = rescue_provenance_as_data()
        assert data["format"] == RESCUE_FORMAT
        assert data["record_format"] == RESCUE_RECORD_FORMAT
        assert data["rescue_doc"] == RESCUE_DOC
        assert (REPO_ROOT / RESCUE_DOC).exists()
        assert data["proof"]["d_trace_1"] == "fixed"
        assert data["proof"]["live_model_rescue_record"] == \
            "not_proven"
        assert data["proof"]["inference_section"] == "unchanged"
        json.dumps(data)

    def test_rule_and_vocabulary_counts(self):
        ids = [r["id"] for r in RESCUE_RULES]
        assert len(ids) == len(set(ids)) == 9
        assert PASS_OUTCOMES == (
            "adopted", "rejected_fabrication", "rejected_voice",
            "not_improved", "failed")
        assert FINAL_SOURCES == (
            "empty", "selected_draft", "refine_pass")

    def test_fixture_counts(self):
        assert len(RESCUE_FIXTURES) == 8
        kinds = [f["kind"] for f in RESCUE_FIXTURES]
        assert kinds.count("refine") == 6
        assert kinds.count("selection") == 1
        assert kinds.count("empty") == 1


# ── The fixtures' records ──────────────────────────────────────


class TestFixtureRecords:
    def test_every_fixture_record_judges_clean(self):
        for fixture in RESCUE_FIXTURES:
            result, record = _record(fixture["id"])
            assert rescue_record_problems(record) == []
            assert rescue_record_result_problems(
                record, result) == []
            assert record["format"] == RESCUE_RECORD_FORMAT

    def test_empty_record(self):
        result, record = _record("empty_candidates")
        assert result.salvage_path == "empty"
        assert record["candidates"] == []
        assert record["selection"]["selected_candidate_id"] is None
        assert record["refine_passes"] == []
        assert record["final"]["final_source"] == "empty"

    def test_best_draft_record(self):
        result, record = _record("best_draft_no_engine")
        assert record["refine_attempted"] is False
        assert record["final"]["final_source"] == "selected_draft"
        assert record["final"]["final_pass"] is None
        selected = record["selection"]["selected_candidate_id"]
        rows = {r["candidate_id"]: r for r in record["candidates"]}
        assert rows[selected]["text_chars"] == len(result.text)
        # sizes and scores, never texts
        assert result.text not in json.dumps(record)

    def test_adopted_pass_record(self):
        _, record = _record("refine_adopted")
        (entry,) = record["refine_passes"]
        assert entry["outcome"] == "adopted"
        assert entry["counted_in_passes_used"] is True
        assert entry["fabrication_problems"] == []
        assert entry["voice_problems"] == []
        assert record["final"]["final_pass"] == 1
        assert record["final"]["confidence"] == entry["score"]

    def test_attempted_is_not_adopted(self):
        # The naming debt in one fixture: the path says
        # pass 2 (attempted), the text came from pass 1.
        result, record = _record("refine_adopt_then_fabricate")
        assert result.salvage_path == "refine_pass_2"
        assert record["final"]["final_source"] == "refine_pass"
        assert record["final"]["final_pass"] == 1
        outcomes = [p["outcome"] for p in record["refine_passes"]]
        assert outcomes == ["adopted", "rejected_fabrication"]

    def test_fabrication_rejection_carries_the_guards_codes(self):
        _, record = _record("refine_rejected_fabrication")
        (entry,) = record["refine_passes"]
        assert entry["outcome"] == "rejected_fabrication"
        assert entry["fabrication_problems"] == ["new_number:50"]
        assert record["final"]["final_source"] == "selected_draft"
        assert record["final"]["final_pass"] is None

    def test_voice_rejection_carries_the_avoid_entry(self):
        _, record = _record("refine_rejected_voice")
        (entry,) = record["refine_passes"]
        assert entry["outcome"] == "rejected_voice"
        assert entry["voice_problems"] == ["vague handwaving"]
        assert record["selection"]["persona_filter_applied"] is True
        assert record["selection"]["persona_consistent_count"] == 1

    def test_not_improved_record(self):
        _, record = _record("refine_not_improved")
        (entry,) = record["refine_passes"]
        assert entry["outcome"] == "not_improved"
        assert entry["counted_in_passes_used"] is True
        assert record["final"]["final_source"] == "selected_draft"

    def test_failed_attempt_is_provenance(self):
        result, record = _record("refine_engine_failure_falls_back")
        assert result.salvage_path == "best_draft"
        assert result.passes_used == 0
        (entry,) = record["refine_passes"]
        assert entry["outcome"] == "failed"
        assert entry["error"] == "engine blew up"
        assert entry["counted_in_passes_used"] is False
        assert entry["score"] is None
        assert record["refine_attempted"] is True

    def test_latencies_named_never_conflated(self):
        _, record = _record("refine_adopted")
        (entry,) = record["refine_passes"]
        assert entry["reported_latency_ms"] == STUB_DURATION_MS
        assert isinstance(entry["latency_ms"], float)
        assert entry["latency_ms"] >= 0


# ── The checker catches tampering ──────────────────────────────


class TestChecker:
    def test_tampered_final_pass_named(self):
        _, record = _record("refine_adopt_then_fabricate")
        bad = copy.deepcopy(record)
        bad["final"]["final_pass"] = 2
        problems = rescue_record_problems(bad)
        assert any("final_pass" in p for p in problems)

    def test_tampered_passes_used_named(self):
        _, record = _record("refine_adopted")
        bad = copy.deepcopy(record)
        bad["final"]["passes_used"] = 0
        problems = rescue_record_problems(bad)
        assert any("passes_used" in p for p in problems)

    def test_rejection_without_reasons_named(self):
        _, record = _record("refine_rejected_fabrication")
        bad = copy.deepcopy(record)
        bad["refine_passes"][0]["fabrication_problems"] = []
        problems = rescue_record_problems(bad)
        assert any("fabrication" in p for p in problems)

    def test_result_mismatch_named(self):
        result, record = _record("refine_adopted")
        bad = copy.deepcopy(record)
        bad["fragments_kept"] += 1
        problems = rescue_record_result_problems(bad, result)
        assert any("fragments_kept" in p for p in problems)


# ── Into the trace ─────────────────────────────────────────────

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


def _core() -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None)


class TestTraceLeg:
    def test_core_turn_trace_and_join_carry_the_record(self):
        core = _core()
        core._engine = _StubEngine()
        core.think("What time does the ferry leave?")
        trace = core.last_turn_trace()
        assert trace is not None
        assert trace_problems(trace) == []
        record = trace["final"]["rescue_record"]
        assert record is not None
        assert rescue_record_problems(record) == []
        assert record["refine_attempted"] is True
        assert [p["outcome"] for p in record["refine_passes"]] == [
            "not_improved", "not_improved"]
        assert record["final"]["final_source"] == "selected_draft"
        joined = core.last_turn_joined_trace()
        assert joined is not None
        assert joined_trace_problems(joined) == []
        assert joined["cognition"]["final"]["rescue_record"] == record

    def test_knowledge_only_final_carries_none_stated(self):
        core = _core()
        core.think("What time does the ferry leave?")
        trace = core.last_turn_trace()
        assert trace is not None
        assert trace_problems(trace) == []
        assert trace["final"]["rescue_record"] is None

    def test_tf084_gap_rule_records_the_closure(self):
        rules = {r["id"]: r["rule"] for r in TRACE_RULES}
        text = rules["salvage_refine_passes_are_a_named_gap"]
        assert "D-TRACE-1" in text
        assert "TF-097" in text
        assert "FIXED" in text
