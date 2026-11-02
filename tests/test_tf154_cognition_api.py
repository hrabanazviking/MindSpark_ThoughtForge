"""
TF-154 — The cognition API: turn-in/turn-out
with traces (TF-084), versioned (TF-034
consumed).

The proof lives as verifiable data in
``src/thoughtforge/cognition_api.py``
(``TURN_IN_FIELDS``, ``TURN_OUT_FIELDS``,
``COGNITION_RULES``, ``COGNITION_PROBES``,
checked by ``cognition_api_problems()``); the
doc ``docs/architecture/COGNITION_API.md`` is
filed verbatim as the module's render.

Pinned here:

  * the turn-out's response fields are the real
    FinalResponseRecord's, plus contract_version
    (TF-034's ``API_VERSION``, consumed, never
    reinvented), trace, trace_format and
    trace_status,
  * a knowledge-only core turn through the API
    (the TF-084 fixture shape) returns a clean
    turn-out with mode knowledge_only and its
    TF-084 joined trace included, validated by
    ``joined_trace_problems()``, its turn_id
    agreeing with the turn-out's,
  * the exported trace is detached JSON-safe
    data: mutating it cannot touch the core's
    retained state (proven by re-exporting from
    the core and comparing),
  * trace_status is honest: ``included`` /
    ``not_requested`` (want_trace=False) /
    ``unavailable: <reason>`` — no trace source,
    the core returned none, or a trace that
    failed its checker is never filed as good,
  * an invalid turn-in raises ValueError naming
    the problem (empty user_text, num_drafts=0,
    per TF-034's ruling) — never silently
    coerced,
  * only the kwargs the caller actually set are
    passed to ``core.think()``; the core is
    built lazily,
  * the cognition-api doc matches its render
    verbatim, the checker is clean, the probes
    pass 7/7, and TF-016's docket totals are
    unchanged (TF-154 has no docket share).

Proof grade: FIXTURE — knowledge-only core on
this VM; a live model turn through the
cognition API is NOT PROVEN, never claimed.
All fixtures synthetic (TF-017); no canon.
"""

from __future__ import annotations

import copy
import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.api_surface import (
    API_VERSION,
    api_manifest,
    verify_api_surface,
)
from thoughtforge.cognition_api import (
    COGNITION_API_DOC,
    COGNITION_API_FORMAT,
    COGNITION_PROBES,
    COGNITION_RULES,
    TURN_IN_FIELDS,
    TURN_IN_IDS,
    TURN_OUT_FIELDS,
    TURN_OUT_IDS,
    CognitionAPI,
    cognition_api_as_data,
    cognition_api_problems,
    cognition_contract,
    cognition_rule_problems,
    render_cognition_api_markdown,
    run_cognition_probes,
    turn_field_problems,
    turn_in_problems,
    turn_out_problems,
)
from thoughtforge.inference.joined_trace import (
    JOINED_FORMAT,
    joined_trace_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERY = "What time does the ferry leave?"


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _stub_record(user_text: str) -> dict:
    return {
        "text": f"stub answer to: {user_text}",
        "response_id": "resp_stub", "turn_id": "turn_stub",
        "mode": "stub_mode", "enforcement_passed": True,
        "citations": ["frag-1"], "token_count": 4,
    }


def _stub_api(**kwargs) -> CognitionAPI:
    return CognitionAPI(turn_callable=_stub_record, **kwargs)


def _knowledge_api():
    from thoughtforge.cognition.core import ThoughtForgeCore

    tmp = tempfile.mkdtemp(prefix="tf154-test-")
    core = ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    return CognitionAPI(core=core), core


def _real_trace():
    _api, core = _knowledge_api()
    record = core.think(QUERY)
    trace = core.last_turn_joined_trace()
    assert trace is not None and joined_trace_problems(trace) == []
    return core, record, trace


class _FakeCore:
    """A core-shaped double recording think() kwargs."""

    def __init__(self, trace=None) -> None:
        self.calls: list = []
        self._trace = trace

    def think(self, user_text, **kwargs):
        self.calls.append((user_text, kwargs))
        return SimpleNamespace(
            text="fake answer", response_id="resp_fake",
            turn_id="turn_fake", mode="fake_mode",
            enforcement_passed=True, citations=["frag-1"],
            token_count=3)

    def last_turn_joined_trace(self):
        return copy.deepcopy(self._trace)


# ── The filed estate ─────────────────────────────────────────────────────────


class TestFiling:
    def test_problems_empty(self) -> None:
        assert cognition_api_problems() == []

    def test_format_and_doc(self) -> None:
        assert COGNITION_API_FORMAT == "thoughtforge-cognition-api/1"
        assert COGNITION_API_DOC == "docs/architecture/COGNITION_API.md"

    def test_turn_in_fields_are_five_unique_and_complete(self) -> None:
        assert TURN_IN_IDS == (
            "user_text", "retrieval_path", "num_drafts",
            "history", "want_trace")
        assert len({f["id"] for f in TURN_IN_FIELDS}) == 5
        for field in TURN_IN_FIELDS:
            assert turn_field_problems(field) == []
        by_id = {f["id"]: f for f in TURN_IN_FIELDS}
        assert by_id["user_text"]["required"] is True
        for optional in ("retrieval_path", "num_drafts",
                         "history", "want_trace"):
            assert by_id[optional]["required"] is False
        assert by_id["want_trace"]["default"] is True
        assert ">= 1" in by_id["num_drafts"]["summary"]

    def test_turn_out_fields_are_eleven_unique_and_complete(self) -> None:
        assert TURN_OUT_IDS == (
            "text", "response_id", "turn_id", "mode",
            "enforcement_passed", "citations", "token_count",
            "contract_version", "trace", "trace_format",
            "trace_status")
        assert len({f["id"] for f in TURN_OUT_FIELDS}) == 11
        for field in TURN_OUT_FIELDS:
            assert turn_field_problems(field) == []
            assert field["source"]
        by_id = {f["id"]: f for f in TURN_OUT_FIELDS}
        assert by_id["text"]["source"] == "FinalResponseRecord.text"
        assert by_id["contract_version"]["source"] == \
            "TF-034 API_VERSION"

    def test_rules_are_six_unique(self) -> None:
        ids = [r["id"] for r in COGNITION_RULES]
        assert len(ids) == len(set(ids)) == 6
        assert set(ids) == {
            "version_is_tf034",
            "turn_out_fields_from_final_response",
            "trace_is_tf084_joined",
            "trace_is_detached_json_safe",
            "no_trace_is_filed_never_faked",
            "invalid_turn_in_raises_named"}
        for rule in COGNITION_RULES:
            assert cognition_rule_problems(rule) == []

    def test_validators_are_empty_safe(self) -> None:
        assert turn_in_problems(None)
        assert turn_in_problems({})
        assert turn_in_problems("not a mapping")
        assert turn_out_problems(None)
        assert turn_out_problems({})
        assert cognition_rule_problems(None)
        assert turn_field_problems(None)

    def test_contract_data_is_tf034_versioned(self) -> None:
        contract = cognition_contract()
        assert contract["format"] == COGNITION_API_FORMAT
        assert contract["contract_version"] == API_VERSION
        assert contract["trace_format"] == JOINED_FORMAT
        assert json.loads(json.dumps(contract)) == contract
        assert verify_api_surface() == []
        data = cognition_api_as_data()
        assert data["format"] == COGNITION_API_FORMAT
        assert data["doc"] == COGNITION_API_DOC
        assert data["manifest"] == api_manifest()
        assert data["manifest"]["version"] == API_VERSION
        assert data["measured"] == {"probes_passed": 7, "probes_total": 7}


# ── The turn ─────────────────────────────────────────────────────────────────


class TestTurn:
    def test_knowledge_only_turn_is_clean(self) -> None:
        api, _core = _knowledge_api()
        out = api.turn(QUERY)
        assert set(out) == set(TURN_OUT_IDS)
        assert out["mode"] == "knowledge_only"
        assert out["text"]
        assert out["response_id"] and out["turn_id"]
        assert isinstance(out["citations"], list)
        assert isinstance(out["enforcement_passed"], bool)
        assert out["contract_version"] == API_VERSION
        assert turn_out_problems(out) == []

    def test_turn_with_injected_callable_maps_record_fields(self) -> None:
        out = _stub_api().turn("hello forge")
        assert out["text"] == "stub answer to: hello forge"
        assert out["response_id"] == "resp_stub"
        assert out["turn_id"] == "turn_stub"
        assert out["mode"] == "stub_mode"
        assert out["enforcement_passed"] is True
        assert out["citations"] == ["frag-1"]
        assert out["token_count"] == 4
        assert out["contract_version"] == API_VERSION

    def test_turn_with_object_record_maps_the_same_fields(self) -> None:
        def turn(user_text: str):
            return SimpleNamespace(
                text="object answer", response_id="resp_obj",
                turn_id="turn_obj", mode="obj_mode",
                enforcement_passed=False, citations=[],
                token_count=0)

        out = CognitionAPI(turn_callable=turn).turn("hi")
        assert out["text"] == "object answer"
        assert out["turn_id"] == "turn_obj"
        assert out["enforcement_passed"] is False
        assert turn_out_problems(out) == []

    def test_construction_builds_no_core(self) -> None:
        assert CognitionAPI()._core is None
        assert _stub_api()._core is None

    def test_core_factory_is_lazy(self) -> None:
        built: list = []
        _api, core = _knowledge_api()

        def factory():
            built.append(1)
            return core

        api = CognitionAPI(core_factory=factory)
        assert built == [] and api._core is None
        out = api.turn(QUERY)
        assert built == [1]
        assert out["mode"] == "knowledge_only"
        assert out["trace_status"] == "included"

    def test_only_set_kwargs_are_passed_to_think(self) -> None:
        core = _FakeCore()
        CognitionAPI(core=core).turn("hello")
        assert core.calls == [("hello", {})]
        core = _FakeCore()
        CognitionAPI(core=core).turn(
            "hello", num_drafts=2, retrieval_path="sql")
        assert core.calls == [(
            "hello", {"num_drafts": 2, "retrieval_path": "sql"})]
        core = _FakeCore()
        history = [{"role": "user", "content": "earlier"}]
        CognitionAPI(core=core).turn("hello", history=history)
        assert core.calls == [("hello", {"history": history})]

    def test_turn_callable_takes_only_user_text(self) -> None:
        # TF-153's injection shape: a one-argument callable
        # serves even when the caller sets turn options.
        seen: list = []

        def turn(user_text: str):
            seen.append(user_text)
            return _stub_record(user_text)

        out = CognitionAPI(turn_callable=turn).turn(
            "hello", num_drafts=2)
        assert seen == ["hello"]
        assert out["text"] == "stub answer to: hello"

    def test_turn_out_is_json_safe(self) -> None:
        api, _core = _knowledge_api()
        out = api.turn(QUERY)
        loaded = json.loads(json.dumps(out))
        # Serialisability and structure, not equality
        # across two turns (the trace carries timings).
        assert set(loaded) == set(TURN_OUT_IDS)
        assert loaded["trace_status"] == "included"
        assert loaded["trace"]["format"] == JOINED_FORMAT
        out = _stub_api().turn("hi")
        assert json.loads(json.dumps(out)) == out


# ── The traces ───────────────────────────────────────────────────────────────


class TestTraces:
    def test_included_trace_is_tf084_joined_and_clean(self) -> None:
        api, _core = _knowledge_api()
        out = api.turn(QUERY)
        trace = out["trace"]
        assert out["trace_status"] == "included"
        assert out["trace_format"] == JOINED_FORMAT
        assert trace["format"] == JOINED_FORMAT
        assert joined_trace_problems(trace) == []
        for section in ("cognition", "inference", "reconciliation"):
            assert section in trace

    def test_trace_turn_id_agrees_with_turn_out(self) -> None:
        api, _core = _knowledge_api()
        out = api.turn(QUERY)
        assert out["trace"]["turn_id"] == out["turn_id"]
        assert out["trace"]["cognition"]["turn_id"] == out["turn_id"]
        assert out["trace"]["inference"]["turn_id"] == out["turn_id"]

    def test_trace_is_detached_from_the_core(self) -> None:
        api, core = _knowledge_api()
        out = api.turn(QUERY)
        original = out["turn_id"]
        out["trace"]["turn_id"] = "turn_mutated"
        out["trace"]["cognition"]["turn_id"] = "turn_mutated"
        reexported = core.last_turn_joined_trace()
        assert reexported["turn_id"] == original
        assert reexported["turn_id"] != "turn_mutated"
        assert joined_trace_problems(reexported) == []

    def test_want_trace_false_files_not_requested(self) -> None:
        api, _core = _knowledge_api()
        out = api.turn(QUERY, want_trace=False)
        assert out["trace_status"] == "not_requested"
        assert out["trace"] is None
        assert out["trace_format"] is None
        assert out["mode"] == "knowledge_only"
        assert turn_out_problems(out) == []

    def test_injected_turn_without_trace_source_is_unavailable(
            self) -> None:
        out = _stub_api().turn("hello forge")
        assert out["trace"] is None
        assert out["trace_format"] is None
        assert out["trace_status"].startswith("unavailable: ")
        assert "no trace source" in out["trace_status"]
        assert turn_out_problems(out) == []

    def test_trace_callable_supplies_the_trace(self) -> None:
        _core, record, trace = _real_trace()

        def turn(user_text: str):
            row = _stub_record(user_text)
            row["turn_id"] = record.turn_id
            return row

        api = CognitionAPI(
            turn_callable=turn,
            trace_callable=lambda: copy.deepcopy(trace))
        out = api.turn("hello")
        assert out["trace_status"] == "included"
        assert out["trace"] == trace
        assert turn_out_problems(out) == []

    def test_fake_core_trace_turn_id_disagreement_is_unavailable(
            self) -> None:
        _core, _record, trace = _real_trace()
        trace = copy.deepcopy(trace)
        # The top-level id is bent to the fake record's,
        # but the cognition / inference sections keep the
        # real id — the joined checker itself disagrees,
        # so the trace must not be filed as this turn's.
        trace["turn_id"] = "turn_fake"
        core = _FakeCore(trace=trace)
        core.think = lambda user_text, **kwargs: SimpleNamespace(
            text="fake", response_id="r", turn_id="turn_fake",
            mode="fake_mode", enforcement_passed=True,
            citations=[], token_count=1)  # type: ignore[method-assign]
        out = CognitionAPI(core=core).turn("hi")
        assert out["trace_status"].startswith("unavailable: ")
        assert "disagrees" in out["trace_status"]

    def test_core_returning_no_trace_is_unavailable(self) -> None:
        core = _FakeCore(trace=None)
        out = CognitionAPI(core=core).turn("hello")
        assert out["trace"] is None
        assert out["trace_status"].startswith("unavailable: ")
        assert "no joined trace" in out["trace_status"]
        assert turn_out_problems(out) == []

    def test_trace_failing_its_checker_is_never_filed_as_good(
            self) -> None:
        _core, record, trace = _real_trace()
        bad = copy.deepcopy(trace)
        bad["reconciliation"]["inference_call_count"] = 99

        def turn(user_text: str):
            row = _stub_record(user_text)
            row["turn_id"] = record.turn_id
            return row

        api = CognitionAPI(
            turn_callable=turn, trace_callable=lambda: bad)
        out = api.turn("hello")
        assert out["trace"] is None
        assert out["trace_status"].startswith("unavailable: ")
        assert "failed its checker" in out["trace_status"]
        assert "reconciliation" in out["trace_status"]
        assert turn_out_problems(out) == []

    def test_trace_turn_id_disagreement_is_unavailable(self) -> None:
        _core, _record, trace = _real_trace()
        # The stub record's turn_id (turn_stub) is not the
        # trace's, so the trace must not be filed as this
        # turn's story.
        api = CognitionAPI(
            turn_callable=_stub_record,
            trace_callable=lambda: copy.deepcopy(trace))
        out = api.turn("hello")
        assert out["trace"] is None
        assert out["trace_status"].startswith("unavailable: ")
        assert "disagrees" in out["trace_status"]
        assert turn_out_problems(out) == []


# ── Refusals and errors ──────────────────────────────────────────────────────


class TestRefusals:
    def test_empty_user_text_raises_named(self) -> None:
        for bad in ("", None, 42):
            with pytest.raises(ValueError, match="user_text"):
                _stub_api().turn(bad)  # type: ignore[arg-type]

    def test_num_drafts_below_one_raises_named(self) -> None:
        for bad in (0, -1):
            with pytest.raises(ValueError, match="num_drafts"):
                _stub_api().turn("hi", num_drafts=bad)

    def test_num_drafts_non_integer_raises_named(self) -> None:
        for bad in ("2", 1.5, True):
            with pytest.raises(ValueError, match="num_drafts"):
                _stub_api().turn("hi", num_drafts=bad)  # type: ignore[arg-type]

    def test_retrieval_path_non_string_raises_named(self) -> None:
        with pytest.raises(ValueError, match="retrieval_path"):
            _stub_api().turn("hi", retrieval_path=42)  # type: ignore[arg-type]

    def test_want_trace_non_bool_raises_named(self) -> None:
        with pytest.raises(ValueError, match="want_trace"):
            _stub_api().turn("hi", want_trace="yes")  # type: ignore[arg-type]

    def test_invalid_turn_in_builds_no_core(self) -> None:
        api = CognitionAPI()
        with pytest.raises(ValueError, match="user_text"):
            api.turn("")
        assert api._core is None

    def test_turn_in_problems_judges_directly(self) -> None:
        assert turn_in_problems({"user_text": "hi"}) == []
        assert turn_in_problems(
            {"user_text": "hi", "num_drafts": 1,
             "want_trace": False}) == []
        assert turn_in_problems({"num_drafts": 2})
        assert any("user_text" in p
                   for p in turn_in_problems({"user_text": ""}))
        assert any("num_drafts" in p for p in turn_in_problems(
            {"user_text": "hi", "num_drafts": 0}))
        assert any("not a filed" in p for p in turn_in_problems(
            {"user_text": "hi", "surprise": 1}))

    def test_turn_out_problems_catches_tampering(self) -> None:
        api, _core = _knowledge_api()
        out = api.turn(QUERY)
        assert turn_out_problems(out) == []
        missing = dict(out)
        del missing["trace_status"]
        assert any("trace_status" in p
                   for p in turn_out_problems(missing))
        wrong_version = dict(out, contract_version="9.9.9")
        assert any("contract_version" in p
                   for p in turn_out_problems(wrong_version))
        bad_status = dict(out, trace_status="filed_anyway")
        assert any("trace_status" in p
                   for p in turn_out_problems(bad_status))
        lying_status = dict(out, trace_status="not_requested")
        assert any("trace" in p
                   for p in turn_out_problems(lying_status))

    def test_turn_out_problems_catches_a_bad_trace(self) -> None:
        api, _core = _knowledge_api()
        out = api.turn(QUERY)
        bad_format = copy.deepcopy(out)
        bad_format["trace"] = dict(out["trace"], format="other/9")
        bad_format["trace_format"] = "other/9"
        problems = turn_out_problems(bad_format)
        assert any("format" in p for p in problems)
        mismatched = copy.deepcopy(out)
        mismatched["trace"]["turn_id"] = "turn_someone_else"
        problems = turn_out_problems(mismatched)
        assert any("turn_id" in p for p in problems)


# ── State, probes, doc, docket ───────────────────────────────────────────────


class TestStateProbesDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_cognition_probes()
        assert len(probes) == len(COGNITION_PROBES) == 7
        assert [p["id"] for p in probes] == [
            "knowledge_turn_with_trace_included",
            "trace_turn_id_agrees_and_sections_present",
            "trace_is_detached_from_core",
            "want_trace_false_files_not_requested",
            "injected_turn_without_trace_source_unavailable",
            "version_is_tf034",
            "invalid_turn_in_raises_named"]
        assert all(p["passed"] for p in probes)

    def test_doc_matches_render(self) -> None:
        assert _read(COGNITION_API_DOC) == \
            render_cognition_api_markdown()
        doc = _read(COGNITION_API_DOC)
        assert "Turn-In / Turn-Out" in doc
        assert "NOT PROVEN" in doc
        assert API_VERSION in doc
        assert JOINED_FORMAT in doc
        assert "TF-155" in doc and "TF-162" in doc
        assert "TF-153" in doc

    def test_docket_untouched_totals_unchanged(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, " \
            "101 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT" in normalized
