"""
TF-157 — The Samtal binding: a Samtal agent
seat's thinking layer via ThoughtForge, under
Samtal's own rules (equal seats, Astrid Law
untouched — the binding is cognition only).

The proof lives as verifiable data in
``src/thoughtforge/samtal_binding.py``
(``SEAT_ROWS``, ``RULE_ROWS``, ``CALL_ROWS``,
``FIELD_ROWS``, ``SAMTAL_BINDING_PROBES``,
checked by ``samtal_binding_problems()``);
the doc ``docs/architecture/SAMTAL_BINDING.md``
is filed verbatim as the module's render.

Pinned here:

  * the seat rows file one row per kind
    Samtal itself files (human / agent /
    subagent / bot / kindroid), every row
    routing through the one shared method,
    no seat privileged, no seat class
    invented by the binding,
  * Samtal's own rules are filed as
    constraints: equal seats; the Astrid Law
    untouched — join_seating / seat_healing /
    presence / room_administration /
    moderation are all filed NOT routed, and
    the binding class files no such method,
  * a seat's thinking call routes to
    TF-154's CognitionAPI by the filed rule
    (the incoming message's body is user_text;
    caller-supplied earlier messages ride as
    history, the seat's own words as
    assistant) and the reply is the filed
    Samtal message shape (the incoming room,
    the turn's text as body, the seat's name
    as author); memory calls route to
    TF-155's MemoryAPI's four ops, called,
    never reimplemented,
  * every field has a disposition — TF-154's
    turn-in and turn-out fields checked live
    against TF-154's own filings, and the
    seat token filed no-home: never filed,
    never passed, never stored,
  * the binding is DEFAULT DISABLED
    (``DEFAULT_ENABLED`` False, filed as
    data): while disabled every routed call
    returns the refusal as data naming the
    binding disabled and touches neither API
    (counting factories prove nothing is
    built); enabling is the explicit
    constructor flag / ``enable()`` act,
  * two differently-named, differently-kinded
    seats route identically in shape,
  * responses are detached JSON-safe data,
    contract_version is TF-034's
    ``API_VERSION``, the doc matches its
    render verbatim, the checker is clean,
    the probes pass 7/7, and TF-016's docket
    totals are unchanged (TF-157 has no
    docket share).

Proof grade: FIXTURE — a fake Samtal
consumer shaped on the filed seat / message
data; NO live Samtal server was routed and
NO Samtal code was imported or copied; live
end-to-end routing is NOT PROVEN (TF-170's
gate owns it), never claimed. All fixtures
synthetic (TF-017); no canon.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from thoughtforge.cognition.api_surface import API_VERSION
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition_api import (
    TURN_IN_FIELDS,
    TURN_OUT_FIELDS,
    CognitionAPI,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.memory_api import MemoryAPI
from thoughtforge.samtal_binding import (
    CALL_IDS,
    CALL_ROWS,
    DEFAULT_ENABLED,
    DISABLED_ERROR,
    FIELD_IDS,
    FIELD_ROWS,
    MESSAGE_FIELDS,
    NOT_ROUTED_CALL_IDS,
    ROUTED_CALL_IDS,
    RULE_IDS,
    RULE_ROWS,
    SAMTAL_BINDING_DOC,
    SAMTAL_BINDING_FORMAT,
    SAMTAL_BINDING_PROBES,
    SEAT_KINDS,
    SEAT_ROWS,
    SamtalBinding,
    assemble_turn_in,
    binding_response_problems,
    call_row_problems,
    field_row_problems,
    field_rows_for,
    render_samtal_binding_markdown,
    reply_message,
    rule_row_problems,
    run_samtal_binding_probes,
    samtal_binding_as_data,
    samtal_binding_contract,
    samtal_binding_problems,
    seat_request_problems,
    seat_row_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


class _RecordingCognitionAPI:
    """A recording stand-in for TF-154's API."""

    def __init__(self) -> None:
        self.turns: list[dict] = []

    def turn(self, user_text: str, **kwargs):
        self.turns.append({"user_text": user_text, **kwargs})
        return {
            "text": f"stub answer to: {user_text}",
            "response_id": "resp_stub",
            "turn_id": "turn_stub",
            "mode": "stub_mode",
            "enforcement_passed": True,
            "citations": [],
            "token_count": 3,
            "contract_version": API_VERSION,
            "trace": None,
            "trace_format": None,
            "trace_status": "unavailable: stub",
        }


def _chat_binding() -> tuple[SamtalBinding, _RecordingCognitionAPI]:
    fake = _RecordingCognitionAPI()
    return SamtalBinding(cognition_api=fake, enabled=True), fake


def _memory_binding(
    tmp_path: Path,
) -> tuple[SamtalBinding, MemoryStore]:
    store = MemoryStore(memory_dir=tmp_path / "store")
    binding = SamtalBinding(
        memory_api=MemoryAPI(store=store), enabled=True)
    return binding, store


_SEAT_A = {"name": "SeatA", "kind": "agent"}
_SEAT_B = {"name": "SeatB", "kind": "kindroid"}
_MESSAGE = {"id": 7, "room": 1, "author": "VolmarrViking",
            "author_kind": "human",
            "body": "What time does the ferry leave?",
            "ts": 0.0}
_HISTORY = [
    {"author": "SeatA", "body": "earlier words of mine"},
    {"author": "VolmarrViking", "body": "earlier question"},
]


# ── The filed estate ─────────────────────────────────────────────────────────


class TestFiling:
    def test_problems_empty(self) -> None:
        assert samtal_binding_problems() == []

    def test_format_doc_and_defaults(self) -> None:
        assert SAMTAL_BINDING_FORMAT == \
            "thoughtforge-samtal-binding/1"
        assert SAMTAL_BINDING_DOC == \
            "docs/architecture/SAMTAL_BINDING.md"
        assert DEFAULT_ENABLED is False
        assert "disabled" in DISABLED_ERROR

    def test_seat_rows_one_per_kind_all_equal(self) -> None:
        assert list(SEAT_KINDS) == [
            "human", "agent", "subagent", "bot", "kindroid"]
        assert [r["kind"] for r in SEAT_ROWS] == \
            list(SEAT_KINDS)
        for row in SEAT_ROWS:
            assert seat_row_problems(row) == []
            assert row["privileged"] is False
            assert row["binding_method"] == \
                "seat_reply(seat, message)"
        assert "token" in SEAT_ROWS[0]["principal_fields"]

    def test_calls_complete_and_unique(self) -> None:
        assert CALL_IDS == (
            "seat_reply", "memory_add", "memory_inspect",
            "memory_prune", "memory_export", "join_seating",
            "seat_healing", "presence",
            "room_administration", "moderation")
        assert ROUTED_CALL_IDS == (
            "seat_reply", "memory_add", "memory_inspect",
            "memory_prune", "memory_export")
        assert NOT_ROUTED_CALL_IDS == (
            "join_seating", "seat_healing", "presence",
            "room_administration", "moderation")
        for row in CALL_ROWS:
            assert call_row_problems(row) == []
        by_id = {r["id"]: r for r in CALL_ROWS}
        assert "CognitionAPI" in \
            by_id["seat_reply"]["thoughtforge_route"]
        for op in ("add", "inspect", "prune", "export"):
            route = by_id[f"memory_{op}"]["thoughtforge_route"]
            assert "MemoryAPI" in route

    def test_astrid_law_calls_all_not_routed(self) -> None:
        by_id = {r["id"]: r for r in CALL_ROWS}
        for call_id in NOT_ROUTED_CALL_IDS:
            assert by_id[call_id]["routed"] is False
            assert by_id[call_id]["surface"] == "samtal_only"
            assert by_id[call_id]["binding_method"] is None
            assert "not routed" in \
                by_id[call_id]["thoughtforge_route"]
        assert "Astrid Law" in \
            by_id["seat_healing"]["thoughtforge_route"]
        for name in ("join", "seat", "heal", "claim",
                     "presence", "moderate", "create_room"):
            assert not hasattr(SamtalBinding, name)

    def test_field_rows_cover_tf154_live(self) -> None:
        ids = [f["id"] for f in FIELD_ROWS]
        assert len(ids) == len(set(ids)) == len(FIELD_IDS)
        for field in FIELD_ROWS:
            assert field_row_problems(field) == []
        in_rows = field_rows_for("in")
        out_rows = field_rows_for("out")
        for field in TURN_IN_FIELDS:
            assert any(field["id"] in r["target"]
                       for r in in_rows)
        out_ids = {r["id"] for r in out_rows}
        for field in TURN_OUT_FIELDS:
            assert f"turn.{field['id']}" in out_ids
        assert "reply.room" in out_ids
        assert "reply.author" in out_ids

    def test_token_filed_no_home(self) -> None:
        by_id = {f["id"]: f for f in FIELD_ROWS}
        row = by_id["seat.token"]
        assert row["disposition"] == "no_home_not_passed"
        assert "never" in row["target"]

    def test_message_fields_are_samtals(self) -> None:
        assert list(MESSAGE_FIELDS) == [
            "id", "room", "author", "author_kind",
            "author_display", "body", "ts", "mtype"]

    def test_rules_are_seven_unique(self) -> None:
        assert list(RULE_IDS) == [
            "equal_seats",
            "astrid_law_untouched_cognition_only",
            "off_by_default",
            "called_never_reimplemented",
            "no_samtal_code_imported_or_copied",
            "every_field_has_a_disposition",
            "responses_detached_json_safe_versioned"]
        for rule in RULE_ROWS:
            assert rule_row_problems(rule) == []

    def test_validators_are_empty_safe(self) -> None:
        assert seat_row_problems(None)
        assert call_row_problems(None)
        assert field_row_problems(None)
        assert rule_row_problems(None)
        assert binding_response_problems(None)
        assert binding_response_problems({})
        assert seat_request_problems(None, None)
        assert seat_request_problems({}, {})
        assert seat_request_problems(
            {"name": "SeatA"}, {"body": ""})

    def test_contract_data(self) -> None:
        contract = samtal_binding_contract()
        assert contract["format"] == SAMTAL_BINDING_FORMAT
        assert contract["contract_version"] == API_VERSION
        assert contract["default_enabled"] is False
        assert contract["seat_kinds"] == list(SEAT_KINDS)
        assert json.loads(json.dumps(contract)) == contract
        data = samtal_binding_as_data()
        assert data["format"] == SAMTAL_BINDING_FORMAT
        assert data["doc"] == SAMTAL_BINDING_DOC
        assert data["measured"] == {
            "probes_passed": 7, "probes_total": 7}


# ── Disabled by default ──────────────────────────────────────────────────────


class TestDisabled:
    def test_default_is_disabled(self) -> None:
        assert SamtalBinding().enabled is False
        assert SamtalBinding(enabled=True).enabled is True

    def test_every_routed_call_refuses_as_data(self) -> None:
        binding = SamtalBinding()
        outs = [
            binding.seat_reply(_SEAT_A, _MESSAGE),
            binding.memory_call(
                "add", {"kind": "user_fact", "summary": "x"}),
            binding.memory_export(),
        ]
        for out in outs:
            assert out["ok"] is False
            assert out["status"] == "refused"
            assert out["error"] == DISABLED_ERROR
            assert out["contract_version"] == API_VERSION
            assert binding_response_problems(out) == []
            assert json.loads(json.dumps(out)) == out

    def test_disabled_seat_reply_names_the_seat(self) -> None:
        out = SamtalBinding().seat_reply(_SEAT_A, _MESSAGE)
        assert out["seat"] == _SEAT_A

    def test_disabled_builds_and_touches_nothing(self) -> None:
        built: list[str] = []

        def cog_factory():
            built.append("cognition")
            return _RecordingCognitionAPI()

        def mem_factory():
            built.append("memory")
            return MemoryAPI()

        binding = SamtalBinding(
            cognition_factory=cog_factory,
            memory_factory=mem_factory)
        binding.seat_reply(_SEAT_A, _MESSAGE)
        binding.memory_inspect()
        assert built == []
        assert binding._cognition_api is None
        assert binding._memory_api is None

    def test_disabled_refuses_garbage_without_raising(self) -> None:
        binding = SamtalBinding()
        out = binding.seat_reply(
            {"name": "SeatA"}, {"body": ""})
        assert out["ok"] is False
        assert out["error"] == DISABLED_ERROR

    def test_enable_and_disable_are_explicit_acts(self) -> None:
        binding, _fake = _chat_binding()
        binding.disable()
        assert binding.seat_reply(
            _SEAT_A, _MESSAGE)["ok"] is False
        binding.enable()
        assert binding.enabled is True
        assert binding.seat_reply(
            _SEAT_A, _MESSAGE)["ok"] is True


# ── Seat reply routing ───────────────────────────────────────────────────────


class TestSeatReply:
    def test_assembly_rule_is_the_filed_one(self) -> None:
        turn_in = assemble_turn_in(_SEAT_A, _MESSAGE, _HISTORY)
        assert turn_in["user_text"] == \
            "What time does the ferry leave?"
        assert turn_in["history"] == [
            {"role": "assistant",
             "content": "earlier words of mine",
             "name": "SeatA"},
            {"role": "user", "content": "earlier question",
             "name": "VolmarrViking"},
        ]
        solo = assemble_turn_in(_SEAT_A, _MESSAGE)
        assert solo == {
            "user_text": "What time does the ferry leave?",
            "history": []}

    def test_reply_routes_by_the_filed_rule(self) -> None:
        binding, fake = _chat_binding()
        out = binding.seat_reply(
            _SEAT_A, _MESSAGE, history=_HISTORY)
        assert fake.turns == [{
            "user_text": "What time does the ferry leave?",
            "history": [
                {"role": "assistant",
                 "content": "earlier words of mine",
                 "name": "SeatA"},
                {"role": "user",
                 "content": "earlier question",
                 "name": "VolmarrViking"},
            ]}]
        assert out["ok"] is True and out["status"] == "replied"
        assert out["call"] == "seat_reply"
        assert out["seat"] == _SEAT_A
        assert out["turn"]["text"] == \
            "stub answer to: What time does the ferry leave?"
        assert binding_response_problems(out) == []

    def test_reply_is_the_filed_message_shape(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.seat_reply(_SEAT_A, _MESSAGE)
        assert out["reply"] == {
            "room": 1, "author": "SeatA",
            "body": "stub answer to: What time does "
                    "the ferry leave?"}
        assert reply_message(
            _SEAT_A, _MESSAGE, out["turn"]) == out["reply"]

    def test_no_history_passes_no_history(self) -> None:
        binding, fake = _chat_binding()
        binding.seat_reply(_SEAT_A, _MESSAGE)
        assert fake.turns == [{
            "user_text": "What time does the ferry leave?"}]

    def test_equal_seats_route_identically(self) -> None:
        binding, fake = _chat_binding()
        out_a = binding.seat_reply(_SEAT_A, _MESSAGE)
        out_b = binding.seat_reply(_SEAT_B, _MESSAGE)
        assert set(out_a) == set(out_b)
        assert out_a["call"] == out_b["call"] == "seat_reply"
        assert out_a["status"] == out_b["status"]
        assert out_a["contract_version"] == \
            out_b["contract_version"] == API_VERSION
        assert set(out_a["turn"]) == set(out_b["turn"])
        assert set(out_a["reply"]) == set(out_b["reply"])
        assert out_a["seat"] == _SEAT_A
        assert out_b["seat"] == _SEAT_B
        assert out_a["reply"]["author"] == "SeatA"
        assert out_b["reply"]["author"] == "SeatB"
        assert fake.turns[0] == fake.turns[1]

    def test_every_kind_routes(self) -> None:
        binding, _fake = _chat_binding()
        for kind in SEAT_KINDS:
            seat = {"name": f"Seat-{kind}", "kind": kind}
            out = binding.seat_reply(seat, _MESSAGE)
            assert out["ok"] is True
            assert out["seat"] == seat

    def test_reply_response_is_detached(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.seat_reply(_SEAT_A, _MESSAGE)
        out["turn"]["text"] = "mutated"
        out["reply"]["body"] = "mutated"
        again = binding.seat_reply(_SEAT_A, _MESSAGE)
        assert again["turn"]["text"] == \
            "stub answer to: What time does the ferry leave?"
        assert again["reply"]["body"] == again["turn"]["text"]

    def test_knowledge_only_core_turn(self, tmp_path: Path) -> None:
        core = ThoughtForgeCore(
            memory_dir=tmp_path / "memory",
            db_path=tmp_path / "nonexistent.db",
            model_path=None)
        binding = SamtalBinding(
            cognition_api=CognitionAPI(core=core),
            enabled=True)
        out = binding.seat_reply(_SEAT_A, _MESSAGE)
        assert out["ok"] is True
        assert out["turn"]["mode"] == "knowledge_only"
        assert out["turn"]["trace_status"] == "included"
        assert out["reply"]["room"] == 1
        assert out["reply"]["author"] == "SeatA"
        assert out["reply"]["body"] == out["turn"]["text"]
        assert binding_response_problems(out) == []

    def test_empty_body_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="body"):
            binding.seat_reply(
                _SEAT_A, {**_MESSAGE, "body": ""})

    def test_token_in_seat_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="token"):
            binding.seat_reply(
                {**_SEAT_A, "token": "secret"}, _MESSAGE)

    def test_unfiled_fields_raise_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="surprise"):
            binding.seat_reply(
                _SEAT_A, {**_MESSAGE, "surprise": 1})
        with pytest.raises(ValueError, match="kind"):
            binding.seat_reply(
                {"name": "SeatA", "kind": "royal"}, _MESSAGE)

    def test_history_without_author_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="author"):
            binding.seat_reply(
                _SEAT_A, _MESSAGE,
                history=[{"body": "orphan words"}])

    def test_response_problems_catches_tampering(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.seat_reply(_SEAT_A, _MESSAGE)
        assert binding_response_problems(out) == []
        wrong = dict(out, contract_version="9.9.9")
        assert any("contract_version" in p
                   for p in binding_response_problems(wrong))
        lying = dict(out, error="boom")
        assert any("error" in p
                   for p in binding_response_problems(lying))
        royal = json.loads(json.dumps(out))
        royal["seat"] = {"name": "SeatA", "kind": "royal"}
        assert any("kind" in p
                   for p in binding_response_problems(royal))


# ── Memory routing ───────────────────────────────────────────────────────────


class TestMemory:
    def test_add_lands_stamped_through_tf155(
            self, tmp_path: Path) -> None:
        binding, store = _memory_binding(tmp_path)
        out = binding.memory_add(
            "user_fact", summary="ferry at 07:30",
            record_id="fct_tf157")
        assert out["ok"] is True and out["call"] == "memory_add"
        assert out["memory"]["status"] == "added"
        assert out["memory"]["record"]["source"] == \
            "manual_entry"
        assert len(store.load_profile_records()) == 1
        assert binding_response_problems(out) == []

    def test_inspect_found_and_miss(self, tmp_path: Path) -> None:
        binding, _store = _memory_binding(tmp_path)
        binding.memory_add("user_fact",
                           summary="ferry at 07:30",
                           record_id="fct_tf157")
        found = binding.memory_inspect(record_id="fct_tf157")
        assert found["memory"]["status"] == "found"
        miss = binding.memory_inspect(record_id="nope")
        assert miss["ok"] is True
        assert miss["memory"]["status"] == "not_found"

    def test_tf155_refusal_surfaces_as_data(
            self, tmp_path: Path) -> None:
        binding, store = _memory_binding(tmp_path)
        out = binding.memory_add("user_fact", summary="")
        assert out["ok"] is False and out["status"] == "refused"
        assert out["memory"]["ok"] is False
        assert out["error"] == out["memory"]["refusal"]
        assert out["error"]
        assert store.load_profile_records() == []
        assert binding_response_problems(out) == []

    def test_export_returns_verified_bundle(
            self, tmp_path: Path) -> None:
        binding, _store = _memory_binding(tmp_path)
        binding.memory_add("user_fact",
                           summary="ferry at 07:30",
                           record_id="fct_tf157")
        out = binding.memory_export(
            destination=str(tmp_path / "bundle.json"))
        assert out["ok"] is True
        assert out["memory"]["verified"] is True
        assert out["memory"]["bundle"]["sections"][
            "profile_records"][0]["source"] == "manual_entry"

    def test_prune_routes_to_tf155(self, tmp_path: Path) -> None:
        binding, _store = _memory_binding(tmp_path)
        binding.memory_add("user_fact",
                           summary="ferry at 07:30",
                           record_id="fct_tf157")
        out = binding.memory_prune(mode="routine")
        assert out["ok"] is True
        assert out["memory"]["status"] == "pruned"
        assert out["memory"]["after"]["user_fact"] == 1

    def test_unknown_op_raises_named(self, tmp_path: Path) -> None:
        binding, _store = _memory_binding(tmp_path)
        with pytest.raises(ValueError, match="explode"):
            binding.memory_call("explode")

    def test_memory_call_dict_form(self, tmp_path: Path) -> None:
        binding, _store = _memory_binding(tmp_path)
        out = binding.memory_call(
            "add", {"kind": "user_fact",
                    "summary": "ferry at 07:30",
                    "record_id": "fct_dict"})
        assert out["ok"] is True
        assert out["memory"]["record_id"] == "fct_dict"
        assert json.loads(json.dumps(out)) == out


# ── Injection and laziness ───────────────────────────────────────────────────


class TestInjection:
    def test_construction_builds_nothing(self) -> None:
        binding = SamtalBinding()
        assert binding._cognition_api is None
        assert binding._memory_api is None
        assert SamtalBinding(
            turn_callable=lambda text: {})._cognition_api is None

    def test_factories_are_lazy(self) -> None:
        built: list[str] = []
        fake = _RecordingCognitionAPI()

        def factory():
            built.append("cognition")
            return fake

        binding = SamtalBinding(
            cognition_factory=factory, enabled=True)
        assert built == []
        binding.seat_reply(_SEAT_A, _MESSAGE)
        assert built == ["cognition"]

    def test_turn_callable_passthrough(self) -> None:
        def stub(user_text: str):
            return {
                "text": f"stub: {user_text}",
                "response_id": "r", "turn_id": "t",
                "mode": "stub_mode",
                "enforcement_passed": True, "citations": [],
                "token_count": 1,
            }

        binding = SamtalBinding(
            turn_callable=stub, enabled=True)
        out = binding.seat_reply(_SEAT_A, _MESSAGE)
        assert out["turn"]["text"] == \
            "stub: What time does the ferry leave?"
        assert out["turn"]["trace_status"].startswith(
            "unavailable: ")
        assert out["reply"]["body"] == out["turn"]["text"]

    def test_memory_store_passthrough(
            self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "store")
        binding = SamtalBinding(
            memory_store=store, enabled=True)
        assert binding._memory_api is None
        out = binding.memory_add(
            "user_fact", summary="ferry at 07:30",
            record_id="fct_store")
        assert out["ok"] is True
        assert len(store.load_profile_records()) == 1


# ── State, probes, doc, docket ───────────────────────────────────────────────


class TestStateProbesDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_samtal_binding_probes()
        assert len(probes) == len(SAMTAL_BINDING_PROBES) == 7
        assert [p["id"] for p in probes] == [
            "disabled_refusal_touches_nothing",
            "enabled_cognition_round_trip_through_binding",
            "enabled_memory_round_trip_temp_store",
            "equal_seats_routed_identically",
            "astrid_law_no_seating_calls_routed",
            "fake_consumer_consistency",
            "doc_matches_render_and_version_is_tf034"]
        assert all(p["passed"] for p in probes)

    def test_doc_matches_render(self) -> None:
        assert _read(SAMTAL_BINDING_DOC) == \
            render_samtal_binding_markdown()
        doc = _read(SAMTAL_BINDING_DOC)
        assert "A Seat's Thinking Through ThoughtForge" in doc
        assert "NOT PROVEN" in doc
        assert "TF-170" in doc and "TF-158" in doc
        assert "Astrid Law" in doc
        assert API_VERSION in doc

    def test_docket_untouched_totals_unchanged(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 32 OPEN, 1 OPEN-ENVIRONMENT, " \
            "100 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT" in normalized
