"""
TF-156 — The Rúnmál binding: Rúnmál's
cognition/memory calls can route through
ThoughtForge (behind its backend table; off
by default; the parity campaign's P-slices
consume the doc).

The proof lives as verifiable data in
``src/thoughtforge/runmal_binding.py``
(``BACKEND_TABLE_ROWS``, ``CALL_ROWS``,
``FIELD_ROWS``, ``RUNMAL_BINDING_RULES``,
``RUNMAL_BINDING_PROBES``, checked by
``runmal_binding_problems()``); the doc
``docs/architecture/RUNMAL_BINDING.md`` is
filed verbatim as the module's render.

Pinned here:

  * the backend-table row files a
    ThoughtForge entry in Rúnmál's
    BackendRegistry under backend_key
    "thoughtforge", answering the ChatBackend
    interface (chat_stream / list_models /
    health_check abstract; count_tokens /
    filter_params concrete), default_model
    None, enabled by default False,
  * every Rúnmál-side call has a CALL_ROWS
    row: chat_stream / list_models /
    health_check route to TF-154's
    CognitionAPI (the filed honest answers
    for the latter two), the four memory ops
    route to TF-155's MemoryAPI, and
    count_tokens / filter_params are filed
    NOT routed — Rúnmál's own concrete
    defaults, nothing invented,
  * the field mapping files the assembly
    rule (the last user message is user_text;
    every other message is history, in
    order, names riding) and gives EVERY
    field a disposition — TF-154's turn-in
    and turn-out fields checked live against
    TF-154's own filings, the SamplingParams
    fields and cancel filed no-home, never
    passed, never claimed honored,
  * the binding is DEFAULT DISABLED
    (``DEFAULT_ENABLED`` False, filed as
    data): while disabled every routed call
    returns the refusal as data naming the
    binding disabled and touches neither API
    (counting factories prove nothing is
    built); enabling is the explicit
    constructor flag / ``enable()`` act,
  * when enabled, cognition routes through
    the injected CognitionAPI and memory
    through the injected MemoryAPI (called,
    never reimplemented — a memory add lands
    stamped manual_entry through TF-057's
    gate, and TF-155 refusals surface
    through the binding envelope as data),
  * responses are detached JSON-safe data,
    contract_version is TF-034's
    ``API_VERSION``, the doc matches its
    render verbatim, the checker is clean,
    the probes pass 7/7, and TF-016's docket
    totals are unchanged (TF-156 has no
    docket share).

Proof grade: FIXTURE — a fake Rúnmál
consumer shaped on the filed ChatBackend /
registry data; NO live Rúnmál process was
routed and NO Rúnmál code was imported or
copied; live end-to-end routing is NOT
PROVEN (TF-170's gate owns it), never
claimed. All fixtures synthetic (TF-017);
no canon.
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
from thoughtforge.runmal_binding import (
    BACKEND_KEY,
    BACKEND_TABLE_ROWS,
    CALL_IDS,
    CALL_ROWS,
    DEFAULT_ENABLED,
    DISABLED_ERROR,
    DONE_REASON,
    FIELD_IDS,
    FIELD_ROWS,
    FILED_HEALTH,
    FILED_MODEL,
    ROUTED_CALL_IDS,
    RUNMAL_BINDING_DOC,
    RUNMAL_BINDING_FORMAT,
    RUNMAL_BINDING_PROBES,
    RUNMAL_BINDING_RULES,
    SAMPLING_PARAM_FIELDS,
    RunmalBinding,
    assemble_turn_in,
    backend_row_problems,
    binding_response_problems,
    binding_rule_problems,
    call_row_problems,
    chat_request_problems,
    field_row_problems,
    field_rows_for,
    render_runmal_binding_markdown,
    reply_events,
    run_runmal_binding_probes,
    runmal_binding_as_data,
    runmal_binding_contract,
    runmal_binding_problems,
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


def _chat_binding() -> tuple[RunmalBinding, _RecordingCognitionAPI]:
    fake = _RecordingCognitionAPI()
    return RunmalBinding(cognition_api=fake, enabled=True), fake


def _memory_binding(
    tmp_path: Path,
) -> tuple[RunmalBinding, MemoryStore]:
    store = MemoryStore(memory_dir=tmp_path / "store")
    binding = RunmalBinding(
        memory_api=MemoryAPI(store=store), enabled=True)
    return binding, store


_MESSAGES = [
    {"role": "system", "content": "be terse", "name": "ops"},
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "reply"},
    {"role": "user", "content": "second"},
]


# ── The filed estate ─────────────────────────────────────────────────────────


class TestFiling:
    def test_problems_empty(self) -> None:
        assert runmal_binding_problems() == []

    def test_format_doc_and_defaults(self) -> None:
        assert RUNMAL_BINDING_FORMAT == \
            "thoughtforge-runmal-binding/1"
        assert RUNMAL_BINDING_DOC == \
            "docs/architecture/RUNMAL_BINDING.md"
        assert BACKEND_KEY == "thoughtforge"
        assert DEFAULT_ENABLED is False
        assert "disabled" in DISABLED_ERROR

    def test_backend_table_row(self) -> None:
        assert len(BACKEND_TABLE_ROWS) == 1
        row = BACKEND_TABLE_ROWS[0]
        assert backend_row_problems(row) == []
        assert row["backend_key"] == "thoughtforge"
        assert row["abstract_methods"] == [
            "chat_stream", "list_models", "health_check"]
        assert row["concrete_methods"] == [
            "count_tokens", "filter_params"]
        assert row["class_attributes"] == [
            "backend_key", "default_model"]
        assert row["default_model"] is None
        assert row["enabled_by_default"] is False
        assert "register" in row["registry_methods"]
        assert "health_all" in row["registry_methods"]

    def test_calls_complete_and_unique(self) -> None:
        assert CALL_IDS == (
            "chat_stream", "list_models", "health_check",
            "count_tokens", "filter_params", "memory_add",
            "memory_inspect", "memory_prune",
            "memory_export")
        assert ROUTED_CALL_IDS == (
            "chat_stream", "list_models", "health_check",
            "memory_add", "memory_inspect", "memory_prune",
            "memory_export")
        for row in CALL_ROWS:
            assert call_row_problems(row) == []
        by_id = {r["id"]: r for r in CALL_ROWS}
        assert by_id["count_tokens"]["routed"] is False
        assert by_id["filter_params"]["routed"] is False
        assert "CognitionAPI" in \
            by_id["chat_stream"]["thoughtforge_route"]
        for op in ("add", "inspect", "prune", "export"):
            route = by_id[f"memory_{op}"]["thoughtforge_route"]
            assert "MemoryAPI" in route
        assert "PARTIAL" in \
            by_id["memory_prune"]["thoughtforge_route"]

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
        assert "events.done" in out_ids

    def test_params_and_cancel_filed_no_home(self) -> None:
        by_id = {f["id"]: f for f in FIELD_ROWS}
        assert list(SAMPLING_PARAM_FIELDS) == [
            "temperature", "top_p", "top_k", "min_p",
            "repetition_penalty", "presence_penalty",
            "frequency_penalty", "max_output_tokens",
            "stop", "seed", "context_length"]
        for param in SAMPLING_PARAM_FIELDS:
            row = by_id[f"params.{param}"]
            assert row["disposition"] == "no_home_not_passed"
        assert by_id["cancel"]["disposition"] == \
            "no_home_not_wired"
        assert by_id["message.name"]["disposition"] == "partial"

    def test_rules_are_seven_unique(self) -> None:
        ids = [r["id"] for r in RUNMAL_BINDING_RULES]
        assert len(ids) == len(set(ids)) == 7
        assert set(ids) == {
            "off_by_default",
            "disabled_refusal_is_data_touches_nothing",
            "called_never_reimplemented",
            "no_runmal_code_imported_or_copied",
            "every_field_has_a_disposition",
            "params_never_claimed_honored",
            "responses_detached_json_safe_versioned"}
        for rule in RUNMAL_BINDING_RULES:
            assert binding_rule_problems(rule) == []

    def test_validators_are_empty_safe(self) -> None:
        assert backend_row_problems(None)
        assert call_row_problems(None)
        assert field_row_problems(None)
        assert binding_rule_problems(None)
        assert binding_response_problems(None)
        assert binding_response_problems({})
        assert chat_request_problems(None)
        assert chat_request_problems([])
        assert chat_request_problems("not a list")

    def test_contract_data(self) -> None:
        contract = runmal_binding_contract()
        assert contract["format"] == RUNMAL_BINDING_FORMAT
        assert contract["contract_version"] == API_VERSION
        assert contract["default_enabled"] is False
        assert contract["backend_key"] == "thoughtforge"
        assert contract["model"] == FILED_MODEL
        assert json.loads(json.dumps(contract)) == contract
        data = runmal_binding_as_data()
        assert data["format"] == RUNMAL_BINDING_FORMAT
        assert data["doc"] == RUNMAL_BINDING_DOC
        assert data["measured"] == {
            "probes_passed": 7, "probes_total": 7}


# ── Disabled by default ──────────────────────────────────────────────────────


class TestDisabled:
    def test_default_is_disabled(self) -> None:
        assert RunmalBinding().enabled is False
        assert RunmalBinding(enabled=True).enabled is True

    def test_every_routed_call_refuses_as_data(self) -> None:
        binding = RunmalBinding()
        outs = [
            binding.chat(_MESSAGES),
            binding.list_models(),
            binding.health_check(),
            binding.memory_call(
                "add", {"kind": "user_fact", "summary": "x"}),
            binding.memory_export(),
        ]
        for out in outs:
            assert out["ok"] is False
            assert out["status"] == "refused"
            assert out["error"] == DISABLED_ERROR
            assert out["backend_key"] == "thoughtforge"
            assert out["contract_version"] == API_VERSION
            assert binding_response_problems(out) == []
            assert json.loads(json.dumps(out)) == out

    def test_disabled_builds_and_touches_nothing(self) -> None:
        built: list[str] = []

        def cog_factory():
            built.append("cognition")
            return _RecordingCognitionAPI()

        def mem_factory():
            built.append("memory")
            return MemoryAPI()

        binding = RunmalBinding(
            cognition_factory=cog_factory,
            memory_factory=mem_factory)
        binding.chat(_MESSAGES)
        binding.memory_inspect()
        assert built == []
        assert binding._cognition_api is None
        assert binding._memory_api is None

    def test_disabled_refuses_garbage_without_raising(self) -> None:
        binding = RunmalBinding()
        out = binding.chat([{"role": "assistant"}])
        assert out["ok"] is False
        assert out["error"] == DISABLED_ERROR

    def test_enable_and_disable_are_explicit_acts(
            self, tmp_path: Path) -> None:
        binding, _fake = _chat_binding()
        binding.disable()
        assert binding.chat(_MESSAGES)["ok"] is False
        binding.enable()
        assert binding.enabled is True
        assert binding.chat(_MESSAGES)["ok"] is True


# ── Chat routing ─────────────────────────────────────────────────────────────


class TestChat:
    def test_assembly_rule_is_the_filed_one(self) -> None:
        turn_in = assemble_turn_in(_MESSAGES)
        assert turn_in["user_text"] == "second"
        assert turn_in["history"] == [
            {"role": "system", "content": "be terse",
             "name": "ops"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        solo = assemble_turn_in(
            [{"role": "user", "content": "only"}])
        assert solo == {"user_text": "only", "history": []}

    def test_chat_routes_by_the_filed_rule(self) -> None:
        binding, fake = _chat_binding()
        out = binding.chat(_MESSAGES)
        assert fake.turns == [{
            "user_text": "second",
            "history": [
                {"role": "system", "content": "be terse",
                 "name": "ops"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
            ]}]
        assert out["ok"] is True and out["status"] == "replied"
        assert out["call"] == "chat_stream"
        assert out["turn"]["text"] == "stub answer to: second"
        assert binding_response_problems(out) == []

    def test_chat_events_are_the_filed_shapes(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.chat(_MESSAGES)
        assert out["events"] == [
            {"type": "text_delta",
             "text": "stub answer to: second"},
            {"type": "usage", "prompt_tokens": None,
             "completion_tokens": None, "total_tokens": 3},
            {"type": "done", "reason": DONE_REASON,
             "usage": {"type": "usage", "prompt_tokens": None,
                       "completion_tokens": None,
                       "total_tokens": 3}},
        ]
        assert reply_events(out["turn"]) == out["events"]

    def test_single_message_passes_no_history(self) -> None:
        binding, fake = _chat_binding()
        binding.chat([{"role": "user", "content": "only"}])
        assert fake.turns == [{"user_text": "only"}]

    def test_params_are_never_passed(self) -> None:
        binding, fake = _chat_binding()
        out = binding.chat(
            [{"role": "user", "content": "hi"}],
            params={"temperature": 0.7, "seed": 3,
                    "max_output_tokens": 64})
        assert out["ok"] is True
        assert fake.turns == [{"user_text": "hi"}]

    def test_chat_response_is_detached(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.chat(_MESSAGES)
        out["turn"]["text"] = "mutated"
        again = binding.chat(_MESSAGES)
        assert again["turn"]["text"] == "stub answer to: second"

    def test_knowledge_only_core_turn(self, tmp_path: Path) -> None:
        core = ThoughtForgeCore(
            memory_dir=tmp_path / "memory",
            db_path=tmp_path / "nonexistent.db",
            model_path=None)
        binding = RunmalBinding(
            cognition_api=CognitionAPI(core=core),
            enabled=True)
        out = binding.chat(
            [{"role": "user",
              "content": "What time does the ferry leave?"}])
        assert out["ok"] is True
        assert out["turn"]["mode"] == "knowledge_only"
        assert out["turn"]["trace_status"] == "included"
        assert [e["type"] for e in out["events"]] == [
            "text_delta", "usage", "done"]
        assert binding_response_problems(out) == []

    def test_no_user_message_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="user"):
            binding.chat([{"role": "assistant",
                           "content": "hi"}])

    def test_empty_last_user_content_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="empty"):
            binding.chat([{"role": "user", "content": ""}])

    def test_unfiled_fields_raise_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="surprise"):
            binding.chat([{"role": "user", "content": "hi",
                           "surprise": 1}])
        with pytest.raises(ValueError, match="surprise"):
            binding.chat([{"role": "user", "content": "hi"}],
                         params={"surprise": 1})

    def test_response_problems_catches_tampering(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.chat(_MESSAGES)
        assert binding_response_problems(out) == []
        wrong = dict(out, contract_version="9.9.9")
        assert any("contract_version" in p
                   for p in binding_response_problems(wrong))
        lying = dict(out, error="boom")
        assert any("error" in p
                   for p in binding_response_problems(lying))
        silent = {"call": "chat_stream", "ok": False,
                  "status": "refused",
                  "backend_key": "thoughtforge",
                  "contract_version": API_VERSION, "error": ""}
        assert any("error" in p
                   for p in binding_response_problems(silent))


# ── list_models / health_check ───────────────────────────────────────────────


class TestModelsAndHealth:
    def test_list_models_returns_the_filed_model(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.list_models()
        assert out["ok"] is True and out["status"] == "listed"
        assert out["models"] == [FILED_MODEL]
        caps = out["models"][0]["capabilities"]
        assert caps["chat"] is True
        assert caps["streaming"] is False
        assert caps["supported_samplers"] == []
        assert caps["token_count_quality"] == "estimated"
        assert binding_response_problems(out) == []

    def test_list_models_response_is_detached(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.list_models()
        out["models"][0]["capabilities"]["chat"] = False
        assert binding.list_models()["models"] == [FILED_MODEL]
        assert FILED_MODEL["capabilities"]["chat"] is True

    def test_health_reports_the_binding_not_a_model(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.health_check()
        assert out["ok"] is True and out["status"] == "reported"
        assert out["health"] == FILED_HEALTH
        assert out["health"]["latency_ms"] is None
        assert "no live model health is claimed" in \
            out["health"]["detail"]
        assert binding_response_problems(out) == []


# ── Memory routing ───────────────────────────────────────────────────────────


class TestMemory:
    def test_add_lands_stamped_through_tf155(
            self, tmp_path: Path) -> None:
        binding, store = _memory_binding(tmp_path)
        out = binding.memory_add(
            "user_fact", summary="ferry at 07:30",
            record_id="fct_tf156")
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
                           record_id="fct_tf156")
        found = binding.memory_inspect(record_id="fct_tf156")
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
                           record_id="fct_tf156")
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
                           record_id="fct_tf156")
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
        binding = RunmalBinding()
        assert binding._cognition_api is None
        assert binding._memory_api is None
        assert RunmalBinding(
            turn_callable=lambda text: {})._cognition_api is None

    def test_factories_are_lazy(self) -> None:
        built: list[str] = []
        fake = _RecordingCognitionAPI()

        def factory():
            built.append("cognition")
            return fake

        binding = RunmalBinding(
            cognition_factory=factory, enabled=True)
        assert built == []
        binding.chat(_MESSAGES)
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

        binding = RunmalBinding(
            turn_callable=stub, enabled=True)
        out = binding.chat(
            [{"role": "user", "content": "hello"}])
        assert out["turn"]["text"] == "stub: hello"
        assert out["turn"]["trace_status"].startswith(
            "unavailable: ")

    def test_memory_store_passthrough(
            self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "store")
        binding = RunmalBinding(
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
        probes = run_runmal_binding_probes()
        assert len(probes) == len(RUNMAL_BINDING_PROBES) == 7
        assert [p["id"] for p in probes] == [
            "disabled_refusal_touches_nothing",
            "enabled_cognition_routes_by_the_filed_rule",
            "enabled_memory_round_trip_temp_store",
            "backend_table_row_consistency",
            "models_and_health_are_honest",
            "params_not_passed_and_invalid_raises_named",
            "doc_matches_render_and_version_is_tf034"]
        assert all(p["passed"] for p in probes)

    def test_doc_matches_render(self) -> None:
        assert _read(RUNMAL_BINDING_DOC) == \
            render_runmal_binding_markdown()
        doc = _read(RUNMAL_BINDING_DOC)
        assert "Rúnmál's Calls Through ThoughtForge" in doc
        assert "NOT PROVEN" in doc
        assert "TF-170" in doc and "TF-157" in doc
        assert API_VERSION in doc
        assert "thoughtforge" in doc

    def test_docket_untouched_totals_unchanged(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 32 OPEN, 1 OPEN-ENVIRONMENT, " \
            "100 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT" in normalized
