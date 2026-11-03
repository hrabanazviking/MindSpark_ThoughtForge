"""
TF-158 — The Hermes binding: ThoughtForge as
a cognition provider for the Hermes hack,
shaped on the volmarr-core shaped contract
(consumes the 2026-10-30 refresh's plugin
surfaces).

The proof lives as verifiable data in
``src/thoughtforge/hermes_binding.py``
(``PROVIDER_ROWS``, ``RULE_ROWS``,
``CALL_ROWS``, ``FIELD_ROWS``,
``HERMES_BINDING_PROBES``, checked by
``hermes_binding_problems()``); the doc
``docs/architecture/HERMES_BINDING.md`` is
filed verbatim as the module's render.

Pinned here:

  * the provider rows file the plugin
    surfaces Hermes actually offers, read
    READ-ONLY from the Hermes hack tree —
    volmarr-core (a literal name there:
    plugins/volmarr-core, plugin.yaml
    manifest_version 2, version 0.26.0), the
    model-provider ProviderProfile surface,
    the runeforge.cognition.route routing
    contract, the MemoryProvider surface —
    and the context-engine surface filed as
    one the binding does NOT answer,
  * a Hermes-routed local turn routes to
    TF-154's CognitionAPI by the filed rule
    (the last user message is user_text;
    every other message rides as history)
    and returns as the filed completion
    shape; the catalog and health calls
    answer from the filed rows; memory calls
    route to TF-155's MemoryAPI's four ops,
    called, never reimplemented,
  * tool execution / gateway messaging /
    session & state custody / credential
    custody / provider resolution / the
    memory-provider lifecycle are all filed
    NOT routed, and the binding class files
    no such method,
  * every field has a disposition — TF-154's
    turn-in and turn-out fields checked live
    against TF-154's own filings, max_tokens
    and any credential filed no-home,
  * the binding is DEFAULT DISABLED
    (``DEFAULT_ENABLED`` False, filed as
    data): while disabled every routed call
    returns the refusal as data naming the
    binding disabled and touches neither API
    (counting factories prove nothing is
    built); enabling is the explicit
    constructor flag / ``enable()`` act,
  * responses are detached JSON-safe data,
    contract_version is TF-034's
    ``API_VERSION``, the doc matches its
    render verbatim, the checker is clean,
    the probes pass 7/7, and TF-016's docket
    totals are unchanged (TF-158 has no
    docket share).

Proof grade: FIXTURE — a fake Hermes
consumer shaped on the filed plugin-surface
data; NO live Hermes process was routed and
NO Hermes code was imported or copied; live
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
from thoughtforge.hermes_binding import (
    CALL_IDS,
    CALL_ROWS,
    DEFAULT_ENABLED,
    DISABLED_ERROR,
    FIELD_IDS,
    FIELD_ROWS,
    FILED_CATALOG,
    FILED_HEALTH,
    FILED_MODEL,
    FINISH_REASON,
    HERMES_BINDING_DOC,
    HERMES_BINDING_FORMAT,
    HERMES_BINDING_PROBES,
    MESSAGE_FIELDS,
    MESSAGE_ROLES,
    NOT_ROUTED_CALL_IDS,
    PLUGIN_NAME,
    PROVIDER_KEY,
    PROVIDER_ROWS,
    REQUEST_FIELDS,
    ROUTED_CALL_IDS,
    RULE_IDS,
    RULE_ROWS,
    HermesBinding,
    assemble_turn_in,
    binding_response_problems,
    call_row_problems,
    completion_for,
    field_row_problems,
    field_rows_for,
    hermes_binding_as_data,
    hermes_binding_contract,
    hermes_binding_problems,
    provider_request_problems,
    provider_row_problems,
    render_hermes_binding_markdown,
    rule_row_problems,
    run_hermes_binding_probes,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.memory_api import MemoryAPI

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


def _chat_binding() -> tuple[HermesBinding, _RecordingCognitionAPI]:
    fake = _RecordingCognitionAPI()
    return HermesBinding(cognition_api=fake, enabled=True), fake


def _memory_binding(
    tmp_path: Path,
) -> tuple[HermesBinding, MemoryStore]:
    store = MemoryStore(memory_dir=tmp_path / "store")
    binding = HermesBinding(
        memory_api=MemoryAPI(store=store), enabled=True)
    return binding, store


_REQUEST = {"model": "thoughtforge",
            "messages": [
                {"role": "user",
                 "content": "What time does the ferry leave?"}],
            "max_tokens": 256}
_MESSAGES = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "reply"},
    {"role": "user",
     "content": "What time does the ferry leave?"},
]


# ── The filed estate ─────────────────────────────────────────────────────────


class TestFiling:
    def test_problems_empty(self) -> None:
        assert hermes_binding_problems() == []

    def test_format_doc_and_defaults(self) -> None:
        assert HERMES_BINDING_FORMAT == \
            "thoughtforge-hermes-binding/1"
        assert HERMES_BINDING_DOC == \
            "docs/architecture/HERMES_BINDING.md"
        assert DEFAULT_ENABLED is False
        assert "disabled" in DISABLED_ERROR
        assert PROVIDER_KEY == "thoughtforge"
        assert PLUGIN_NAME == "volmarr-core"
        assert FINISH_REASON == "complete"

    def test_provider_rows_file_the_real_surfaces(self) -> None:
        assert [r["id"] for r in PROVIDER_ROWS] == [
            "provider_volmarr_core",
            "provider_model_provider",
            "provider_cognition_routing",
            "provider_memory_provider",
            "provider_context_engine"]
        for row in PROVIDER_ROWS:
            assert provider_row_problems(row) == []
            assert row["enabled_by_default"] is False
        by_id = {r["id"]: r for r in PROVIDER_ROWS}
        core = by_id["provider_volmarr_core"]
        assert "volmarr-core" in core["hermes_side"]
        assert "register_cli_command" in core["registration"]
        routing = by_id["provider_cognition_routing"]
        assert "runeforge.cognition.route" in \
            routing["hermes_side"]
        engine = by_id["provider_context_engine"]
        assert engine["answers_binding"] is False
        assert engine["binding_method"] is None

    def test_calls_complete_and_unique(self) -> None:
        assert CALL_IDS == (
            "provider_turn", "provider_catalog",
            "provider_health", "memory_add",
            "memory_inspect", "memory_prune",
            "memory_export", "tool_execution",
            "gateway_messaging", "session_state_custody",
            "credential_custody", "provider_resolution",
            "memory_lifecycle")
        assert ROUTED_CALL_IDS == (
            "provider_turn", "provider_catalog",
            "provider_health", "memory_add",
            "memory_inspect", "memory_prune",
            "memory_export")
        assert NOT_ROUTED_CALL_IDS == (
            "tool_execution", "gateway_messaging",
            "session_state_custody", "credential_custody",
            "provider_resolution", "memory_lifecycle")
        for row in CALL_ROWS:
            assert call_row_problems(row) == []
        by_id = {r["id"]: r for r in CALL_ROWS}
        assert "CognitionAPI" in \
            by_id["provider_turn"]["thoughtforge_route"]
        for op in ("add", "inspect", "prune", "export"):
            route = by_id[f"memory_{op}"]["thoughtforge_route"]
            assert "MemoryAPI" in route

    def test_hermes_only_calls_all_not_routed(self) -> None:
        by_id = {r["id"]: r for r in CALL_ROWS}
        for call_id in NOT_ROUTED_CALL_IDS:
            assert by_id[call_id]["routed"] is False
            assert by_id[call_id]["surface"] == "hermes_only"
            assert by_id[call_id]["binding_method"] is None
            assert "not routed" in \
                by_id[call_id]["thoughtforge_route"]
        assert "TF-160" in \
            by_id["credential_custody"]["thoughtforge_route"]
        for name in ("execute_tool", "send", "send_message",
                     "open_session", "get_credential",
                     "decide_route", "resolve_provider",
                     "prefetch", "sync_turn"):
            assert not hasattr(HermesBinding, name)

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
        assert "completion.finish_reason" in out_ids
        assert "completion.model" in out_ids

    def test_credential_and_max_tokens_filed_no_home(self) -> None:
        by_id = {f["id"]: f for f in FIELD_ROWS}
        key = by_id["request.api_key"]
        assert key["disposition"] == "no_home_not_passed"
        assert "never" in key["target"]
        tokens = by_id["request.max_tokens"]
        assert tokens["disposition"] == "no_home_not_passed"

    def test_request_and_message_fields_are_hermes(self) -> None:
        assert list(REQUEST_FIELDS) == [
            "model", "messages", "max_tokens"]
        assert list(MESSAGE_FIELDS) == ["role", "content"]
        assert list(MESSAGE_ROLES) == [
            "system", "user", "assistant"]

    def test_catalog_and_health_filings_are_honest(self) -> None:
        assert FILED_CATALOG == {
            "object": "list", "data": [FILED_MODEL]}
        assert FILED_MODEL["id"] == PROVIDER_KEY
        assert FILED_HEALTH["healthy"] is True
        assert FILED_HEALTH["models"] == [PROVIDER_KEY]
        assert FILED_HEALTH["latency_ms"] is None
        assert "no live model health is claimed" in \
            FILED_HEALTH["detail"]

    def test_rules_are_seven_unique(self) -> None:
        assert list(RULE_IDS) == [
            "volmarr_core_real_and_named",
            "composition_boundary_holds",
            "off_by_default",
            "called_never_reimplemented",
            "no_hermes_code_imported_or_copied",
            "every_field_has_a_disposition",
            "responses_detached_json_safe_versioned"]
        for rule in RULE_ROWS:
            assert rule_row_problems(rule) == []

    def test_validators_are_empty_safe(self) -> None:
        assert provider_row_problems(None)
        assert call_row_problems(None)
        assert field_row_problems(None)
        assert rule_row_problems(None)
        assert binding_response_problems(None)
        assert binding_response_problems({})
        assert provider_request_problems(None)
        assert provider_request_problems({})
        assert provider_request_problems(
            {"messages": []})
        assert provider_request_problems(
            {"messages": [{"role": "user", "content": ""}]})

    def test_contract_data(self) -> None:
        contract = hermes_binding_contract()
        assert contract["format"] == HERMES_BINDING_FORMAT
        assert contract["contract_version"] == API_VERSION
        assert contract["default_enabled"] is False
        assert contract["provider_key"] == PROVIDER_KEY
        assert contract["plugin_name"] == PLUGIN_NAME
        assert json.loads(json.dumps(contract)) == contract
        data = hermes_binding_as_data()
        assert data["format"] == HERMES_BINDING_FORMAT
        assert data["doc"] == HERMES_BINDING_DOC
        assert data["measured"] == {
            "probes_passed": 7, "probes_total": 7}


# ── Disabled by default ──────────────────────────────────────────────────────


class TestDisabled:
    def test_default_is_disabled(self) -> None:
        assert HermesBinding().enabled is False
        assert HermesBinding(enabled=True).enabled is True

    def test_every_routed_call_refuses_as_data(self) -> None:
        binding = HermesBinding()
        outs = [
            binding.provider_turn(_REQUEST),
            binding.catalog(),
            binding.health_check(),
            binding.memory_call(
                "add", {"kind": "user_fact", "summary": "x"}),
            binding.memory_export(),
        ]
        for out in outs:
            assert out["ok"] is False
            assert out["status"] == "refused"
            assert out["error"] == DISABLED_ERROR
            assert out["provider_key"] == PROVIDER_KEY
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

        binding = HermesBinding(
            cognition_factory=cog_factory,
            memory_factory=mem_factory)
        binding.provider_turn(_REQUEST)
        binding.memory_inspect()
        assert built == []
        assert binding._cognition_api is None
        assert binding._memory_api is None

    def test_disabled_refuses_garbage_without_raising(self) -> None:
        binding = HermesBinding()
        out = binding.provider_turn(
            {"messages": [], "api_key": "secret"})
        assert out["ok"] is False
        assert out["error"] == DISABLED_ERROR

    def test_enable_and_disable_are_explicit_acts(self) -> None:
        binding, _fake = _chat_binding()
        binding.disable()
        assert binding.provider_turn(
            _REQUEST)["ok"] is False
        binding.enable()
        assert binding.enabled is True
        assert binding.provider_turn(
            _REQUEST)["ok"] is True


# ── Provider turn routing ────────────────────────────────────────────────────


class TestProviderTurn:
    def test_assembly_rule_is_the_filed_one(self) -> None:
        turn_in = assemble_turn_in(_MESSAGES)
        assert turn_in["user_text"] == \
            "What time does the ferry leave?"
        assert turn_in["history"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        solo = assemble_turn_in(_REQUEST["messages"])
        assert solo == {
            "user_text": "What time does the ferry leave?",
            "history": []}

    def test_turn_routes_by_the_filed_rule(self) -> None:
        binding, fake = _chat_binding()
        out = binding.provider_turn(
            {"model": "thoughtforge", "messages": _MESSAGES,
             "max_tokens": 256})
        assert fake.turns == [{
            "user_text": "What time does the ferry leave?",
            "history": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
            ]}]
        assert out["ok"] is True and out["status"] == "replied"
        assert out["call"] == "provider_turn"
        assert out["model"] == "thoughtforge"
        assert out["turn"]["text"] == \
            "stub answer to: What time does the ferry leave?"
        assert binding_response_problems(out) == []

    def test_completion_is_the_filed_shape(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.provider_turn(_REQUEST)
        completion = out["completion"]
        assert completion["model"] == "thoughtforge"
        assert completion["choices"] == [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "stub answer to: What time "
                           "does the ferry leave?"},
            "finish_reason": FINISH_REASON}]
        assert completion["usage"] == {
            "prompt_tokens": None, "completion_tokens": None,
            "total_tokens": 3}
        assert completion_for(
            out["turn"], "thoughtforge") == completion

    def test_no_history_passes_no_history(self) -> None:
        binding, fake = _chat_binding()
        binding.provider_turn(_REQUEST)
        assert fake.turns == [{
            "user_text": "What time does the ferry leave?"}]

    def test_max_tokens_never_reaches_the_turn(self) -> None:
        binding, fake = _chat_binding()
        out = binding.provider_turn(
            {"messages": _REQUEST["messages"],
             "max_tokens": 999})
        assert out["ok"] is True
        assert "max_tokens" not in json.dumps(fake.turns)
        assert out["model"] == PROVIDER_KEY

    def test_catalog_and_health_through_binding(self) -> None:
        binding, _fake = _chat_binding()
        catalog = binding.catalog()
        assert catalog["ok"] is True
        assert catalog["catalog"] == FILED_CATALOG
        health = binding.health_check()
        assert health["ok"] is True
        assert health["health"] == FILED_HEALTH
        assert binding_response_problems(catalog) == []
        assert binding_response_problems(health) == []

    def test_turn_response_is_detached(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.provider_turn(_REQUEST)
        out["turn"]["text"] = "mutated"
        out["completion"]["choices"][0]["message"][
            "content"] = "mutated"
        again = binding.provider_turn(_REQUEST)
        assert again["turn"]["text"] == \
            "stub answer to: What time does the ferry leave?"
        assert again["completion"]["choices"][0][
            "message"]["content"] == again["turn"]["text"]

    def test_knowledge_only_core_turn(self, tmp_path: Path) -> None:
        core = ThoughtForgeCore(
            memory_dir=tmp_path / "memory",
            db_path=tmp_path / "nonexistent.db",
            model_path=None)
        binding = HermesBinding(
            cognition_api=CognitionAPI(core=core),
            enabled=True)
        out = binding.provider_turn(_REQUEST)
        assert out["ok"] is True
        assert out["turn"]["mode"] == "knowledge_only"
        assert out["turn"]["trace_status"] == "included"
        assert out["completion"]["choices"][0][
            "message"]["content"] == out["turn"]["text"]
        assert binding_response_problems(out) == []

    def test_empty_last_user_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="empty"):
            binding.provider_turn({"messages": [
                {"role": "user", "content": ""}]})

    def test_credential_in_request_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="api_key"):
            binding.provider_turn(
                {**_REQUEST, "api_key": "secret"})

    def test_unfiled_fields_raise_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="name"):
            binding.provider_turn({"messages": [
                {"role": "user", "content": "hi",
                 "name": "ops"}]})
        with pytest.raises(ValueError, match="role"):
            binding.provider_turn({"messages": [
                {"role": "tool", "content": "hi"},
                {"role": "user", "content": "hi"}]})

    def test_not_ending_in_user_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="last message"):
            binding.provider_turn({"messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]})

    def test_response_problems_catches_tampering(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.provider_turn(_REQUEST)
        assert binding_response_problems(out) == []
        wrong = dict(out, contract_version="9.9.9")
        assert any("contract_version" in p
                   for p in binding_response_problems(wrong))
        lying = dict(out, error="boom")
        assert any("error" in p
                   for p in binding_response_problems(lying))
        other = dict(out, provider_key="other")
        assert any("provider_key" in p
                   for p in binding_response_problems(other))


# ── Memory routing ───────────────────────────────────────────────────────────


class TestMemory:
    def test_add_lands_stamped_through_tf155(
            self, tmp_path: Path) -> None:
        binding, store = _memory_binding(tmp_path)
        out = binding.memory_add(
            "user_fact", summary="ferry at 07:30",
            record_id="fct_tf158")
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
                           record_id="fct_tf158")
        found = binding.memory_inspect(record_id="fct_tf158")
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
                           record_id="fct_tf158")
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
                           record_id="fct_tf158")
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
        binding = HermesBinding()
        assert binding._cognition_api is None
        assert binding._memory_api is None
        assert HermesBinding(
            turn_callable=lambda text: {})._cognition_api is None

    def test_factories_are_lazy(self) -> None:
        built: list[str] = []
        fake = _RecordingCognitionAPI()

        def factory():
            built.append("cognition")
            return fake

        binding = HermesBinding(
            cognition_factory=factory, enabled=True)
        assert built == []
        binding.provider_turn(_REQUEST)
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

        binding = HermesBinding(
            turn_callable=stub, enabled=True)
        out = binding.provider_turn(_REQUEST)
        assert out["turn"]["text"] == \
            "stub: What time does the ferry leave?"
        assert out["turn"]["trace_status"].startswith(
            "unavailable: ")
        assert out["completion"]["choices"][0][
            "message"]["content"] == out["turn"]["text"]

    def test_memory_store_passthrough(
            self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "store")
        binding = HermesBinding(
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
        probes = run_hermes_binding_probes()
        assert len(probes) == len(HERMES_BINDING_PROBES) == 7
        assert [p["id"] for p in probes] == [
            "disabled_refusal_touches_nothing",
            "enabled_cognition_round_trip_through_binding",
            "enabled_memory_round_trip_temp_store",
            "catalog_and_health_are_honest",
            "hermes_only_calls_stay_hermes",
            "fake_consumer_consistency",
            "doc_matches_render_and_version_is_tf034"]
        assert all(p["passed"] for p in probes)

    def test_doc_matches_render(self) -> None:
        assert _read(HERMES_BINDING_DOC) == \
            render_hermes_binding_markdown()
        doc = _read(HERMES_BINDING_DOC)
        assert "Cognition Provider for the Hermes" in doc
        assert "NOT PROVEN" in doc
        assert "TF-170" in doc and "TF-159" in doc
        assert "volmarr-core" in doc
        assert API_VERSION in doc

    def test_docket_untouched_totals_unchanged(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, " \
            "101 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT" in normalized
