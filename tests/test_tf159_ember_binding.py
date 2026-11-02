"""
TF-159 — The Ember binding: the tiny profile
as Ember's cognition layer candidate —
budgets proven on phone_low first.

The proof lives as verifiable data in
``src/thoughtforge/ember_binding.py``
(``PROFILE_ROWS``, ``BUDGET_ROWS``,
``RULE_ROWS``, ``CALL_ROWS``, ``FIELD_ROWS``,
``EMBER_BINDING_PROBES``, checked by
``ember_binding_problems()``); the doc
``docs/architecture/EMBER_BINDING.md`` is
filed verbatim as the module's render.

Pinned here:

  * the profile rows file Ember's E-024 gate
    as read READ-ONLY from the Ember tree —
    toaster cannot afford the layer, phone /
    pi draft cap 1, laptop 2 — and only
    phone is paired with a ThoughtForge
    profile (phone_low, the tiny pairing
    this slice proves),
  * the budget rows are phone_low's own
    numbers, copied exactly from the
    profile's definitions in this tree
    (TF-037 floor, TF-056 memory, TF-033
    latency, TF-079 cold-start, over the
    shipped profile JSON) and re-derived
    live in the tests — no invented number,
  * an Ember handle turn routes to TF-154's
    CognitionAPI by the filed rule (the
    request's user_text; each Episode as two
    history entries; the resolved draft cap
    as num_drafts) and returns as the filed
    CognitionReply shape; the gate / registry
    / health calls answer from the filed
    rows; memory calls route to TF-155's
    MemoryAPI's four ops, called, never
    reimplemented,
  * chat rewire / Funi turn / tool execution
    / persona custody / Episode-Well custody
    / credential custody are all filed NOT
    routed, and the binding class files no
    such method,
  * every field has a disposition — TF-154's
    turn-in and turn-out fields checked live
    against TF-154's own filings, any
    credential filed no-home,
  * the binding is DEFAULT DISABLED
    (``DEFAULT_ENABLED`` False, filed as
    data): while disabled every routed call
    returns the refusal as data naming the
    binding disabled and touches neither API
    (counting factories prove nothing is
    built); enabling is the explicit
    constructor flag / ``enable()`` act; an
    enabled toaster turn returns the filed
    cannot-afford refusal and routes nothing,
  * responses are detached JSON-safe data,
    contract_version is TF-034's
    ``API_VERSION``, the doc matches its
    render verbatim, the checker is clean,
    the probes pass 7/7, and TF-016's docket
    totals are unchanged (TF-159 has no
    docket share).

Proof grade: FIXTURE — a fake Ember consumer
shaped on the filed E-024 data; NO live
Ember process was routed and NO Ember code
was imported or copied; live end-to-end
routing is NOT PROVEN (TF-170's gate owns
it), never claimed. Budgets are proven as
filed data re-derived live from phone_low's
own definitions plus TF-037's floor proof —
no phone hardware ran. All fixtures
synthetic (TF-017); no canon.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from thoughtforge.cognition.api_surface import API_VERSION
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.latency_budgets import (
    budget_for_profile as latency_budget_for,
)
from thoughtforge.cognition.small_model_floor import (
    floor_budget,
)
from thoughtforge.cognition_api import (
    TURN_IN_FIELDS,
    TURN_OUT_FIELDS,
    CognitionAPI,
)
from thoughtforge.ember_binding import (
    BUDGET_IDS,
    BUDGET_ROWS,
    CALL_IDS,
    CALL_ROWS,
    DEFAULT_EMBER_PROFILE,
    DEFAULT_ENABLED,
    DISABLED_ERROR,
    EMBER_BINDING_DOC,
    EMBER_BINDING_FORMAT,
    EMBER_BINDING_PROBES,
    EMBER_PACKAGE,
    EMBER_PROFILES,
    EMBER_VERSION,
    EPISODE_FIELDS,
    FIELD_IDS,
    FIELD_ROWS,
    FILED_HEALTH,
    FILED_REGISTRY,
    NOT_ROUTED_CALL_IDS,
    PROFILE_ROWS,
    PROVIDER_KEY,
    REPLY_FIELDS,
    REQUEST_FIELDS,
    ROUTED_CALL_IDS,
    RULE_IDS,
    RULE_ROWS,
    TINY_PROFILE_ID,
    TOASTER_ERROR,
    EmberBinding,
    assemble_history,
    binding_response_problems,
    budget_row_problems,
    budget_value,
    call_row_problems,
    ember_binding_as_data,
    ember_binding_contract,
    ember_binding_problems,
    ember_request_problems,
    expected_budget_values,
    field_row_problems,
    field_rows_for,
    profile_row_for,
    profile_row_problems,
    render_ember_binding_markdown,
    reply_for,
    resolved_num_drafts,
    rule_row_problems,
    run_ember_binding_probes,
)
from thoughtforge.inference.cold_start_budgets import (
    budget_for_profile as cold_budget_for,
)
from thoughtforge.knowledge.memory_budgets import (
    budget_for_profile as memory_budget_for,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.memory_api import MemoryAPI
from thoughtforge.utils.config import load_hardware_profile

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


def _chat_binding() -> tuple[EmberBinding, _RecordingCognitionAPI]:
    fake = _RecordingCognitionAPI()
    return EmberBinding(cognition_api=fake, enabled=True), fake


def _memory_binding(
    tmp_path: Path,
) -> tuple[EmberBinding, MemoryStore]:
    store = MemoryStore(memory_dir=tmp_path / "store")
    binding = EmberBinding(
        memory_api=MemoryAPI(store=store), enabled=True)
    return binding, store


_REQUEST = {"user_text": "What time does the ferry leave?",
            "profile": "phone",
            "episodes": [
                {"operator_input": "hello",
                 "ember_reply": "hi there"}]}
_EPISODES = [
    {"operator_input": "hello", "ember_reply": "hi there"},
    {"operator_input": "and the bus?",
     "ember_reply": "at nine"},
]


# ── The filed estate ─────────────────────────────────────────────────────────


class TestFiling:
    def test_problems_empty(self) -> None:
        assert ember_binding_problems() == []

    def test_format_doc_and_defaults(self) -> None:
        assert EMBER_BINDING_FORMAT == \
            "thoughtforge-ember-binding/1"
        assert EMBER_BINDING_DOC == \
            "docs/architecture/EMBER_BINDING.md"
        assert DEFAULT_ENABLED is False
        assert "disabled" in DISABLED_ERROR
        assert "cannot afford" in TOASTER_ERROR
        assert PROVIDER_KEY == "thoughtforge"
        assert EMBER_PACKAGE == "ember-agent"
        assert EMBER_VERSION == "0.2.0"
        assert TINY_PROFILE_ID == "phone_low"
        assert DEFAULT_EMBER_PROFILE == "phone"

    def test_profile_rows_file_e024_gate(self) -> None:
        assert [r["ember_profile"] for r in PROFILE_ROWS] == [
            "toaster", "phone", "pi", "laptop"]
        for row in PROFILE_ROWS:
            assert profile_row_problems(row) == []
            assert row["enabled_by_default"] is False
        gate = {
            r["ember_profile"]: (
                r["affords"], r["draft_cap"],
                r["thoughtforge_profile"])
            for r in PROFILE_ROWS}
        assert gate == {
            "toaster": (False, None, None),
            "phone": (True, 1, "phone_low"),
            "pi": (True, 1, None),
            "laptop": (True, 2, None)}
        assert profile_row_for("phone")["id"] == \
            "profile_phone"
        assert profile_row_for("watch") is None

    def test_only_phone_pairs_with_phone_low(self) -> None:
        paired = [
            r["ember_profile"] for r in PROFILE_ROWS
            if r["thoughtforge_profile"] is not None]
        assert paired == ["phone"]

    def test_budget_rows_match_live_phone_low(self) -> None:
        expected = expected_budget_values()
        assert {r["id"]: r["value"]
                for r in BUDGET_ROWS} == expected
        assert list(BUDGET_IDS) == list(expected)
        for row in BUDGET_ROWS:
            assert budget_row_problems(row) == []
            assert row["source"]
        floor = floor_budget("phone_low")
        assert budget_value(
            "generation.max_response_tokens") == \
            floor.max_response_tokens == 180
        assert budget_value(
            "generation.token_budget_per_turn") == 180
        assert budget_value(
            "generation.max_context_tokens") == \
            floor.max_context_tokens == 512
        assert budget_value("generation.draft_count") == \
            floor.draft_count == 2
        assert budget_value(
            "generation.max_refinement_passes") == 1
        assert budget_value(
            "memory.max_memory_bundle_facts") == \
            floor.max_memory_bundle_facts == 5

    def test_budget_rows_cite_their_sources(self) -> None:
        by_id = {r["id"]: r for r in BUDGET_ROWS}
        profile = load_hardware_profile("phone_low")
        assert budget_value("profile.ram_gb") == \
            profile["hardware"]["ram_gb"] == 2
        assert budget_value("profile.max_params_b") == \
            profile["model"]["max_params_b"] == 1.1
        assert budget_value("profile.quantization") == "q2_k"
        assert budget_value("profile.recommended_models") == [
            "TinyLlama-1.1B-Chat", "Granite-1B-Instruct"]
        latency = latency_budget_for("phone_low")
        assert (budget_value("latency.retrieval_ms"),
                budget_value("latency.generation_ms"),
                budget_value("latency.salvage_ms"),
                budget_value("latency.total_ms")) == (
                    latency.retrieval_ms,
                    latency.generation_ms,
                    latency.salvage_ms,
                    latency.total_ms) == (500, 3000, 1500, 5000)
        cold = cold_budget_for("phone_low")
        assert (budget_value("cold_start.profile_load_ms"),
                budget_value("cold_start.total_ms")) == (
                    cold.profile_load_ms,
                    cold.total_ms) == (150, 7650)
        memory = memory_budget_for("phone_low")
        assert budget_value(
            "memory.user_facts_cap_hard") == \
            memory.caps["user_fact"]["hard"] == 64
        assert budget_value("retrieval.sql_max_results") == \
            memory.sql_max_results == 5
        assert budget_value("retrieval.vector_top_k") == \
            memory.vector_top_k == 3
        assert "small_model_floor" in by_id[
            "generation.max_response_tokens"]["source"]
        assert "latency_budgets" in by_id[
            "latency.total_ms"]["source"]
        assert "cold_start_budgets" in by_id[
            "cold_start.total_ms"]["source"]
        assert "memory_budgets" in by_id[
            "memory.user_facts_cap_hard"]["source"]

    def test_calls_complete_and_unique(self) -> None:
        assert CALL_IDS == (
            "ember_turn", "profile_gate", "registry",
            "health_check", "memory_add", "memory_inspect",
            "memory_prune", "memory_export", "chat_rewire",
            "funi_model_turn", "tool_execution",
            "persona_custody", "episode_well_custody",
            "credential_custody")
        assert ROUTED_CALL_IDS == (
            "ember_turn", "profile_gate", "registry",
            "health_check", "memory_add", "memory_inspect",
            "memory_prune", "memory_export")
        assert NOT_ROUTED_CALL_IDS == (
            "chat_rewire", "funi_model_turn",
            "tool_execution", "persona_custody",
            "episode_well_custody", "credential_custody")
        for row in CALL_ROWS:
            assert call_row_problems(row) == []
        by_id = {r["id"]: r for r in CALL_ROWS}
        assert "CognitionAPI" in \
            by_id["ember_turn"]["thoughtforge_route"]
        for op in ("add", "inspect", "prune", "export"):
            route = by_id[f"memory_{op}"]["thoughtforge_route"]
            assert "MemoryAPI" in route

    def test_ember_only_calls_all_not_routed(self) -> None:
        by_id = {r["id"]: r for r in CALL_ROWS}
        for call_id in NOT_ROUTED_CALL_IDS:
            assert by_id[call_id]["routed"] is False
            assert by_id[call_id]["surface"] == "ember_only"
            assert by_id[call_id]["binding_method"] is None
            assert "not routed" in \
                by_id[call_id]["thoughtforge_route"]
        assert "TF-160" in \
            by_id["credential_custody"]["thoughtforge_route"]
        for name in ("rewire_chat", "run_funi", "execute_tool",
                     "resolve_persona", "read_well",
                     "get_credential", "open_cognition"):
            assert not hasattr(EmberBinding, name)

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
        assert "reply.provider" in out_ids
        assert "reply.api_version" in out_ids

    def test_credential_filed_no_home(self) -> None:
        by_id = {f["id"]: f for f in FIELD_ROWS}
        key = by_id["request.api_key"]
        assert key["disposition"] == "no_home_not_passed"
        assert "never" in key["target"]

    def test_request_and_episode_fields_are_ember(self) -> None:
        assert list(REQUEST_FIELDS) == [
            "user_text", "episodes", "profile", "num_drafts"]
        assert list(EPISODE_FIELDS) == [
            "operator_input", "ember_reply"]
        assert list(EMBER_PROFILES) == [
            "toaster", "phone", "pi", "laptop"]
        assert list(REPLY_FIELDS) == [
            "text", "provider", "api_version", "turn_id",
            "mode", "enforcement_passed", "citations",
            "token_count"]

    def test_registry_and_health_filings_are_honest(self) -> None:
        assert FILED_REGISTRY == {
            "provider": PROVIDER_KEY, "implemented": True,
            "api_major": 1,
            "afforded_profiles": ["phone", "pi", "laptop"]}
        assert FILED_HEALTH["provider"] == PROVIDER_KEY
        assert FILED_HEALTH["api_version"] == API_VERSION
        assert FILED_HEALTH["last_ok"] is None
        assert FILED_HEALTH["healthy"] is True
        assert "no live model health is claimed" in \
            FILED_HEALTH["detail"]

    def test_rules_are_nine_unique(self) -> None:
        assert list(RULE_IDS) == [
            "ember_candidate_real_and_named",
            "candidate_never_a_chat_rewire",
            "budgets_proven_on_phone_low_first",
            "toaster_cannot_afford",
            "off_by_default",
            "called_never_reimplemented",
            "no_ember_code_imported_or_copied",
            "every_field_has_a_disposition",
            "responses_detached_json_safe_versioned"]
        for rule in RULE_ROWS:
            assert rule_row_problems(rule) == []

    def test_validators_are_empty_safe(self) -> None:
        assert profile_row_problems(None)
        assert budget_row_problems(None)
        assert call_row_problems(None)
        assert field_row_problems(None)
        assert rule_row_problems(None)
        assert binding_response_problems(None)
        assert binding_response_problems({})
        assert ember_request_problems(None)
        assert ember_request_problems({})
        assert ember_request_problems({"user_text": ""})
        assert ember_request_problems(
            {"user_text": "hi", "profile": "watch"})
        assert ember_request_problems(
            {"user_text": "hi", "num_drafts": 0})
        assert ember_request_problems(
            {"user_text": "hi",
             "episodes": [{"operator_input": "x"}]})

    def test_contract_data(self) -> None:
        contract = ember_binding_contract()
        assert contract["format"] == EMBER_BINDING_FORMAT
        assert contract["contract_version"] == API_VERSION
        assert contract["default_enabled"] is False
        assert contract["provider_key"] == PROVIDER_KEY
        assert contract["tiny_profile"] == "phone_low"
        assert json.loads(json.dumps(contract)) == contract
        data = ember_binding_as_data()
        assert data["format"] == EMBER_BINDING_FORMAT
        assert data["doc"] == EMBER_BINDING_DOC
        assert data["measured"] == {
            "probes_passed": 7, "probes_total": 7}


# ── Disabled by default ──────────────────────────────────────────────────────


class TestDisabled:
    def test_default_is_disabled(self) -> None:
        assert EmberBinding().enabled is False
        assert EmberBinding(enabled=True).enabled is True

    def test_every_routed_call_refuses_as_data(self) -> None:
        binding = EmberBinding()
        outs = [
            binding.ember_turn(_REQUEST),
            binding.profile_gate("phone"),
            binding.registry(),
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

        binding = EmberBinding(
            cognition_factory=cog_factory,
            memory_factory=mem_factory)
        binding.ember_turn(_REQUEST)
        binding.memory_inspect()
        assert built == []
        assert binding._cognition_api is None
        assert binding._memory_api is None

    def test_disabled_refuses_garbage_without_raising(self) -> None:
        binding = EmberBinding()
        out = binding.ember_turn(
            {"user_text": "", "api_key": "secret"})
        assert out["ok"] is False
        assert out["error"] == DISABLED_ERROR

    def test_enable_and_disable_are_explicit_acts(self) -> None:
        binding, _fake = _chat_binding()
        binding.disable()
        assert binding.ember_turn(
            _REQUEST)["ok"] is False
        binding.enable()
        assert binding.enabled is True
        assert binding.ember_turn(
            _REQUEST)["ok"] is True


# ── Ember turn routing ───────────────────────────────────────────────────────


class TestEmberTurn:
    def test_assembly_rule_is_the_filed_one(self) -> None:
        assert assemble_history(_EPISODES) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "and the bus?"},
            {"role": "assistant", "content": "at nine"},
        ]
        assert assemble_history([]) == []
        assert assemble_history(None) == []

    def test_turn_routes_by_the_filed_rule(self) -> None:
        binding, fake = _chat_binding()
        out = binding.ember_turn(
            {"user_text": "What time does the ferry leave?",
             "profile": "phone", "episodes": _EPISODES})
        assert len(fake.turns) == 1
        call = fake.turns[0]
        assert call["user_text"] == \
            "What time does the ferry leave?"
        assert call["num_drafts"] == 1
        history = call["history"]
        # A real ChatHistory (E-024's rule), not a list:
        # a real core reads it via to_prompt_context().
        assert not isinstance(history, list)
        assert history.to_messages() == \
            assemble_history(_EPISODES)
        assert "hello" in history.to_prompt_context()
        assert out["ok"] is True and out["status"] == "replied"
        assert out["call"] == "ember_turn"
        assert out["profile"] == "phone"
        assert out["turn"]["text"] == \
            "stub answer to: What time does the ferry leave?"
        assert binding_response_problems(out) == []

    def test_reply_is_the_filed_cognition_reply(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.ember_turn(_REQUEST)
        reply = out["reply"]
        assert set(reply) == set(REPLY_FIELDS)
        assert reply == {
            "text": "stub answer to: What time does "
                    "the ferry leave?",
            "provider": PROVIDER_KEY,
            "api_version": API_VERSION,
            "turn_id": "turn_stub",
            "mode": "stub_mode",
            "enforcement_passed": True,
            "citations": [],
            "token_count": 3}
        assert reply_for(out["turn"]) == reply

    def test_no_episodes_passes_no_history(self) -> None:
        binding, fake = _chat_binding()
        binding.ember_turn(
            {"user_text": "What time does the ferry leave?"})
        assert fake.turns == [{
            "user_text": "What time does the ferry leave?",
            "num_drafts": 1}]
        # The default profile is the filed phone seat.
        assert binding.ember_turn(
            {"user_text": "hi"})["profile"] == "phone"

    def test_draft_caps_follow_e024(self) -> None:
        assert resolved_num_drafts("phone") == 1
        assert resolved_num_drafts("pi") == 1
        assert resolved_num_drafts("laptop") == 2
        assert resolved_num_drafts("toaster") is None
        assert resolved_num_drafts("phone", 2) == 2
        binding, fake = _chat_binding()
        binding.ember_turn({"user_text": "hi",
                            "profile": "laptop"})
        assert fake.turns[0]["num_drafts"] == 2
        binding.ember_turn({"user_text": "hi",
                            "profile": "phone",
                            "num_drafts": 2})
        assert fake.turns[1]["num_drafts"] == 2

    def test_toaster_refused_and_routes_nothing(self) -> None:
        binding, fake = _chat_binding()
        out = binding.ember_turn(
            {**_REQUEST, "profile": "toaster"})
        assert out["ok"] is False
        assert out["status"] == "refused"
        assert out["error"] == TOASTER_ERROR
        assert out["profile"] == "toaster"
        assert fake.turns == []
        assert binding_response_problems(out) == []
        # No explicit draft count buys affordance.
        out = binding.ember_turn(
            {"user_text": "hi", "profile": "toaster",
             "num_drafts": 2})
        assert out["ok"] is False and fake.turns == []

    def test_gate_registry_and_health_through_binding(
            self) -> None:
        binding, _fake = _chat_binding()
        gate = binding.profile_gate("phone")
        assert gate["ok"] is True
        assert gate["status"] == "afforded"
        assert gate["gate"]["draft_cap"] == 1
        assert gate["gate"]["thoughtforge_profile"] == \
            "phone_low"
        toaster = binding.profile_gate("toaster")
        assert toaster["status"] == "cannot_afford"
        assert toaster["gate"]["affords"] is False
        registry = binding.registry()
        assert registry["registry"] == FILED_REGISTRY
        health = binding.health_check()
        assert health["health"] == FILED_HEALTH
        assert binding_response_problems(gate) == []
        assert binding_response_problems(registry) == []
        assert binding_response_problems(health) == []
        with pytest.raises(ValueError, match="watch"):
            binding.profile_gate("watch")

    def test_turn_response_is_detached(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.ember_turn(_REQUEST)
        out["turn"]["text"] = "mutated"
        out["reply"]["text"] = "mutated"
        again = binding.ember_turn(_REQUEST)
        assert again["turn"]["text"] == \
            "stub answer to: What time does the ferry leave?"
        assert again["reply"]["text"] == again["turn"]["text"]

    def test_knowledge_only_core_turn_within_budget(
            self, tmp_path: Path) -> None:
        core = ThoughtForgeCore(
            memory_dir=tmp_path / "memory",
            db_path=tmp_path / "nonexistent.db",
            model_path=None)
        binding = EmberBinding(
            cognition_api=CognitionAPI(core=core),
            enabled=True)
        out = binding.ember_turn(_REQUEST)
        assert out["ok"] is True
        assert out["turn"]["mode"] == "knowledge_only"
        assert out["turn"]["trace_status"] == "included"
        assert out["reply"]["text"] == out["turn"]["text"]
        assert out["reply"]["provider"] == PROVIDER_KEY
        # The filed phone_low response cap, proven first.
        assert out["turn"]["token_count"] <= \
            budget_value("generation.max_response_tokens") == 180
        assert binding_response_problems(out) == []

    def test_empty_user_text_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="user_text"):
            binding.ember_turn({"user_text": ""})

    def test_credential_in_request_raises_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="api_key"):
            binding.ember_turn(
                {**_REQUEST, "api_key": "secret"})

    def test_unfiled_fields_raise_named(self) -> None:
        binding, _fake = _chat_binding()
        with pytest.raises(ValueError, match="persona"):
            binding.ember_turn(
                {**_REQUEST, "persona": "someone"})
        with pytest.raises(ValueError, match="watch"):
            binding.ember_turn(
                {**_REQUEST, "profile": "watch"})
        with pytest.raises(ValueError, match="num_drafts"):
            binding.ember_turn(
                {**_REQUEST, "num_drafts": 0})
        with pytest.raises(ValueError, match="ember_reply"):
            binding.ember_turn(
                {"user_text": "hi",
                 "episodes": [{"operator_input": "x"}]})

    def test_response_problems_catches_tampering(self) -> None:
        binding, _fake = _chat_binding()
        out = binding.ember_turn(_REQUEST)
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
            record_id="fct_tf159")
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
                           record_id="fct_tf159")
        found = binding.memory_inspect(record_id="fct_tf159")
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
                           record_id="fct_tf159")
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
                           record_id="fct_tf159")
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
        binding = EmberBinding()
        assert binding._cognition_api is None
        assert binding._memory_api is None
        assert EmberBinding(
            turn_callable=lambda text: {})._cognition_api is None

    def test_factories_are_lazy(self) -> None:
        built: list[str] = []
        fake = _RecordingCognitionAPI()

        def factory():
            built.append("cognition")
            return fake

        binding = EmberBinding(
            cognition_factory=factory, enabled=True)
        assert built == []
        binding.ember_turn(_REQUEST)
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

        binding = EmberBinding(
            turn_callable=stub, enabled=True)
        out = binding.ember_turn(_REQUEST)
        assert out["turn"]["text"] == \
            "stub: What time does the ferry leave?"
        assert out["turn"]["trace_status"].startswith(
            "unavailable: ")
        assert out["reply"]["text"] == out["turn"]["text"]

    def test_memory_store_passthrough(
            self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "store")
        binding = EmberBinding(
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
        probes = run_ember_binding_probes()
        assert len(probes) == len(EMBER_BINDING_PROBES) == 7
        assert [p["id"] for p in probes] == [
            "disabled_refusal_touches_nothing",
            "enabled_cognition_round_trip_through_binding",
            "enabled_memory_round_trip_temp_store",
            "budgets_are_phone_low_live",
            "toaster_refused_and_gate_is_honest",
            "fake_consumer_consistency",
            "doc_matches_render_and_version_is_tf034"]
        assert all(p["passed"] for p in probes)

    def test_doc_matches_render(self) -> None:
        assert _read(EMBER_BINDING_DOC) == \
            render_ember_binding_markdown()
        doc = _read(EMBER_BINDING_DOC)
        assert "Tiny Profile as Ember's" in doc
        assert "NOT PROVEN" in doc
        assert "TF-170" in doc and "TF-160" in doc
        assert "phone_low" in doc
        assert "180" in doc and "512" in doc
        assert "5000" in doc and "7650" in doc
        assert API_VERSION in doc

    def test_docket_untouched_totals_unchanged(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 32 OPEN, 1 OPEN-ENVIRONMENT, " \
            "100 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT" in normalized
