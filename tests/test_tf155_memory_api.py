"""
TF-155 — The memory API: store operations
(TF-057's toolset) as service calls with the
same enforcement.

The proof lives as verifiable data in
``src/thoughtforge/memory_api.py``
(``MEMORY_OPS``, ``MEMORY_REQUEST_FIELDS``,
``MEMORY_RESPONSE_FIELDS``, ``MEMORY_RULES``,
``MEMORY_PROBES``, checked by
``memory_api_problems()``); the doc
``docs/architecture/MEMORY_API.md`` is filed
verbatim as the module's render.

Pinned here:

  * the four ops are TF-057's toolset
    operations, each mapped to its OPERATIONS
    row and its underlying forge_toolset
    function(s) — the API calls those
    functions and reimplements no store logic,
  * an add through the API lands stamped
    source='manual_entry' through the TF-042
    gate; enforcement refusals (gate verdict,
    duplicate id, unknown kind, unknown prune
    mode, unknown inspect filter kind) surface
    as ok=False refusal DATA naming the
    refusal, with nothing written — never a
    silent success,
  * an invalid request (unknown op, unfiled
    field, missing kind, wrong-typed value)
    raises ValueError naming the problem —
    never silently coerced,
  * inspect through the API is read-only
    (byte-for-byte), export returns the
    TF-052 bundle verified and detached
    (mutating it cannot touch the store),
  * the contract version is TF-034's
    ``API_VERSION``, consumed, never
    reinvented; TF-057's own checker stays
    clean underneath,
  * the store is built lazily and a store /
    store factory / toolset callables can be
    injected (TF-154's injection shape),
  * the memory-api doc matches its render
    verbatim, the checker is clean, the probes
    pass 7/7, and TF-016's docket totals are
    unchanged (TF-155 has no docket share).

Proof grade: FIXTURE — synthetic temp stores
on this VM; behaviour over a real, lived-in
store and any live model behaviour are NOT
PROVEN (TF-173 class), never claimed.
All fixtures synthetic (TF-017); no canon.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from thoughtforge.cognition.api_surface import (
    API_VERSION,
    api_manifest,
    verify_api_surface,
)
from thoughtforge.knowledge.forge_toolset import (
    OPERATIONS as TOOLSET_OPERATIONS,
)
from thoughtforge.knowledge.forge_toolset import forge_toolset_problems
from thoughtforge.knowledge.portability import (
    BUNDLE_FORMAT,
    read_bundle,
    verify_bundle,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.memory_api import (
    MEMORY_API_DOC,
    MEMORY_API_FORMAT,
    MEMORY_OPS,
    MEMORY_PROBES,
    MEMORY_REQUEST_FIELDS,
    MEMORY_RESPONSE_FIELDS,
    MEMORY_RULES,
    OP_IDS,
    RESPONSE_IDS,
    MemoryAPI,
    add_request_problems,
    export_request_problems,
    inspect_request_problems,
    memory_api_as_data,
    memory_api_problems,
    memory_contract,
    memory_op_problems,
    memory_rule_problems,
    prune_request_problems,
    render_memory_api_markdown,
    request_field_ids,
    request_field_problems,
    request_fields_for,
    request_problems,
    response_problems,
    run_memory_probes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _api(tmp_path: Path) -> tuple[MemoryAPI, MemoryStore]:
    store = MemoryStore(memory_dir=tmp_path / "store")
    return MemoryAPI(store=store), store


def _seeded(tmp_path: Path) -> tuple[MemoryAPI, MemoryStore]:
    api, store = _api(tmp_path)
    out = api.add("user_fact", summary="ferry at 07:30",
                  tags=["ferry"], record_id="fct_seed")
    assert out["ok"] is True
    return api, store


def _dir_bytes(store: MemoryStore) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(store.dir.iterdir())
            if p.is_file()}


# ── The filed estate ─────────────────────────────────────────────────────────


class TestFiling:
    def test_problems_empty(self) -> None:
        assert memory_api_problems() == []

    def test_format_and_doc(self) -> None:
        assert MEMORY_API_FORMAT == "thoughtforge-memory-api/1"
        assert MEMORY_API_DOC == "docs/architecture/MEMORY_API.md"

    def test_ops_are_tf057s_four_mapped(self) -> None:
        assert OP_IDS == ("add", "inspect", "prune", "export")
        toolset = {r["name"]: r for r in TOOLSET_OPERATIONS}
        for op in MEMORY_OPS:
            assert memory_op_problems(op) == []
            row = toolset[op["toolset_operation"]]
            assert row["writes_store"] is op["writes_store"]
        by_id = {op["id"]: op for op in MEMORY_OPS}
        assert by_id["add"]["functions"] == ["add_record"]
        assert by_id["inspect"]["functions"] == [
            "inspect_store", "inspect_records", "find_record"]
        assert by_id["prune"]["functions"] == ["prune_store"]
        assert by_id["export"]["functions"] == ["export_store_bundle"]
        writes = {op["id"]: op["writes_store"] for op in MEMORY_OPS}
        assert writes == {"add": True, "inspect": False,
                          "prune": True, "export": False}

    def test_request_fields_per_op(self) -> None:
        assert list(request_field_ids("add")) == [
            "kind", "summary", "tags", "category", "value",
            "fact_value", "pattern_shape", "tone", "importance",
            "confidence", "quality", "record_id"]
        assert list(request_field_ids("inspect")) == [
            "record_id", "kind", "tag", "status", "text", "profile_id"]
        assert list(request_field_ids("prune")) == ["mode", "profile_id"]
        assert list(request_field_ids("export")) == ["destination"]
        for field in MEMORY_REQUEST_FIELDS:
            assert request_field_problems(field) == []
        kind = next(f for f in request_fields_for("add")
                    if f["id"] == "kind")
        assert kind["required"] is True
        mode = next(f for f in request_fields_for("prune")
                    if f["id"] == "mode")
        assert mode["default"] == "routine"

    def test_response_fields_are_five(self) -> None:
        assert RESPONSE_IDS == (
            "op", "ok", "status", "contract_version", "refusal")
        for field in MEMORY_RESPONSE_FIELDS:
            assert field["source"] and field["summary"]
        by_id = {f["id"]: f for f in MEMORY_RESPONSE_FIELDS}
        assert by_id["contract_version"]["source"] == "TF-034 API_VERSION"

    def test_rules_are_six_unique(self) -> None:
        ids = [r["id"] for r in MEMORY_RULES]
        assert len(ids) == len(set(ids)) == 6
        assert set(ids) == {
            "ops_are_tf057s_toolset",
            "same_enforcement_by_construction",
            "invalid_request_raises_named",
            "enforcement_refusal_is_data",
            "export_is_verified_detached_bundle",
            "version_is_tf034"}
        for rule in MEMORY_RULES:
            assert memory_rule_problems(rule) == []

    def test_validators_are_empty_safe(self) -> None:
        assert request_problems("explode", {})  # unknown op named
        assert add_request_problems(None)
        assert add_request_problems("not a mapping")
        assert inspect_request_problems(None) == []
        assert prune_request_problems(None) == []
        assert export_request_problems(None) == []
        assert response_problems(None)
        assert response_problems({})
        assert memory_op_problems(None)
        assert memory_rule_problems(None)
        assert request_field_problems(None)

    def test_contract_data_is_tf034_versioned(self) -> None:
        contract = memory_contract()
        assert contract["format"] == MEMORY_API_FORMAT
        assert contract["contract_version"] == API_VERSION
        assert contract["bundle_format"] == BUNDLE_FORMAT
        assert json.loads(json.dumps(contract)) == contract
        assert verify_api_surface() == []
        assert forge_toolset_problems() == []
        data = memory_api_as_data()
        assert data["format"] == MEMORY_API_FORMAT
        assert data["doc"] == MEMORY_API_DOC
        assert data["manifest"] == api_manifest()
        assert data["measured"] == {"probes_passed": 7, "probes_total": 7}


# ── Add ──────────────────────────────────────────────────────────────────────


class TestAdd:
    def test_each_kind_lands_stamped(self, tmp_path: Path) -> None:
        api, store = _api(tmp_path)
        outs = [
            api.add("user_preference", summary="tone: terse",
                    category="tone", value="terse", record_id="usr_a"),
            api.add("user_fact", summary="ferry at 07:30",
                    record_id="fct_a"),
            api.add("episode", summary="talked ferries",
                    record_id="ep_a"),
            api.add("response_pattern", summary="a shape",
                    pattern_shape=["ack", "answer"], record_id="pat_a"),
        ]
        assert all(o["ok"] is True and o["status"] == "added"
                   for o in outs)
        assert all(o["source"] == "manual_entry" for o in outs)
        assert all(o["contract_version"] == API_VERSION for o in outs)
        assert all(response_problems(o) == [] for o in outs)
        loaded = (store.load_profile_records() + store.load_episodes()
                  + store.load_patterns())
        assert len(loaded) == 4
        assert all(r.source == "manual_entry" for r in loaded)

    def test_add_response_is_detached(self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        out = api.inspect(record_id="fct_seed")
        out["record"]["summary"] = "mutated"
        again = api.inspect(record_id="fct_seed")
        assert again["record"]["summary"] == "ferry at 07:30"

    def test_generated_ids_use_the_store_factories(
            self, tmp_path: Path) -> None:
        api, _store = _api(tmp_path)
        assert api.add("user_fact", summary="x")[
            "record_id"].startswith("fct_")
        assert api.add("episode", summary="x")[
            "record_id"].startswith("ep_")

    def test_duplicate_id_refused_as_data_nothing_written(
            self, tmp_path: Path) -> None:
        api, store = _seeded(tmp_path)
        out = api.add("episode", summary="two", record_id="fct_seed")
        assert out["ok"] is False and out["status"] == "refused"
        assert "already exists" in out["refusal"]
        assert response_problems(out) == []
        assert store.load_episodes() == []

    def test_gate_refusal_surfaces_as_data(self, tmp_path: Path) -> None:
        api, store = _api(tmp_path)
        for fields in ({"summary": ""}, {"summary": "x" * 181},
                       {"summary": "ok", "confidence": 2.0}):
            out = api.add("user_fact", **fields)
            assert out["ok"] is False and out["status"] == "refused"
            assert out["refusal"]
        assert store.load_profile_records() == []

    def test_unknown_kind_refused_as_data(self, tmp_path: Path) -> None:
        api, store = _api(tmp_path)
        out = api.add("personality_core", summary="s")
        assert out["ok"] is False and out["status"] == "refused"
        assert "user_fact" in out["refusal"]
        assert api.inspect()["total_records"] == 0

    def test_request_dict_form(self, tmp_path: Path) -> None:
        api, _store = _api(tmp_path)
        out = api.call("add", {"kind": "user_fact",
                               "summary": "ferry at 07:30",
                               "record_id": "fct_dict"})
        assert out["ok"] is True and out["record_id"] == "fct_dict"
        assert json.loads(json.dumps(out)) == out


# ── Inspect ──────────────────────────────────────────────────────────────────


class TestInspect:
    def test_store_report(self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        out = api.inspect()
        assert out["ok"] is True and out["status"] == "inspected"
        assert out["counts"]["user_fact"] == 1
        assert out["total_records"] == 1
        assert out["recent_queue_length"] == 1
        assert out["tag_count"] == 1
        assert response_problems(out) == []

    def test_find_found_and_miss(self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        found = api.inspect(record_id="fct_seed")
        assert found["status"] == "found" and found["found"] is True
        assert found["kind"] == "user_fact" and found["home"] == "profile"
        miss = api.inspect(record_id="nope")
        assert miss["ok"] is True and miss["status"] == "not_found"
        assert miss["found"] is False

    def test_listing_filters(self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        api.add("episode", summary="talked ferries", tags=["ferry"],
                record_id="ep_seed")
        by_kind = api.inspect(kind="user_fact")
        assert [r["record_id"] for r in by_kind["records"]] == ["fct_seed"]
        by_tag = api.inspect(tag="ferry")
        assert {r["record_id"] for r in by_tag["records"]} == {
            "fct_seed", "ep_seed"}
        by_text = api.inspect(text="FERRY")
        assert [r["record_id"] for r in by_text["records"]] == ["fct_seed"]
        assert api.inspect(text="absent")["count"] == 0

    def test_unknown_filter_kind_refused_as_data(
            self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        out = api.inspect(kind="bogus_kind")
        assert out["ok"] is False and out["status"] == "refused"
        assert "user_fact" in out["refusal"]

    def test_budget_only_when_profile_named(self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        assert "budget" not in api.inspect()
        with_budget = api.inspect(profile_id="phone_low")
        assert with_budget["budget"]["profile_id"] == "phone_low"

    def test_inspect_is_read_only(self, tmp_path: Path) -> None:
        api, store = _seeded(tmp_path)
        before = _dir_bytes(store)
        api.inspect()
        api.inspect(tag="ferry")
        api.inspect(record_id="fct_seed")
        assert _dir_bytes(store) == before


# ── Prune ────────────────────────────────────────────────────────────────────


class TestPrune:
    def test_routine_prune_reports_before_after(
            self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        out = api.prune()
        assert out["ok"] is True and out["status"] == "pruned"
        assert out["mode"] == "routine"
        assert out["counts"] == {
            "archived": 0, "deleted": 0, "deduplicated": 0}
        assert out["before"] == out["after"]
        assert out["after"]["user_fact"] == 1

    def test_unknown_mode_refused_as_data_before_any_write(
            self, tmp_path: Path) -> None:
        api, store = _seeded(tmp_path)
        before = _dir_bytes(store)
        out = api.prune(mode="scorched")
        assert out["ok"] is False and out["status"] == "refused"
        assert "valid modes" in out["refusal"]
        assert _dir_bytes(store) == before

    def test_profile_caps_reach_the_prune(self, tmp_path: Path) -> None:
        api, _store = _api(tmp_path)
        for i in range(70):
            out = api.add("user_fact", summary=f"fact {i}",
                          tags=[f"t{i}"], record_id=f"fct_cap{i:03d}")
            assert out["ok"] is True
        out = api.prune(mode="light", profile_id="phone_low")
        assert out["before"]["user_fact"] == 70
        assert out["after"]["user_fact"] == 64  # phone_low hard cap


# ── Export ───────────────────────────────────────────────────────────────────


class TestExport:
    def test_export_returns_a_verified_bundle(
            self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        dest = tmp_path / "bundle.json"
        out = api.export(destination=str(dest))
        assert out["ok"] is True and out["status"] == "exported"
        assert out["verified"] is True and out["problems"] == []
        assert out["counts"]["profile_records"] == 1
        assert out["total_records"] == 1
        assert verify_bundle(out["bundle"]) == []
        assert out["bundle"]["sections"]["profile_records"][0][
            "source"] == "manual_entry"
        assert verify_bundle(read_bundle(dest)) == []
        assert response_problems(out) == []

    def test_export_without_destination_returns_the_bundle(
            self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        out = api.export()
        assert out["verified"] is True
        assert out["bundle"]["format"] == BUNDLE_FORMAT
        assert json.loads(json.dumps(out)) == out

    def test_export_bundle_is_detached(self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        out = api.export(destination=str(tmp_path / "b1.json"))
        out["bundle"]["sections"]["profile_records"][0][
            "summary"] = "mutated"
        again = api.export(destination=str(tmp_path / "b2.json"))
        assert again["bundle"]["sections"]["profile_records"][0][
            "summary"] == "ferry at 07:30"
        assert api.inspect(record_id="fct_seed")[
            "record"]["summary"] == "ferry at 07:30"

    def test_export_does_not_write_the_store(
            self, tmp_path: Path) -> None:
        api, store = _seeded(tmp_path)
        before = _dir_bytes(store)
        api.export(destination=str(tmp_path / "out.json"))
        assert _dir_bytes(store) == before


# ── Invalid requests raise; enforcement refuses as data ─────────────────────


class TestRequestShape:
    def test_unknown_op_raises_named(self, tmp_path: Path) -> None:
        api, _store = _api(tmp_path)
        with pytest.raises(ValueError, match="explode"):
            api.call("explode")

    def test_unfiled_field_raises_named(self, tmp_path: Path) -> None:
        api, _store = _api(tmp_path)
        with pytest.raises(ValueError, match="surprise"):
            api.add("user_fact", summary="x", surprise=1)
        with pytest.raises(ValueError, match="surprise"):
            api.call("prune", {"surprise": 1})

    def test_missing_kind_raises_named(self, tmp_path: Path) -> None:
        api, _store = _api(tmp_path)
        with pytest.raises(ValueError, match="kind"):
            api.call("add", {})
        with pytest.raises(ValueError, match="kind"):
            api.call("add", {"kind": ""})

    def test_wrong_typed_values_raise_named(self, tmp_path: Path) -> None:
        api, _store = _api(tmp_path)
        with pytest.raises(ValueError, match="tags"):
            api.add("user_fact", summary="x", tags="ferry")
        with pytest.raises(ValueError, match="importance"):
            api.add("user_fact", summary="x", importance="high")
        with pytest.raises(ValueError, match="mode"):
            api.prune(mode=42)

    def test_non_dict_request_raises_named(self, tmp_path: Path) -> None:
        api, _store = _api(tmp_path)
        with pytest.raises(ValueError, match="not a dict"):
            api.call("inspect", "not-a-dict")

    def test_invalid_request_builds_no_store(self) -> None:
        api = MemoryAPI()
        with pytest.raises(ValueError, match="kind"):
            api.call("add", {})
        assert api._store is None

    def test_response_problems_catches_tampering(
            self, tmp_path: Path) -> None:
        api, _store = _seeded(tmp_path)
        out = api.inspect()
        assert response_problems(out) == []
        wrong_version = dict(out, contract_version="9.9.9")
        assert any("contract_version" in p
                   for p in response_problems(wrong_version))
        lying = dict(out, refusal="refused really")
        assert any("refusal" in p for p in response_problems(lying))
        silent = {"op": "add", "ok": False, "status": "refused",
                  "contract_version": API_VERSION, "refusal": ""}
        assert any("refusal" in p for p in response_problems(silent))


# ── Injection and laziness ───────────────────────────────────────────────────


class TestInjection:
    def test_construction_builds_no_store(self) -> None:
        assert MemoryAPI()._store is None
        assert MemoryAPI(store_factory=lambda: None)._store is None

    def test_store_factory_is_lazy(self, tmp_path: Path) -> None:
        built: list = []
        store = MemoryStore(memory_dir=tmp_path / "store")

        def factory():
            built.append(1)
            return store

        api = MemoryAPI(store_factory=factory)
        assert built == [] and api._store is None
        out = api.add("user_fact", summary="ferry at 07:30",
                      record_id="fct_lazy")
        assert built == [1] and out["ok"] is True

    def test_injected_toolset_callable_is_used(
            self, tmp_path: Path) -> None:
        _api_inner, store = _api(tmp_path)
        seen: list = []

        def fake_add(store_arg, kind, **kwargs):
            seen.append((store_arg is store, kind))
            return {"status": "added", "kind": kind,
                    "record_id": "fct_fake", "source": "manual_entry",
                    "record": {"record_id": "fct_fake"}}

        api = MemoryAPI(store=store, add_callable=fake_add)
        out = api.add("user_fact", summary="anything")
        assert seen == [(True, "user_fact")]
        assert out["ok"] is True and out["record_id"] == "fct_fake"
        assert store.load_profile_records() == []


# ── State, probes, doc, docket ───────────────────────────────────────────────


class TestStateProbesDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_memory_probes()
        assert len(probes) == len(MEMORY_PROBES) == 7
        assert [p["id"] for p in probes] == [
            "add_lands_stamped_and_findable",
            "inspect_is_read_only",
            "gate_refusal_surfaces_as_data",
            "export_verified_and_detached",
            "prune_reports_and_refuses_as_data",
            "version_is_tf034_and_toolset_clean",
            "invalid_request_raises_named"]
        assert all(p["passed"] for p in probes)

    def test_doc_matches_render(self) -> None:
        assert _read(MEMORY_API_DOC) == render_memory_api_markdown()
        doc = _read(MEMORY_API_DOC)
        assert "Store Operations as Service Calls" in doc
        assert "NOT PROVEN" in doc
        assert API_VERSION in doc
        assert BUNDLE_FORMAT in doc
        assert "TF-156" in doc and "TF-162" in doc
        assert "TF-057" in doc

    def test_docket_untouched_totals_unchanged(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, " \
            "101 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT" in normalized
