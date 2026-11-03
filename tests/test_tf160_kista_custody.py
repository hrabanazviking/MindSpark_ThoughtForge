"""
TF-160 — Kista custody: any credential a
binding needs lives in Kista-shaped custody,
never in configs.

The proof lives as verifiable data in
``src/thoughtforge/kista_custody.py``
(``BINDING_NEED_ROWS``, ``CUSTODY_ROWS``,
``CONFIG_KEY_ROWS``, ``FIELD_ROWS``,
``RULE_ROWS``, ``KISTA_CUSTODY_PROBES``,
checked by ``kista_custody_problems()``);
the doc ``docs/architecture/KISTA_CUSTODY.md``
is filed verbatim as the module's render.

Pinned here:

  * the Kista shape is filed from the real
    plugin, read READ-ONLY from the Hermes
    tree — kista-secret-source 0.2.0, the
    reference pattern verbatim, the vault,
    the CLI resolution, the source config
    with enabled default False,
  * no binding needs a credential today —
    re-derived live from TF-156..TF-159's
    own filings (hermes/ember
    credential_custody NOT routed, samtal
    seat.token no-home, runmal no credential
    surface); only the backend factory row
    needs one, custody-referenced,
  * the shipped ``configs/user_config.yaml``
    is migrated: the raw keys
    ``openai_api_key`` / ``hf_token`` are
    gone and the filed ``*_ref`` custody
    references stand in their place; every
    shipped config scans clean live,
  * the validator refuses the raw shape —
    a non-empty raw value, an empty raw key,
    a malformed reference, at any depth,
  * custody is DEFAULT DISABLED: while
    disabled every resolution refuses as
    data and builds no resolver; an enabled
    custody with no resolver refuses as
    not-configured (no live Kista shipped),
  * a resolved value is delivered only to
    the consumer callable and appears in NO
    serialised output (sentinel-proven),
  * D-CFG-4 is FIXED by TF-160 and TF-016's
    totals moved in the open (137 — 31 OPEN
    / 1 OPEN-ENVIRONMENT / 101 FIXED / 4
    CLOSED-BY-VERDICT).

Proof grade: FIXTURE — a fake resolver
shaped on the filed Kista data; NO live
Kista vault or CLI was called and NO Kista
or Hermes code was imported or copied;
live end-to-end resolution is NOT PROVEN
(TF-170's gate owns it), never claimed.
All fixtures synthetic (TF-017); no canon.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from thoughtforge.cognition.api_surface import API_VERSION
from thoughtforge.ember_binding import (
    NOT_ROUTED_CALL_IDS as EMBER_NOT_ROUTED,
)
from thoughtforge.hermes_binding import (
    NOT_ROUTED_CALL_IDS as HERMES_NOT_ROUTED,
)
from thoughtforge.inference.unified_backend import (
    load_backend_from_config,
)
from thoughtforge.kista_custody import (
    BINDING_NEED_IDS,
    BINDING_NEED_ROWS,
    CONFIG_KEY_IDS,
    CONFIG_KEY_ROWS,
    CREDENTIAL_CONFIG_KEYS,
    CREDENTIAL_REF_KEYS,
    CUSTODY_ROW_IDS,
    CUSTODY_ROWS,
    DEFAULT_ENABLED,
    DISABLED_ERROR,
    FIELD_IDS,
    FIELD_ROWS,
    KISTA_CUSTODY_DOC,
    KISTA_CUSTODY_FORMAT,
    KISTA_CUSTODY_PROBES,
    NOT_CONFIGURED_ERROR,
    PLUGIN_NAME,
    PLUGIN_VERSION,
    REFERENCE_PATTERN,
    RULE_IDS,
    RULE_ROWS,
    SENTINEL_SECRET,
    SOURCE_NAME,
    KistaCustody,
    binding_need_row_for,
    binding_need_row_problems,
    config_key_row_for,
    config_key_row_problems,
    config_mapping_problems,
    custody_response_problems,
    custody_row_problems,
    field_row_problems,
    field_rows_for,
    kista_custody_as_data,
    kista_custody_contract,
    kista_custody_problems,
    parse_reference,
    reference_for_service_field,
    reference_problems,
    render_kista_custody_markdown,
    resolve_request_problems,
    rule_row_problems,
    run_kista_custody_probes,
    shipped_config_problems,
)
from thoughtforge.runmal_binding import (
    CALL_ROWS as RUNMAL_CALL_ROWS,
)
from thoughtforge.runmal_binding import (
    FIELD_ROWS as RUNMAL_FIELD_ROWS,
)
from thoughtforge.samtal_binding import (
    FIELD_ROWS as SAMTAL_FIELD_ROWS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_REF = "kista://thoughtforge/openai_api_key"


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


class _FakeResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, service: str, field: str):
        self.calls.append((service, field))
        if service == "thoughtforge":
            return SENTINEL_SECRET
        return None


def _enabled() -> tuple[KistaCustody, _FakeResolver]:
    fake = _FakeResolver()
    return KistaCustody(resolver=fake, enabled=True), fake


# ── The filed estate ─────────────────────────────────────────────────────────


class TestFiling:
    def test_problems_empty(self) -> None:
        assert kista_custody_problems() == []

    def test_format_doc_and_defaults(self) -> None:
        assert KISTA_CUSTODY_FORMAT == \
            "thoughtforge-kista-custody/1"
        assert KISTA_CUSTODY_DOC == \
            "docs/architecture/KISTA_CUSTODY.md"
        assert DEFAULT_ENABLED is False
        assert "disabled" in DISABLED_ERROR
        assert "not configured" in NOT_CONFIGURED_ERROR
        assert PLUGIN_NAME == "kista-secret-source"
        assert PLUGIN_VERSION == "0.2.0"
        assert SOURCE_NAME == "kista"

    def test_reference_pattern_is_the_plugins_own(self) -> None:
        assert REFERENCE_PATTERN == (
            r"^kista://(?P<service>[a-z0-9][a-z0-9_.-]{0,127})"
            r"/(?P<field>[a-z_][a-z0-9_]{0,63})$")
        assert parse_reference(_REF) == (
            "thoughtforge", "openai_api_key")
        assert parse_reference(
            "kista://thoughtforge/hf_token") == (
                "thoughtforge", "hf_token")
        assert reference_for_service_field(
            "thoughtforge", "hf_token") == \
            "kista://thoughtforge/hf_token"

    def test_reference_refusals(self) -> None:
        for bad in ("http://thoughtforge/openai_api_key",
                    "kista://ThoughtForge/openai_api_key",
                    "kista://thoughtforge",
                    "kista://thoughtforge/",
                    "kista://thoughtforge/OpenAI",
                    "kista://thought forge/hf_token",
                    "sk-live-value", "", None, 42):
            assert parse_reference(bad) is None
            assert reference_problems(bad)

    def test_binding_need_rows(self) -> None:
        assert list(BINDING_NEED_IDS) == [
            "need_runmal_binding", "need_samtal_binding",
            "need_hermes_binding", "need_ember_binding",
            "need_service_shape", "need_backend_factory"]
        for row in BINDING_NEED_ROWS:
            assert binding_need_row_problems(row) == []
        needing = [r["id"] for r in BINDING_NEED_ROWS
                   if r["needs_credential"]]
        assert needing == ["need_backend_factory"]
        assert binding_need_row_for(
            "backend factory")["id"] == \
            "need_backend_factory"
        assert binding_need_row_for("watch") is None

    def test_binding_needs_rederived_live(self) -> None:
        assert "credential_custody" in HERMES_NOT_ROUTED
        assert "credential_custody" in EMBER_NOT_ROUTED
        token = next(f for f in SAMTAL_FIELD_ROWS
                     if f["id"] == "seat.token")
        assert token["disposition"] == "no_home_not_passed"
        assert not any(
            "credential" in r["id"] or "api_key" in r["id"]
            for r in RUNMAL_CALL_ROWS)
        assert not any(
            "credential" in f["id"] or "api_key" in f["id"]
            for f in RUNMAL_FIELD_ROWS)

    def test_custody_rows_file_the_plugin_shape(self) -> None:
        assert list(CUSTODY_ROW_IDS) == [
            "custody_plugin", "custody_reference_form",
            "custody_vault", "custody_cli_resolution",
            "custody_source_config",
            "custody_hermes_profile_key_file"]
        for row in CUSTODY_ROWS:
            assert custody_row_problems(row) == []
        by_id = {r["id"]: r for r in CUSTODY_ROWS}
        assert "register_secret_source" in \
            by_id["custody_plugin"]["shape"]
        assert "_REFERENCE_RE" in \
            by_id["custody_reference_form"]["verified"]
        assert ".vault_key" in by_id["custody_vault"]["shape"]
        assert "vault.json.enc" in \
            by_id["custody_vault"]["shape"]
        assert "kista get" in \
            by_id["custody_cli_resolution"]["shape"]
        assert "secrets/aesir.key" in by_id[
            "custody_hermes_profile_key_file"]["shape"]

    def test_config_key_rows(self) -> None:
        assert list(CONFIG_KEY_IDS) == [
            "config_openai_api_key", "config_hf_token"]
        assert CREDENTIAL_CONFIG_KEYS == (
            "openai_api_key", "hf_token")
        assert CREDENTIAL_REF_KEYS == (
            "openai_api_key_ref", "hf_token_ref")
        for row in CONFIG_KEY_ROWS:
            assert config_key_row_problems(row) == []
            assert row["reference"] == \
                f"kista://thoughtforge/{row['raw_key']}"
        assert config_key_row_for(
            "hf_token")["ref_key"] == "hf_token_ref"
        assert config_key_row_for("watch") is None

    def test_field_rows_and_value_no_home(self) -> None:
        ids = [f["id"] for f in FIELD_ROWS]
        assert len(ids) == len(set(ids)) == len(FIELD_IDS)
        for field in FIELD_ROWS:
            assert field_row_problems(field) == []
        by_id = {f["id"]: f for f in FIELD_ROWS}
        assert by_id["response.value"]["disposition"] == \
            "no_home_never_filed"
        assert by_id["config.raw_credential_key"][
            "disposition"] == "no_home_not_passed"
        assert by_id["request.secret_value"][
            "disposition"] == "no_home_not_passed"
        assert len(field_rows_for("in")) == 4
        assert len(field_rows_for("out")) == 4

    def test_rules_are_ten_unique(self) -> None:
        assert list(RULE_IDS) == [
            "never_in_configs",
            "reference_never_value",
            "resolved_value_never_serialised",
            "off_by_default",
            "resolution_only_through_injected_resolver",
            "failures_name_a_kind_never_a_value",
            "no_binding_needs_a_credential_today",
            "called_never_reimplemented",
            "every_field_has_a_disposition",
            "responses_detached_json_safe_versioned"]
        for rule in RULE_ROWS:
            assert rule_row_problems(rule) == []

    def test_validators_are_empty_safe(self) -> None:
        assert binding_need_row_problems(None)
        assert custody_row_problems(None)
        assert config_key_row_problems(None)
        assert field_row_problems(None)
        assert rule_row_problems(None)
        assert custody_response_problems(None)
        assert custody_response_problems({})
        assert resolve_request_problems(None)
        assert resolve_request_problems({})
        assert config_mapping_problems(None)
        assert config_mapping_problems("not-a-mapping")

    def test_contract_data(self) -> None:
        contract = kista_custody_contract()
        assert contract["format"] == KISTA_CUSTODY_FORMAT
        assert contract["contract_version"] == API_VERSION
        assert contract["default_enabled"] is False
        assert contract["reference_pattern"] == \
            REFERENCE_PATTERN
        assert json.loads(json.dumps(contract)) == contract
        assert SENTINEL_SECRET not in json.dumps(contract)
        data = kista_custody_as_data()
        assert data["format"] == KISTA_CUSTODY_FORMAT
        assert data["doc"] == KISTA_CUSTODY_DOC
        assert data["measured"] == {
            "probes_passed": 7, "probes_total": 7}
        assert SENTINEL_SECRET not in json.dumps(data)


# ── The shipped-config migration ─────────────────────────────────────────────


class TestShippedConfigs:
    def test_shipped_configs_scan_clean(self) -> None:
        assert shipped_config_problems() == []

    def test_user_config_carries_references(self) -> None:
        import yaml

        parsed = yaml.safe_load(
            _read("configs/user_config.yaml"))
        assert "openai_api_key" not in parsed
        assert "hf_token" not in parsed
        assert parsed["openai_api_key_ref"] == \
            "kista://thoughtforge/openai_api_key"
        assert parsed["hf_token_ref"] == \
            "kista://thoughtforge/hf_token"
        assert parsed["backend"] == "none"

    def test_no_config_file_names_a_raw_key(self) -> None:
        for path in (REPO_ROOT / "configs").glob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            for key in CREDENTIAL_CONFIG_KEYS:
                assert not re.search(
                    rf"^\s*{key}\s*:", text, re.M)

    def test_factory_behaviour_on_shipped_file_unchanged(
            self) -> None:
        # The shipped values were empty; the migrated
        # file (keys absent) must drive the factory
        # exactly as before: backend "none" -> the
        # factory's own no-backend outcome.
        result = load_backend_from_config(
            REPO_ROOT / "configs" / "user_config.yaml")
        assert result is None or \
            getattr(result, "backend_name", None) is None \
            or getattr(result, "name", "none") in (
                None, "none")


# ── The config validator ─────────────────────────────────────────────────────


class TestConfigValidator:
    def test_raw_value_refused(self) -> None:
        problems = config_mapping_problems(
            {"openai_api_key": "sk-live-value"})
        assert any("raw credential value" in p
                   and "D-CFG-4" in p for p in problems)

    def test_empty_raw_key_is_the_defect_shape(self) -> None:
        problems = config_mapping_problems({"hf_token": ""})
        assert any("shape is the defect" in p
                   for p in problems)

    def test_nested_raw_key_refused(self) -> None:
        problems = config_mapping_problems(
            {"section": {"deep": {"hf_token": "x"}}})
        assert any("hf_token" in p for p in problems)

    def test_migrated_config_accepted(self) -> None:
        assert config_mapping_problems({
            "backend": "none",
            "openai_api_key_ref":
                "kista://thoughtforge/openai_api_key",
            "hf_token_ref": "kista://thoughtforge/hf_token",
        }) == []

    def test_malformed_reference_named(self) -> None:
        problems = config_mapping_problems(
            {"hf_token_ref": "just-the-token-value"})
        assert any("hf_token_ref" in p for p in problems)
        problems = config_mapping_problems(
            {"other": "kista://Bad Service/field"})
        assert problems

    def test_validate_config_envelope(self) -> None:
        custody = KistaCustody()
        ok = custody.validate_config({"backend": "none"})
        assert ok["ok"] is True and ok["status"] == "valid"
        assert custody_response_problems(ok) == []
        bad = custody.validate_config(
            {"openai_api_key": "sk-live"})
        assert bad["ok"] is False
        assert bad["status"] == "refused"
        assert bad["problems"]
        assert custody_response_problems(bad) == []
        assert json.loads(json.dumps(bad)) == bad


# ── Disabled by default ──────────────────────────────────────────────────────


class TestDisabled:
    def test_default_is_disabled(self) -> None:
        assert KistaCustody().enabled is False
        assert KistaCustody(enabled=True).enabled is True
        assert KistaCustody(
            resolver=_FakeResolver()).enabled is False

    def test_resolution_refuses_as_data(self) -> None:
        custody = KistaCustody(resolver=_FakeResolver())
        out = custody.resolve(_REF, consumer=lambda v: v)
        assert out["ok"] is False
        assert out["status"] == "refused"
        assert out["error"] == DISABLED_ERROR
        assert out["contract_version"] == API_VERSION
        assert custody_response_problems(out) == []
        assert json.loads(json.dumps(out)) == out

    def test_disabled_builds_and_touches_nothing(self) -> None:
        built: list[str] = []
        fake = _FakeResolver()

        def factory():
            built.append("resolver")
            return fake

        custody = KistaCustody(resolver_factory=factory)
        custody.resolve(_REF)
        assert built == []
        assert custody._resolver is None
        assert fake.calls == []

    def test_disabled_refuses_garbage_without_raising(
            self) -> None:
        custody = KistaCustody()
        out = custody.resolve("not a reference")
        assert out["ok"] is False
        assert out["error"] == DISABLED_ERROR

    def test_enable_and_disable_are_explicit_acts(self) -> None:
        custody, _fake = _enabled()
        custody.disable()
        assert custody.resolve(_REF)["ok"] is False
        custody.enable()
        assert custody.enabled is True
        assert custody.resolve(_REF)["ok"] is True


# ── Resolution ───────────────────────────────────────────────────────────────


class TestResolution:
    def test_resolve_delivers_only_to_consumer(self) -> None:
        custody, fake = _enabled()
        received: list = []
        out = custody.resolve(
            _REF,
            consumer=lambda v: received.append(v) or len(v),
            binding="backend_factory")
        assert fake.calls == [
            ("thoughtforge", "openai_api_key")]
        assert received == [SENTINEL_SECRET]
        assert out["ok"] is True
        assert out["status"] == "resolved"
        assert out["reference"] == _REF
        assert out["service"] == "thoughtforge"
        assert out["field"] == "openai_api_key"
        assert out["binding"] == "backend_factory"
        assert out["consumer_result"] == len(SENTINEL_SECRET)
        assert "value" not in out and "secret" not in out
        assert custody_response_problems(out) == []

    def test_sentinel_in_no_serialised_output(self) -> None:
        custody, _fake = _enabled()
        out = custody.resolve(_REF, consumer=lambda v: len(v))
        assert SENTINEL_SECRET not in json.dumps(out)
        assert SENTINEL_SECRET not in json.dumps(
            kista_custody_contract())
        assert SENTINEL_SECRET not in json.dumps(
            {k: str(v) for k, v in vars(custody).items()})
        assert SENTINEL_SECRET not in json.dumps(
            kista_custody_as_data())

    def test_resolution_without_consumer_discards(self) -> None:
        custody, fake = _enabled()
        out = custody.resolve(_REF)
        assert out["ok"] is True
        assert out["consumer_result"] is None
        assert fake.calls == [
            ("thoughtforge", "openai_api_key")]

    def test_unconfigured_refuses_as_data(self) -> None:
        custody = KistaCustody(enabled=True)
        out = custody.resolve(_REF)
        assert out["ok"] is False
        assert out["error"] == NOT_CONFIGURED_ERROR
        assert custody_response_problems(out) == []

    def test_resolver_returning_nothing_refuses(self) -> None:
        custody, _fake = _enabled()
        out = custody.resolve("kista://other/service_field")
        assert out["ok"] is False
        assert "returned no value" in out["error"]
        assert custody_response_problems(out) == []

    def test_resolver_error_names_kind_only(self) -> None:
        def boom(service: str, field: str):
            raise RuntimeError(
                f"vault exploded with {SENTINEL_SECRET}")

        custody = KistaCustody(resolver=boom, enabled=True)
        out = custody.resolve(_REF)
        assert out["ok"] is False
        assert "resolver failed" in out["error"]
        assert SENTINEL_SECRET not in json.dumps(out)
        assert "vault exploded" not in json.dumps(out)

    def test_invalid_reference_raises_named(self) -> None:
        custody, _fake = _enabled()
        with pytest.raises(ValueError, match="reference"):
            custody.resolve("sk-live-value")
        with pytest.raises(ValueError, match="reference"):
            custody.resolve("kista://Bad/openai_api_key")

    def test_reference_for_keys(self) -> None:
        custody = KistaCustody()
        assert custody.reference_for("openai_api_key") == \
            "kista://thoughtforge/openai_api_key"
        assert custody.reference_for("hf_token") == \
            "kista://thoughtforge/hf_token"
        assert custody.reference_for("backend") is None

    def test_response_problems_catches_value_field(self) -> None:
        custody, _fake = _enabled()
        out = custody.resolve(_REF, consumer=lambda v: 1)
        assert custody_response_problems(out) == []
        lying = dict(out, value=SENTINEL_SECRET)
        assert any("value" in p
                   for p in custody_response_problems(lying))
        wrong = dict(out, contract_version="9.9.9")
        assert any("contract_version" in p
                   for p in custody_response_problems(wrong))

    def test_factory_is_lazy(self) -> None:
        built: list[str] = []
        fake = _FakeResolver()

        def factory():
            built.append("resolver")
            return fake

        custody = KistaCustody(
            resolver_factory=factory, enabled=True)
        assert built == []
        custody.resolve(_REF)
        assert built == ["resolver"]


# ── State, probes, doc, docket ───────────────────────────────────────────────


class TestStateProbesDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_kista_custody_probes()
        assert len(probes) == len(KISTA_CUSTODY_PROBES) == 7
        assert [p["id"] for p in probes] == [
            "disabled_refusal_touches_nothing",
            "reference_shape_is_kistas_own",
            "resolved_value_never_serialised",
            "configs_on_disk_carry_references_not_values",
            "raw_config_values_refused_by_validator",
            "bindings_need_no_credential_today",
            "doc_matches_render_and_version_is_tf034"]
        assert all(p["passed"] for p in probes)

    def test_doc_matches_render(self) -> None:
        assert _read(KISTA_CUSTODY_DOC) == \
            render_kista_custody_markdown()
        doc = _read(KISTA_CUSTODY_DOC)
        assert "Credentials Live in Custody" in doc
        assert "NOT PROVEN" in doc
        assert "TF-170" in doc and "D-CFG-4" in doc
        assert "kista://thoughtforge/openai_api_key" in doc
        assert "kista-secret-source" in doc
        assert API_VERSION in doc

    def test_docket_records_the_fix(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, " \
            "101 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT" in normalized
        row = next(line for line in docket.splitlines()
                   if line.startswith("| D-CFG-4"))
        assert "FIXED by TF-160" in row
