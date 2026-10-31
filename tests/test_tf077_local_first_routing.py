"""
TF-077 — Local-first routing: profile-aware backend
choice (privacy + cost + capability), the router's table
published.

The router lives in
``src/thoughtforge/inference/local_first_routing.py``
(``BACKEND_ROSTER`` / ``ROUTING_TABLE`` / ``ROUTING_RULES``
as data, ``route()`` / ``build_chain()`` /
``model_available()``, format
``thoughtforge-local-first-routing/1``).
Write-up: ``docs/architecture/LOCAL_FIRST_ROUTING.md``.

Measured start: no routing module existed in inference
(the tree's only router routes intent); the factory loads
one backend by config key alone (no profile / privacy /
cost in its source); the profiles' inference_backend
field names an engine ("cpu" / "cuda"), not a backend
family; TF-065 placements existed unconsumed; TF-073's
chain waited for an author of its order.

D-HEALTH-2 FIXED in scope at routing grade: ollama /
lmstudio / aesir gained tri-state ``model_listed()``
(mirroring TF-076's surface — their list methods return
[] alike for down and absent), and the router's
``model_available()`` consumes every family's per-model
surface; health_check() semantics themselves unchanged.

Pinned here:

  * the filed estate: roster/table/rule counts, both
    checkers clean, the table judged against the live
    profile JSONs (hosted last everywhere, constrained
    profiles lead in-process, GPU profiles lead aesir),
  * classification: family from the real classes' names,
    placement resolved per instance (loopback vs off-box
    for local_or_remote; unresolvable counts remote),
  * routing: local-first absolute across tiers, remote
    excluded by default with reason, admitted last when
    allowed, privacy/cost classes resolved per entry,
  * the gates: unknown family and generate=False
    excluded with reasons; the model gate excludes a
    False, keeps + annotates a None,
  * the new model_listed() legs over mocked transports
    (listed / absent / down / unconfigured, ollama's
    implied :latest),
  * model_available() per surface (file existence for
    turboquant, per-model health for huggingface),
  * the chain: build_chain's order is the decision's,
    and it serves through TF-073's wrapper; an empty
    decision refuses to chain; an unknown profile raises.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from thoughtforge.inference.aesir_backend import AesirBackend
from thoughtforge.inference.backend_fallback import FallbackBackend
from thoughtforge.inference.hf_backend import HuggingFaceBackend
from thoughtforge.inference.lmstudio_backend import LMStudioBackend
from thoughtforge.inference.local_first_routing import (
    BACKEND_ROSTER,
    ROUTING_RULES,
    ROUTING_TABLE,
    RoutingDecision,
    build_chain,
    family_of,
    is_loopback_url,
    model_available,
    resolved_placement,
    route,
    routing_as_data,
    routing_live_problems,
    routing_problems,
)
from thoughtforge.inference.ollama_backend import OllamaBackend
from thoughtforge.inference.openai_compatible import (
    OpenAICompatibleBackend,
)
from thoughtforge.inference.turboquant_backend import TurboQuantBackend
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    UnifiedBackend,
)

OLLAMA_GET = "thoughtforge.inference.ollama_backend.requests.get"
LM_GET = "thoughtforge.inference.lmstudio_backend.requests.get"
AESIR_GET = "thoughtforge.inference.aesir_backend.requests.get"


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def real_set(tmp_path=None):
    return [
        HuggingFaceBackend(),
        OpenAICompatibleBackend(),
        LMStudioBackend(),
        OllamaBackend(),
        AesirBackend(),
        TurboQuantBackend(model_path=Path("/nonexistent/x.gguf")),
    ]


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert routing_problems() == []
        assert routing_live_problems() == []

    def test_as_data_shape(self):
        data = routing_as_data()
        assert data["format"] == "thoughtforge-local-first-routing/1"
        assert len(data["roster"]) == len(BACKEND_ROSTER) == 6
        assert len(data["table"]) == len(ROUTING_TABLE) == 6
        assert len(data["rules"]) == len(ROUTING_RULES) == 9
        assert data["privacy_classes"] == [
            "data_stays_on_machine", "data_leaves_machine"]

    def test_every_table_row_is_every_family_hosted_last(self):
        families = {r["family"] for r in BACKEND_ROSTER}
        for row in ROUTING_TABLE:
            assert set(row["order"]) == families
            assert len(row["order"]) == 6
            assert row["order"][-1] == "huggingface"


# ── Classification ─────────────────────────────────────────────


class TestClassification:
    def test_family_of_real_instances(self):
        expected = {
            "turboquant", "aesir", "ollama", "lmstudio",
            "openai_compatible", "huggingface"}
        assert {family_of(b) for b in real_set()} == expected

    @pytest.mark.parametrize("url,local", [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:18434", True),
        ("http://[::1]:1234", True),
        ("https://models.example.org:11434", False),
        ("http://192.168.1.20:11434", False),
        ("", False),
        (None, False),
    ])
    def test_loopback_urls(self, url, local):
        assert is_loopback_url(url) is local

    def test_resolved_placement_per_instance(self):
        assert resolved_placement(OllamaBackend()) == "local"
        assert resolved_placement(OllamaBackend(
            base_url="https://models.example.org:11434")) == "remote"
        assert resolved_placement(HuggingFaceBackend()) == "remote"
        assert resolved_placement(AesirBackend()) == "local"
        assert resolved_placement(TurboQuantBackend(
            model_path=Path("/nonexistent/x.gguf"))) == "local"


# ── Routing over the real classes (classification only) ───────


class TestRealRouting:
    def test_desktop_cpu_default_route(self):
        decision = route("desktop_cpu", real_set())
        assert [e.family for e in decision.ordered] == [
            "turboquant", "aesir", "ollama", "lmstudio",
            "openai_compatible"]
        assert [e.reason for e in decision.excluded] == [
            "remote_not_allowed"]
        assert decision.excluded[0].name.startswith("huggingface:")
        assert decision.chosen.family == "turboquant"
        for entry in decision.ordered:
            assert entry.privacy == "data_stays_on_machine"
            assert entry.cost == "local_compute"
            assert entry.model_listed is None  # not probed, stated

    def test_gpu_profiles_lead_with_aesir(self):
        for profile in ("desktop_gpu", "server_gpu"):
            decision = route(profile, real_set())
            assert decision.chosen.family == "aesir"

    def test_constrained_profiles_lead_in_process(self):
        for profile in ("pi_zero", "phone_low", "pi_5"):
            decision = route(profile, real_set())
            assert decision.chosen.family == "turboquant"

    def test_remote_admitted_last_with_classes(self):
        decision = route("desktop_cpu", real_set(), allow_remote=True)
        last = decision.ordered[-1]
        assert last.family == "huggingface"
        assert last.privacy == "data_leaves_machine"
        assert last.cost == "hosted_metered"

    def test_remote_configured_server_falls_behind_locals(self):
        remote_ollama = OllamaBackend(
            base_url="https://models.example.org:11434")
        decision = route(
            "desktop_cpu",
            [remote_ollama, LMStudioBackend()],
            allow_remote=True)
        assert [e.family for e in decision.ordered] == [
            "lmstudio", "ollama"]
        assert decision.ordered[-1].cost == \
            "remote_declared_unknown"

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            route("no_such_profile", real_set())


# ── The new model_listed() legs (D-HEALTH-2) ───────────────────


class TestModelListedLegs:
    def test_ollama_listed_absent_and_latest(self):
        backend = OllamaBackend(model="llama3.2:3b")
        payload = {"models": [{"name": "llama3.2:3b"},
                              {"name": "other:1b"}]}
        with patch(OLLAMA_GET, return_value=_Resp(200, payload)):
            assert backend.model_listed() is True
        with patch(OLLAMA_GET, return_value=_Resp(
                200, {"models": [{"name": "other:1b"}]})):
            assert backend.model_listed() is False
        latest = OllamaBackend(model="llama3.2")
        with patch(OLLAMA_GET, return_value=_Resp(
                200, {"models": [{"name": "llama3.2:latest"}]})):
            assert latest.model_listed() is True

    def test_ollama_down_or_error_is_unknown(self):
        backend = OllamaBackend()
        with patch(OLLAMA_GET, return_value=_Resp(500, {})):
            assert backend.model_listed() is None
        with patch(OLLAMA_GET, side_effect=ConnectionError("down")):
            assert backend.model_listed() is None

    def test_lmstudio_listed_absent_unconfigured(self):
        backend = LMStudioBackend(model="m-1")
        payload = {"data": [{"id": "m-1"}, {"id": "m-2"}]}
        with patch(LM_GET, return_value=_Resp(200, payload)):
            assert backend.model_listed() is True
        with patch(LM_GET, return_value=_Resp(
                200, {"data": [{"id": "m-2"}]})):
            assert backend.model_listed() is False
        assert LMStudioBackend().model_listed() is None
        with patch(LM_GET, side_effect=ConnectionError("down")):
            assert backend.model_listed() is None

    def test_aesir_catalog_listed_absent_unconfigured(self):
        backend = AesirBackend(model="res-1")
        payload = {"data": [{"id": "res-1"}]}
        with patch(AESIR_GET, return_value=_Resp(200, payload)):
            assert backend.model_listed() is True
        with patch(AESIR_GET, return_value=_Resp(
                200, {"data": []})):
            assert backend.model_listed() is False
        assert AesirBackend().model_listed() is None
        with patch(AESIR_GET, return_value=_Resp(503, {})):
            assert backend.model_listed() is None


# ── model_available() per surface ──────────────────────────────


class TestModelAvailable:
    def test_turboquant_file_existence(self, tmp_path):
        present = tmp_path / "m.gguf"
        present.write_bytes(b"GGUF")
        assert model_available(
            TurboQuantBackend(model_path=present)) is True
        assert model_available(TurboQuantBackend(
            model_path=tmp_path / "absent.gguf")) is False

    def test_huggingface_per_model_health(self):
        backend = HuggingFaceBackend()
        with patch.object(
                HuggingFaceBackend, "health_check",
                return_value=True):
            assert model_available(backend) is True
        with patch.object(
                HuggingFaceBackend, "health_check",
                return_value=False):
            assert model_available(backend) is False

class TestModelGateMocked:
    def test_gate_over_mocked_transports(self):
        ollama = OllamaBackend(model="ghost:1b")
        lm = LMStudioBackend(model="m-1")
        with patch(OLLAMA_GET, return_value=_Resp(
                200, {"models": [{"name": "other:1b"}]})), \
                patch(LM_GET, return_value=_Resp(
                    200, {"data": [{"id": "m-1"}]})):
            decision = route(
                "desktop_cpu", [ollama, lm],
                require_model_listed=True)
        assert [e.family for e in decision.ordered] == ["lmstudio"]
        assert decision.ordered[0].model_listed is True
        assert [e.reason for e in decision.excluded] == [
            "model_not_listed"]

    def test_unknown_is_kept_and_annotated(self):
        lm = LMStudioBackend(model="m-1")
        with patch(LM_GET, side_effect=ConnectionError("down")):
            decision = route(
                "desktop_cpu", [lm], require_model_listed=True)
        assert [e.family for e in decision.ordered] == ["lmstudio"]
        assert decision.ordered[0].model_listed is None
        assert decision.excluded == ()


# ── Gates + the chain ──────────────────────────────────────────


class _Stub(UnifiedBackend):
    def __init__(self, name, *, placement="local", generate=True):
        self._name = name
        self._placement = placement
        self._generate = generate

    def backend_name(self):
        return self._name

    def generate(self, request):
        from thoughtforge.inference.unified_backend import (
            GenerationResponse,
        )

        return GenerationResponse(
            text=f"text from {self._name}",
            backend_used=self._name)

    def health_check(self):
        return True

    def capabilities(self):
        from thoughtforge.inference.backend_contract import (
            BackendCapabilities,
        )

        return BackendCapabilities(
            name=self._name, generate=self._generate,
            placement=self._placement)


class TestGatesAndChain:
    def test_unknown_family_and_no_generate_excluded(self):
        decision = route("pi_5", [
            _Stub("mystery:1"),
            _Stub("aesir:nogen", generate=False),
            _Stub("turboquant:ok.gguf"),
        ])
        reasons = {e.name: e.reason for e in decision.excluded}
        assert reasons == {
            "mystery:1": "unknown_family",
            "aesir:nogen": "generate_not_offered",
        }
        assert [e.name for e in decision.ordered] == [
            "turboquant:ok.gguf"]

    def test_build_chain_serves_the_decision_order(self):
        decision = route("desktop_cpu", [
            _Stub("ollama:a"),
            _Stub("turboquant:b.gguf"),
        ])
        chain = build_chain(decision)
        assert isinstance(chain, FallbackBackend)
        assert chain.chain_names() == [
            "turboquant:b.gguf", "ollama:a"]
        resp = chain.generate(GenerationRequest(prompt="hi"))
        assert resp.text == "text from turboquant:b.gguf"
        assert chain.last_report.served_by == "turboquant:b.gguf"

    def test_empty_decision_refuses_to_chain(self):
        with pytest.raises(ValueError):
            build_chain(RoutingDecision(profile="pi_5"))

    def test_decision_as_data(self):
        decision = route("pi_5", [_Stub("turboquant:b.gguf")])
        data = decision.to_dict()
        assert data["format"] == \
            "thoughtforge-local-first-routing/1"
        assert data["chosen"] == "turboquant:b.gguf"
        assert data["ordered"][0]["privacy"] == \
            "data_stays_on_machine"
