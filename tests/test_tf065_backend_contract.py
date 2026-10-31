"""TF-065 — backend contract v1.

Pins: the duties + vocabularies as data; the contract
estimate (empty 0, ~4 chars/token, monotone); capability
records judged as data (clean record passes, every
vocabulary breach named); the base class defaults (a
backend declaring nothing gets the conservative record —
simulated unless it overrides generate_stream, estimate
counting, seed not_supported — and the shared count_tokens);
the shipped set (four backends, factory ids, placements
declared, records clean and consistent with their classes);
seed forwarding measured in the actual payloads (ollama
options.seed, lmstudio seed field, HF client kwargs + raw
parameters, TurboQuant GenerationParams) and ABSENT when
the request carries no seed (pre-slice payloads unchanged);
TurboQuant's native stream delegating to a fake engine,
its count_tokens flipping estimate -> model with load
state, and its load failure yielding nothing; error-as-data
on the turboquant missing-model path; and the live
checkers (dependency rulings against pyproject + AST,
factory keys, the whole contract re-derived).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from thoughtforge.inference.backend_contract import (
    BACKEND_CONFIG_KEYS,
    CONTRACT_DUTIES,
    CONTRACT_FORMAT,
    DEPENDENCY_RULINGS,
    FINISH_REASONS,
    OPTIONAL_PACKAGES,
    REQUESTS_CONSUMERS,
    SEED_MODES,
    SHIPPED_BACKENDS,
    STREAMING_MODES,
    TOKEN_COUNTING_MODES,
    BackendCapabilities,
    backend_contract_problems,
    capabilities_problems,
    dependency_problems,
    estimate_tokens,
    factory_problems,
    shipped_backends,
)
from thoughtforge.inference.hf_backend import HuggingFaceBackend
from thoughtforge.inference.lmstudio_backend import LMStudioBackend
from thoughtforge.inference.ollama_backend import OllamaBackend
from thoughtforge.inference.turboquant_backend import TurboQuantBackend
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)


class _Minimal(UnifiedBackend):
    def generate(self, request):
        return GenerationResponse(text="ok", backend_used=self.backend_name())

    def health_check(self):
        return True

    def backend_name(self):
        return "minimal:test"


class _NativeMinimal(_Minimal):
    def generate_stream(self, request):
        yield "o"
        yield "k"


def _ok_response(payload):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = payload
    return mock


def _capture_post(payload):
    captured: dict = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["payload"] = json
        return _ok_response(payload)

    return captured, fake_post


# ── The contract as data ──────────────────────────────────────────────────────

class TestContractData:
    def test_duties_cover_the_roadmap_names(self):
        duties = {d["duty"] for d in CONTRACT_DUTIES}
        assert duties == {
            "generate", "generate_stream", "count_tokens", "capabilities",
            "seed", "health_check", "backend_name", "finish_reason",
        }

    def test_required_duties(self):
        required = {d["duty"] for d in CONTRACT_DUTIES if d["required"]}
        assert required == {
            "generate", "generate_stream", "count_tokens", "capabilities",
            "health_check", "backend_name", "finish_reason",
        }

    def test_seed_duty_names_tf036(self):
        seed = next(d for d in CONTRACT_DUTIES if d["duty"] == "seed")
        assert seed["required"] is False
        assert "TF-036" in seed["rule"]

    def test_vocabularies(self):
        assert STREAMING_MODES == ("native", "simulated", "none")
        assert TOKEN_COUNTING_MODES == ("model", "estimate", "none")
        assert SEED_MODES == ("honoured", "not_supported")
        assert FINISH_REASONS == ("stop", "length", "error")
        assert CONTRACT_FORMAT == "thoughtforge-backend-contract/1"

    def test_shipped_set_is_the_factory_set(self):
        assert [r["id"] for r in SHIPPED_BACKENDS] == [
            "ollama", "lmstudio", "huggingface", "turboquant"]

    def test_estimate_tokens(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 40) == 10
        text = "the ferry leaves at dawn"
        assert 0 < estimate_tokens(text) <= estimate_tokens(text * 4)


# ── Capability records as data ────────────────────────────────────────────────

class TestCapabilityRecords:
    def test_clean_record_passes(self):
        caps = BackendCapabilities(name="x:y").to_dict()
        assert caps["format"] == CONTRACT_FORMAT
        assert capabilities_problems(caps) == []

    def test_every_breach_named(self):
        caps = BackendCapabilities(name="x:y").to_dict()
        caps.update({"format": "nope", "name": "", "generate": False,
                     "health": False, "streaming": "fast",
                     "token_counting": "exact", "seed": "maybe",
                     "placement": "cloud"})
        problems = capabilities_problems(caps)
        assert len(problems) == 8

    def test_default_record_for_undeclaring_backend(self):
        caps = _Minimal().capabilities()
        assert caps.name == "minimal:test"
        assert caps.streaming == "simulated"
        assert caps.token_counting == "estimate"
        assert caps.seed == "not_supported"
        assert capabilities_problems(caps.to_dict()) == []

    def test_default_record_detects_native_override(self):
        assert _NativeMinimal().capabilities().streaming == "native"

    def test_base_count_tokens_is_the_contract_estimate(self):
        backend = _Minimal()
        assert backend.count_tokens("") == 0
        assert backend.count_tokens("the ferry leaves at dawn") == \
            estimate_tokens("the ferry leaves at dawn")


# ── The shipped backends' records ─────────────────────────────────────────────

class TestShippedRecords:
    def test_four_backends_instantiate(self):
        backends = shipped_backends()
        assert len(backends) == 4
        assert all(isinstance(b, UnifiedBackend) for b in backends)

    def test_ollama_record(self):
        caps = OllamaBackend(model="m1").capabilities()
        assert (caps.streaming, caps.token_counting, caps.seed) == \
            ("simulated", "estimate", "honoured")
        assert caps.placement == "local_or_remote"
        assert caps.name == "ollama:m1"

    def test_lmstudio_record(self):
        caps = LMStudioBackend(model="m2").capabilities()
        assert (caps.streaming, caps.token_counting, caps.seed) == \
            ("simulated", "estimate", "honoured")
        assert caps.placement == "local_or_remote"

    def test_hf_record_is_remote_and_honest_about_tokens(self):
        caps = HuggingFaceBackend(model="m3").capabilities()
        assert caps.placement == "remote"
        assert caps.token_counting == "estimate"
        # TF-071 amended this pin in scope (D-TOK-3): the note
        # no longer declares a word-split approximation —
        # tokens_generated now IS the contract estimate.
        assert any("contract estimate" in note for note in caps.notes)

    def test_turboquant_record_before_load(self):
        caps = TurboQuantBackend(
            model_path=Path("/nonexistent/x.gguf")).capabilities()
        assert caps.streaming == "native"
        assert caps.token_counting == "estimate"  # never claimed early
        assert caps.seed == "honoured"
        assert caps.placement == "local"

    def test_all_records_clean(self):
        for backend in shipped_backends():
            assert capabilities_problems(backend.capabilities().to_dict()) == []


# ── Seed forwarding, measured in payloads ─────────────────────────────────────

class TestSeedForwarding:
    def test_ollama_seed_in_options(self):
        captured, fake = _capture_post(
            {"message": {"content": "ok"}, "done": True, "eval_count": 1})
        with patch("requests.post", side_effect=fake):
            OllamaBackend().generate(GenerationRequest(prompt="hi", seed=42))
        assert captured["payload"]["options"]["seed"] == 42

    def test_ollama_no_seed_no_key(self):
        captured, fake = _capture_post(
            {"message": {"content": "ok"}, "done": True, "eval_count": 1})
        with patch("requests.post", side_effect=fake):
            OllamaBackend().generate(GenerationRequest(prompt="hi"))
        assert "seed" not in captured["payload"]["options"]

    def test_lmstudio_seed_field(self):
        captured, fake = _capture_post(
            {"choices": [{"message": {"content": "ok"},
                          "finish_reason": "stop"}],
             "usage": {"completion_tokens": 1}})
        with patch("requests.post", side_effect=fake):
            LMStudioBackend().generate(GenerationRequest(prompt="hi", seed=7))
        assert captured["payload"]["seed"] == 7

    def test_lmstudio_no_seed_no_key(self):
        captured, fake = _capture_post(
            {"choices": [{"message": {"content": "ok"},
                          "finish_reason": "stop"}],
             "usage": {"completion_tokens": 1}})
        with patch("requests.post", side_effect=fake):
            LMStudioBackend().generate(GenerationRequest(prompt="hi"))
        assert "seed" not in captured["payload"]

    def test_hf_raw_leg_seed_in_parameters(self):
        backend = HuggingFaceBackend(model="m")
        backend._client = None
        captured, fake = _capture_post([{"generated_text": "ok"}])
        with patch("requests.post", side_effect=fake):
            backend.generate(GenerationRequest(prompt="hi", seed=11))
        assert captured["payload"]["parameters"]["seed"] == 11

    def test_hf_client_leg_seed_kwarg(self):
        backend = HuggingFaceBackend(model="m")
        calls: dict = {}

        class FakeClient:
            def text_generation(self, prompt, **kwargs):
                calls.update(kwargs)
                return "ok"

        backend._client = FakeClient()
        backend.generate(GenerationRequest(prompt="hi", seed=13))
        assert calls["seed"] == 13

    def test_hf_client_leg_no_seed_no_kwarg(self):
        backend = HuggingFaceBackend(model="m")
        calls: dict = {}

        class FakeClient:
            def text_generation(self, prompt, **kwargs):
                calls.update(kwargs)
                return "ok"

        backend._client = FakeClient()
        backend.generate(GenerationRequest(prompt="hi"))
        assert "seed" not in calls


# ── TurboQuant conformance ────────────────────────────────────────────────────

class _FakeEngine:
    def __init__(self):
        self.params = None
        self.prompt = None

    def generate(self, prompt, params=None):
        self.prompt, self.params = prompt, params
        result = MagicMock()
        result.text = "engine text"
        result.completion_tokens = 2
        result.finish_reason = "stop"
        return result

    def generate_stream(self, prompt, params=None):
        self.prompt, self.params = prompt, params
        yield "eng"
        yield "ine"

    def estimate_tokens(self, text):
        return 99 if text else 0


class TestTurboQuantConformance:
    def _backend(self):
        backend = TurboQuantBackend(model_path=Path("/nonexistent/x.gguf"))
        backend._engine = _FakeEngine()
        return backend

    def test_seed_reaches_engine_params(self):
        backend = self._backend()
        resp = backend.generate(GenerationRequest(prompt="hi", seed=5))
        assert resp.text == "engine text"
        assert backend._engine.params.seed == 5

    def test_native_stream_delegates(self):
        backend = self._backend()
        chunks = list(backend.generate_stream(
            GenerationRequest(prompt="hi", seed=5)))
        assert chunks == ["eng", "ine"]
        assert backend._engine.params.seed == 5
        # the core labels this native by its override rule
        assert type(backend).generate_stream is not \
            UnifiedBackend.generate_stream

    def test_count_tokens_uses_engine_when_loaded(self):
        backend = self._backend()
        assert backend.count_tokens("anything") == 99
        assert backend.count_tokens("") == 0
        assert backend.capabilities().token_counting == "model"

    def test_missing_model_generate_is_error_data(self):
        backend = TurboQuantBackend(model_path=Path("/nonexistent/x.gguf"))
        resp = backend.generate(GenerationRequest(prompt="hi"))
        assert resp.finish_reason == "error"
        assert resp.error != ""
        assert resp.backend_used == backend.backend_name()

    def test_missing_model_stream_yields_nothing(self):
        backend = TurboQuantBackend(model_path=Path("/nonexistent/x.gguf"))
        assert list(backend.generate_stream(GenerationRequest(prompt="hi"))) == []


# ── Dependencies + factory + the live checker ─────────────────────────────────

class TestDependenciesAndFactory:
    def test_dependency_rulings(self):
        rulings = {r["package"]: r for r in DEPENDENCY_RULINGS}
        assert rulings["requests"]["ruling"] == "core"
        assert set(OPTIONAL_PACKAGES) == {
            "huggingface_hub", "transformers", "onnxruntime",
            "optimum", "psutil"}
        assert len(REQUESTS_CONSUMERS) == 4

    def test_dependency_problems_clean(self):
        assert dependency_problems() == []

    def test_factory_keys(self):
        assert factory_problems() == []
        assert "openai_api_key" in BACKEND_CONFIG_KEYS
        assert "profile" not in BACKEND_CONFIG_KEYS

    def test_live_checker_clean(self):
        assert backend_contract_problems() == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
