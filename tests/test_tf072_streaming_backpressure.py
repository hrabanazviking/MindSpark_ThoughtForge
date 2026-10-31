"""
TF-072 — Streaming backpressure: slow consumers don't
corrupt generation; cancellation clean on every backend.

The discipline lives in
``src/thoughtforge/inference/streaming_discipline.py``
(STREAMING_RULES + BACKEND_STREAM_SHAPES as data, format
``thoughtforge-streaming-discipline/1``); the one fix
lives in ``TurboQuantEngine.generate_stream`` (D-STREAM-1:
the llama stream iterator is closed in a finally on every
exit). Write-up: ``docs/architecture/STREAMING_BACKPRESSURE.md``.

Pinned here:

  * the filed estate: format, rule ids, the shape table
    judged against each shipped class's override status
    and TF-065 capability record; both checkers clean,
    filed and live,
  * pull-based production: produced == pulled at every
    pause, at the engine layer and through the backend's
    yield-from,
  * slow-consumer integrity: a stream consumed with
    arbitrary pauses joins byte-identically to a quickly
    consumed one and to the raw completion; a simulated
    backend's single chunk equals its generate() text,
  * free cancel before the first pull: no generate()
    call, no engine load,
  * cancel closes every layer (D-STREAM-1): the model's
    stream iterator records closed on early close, on a
    mid-stream error, and on exhaustion; the backend's
    close reaches the engine generator; a closed stream
    yields nothing further and its yielded prefix is
    untouched,
  * reuse after cancel: the next stream from the same
    instance is whole,
  * the core leg: a consumer closing think_stream
    mid-turn closes the stub engine's generator and
    lands no partial candidate (TF-027's discard rule
    standing on this slice's backend surface).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.inference.streaming_discipline import (
    BACKEND_STREAM_SHAPES,
    STREAM_FORMAT,
    STREAMING_RULES,
    streaming_as_data,
    streaming_live_problems,
    streaming_problems,
)
from thoughtforge.inference.turboquant import TurboQuantEngine
from thoughtforge.inference.turboquant_backend import TurboQuantBackend
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)

TEXTS = [f"delta-{i} " for i in range(6)]
WHOLE = "".join(TEXTS)
REQ = GenerationRequest(prompt="prompt")


# ── Stubs ──────────────────────────────────────────────────────


class ClosableStream:
    """A llama-style stream iterator recording close()."""

    def __init__(self, texts=TEXTS, fail_at=None):
        self._texts = list(texts)
        self._fail_at = fail_at
        self._i = 0
        self.closed = False
        self.produced = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._fail_at is not None and self._i == self._fail_at:
            raise ValueError("stub llama died mid-stream")
        if self._i >= len(self._texts):
            raise StopIteration
        text = self._texts[self._i]
        self._i += 1
        self.produced += 1
        return {"choices": [{"text": text}]}

    def close(self):
        self.closed = True


def engine_with(stream) -> TurboQuantEngine:
    from thoughtforge.inference.profiles import (
        ProfileSelector,
        resolve_generation_params,
    )

    engine = TurboQuantEngine()
    profile = ProfileSelector().load("desktop_cpu")
    engine._params = resolve_generation_params(profile.profile)
    engine._model = lambda prompt, **kwargs: stream
    engine._loaded = True
    return engine


class CountingBackend(UnifiedBackend):
    """On the default (simulated) stream path."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return GenerationResponse(
            text="one whole response", backend_used="counting:test")

    def health_check(self):
        return True

    def backend_name(self):
        return "counting:test"


class RecordingEngine:
    """Stub native engine for the backend layer."""

    def __init__(self, texts=TEXTS):
        self.texts = list(texts)
        self.produced = 0
        self.closed = False

    def generate_stream(self, prompt, params=None):
        try:
            for text in self.texts:
                self.produced += 1
                yield text
        finally:
            self.closed = True


def backend_with(engine) -> TurboQuantBackend:
    backend = TurboQuantBackend(model_path=Path("stub.gguf"))
    backend._engine = engine
    return backend


# ── The filed estate ───────────────────────────────────────────


class TestFiledEstate:
    def test_format_and_data_shape(self):
        data = streaming_as_data()
        assert data["format"] == STREAM_FORMAT
        assert STREAM_FORMAT == "thoughtforge-streaming-discipline/1"
        assert [r["id"] for r in data["rules"]] == [
            r["id"] for r in STREAMING_RULES
        ]
        assert [s["backend"] for s in data["backend_shapes"]] == [
            "turboquant", "ollama", "lmstudio", "huggingface",
        ]

    def test_rule_ids_exact(self):
        assert [r["id"] for r in STREAMING_RULES] == [
            "pull_based_no_run_ahead",
            "slow_consumer_integrity",
            "cancel_before_first_pull_is_free",
            "cancel_closes_every_layer",
            "cancel_leaves_no_partial",
            "reuse_after_cancel",
            "simulated_cancellation_point_is_stated",
        ]
        assert all(r["rule"] and r["measured_by"] for r in STREAMING_RULES)

    def test_shapes_match_capability_records(self):
        from thoughtforge.inference.hf_backend import HuggingFaceBackend
        from thoughtforge.inference.lmstudio_backend import LMStudioBackend
        from thoughtforge.inference.ollama_backend import OllamaBackend

        instances = {
            "turboquant": TurboQuantBackend(model_path=Path("x.gguf")),
            "ollama": OllamaBackend(),
            "lmstudio": LMStudioBackend(),
            "huggingface": HuggingFaceBackend(model="m"),
        }
        for shape in BACKEND_STREAM_SHAPES:
            caps = instances[shape["backend"]].capabilities()
            assert caps.streaming == shape["shape"], shape["backend"]
            assert shape["cancellation_point"]
            assert shape["underlying"]

    def test_checkers_clean(self):
        assert streaming_problems() == []
        assert streaming_live_problems() == []


# ── Backpressure: pull-based, slow-consumer integrity ─────────


class TestBackpressure:
    def test_engine_produces_only_on_pull(self):
        stream = ClosableStream()
        gen = engine_with(stream).generate_stream("prompt")
        assert stream.produced == 0  # creation alone produces nothing
        for pulled in range(1, 4):
            next(gen)
            assert stream.produced == pulled
        gen.close()

    def test_backend_produces_only_on_pull(self):
        engine = RecordingEngine()
        gen = backend_with(engine).generate_stream(REQ)
        assert engine.produced == 0
        next(gen)
        next(gen)
        assert engine.produced == 2
        gen.close()

    def test_slow_consumer_integrity_native(self):
        slow_stream = ClosableStream()
        slow = []
        gen = engine_with(slow_stream).generate_stream("prompt")
        for _ in range(len(TEXTS)):
            slow.append(next(gen))
            # Arbitrary consumer-side work between pulls.
            _ = sum(len(s) for s in slow) * 7
        fast = list(engine_with(ClosableStream()).generate_stream("prompt"))
        assert "".join(slow) == "".join(fast) == WHOLE

    def test_slow_consumer_integrity_simulated(self):
        backend = CountingBackend()
        gen = backend.generate_stream(REQ)
        chunk = next(gen)  # the consumer may now pause indefinitely
        _ = [len(chunk) for _ in range(100)]
        with pytest.raises(StopIteration):
            next(gen)
        assert chunk == backend.generate(REQ).text


# ── Cancellation ───────────────────────────────────────────────


class TestCancelBeforeFirstPull:
    def test_default_stream_runs_no_generate(self):
        backend = CountingBackend()
        gen = backend.generate_stream(REQ)
        gen.close()
        assert backend.calls == 0

    def test_native_backend_loads_no_engine(self):
        backend = TurboQuantBackend(model_path=Path("stub.gguf"))
        gen = backend.generate_stream(REQ)
        gen.close()
        assert backend._engine is None


class TestCancelClosesEveryLayer:
    def test_engine_close_reaches_model_stream(self):
        stream = ClosableStream()
        gen = engine_with(stream).generate_stream("prompt")
        assert next(gen) == TEXTS[0]
        gen.close()
        assert stream.closed is True

    def test_engine_error_closes_model_stream(self):
        stream = ClosableStream(fail_at=2)
        with pytest.raises(RuntimeError):
            list(engine_with(stream).generate_stream("prompt"))
        assert stream.closed is True

    def test_engine_exhaustion_closes_model_stream(self):
        stream = ClosableStream()
        assert "".join(engine_with(stream).generate_stream("prompt")) == WHOLE
        assert stream.closed is True

    def test_closed_stream_yields_nothing_and_prefix_stands(self):
        stream = ClosableStream()
        gen = engine_with(stream).generate_stream("prompt")
        prefix = [next(gen), next(gen)]
        gen.close()
        assert prefix == TEXTS[:2]
        with pytest.raises(StopIteration):
            next(gen)

    def test_backend_close_reaches_engine_generator(self):
        engine = RecordingEngine()
        gen = backend_with(engine).generate_stream(REQ)
        next(gen)
        gen.close()
        assert engine.closed is True


class TestReuseAfterCancel:
    def test_engine_reuse(self):
        engine = engine_with(ClosableStream())
        gen = engine.generate_stream("prompt")
        next(gen)
        gen.close()
        engine._model = lambda prompt, **kwargs: ClosableStream(
            ["again ", "whole"])
        assert "".join(engine.generate_stream("prompt")) == "again whole"

    def test_backend_reuse(self):
        backend = backend_with(RecordingEngine())
        gen = backend.generate_stream(REQ)
        next(gen)
        gen.close()
        backend._engine = RecordingEngine(["second ", "stream"])
        assert "".join(backend.generate_stream(REQ)) == "second stream"


class TestSimulatedCancellationPoint:
    def test_ollama_stream_is_one_completed_chunk(self):
        from thoughtforge.inference.ollama_backend import OllamaBackend

        calls = {"n": 0}

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"content": "ollama whole"},
                        "done": True, "eval_count": 3}

        def fake_post(url, **kwargs):
            calls["n"] += 1
            return Resp()

        backend = OllamaBackend()
        with patch("requests.post", side_effect=fake_post):
            gen = backend.generate_stream(REQ)
            gen.close()  # before the first pull: no HTTP at all
            assert calls["n"] == 0
            chunks = list(backend.generate_stream(REQ))
        assert chunks == ["ollama whole"]
        assert calls["n"] == 1


# ── The core leg ───────────────────────────────────────────────


class TestCoreConsumerClose:
    def test_closing_think_stream_closes_engine_and_lands_no_partial(self):
        tmp = tempfile.mkdtemp()
        core = ThoughtForgeCore(
            memory_dir=Path(tmp) / "memory",
            db_path=Path(tmp) / "nonexistent.db",
            model_path=None,
        )

        class NativeStub:
            def __init__(self):
                self.closed = False

            def generate_stream(self, prompt, params=None):
                try:
                    for i in range(50):
                        yield f"word{i} "
                finally:
                    self.closed = True

            def generate(self, prompt, params=None):  # pragma: no cover
                raise AssertionError("stream path must be used")

        stub = NativeStub()
        core._engine = stub
        events = core.think_stream("What is Yggdrasil?")
        chunks = 0
        for event in events:
            if event.kind == "chunk":
                chunks += 1
                if chunks == 3:
                    break
        events.close()
        assert stub.closed is True
        turn = core.last_turn_state
        assert turn is not None
        assert turn.candidates == []  # the partial was discarded whole
        assert not turn.is_complete
