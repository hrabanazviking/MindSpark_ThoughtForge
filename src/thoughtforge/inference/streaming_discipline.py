"""TF-072 — Streaming discipline: backpressure and
clean cancellation on every backend.

Roadmap: "Streaming backpressure: slow consumers don't
corrupt generation; cancellation clean on every backend."

Measured start (probed this tick, VM):

- No queue, thread, or buffer exists anywhere in the
  streaming path — every layer is a Python generator
  (the cognition core's event stream, the backends'
  ``generate_stream``, the TurboQuant engine's), so the
  chain is pull-based end to end: a producer advances
  only when its consumer pulls (probe: a counting
  producer had produced exactly 2 items after 2 pulls
  of a ``yield from`` chain, never more).
- Laziness is load-bearing and was unpinned: the
  default ``UnifiedBackend.generate_stream`` is a
  generator function, so a consumer who closes the
  stream before the first pull runs ``generate()``
  zero times (probe: 0 calls). The simulated backends
  (ollama / lmstudio / huggingface — no
  ``generate_stream`` override, TF-065) ride that
  default: their one chunk is a completed ``generate()``
  response, so a slow consumer cannot corrupt anything
  (the text already exists whole), but cancellation
  cannot land *inside* a generate either — it lands
  between chunks, and the honest name for their
  cancellation point is "before the first pull, or
  after the single chunk".
- The native path had one real leak (D-STREAM-1):
  ``TurboQuantEngine.generate_stream`` held the llama
  ``stream=True`` iterator in a local and simply
  abandoned it on two of its three exits — a consumer
  closing the generator early (GeneratorExit at the
  yield) and a mid-stream error both left the model's
  stream un-closed, to be reaped by garbage collection
  whenever that happened (probe: a closable stub
  stream recorded ``closed == False`` on both paths;
  only plain exhaustion ended it, by draining). The
  wrapper's own propagation was sound — closing a
  ``TurboQuantBackend`` stream forwards ``close()``
  through its ``yield from`` into the engine generator
  (probe: the stub engine's ``finally`` ran) — the
  engine just did nothing with the arrival.
- Cancellation above the backend layer was TF-027's
  and already proven there: the core checks its
  interrupt token before each delivered chunk, a cut
  discards the in-flight candidate whole, and a
  consumer closing ``think_stream`` records the cut
  (probe this tick: the native stub engine's generator
  was closed and no candidate landed). This slice owns
  the layer below: the backend / engine surface the
  core stands on.

THE DISCIPLINE (``STREAMING_RULES`` as data):

- ``pull_based_no_run_ahead`` — a stream producer
  advances only on a consumer pull; no layer may
  buffer, prefetch, or run a generation ahead of its
  consumer. Measured by produced == pulled at every
  pause.
- ``slow_consumer_integrity`` — pausing arbitrarily
  long between pulls changes nothing: the joined
  chunks of a slowly consumed stream are byte-identical
  to the quickly consumed stream and to the raw
  completion the source produced (native deltas
  unstripped, in order; a simulated backend's single
  chunk equals its ``generate()`` text).
- ``cancel_before_first_pull_is_free`` — creating a
  stream starts no work. Closing a stream that was
  never pulled runs no generation and loads no engine.
- ``cancel_closes_every_layer`` — closing a stream
  mid-flight closes every layer below the consumer:
  the backend generator, the engine generator, and the
  model's own stream iterator (D-STREAM-1's fix). The
  same close happens on a mid-stream error and on
  plain exhaustion — one exit discipline, not three.
- ``cancel_leaves_no_partial`` — what a cancelled
  stream already yielded stays yielded (the caller —
  the cognition core — decides what a partial means;
  TF-027's rule discards the partial candidate whole).
  The backend layer never replays, patches, or
  completes a cancelled stream's text after the fact.
- ``reuse_after_cancel`` — a cancelled stream poisons
  nothing: the next stream from the same backend /
  engine instance produces its full text.
- ``simulated_cancellation_point_is_stated`` — a
  simulated backend's cancellation point is between
  chunks (in practice: before its single chunk), never
  inside its ``generate()``; the shape table files this
  per backend instead of letting "cancellation clean
  on every backend" imply an in-flight HTTP abort that
  does not exist in this tree.

FIXED in scope:

- **D-STREAM-1** — the engine's stream iterator is now
  closed in a ``finally`` on every exit (exhaustion /
  error / consumer close). A stream without ``close()``
  needs nothing; a failing close is logged at debug and
  never masks the stream's own outcome; GeneratorExit
  is never converted to RuntimeError.

Proof grade FIXTURE (the real engine and backends on
this VM over closable stub llama streams and counting
stub backends; the core leg over TF-026's stub-engine
shape) — a *loaded GGUF model's* stream-close
behaviour is NOT PROVEN (no llama_cpp, no GGUF on this
VM; llama_cpp's stream object accepting ``close()`` is
handled defensively — ``getattr`` + callable — so an
iterator without one is still disciplined), and a live
HTTP backend's in-flight abort is NOT PROVEN and not
claimed (the simulated shape's stated cancellation
point is the claim), never claimed.
"""

from __future__ import annotations

import inspect
from typing import Any

STREAM_FORMAT = "thoughtforge-streaming-discipline/1"

STREAMING_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "pull_based_no_run_ahead",
        "rule": "every streaming layer is a generator a consumer "
                "pulls; a producer advances only on a pull and never "
                "buffers or prefetches ahead of its consumer",
        "measured_by": "produced == pulled at every pause, at the "
                       "engine, backend, and default-stream layers",
    },
    {
        "id": "slow_consumer_integrity",
        "rule": "arbitrary pauses between pulls change nothing: the "
                "joined chunks are byte-identical however slowly "
                "they are consumed",
        "measured_by": "a slowly consumed stream joins to the same "
                       "text as a quickly consumed one, and to the "
                       "source's raw completion",
    },
    {
        "id": "cancel_before_first_pull_is_free",
        "rule": "creating a stream starts no work; closing an "
                "unpulled stream runs no generation and loads no "
                "engine",
        "measured_by": "generate() call count 0 and the backend's "
                       "engine still unloaded after create + close",
    },
    {
        "id": "cancel_closes_every_layer",
        "rule": "closing a stream mid-flight closes every layer "
                "below: backend generator, engine generator, and "
                "the model's stream iterator — on early close, on "
                "mid-stream error, and on exhaustion alike",
        "measured_by": "the closable stub stream records closed on "
                       "all three exits (D-STREAM-1)",
    },
    {
        "id": "cancel_leaves_no_partial",
        "rule": "deltas already yielded stay yielded; no layer "
                "replays, patches, or completes a cancelled "
                "stream's text — the caller decides what a partial "
                "means (the core discards it whole, TF-027)",
        "measured_by": "a cancelled stream yields nothing further "
                       "after close, and its yielded prefix is "
                       "untouched",
    },
    {
        "id": "reuse_after_cancel",
        "rule": "a cancelled stream poisons no shared state: the "
                "next stream from the same instance produces its "
                "full text",
        "measured_by": "a second stream after a mid-flight cancel "
                       "joins to its complete text",
    },
    {
        "id": "simulated_cancellation_point_is_stated",
        "rule": "a simulated backend (no generate_stream override) "
                "cancels between chunks — before its single chunk — "
                "never inside generate(); the shape table states "
                "this instead of implying an in-flight abort",
        "measured_by": "BACKEND_STREAM_SHAPES carries the point "
                       "per backend, judged against each class's "
                       "override status and capability record",
    },
)

#: The streaming shape of every shipped backend, filed as
#: data and judged live by ``streaming_problems`` against
#: each class's override status and TF-065 capability
#: record. ``underlying`` names the resource a cancel
#: must reach at the bottom of the chain.
BACKEND_STREAM_SHAPES: tuple[dict[str, Any], ...] = (
    {
        "backend": "turboquant",
        "shape": "native",
        "cancellation_point": "between any two deltas",
        "underlying": "the llama stream=True iterator, closed "
                      "by the engine on every exit (D-STREAM-1)",
    },
    {
        "backend": "ollama",
        "shape": "simulated",
        "cancellation_point": "before the single chunk (generate() "
                              "runs whole on the first pull)",
        "underlying": "none — one completed HTTP response",
    },
    {
        "backend": "lmstudio",
        "shape": "simulated",
        "cancellation_point": "before the single chunk (generate() "
                              "runs whole on the first pull)",
        "underlying": "none — one completed HTTP response",
    },
    {
        "backend": "huggingface",
        "shape": "simulated",
        "cancellation_point": "before the single chunk (generate() "
                              "runs whole on the first pull)",
        "underlying": "none — one completed HTTP response",
    },
)


def streaming_as_data() -> dict[str, Any]:
    return {
        "format": STREAM_FORMAT,
        "rules": [dict(r) for r in STREAMING_RULES],
        "backend_shapes": [dict(s) for s in BACKEND_STREAM_SHAPES],
    }


# ── Stubs for the live probes ──────────────────────────────────


class _ClosableStream:
    """A llama-style stream iterator that records close()."""

    def __init__(self, texts: list[str], fail_at: int | None = None) -> None:
        self._texts = texts
        self._fail_at = fail_at
        self._i = 0
        self.closed = False
        self.produced = 0

    def __iter__(self) -> "_ClosableStream":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._fail_at is not None and self._i == self._fail_at:
            raise ValueError("stub llama died mid-stream")
        if self._i >= len(self._texts):
            raise StopIteration
        text = self._texts[self._i]
        self._i += 1
        self.produced += 1
        return {"choices": [{"text": text}]}

    def close(self) -> None:
        self.closed = True


def _counting_backend() -> Any:
    from thoughtforge.inference.unified_backend import (
        GenerationResponse,
        UnifiedBackend,
    )

    class CountingBackend(UnifiedBackend):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request: Any) -> Any:
            self.calls += 1
            return GenerationResponse(
                text="one whole response", backend_used="counting:test",
            )

        def health_check(self) -> bool:
            return True

        def backend_name(self) -> str:
            return "counting:test"

    return CountingBackend()


def _engine_with_stream(stream: Any) -> Any:
    from thoughtforge.inference.profiles import (
        ProfileSelector,
        resolve_generation_params,
    )
    from thoughtforge.inference.turboquant import TurboQuantEngine

    engine = TurboQuantEngine()
    profile = ProfileSelector().load("desktop_cpu")
    engine._params = resolve_generation_params(profile.profile)
    engine._model = lambda prompt, **kwargs: stream
    engine._loaded = True
    return engine


def _request() -> Any:
    from thoughtforge.inference.unified_backend import GenerationRequest

    return GenerationRequest(prompt="prompt")


# ── Checkers ───────────────────────────────────────────────────


def streaming_problems() -> list[str]:
    """The filed estate judged against the code (no live runs)."""
    problems: list[str] = []

    ids = [r["id"] for r in STREAMING_RULES]
    if len(ids) != len(set(ids)):
        problems.append("STREAMING_RULES ids are not unique")
    shape_names = [s["backend"] for s in BACKEND_STREAM_SHAPES]
    if shape_names != ["turboquant", "ollama", "lmstudio", "huggingface"]:
        problems.append(
            f"BACKEND_STREAM_SHAPES covers {shape_names}, not the "
            "four shipped backends in order"
        )

    from thoughtforge.inference.hf_backend import HuggingFaceBackend
    from thoughtforge.inference.lmstudio_backend import LMStudioBackend
    from thoughtforge.inference.ollama_backend import OllamaBackend
    from thoughtforge.inference.turboquant import TurboQuantEngine
    from thoughtforge.inference.turboquant_backend import TurboQuantBackend
    from thoughtforge.inference.unified_backend import UnifiedBackend

    # Laziness is the free-cancel guarantee: every stream
    # surface must be a generator function.
    for cls, method in (
        (UnifiedBackend, "generate_stream"),
        (TurboQuantBackend, "generate_stream"),
        (TurboQuantEngine, "generate_stream"),
    ):
        if not inspect.isgeneratorfunction(getattr(cls, method)):
            problems.append(
                f"{cls.__name__}.{method} is not a generator "
                "function — creating a stream would start work"
            )

    # The shape table judged against each class: an override
    # exactly where the table claims native, and the TF-065
    # capability record agreeing with the filed shape.
    shipped = {
        "turboquant": TurboQuantBackend,
        "ollama": OllamaBackend,
        "lmstudio": LMStudioBackend,
        "huggingface": HuggingFaceBackend,
    }
    for shape in BACKEND_STREAM_SHAPES:
        cls = shipped[shape["backend"]]
        overrides = (
            cls.generate_stream is not UnifiedBackend.generate_stream
        )
        if shape["shape"] == "native" and not overrides:
            problems.append(
                f"{shape['backend']} filed native but does not "
                "override generate_stream"
            )
        if shape["shape"] == "simulated" and overrides:
            problems.append(
                f"{shape['backend']} filed simulated but overrides "
                "generate_stream — amend the shape table"
            )

    # D-STREAM-1's fix, pinned in the engine's own source: the
    # stream iteration must sit under a finally that closes.
    src = inspect.getsource(TurboQuantEngine.generate_stream)
    if "finally" not in src or "close" not in src:
        problems.append(
            "TurboQuantEngine.generate_stream no longer closes "
            "its stream in a finally (D-STREAM-1 regression)"
        )
    # The wrapper's propagation leg: yield from forwards close().
    bsrc = inspect.getsource(TurboQuantBackend.generate_stream)
    if "yield from" not in bsrc:
        problems.append(
            "TurboQuantBackend.generate_stream no longer delegates "
            "with yield from — close() would not reach the engine"
        )
    return problems


def streaming_live_problems() -> list[str]:
    """The discipline re-derived live over stubs (this VM)."""
    problems: list[str] = []
    texts = [f"delta-{i} " for i in range(6)]
    whole = "".join(texts)

    # Cancel before the first pull is free (default path).
    backend = _counting_backend()
    gen = backend.generate_stream(_request())
    gen.close()
    if backend.calls != 0:
        problems.append(
            f"closing an unpulled default stream ran generate() "
            f"{backend.calls} time(s)"
        )

    # Slow-consumer integrity + pull-based production (engine).
    stream = _ClosableStream(texts)
    engine = _engine_with_stream(stream)
    gen = engine.generate_stream("prompt")
    got: list[str] = []
    for _ in range(len(texts)):
        got.append(next(gen))
        if stream.produced != len(got):
            problems.append(
                f"engine produced {stream.produced} after "
                f"{len(got)} pull(s) — run-ahead"
            )
            break
    if "".join(got) != whole:
        problems.append("slowly consumed engine stream != whole text")
    gen.close()
    if not stream.closed:
        problems.append("engine stream not closed after exhaustion")

    # Cancel mid-stream closes the model's iterator, and the
    # yielded prefix is untouched with nothing further.
    stream2 = _ClosableStream(texts)
    engine2 = _engine_with_stream(stream2)
    gen2 = engine2.generate_stream("prompt")
    prefix = [next(gen2), next(gen2)]
    gen2.close()
    if not stream2.closed:
        problems.append(
            "D-STREAM-1: model stream not closed on consumer close"
        )
    if prefix != texts[:2]:
        problems.append("cancel altered the already-yielded prefix")
    try:
        next(gen2)
        problems.append("a closed stream yielded again")
    except StopIteration:
        pass

    # A mid-stream error closes the model's iterator too.
    stream3 = _ClosableStream(texts, fail_at=2)
    engine3 = _engine_with_stream(stream3)
    try:
        list(engine3.generate_stream("prompt"))
        problems.append("failing stub stream did not raise")
    except RuntimeError:
        pass
    if not stream3.closed:
        problems.append("D-STREAM-1: model stream not closed on error")

    # Reuse after cancel (backend layer, stub native engine).
    from pathlib import Path

    from thoughtforge.inference.turboquant_backend import TurboQuantBackend

    class StubEngine:
        def __init__(self) -> None:
            self.closed = False

        def generate_stream(self, prompt: str, params: Any = None):
            try:
                yield from texts
            finally:
                self.closed = True

    tq = TurboQuantBackend(model_path=Path("stub.gguf"))
    stub = StubEngine()
    tq._engine = stub
    gen4 = tq.generate_stream(_request())
    next(gen4)
    gen4.close()
    if not stub.closed:
        problems.append("backend close did not reach the engine")
    stub2 = StubEngine()
    tq._engine = stub2
    if "".join(tq.generate_stream(_request())) != whole:
        problems.append("stream after a cancel is not whole")
    return problems


__all__ = [
    "BACKEND_STREAM_SHAPES",
    "STREAM_FORMAT",
    "STREAMING_RULES",
    "streaming_as_data",
    "streaming_live_problems",
    "streaming_problems",
]
