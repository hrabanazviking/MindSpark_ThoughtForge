"""
TF-026 — Streaming turns: cognition works on streamed generation,
scaffold applied pre/post stream honestly.

The contract lives in ``src/thoughtforge/cognition/streaming.py``
(StreamEvent + chunk_words); the core's streamed turn is
``ThoughtForgeCore.think_stream``; the sources are
``TurboQuantEngine.generate_stream`` and ``UnifiedBackend``'s
default/overridable ``generate_stream``. Write-up:
``docs/architecture/STREAMING_TURNS.md``.

Pinned here:

  * the event contract: fixed kinds, turn_started first, exactly
    one terminal final; the final event's record IS the retained
    turn state's final_response,
  * honesty of the ``native`` flag: chunks from a truly streaming
    source are native; chunks re-cut from completed text (engine
    without generate_stream, default unified backend) are not,
  * pre-stream honesty: the prompt a streamed candidate is built
    from is the same scaffolded candidate prompt think() builds
    for the same turn shape,
  * post-stream honesty: salvage/enforcement still run on the
    completed candidate text — the final event text is the
    composed final, not a concatenation of streamed chunks,
  * failure honesty: a candidate whose stream raises mid-way is
    discarded whole and surfaces as candidate_error; the turn
    still completes; if every candidate fails the turn degrades
    to the knowledge-only final,
  * knowledge-only turns stream nothing but turn_started + final,
  * think() is unchanged in shape: same final for the same stub
    engine, no events involved,
  * TurboQuantEngine.generate_stream over a fake llama: deltas in
    order, unstripped join, stream=True passed to the model call.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.prompt_builder import PromptBuilder
from thoughtforge.cognition.streaming import (
    EVENT_KINDS,
    WORDS_PER_CHUNK,
    StreamEvent,
    chunk_words,
)
from thoughtforge.inference.turboquant import TurboQuantEngine
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)

TEXT = (
    "Yggdrasil is the world tree that joins the nine realms. "
    "Its roots reach into deep wells of wisdom and fate. "
    "Odin hung upon its branches for nine nights seeking runes."
)


def _core() -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )


class _NativeStubEngine:
    """Engine whose generate_stream yields fixed-size deltas."""

    def __init__(self, text: str = TEXT, step: int = 17) -> None:
        self.text = text
        self.step = step
        self.prompts: list[str] = []

    def generate_stream(self, prompt, params=None):
        self.prompts.append(prompt)
        for i in range(0, len(self.text), self.step):
            yield self.text[i:i + self.step]

    def generate(self, prompt, params=None):  # pragma: no cover
        raise AssertionError("generate() must not be used when generate_stream exists")


class _CompletedStubEngine:
    """Engine with only the completed-text generate()."""

    def __init__(self, text: str = TEXT) -> None:
        self.text = text

    def generate(self, prompt, params=None):
        return SimpleNamespace(text=self.text, tokens_generated=len(self.text.split()))


class _FlakyStreamEngine:
    """First mode's stream dies mid-way; second mode completes."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_stream(self, prompt, params=None):
        self.calls += 1
        if self.calls == 1:
            yield TEXT[:20]
            raise RuntimeError("stream broke mid-way")
        yield TEXT


# ── The event contract ─────────────────────────────────────────────────────────


class TestEventContract:
    def test_event_kinds_are_the_six(self) -> None:
        # TF-027 extended the contract with ONE kind: "interrupted",
        # the terminal event of a cut turn (it replaces "final" for
        # that turn; the original six and their order are unchanged).
        assert EVENT_KINDS == (
            "turn_started", "candidate_start", "chunk",
            "candidate_end", "candidate_error", "final",
            "interrupted",
        )

    def test_event_defaults(self) -> None:
        ev = StreamEvent(kind="chunk", turn_id="t1")
        assert ev.mode == "" and ev.candidate_id == ""
        assert ev.text == "" and ev.native is False and ev.final is None

    def test_chunk_words_rejoins_to_normalised_text(self) -> None:
        text = "one  two\nthree four five six seven eight nine ten"
        chunks = list(chunk_words(text, words_per_chunk=3))
        assert " ".join(chunks) == "one two three four five six seven eight nine ten"
        assert all(len(c.split()) <= 3 for c in chunks)

    def test_chunk_words_default_size_and_empty(self) -> None:
        assert list(chunk_words("")) == []
        words = " ".join(f"w{i}" for i in range(WORDS_PER_CHUNK + 1))
        assert len(list(chunk_words(words))) == 2

    def test_chunk_words_rejects_bad_size(self) -> None:
        with pytest.raises(ValueError):
            list(chunk_words("x", words_per_chunk=0))


# ── Knowledge-only turns ───────────────────────────────────────────────────────


class TestKnowledgeOnlyStream:
    def test_only_started_and_final(self) -> None:
        core = _core()
        events = list(core.think_stream("What is Yggdrasil?"))
        assert [e.kind for e in events] == ["turn_started", "final"]
        assert events[0].turn_id == events[1].turn_id
        final = events[1].final
        assert final is not None
        assert events[1].text == final.text
        turn = core.last_turn_state
        assert turn is not None and turn.final_response is final
        assert turn.phase == "completed"


# ── Native streaming ───────────────────────────────────────────────────────────


class TestNativeStream:
    def test_event_shape_and_native_flag(self) -> None:
        core = _core()
        engine = _NativeStubEngine()
        core._engine = engine
        events = list(core.think_stream("What is Yggdrasil?"))
        kinds = [e.kind for e in events]
        assert kinds[0] == "turn_started" and kinds[-1] == "final"
        assert kinds.count("final") == 1
        chunks = [e for e in events if e.kind == "chunk"]
        assert chunks and all(e.native for e in chunks)
        # Per-candidate: chunk texts concatenate to the candidate text.
        ends = [e for e in events if e.kind == "candidate_end"]
        assert len(ends) == 2  # the scaffold's two candidate modes
        for end in ends:
            own = [c.text for c in chunks if c.candidate_id == end.candidate_id]
            assert "".join(own) == end.text == TEXT
        turn = core.last_turn_state
        assert turn is not None
        assert [c.text for c in turn.candidates] == [TEXT, TEXT]

    def test_streamed_prompt_is_the_scaffolded_candidate_prompt(self) -> None:
        # Pre-stream honesty: the streamed generation consumes the
        # same prompt think() would build for the turn's first mode.
        core = _core()
        engine = _NativeStubEngine()
        core._engine = engine
        list(core.think_stream("What is Yggdrasil?"))
        turn = core.last_turn_state
        assert turn is not None and turn.input_sketch is not None and turn.scaffold is not None
        expected = PromptBuilder().build_candidate_prompt(
            sketch=turn.input_sketch,
            scaffold=turn.scaffold,
            mode=turn.scaffold.candidate_modes[0],
            memory_cues=PromptBuilder().extract_memory_cues(turn.memory_bundle),
        )
        assert engine.prompts[0] == expected

    def test_final_is_composed_not_concatenated_chunks(self) -> None:
        # Post-stream honesty: salvage + enforcement ran on the
        # completed candidates; the final event carries the turn's
        # composed FinalResponseRecord, whatever its text is.
        core = _core()
        core._engine = _NativeStubEngine()
        events = list(core.think_stream("What is Yggdrasil?"))
        turn = core.last_turn_state
        assert turn is not None
        assert events[-1].final is turn.final_response
        assert events[-1].text == turn.final_response.text


# ── Simulated streaming is labelled ────────────────────────────────────────────


class TestSimulatedStream:
    def test_completed_engine_chunks_are_not_native(self) -> None:
        core = _core()
        core._engine = _CompletedStubEngine()
        events = list(core.think_stream("What is Yggdrasil?"))
        chunks = [e for e in events if e.kind == "chunk"]
        assert chunks and all(not e.native for e in chunks)
        ends = [e for e in events if e.kind == "candidate_end"]
        assert ends and all(e.text == TEXT for e in ends)
        # The candidate keeps the source's original text exactly,
        # not a chunk reconstruction.
        turn = core.last_turn_state
        assert turn is not None
        assert all(c.text == TEXT for c in turn.candidates)
        assert all(c.token_estimate == len(TEXT.split()) for c in turn.candidates)

    def test_unified_default_backend_chunks_are_not_native(self) -> None:
        class _Backend(UnifiedBackend):
            def generate(self, request: GenerationRequest) -> GenerationResponse:
                return GenerationResponse(text=TEXT, tokens_generated=len(TEXT.split()))

            def health_check(self) -> bool:
                return True

            def backend_name(self) -> str:
                return "stub"

        core = _core()
        core._unified_backend = _Backend()
        events = list(core.think_stream("What is Yggdrasil?"))
        chunks = [e for e in events if e.kind == "chunk"]
        assert chunks and all(not e.native for e in chunks)
        ends = [e for e in events if e.kind == "candidate_end"]
        assert len(ends) == 1 and ends[0].text == TEXT

    def test_unified_native_override_chunks_are_native(self) -> None:
        class _StreamingBackend(UnifiedBackend):
            def generate(self, request: GenerationRequest) -> GenerationResponse:
                raise AssertionError("generate() must not be used")

            def generate_stream(self, request: GenerationRequest):
                for i in range(0, len(TEXT), 11):
                    yield TEXT[i:i + 11]

            def health_check(self) -> bool:
                return True

            def backend_name(self) -> str:
                return "stub-stream"

        core = _core()
        core._unified_backend = _StreamingBackend()
        events = list(core.think_stream("What is Yggdrasil?"))
        chunks = [e for e in events if e.kind == "chunk"]
        assert chunks and all(e.native for e in chunks)
        assert "".join(c.text for c in chunks) == TEXT

    def test_unified_backend_error_yields_candidate_error(self) -> None:
        class _ErrBackend(UnifiedBackend):
            def generate(self, request: GenerationRequest) -> GenerationResponse:
                return GenerationResponse(text="", error="backend down")

            def health_check(self) -> bool:
                return False

            def backend_name(self) -> str:
                return "stub-err"

        core = _core()
        core._unified_backend = _ErrBackend()
        events = list(core.think_stream("What is Yggdrasil?"))
        kinds = [e.kind for e in events]
        assert "candidate_error" in kinds
        assert kinds[-1] == "final"
        turn = core.last_turn_state
        assert turn is not None and turn.candidates == []
        assert events[-1].final is not None
        assert events[-1].final.mode == "knowledge_only"


# ── Failure honesty ────────────────────────────────────────────────────────────


class TestStreamFailure:
    def test_mid_stream_failure_discards_partial_candidate(self) -> None:
        core = _core()
        core._engine = _FlakyStreamEngine()
        events = list(core.think_stream("What is Yggdrasil?"))
        errors = [e for e in events if e.kind == "candidate_error"]
        assert len(errors) == 1 and "stream broke" in errors[0].text
        ends = [e for e in events if e.kind == "candidate_end"]
        assert len(ends) == 1 and ends[0].text == TEXT
        turn = core.last_turn_state
        assert turn is not None
        # The partial first candidate never became a candidate.
        assert len(turn.candidates) == 1 and turn.candidates[0].text == TEXT
        assert events[-1].kind == "final"

    def test_all_candidates_fail_degrades_to_knowledge_only(self) -> None:
        class _DeadEngine:
            def generate_stream(self, prompt, params=None):
                raise RuntimeError("engine dead")
                yield  # pragma: no cover — makes this a generator

        core = _core()
        core._engine = _DeadEngine()
        events = list(core.think_stream("What is Yggdrasil?"))
        assert [e.kind for e in events].count("candidate_error") == 2
        assert events[-1].kind == "final"
        assert events[-1].final is not None
        assert events[-1].final.mode == "knowledge_only"


# ── think() parity ─────────────────────────────────────────────────────────────


class TestThinkParity:
    def test_think_with_native_engine_still_returns_final(self) -> None:
        core = _core()
        core._engine = _CompletedStubEngine()
        result = core.think("What is Yggdrasil?")
        turn = core.last_turn_state
        assert turn is not None and turn.final_response is result
        assert turn.candidates and all(c.text == TEXT for c in turn.candidates)


# ── TurboQuantEngine.generate_stream ───────────────────────────────────────────


class TestEngineStream:
    def _engine_with_fake_llama(self, chunks: list[dict]) -> tuple[TurboQuantEngine, dict]:
        from thoughtforge.inference.profiles import ProfileSelector, resolve_generation_params

        engine = TurboQuantEngine()
        seen: dict = {}

        def fake_llama(prompt, **kwargs):
            seen.update(kwargs)
            seen["prompt"] = prompt
            return iter(chunks)

        profile = ProfileSelector().load("desktop_cpu")
        engine._params = resolve_generation_params(profile.profile)
        engine._model = fake_llama
        engine._loaded = True
        return engine, seen

    def test_deltas_in_order_unstripped_and_stream_flag(self) -> None:
        chunks = [
            {"choices": [{"text": " Ygg"}]},
            {"choices": [{"text": "drasil"}]},
            {"choices": []},
            {"choices": [{"text": " rises."}]},
        ]
        engine, seen = self._engine_with_fake_llama(chunks)
        deltas = list(engine.generate_stream("prompt"))
        assert deltas == [" Ygg", "drasil", " rises."]
        assert "".join(deltas) == " Yggdrasil rises."
        assert seen["stream"] is True
        assert seen["prompt"] == "prompt"

    def test_mid_stream_model_error_raises_runtime(self) -> None:
        def bad_iter():
            yield {"choices": [{"text": "partial"}]}
            raise ValueError("llama died")

        engine, _ = self._engine_with_fake_llama([])
        engine._model = lambda prompt, **kwargs: bad_iter()
        with pytest.raises(RuntimeError):
            list(engine.generate_stream("prompt"))
