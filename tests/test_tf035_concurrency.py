"""
TF-035 — Concurrency: two conversations at once without
state bleed (fixtures prove isolation).

The contract lives in
``src/thoughtforge/cognition/concurrency.py`` (the surface
table, the isolation checker, the concurrent runner); the
same-core fix lives in ``core.py`` (the per-core turn lock);
the write-up is ``docs/architecture/CONCURRENCY.md``.

Pinned here:

  * the surface table: the ten named surfaces, each with a
    valid owner and a stated rule; lookup by name, with a
    bogus name raising and naming the valid surfaces,
  * the checker on retained state: fresh cores report their
    missing turns; one core on both sides is named as one
    conversation, not two; a tampered shared turn state is
    caught by name; clean sequential turns on two cores pass,
  * two cores in parallel (barrier-synchronised backends, so
    both generations are provably in flight together): each
    caller gets its own answer; turn ids, histories, and
    scaffolds stay separate; one conversation's standing goal
    steers its own next turn and is absent from the other's;
    per-conversation config (depth / candidate modes) and
    persona each hold on their own core; the checker finds
    no problems,
  * shared-read surfaces under concurrency: two cores on one
    knowledge DB path answer their own turns, two
    KnowledgeForge instances retrieve from one DB at once,
    and the perf singleton is one thread-safe aggregate,
  * the same-core fix (this slice's discovered defect): two
    threads on ONE core — which before the fix handed both
    callers the same final text in 3/3 probe trials — now
    serialise: each caller gets its own answer, the backend's
    generate intervals never overlap, two concurrent streams
    each end in their own final, and a stream closed early
    releases the lock so the next turn still runs.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import pytest

from thoughtforge.cognition.chat_history import ChatHistory
from thoughtforge.cognition.coherence import extract_request
from thoughtforge.cognition.concurrency import (
    CONCURRENCY_SURFACES,
    VALID_OWNERS,
    isolation_problems,
    run_concurrent_turns,
    surface_rule,
)
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.etl.schema import initialize_schema

from tests.fixtures.synthetic import build_synthetic_store

FILLER = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)

PLAN_A = "Help me plan the lantern festival route for Lantern Hollow"
PLAN_B = "Help me plan the harbour winter market for the mill landing"
FERRY = "What time does the ferry leave?"
WEATHER = "What is the weather pattern over the northern sea?"


# ── Fixture backend ───────────────────────────────────────────────────────────


class EchoBackend(UnifiedBackend):
    """Answers about the prompt's own user request, so a caller
    who receives another conversation's answer is visible at a
    glance. Optionally barrier-synchronised (both conversations
    provably in generation at once) or slowed (interval overlap
    becomes measurable)."""

    def __init__(self, barrier: threading.Barrier | None = None,
                 delay: float = 0.0) -> None:
        self.barrier = barrier
        self.delay = delay
        self.prompts: list[str] = []
        self.intervals: list[tuple[float, float]] = []

    def generate(self, request):
        started = time.perf_counter()
        self.prompts.append(request.prompt)
        if self.barrier is not None:
            self.barrier.wait(timeout=15)
        if self.delay:
            time.sleep(self.delay)
        req = extract_request(request.prompt)
        text = f"Answer about {req}. {FILLER}"
        self.intervals.append((started, time.perf_counter()))
        return GenerationResponse(
            text=text, tokens_generated=len(text.split())
        )

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf035-echo"


def _core(backend=None, config=None, memory_dir=None, db_path=None,
          synthetic_store: bool = False) -> ThoughtForgeCore:
    tmp = Path(tempfile.mkdtemp())
    mem = memory_dir if memory_dir is not None else tmp / "memory"
    if synthetic_store:
        build_synthetic_store(mem)
    return ThoughtForgeCore(
        config=config if config is not None else {"cognition": {}},
        memory_dir=mem,
        db_path=db_path if db_path is not None else tmp / "none.db",
        backend=backend,
    )


def _pair(barrier=None, delay: float = 0.0):
    return (
        _core(backend=EchoBackend(barrier, delay)),
        _core(backend=EchoBackend(barrier, delay)),
    )


# ── The surface table ─────────────────────────────────────────────────────────


class TestSurfaceTable:
    def test_ten_surfaces_named(self) -> None:
        assert {r.surface for r in CONCURRENCY_SURFACES} == {
            "conversation_config", "will_ledger", "persona_expectation",
            "last_turn_state", "generation_source", "chat_history",
            "knowledge_database", "memory_dir", "perf_tracker",
            "same_core_turns",
        }

    def test_every_rule_has_valid_owner_and_text(self) -> None:
        for rule in CONCURRENCY_SURFACES:
            assert rule.owner in VALID_OWNERS, rule.surface
            assert len(rule.rule) > 40, rule.surface

    def test_lookup_by_name(self) -> None:
        rule = surface_rule("same_core_turns")
        assert rule.owner == "serialised"
        assert "turn lock" in rule.rule

    def test_lookup_bogus_raises_naming_valid(self) -> None:
        with pytest.raises(ValueError) as exc:
            surface_rule("telepathy")
        assert "telepathy" in str(exc.value)
        assert "will_ledger" in str(exc.value)

    def test_one_core_per_conversation_is_the_serialised_rule(self) -> None:
        owners = {r.surface: r.owner for r in CONCURRENCY_SURFACES}
        assert owners["will_ledger"] == "conversation"
        assert owners["chat_history"] == "caller"
        assert owners["knowledge_database"] == "shared-read"
        assert owners["perf_tracker"] == "shared-aggregate"


# ── The checker (pure, on retained state) ─────────────────────────────────────


class TestChecker:
    def test_fresh_cores_report_missing_turns(self) -> None:
        core_a, core_b = _pair()
        problems = isolation_problems(core_a, core_b)
        assert any("conversation A" in p for p in problems)
        assert any("conversation B" in p for p in problems)

    def test_same_core_on_both_sides_is_named(self) -> None:
        core = _core(backend=EchoBackend())
        core.think(FERRY)
        problems = isolation_problems(core, core)
        assert any("same core instance" in p for p in problems)
        assert any("SAME turn state" in p for p in problems)

    def test_clean_sequential_turns_pass(self) -> None:
        core_a, core_b = _pair()
        core_a.think(PLAN_A)
        core_b.think(WEATHER)
        assert isolation_problems(core_a, core_b) == []

    def test_tampered_shared_turn_state_is_caught(self) -> None:
        core_a, core_b = _pair()
        core_a.think(PLAN_A)
        core_b.think(WEATHER)
        core_b._last_turn_state = core_a._last_turn_state
        problems = isolation_problems(core_a, core_b)
        assert any("SAME turn state" in p for p in problems)
        assert any("turn_id collision" in p for p in problems)

    def test_shared_history_object_is_named(self) -> None:
        core_a, core_b = _pair()
        shared = ChatHistory()
        core_a.think(FERRY, history=shared)
        core_b.think(FERRY, history=shared)
        problems = isolation_problems(core_a, core_b, shared, shared)
        assert any("SAME chat history" in p for p in problems)

    def test_separate_histories_pass(self) -> None:
        core_a, core_b = _pair()
        hist_a, hist_b = ChatHistory(), ChatHistory()
        hist_a.add_user(PLAN_A)
        hist_b.add_user(WEATHER)
        core_a.think(PLAN_A, history=hist_a)
        core_b.think(WEATHER, history=hist_b)
        assert isolation_problems(core_a, core_b, hist_a, hist_b) == []


# ── Two cores in parallel ─────────────────────────────────────────────────────


class TestTwoCoresConcurrent:
    def test_barrier_pair_each_gets_own_answer(self) -> None:
        core_a, core_b = _pair(barrier=threading.Barrier(2))
        run = run_concurrent_turns(core_a, PLAN_A, core_b, WEATHER)
        assert run.errors == {}
        assert run.problems == []
        assert PLAN_A in run.finals["a"].text
        assert WEATHER in run.finals["b"].text
        assert WEATHER not in run.finals["a"].text
        assert PLAN_A not in run.finals["b"].text
        assert (
            core_a.last_turn_state.turn_id != core_b.last_turn_state.turn_id
        )

    def test_will_steers_only_its_own_conversation(self) -> None:
        core_a, core_b = _pair(barrier=threading.Barrier(2))
        hist_a, hist_b = ChatHistory(), ChatHistory()
        run1 = run_concurrent_turns(
            core_a, PLAN_A, core_b, WEATHER, hist_a, hist_b
        )
        assert run1.problems == []
        assert core_a.will_ledger.active is not None
        assert core_b.will_ledger.active is None
        goal = core_a.will_ledger.active.statement

        run2 = run_concurrent_turns(core_a, FERRY, core_b, FERRY)
        assert run2.problems == []
        fact_a = core_a.last_turn_state.scaffold.fact_block or ""
        fact_b = core_b.last_turn_state.scaffold.fact_block or ""
        assert f"Standing goal: {goal}" in fact_a
        assert goal not in fact_b

    def test_config_selections_hold_per_conversation(self) -> None:
        barrier = threading.Barrier(2)
        core_a = _core(
            backend=EchoBackend(barrier),
            config={"cognition": {},
                    "conversation": {"depth": "light",
                                     "candidate_modes": ["practical"]}},
        )
        core_b = _core(backend=EchoBackend(barrier))
        run = run_concurrent_turns(core_a, PLAN_A, core_b, PLAN_B)
        assert run.errors == {} and run.problems == []
        scaffold_a = core_a.last_turn_state.scaffold
        scaffold_b = core_b.last_turn_state.scaffold
        assert scaffold_a.depth == "light"
        assert scaffold_a.candidate_modes == ["practical"]
        # B asked a planning question with no selections: the
        # builder's own derivation stands (expert depth, the
        # planning mode pair) — A's selections never reached it.
        assert scaffold_b.depth == "expert"
        assert scaffold_b.candidate_modes == ["strict_spec", "practical"]

    def test_persona_holds_per_conversation(self) -> None:
        barrier = threading.Barrier(2)
        core_a = _core(backend=EchoBackend(barrier), synthetic_store=True)
        core_b = _core(backend=EchoBackend(barrier))
        assert core_a.persona_expectation is not None
        assert core_a.persona_expectation.record_id == "syn_wrenna_001"
        assert core_b.persona_expectation is None
        run = run_concurrent_turns(core_a, FERRY, core_b, FERRY)
        assert run.errors == {} and run.problems == []
        fact_a = core_a.last_turn_state.scaffold.fact_block or ""
        fact_b = core_b.last_turn_state.scaffold.fact_block or ""
        assert "Persona traits" in fact_a and "plain-spoken" in fact_a
        assert "Persona traits" not in fact_b

    def test_histories_stay_separate_under_concurrency(self) -> None:
        core_a, core_b = _pair(barrier=threading.Barrier(2))
        hist_a, hist_b = ChatHistory(), ChatHistory()
        hist_a.add_user(PLAN_A)
        hist_b.add_user(WEATHER)
        run = run_concurrent_turns(
            core_a, PLAN_A, core_b, WEATHER, hist_a, hist_b
        )
        assert run.problems == []
        contents_a = [m.content for m in hist_a.messages]
        contents_b = [m.content for m in hist_b.messages]
        assert contents_a == [PLAN_A]
        assert contents_b == [WEATHER]


# ── Shared-read surfaces ──────────────────────────────────────────────────────


class TestSharedReadSurfaces:
    def test_two_cores_one_knowledge_db_path(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        db = tmp / "shared.db"
        initialize_schema(db)
        barrier = threading.Barrier(2)
        core_a = _core(backend=EchoBackend(barrier), db_path=db)
        core_b = _core(backend=EchoBackend(barrier), db_path=db)
        run = run_concurrent_turns(core_a, PLAN_A, core_b, WEATHER)
        assert run.errors == {} and run.problems == []
        # The shared store was empty, and the core loop's
        # retrieval gate (TF-019 §8-F2, owned by TF-041) reads
        # the default path — both bundles are honestly empty;
        # what this test pins is that sharing the path neither
        # crashes nor crosses the conversations.
        assert core_a.last_turn_state.memory_bundle.total_retrieved == 0
        assert core_b.last_turn_state.memory_bundle.total_retrieved == 0

    def test_two_forges_retrieve_from_one_db_at_once(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        db = tmp / "shared.db"
        initialize_schema(db)
        barrier = threading.Barrier(2)
        bundles: dict[str, object] = {}
        errors: dict[str, str] = {}

        def retrieve(label: str) -> None:
            try:
                barrier.wait(timeout=15)
                bundles[label] = KnowledgeForge(db_path=db).retrieve(
                    "lantern festival", path="sql"
                )
            except Exception as exc:  # noqa: BLE001 — captured as data
                errors[label] = f"{type(exc).__name__}: {exc}"

        threads = [
            threading.Thread(target=retrieve, args=(label,))
            for label in ("a", "b")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == {}
        assert bundles["a"] is not bundles["b"]
        assert bundles["a"].total_retrieved == 0
        assert bundles["b"].total_retrieved == 0

    def test_perf_tracker_is_one_thread_safe_aggregate(self) -> None:
        from thoughtforge.utils.perf import get_perf_tracker

        core_a, core_b = _pair()
        assert core_a._perf is core_b._perf is get_perf_tracker()
        tracker = get_perf_tracker()

        def record_many() -> None:
            for _ in range(50):
                tracker.record("tf035.probe", 1.0)

        threads = [threading.Thread(target=record_many) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Counted by this test's unique event name inside the
        # tracker's window — the ring buffer is capped, so a
        # total-count delta would lie once the suite has filled
        # it (found in this slice's first full-suite run).
        stats = tracker.summary(last_n=1000).events.get("tf035.probe")
        assert stats is not None and stats.count == 100


# ── One core, two threads: serialised, never bleeding ─────────────────────────


class TestSameCoreSerialised:
    def test_two_threads_each_get_their_own_answer(self) -> None:
        backend = EchoBackend(delay=0.25)
        core = _core(backend=backend)
        run = run_concurrent_turns(
            core, "Tell me about the ferry crossing at Lantern Hollow",
            core, "Tell me about the lighthouse keeper duties at dawn",
        )
        # The checker names the shared core for what it is (one
        # conversation, serialised) — the ANSWERS are what the
        # fix guarantees: each caller receives its own.
        assert run.errors == {}
        assert any("same core instance" in p for p in run.problems)
        assert "ferry crossing" in run.finals["a"].text
        assert "lighthouse keeper" in run.finals["b"].text
        assert run.finals["a"].text != run.finals["b"].text

    def test_generate_intervals_never_overlap_on_one_core(self) -> None:
        backend = EchoBackend(delay=0.2)
        core = _core(backend=backend)
        run_concurrent_turns(
            core, "Tell me about the ferry crossing at Lantern Hollow",
            core, "Tell me about the lighthouse keeper duties at dawn",
        )
        intervals = sorted(backend.intervals)
        assert len(intervals) == 2
        (a_start, a_end), (b_start, b_end) = intervals
        assert a_end <= b_start or b_end <= a_start
        assert intervals[0][1] <= intervals[1][0]

    def test_two_streams_on_one_core_each_end_in_own_final(self) -> None:
        backend = EchoBackend(delay=0.1)
        core = _core(backend=backend)
        finals: dict[str, str] = {}

        def drain(label: str, text: str) -> None:
            for event in core.think_stream(text):
                if event.kind == "final":
                    finals[label] = event.text

        threads = [
            threading.Thread(
                target=drain,
                args=("a", "Tell me about the ferry crossing at Lantern Hollow"),
            ),
            threading.Thread(
                target=drain,
                args=("b", "Tell me about the lighthouse keeper duties at dawn"),
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert "ferry crossing" in finals["a"]
        assert "lighthouse keeper" in finals["b"]
        assert finals["a"] != finals["b"]

    def test_closed_stream_releases_the_lock(self) -> None:
        core = _core(backend=EchoBackend())
        stream = core.think_stream(FERRY)
        first = next(stream)
        assert first.kind == "turn_started"
        stream.close()
        # The lock was released on close: a fresh turn on the
        # same core completes instead of deadlocking.
        final = core.think(WEATHER)
        assert final is not None and final.text

    def test_resume_gate_still_answers_under_the_lock(self) -> None:
        core = _core(backend=EchoBackend())
        with pytest.raises(ValueError):
            core.resume_turn()
