"""
TF-036 — Determinism mode: seeded runs reproducible for tests
and benchmarks.

The mode lives in ``src/thoughtforge/cognition/determinism.py``
(the source census, the mode + resolver, the seeded id stream,
the reset, the fingerprint + reproducibility checker, the
seeded script runner); the core plumbing lives in ``core.py``
(``_mint_id`` / ``_next_generation_seed`` and their call sites);
the seed fields live on ``GenerationParams`` /
``GenerationRequest`` with the TurboQuant forward in
``inference/turboquant.py`` and the salvage-refine seed in
``refinement/salvage.py``. The write-up is
``docs/architecture/DETERMINISM_MODE.md``.

Pinned here:

  * the census: the seven named surfaces, each with a valid
    ruling; lookup by name, bogus names raising,
  * resolution: no config / no key = mode off; a seed (0
    included) = mode on; non-int, negative, float, string and
    bool seeds raise at resolution and at core construction,
  * the id stream: same seed = same sequence in the standing
    shapes, different seed = different sequence, ids advance
    within one stream,
  * seeded runs through a seed-honouring stochastic backend:
    two fresh cores with the same seed reproduce exactly
    (fingerprints equal, checker silent, the backend saw the
    same seed sequence); a different seed is caught by name;
    unseeded cores carry seed=None on every request and mint
    differing ids,
  * the seeded script runner + reset: a multi-turn script
    re-run on the SAME core reproduces (the will-ledger
    carry-over regression: finals matched but scaffolds wore
    the previous run's standing goal before the reset learned
    to restart the ledger); reset/runner on an unseeded core
    raise before running anything,
  * two seeded cores interleaved turn-by-turn still each
    reproduce their solo run (dedicated per-core streams —
    the global RNG is never seeded, and its state is proven
    untouched by a seeded turn),
  * knowledge-only seeded cores reproduce, and the fingerprint
    excludes timestamps/timings (mutating them post-run still
    compares clean) while naming real drift (a tampered final
    text / candidate id is caught by field path),
  * the local-engine paths: plain and streamed seeded turns
    reproduce, the engine recorded the same non-None seeds on
    both cores; unseeded engine calls carry seed=None,
  * the inference plumbing: both seed fields default to None
    and _merge_params carries a set seed through.
"""

from __future__ import annotations

import random
import re
import tempfile
from pathlib import Path

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.determinism import (
    DETERMINISM_SOURCES,
    VALID_RULINGS,
    DeterministicIds,
    reproducibility_problems,
    reset_deterministic_state,
    resolve_determinism_mode,
    run_seeded_script,
    source_ruling,
    turn_fingerprint,
)
from thoughtforge.inference.turboquant import (
    GenerationParams,
    GenerationResult,
    TurboQuantEngine,
)
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.knowledge.models import RuntimeTurnState

FERRY = "What time does the ferry leave?"
PLAN = "Help me plan the lantern festival route for Lantern Hollow"

VARIANTS = [
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning.",
    "The ferry over the River Sarn needs eleven minutes on a calm "
    "morning, passing the willow narrows before reaching the mill "
    "landing where the lantern-keeper waits.",
    "Eleven minutes is the calm-water time for the River Sarn "
    "ferry from the willow narrows to the mill landing, as the "
    "lantern-keeper records each morning at dawn.",
]


# ── Fixture generation sources ────────────────────────────────────────────────


class StochasticBackend(UnifiedBackend):
    """Samples among phrasings with random.Random(request.seed)
    when a seed arrives, and with a system-seeded Random when
    none does — so unseeded runs genuinely vary in principle,
    and seeded runs must match exactly. Records every seed."""

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def generate(self, request):
        self.seeds.append(request.seed)
        rng = (
            random.Random(request.seed)
            if request.seed is not None
            else random.Random()
        )
        text = rng.choice(VARIANTS)
        return GenerationResponse(
            text=text, tokens_generated=len(text.split())
        )

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf036-stochastic"


class StochasticEngine:
    """The local-engine twin: generate + generate_stream in the
    TurboQuantEngine's shape, sampling by params.seed."""

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def _pick(self, params) -> str:
        seed = params.seed if params is not None else None
        self.seeds.append(seed)
        rng = random.Random(seed) if seed is not None else random.Random()
        return rng.choice(VARIANTS)

    def generate(self, prompt, params=None):
        text = self._pick(params)
        return GenerationResult(
            text=text,
            tokens_generated=len(text.split()),
            tokens_prompt=10,
            total_tokens=10 + len(text.split()),
            duration_ms=1,
            tokens_per_second=100.0,
        )

    def generate_stream(self, prompt, params=None):
        yield self._pick(params)


def _core(backend=None, engine=None, seed=None) -> ThoughtForgeCore:
    tmp = Path(tempfile.mkdtemp())
    config: dict = {"cognition": {}}
    if seed is not None:
        config["determinism"] = {"seed": seed}
    core = ThoughtForgeCore(
        config=config,
        memory_dir=tmp / "memory",
        db_path=tmp / "none.db",
        backend=backend,
    )
    if engine is not None:
        core._engine = engine
    return core


# ── The census ────────────────────────────────────────────────────────────────


class TestCensus:
    def test_seven_surfaces_named(self) -> None:
        assert {s.surface for s in DETERMINISM_SOURCES} == {
            "router_scaffold_prompts", "scoring_salvage_enforcement",
            "generation_sampling", "record_ids",
            "timestamps_and_timings", "chat_history_ids", "global_rng",
        }

    def test_every_source_has_valid_ruling_and_detail(self) -> None:
        for source in DETERMINISM_SOURCES:
            assert source.ruling in VALID_RULINGS, source.surface
            assert len(source.detail) > 40, source.surface

    def test_lookup_by_name(self) -> None:
        assert source_ruling("generation_sampling").ruling == "seeded-in-mode"
        assert source_ruling("global_rng").ruling == "excluded-from-reproducibility"

    def test_lookup_bogus_raises_naming_valid(self) -> None:
        with pytest.raises(ValueError) as exc:
            source_ruling("astrology")
        assert "astrology" in str(exc.value)
        assert "record_ids" in str(exc.value)


# ── Resolution ────────────────────────────────────────────────────────────────


class TestResolveMode:
    def test_no_config_is_off(self) -> None:
        assert not resolve_determinism_mode(None).enabled
        assert not resolve_determinism_mode({}).enabled
        assert not resolve_determinism_mode({"determinism": {}}).enabled

    def test_seed_turns_mode_on_and_round_trips(self) -> None:
        mode = resolve_determinism_mode({"determinism": {"seed": 42}})
        assert mode.enabled and mode.seed == 42
        assert mode.to_dict() == {"seed": 42}

    def test_seed_zero_is_a_real_seed(self) -> None:
        mode = resolve_determinism_mode({"determinism": {"seed": 0}})
        assert mode.enabled and mode.seed == 0

    @pytest.mark.parametrize("bad", [-1, 1.5, "42", True])
    def test_bad_seeds_raise(self, bad) -> None:
        with pytest.raises(ValueError):
            resolve_determinism_mode({"determinism": {"seed": bad}})

    def test_non_mapping_section_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_determinism_mode({"determinism": 42})

    def test_core_construction_rejects_bad_seed(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        with pytest.raises(ValueError):
            ThoughtForgeCore(
                config={"cognition": {}, "determinism": {"seed": -3}},
                memory_dir=tmp / "memory",
                db_path=tmp / "none.db",
            )

    def test_core_reports_its_mode(self) -> None:
        assert _core(seed=42).determinism_mode.seed == 42
        assert not _core().determinism_mode.enabled


# ── The id stream ─────────────────────────────────────────────────────────────


class TestDeterministicIds:
    def test_same_seed_same_sequence(self) -> None:
        a, b = DeterministicIds(42), DeterministicIds(42)
        seq_a = [a.new_id("turn", 8), a.new_id("cand", 6), a.new_id("resp", 8)]
        seq_b = [b.new_id("turn", 8), b.new_id("cand", 6), b.new_id("resp", 8)]
        assert seq_a == seq_b

    def test_different_seed_different_sequence(self) -> None:
        assert DeterministicIds(42).new_id("turn", 8) != (
            DeterministicIds(7).new_id("turn", 8)
        )

    def test_shapes_match_the_standing_id_shapes(self) -> None:
        ids = DeterministicIds(1)
        assert re.fullmatch(r"turn_[0-9a-f]{8}", ids.new_id("turn", 8))
        assert re.fullmatch(r"cand_[0-9a-f]{6}", ids.new_id("cand", 6))
        assert re.fullmatch(r"frag_[0-9a-f]{6}", ids.new_id("frag", 6))
        assert re.fullmatch(r"resp_[0-9a-f]{8}", ids.new_id("resp", 8))

    def test_ids_advance_within_one_stream(self) -> None:
        ids = DeterministicIds(42)
        first, second = ids.new_id("turn", 8), ids.new_id("turn", 8)
        assert first != second
        assert ids.minted == 2


# ── Seeded runs through the backend ───────────────────────────────────────────


class TestSeededBackendRuns:
    def test_two_fresh_cores_reproduce_exactly(self) -> None:
        backend_a, backend_b = StochasticBackend(), StochasticBackend()
        core_a, core_b = _core(backend_a, seed=42), _core(backend_b, seed=42)
        final_a, final_b = core_a.think(FERRY), core_b.think(FERRY)
        assert final_a.text == final_b.text
        assert backend_a.seeds == backend_b.seeds
        assert backend_a.seeds and all(s is not None for s in backend_a.seeds)
        assert reproducibility_problems(
            core_a.last_turn_state, core_b.last_turn_state
        ) == []

    def test_different_seed_is_caught_by_name(self) -> None:
        core_a = _core(StochasticBackend(), seed=42)
        core_b = _core(StochasticBackend(), seed=7)
        core_a.think(FERRY)
        core_b.think(FERRY)
        problems = reproducibility_problems(
            core_a.last_turn_state, core_b.last_turn_state
        )
        assert problems
        assert any(p.startswith("turn_id") for p in problems)

    def test_unseeded_cores_carry_no_seed_and_mint_fresh_ids(self) -> None:
        backend_a, backend_b = StochasticBackend(), StochasticBackend()
        core_a, core_b = _core(backend_a), _core(backend_b)
        core_a.think(FERRY)
        core_b.think(FERRY)
        assert backend_a.seeds and all(s is None for s in backend_a.seeds)
        assert (
            core_a.last_turn_state.turn_id != core_b.last_turn_state.turn_id
        )

    def test_script_on_two_fresh_cores_reproduces(self) -> None:
        run_a = run_seeded_script(
            _core(StochasticBackend(), seed=42), [FERRY, PLAN]
        )
        run_b = run_seeded_script(
            _core(StochasticBackend(), seed=42), [FERRY, PLAN]
        )
        assert run_a.fingerprints == run_b.fingerprints
        assert [f.text for f in run_a.finals] == [f.text for f in run_b.finals]

    def test_same_core_rerun_with_reset_reproduces(self) -> None:
        # Regression: the script's second turn sets a standing
        # goal; before the reset restarted the will ledger, the
        # re-run's scaffolds carried the previous run's goal and
        # the fingerprints differed while the finals matched.
        core = _core(StochasticBackend(), seed=42)
        run_1 = run_seeded_script(core, [FERRY, PLAN])
        run_2 = run_seeded_script(core, [FERRY, PLAN])
        assert run_1.fingerprints == run_2.fingerprints

    def test_reset_on_unseeded_core_raises(self) -> None:
        with pytest.raises(ValueError):
            reset_deterministic_state(_core(StochasticBackend()))

    def test_runner_on_unseeded_core_raises_before_running(self) -> None:
        core = _core(StochasticBackend())
        with pytest.raises(ValueError):
            run_seeded_script(core, [FERRY])
        assert core.last_turn_state is None

    def test_interleaved_seeded_cores_still_reproduce(self) -> None:
        solo_a = run_seeded_script(
            _core(StochasticBackend(), seed=42), [FERRY, PLAN]
        )
        solo_b = run_seeded_script(
            _core(StochasticBackend(), seed=7), [FERRY, PLAN]
        )
        core_a = _core(StochasticBackend(), seed=42)
        core_b = _core(StochasticBackend(), seed=7)
        fp_a, fp_b = [], []
        for text in (FERRY, PLAN):
            core_a.think(text)
            fp_a.append(turn_fingerprint(core_a.last_turn_state))
            core_b.think(text)
            fp_b.append(turn_fingerprint(core_b.last_turn_state))
        assert fp_a == solo_a.fingerprints
        assert fp_b == solo_b.fingerprints

    def test_global_rng_state_untouched_by_seeded_turn(self) -> None:
        random.seed(20261031)
        before = random.getstate()
        _core(StochasticBackend(), seed=42).think(FERRY)
        assert random.getstate() == before


# ── Knowledge-only runs + the fingerprint's terms ─────────────────────────────


class TestKnowledgeOnlyAndFingerprint:
    def test_seeded_knowledge_only_cores_reproduce(self) -> None:
        core_a, core_b = _core(seed=42), _core(seed=42)
        core_a.think(FERRY)
        core_b.think(FERRY)
        assert reproducibility_problems(
            core_a.last_turn_state, core_b.last_turn_state
        ) == []

    def test_timings_and_timestamps_are_excluded(self) -> None:
        core_a, core_b = _core(seed=42), _core(seed=42)
        core_a.think(FERRY)
        core_b.think(FERRY)
        turn_b = core_b.last_turn_state
        turn_b.retrieval_ms = 999
        turn_b.generation_ms = 888
        turn_b.salvage_ms = 777
        turn_b.completed_at = "2999-01-01T00:00:00+00:00"
        assert reproducibility_problems(core_a.last_turn_state, turn_b) == []

    def test_tampered_final_text_is_named(self) -> None:
        core_a, core_b = _core(seed=42), _core(seed=42)
        core_a.think(FERRY)
        core_b.think(FERRY)
        core_b.last_turn_state.final_response.text = "A different answer."
        problems = reproducibility_problems(
            core_a.last_turn_state, core_b.last_turn_state
        )
        assert any(p.startswith("final.text") for p in problems)

    def test_unfinished_turn_fingerprint_raises(self) -> None:
        with pytest.raises(ValueError):
            turn_fingerprint(RuntimeTurnState(turn_id="turn_deadbeef"))


# ── The local-engine paths ────────────────────────────────────────────────────


class TestSeededEngineRuns:
    def test_plain_engine_turns_reproduce(self) -> None:
        engine_a, engine_b = StochasticEngine(), StochasticEngine()
        core_a = _core(engine=engine_a, seed=42)
        core_b = _core(engine=engine_b, seed=42)
        final_a, final_b = core_a.think(FERRY), core_b.think(FERRY)
        assert final_a.text == final_b.text
        assert engine_a.seeds == engine_b.seeds
        assert engine_a.seeds and all(s is not None for s in engine_a.seeds)
        assert reproducibility_problems(
            core_a.last_turn_state, core_b.last_turn_state
        ) == []

    def test_streamed_engine_turns_reproduce(self) -> None:
        engine_a, engine_b = StochasticEngine(), StochasticEngine()
        core_a = _core(engine=engine_a, seed=42)
        core_b = _core(engine=engine_b, seed=42)

        def events_of(core):
            return [
                (e.kind, e.mode, e.candidate_id, e.text, e.native)
                for e in core.think_stream(FERRY)
            ]

        events_a, events_b = events_of(core_a), events_of(core_b)
        assert events_a == events_b
        assert events_a[-1][0] == "final"
        assert reproducibility_problems(
            core_a.last_turn_state, core_b.last_turn_state
        ) == []

    def test_unseeded_engine_calls_carry_no_seed(self) -> None:
        engine = StochasticEngine()
        _core(engine=engine).think(FERRY)
        assert engine.seeds and all(s is None for s in engine.seeds)


# ── The inference plumbing ────────────────────────────────────────────────────


class TestInferenceSeedPlumbing:
    def test_generation_params_seed_defaults_to_none(self) -> None:
        assert GenerationParams().seed is None
        assert GenerationParams(seed=5).seed == 5

    def test_generation_request_seed_defaults_to_none(self) -> None:
        assert GenerationRequest(prompt="hi").seed is None
        assert GenerationRequest(prompt="hi", seed=5).seed == 5

    def test_merge_params_carries_the_seed(self) -> None:
        engine = TurboQuantEngine(model_path=None, profile_id="desktop_cpu")
        engine.load()
        try:
            assert engine._merge_params(GenerationParams(seed=99))["seed"] == 99
            assert engine._merge_params(None)["seed"] is None
        finally:
            engine.unload()
