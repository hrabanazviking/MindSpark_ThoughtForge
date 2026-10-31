"""TF-079 — Cold-start budgets: time-to-first-token per
profile measured + capped in tests where the VM allows.

Roadmap: "Cold-start budgets: time-to-first-token per
profile measured + capped in tests where the VM allows."

Measured start (probed live this tick, VM):

- **No cold-start or TTFT measurement existed
  anywhere.** Grep over src / tests / docs for cold /
  ttft / first-token / first_token finds no measurement
  surface at all — only unrelated uses of the word
  "cold" (scaffold temperatures, cold thread state in
  the specs). D-COLD-1 in the TF-016 docket.
- **TF-033 covers warm turns only, and says so.**
  ``cognition/latency_budgets.py`` budgets the three
  stages of a turn that is already running (retrieval /
  generation / salvage on a constructed core); its
  docstring states plainly that no real model ran for
  that slice and that "real-model latency is
  TF-079/TF-175's road". Its first-turn note (51 ms
  warm-up against ~1 ms warm turns) is an observation,
  not a cold-start budget: nothing timed a boot.
- **TF-069 boots every profile but records no
  timing.** ``profile_runner.run_profile_boot`` runs
  nine legs per profile in simulation and files
  verdicts — no leg carries a duration, and the runner
  measures none.
- **The engine load is unmeasured.**
  ``TurboQuantEngine.load`` resolves the profile,
  selects a host backend and (with a model path) would
  construct llama_cpp — nothing times any of it, and
  the backends' ``latency_ms`` fields (TF-065) time
  *generation calls on a loaded backend*, never a load.

WHAT COLD START MEANS HERE, in parts (``STAGES``):

1. ``profile_load`` — the shipped profile JSON loads
   through the tree's own loader
   (``load_hardware_profile``).
2. ``core_construction`` — a real ``ThoughtForgeCore``
   is constructed on that profile in temp dirs (the
   TF-069 boot shape: hardware from the profile JSON,
   not host detection).
3. ``engine_load`` — a real no-model
   ``TurboQuantEngine`` loads for that profile and
   records TF-068's ``no_model`` quant verdict. No
   model file is loaded: there is no GGUF anywhere in
   this tree and ``llama_cpp`` is not installed on
   this VM (TF-066/TF-078's standing), so a real
   model's file load is not part of what is measured
   and is NOT PROVEN here.
4. ``first_token`` — time from the start of a
   streaming generation through a fixture backend to
   its first yielded chunk: this slice's TTFT. The
   fixture yields three chunks, so the first chunk is
   distinct from the completed generation (below).

The **time-to-first-token** of a cold start is the sum
of the four stages, by construction
(``ColdStartMeasurement.ttft_ms`` /
``ColdStartBudget.total_ms``) — a total is never an
independent number that could contradict its parts.

Two further quantities are measured on the same
fixture run and filed beside the stages, never summed
into TTFT (each starts its own clock at generation
start / answer start, so summing them would double
count):

- ``generation_complete_ms`` — generation start to
  stream exhaustion. For a multi-chunk (native-shaped)
  stream this is strictly greater than
  ``first_token``; for a *simulated* backend (TF-072's
  shape table: ollama / lmstudio / huggingface yield a
  single chunk that is a completed ``generate()``) the
  first chunk IS the complete generation, and the two
  numbers are equal by construction — stated, not
  hidden.
- ``first_answer_ms`` — one non-streaming
  ``generate()`` through the same fixture backend:
  the first complete answer, as opposed to its first
  chunk.

THE BUDGETS (``COLD_START_RULES`` as data; format
``thoughtforge-cold-start-budgets/1``):

- ``budgets_reuse_tf033_factors`` — per-profile
  scaling reuses TF-033's ``PROFILE_FACTORS`` table
  imported live (server_gpu x0.5, desktop_gpu x0.75,
  desktop_cpu x1, pi_5 x2, phone_low x3, pi_zero x5);
  no second factor table exists to drift.
- ``base_from_vm_measurements_with_headroom`` — the
  desktop_cpu bases are set from this tick's VM
  measurements (median of 3 simulated boots per
  profile, filed as ``MEASURED_ROWS``; see
  docs/architecture/COLD_START_BUDGETS.md for the
  table): profile_load medians 0.234-3.311 ms -> base 50 ms;
  core_construction medians 10.160-91.026 ms -> base
  500 ms; engine_load medians 70.053-117.855 ms ->
  base 1000 ms; filed TTFT medians 95.231-199.084 ms
  against the 2550 ms desktop_cpu total. Each base carries roughly an order of
  magnitude of headroom over the measured fixture
  reality — deliberately, because this VM is shared
  and walls measured under contention are not
  baselines (TF-033's standing warning, Aesir D-14's
  same lesson).
- ``first_token_base_is_campaign_set`` — the fixture's
  first chunk measures ~0.01 ms (no model computes
  anything), so a fixture-derived first_token base
  would be a flaky sub-millisecond cap pretending to
  be a model number. The first_token base is instead
  campaign-set at TF-033's generation base, 1000 ms,
  stated as a ceiling for the shape, not a prediction
  of any model's TTFT.
- ``vm_profile_enforced_real_wall_clock`` — where the
  VM allows means exactly this: the VM profile
  (``desktop_cpu``, TF-033's ``VM_PROFILE_ID``,
  re-verified as what this machine auto-selects) gets
  real wall-clock enforcement in tests — a live
  ``measure_cold_start`` must land inside every stage
  budget and the TTFT total.
- ``other_profiles_simulation_only`` — the other five
  profiles' budgets are filed data; their filed rows
  are VM timings of *simulated* boots (hardware from
  each profile's JSON, run on this host), judged
  against their scaled budgets as data. Enforcement
  on their real hardware is NOT PROVEN: TF-082 owns
  the real Pi, TF-175 the laptop line.
- ``deterministic_cap_proof_scripted`` — wall timing
  alone cannot prove a cap catches a slow start
  without flakiness, so the verdict is also proven
  deterministically: ``scripted_measurement`` builds a
  measurement from stated stage delays with no clock
  at all, and ``measure_cold_start`` accepts an
  injectable clock (``FakeClock``) that a scripted
  fixture backend advances by exactly its stated
  first-chunk delay. A deliberately slow first token
  is caught BY NAME (stage + profile in the verdict),
  a fast one passes.
- ``live_model_ttft_not_proven`` — no real model
  exists on this VM, so a live model's
  time-to-first-token is NOT PROVEN for every
  profile, stated everywhere it matters. TF-082 /
  TF-175 own the roads that could prove one, and
  TF-086's truth table must carry this line.

Proof grade FIXTURE/SIMULATION (real profile loads,
core constructions and no-model engine loads on this
VM, fixture streaming backend, medians of 3 runs per
profile) — a real model's cold start, a GGUF file
load, and every non-VM profile's real hardware are
NOT PROVEN, never claimed.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from thoughtforge.cognition.latency_budgets import PROFILE_FACTORS, VM_PROFILE_ID
from thoughtforge.utils.config import (
    list_hardware_profiles,
    load_hardware_profile,
)

COLD_START_FORMAT = "thoughtforge-cold-start-budgets/1"

#: The cold-start stages, in boot order. TTFT is their
#: sum by construction.
STAGES: tuple[str, ...] = (
    "profile_load", "core_construction", "engine_load", "first_token",
)

# Campaign-set base ceilings for the VM profile
# (desktop_cpu), in milliseconds. Derivation in the
# module docstring + the architecture doc: medians of
# 3 simulated boots per profile measured this tick on
# this VM, each base carrying roughly an order of
# magnitude of headroom (first_token's base is
# campaign-set at TF-033's generation base — the
# fixture measures ~0.01 ms and proves only the shape).
BASE_PROFILE_LOAD_MS = 50
BASE_CORE_CONSTRUCTION_MS = 500
BASE_ENGINE_LOAD_MS = 1000
BASE_FIRST_TOKEN_MS = 1000

_BASES: dict[str, int] = {
    "profile_load": BASE_PROFILE_LOAD_MS,
    "core_construction": BASE_CORE_CONSTRUCTION_MS,
    "engine_load": BASE_ENGINE_LOAD_MS,
    "first_token": BASE_FIRST_TOKEN_MS,
}

# ── The rules ───────────────────────────────────────────────────────────────

COLD_START_RULES: tuple[dict, ...] = (
    {"id": "cold_start_defined_in_parts",
     "rule": "cold start is four stages in boot order — profile "
             "load, core construction, no-model engine load, "
             "first token (first streaming chunk through a "
             "fixture backend); time-to-first-token is their "
             "sum by construction, never an independent number"},
    {"id": "budgets_reuse_tf033_factors",
     "rule": "per-profile budget scaling reuses TF-033's "
             "PROFILE_FACTORS imported live; no second factor "
             "table exists to drift"},
    {"id": "base_from_vm_measurements_with_headroom",
     "rule": "the desktop_cpu bases are set from this tick's VM "
             "measurements (MEASURED_ROWS, medians of 3 simulated "
             "boots per profile) with roughly an order of "
             "magnitude of headroom, stated as such — this VM is "
             "shared and walls under contention are not baselines"},
    {"id": "first_token_base_is_campaign_set",
     "rule": "the first_token base (1000 ms) is campaign-set at "
             "TF-033's generation base: the fixture's first chunk "
             "measures ~0.01 ms and proves only the shape, so a "
             "fixture-derived cap would pretend to be a model "
             "number"},
    {"id": "total_is_sum_of_parts",
     "rule": "a budget's total and a measurement's TTFT are each "
             "the sum of their four stage values by construction"},
    {"id": "vm_profile_enforced_real_wall_clock",
     "rule": "the VM profile (desktop_cpu, TF-033's VM_PROFILE_ID) "
             "is enforced in tests with real wall-clock "
             "measurements against the generous caps"},
    {"id": "other_profiles_simulation_only",
     "rule": "the other profiles' budgets are filed data judged "
             "against VM timings of simulated boots (hardware "
             "from each profile's JSON); real-hardware cold start "
             "is NOT PROVEN (TF-082 owns the real Pi, TF-175 the "
             "laptop line)"},
    {"id": "streaming_first_chunk_vs_complete",
     "rule": "first chunk and generation complete are measured "
             "distinctly on the same stream run: distinct for a "
             "multi-chunk (native-shaped) stream, equal by "
             "construction for a simulated backend whose single "
             "chunk is a completed generate() (TF-072's shapes)"},
    {"id": "deterministic_cap_proof_scripted",
     "rule": "an over-budget cold start is proven caught by name "
             "(stage + profile) without flaky wall timing: "
             "scripted measurements with no clock, and an "
             "injectable FakeClock a scripted fixture backend "
             "advances by exactly its stated delay"},
    {"id": "live_model_ttft_not_proven",
     "rule": "no real model exists on this VM (no GGUF in-tree, "
             "llama_cpp absent): a live model's TTFT, a GGUF "
             "file load, and real-hardware cold starts are NOT "
             "PROVEN for every profile; TF-082/TF-175 own those "
             "roads and TF-086's truth table must carry the line"},
)

# ── Filed measured rows (this tick, VM) ─────────────────────────────────────
# Medians of MEASURED_RUNS simulated boots per profile,
# measured live by measure_cold_start on this VM this
# tick. Filed as the tick's record — timings vary run to
# run on a shared VM, so the checker judges these rows
# for shape, positivity and within-budget truth, and
# re-measures the VM profile live; it never demands a
# re-measurement reproduce a filed millisecond.

MEASURED_RUNS = 3

MEASURED_ROWS: dict[str, dict[str, float]] = {
    "desktop_cpu": {"profile_load_ms": 3.311, "core_construction_ms": 91.026,
                    "engine_load_ms": 87.516, "first_token_ms": 0.007,
                    "generation_complete_ms": 0.01, "first_answer_ms": 0.012,
                    "ttft_ms": 181.86},
    "desktop_gpu": {"profile_load_ms": 2.846, "core_construction_ms": 73.278,
                    "engine_load_ms": 82.603, "first_token_ms": 0.005,
                    "generation_complete_ms": 0.008, "first_answer_ms": 0.012,
                    "ttft_ms": 158.732},
    "phone_low": {"profile_load_ms": 0.27, "core_construction_ms": 29.883,
                    "engine_load_ms": 70.053, "first_token_ms": 0.005,
                    "generation_complete_ms": 0.008, "first_answer_ms": 0.011,
                    "ttft_ms": 100.211},
    "pi_5": {"profile_load_ms": 0.294, "core_construction_ms": 11.47,
                    "engine_load_ms": 89.787, "first_token_ms": 0.005,
                    "generation_complete_ms": 0.008, "first_answer_ms": 0.012,
                    "ttft_ms": 101.556},
    "pi_zero": {"profile_load_ms": 0.234, "core_construction_ms": 80.989,
                    "engine_load_ms": 117.855, "first_token_ms": 0.006,
                    "generation_complete_ms": 0.01, "first_answer_ms": 0.014,
                    "ttft_ms": 199.084},
    "server_gpu": {"profile_load_ms": 0.729, "core_construction_ms": 10.16,
                    "engine_load_ms": 84.338, "first_token_ms": 0.004,
                    "generation_complete_ms": 0.008, "first_answer_ms": 0.011,
                    "ttft_ms": 95.231},
}

_MEASURED_KEYS = (
    "profile_load_ms", "core_construction_ms", "engine_load_ms",
    "first_token_ms", "generation_complete_ms", "first_answer_ms",
    "ttft_ms",
)

# ── Budgets ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColdStartBudget:
    """One profile's cold-start ceilings, in milliseconds."""

    profile_id: str
    profile_load_ms: int
    core_construction_ms: int
    engine_load_ms: int
    first_token_ms: int

    @property
    def total_ms(self) -> int:
        """The TTFT ceiling: the stage budgets' sum, by
        construction — never an independent number."""
        return (self.profile_load_ms + self.core_construction_ms
                + self.engine_load_ms + self.first_token_ms)

    def stage_budget(self, stage: str) -> int:
        if stage == "profile_load":
            return self.profile_load_ms
        if stage == "core_construction":
            return self.core_construction_ms
        if stage == "engine_load":
            return self.engine_load_ms
        if stage == "first_token":
            return self.first_token_ms
        if stage == "total":
            return self.total_ms
        raise ValueError(f"unknown cold-start stage: {stage!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_load_ms": self.profile_load_ms,
            "core_construction_ms": self.core_construction_ms,
            "engine_load_ms": self.engine_load_ms,
            "first_token_ms": self.first_token_ms,
            "total_ms": self.total_ms,
        }


def budget_for_profile(profile_id: str) -> ColdStartBudget:
    """Resolve one shipped profile's cold-start budget: the
    campaign bases scaled by TF-033's stated factor for the
    profile. An unknown profile — or a shipped profile with
    no TF-033 factor — raises ValueError naming the valid
    ids: a budget never silently defaults."""
    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile: {profile_id!r} "
            f"(valid: {', '.join(valid)})"
        )
    if profile_id not in PROFILE_FACTORS:
        raise ValueError(
            f"no TF-033 factor recorded for profile: {profile_id!r}"
        )
    factor = PROFILE_FACTORS[profile_id]
    return ColdStartBudget(
        profile_id=profile_id,
        profile_load_ms=int(round(BASE_PROFILE_LOAD_MS * factor)),
        core_construction_ms=int(round(BASE_CORE_CONSTRUCTION_MS * factor)),
        engine_load_ms=int(round(BASE_ENGINE_LOAD_MS * factor)),
        first_token_ms=int(round(BASE_FIRST_TOKEN_MS * factor)),
    )


def all_budgets() -> dict[str, ColdStartBudget]:
    """Every shipped profile's cold-start budget, keyed by id."""
    return {pid: budget_for_profile(pid) for pid in list_hardware_profiles()}


# ── Measurements ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColdStartMeasurement:
    """One cold start's measured stage times, in milliseconds."""

    profile_id: str
    profile_load_ms: float
    core_construction_ms: float
    engine_load_ms: float
    first_token_ms: float
    generation_complete_ms: float = 0.0
    first_answer_ms: float = 0.0
    first_chunk: str = ""
    answer_text: str = ""

    @property
    def ttft_ms(self) -> float:
        """Time-to-first-token: the four stage times' sum,
        by construction."""
        return (self.profile_load_ms + self.core_construction_ms
                + self.engine_load_ms + self.first_token_ms)

    def stage_measurements(self) -> dict[str, float]:
        return {
            "profile_load": self.profile_load_ms,
            "core_construction": self.core_construction_ms,
            "engine_load": self.engine_load_ms,
            "first_token": self.first_token_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_load_ms": self.profile_load_ms,
            "core_construction_ms": self.core_construction_ms,
            "engine_load_ms": self.engine_load_ms,
            "first_token_ms": self.first_token_ms,
            "generation_complete_ms": self.generation_complete_ms,
            "first_answer_ms": self.first_answer_ms,
            "ttft_ms": self.ttft_ms,
            "first_chunk": self.first_chunk,
            "answer_text": self.answer_text,
        }


def scripted_measurement(
    profile_id: str, delays_ms: dict[str, float],
) -> ColdStartMeasurement:
    """A measurement built from stated stage delays, with no
    clock at all — the deterministic cap proof's raw material.
    Missing stages measure 0; ``generation_complete_ms``
    defaults to the first_token delay (a simulated backend's
    single-chunk shape) and ``first_answer_ms`` to it too,
    unless stated under those keys."""
    unknown = set(delays_ms) - set(STAGES) - {
        "generation_complete", "first_answer"}
    if unknown:
        raise ValueError(f"unknown scripted stages: {sorted(unknown)}")
    first_token = float(delays_ms.get("first_token", 0.0))
    return ColdStartMeasurement(
        profile_id=profile_id,
        profile_load_ms=float(delays_ms.get("profile_load", 0.0)),
        core_construction_ms=float(delays_ms.get("core_construction", 0.0)),
        engine_load_ms=float(delays_ms.get("engine_load", 0.0)),
        first_token_ms=first_token,
        generation_complete_ms=float(
            delays_ms.get("generation_complete", first_token)),
        first_answer_ms=float(delays_ms.get("first_answer", first_token)),
    )


# ── Verdicts ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColdStageVerdict:
    """One stage's measured time against its budget."""

    stage: str
    measured_ms: float
    budget_ms: int

    @property
    def within(self) -> bool:
        return self.measured_ms <= self.budget_ms


@dataclass(frozen=True)
class ColdStartVerdict:
    """A whole cold start's verdict against one budget."""

    profile_id: str
    stages: tuple[ColdStageVerdict, ...]

    @property
    def within_budget(self) -> bool:
        return all(stage.within for stage in self.stages)

    @property
    def overruns(self) -> tuple[ColdStageVerdict, ...]:
        """The stages that broke their budget, in boot order
        (the total, when judged, comes last)."""
        return tuple(stage for stage in self.stages if not stage.within)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "within_budget": self.within_budget,
            "stages": [
                {"stage": s.stage, "measured_ms": s.measured_ms,
                 "budget_ms": s.budget_ms, "within": s.within}
                for s in self.stages
            ],
        }


def check_cold_start(
    measurement: ColdStartMeasurement, budget: ColdStartBudget,
) -> ColdStartVerdict:
    """Judge one cold start against ``budget``: per-stage
    verdicts in boot order plus a ``total`` verdict against
    the budget's summed TTFT ceiling. Pure — the same
    measurement and budget always give the same verdict,
    which is what makes the scripted cap proof possible."""
    if measurement.profile_id != budget.profile_id:
        raise ValueError(
            f"measurement profile {measurement.profile_id!r} != "
            f"budget profile {budget.profile_id!r}"
        )
    measured = measurement.stage_measurements()
    verdicts = [
        ColdStageVerdict(
            stage=stage, measured_ms=measured[stage],
            budget_ms=budget.stage_budget(stage))
        for stage in STAGES
    ]
    verdicts.append(ColdStageVerdict(
        stage="total", measured_ms=measurement.ttft_ms,
        budget_ms=budget.total_ms))
    return ColdStartVerdict(
        profile_id=budget.profile_id, stages=tuple(verdicts))


# ── The fixture backend + injectable clock ──────────────────────────────────

class FakeClock:
    """An injectable clock for deterministic measurement:
    calling it returns its current (fake) time in seconds;
    only an explicit ``advance_ms`` moves it. Real work
    measured against it reads 0 — that is the point: the
    scripted delay is the only time that passes."""

    def __init__(self) -> None:
        self._now_s = 0.0

    def __call__(self) -> float:
        return self._now_s

    def advance_ms(self, ms: float) -> None:
        self._now_s += ms / 1000.0


#: The fixture stream's chunks: a synthetic answer in
#: TF-017's world, no canon. Three chunks, so the first
#: chunk is distinct from the completed generation.
FIXTURE_CHUNKS: tuple[str, ...] = (
    "The ", "ferry ", "crosses in eleven minutes.")


class FixtureStreamingBackend:
    """A stub backend in TF-065's shape (duck-typed like
    the tree's other fixture backends): streams
    FIXTURE_CHUNKS, answers their join from generate().

    ``first_chunk_delay_ms`` / ``answer_delay_ms`` are
    scripted delays: with a ``clock`` (FakeClock) they
    advance it by exactly the stated amount; without
    one they sleep in real time. Zero delay (the
    default) does neither."""

    def __init__(
        self, *, first_chunk_delay_ms: float = 0.0,
        answer_delay_ms: float = 0.0, clock: FakeClock | None = None,
    ) -> None:
        self._first_chunk_delay_ms = float(first_chunk_delay_ms)
        self._answer_delay_ms = float(answer_delay_ms)
        self._clock = clock

    def _delay(self, ms: float) -> None:
        if ms <= 0:
            return
        if self._clock is not None:
            self._clock.advance_ms(ms)
        else:
            time.sleep(ms / 1000.0)

    def generate_stream(self, request: Any):
        self._delay(self._first_chunk_delay_ms)
        yield from FIXTURE_CHUNKS

    def generate(self, request: Any) -> Any:
        from thoughtforge.inference.unified_backend import GenerationResponse

        self._delay(self._answer_delay_ms)
        text = "".join(FIXTURE_CHUNKS)
        return GenerationResponse(
            text=text, tokens_generated=len(text.split()),
            backend_used="fixture:cold-start")

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "fixture:cold-start"


# ── Live measurement ────────────────────────────────────────────────────────

def measure_cold_start(
    profile_id: str, *, backend: Any = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ColdStartMeasurement:
    """Measure one simulated cold start for ``profile_id``
    on this machine, stage by stage, with ``clock``.

    The boot is TF-069's simulation: the profile JSON is
    the hardware truth; the core is constructed in temp
    dirs on that profile; the engine is a real no-model
    TurboQuantEngine load; the first token and first
    answer come through ``backend`` (the instant fixture
    backend by default). Unknown profiles raise
    ValueError, as everywhere in this tree."""
    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile: {profile_id!r} "
            f"(valid: {', '.join(valid)})"
        )
    if backend is None:
        backend = FixtureStreamingBackend()
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.inference.turboquant import TurboQuantEngine
    from thoughtforge.inference.unified_backend import GenerationRequest

    request = GenerationRequest(prompt="What is the ferry?")
    with tempfile.TemporaryDirectory(prefix="tf079-") as tmp:
        root = Path(tmp)
        t0 = clock()
        load_hardware_profile(profile_id)
        t1 = clock()
        ThoughtForgeCore(
            config={"conversation": {"profile_id": profile_id}},
            memory_dir=root / "memory", db_path=root / "knowledge.db",
            backend=backend,
        )
        t2 = clock()
        engine = TurboQuantEngine(profile_id=profile_id)
        try:
            engine.load()
        finally:
            engine.unload()
        t3 = clock()
        stream = backend.generate_stream(request)
        first_chunk = next(stream)
        t4 = clock()
        answer_text = first_chunk + "".join(stream)
        t5 = clock()
        answer_start = clock()
        response = backend.generate(request)
        t6 = clock()
    return ColdStartMeasurement(
        profile_id=profile_id,
        profile_load_ms=(t1 - t0) * 1000.0,
        core_construction_ms=(t2 - t1) * 1000.0,
        engine_load_ms=(t3 - t2) * 1000.0,
        first_token_ms=(t4 - t3) * 1000.0,
        generation_complete_ms=(t5 - t3) * 1000.0,
        first_answer_ms=(t6 - answer_start) * 1000.0,
        first_chunk=first_chunk,
        answer_text=response.text or answer_text,
    )


def median_cold_start(
    profile_id: str, runs: int = MEASURED_RUNS,
) -> ColdStartMeasurement:
    """The median cold start over ``runs`` live simulated
    boots — the filed rows' measuring instrument."""
    rows = [measure_cold_start(profile_id) for _ in range(runs)]
    fields = (
        "profile_load_ms", "core_construction_ms", "engine_load_ms",
        "first_token_ms", "generation_complete_ms", "first_answer_ms",
    )
    medians = {
        f: float(statistics.median(getattr(r, f) for r in rows))
        for f in fields
    }
    return ColdStartMeasurement(
        profile_id=profile_id, first_chunk=rows[0].first_chunk,
        answer_text=rows[0].answer_text, **medians)


# ── Export ──────────────────────────────────────────────────────────────────

def cold_start_as_data() -> dict[str, Any]:
    return {
        "format": COLD_START_FORMAT,
        "stages": list(STAGES),
        "rules": [dict(r) for r in COLD_START_RULES],
        "factors": dict(PROFILE_FACTORS),
        "vm_profile_id": VM_PROFILE_ID,
        "bases_ms": dict(_BASES),
        "budgets": {pid: b.to_dict() for pid, b in all_budgets().items()},
        "measured_runs": MEASURED_RUNS,
        "measured_rows": {pid: dict(row) for pid, row in MEASURED_ROWS.items()},
        "live_proof": "not_proven",
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def cold_start_problems() -> list[str]:
    """The filed estate judged against the live tree."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in COLD_START_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("COLD_START_RULES ids are not unique")
    if STAGES != ("profile_load", "core_construction",
                  "engine_load", "first_token"):
        problems.append(f"STAGES drifted: {STAGES}")

    # Budgets: exactly the shipped profiles, TF-033's
    # factors applied to the filed bases, totals as sums.
    budgets = all_budgets()
    if set(budgets) != set(list_hardware_profiles()):
        problems.append("budgets do not cover exactly the shipped profiles")
    if set(budgets) != set(PROFILE_FACTORS):
        problems.append("budgets and TF-033 PROFILE_FACTORS disagree "
                        "on the profile set")
    for pid, budget in budgets.items():
        factor = PROFILE_FACTORS[pid]
        for stage, base in _BASES.items():
            if budget.stage_budget(stage) != int(round(base * factor)):
                problems.append(f"{pid}: {stage} budget is not "
                                f"base x TF-033 factor")
        if budget.total_ms != sum(
                budget.stage_budget(s) for s in STAGES):
            problems.append(f"{pid}: total is not the stage sum")

    # Filed measured rows: every profile, every key,
    # positive, TTFT the stage sum, complete >= first
    # token, and every filed median inside its budget.
    if set(MEASURED_ROWS) != set(budgets):
        problems.append("measured rows do not cover exactly "
                        "the shipped profiles")
    for pid, row in MEASURED_ROWS.items():
        if tuple(row) != _MEASURED_KEYS:
            problems.append(f"{pid}: measured row keys drifted")
            continue
        if any(row[k] <= 0 for k in _MEASURED_KEYS):
            problems.append(f"{pid}: a filed median is not positive")
        stage_sum = sum(row[f"{s}_ms"] for s in STAGES)
        if abs(row["ttft_ms"] - stage_sum) > 0.01:
            problems.append(f"{pid}: filed ttft {row['ttft_ms']} != "
                            f"stage sum {stage_sum}")
        if row["generation_complete_ms"] < row["first_token_ms"] - 1e-9:
            problems.append(f"{pid}: generation complete precedes "
                            f"its first token")
        budget = budgets[pid]
        for stage in STAGES:
            if row[f"{stage}_ms"] > budget.stage_budget(stage):
                problems.append(f"{pid}: filed {stage} median "
                                f"{row[f'{stage}_ms']} over its budget")
        if row["ttft_ms"] > budget.total_ms:
            problems.append(f"{pid}: filed ttft over its total budget")

    # The VM profile is TF-033's, and this machine
    # auto-selects it (TF-033's pin, re-verified here —
    # the wall-clock enforcement binds to it).
    if VM_PROFILE_ID != "desktop_cpu":
        problems.append("VM_PROFILE_ID is not desktop_cpu")
    from thoughtforge.inference.profiles import ProfileSelector
    if ProfileSelector().auto_select().profile_id != VM_PROFILE_ID:
        problems.append("this machine no longer auto-selects the "
                        "VM profile — enforcement would bind wrongly")

    # Live-proof honesty: the checker itself re-probes
    # the grounds (no GGUF in the sparse tree, no
    # llama_cpp here).
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    if list(root.rglob("*.gguf")):
        problems.append("a GGUF now exists in-tree — the "
                        "not_proven filing must be revisited")
    if importlib.util.find_spec("llama_cpp") is not None:
        problems.append("llama_cpp is now importable — the "
                        "not_proven filing must be revisited")
    return problems


def cold_start_live_problems() -> list[str]:
    """The budgets re-derived behaviourally on named cases."""
    problems: list[str] = []

    # Deterministic cap proof, no clock: a slow first
    # token is caught by name (stage + profile), a fast
    # one passes; a slow core construction likewise.
    budget = budget_for_profile(VM_PROFILE_ID)
    fast = check_cold_start(
        scripted_measurement(VM_PROFILE_ID, {
            "profile_load": 1.0, "core_construction": 10.0,
            "engine_load": 20.0, "first_token": 30.0}), budget)
    if not fast.within_budget:
        problems.append(f"scripted fast cold start failed: {fast.to_dict()}")
    slow = check_cold_start(
        scripted_measurement(VM_PROFILE_ID, {
            "profile_load": 1.0, "core_construction": 10.0,
            "engine_load": 20.0, "first_token": 60_000.0}), budget)
    if slow.within_budget or [s.stage for s in slow.overruns] != [
            "first_token", "total"]:
        problems.append(f"scripted slow first token not caught by name: "
                        f"{slow.to_dict()}")
    if slow.profile_id != VM_PROFILE_ID:
        problems.append("the verdict does not carry its profile")

    # Injectable clock: the scripted fixture delay is
    # exactly the measured first token.
    clock = FakeClock()
    backend = FixtureStreamingBackend(
        first_chunk_delay_ms=1234.0, clock=clock)
    measured = measure_cold_start(
        VM_PROFILE_ID, backend=backend, clock=clock)
    if abs(measured.first_token_ms - 1234.0) > 1e-6:
        problems.append(f"fake-clock first token {measured.first_token_ms} "
                        f"!= scripted 1234.0")
    verdict = check_cold_start(measured, budget)
    if verdict.within_budget or "first_token" not in [
            s.stage for s in verdict.overruns]:
        problems.append("fake-clock slow first token not caught")

    # Streaming shape: the fixture's first chunk is
    # distinct from its completed answer.
    live = measure_cold_start(VM_PROFILE_ID)
    if live.first_chunk != FIXTURE_CHUNKS[0] \
            or live.answer_text != "".join(FIXTURE_CHUNKS):
        problems.append("fixture stream shape drifted (first chunk "
                        "is not distinct from the whole answer)")
    if live.generation_complete_ms < live.first_token_ms - 1e-9:
        problems.append("live generation complete precedes first token")

    # Real wall-clock enforcement on the VM profile.
    live_verdict = check_cold_start(live, budget)
    if not live_verdict.within_budget:
        problems.append(f"live VM cold start over budget: "
                        f"{live_verdict.to_dict()} vs {live.to_dict()}")
    return problems


__all__ = [
    "BASE_CORE_CONSTRUCTION_MS",
    "BASE_ENGINE_LOAD_MS",
    "BASE_FIRST_TOKEN_MS",
    "BASE_PROFILE_LOAD_MS",
    "COLD_START_FORMAT",
    "COLD_START_RULES",
    "FIXTURE_CHUNKS",
    "MEASURED_ROWS",
    "MEASURED_RUNS",
    "STAGES",
    "ColdStartBudget",
    "ColdStartMeasurement",
    "ColdStartVerdict",
    "ColdStageVerdict",
    "FakeClock",
    "FixtureStreamingBackend",
    "all_budgets",
    "budget_for_profile",
    "check_cold_start",
    "cold_start_as_data",
    "cold_start_live_problems",
    "cold_start_problems",
    "measure_cold_start",
    "median_cold_start",
    "scripted_measurement",
]
