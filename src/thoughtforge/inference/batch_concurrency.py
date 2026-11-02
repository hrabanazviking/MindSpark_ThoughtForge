"""TF-081 — Batch & concurrency per profile, measured
honestly: there is no batched generation anywhere in
the tree, and concurrency has exactly one parallel
shape.

Roadmap: "Batch & concurrency per profile, measured
honestly."

Measured start (probed live this tick, VM):

- **No backend or engine exposes a batch.**
  ``UnifiedBackend``'s method surface is generate /
  generate_stream / health_check / backend_name /
  count_tokens / capabilities — one request in, one
  response out, no batch entry on any leg (the class's
  method list is pinned by this slice's checker). The
  one true batch in the whole inference tree is not
  generation at all: ``onnx_export``'s tokenisation
  loop (``batch_size`` chunks over texts) in the
  embedding-export path (TF-070's estate).
- **The tree's one multi-generation,
  ``TurboQuantEngine.generate_drafts``, is a
  sequential loop.** Its source is a plain
  ``for i in range(count): self.generate(...)`` —
  draft *i+1* starts only after draft *i* returns.
  The engine module imports no threading and holds no
  lock: concurrent calls on ONE engine are an
  unguarded, unproven shape, not a supported one.
- **Concurrency at the turn level is TF-035's
  contract, and TF-035 deferred its timing here.**
  One core serialises its turns under a turn lock;
  parallel conversations run on parallel cores.
  TF-035 proved the isolation with stub backends and
  stated plainly: "Two *live models* generating
  concurrently (and what that does to latency
  budgets) is TF-081's measurement, not this slice's
  claim." This slice makes that measurement — at
  fixture grade, because no live model exists on
  this VM (no GGUF in-tree, ``llama_cpp`` absent):
  the live-model share stays NOT PROVEN, now with
  the fixture panel standing where no measurement
  stood before.
- **Per profile, nothing declares a concurrency or
  batch knob.** No shipped profile JSON carries a
  concurrency, batch, or thread key (checker-pinned
  over all six); the engine's ``n_threads`` defaults
  to None (auto) on every profile; the only
  per-profile number that shapes multi-generation at
  all is ``draft_count`` (pi_zero 1, phone_low 2,
  pi_5 / desktop_cpu / desktop_gpu 3, server_gpu 5 —
  read live by this slice's checker). What a profile
  changes in the panel below is its core's resolved
  configuration riding along (context, drafts,
  refinement), not a declared parallelism. D-BATCH-2
  in the TF-016 docket, owned by TF-153/TF-086.
- **The server-shaped backends differ, and the
  differences are filed, not smoothed.** Aesir
  serialises server-side behind one resident model
  (TF-074's ledger: exactly one resident generates;
  requests are serialized). The HTTP backends
  (ollama / lmstudio / huggingface) are stateless
  per call on the client — each generate is its own
  ``requests`` call, no session, no client lock — so
  concurrent client calls are possible by
  construction; whether any given server actually
  generates concurrently is the server's truth and
  is NOT PROVEN here (no live servers on this VM).

THE PANEL (``measure_batch_concurrency`` /
``MEASURED_ROWS``, format
``thoughtforge-batch-concurrency/1``): the same
batch on every profile — ``BATCH_SIZE`` = 4 distinct
synthetic prompts (TF-017's world), each generation
carrying a scripted ``DELAY_MS`` = 50 ms delay in an
instrumented fixture backend (``CountingBackend``,
which records the maximum generations ever
simultaneously in flight) — run four ways:

1. ``sequential`` — the batch one turn at a time on
   one core;
2. ``parallel_cores`` — one core per prompt, all
   threads at once (TF-035's parallel shape);
3. ``same_core`` — all threads on ONE core (TF-035's
   serialised shape: the turn lock queues them);
4. ``drafts`` — the real ``TurboQuantEngine`` on the
   profile, its ``_model`` a stub llama carrying the
   same scripted delay and recorder,
   ``generate_drafts`` at the profile's own
   ``draft_count``.

Filed rows are medians of ``MEASURED_RUNS`` = 3 full
panels per profile measured live on this VM this
tick. What the panel proves is the *orchestration
shape* per profile: parallel cores genuinely overlap
(max in flight 4 of 4 in every filed median row;
speedups 2.375–5.353×), the same core never overlaps
(max in flight 1 — the lock, re-proven at batch
scale; its wall ratio to sequential is 0.975–1.071,
the serialised shape measured, not asserted), and
drafts never overlap (max in flight 1 — the
sequential loop, measured through the real engine;
server_gpu's 5 drafts cost 458.807 ms of scripted
delay + engine overhead, pi_zero's single draft
65.773 ms).
The millisecond rows are VM fixture timings: core
turn overhead (~50 ms a turn beyond the scripted
delay on this shared VM) dominates them, they vary
run to run under contention, and the checker judges
them for shape and ordering only — never as a
prediction of any model's throughput. A live
model's concurrent throughput, a real server's
concurrency, and every profile's real hardware
remain NOT PROVEN (TF-175's laptop line and TF-082's
Pi own those roads; TF-086's truth table carries
this line).

Proof grade FIXTURE (instrumented fixture backends
through real cores, a stub llama through the real
engine, medians of 3 panels per profile, VM) — live
concurrent generation NOT PROVEN, never claimed.
"""

from __future__ import annotations

import statistics
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thoughtforge.utils.config import list_hardware_profiles

BATCH_FORMAT = "thoughtforge-batch-concurrency/1"

# ── The panel's fixed shape (comparable across profiles) ────────────────────

#: Batch size: the same four prompts on every profile,
#: so rows compare orchestration, not workload.
BATCH_SIZE = 4

#: The scripted per-generation delay, in milliseconds.
#: It stands in for a model's compute time: long enough
#: that overlap (or its absence) is measurable, short
#: enough that a 6-profile × 3-run panel stays cheap.
DELAY_MS = 50.0

MEASURED_RUNS = 3

#: The panel's prompts: distinct synthetic asks in
#: TF-017's world, no canon. Each must extract to a
#: distinct request so a crossed answer is visible.
PANEL_PROMPTS: tuple[str, ...] = (
    "Tell me about the ferry crossing at Lantern Hollow",
    "Tell me about the lighthouse keeper duties at dawn",
    "Tell me about the harbour winter market stalls",
    "Tell me about the willow narrows river gauge",
)

_FILLER = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)

# ── The concurrency shapes, filed as data ───────────────────────────────────

CONCURRENCY_SHAPES: tuple[dict[str, str], ...] = (
    {"surface": "unified_backend_contract",
     "shape": "single_request_only",
     "evidence": "UnifiedBackend's methods are generate / "
                 "generate_stream / health_check / backend_name / "
                 "count_tokens / capabilities — no batch entry "
                 "exists on the contract any backend binds to"},
    {"surface": "turboquant_engine_drafts",
     "shape": "sequential_loop",
     "evidence": "generate_drafts is a for-loop over "
                 "self.generate(...); the engine module imports "
                 "no threading and holds no lock, so concurrent "
                 "calls on one engine are unguarded and unproven"},
    {"surface": "one_core_turns",
     "shape": "serialised_by_turn_lock",
     "evidence": "TF-035: ThoughtForgeCore holds _turn_lock; "
                 "turns on one instance queue, never overlap, "
                 "never bleed"},
    {"surface": "parallel_cores",
     "shape": "parallel_isolated",
     "evidence": "TF-035: one core per conversation, run on "
                 "parallel threads, is the tree's one parallel "
                 "shape; isolation is TF-035's proven contract"},
    {"surface": "aesir_server",
     "shape": "serialised_server_side",
     "evidence": "TF-074's ledger: exactly one resident model "
                 "generates and requests are serialized on the "
                 "server, whatever the client does"},
    {"surface": "http_backends",
     "shape": "stateless_per_call_client",
     "evidence": "ollama / lmstudio / huggingface each make "
                 "their own requests call per generate — no "
                 "session, no client lock; concurrent client "
                 "calls are possible by construction, server-side "
                 "concurrency is the server's truth, NOT PROVEN"},
    {"surface": "onnx_tokenisation",
     "shape": "batched_not_generation",
     "evidence": "the inference tree's one true batch: "
                 "onnx_export tokenises texts in batch_size "
                 "chunks — an embedding-export step, not "
                 "generation"},
)

# ── The rules ───────────────────────────────────────────────────────────────

BATCH_RULES: tuple[dict, ...] = (
    {"id": "no_native_batch_generation",
     "rule": "no backend and no engine in the tree exposes "
             "batched generation: the contract is one request "
             "in, one response out; the tree's one batch "
             "(ONNX tokenisation) is an export step, never "
             "generation"},
    {"id": "drafts_are_sequential",
     "rule": "the one multi-generation, generate_drafts, runs "
             "its drafts one at a time in a plain loop — a "
             "profile's draft_count multiplies wall time, it "
             "does not parallelise it (measured: max in flight "
             "1 through the real engine on every profile)"},
    {"id": "parallel_only_across_cores",
     "rule": "parallel generation exists in exactly one shape: "
             "one core per conversation on parallel threads "
             "(TF-035's contract); nothing else in the tree "
             "overlaps two generations"},
    {"id": "same_core_serialised",
     "rule": "threads sharing one core queue under its turn "
             "lock: at batch scale the max in flight stays 1 "
             "and the wall stays sequential-shaped — safe and "
             "correct, never parallel"},
    {"id": "one_engine_concurrent_calls_unproven",
     "rule": "TurboQuantEngine holds no lock and imports no "
             "threading: concurrent generate calls on ONE "
             "engine are unguarded and NOT PROVEN; the "
             "supported concurrent shape is a backend/core "
             "per conversation"},
    {"id": "aesir_serialised_server_side",
     "rule": "against Aesir, client concurrency cannot create "
             "server parallelism: one resident model serves, "
             "serialized (TF-074's ledger) — a batch through "
             "Aesir is a queue by the server's own design"},
    {"id": "http_server_concurrency_not_proven",
     "rule": "the HTTP backends are stateless per call on the "
             "client, so concurrent calls can be issued; what "
             "a given server does with them is NOT PROVEN (no "
             "live servers on this VM)"},
    {"id": "profiles_declare_no_concurrency_knob",
     "rule": "no shipped profile declares a concurrency, "
             "batch, or thread parameter (checker-pinned over "
             "all six JSONs); the engine's n_threads is auto "
             "(None) everywhere, and draft_count is the only "
             "per-profile number shaping multi-generation — "
             "D-BATCH-2, owned by TF-153/TF-086"},
    {"id": "panel_is_comparable_by_construction",
     "rule": "the panel is identical on every profile — same "
             "4 prompts, same scripted 50 ms generation delay, "
             "medians of 3 — so per-profile rows compare "
             "orchestration and resolved configuration, never "
             "different workloads"},
    {"id": "live_concurrency_not_proven",
     "rule": "the panel measures fixture generations through "
             "real cores and the real engine: a live model's "
             "concurrent throughput, a real server's "
             "concurrency, and real-hardware panels are NOT "
             "PROVEN for every profile; TF-082/TF-175 own those "
             "roads and TF-086's truth table must carry this "
             "line"},
)


# ── The instrumented fixture backend ────────────────────────────────────────

class FlightRecorder:
    """Thread-safe in-flight accounting: how many
    generations are running right now, and the most
    ever seen at once. The panel's overlap witness."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    def enter(self) -> None:
        with self._lock:
            self.in_flight += 1
            self.calls += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)

    def exit(self) -> None:
        with self._lock:
            self.in_flight -= 1


class CountingBackend:
    """A UnifiedBackend-shaped fixture (duck-typed like
    the tree's other fixture backends): each generate
    registers with the recorder, sleeps the scripted
    delay, and answers about the prompt's own request
    (TF-035's echo shape, so a crossed answer shows)."""

    def __init__(self, recorder: FlightRecorder,
                 delay_ms: float = DELAY_MS) -> None:
        self._recorder = recorder
        self._delay_s = delay_ms / 1000.0

    def generate(self, request: Any) -> Any:
        from thoughtforge.cognition.coherence import extract_request
        from thoughtforge.inference.unified_backend import GenerationResponse

        self._recorder.enter()
        try:
            time.sleep(self._delay_s)
            req = extract_request(request.prompt)
            text = f"Answer about {req}. {_FILLER}"
            return GenerationResponse(
                text=text, tokens_generated=len(text.split()))
        finally:
            self._recorder.exit()

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf081-counting"


class _StubLlama:
    """The engine's ``_model`` stand-in: a callable
    returning llama-shaped output after the scripted
    delay, registered with the recorder (TF-072's stub
    shape, extended with timing)."""

    def __init__(self, recorder: FlightRecorder,
                 delay_ms: float = DELAY_MS) -> None:
        self._recorder = recorder
        self._delay_s = delay_ms / 1000.0

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self._recorder.enter()
        try:
            time.sleep(self._delay_s)
            return {
                "choices": [{
                    "text": " A lantern draft answer with enough "
                            "words to stand on its own.",
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 9,
                          "total_tokens": 14},
            }
        finally:
            self._recorder.exit()


# ── Measurement ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BatchMeasurement:
    """One profile's panel, as data: four walls and the
    three overlap witnesses."""

    profile_id: str
    batch_size: int
    delay_ms: float
    draft_count: int
    sequential_ms: float
    parallel_ms: float
    same_core_ms: float
    drafts_ms: float
    max_in_flight_parallel: int
    max_in_flight_same_core: int
    max_in_flight_drafts: int

    @property
    def speedup_parallel(self) -> float:
        """Sequential wall / parallel-cores wall."""
        return self.sequential_ms / self.parallel_ms

    @property
    def same_core_ratio(self) -> float:
        """Same-core wall / sequential wall: ~1 is the
        serialised shape (the lock working), not a
        failure."""
        return self.same_core_ms / self.sequential_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "batch_size": self.batch_size,
            "delay_ms": self.delay_ms,
            "draft_count": self.draft_count,
            "sequential_ms": self.sequential_ms,
            "parallel_ms": self.parallel_ms,
            "same_core_ms": self.same_core_ms,
            "drafts_ms": self.drafts_ms,
            "max_in_flight_parallel": self.max_in_flight_parallel,
            "max_in_flight_same_core": self.max_in_flight_same_core,
            "max_in_flight_drafts": self.max_in_flight_drafts,
            "speedup_parallel": round(self.speedup_parallel, 3),
            "same_core_ratio": round(self.same_core_ratio, 3),
        }


def _make_core(profile_id: str, recorder: FlightRecorder,
               root: Path) -> Any:
    from thoughtforge.cognition.core import ThoughtForgeCore

    return ThoughtForgeCore(
        config={"conversation": {"profile_id": profile_id}},
        memory_dir=root / "memory", db_path=root / "knowledge.db",
        backend=CountingBackend(recorder),
    )


def measure_batch_concurrency(profile_id: str) -> BatchMeasurement:
    """Run the panel once for ``profile_id``: a warm-up
    turn first (TF-033's first-turn note — warm-up is
    not measured), then the four legs in filed order.
    Unknown profiles raise ValueError, as everywhere
    in this tree."""
    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile: {profile_id!r} "
            f"(valid: {', '.join(valid)})"
        )
    from thoughtforge.inference.turboquant import TurboQuantEngine

    prompts = list(PANEL_PROMPTS[:BATCH_SIZE])
    with tempfile.TemporaryDirectory(prefix="tf081-") as tmp:
        root = Path(tmp)

        warm = _make_core(profile_id, FlightRecorder(), root / "warm")
        warm.think(prompts[0])

        # 1 — sequential on one core
        rec_seq = FlightRecorder()
        core_seq = _make_core(profile_id, rec_seq, root / "seq")
        t0 = time.perf_counter()
        for prompt in prompts:
            core_seq.think(prompt)
        sequential_ms = (time.perf_counter() - t0) * 1000.0

        # 2 — parallel cores
        rec_par = FlightRecorder()
        cores = [
            _make_core(profile_id, rec_par, root / f"par{i}")
            for i in range(BATCH_SIZE)
        ]
        t0 = time.perf_counter()
        threads = [
            threading.Thread(target=core.think, args=(prompt,))
            for core, prompt in zip(cores, prompts)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        parallel_ms = (time.perf_counter() - t0) * 1000.0

        # 3 — same core, all threads
        rec_same = FlightRecorder()
        core_same = _make_core(profile_id, rec_same, root / "same")
        t0 = time.perf_counter()
        threads = [
            threading.Thread(target=core_same.think, args=(prompt,))
            for prompt in prompts
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        same_core_ms = (time.perf_counter() - t0) * 1000.0

        # 4 — drafts through the real engine
        rec_drafts = FlightRecorder()
        engine = TurboQuantEngine(profile_id=profile_id)
        try:
            engine.load()
            engine._model = _StubLlama(rec_drafts)  # type: ignore[assignment]
            t0 = time.perf_counter()
            draft_set = engine.generate_drafts(prompts[0])
            drafts_ms = (time.perf_counter() - t0) * 1000.0
            draft_count = len(draft_set.drafts)
        finally:
            engine.unload()

    return BatchMeasurement(
        profile_id=profile_id, batch_size=BATCH_SIZE,
        delay_ms=DELAY_MS, draft_count=draft_count,
        sequential_ms=round(sequential_ms, 3),
        parallel_ms=round(parallel_ms, 3),
        same_core_ms=round(same_core_ms, 3),
        drafts_ms=round(drafts_ms, 3),
        max_in_flight_parallel=rec_par.max_in_flight,
        max_in_flight_same_core=rec_same.max_in_flight,
        max_in_flight_drafts=rec_drafts.max_in_flight,
    )


def median_batch_concurrency(
    profile_id: str, runs: int = MEASURED_RUNS,
) -> BatchMeasurement:
    """The median panel over ``runs`` live panels — the
    filed rows' measuring instrument. Walls are field
    medians; draft_count and the overlap witnesses are
    the first run's (the witnesses are integers the
    shape pins, re-derived by the live checker)."""
    rows = [measure_batch_concurrency(profile_id) for _ in range(runs)]
    med = {
        f: float(statistics.median(getattr(r, f) for r in rows))
        for f in ("sequential_ms", "parallel_ms", "same_core_ms",
                  "drafts_ms")
    }
    return BatchMeasurement(
        profile_id=profile_id, batch_size=BATCH_SIZE,
        delay_ms=DELAY_MS, draft_count=rows[0].draft_count,
        max_in_flight_parallel=rows[0].max_in_flight_parallel,
        max_in_flight_same_core=rows[0].max_in_flight_same_core,
        max_in_flight_drafts=rows[0].max_in_flight_drafts,
        **{k: round(v, 3) for k, v in med.items()},
    )


# ── Filed measured rows (this tick, VM) ─────────────────────────────────────
# Medians of MEASURED_RUNS panels per profile, measured
# live by median_batch_concurrency on this VM this
# tick. Timings vary run to run on a shared VM, so the
# checker judges these rows for shape and ordering
# (positive walls, parallel faster than sequential,
# serialised legs at max in flight 1) and re-measures
# the VM profile live; it never demands a re-measurement
# reproduce a filed millisecond.

MEASURED_ROWS: dict[str, dict[str, Any]] = {
    "desktop_cpu": {
        "profile_id": "desktop_cpu", "batch_size": 4, "delay_ms": 50.0,
        "draft_count": 3, "sequential_ms": 390.031, "parallel_ms": 110.311,
        "same_core_ms": 381.038, "drafts_ms": 278.434,
        "max_in_flight_parallel": 4, "max_in_flight_same_core": 1,
        "max_in_flight_drafts": 1, "speedup_parallel": 3.536,
        "same_core_ratio": 0.977},
    "desktop_gpu": {
        "profile_id": "desktop_gpu", "batch_size": 4, "delay_ms": 50.0,
        "draft_count": 3, "sequential_ms": 387.886, "parallel_ms": 163.314,
        "same_core_ms": 407.45, "drafts_ms": 264.458,
        "max_in_flight_parallel": 4, "max_in_flight_same_core": 1,
        "max_in_flight_drafts": 1, "speedup_parallel": 2.375,
        "same_core_ratio": 1.05},
    "phone_low": {
        "profile_id": "phone_low", "batch_size": 4, "delay_ms": 50.0,
        "draft_count": 2, "sequential_ms": 354.313, "parallel_ms": 81.367,
        "same_core_ms": 379.604, "drafts_ms": 182.156,
        "max_in_flight_parallel": 4, "max_in_flight_same_core": 1,
        "max_in_flight_drafts": 1, "speedup_parallel": 4.355,
        "same_core_ratio": 1.071},
    "pi_5": {
        "profile_id": "pi_5", "batch_size": 4, "delay_ms": 50.0,
        "draft_count": 3, "sequential_ms": 392.487, "parallel_ms": 152.691,
        "same_core_ms": 382.569, "drafts_ms": 277.758,
        "max_in_flight_parallel": 4, "max_in_flight_same_core": 1,
        "max_in_flight_drafts": 1, "speedup_parallel": 2.57,
        "same_core_ratio": 0.975},
    "pi_zero": {
        "profile_id": "pi_zero", "batch_size": 4, "delay_ms": 50.0,
        "draft_count": 1, "sequential_ms": 377.347, "parallel_ms": 70.487,
        "same_core_ms": 395.11, "drafts_ms": 65.773,
        "max_in_flight_parallel": 4, "max_in_flight_same_core": 1,
        "max_in_flight_drafts": 1, "speedup_parallel": 5.353,
        "same_core_ratio": 1.047},
    "server_gpu": {
        "profile_id": "server_gpu", "batch_size": 4, "delay_ms": 50.0,
        "draft_count": 5, "sequential_ms": 389.931, "parallel_ms": 84.234,
        "same_core_ms": 395.236, "drafts_ms": 458.807,
        "max_in_flight_parallel": 4, "max_in_flight_same_core": 1,
        "max_in_flight_drafts": 1, "speedup_parallel": 4.629,
        "same_core_ratio": 1.014},
}

_ROW_KEYS = (
    "profile_id", "batch_size", "delay_ms", "draft_count",
    "sequential_ms", "parallel_ms", "same_core_ms", "drafts_ms",
    "max_in_flight_parallel", "max_in_flight_same_core",
    "max_in_flight_drafts", "speedup_parallel", "same_core_ratio",
)


# ── Export ──────────────────────────────────────────────────────────────────

def batch_as_data() -> dict[str, Any]:
    return {
        "format": BATCH_FORMAT,
        "batch_size": BATCH_SIZE,
        "delay_ms": DELAY_MS,
        "measured_runs": MEASURED_RUNS,
        "prompts": list(PANEL_PROMPTS),
        "shapes": [dict(s) for s in CONCURRENCY_SHAPES],
        "rules": [dict(r) for r in BATCH_RULES],
        "measured_rows": {pid: dict(row)
                          for pid, row in MEASURED_ROWS.items()},
        "live_proof": "not_proven",
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def batch_problems() -> list[str]:
    """The filed estate judged against the live tree."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in BATCH_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("BATCH_RULES ids are not unique")
    shape_surfaces = [s["surface"] for s in CONCURRENCY_SHAPES]
    if shape_surfaces != [
            "unified_backend_contract", "turboquant_engine_drafts",
            "one_core_turns", "parallel_cores", "aesir_server",
            "http_backends", "onnx_tokenisation"]:
        problems.append(f"CONCURRENCY_SHAPES drifted: {shape_surfaces}")
    if len(PANEL_PROMPTS) != BATCH_SIZE:
        problems.append("the panel does not carry BATCH_SIZE prompts")

    import inspect
    import json

    from thoughtforge.inference import turboquant as turboquant_module
    from thoughtforge.inference.unified_backend import UnifiedBackend

    # The contract has no batch entry.
    methods = {name for name, _ in inspect.getmembers(
        UnifiedBackend, predicate=inspect.isfunction)}
    if "batch" in " ".join(sorted(methods)) \
            or any("batch" in name for name in methods):
        problems.append(f"UnifiedBackend grew a batch method: {methods}")

    # Drafts are a sequential loop in an unguarded engine.
    engine_src = inspect.getsource(turboquant_module.TurboQuantEngine)
    if "for i in range(count):" not in engine_src \
            or "result = self.generate(prompt, draft_params)" \
            not in engine_src:
        problems.append("generate_drafts is no longer the sequential "
                        "loop this slice measured — re-read it")
    module_src = Path(turboquant_module.__file__).read_text(
        encoding="utf-8")
    if "import threading" in module_src or "threading.Lock" in module_src:
        problems.append("the engine grew threading/a lock — the "
                        "unguarded filing must be re-read")

    # TF-035's turn lock stands in the core.
    from thoughtforge.cognition import core as core_module

    if "self._turn_lock = threading.Lock()" not in inspect.getsource(
            core_module.ThoughtForgeCore):
        problems.append("the core's turn lock drifted — TF-035's "
                        "serialised shape must be re-read")

    # Aesir's serialised resident stands in its ledger.
    from thoughtforge.inference import aesir_backend

    aesir_rules = {r["id"] for r in aesir_backend.AESIR_RULES}
    if "one_resident_model" not in aesir_rules:
        problems.append("Aesir's one-resident rule drifted")

    # The HTTP legs are per-call, session-free.
    inference_dir = Path(turboquant_module.__file__).resolve().parent
    for name in ("ollama_backend.py", "lmstudio_backend.py"):
        src = inference_dir.joinpath(name).read_text(encoding="utf-8")
        if "requests.post" not in src or "Session(" in src:
            problems.append(f"{name}: the per-call shape drifted")

    # The tree's one batch is ONNX tokenisation.
    onnx_src = inference_dir.joinpath("onnx_export.py").read_text(
        encoding="utf-8")
    if "for start in range(0, len(texts), batch_size)" not in onnx_src:
        problems.append("onnx_export's tokenisation batch drifted")

    # Profiles declare no concurrency knob; draft_count
    # is read live and the filed rows carry it.
    from thoughtforge.inference.profiles import resolve_generation_params
    from thoughtforge.utils.config import load_hardware_profile

    profiles_dir = inference_dir.resolve().parents[2] / "hardware_profiles"
    for pid in list_hardware_profiles():
        raw = json.loads(
            profiles_dir.joinpath(f"{pid}.json").read_text(
                encoding="utf-8"))
        flat = json.dumps(raw).lower()
        for knob in ("concurr", "batch", "thread"):
            if knob in flat:
                problems.append(f"{pid}: profile grew a {knob!r} key — "
                                f"D-BATCH-2's filing must be re-read")
        params = resolve_generation_params(load_hardware_profile(pid))
        row = MEASURED_ROWS.get(pid)
        if row is not None and row["draft_count"] != params.draft_count:
            problems.append(f"{pid}: filed draft_count "
                            f"{row['draft_count']} != live "
                            f"{params.draft_count}")

    # Filed rows: shape and ordering only (shared-VM
    # walls are not baselines).
    if set(MEASURED_ROWS) != set(list_hardware_profiles()):
        problems.append("measured rows do not cover exactly "
                        "the shipped profiles")
    for pid, row in MEASURED_ROWS.items():
        if tuple(row) != _ROW_KEYS:
            problems.append(f"{pid}: measured row keys drifted")
            continue
        if row["batch_size"] != BATCH_SIZE or row["delay_ms"] != DELAY_MS:
            problems.append(f"{pid}: the panel shape drifted in the row")
        for wall in ("sequential_ms", "parallel_ms", "same_core_ms",
                     "drafts_ms"):
            if row[wall] <= 0:
                problems.append(f"{pid}: {wall} is not positive")
        if not row["parallel_ms"] < row["sequential_ms"]:
            problems.append(f"{pid}: filed parallel wall is not below "
                            f"the sequential wall")
        if not row["same_core_ms"] > row["parallel_ms"]:
            problems.append(f"{pid}: filed same-core wall is not above "
                            f"the parallel wall")
        if row["max_in_flight_same_core"] != 1:
            problems.append(f"{pid}: same-core max in flight != 1")
        if row["max_in_flight_drafts"] != 1:
            problems.append(f"{pid}: drafts max in flight != 1")
        if not 2 <= row["max_in_flight_parallel"] <= BATCH_SIZE:
            problems.append(f"{pid}: parallel max in flight "
                            f"{row['max_in_flight_parallel']} outside "
                            f"[2, {BATCH_SIZE}]")
        if row["drafts_ms"] < DELAY_MS * row["draft_count"] * 0.8:
            problems.append(f"{pid}: drafts wall below the sequential "
                            f"floor (draft_count x delay x 0.8)")
    return problems


def batch_live_problems() -> list[str]:
    """The panel re-run behaviourally on the VM profile."""
    problems: list[str] = []
    live = measure_batch_concurrency("desktop_cpu")
    if live.draft_count != 3:
        problems.append(f"desktop_cpu draft_count {live.draft_count} != 3")
    if live.max_in_flight_drafts != 1:
        problems.append("drafts overlapped through the real engine")
    if live.max_in_flight_same_core != 1:
        problems.append("same-core generations overlapped — the "
                        "turn lock is not serialising at batch scale")
    if live.max_in_flight_parallel < 2:
        problems.append("parallel cores did not overlap")
    if not live.parallel_ms < live.sequential_ms:
        problems.append(f"live parallel wall {live.parallel_ms} not "
                        f"below sequential {live.sequential_ms}")
    if not live.same_core_ms > live.parallel_ms:
        problems.append("live same-core wall not above the parallel "
                        "wall — the serialised shape is not visible")
    return problems


__all__ = [
    "BATCH_FORMAT",
    "BATCH_RULES",
    "BATCH_SIZE",
    "CONCURRENCY_SHAPES",
    "DELAY_MS",
    "MEASURED_ROWS",
    "MEASURED_RUNS",
    "PANEL_PROMPTS",
    "BatchMeasurement",
    "CountingBackend",
    "FlightRecorder",
    "batch_as_data",
    "batch_live_problems",
    "batch_problems",
    "measure_batch_concurrency",
    "median_batch_concurrency",
]
