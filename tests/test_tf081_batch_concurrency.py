"""
TF-081 — Batch & concurrency per profile, measured
honestly.

The estate lives in
``src/thoughtforge/inference/batch_concurrency.py``
(``CONCURRENCY_SHAPES`` / ``BATCH_RULES`` as data,
``measure_batch_concurrency()`` /
``median_batch_concurrency()``, format
``thoughtforge-batch-concurrency/1``).
Write-up: ``docs/architecture/BATCH_CONCURRENCY.md``.

Measured start: no backend or engine exposed a batch
(the contract is one request in, one response out;
the tree's one batch is ONNX tokenisation, an export
step); ``generate_drafts`` was a sequential loop in an
unguarded engine (no threading, no lock); TF-035
proved concurrent isolation on two cores but deferred
concurrent timing to this slice; and no profile
declared any concurrency, batch, or thread parameter
— ``draft_count`` was the only per-profile number
shaping multi-generation.

Pinned here:

  * the filed estate: both checkers clean, rule ids
    unique, the seven shapes in order,
  * the shapes against the live tree: no batch method
    on the contract, the drafts loop sequential in
    source, TF-035's turn lock standing, Aesir's
    one-resident rule standing, the HTTP legs per-call
    and session-free, the ONNX batch in place,
  * the filed panel: every profile covered, the panel
    shape identical per row, walls positive, parallel
    below sequential, same-core above parallel, the
    serialised legs at max in flight 1, parallel at
    2..batch_size, draft counts recomputed live from
    the resolved profile params,
  * the live panel on the VM profile (desktop_cpu):
    parallel cores overlap (max in flight >= 2) and
    beat the sequential wall; the same core never
    overlaps and stays above the parallel wall;
    drafts through the real engine never overlap and
    number the profile's draft_count,
  * the recorder/backend fixtures themselves: the
    in-flight witness counts honestly under threads,
  * proof honesty: the export states live proof
    not_proven, and unknown profiles raise naming the
    valid ids.
"""

from __future__ import annotations

import threading
import time

import pytest

from thoughtforge.inference.batch_concurrency import (
    BATCH_FORMAT,
    BATCH_RULES,
    BATCH_SIZE,
    CONCURRENCY_SHAPES,
    DELAY_MS,
    MEASURED_ROWS,
    PANEL_PROMPTS,
    CountingBackend,
    FlightRecorder,
    batch_as_data,
    batch_live_problems,
    batch_problems,
    measure_batch_concurrency,
)
from thoughtforge.inference.profiles import resolve_generation_params
from thoughtforge.utils.config import list_hardware_profiles, load_hardware_profile


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert batch_problems() == []
        assert batch_live_problems() == []

    def test_rules_shapes_and_format(self):
        ids = [r["id"] for r in BATCH_RULES]
        assert len(ids) == len(set(ids)) == 10
        assert [s["surface"] for s in CONCURRENCY_SHAPES] == [
            "unified_backend_contract", "turboquant_engine_drafts",
            "one_core_turns", "parallel_cores", "aesir_server",
            "http_backends", "onnx_tokenisation"]
        assert batch_as_data()["format"] == BATCH_FORMAT

    def test_panel_shape_is_fixed_and_synthetic(self):
        assert BATCH_SIZE == 4 and len(PANEL_PROMPTS) == 4
        assert DELAY_MS == 50.0
        assert len(set(PANEL_PROMPTS)) == 4  # distinct asks


# ── The filed panel ────────────────────────────────────────────


class TestFiledPanel:
    def test_every_profile_covered_with_live_draft_counts(self):
        assert set(MEASURED_ROWS) == set(list_hardware_profiles())
        for pid, row in MEASURED_ROWS.items():
            params = resolve_generation_params(load_hardware_profile(pid))
            assert row["draft_count"] == params.draft_count
            assert row["batch_size"] == BATCH_SIZE
            assert row["delay_ms"] == DELAY_MS

    def test_sharp_draft_counts(self):
        counts = {pid: row["draft_count"]
                  for pid, row in MEASURED_ROWS.items()}
        assert counts == {
            "desktop_cpu": 3, "desktop_gpu": 3, "phone_low": 2,
            "pi_5": 3, "pi_zero": 1, "server_gpu": 5}

    def test_ordering_and_witnesses(self):
        for pid, row in MEASURED_ROWS.items():
            assert row["parallel_ms"] < row["sequential_ms"], pid
            assert row["same_core_ms"] > row["parallel_ms"], pid
            assert row["max_in_flight_parallel"] == 4, pid
            assert row["max_in_flight_same_core"] == 1, pid
            assert row["max_in_flight_drafts"] == 1, pid
            # The serialised shape, measured: same-core wall
            # within a generous band of the sequential wall.
            assert 0.5 < row["same_core_ratio"] < 2.0, pid
            assert row["speedup_parallel"] > 1.0, pid

    def test_drafts_wall_respects_the_sequential_floor(self):
        for pid, row in MEASURED_ROWS.items():
            assert row["drafts_ms"] >= \
                DELAY_MS * row["draft_count"] * 0.8, pid


# ── The fixtures themselves ────────────────────────────────────


class TestFixtures:
    def test_recorder_counts_overlap_honestly(self):
        recorder = FlightRecorder()
        barrier = threading.Barrier(3)

        def hold():
            recorder.enter()
            try:
                barrier.wait(timeout=10)
                time.sleep(0.02)
            finally:
                recorder.exit()

        threads = [threading.Thread(target=hold) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert recorder.max_in_flight == 3
        assert recorder.calls == 3 and recorder.in_flight == 0

    def test_counting_backend_answers_about_its_own_prompt(self):
        from thoughtforge.inference.unified_backend import GenerationRequest

        recorder = FlightRecorder()
        backend = CountingBackend(recorder, delay_ms=1.0)
        response = backend.generate(GenerationRequest(
            prompt="Tell me about the ferry crossing at Lantern Hollow"))
        assert "ferry crossing" in response.text
        assert recorder.calls == 1 and recorder.max_in_flight == 1


# ── The live panel (VM profile) ────────────────────────────────


class TestLivePanel:
    def test_desktop_cpu_panel_shapes(self):
        live = measure_batch_concurrency("desktop_cpu")
        assert live.draft_count == 3
        assert live.max_in_flight_drafts == 1
        assert live.max_in_flight_same_core == 1
        assert live.max_in_flight_parallel >= 2
        assert live.parallel_ms < live.sequential_ms
        assert live.same_core_ms > live.parallel_ms
        data = live.to_dict()
        assert data["speedup_parallel"] > 1.0
        assert 0.5 < data["same_core_ratio"] < 2.0

    def test_pi_zero_single_draft(self):
        live = measure_batch_concurrency("pi_zero")
        assert live.draft_count == 1
        assert live.max_in_flight_drafts == 1

    def test_unknown_profile_raises_naming_valid_ids(self):
        with pytest.raises(ValueError) as excinfo:
            measure_batch_concurrency("bogus_profile")
        for pid in list_hardware_profiles():
            assert pid in str(excinfo.value)


# ── Proof honesty ──────────────────────────────────────────────


class TestProofHonesty:
    def test_export_states_not_proven(self):
        data = batch_as_data()
        assert data["live_proof"] == "not_proven"
        assert set(data["measured_rows"]) == set(list_hardware_profiles())
        rule = {r["id"]: r["rule"] for r in BATCH_RULES}
        assert "NOT PROVEN" in rule["live_concurrency_not_proven"]
        assert "NOT PROVEN" in rule["one_engine_concurrent_calls_unproven"]
