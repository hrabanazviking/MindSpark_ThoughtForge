"""
TF-082 — The Pi profile: the whole stack on
Mimir-class hardware, measured on a real Pi 5.

The estate lives in
``src/thoughtforge/inference/pi_profile.py``
(``PI_RULES`` as data, the on-device measurements
filed as ``HOST_EVIDENCE`` / ``DETECTOR_ON_PI`` /
``BOOT_ON_PI`` / ``EMBEDDINGS_ON_PI`` /
``ENCODE_ON_PI`` / ``COLD_START_ON_PI`` /
``BATCH_ON_PI`` / ``GGUF_ON_PI``, format
``thoughtforge-pi-profile/1``).
Write-up: ``docs/architecture/PI_PROFILE.md``.

Measured start: every prior Pi number was a VM
simulation (TF-069 boots, TF-079 cold starts, TF-081
batch, TF-053 embeddings — D-ENV-1's package-absent
verdict), each filed NOT PROVEN for real hardware.

Pinned here:

  * the filed estate: both checkers clean, rule ids
    unique, the export's proof grades stated,
  * the host record: a coherent Pi 5 (aarch64,
    device-tree == cpuinfo model, a commit-shaped
    tree head, sane thermals),
  * the detector record + D-DET-1: the Pi selected
    pi_5, the filed is_mobile defect measurement
    stands (True on a >4 GB machine), the live source
    detects RAM before the mobile check, and the
    live behaviour is fixed both ways (16 GB ARM not
    mobile and selecting pi_5, Pi or not; 2 GB ARM
    still mobile),
  * the boots: on-Pi verdicts and cap rows equal
    TF-069's filed simulation rows leg for leg —
    pi_5 passing all nine legs, pi_zero failing
    exactly ram_ceiling + recommended_models,
  * the embeddings: filed rows consistent with
    TF-053's live assignments (pi_5 available_offline
    with package, pi_zero not_assigned), the real
    encode row's shape (384-dim, 2 rows),
  * the cold rows: TTFT is the stage sum per profile,
    all stages positive,
  * the batch row: TF-081's ordering and witnesses
    reproduced (parallel < sequential, same-core >
    parallel, in flight 4/1/1, speedup recomputes),
  * the GGUF inventory: every filed quant is the
    tree's own parser's verdict on the filed name,
  * proof honesty: pi_zero hardware and live-model
    generation are filed not_proven, never proven.
"""

from __future__ import annotations

from thoughtforge.inference.pi_profile import (
    BATCH_ON_PI,
    BOOT_ON_PI,
    COLD_START_ON_PI,
    DETECTOR_ON_PI,
    EMBEDDINGS_ON_PI,
    ENCODE_ON_PI,
    GGUF_ON_PI,
    HOST_EVIDENCE,
    PI_FORMAT,
    PI_PROFILES,
    PI_RULES,
    pi_as_data,
    pi_live_problems,
    pi_problems,
)
from thoughtforge.inference.profile_runner import BOOT_VERDICTS, RAM_ROWS
from thoughtforge.inference.turboquant_map import quant_of_filename


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert pi_problems() == []
        assert pi_live_problems() == []

    def test_rules_and_format(self):
        ids = [r["id"] for r in PI_RULES]
        assert len(ids) == len(set(ids)) == 11
        assert pi_as_data()["format"] == PI_FORMAT
        assert PI_PROFILES == ("pi_5", "pi_zero")


# ── The host record ────────────────────────────────────────────


class TestHostRecord:
    def test_a_coherent_pi5(self):
        host = HOST_EVIDENCE
        assert host["machine"] == "aarch64"
        assert "Raspberry Pi 5" in host["device_tree_model"]
        assert host["device_tree_model"] == host["cpuinfo_model"]
        assert host["cpu_count"] == 4
        assert host["mem_total_mb"] > host["mem_available_mb"] > 0

    def test_head_and_thermals(self):
        host = HOST_EVIDENCE
        assert len(host["tree_head"]) == 40
        assert 0 < host["thermal_c_start"] < 85
        assert 0 < host["thermal_c_end"] < 85
        # Colour with a point: the probe peaked at
        # roughly one pi_zero of RAM.
        assert host["peak_rss_mb"] > 256


# ── The detector and D-DET-1 ───────────────────────────────────


class TestDetector:
    def test_the_pi_selected_its_profile(self):
        det = DETECTOR_ON_PI
        assert det["selected_profile"] == "pi_5"
        assert det["is_raspberry_pi"] is True
        assert det["is_arm"] is True and det["cpu_arch"] == "arm64"
        assert det["ram_gb"] > 4.0

    def test_the_defect_stays_filed_as_measured(self):
        # The on-device run measured is_mobile True on a
        # 16 GB machine — that measurement is the
        # defect's evidence and is never edited away.
        assert DETECTOR_ON_PI["is_mobile_as_measured"] is True

    def test_live_behaviour_after_the_fix(self):
        from thoughtforge.inference.profiles import (
            HardwareDetector,
            HardwareSnapshot,
            ProfileSelector,
        )

        detector = HardwareDetector()
        big = HardwareSnapshot(os_name="linux", cpu_arch="arm64",
                               is_arm=True, ram_gb=16.0)
        assert detector._detect_mobile(big) is False
        assert ProfileSelector._select_id(big) == "pi_5"
        small = HardwareSnapshot(os_name="linux", cpu_arch="arm64",
                                 is_arm=True, ram_gb=2.0)
        assert detector._detect_mobile(small) is True


# ── The boots ──────────────────────────────────────────────────


class TestBootsOnPi:
    def test_verdicts_match_the_simulation(self):
        for pid, row in BOOT_ON_PI.items():
            assert row["verdict"] == BOOT_VERDICTS[pid], pid
            assert row["cap_status"] == RAM_ROWS[pid], pid

    def test_pi5_passes_all_nine_legs(self):
        assert len(BOOT_ON_PI["pi_5"]["legs"]) == 9
        assert set(BOOT_ON_PI["pi_5"]["legs"].values()) == {"pass"}

    def test_pi_zero_fails_the_simulations_two(self):
        failing = {k for k, s in BOOT_ON_PI["pi_zero"]["legs"].items()
                   if s != "pass"}
        assert failing == {"ram_ceiling", "recommended_models"}


# ── Embeddings and cold/batch rows ─────────────────────────────


class TestMeasuredRows:
    def test_embeddings_rows(self):
        assert EMBEDDINGS_ON_PI["pi_5"]["status"] == "available_offline"
        assert EMBEDDINGS_ON_PI["pi_5"]["package_installed"] is True
        assert EMBEDDINGS_ON_PI["pi_zero"]["status"] == "not_assigned"
        assert EMBEDDINGS_ON_PI["phone_low"]["status"] == \
            "unavailable_onnx_artefact_missing"
        assert ENCODE_ON_PI["dim"] == 384
        assert ENCODE_ON_PI["rows"] == 2

    def test_cold_rows_sum_to_ttft(self):
        for pid, row in COLD_START_ON_PI.items():
            stage_sum = (row["profile_load_ms"]
                         + row["core_construction_ms"]
                         + row["engine_load_ms"]
                         + row["first_token_ms"])
            assert abs(stage_sum - row["ttft_ms"]) <= 0.01, pid
            assert all(v > 0 for v in row.values()), pid

    def test_batch_row_shapes(self):
        row = BATCH_ON_PI
        assert row["parallel_ms"] < row["sequential_ms"]
        assert row["same_core_ms"] > row["parallel_ms"]
        assert (row["max_in_flight_parallel"],
                row["max_in_flight_same_core"],
                row["max_in_flight_drafts"]) == (4, 1, 1)
        assert abs(row["sequential_ms"] / row["parallel_ms"]
                   - row["speedup_parallel"]) <= 0.01

    def test_gguf_quants_are_the_parsers_verdicts(self):
        assert len(GGUF_ON_PI) == 4
        for row in GGUF_ON_PI:
            assert quant_of_filename(row["name"]) == row["quant"]
            assert row["size_mb"] > 0


# ── Proof honesty ──────────────────────────────────────────────


class TestProofHonesty:
    def test_not_proven_grades_stated(self):
        proof = pi_as_data()["proof"]
        assert proof["pi_5_stack"] == "proven_on_real_pi_5"
        assert proof["pi_zero_hardware"] == "not_proven"
        assert proof["live_model_generation"] == "not_proven"
        rules = {r["id"]: r["rule"] for r in PI_RULES}
        assert "NOT PROVEN" in rules["live_model_generation_not_proven_on_pi"]
        assert "NOT PROVEN" in rules["pi_zero_hardware_not_proven"]
