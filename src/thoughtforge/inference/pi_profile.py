"""TF-082 — The Pi profile: the whole stack on
Mimir-class hardware, measured on a real Raspberry
Pi 5 — and the one hardware class this campaign had
never touched stops being a simulation.

Roadmap: "The Pi profile: the whole stack on
Mimir-class hardware (phone_low/pi budgets) — proven
on the real Pi or NOT PROVEN, stated plainly."

Measured start (probed live this tick):

- **Every prior Pi number in this campaign was a
  simulation on an x86_64 VM.** TF-069 boots the pi
  profiles from their own JSONs on the VM, TF-079's
  cold starts and TF-081's batch panel are VM fixture
  timings, TF-053's embedding verdicts are the VM's
  (package absent there — D-ENV-1), and each of those
  slices stated plainly: real Pi NOT PROVEN, TF-082
  owns the road.
- **The house has a real Pi, and it was reachable.**
  ``purpleviking1``: Raspberry Pi 5 Model B Rev 1.1,
  aarch64, 4 cores, 16 GB RAM, kernel
  6.12.112-v8-16k-hailo+, Python 3.11.2. The campaign
  tree was cloned onto it (sparse, branch head
  52f7630 — the exact tree this slice banks on) and
  the stack legs below were executed there; their
  outputs are filed as data in this module, in the
  shapes the earlier slices defined.
- **One real defect fell out of the hardware
  (D-DET-1, FIXED in scope).**
  ``HardwareDetector.detect()`` computed ``is_mobile``
  BEFORE detecting RAM, so ``_detect_mobile`` read the
  snapshot's default ``ram_gb = 0.0`` and its
  ``ram < 4.0`` clause labelled *every* ARM machine
  mobile — on the Pi, a 16 GB machine came back
  ``is_mobile = True``. Selection survived only
  because ``_select_id`` checks ``is_raspberry_pi``
  before the mobile branch; a non-Pi ARM Linux
  machine would have been misrouted to phone_low.
  The fix is an ordering change in ``detect()`` (RAM
  first), pinned by this slice's checker against the
  live source. On x86 the bug was invisible
  (``is_arm`` False short-circuits the clause).
- **What the Pi proved, leg by leg** (filed below):
  the detector identifies the Pi and selects pi_5;
  TF-069's boots run on-device with verdicts
  *identical* to the VM simulation (pi_5
  boots_with_ceiling_gaps with all nine legs passing,
  pi_zero the same verdict with the same two failing
  legs — the q2_k ceiling unprovable and the
  over-cap TinyLlama recommendation); TF-053's
  embedding assignment for pi_5 is **available
  offline** on the Pi (package present, model
  cached — the opposite of the VM's D-ENV-1 verdict)
  and a real offline encode of two sentences through
  sentence-transformers 5.4.1 produced 384-dim
  vectors; TF-079's cold-start medians and TF-081's
  batch panel reproduce their shapes on ARM (parallel
  max in flight 4/4, serialised legs at 1).
- **What the Pi did not prove, stated plainly.**
  Live-model generation is NOT PROVEN on the Pi
  either: ``llama_cpp`` is absent from the Pi's
  system Python and both house venvs probed, and the
  Pi's package index serves **no binary wheel** for
  it (``pip install --only-binary :all:
  llama-cpp-python`` → "No matching distribution
  found", measured this tick) — a source build is
  TF-174's road, not this tick's. Four real GGUFs
  *are* on the Pi (TinyLlama-1.1B at q4_0 / q5_k_m /
  q6_k / q8_0, inventoried below through the tree's
  own filename parser) waiting for a loader. And
  **pi_zero hardware is not in the house**: the only
  Pi is this 16 GB Pi 5, so every pi_zero row here is
  pi_zero's *profile* executed on the Pi 5 — a 512 MB
  Pi Zero was never run, and its hardware verdict
  stays NOT PROVEN (TF-174). Colour, measured: the
  whole on-Pi probe process peaked at 512.1 MB RSS —
  one pi_zero's entire RAM, spent largely by the
  torch/sentence-transformers leg pi_zero is honestly
  assigned none of (TF-053).

Proof grade REAL HARDWARE, fixture legs (the real
tree's real boots, detector, embedding encode, cold
starts and batch panel, executed on a real Raspberry
Pi 5 at the campaign head) — a live model's
generation on the Pi NOT PROVEN (no llama_cpp), a
Pi Zero NOT PROVEN (no such hardware in the house),
never claimed.
"""

from __future__ import annotations

from typing import Any

PI_FORMAT = "thoughtforge-pi-profile/1"

#: The profiles this slice is about, in roadmap order.
PI_PROFILES: tuple[str, ...] = ("pi_5", "pi_zero")

# ── The device, as measured ─────────────────────────────────────────────────
# purpleviking1, probed live on-device this tick at
# campaign head 52f7630 (sparse clone of this branch).

HOST_EVIDENCE: dict[str, Any] = {
    "hostname": "purpleviking1",
    "device_tree_model": "Raspberry Pi 5 Model B Rev 1.1",
    "cpuinfo_model": "Raspberry Pi 5 Model B Rev 1.1",
    "machine": "aarch64",
    "kernel": "6.12.112-v8-16k-hailo+",
    "python": "3.11.2",
    "cpu_count": 4,
    "mem_total_mb": 16209,
    "mem_available_mb": 14394,
    "thermal_c_start": 44.65,
    "thermal_c_end": 43.55,
    "peak_rss_mb": 512.1,
    "tree_head": "52f76303fad1a41f0a86261c521ffd4a5a3bc421",
}

#: The detector's snapshot on the real Pi. is_mobile
#: is filed as measured — True, the D-DET-1 defect —
#: and the fix's post-state is judged live by the
#: checker (a 16 GB ARM snapshot is not mobile).
DETECTOR_ON_PI: dict[str, Any] = {
    "ram_gb": 15.83, "vram_gb": 0.0, "cpu_arch": "arm64",
    "os_name": "linux", "cpu_count": 4, "gpu_name": None,
    "has_cuda": False, "is_arm": True, "is_raspberry_pi": True,
    "is_mobile_as_measured": True,
    "selected_profile": "pi_5",
}

# ── TF-069's boots, executed on the Pi ──────────────────────────────────────

BOOT_ON_PI: dict[str, dict[str, Any]] = {
    "pi_5": {
        "verdict": "boots_with_ceiling_gaps",
        "legs": {
            "profile": "pass", "params": "pass",
            "selector": "pass", "ram_ceiling": "pass",
            "recommended_models": "pass", "deployment": "pass",
            "core": "pass", "memory_budget": "pass",
            "engine": "pass",
        },
        "cap_status": "fit", "host_backend": "cpu",
    },
    "pi_zero": {
        "verdict": "boots_with_ceiling_gaps",
        "legs": {
            "profile": "pass", "params": "pass",
            "selector": "pass", "ram_ceiling": "fail",
            "recommended_models": "fail", "deployment": "pass",
            "core": "pass", "memory_budget": "pass",
            "engine": "pass",
        },
        "cap_status": "unprovable", "host_backend": "cpu",
    },
}

# ── Embeddings on the Pi (TF-053's discipline) ─────────────────────────────

EMBEDDINGS_ON_PI: dict[str, dict[str, Any]] = {
    "pi_5": {"status": "available_offline",
             "runtime": "sentence_transformers",
             "package_installed": True,
             "model": "all-MiniLM-L6-v2"},
    "pi_zero": {"status": "not_assigned", "runtime": "none",
                "package_installed": None, "model": None},
    "phone_low": {"status": "unavailable_onnx_artefact_missing",
                  "runtime": "onnx", "package_installed": True,
                  "model": "all-MiniLM-L6-v2"},
}

#: A real offline encode on the Pi: the registered
#: model loaded from the local cache (HF_HUB_OFFLINE=1)
#: and two synthetic sentences encoded.
ENCODE_ON_PI: dict[str, Any] = {
    "st_version": "5.4.1", "dim": 384, "rows": 2,
    "load_ms": 645.346, "encode_ms": 824.75,
}

# ── Cold starts on the Pi (TF-079's medians of 3) ──────────────────────────

COLD_START_ON_PI: dict[str, dict[str, float]] = {
    "pi_5": {"profile_load_ms": 0.115, "core_construction_ms": 3.848,
             "engine_load_ms": 3.074, "first_token_ms": 0.006,
             "ttft_ms": 7.043},
    "pi_zero": {"profile_load_ms": 0.093, "core_construction_ms": 2.104,
                "engine_load_ms": 2.465, "first_token_ms": 0.005,
                "ttft_ms": 4.667},
}

# ── TF-081's batch panel, one run on the Pi (pi_5) ─────────────────────────

BATCH_ON_PI: dict[str, Any] = {
    "profile_id": "pi_5", "batch_size": 4, "delay_ms": 50.0,
    "draft_count": 3, "sequential_ms": 204.578, "parallel_ms": 53.275,
    "same_core_ms": 204.037, "drafts_ms": 150.329,
    "max_in_flight_parallel": 4, "max_in_flight_same_core": 1,
    "max_in_flight_drafts": 1, "speedup_parallel": 3.84,
    "same_core_ratio": 0.997,
}

# ── The GGUFs waiting on the Pi ─────────────────────────────────────────────
# Inventoried on-device through TF-066's own filename
# parser. Sizes are the files' measured sizes in MB.

GGUF_ON_PI: tuple[dict[str, Any], ...] = (
    {"name": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
     "size_mb": 637.7, "quant": "q4_0"},
    {"name": "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
     "size_mb": 783.0, "quant": "q5_k_m"},
    {"name": "tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
     "size_mb": 904.4, "quant": "q6_k"},
    {"name": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
     "size_mb": 1170.8, "quant": "q8_0"},
)

# ── The rules ───────────────────────────────────────────────────────────────

PI_RULES: tuple[dict, ...] = (
    {"id": "pi5_stack_boots_on_real_hardware",
     "rule": "TF-069's whole boot runs on the real Pi 5 "
             "for pi_5: all nine legs pass on-device "
             "(profile / params / selector / ram_ceiling / "
             "recommended_models / deployment / core / "
             "memory_budget / engine), the engine's host "
             "backend honestly cpu"},
    {"id": "simulation_verdicts_reproduced",
     "rule": "the on-device boot verdicts and cap rows "
             "are identical to TF-069's VM simulation for "
             "both Pi profiles (pi_5 boots_with_ceiling_"
             "gaps / cap fit; pi_zero the same verdict / "
             "cap unprovable, its ram_ceiling and "
             "recommended_models legs failing for the "
             "same filed reasons) — the simulation was "
             "telling the truth about the stack's shape"},
    {"id": "detector_selects_pi5_on_the_pi",
     "rule": "the tree's own detector identifies the "
             "real machine (aarch64, is_raspberry_pi via "
             "the cpuinfo Model line, 15.83 GB detected) "
             "and its selector chooses pi_5 — the profile "
             "this hardware is for"},
    {"id": "detector_mobile_ordering_defect",
     "rule": "D-DET-1 (found on the Pi, FIXED in scope): "
             "detect() computed is_mobile before RAM was "
             "detected, so the check read the 0.0 default "
             "and labelled every ARM machine mobile "
             "(measured True on the 16 GB Pi); selection "
             "survived only because _select_id checks "
             "is_raspberry_pi first — a non-Pi ARM Linux "
             "machine would have been misrouted to "
             "phone_low. detect() now detects RAM first; "
             "the checker pins the source order"},
    {"id": "embeddings_available_offline_on_pi5",
     "rule": "TF-053's pi_5 assignment (sentence_"
             "transformers, all-MiniLM-L6-v2) is "
             "available_offline on the Pi — package "
             "importable AND model cached — the opposite "
             "of this VM's verdict (D-ENV-1, package "
             "absent); and the leg genuinely runs: a real "
             "offline encode on-device produced 384-dim "
             "vectors (load 645.346 ms, encode 824.75 ms, "
             "sentence-transformers 5.4.1)"},
    {"id": "pi_zero_embeddings_not_assigned_anywhere",
     "rule": "pi_zero's embedding verdict on the Pi is "
             "not_assigned, exactly as on the VM: the "
             "512 MB profile carries no embedding model "
             "on any machine, SQL-only by TF-053's rule"},
    {"id": "phone_low_onnx_artefact_missing_on_pi_too",
     "rule": "phone_low's ONNX leg is unavailable on the "
             "Pi for the artefact reason, not the package "
             "reason (onnxruntime importable there, no "
             "exported artefact present) — TF-070's estate "
             "owns the export road"},
    {"id": "cold_and_batch_shapes_reproduce_on_arm",
     "rule": "TF-079's cold-start medians and TF-081's "
             "batch panel reproduce their shapes on the "
             "Pi (cold TTFT 7.043 ms pi_5 / 4.667 ms "
             "pi_zero at fixture grade; batch parallel "
             "max in flight 4 of 4 with speedup 3.84, "
             "serialised legs at max in flight 1, "
             "same/seq ratio 0.997). The Pi's walls are "
             "lower than the contended VM's filed rows — "
             "recorded as measured, never as a model-"
             "speed claim: the first token is still a "
             "fixture's, the engine still model-less"},
    {"id": "live_model_generation_not_proven_on_pi",
     "rule": "live-model generation on the Pi is NOT "
             "PROVEN: no live model generated there — "
             "llama_cpp is absent from the Pi's system "
             "Python and both house venvs probed, and the "
             "Pi's index serves no binary wheel for it "
             "(measured pip refusal this tick). Four real "
             "TinyLlama GGUFs are on the device (filed in "
             "GGUF_ON_PI through the tree's own parser) "
             "waiting for a loader; a source build or a "
             "binary backend is TF-174's road, and TF-086's "
             "truth table must carry the line"},
    {"id": "pi_zero_hardware_not_proven",
     "rule": "the house's only Pi is a 16 GB Pi 5: every "
             "pi_zero row in this slice is pi_zero's "
             "profile executed on that Pi 5, never a "
             "512 MB Pi Zero — pi_zero hardware stays NOT "
             "PROVEN (TF-174). The probe process's own "
             "peak RSS, 512.1 MB, is filed as colour: it "
             "equals pi_zero's whole declared RAM"},
    {"id": "thermal_headroom_measured_once",
     "rule": "the probe ran at 44.65 °C → 43.55 °C SoC "
             "temperature: no thermal pressure was "
             "observed during the stack's legs (one run, "
             "stated as one run; a soak is TF-176's)"},
)


# ── Export ──────────────────────────────────────────────────────────────────

def pi_as_data() -> dict[str, Any]:
    return {
        "format": PI_FORMAT,
        "pi_profiles": list(PI_PROFILES),
        "host": dict(HOST_EVIDENCE),
        "detector": dict(DETECTOR_ON_PI),
        "boots": {pid: dict(row) for pid, row in BOOT_ON_PI.items()},
        "embeddings": {pid: dict(row)
                       for pid, row in EMBEDDINGS_ON_PI.items()},
        "encode": dict(ENCODE_ON_PI),
        "cold_start": {pid: dict(row)
                       for pid, row in COLD_START_ON_PI.items()},
        "batch_pi5": dict(BATCH_ON_PI),
        "gguf": [dict(r) for r in GGUF_ON_PI],
        "rules": [dict(r) for r in PI_RULES],
        "proof": {
            "pi_5_stack": "proven_on_real_pi_5",
            "pi_zero_profile_on_pi_5": "proven_on_real_pi_5",
            "pi_zero_hardware": "not_proven",
            "live_model_generation": "not_proven",
        },
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def pi_problems() -> list[str]:
    """The filed estate judged against the live tree."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in PI_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("PI_RULES ids are not unique")

    import inspect

    from thoughtforge.etl.embedding_discipline import PROFILE_EMBEDDINGS
    from thoughtforge.inference import profiles as profiles_module
    from thoughtforge.inference.profile_runner import BOOT_VERDICTS, RAM_ROWS
    from thoughtforge.inference.turboquant_map import quant_of_filename
    from thoughtforge.utils.config import list_hardware_profiles

    # Host evidence shape: a real Pi 5, the campaign head.
    host = HOST_EVIDENCE
    if host["machine"] != "aarch64" \
            or "Raspberry Pi 5" not in host["device_tree_model"] \
            or host["device_tree_model"] != host["cpuinfo_model"]:
        problems.append("host evidence is not a coherent Pi 5 record")
    if len(host["tree_head"]) != 40 \
            or any(c not in "0123456789abcdef" for c in host["tree_head"]):
        problems.append("host tree_head is not a commit sha")
    if not (0 < host["thermal_c_start"] < 85
            and 0 < host["thermal_c_end"] < 85):
        problems.append("filed thermals are outside a sane SoC band")

    # The detector record: the Pi's identity, its one
    # defect, and the correct selection.
    det = DETECTOR_ON_PI
    if det["selected_profile"] != "pi_5" or not det["is_raspberry_pi"] \
            or not det["is_arm"] or det["cpu_arch"] != "arm64":
        problems.append("detector record drifted from the Pi's identity")
    if det["is_mobile_as_measured"] is not True:
        problems.append("the D-DET-1 measurement (is_mobile True on "
                        "the 16 GB Pi) must stay filed as measured")
    if not det["ram_gb"] > 4.0:
        problems.append("the Pi's detected RAM must exceed the "
                        "mobile clause's 4 GB line — that is what "
                        "makes the filed is_mobile a defect")

    # D-DET-1's fix stands in the live source: RAM is
    # detected before the mobile check reads it, and the
    # selector still checks the Pi branch before mobile.
    detect_src = inspect.getsource(
        profiles_module.HardwareDetector.detect)
    ram_at = detect_src.find("snap.ram_gb = self._detect_ram_gb()")
    mobile_at = detect_src.find("snap.is_mobile = self._detect_mobile(snap)")
    if not (0 <= ram_at < mobile_at):
        problems.append("D-DET-1 regressed: detect() no longer "
                        "detects RAM before is_mobile")
    select_src = inspect.getsource(
        profiles_module.ProfileSelector._select_id)
    if not (0 <= select_src.find("is_raspberry_pi")
            < select_src.find("hw.is_mobile")):
        problems.append("_select_id no longer checks the Pi "
                        "branch before the mobile branch")

    # Boots: the Pi's verdicts are TF-069's filed
    # simulation verdicts, leg for leg.
    if set(BOOT_ON_PI) != set(PI_PROFILES):
        problems.append("boot rows do not cover exactly the Pi profiles")
    for pid, row in BOOT_ON_PI.items():
        if pid not in list_hardware_profiles():
            problems.append(f"{pid}: not a shipped profile")
        if row["verdict"] != BOOT_VERDICTS[pid]:
            problems.append(f"{pid}: on-Pi verdict {row['verdict']} != "
                            f"TF-069's filed {BOOT_VERDICTS[pid]}")
        if row["cap_status"] != RAM_ROWS[pid]:
            problems.append(f"{pid}: on-Pi cap {row['cap_status']} != "
                            f"TF-069's filed {RAM_ROWS[pid]}")
        if set(row["legs"]) != {
                "profile", "params", "selector", "ram_ceiling",
                "recommended_models", "deployment", "core",
                "memory_budget", "engine"}:
            problems.append(f"{pid}: boot legs drifted from TF-069's nine")
    if any(s != "pass" for s in BOOT_ON_PI["pi_5"]["legs"].values()):
        problems.append("pi_5 on the Pi must pass all nine legs")
    pi_zero_fail = {k for k, s in BOOT_ON_PI["pi_zero"]["legs"].items()
                    if s != "pass"}
    if pi_zero_fail != {"ram_ceiling", "recommended_models"}:
        problems.append(f"pi_zero's on-Pi failing legs {pi_zero_fail} "
                        f"!= the simulation's two")

    # Embeddings: filed statuses consistent with
    # TF-053's live assignments.
    for pid, row in EMBEDDINGS_ON_PI.items():
        assignment = PROFILE_EMBEDDINGS[pid]
        if row["runtime"] != assignment["runtime"]:
            problems.append(f"{pid}: filed runtime != TF-053's assignment")
        if row["model"] != assignment["model"]:
            problems.append(f"{pid}: filed model != TF-053's assignment")
    if EMBEDDINGS_ON_PI["pi_5"]["status"] != "available_offline" \
            or EMBEDDINGS_ON_PI["pi_5"]["package_installed"] is not True:
        problems.append("pi_5's on-Pi embedding verdict must stay "
                        "available_offline with the package installed")
    if EMBEDDINGS_ON_PI["pi_zero"]["status"] != "not_assigned":
        problems.append("pi_zero must stay not_assigned on the Pi too")
    if ENCODE_ON_PI["dim"] != 384 or ENCODE_ON_PI["rows"] != 2 \
            or ENCODE_ON_PI["load_ms"] <= 0 \
            or ENCODE_ON_PI["encode_ms"] <= 0:
        problems.append("the on-Pi encode row drifted (dim 384, "
                        "2 rows, positive walls)")

    # Cold rows: positive, and TTFT is the stage sum.
    if set(COLD_START_ON_PI) != set(PI_PROFILES):
        problems.append("cold rows do not cover exactly the Pi profiles")
    for pid, row in COLD_START_ON_PI.items():
        stage_sum = (row["profile_load_ms"] + row["core_construction_ms"]
                     + row["engine_load_ms"] + row["first_token_ms"])
        if abs(stage_sum - row["ttft_ms"]) > 0.01:
            problems.append(f"{pid}: filed TTFT {row['ttft_ms']} != "
                            f"stage sum {stage_sum}")
        if any(v <= 0 for v in row.values()):
            problems.append(f"{pid}: a cold-start stage is not positive")

    # Batch row: TF-081's shape, on ARM.
    batch = BATCH_ON_PI
    if batch["profile_id"] != "pi_5" or batch["batch_size"] != 4 \
            or batch["draft_count"] != 3:
        problems.append("the on-Pi batch row drifted from TF-081's panel")
    if not batch["parallel_ms"] < batch["sequential_ms"]:
        problems.append("on-Pi parallel wall is not below sequential")
    if not batch["same_core_ms"] > batch["parallel_ms"]:
        problems.append("on-Pi same-core wall is not above parallel")
    if (batch["max_in_flight_parallel"], batch["max_in_flight_same_core"],
            batch["max_in_flight_drafts"]) != (4, 1, 1):
        problems.append("on-Pi in-flight witnesses drifted from (4, 1, 1)")
    if abs(batch["sequential_ms"] / batch["parallel_ms"]
           - batch["speedup_parallel"]) > 0.01:
        problems.append("on-Pi speedup does not recompute from the walls")

    # GGUF inventory: every filed quant is the tree's
    # own parser's verdict on the filed name.
    for row in GGUF_ON_PI:
        if quant_of_filename(row["name"]) != row["quant"]:
            problems.append(f"{row['name']}: filed quant "
                            f"{row['quant']} != the parser's verdict")
        if row["size_mb"] <= 0:
            problems.append(f"{row['name']}: size is not positive")
    return problems


def pi_live_problems() -> list[str]:
    """The D-DET-1 fix and the selection rules, judged
    behaviourally on this machine's tree."""
    problems: list[str] = []
    from thoughtforge.inference.profiles import (
        HardwareDetector,
        HardwareSnapshot,
        ProfileSelector,
    )

    detector = HardwareDetector()
    big_arm = HardwareSnapshot(
        os_name="linux", cpu_arch="arm64", is_arm=True, ram_gb=16.0)
    if detector._detect_mobile(big_arm) is not False:
        problems.append("a 16 GB ARM snapshot is still labelled mobile")
    small_arm = HardwareSnapshot(
        os_name="linux", cpu_arch="arm64", is_arm=True, ram_gb=2.0)
    if detector._detect_mobile(small_arm) is not True:
        problems.append("a 2 GB ARM snapshot is no longer mobile — "
                        "the clause itself must stay")
    pi_snap = HardwareSnapshot(
        os_name="linux", cpu_arch="arm64", is_arm=True,
        is_raspberry_pi=True, ram_gb=16.0)
    if ProfileSelector._select_id(pi_snap) != "pi_5":
        problems.append("a 16 GB Pi snapshot no longer selects pi_5")
    arm_server = HardwareSnapshot(
        os_name="linux", cpu_arch="arm64", is_arm=True,
        is_mobile=detector._detect_mobile(big_arm), ram_gb=16.0)
    if ProfileSelector._select_id(arm_server) != "pi_5":
        problems.append("a non-Pi 16 GB ARM machine no longer lands "
                        "on pi_5 — D-DET-1's misroute is back")
    this_machine = detector.detect()
    if this_machine.is_mobile:
        problems.append("this (x86_64) machine is labelled mobile")
    return problems


__all__ = [
    "BATCH_ON_PI",
    "BOOT_ON_PI",
    "COLD_START_ON_PI",
    "DETECTOR_ON_PI",
    "EMBEDDINGS_ON_PI",
    "ENCODE_ON_PI",
    "GGUF_ON_PI",
    "HOST_EVIDENCE",
    "PI_FORMAT",
    "PI_PROFILES",
    "PI_RULES",
    "pi_as_data",
    "pi_live_problems",
    "pi_problems",
]
