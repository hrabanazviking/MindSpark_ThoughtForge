"""TF-083 — GPU-optional proof: no profile secretly
requires a GPU, and the CPU paths are measured on the
laptop line as well as on this GPU-less VM.

Roadmap: "GPU-optional proof: no profile secretly
requires a GPU; CPU paths measured on the laptop
line."

Measured start (probed live this tick):

- **The profile estate, read plainly.** Four of the
  six shipped profiles declare a CPU path outright
  (desktop_cpu / phone_low / pi_5 / pi_zero:
  ``inference_backend: "cpu"``, ``vram_gb: 0``,
  ``gpu: null``). The two GPU profiles declare CPU
  as a fallback in their own JSONs (desktop_gpu:
  cuda with fallbacks [vulkan, directml, cpu];
  server_gpu: cuda with fallback [cpu]). No profile
  declares a GPU as its only path — but a declaration
  is a claim, and this slice measured the machinery
  that consumes it.
- **This VM has no GPU, and the stack runs here.**
  The detector's snapshot on this VM: vram 0.0,
  gpu_name None, has_cuda False, auto-selection
  desktop_cpu. ``detect_backends()`` finds exactly
  vulkan (library presence — see D-GPU-2) and cpu.
  Every Part IV panel before this one (TF-069 boots,
  TF-079 cold starts, TF-081 batch) already ran on
  this GPU-less host; this slice files the CPU panel
  as its own measurement (below).
- **One secret GPU requirement was real, and it is
  FIXED in scope (D-GPU-1).** The engine resolves
  ``n_gpu_layers`` from the *profile's declared
  VRAM* (the resolver ladder: server_gpu -1,
  desktop_gpu 40, everything else 0), selects a
  backend by the profile's declared backend +
  fallbacks, and ``build_llama_kwargs`` merged the
  resolved count over the selected backend's own
  kwargs. Measured pre-fix on this GPU-less VM:
  server_gpu **selected the cpu backend and still
  carried n_gpu_layers=-1** into the kwargs a model
  load would receive — CPU in name, all-layers-to-
  GPU in fact; desktop_gpu with cpu selected would
  have carried 40. The fix lives in
  ``build_llama_kwargs`` (the tree's one kwargs
  constructor): a backend whose own declared kwargs
  carry ``n_gpu_layers == 0`` (cpu; directml, whose
  offload is a build property, not a kwarg) is
  clamped to 0 whatever the caller resolved, and the
  engine's load log now reports the effective count
  from the built kwargs. GPU-capable backends keep
  the caller's count exactly (cuda + 32 → 32,
  checker-pinned). Whether a CPU-only llama.cpp
  build would have errored or silently ignored -1
  is NOT PROVEN (``llama_cpp`` is absent on both
  hosts measured) — the defect is that the forge's
  own contract said cpu and handed over a GPU
  instruction, and that is what is fixed.
- **The laptop line, measured.** ``purplelongship``
  (Ryzen 9 4900HS, 16 threads, 38.59 GB RAM, RTX
  2060 Max-Q 6 GB, has_cuda True, auto-selection
  desktop_gpu, backends cuda / vulkan / cpu) took a
  sparse clone of this branch at head 2222a52 and
  ran this slice's panel there: **every profile's
  forced-CPU engine load succeeds on the GPU
  machine too** (backend cpu for all six), the two
  GPU profiles' declared loads bind cuda there, and
  every profile's core answers a knowledge-only
  turn. The laptop walls (~1.0–1.4 s per engine
  load) were measured under a load average of
  ~15.6 on 16 threads — contention, filed as
  colour, never as baselines (the Aesir D-14 rule
  this campaign has kept since TF-079).
- **The rest of the tree asks for a GPU nowhere.**
  A sweep over ``src/`` for hard CUDA/GPU
  requirements finds none: the ONNX embedder
  defaults ``use_gpu=False`` and always keeps the
  CPU execution provider in its list (CUDA is
  prepended only on request); the HF backend is a
  remote API client; the embeddings module is
  CPU-shaped (its per-profile truth is TF-053's);
  the cognition core names no device at all.
- **One probe overstates, filed OPEN (D-GPU-2).**
  Both vulkan probes (``backends._probe_vulkan``
  and ``profiles._check_vulkan``) accept the mere
  presence of ``libvulkan.so.1`` as a working GPU
  backend: on this VM the library exists,
  ``vulkaninfo`` does not, and no GPU device
  exists, yet vulkan reports available and
  desktop_gpu's declared load selects it here. A
  library is not a device. The selection is not a
  *requirement* (cpu stands behind it in the
  profile's own fallback order, and the clamp
  above bounds what a non-offload backend can be
  handed) — but the probe's truth grade is a
  library's, not a device's, and TF-086's truth
  table v2 owns the line.

Proof grade FIXTURE (real per-profile engine loads
— forced-CPU and as-declared — and real core turns
on two hosts: this GPU-less VM and the laptop line;
the kwargs each profile would hand a model load,
judged live) — a live model's generation on CPU is
NOT PROVEN on either host (``llama_cpp`` absent on
both, no GGUF in the tree), a GPU-less *laptop* is
NOT PROVEN (the laptop line's machine carries a
GPU; its CPU paths are proven with the GPU present
but unselected), never claimed.
"""

from __future__ import annotations

from typing import Any

GPU_FORMAT = "thoughtforge-gpu-optional/1"

#: All shipped profiles, in the loader's order.
ALL_PROFILES: tuple[str, ...] = (
    "desktop_cpu", "desktop_gpu", "phone_low",
    "pi_5", "pi_zero", "server_gpu",
)

# ── The hosts, as measured ──────────────────────────────────────────────────
# This VM (the campaign host), probed live this tick.
# It has no GPU: no nvidia-smi, no torch, vram 0.0.

HOST_VM: dict[str, Any] = {
    "hostname": "htch-runtime",
    "machine": "x86_64", "cpu_count": 4,
    "ram_gb": 7.75, "vram_gb": 0.0, "gpu_name": None,
    "has_cuda": False, "has_vulkan": True,
    "auto_selected": "desktop_cpu",
    "backends_available": ["vulkan", "cpu"],
    "vulkan_basis": "library_presence_only",
    "tree_head": "2222a520b0c5ea3348c464a3f72051814f5af71f",
}

# The laptop line: purplelongship, probed live
# on-device this tick at the same tree head (sparse
# clone of this branch). GPU present (RTX 2060
# Max-Q, 6 GB — the detector auto-selects
# desktop_gpu there). The panel below ran under a
# load average of ~15.6 on 16 threads; its walls
# are filed as colour, never baselines.

HOST_LAPTOP: dict[str, Any] = {
    "hostname": "purplelongship",
    "machine": "x86_64", "cpu_count": 16,
    "cpu_model": "AMD Ryzen 9 4900HS with Radeon Graphics",
    "ram_gb": 38.59, "vram_gb": 6.0,
    "gpu_name": "NVIDIA GeForce RTX 2060 with Max-Q Design",
    "has_cuda": True, "has_vulkan": True,
    "auto_selected": "desktop_gpu",
    "backends_available": ["cuda", "vulkan", "cpu"],
    "python": "3.14.4",
    "load_average": [15.62, 15.13, 13.94],
    "tree_head": "2222a520b0c5ea3348c464a3f72051814f5af71f",
}

# ── The profile estate, judged live by the checker ──────────────────────────
# One row per shipped profile: what its JSON declares,
# what the resolver derives, what this GPU-less VM
# selects for it through the engine's own path, and
# the n_gpu_layers its model load would now receive
# (post-D-GPU-1-fix; the pre-fix server_gpu value was
# -1 under a cpu selection, desktop_gpu's cpu-selected
# value was 40 — the measured start above).

PROFILE_GPU_ROWS: dict[str, dict[str, Any]] = {
    "desktop_cpu": {
        "declared_backend": "cpu", "declared_fallbacks": [],
        "declared_gpu": None, "declared_vram_gb": 0,
        "resolved_n_gpu_layers": 0, "cpu_path": "declared_cpu",
        "selected_on_vm": "cpu", "kwargs_n_gpu_layers_on_vm": 0,
    },
    "desktop_gpu": {
        "declared_backend": "cuda",
        "declared_fallbacks": ["vulkan", "directml", "cpu"],
        "declared_gpu": "consumer", "declared_vram_gb": 12,
        "resolved_n_gpu_layers": 40, "cpu_path": "declared_fallback_cpu",
        "selected_on_vm": "vulkan", "kwargs_n_gpu_layers_on_vm": 40,
    },
    "phone_low": {
        "declared_backend": "cpu", "declared_fallbacks": [],
        "declared_gpu": None, "declared_vram_gb": 0,
        "resolved_n_gpu_layers": 0, "cpu_path": "declared_cpu",
        "selected_on_vm": "cpu", "kwargs_n_gpu_layers_on_vm": 0,
    },
    "pi_5": {
        "declared_backend": "cpu", "declared_fallbacks": [],
        "declared_gpu": None, "declared_vram_gb": 0,
        "resolved_n_gpu_layers": 0, "cpu_path": "declared_cpu",
        "selected_on_vm": "cpu", "kwargs_n_gpu_layers_on_vm": 0,
    },
    "pi_zero": {
        "declared_backend": "cpu", "declared_fallbacks": [],
        "declared_gpu": None, "declared_vram_gb": 0,
        "resolved_n_gpu_layers": 0, "cpu_path": "declared_cpu",
        "selected_on_vm": "cpu", "kwargs_n_gpu_layers_on_vm": 0,
    },
    "server_gpu": {
        "declared_backend": "cuda", "declared_fallbacks": ["cpu"],
        "declared_gpu": "datacenter", "declared_vram_gb": 24,
        "resolved_n_gpu_layers": -1, "cpu_path": "declared_fallback_cpu",
        "selected_on_vm": "cpu", "kwargs_n_gpu_layers_on_vm": 0,
    },
}

#: The backend estate's own offload declarations, as
#: each prober files them: -1 (all layers) for the
#: GPU-capable backends, 0 for directml (a build
#: property, not a kwarg) and cpu (always available,
#: priority 100 — the ultimate fallback, probed live
#: by the checker).
BACKEND_OFFLOAD: dict[str, int] = {
    "cuda": -1, "rocm": -1, "metal": -1, "vulkan": -1,
    "directml": 0, "cpu": 0,
}

# ── The CPU panels (medians of 3 runs per leg) ──────────────────────────────
# Per profile: a no-model TurboQuantEngine load with
# the backend forced to cpu, a no-model load as the
# profile declares it, and a real ThoughtForgeCore
# construction + one knowledge-only turn in temp dirs
# (the TF-069 core shape). Walls are fixture walls on
# shared machines — shapes, not model speeds.

CPU_PANEL_VM: dict[str, dict[str, Any]] = {
    "desktop_cpu": {"engine_load_cpu_ms": 63.988, "engine_cpu_backend": "cpu",
                    "engine_load_declared_ms": 32.037, "engine_declared_backend": "cpu",
                    "core_plus_turn_ms": 58.898, "turn_ok": True},
    "desktop_gpu": {"engine_load_cpu_ms": 20.592, "engine_cpu_backend": "cpu",
                    "engine_load_declared_ms": 22.481, "engine_declared_backend": "vulkan",
                    "core_plus_turn_ms": 4.012, "turn_ok": True},
    "phone_low": {"engine_load_cpu_ms": 6.218, "engine_cpu_backend": "cpu",
                  "engine_load_declared_ms": 13.373, "engine_declared_backend": "cpu",
                  "core_plus_turn_ms": 3.65, "turn_ok": True},
    "pi_5": {"engine_load_cpu_ms": 29.392, "engine_cpu_backend": "cpu",
             "engine_load_declared_ms": 11.25, "engine_declared_backend": "cpu",
             "core_plus_turn_ms": 10.874, "turn_ok": True},
    "pi_zero": {"engine_load_cpu_ms": 15.763, "engine_cpu_backend": "cpu",
                "engine_load_declared_ms": 15.25, "engine_declared_backend": "cpu",
                "core_plus_turn_ms": 6.499, "turn_ok": True},
    "server_gpu": {"engine_load_cpu_ms": 18.57, "engine_cpu_backend": "cpu",
                   "engine_load_declared_ms": 66.504, "engine_declared_backend": "cpu",
                   "core_plus_turn_ms": 4.123, "turn_ok": True},
}

#: The same panel on the laptop line (purplelongship,
#: load average ~15.6 — see HOST_LAPTOP). The declared
#: leg binds cuda for the two GPU profiles there, as
#: their JSONs intend on a GPU machine.
CPU_PANEL_LAPTOP: dict[str, dict[str, Any]] = {
    "desktop_cpu": {"engine_load_cpu_ms": 975.493, "engine_cpu_backend": "cpu",
                    "engine_load_declared_ms": 1325.467, "engine_declared_backend": "cpu",
                    "core_plus_turn_ms": 1.282, "turn_ok": True},
    "desktop_gpu": {"engine_load_cpu_ms": 1184.621, "engine_cpu_backend": "cpu",
                    "engine_load_declared_ms": 1318.055, "engine_declared_backend": "cuda",
                    "core_plus_turn_ms": 1.14, "turn_ok": True},
    "phone_low": {"engine_load_cpu_ms": 1233.098, "engine_cpu_backend": "cpu",
                  "engine_load_declared_ms": 1125.153, "engine_declared_backend": "cpu",
                  "core_plus_turn_ms": 1.337, "turn_ok": True},
    "pi_5": {"engine_load_cpu_ms": 1205.072, "engine_cpu_backend": "cpu",
             "engine_load_declared_ms": 1271.91, "engine_declared_backend": "cpu",
             "core_plus_turn_ms": 2.043, "turn_ok": True},
    "pi_zero": {"engine_load_cpu_ms": 1357.764, "engine_cpu_backend": "cpu",
                "engine_load_declared_ms": 1371.904, "engine_declared_backend": "cpu",
                "core_plus_turn_ms": 1.356, "turn_ok": True},
    "server_gpu": {"engine_load_cpu_ms": 1040.627, "engine_cpu_backend": "cpu",
                   "engine_load_declared_ms": 1158.987, "engine_declared_backend": "cuda",
                   "core_plus_turn_ms": 1.074, "turn_ok": True},
}

# ── The rules ───────────────────────────────────────────────────────────────

GPU_RULES: tuple[dict, ...] = (
    {"id": "four_profiles_declare_cpu_outright",
     "rule": "desktop_cpu / phone_low / pi_5 / pi_zero declare "
             "inference_backend cpu with vram_gb 0 and gpu null "
             "in their shipped JSONs — their CPU path is their "
             "declared path, not a degradation"},
    {"id": "gpu_profiles_declare_a_cpu_fallback",
     "rule": "the two GPU profiles name cpu in their own "
             "declared fallback lists (desktop_gpu: vulkan / "
             "directml / cpu; server_gpu: cpu) — no shipped "
             "profile's only path is a GPU"},
    {"id": "cpu_backend_is_always_available",
     "rule": "the cpu backend prober is unconditionally "
             "available (priority 100, 'always available'), and "
             "select_backend falls back to it — constructing a "
             "minimal cpu backend even if detection itself "
             "failed — so backend selection can never strand a "
             "profile for want of a GPU"},
    {"id": "selected_backend_governs_offload",
     "rule": "D-GPU-1 (found + FIXED by TF-083): the resolved "
             "n_gpu_layers ladder is derived from the profile's "
             "declared VRAM, so before the fix a GPU profile "
             "that fell back to cpu on a GPU-less machine still "
             "handed its GPU count to Llama() (measured: "
             "server_gpu selected cpu, kwargs -1; desktop_gpu "
             "cpu-selected would have carried 40). "
             "build_llama_kwargs now clamps the count to 0 for "
             "any backend whose own declared kwargs carry 0 "
             "(cpu, directml); a GPU-capable backend keeps the "
             "caller's count exactly, and the engine's load log "
             "reports the effective count from the built kwargs"},
    {"id": "gpu_less_vm_runs_every_profile",
     "rule": "on this GPU-less VM (vram 0.0, has_cuda False, "
             "detector auto-selects desktop_cpu) every "
             "profile's forced-CPU engine load succeeds with "
             "backend cpu, every profile's as-declared load "
             "succeeds (desktop_gpu via vulkan — D-GPU-2's "
             "library-grade probe; server_gpu via its declared "
             "cpu fallback), and every profile's core answers "
             "a knowledge-only turn (CPU_PANEL_VM, medians "
             "of 3)"},
    {"id": "laptop_line_cpu_paths_measured",
     "rule": "on purplelongship (RTX 2060 Max-Q present, "
             "detector auto-selects desktop_gpu) every "
             "profile's forced-CPU engine load also succeeds "
             "with backend cpu, the GPU profiles' declared "
             "loads bind cuda, and every core answers "
             "(CPU_PANEL_LAPTOP, medians of 3 under a load "
             "average of ~15.6 on 16 threads — the walls are "
             "contention colour, never baselines)"},
    {"id": "no_other_hard_gpu_requirement",
     "rule": "a sweep of src/ finds no other hard GPU "
             "requirement: the ONNX embedder defaults "
             "use_gpu=False and always lists the CPU execution "
             "provider (CUDA prepended only on request), the "
             "HF backend is a remote API client, embeddings "
             "are CPU-shaped per TF-053, and the cognition "
             "core names no device"},
    {"id": "vulkan_probe_is_library_grade",
     "rule": "D-GPU-2 (OPEN, TF-086's truth table v2): both "
             "vulkan probes accept libvulkan.so.1's presence "
             "as a working backend — on this VM the library "
             "exists, vulkaninfo does not, and no GPU device "
             "exists, yet vulkan reports available and "
             "desktop_gpu selects it. A library is not a "
             "device; the probe's truth grade is filed, the "
             "verdict is TF-086's"},
    {"id": "live_model_cpu_generation_not_proven",
     "rule": "a live model's generation on CPU is NOT "
             "PROVEN on either host: llama_cpp is absent on "
             "this VM and on the laptop (measured this tick), "
             "and no GGUF exists in the tree — every load "
             "above is a no-model load and every turn a "
             "knowledge-only one. TF-127's live panel / "
             "TF-175's laptop panel own the road, and "
             "TF-086's truth table must carry the line"},
    {"id": "gpu_less_laptop_not_proven",
     "rule": "a GPU-less laptop is NOT PROVEN as hardware: "
             "the laptop line's machine carries a GPU, and "
             "its CPU paths are proven with the GPU present "
             "but unselected (forced-cpu loads, the "
             "detector's desktop_gpu auto-selection left "
             "unused by those loads). The GPU-less host "
             "proof is this VM's, at desktop_cpu grade"},
)

# ── Export ──────────────────────────────────────────────────────────────────


def gpu_as_data() -> dict[str, Any]:
    return {
        "format": GPU_FORMAT,
        "profiles": list(ALL_PROFILES),
        "host_vm": dict(HOST_VM),
        "host_laptop": dict(HOST_LAPTOP),
        "profile_rows": {pid: dict(row)
                         for pid, row in PROFILE_GPU_ROWS.items()},
        "backend_offload": dict(BACKEND_OFFLOAD),
        "cpu_panel_vm": {pid: dict(row)
                         for pid, row in CPU_PANEL_VM.items()},
        "cpu_panel_laptop": {pid: dict(row)
                             for pid, row in CPU_PANEL_LAPTOP.items()},
        "rules": [dict(r) for r in GPU_RULES],
        "proof": {
            "no_profile_requires_gpu": "proven_fixture_two_hosts",
            "cpu_paths_vm": "measured_medians_of_3",
            "cpu_paths_laptop_line": "measured_medians_of_3_under_contention",
            "live_model_cpu_generation": "not_proven",
            "gpu_less_laptop_hardware": "not_proven",
        },
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def _panels_problems() -> list[str]:
    problems: list[str] = []
    for label, panel in (("vm", CPU_PANEL_VM),
                         ("laptop", CPU_PANEL_LAPTOP)):
        if set(panel) != set(ALL_PROFILES):
            problems.append(f"{label} panel does not cover all profiles")
        for pid, row in panel.items():
            if row["engine_cpu_backend"] != "cpu":
                problems.append(f"{label}/{pid}: forced-CPU load did "
                                f"not land on cpu")
            if row["turn_ok"] is not True:
                problems.append(f"{label}/{pid}: core turn not ok")
            if any(row[k] <= 0 for k in ("engine_load_cpu_ms",
                                         "engine_load_declared_ms",
                                         "core_plus_turn_ms")):
                problems.append(f"{label}/{pid}: a panel wall is "
                                f"not positive")
    laptop_declared = {pid: CPU_PANEL_LAPTOP[pid]["engine_declared_backend"]
                       for pid in ("desktop_gpu", "server_gpu")}
    if laptop_declared != {"desktop_gpu": "cuda", "server_gpu": "cuda"}:
        problems.append("the GPU profiles' declared laptop loads no "
                        "longer bind cuda there")
    if CPU_PANEL_VM["server_gpu"]["engine_declared_backend"] != "cpu":
        problems.append("server_gpu's declared VM load no longer "
                        "falls back to cpu")
    return problems


def gpu_problems() -> list[str]:
    """The filed estate judged against the live tree."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in GPU_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("GPU_RULES ids are not unique")

    from thoughtforge.inference import backends as backends_module
    from thoughtforge.inference.profiles import resolve_generation_params
    from thoughtforge.utils.config import (
        list_hardware_profiles,
        load_hardware_profile,
    )

    if tuple(sorted(list_hardware_profiles())) != tuple(sorted(ALL_PROFILES)):
        problems.append("the shipped profile set drifted from ALL_PROFILES")
    if set(PROFILE_GPU_ROWS) != set(ALL_PROFILES):
        problems.append("profile rows do not cover exactly the profiles")

    probers = {
        "cuda": backends_module._probe_cuda,
        "rocm": backends_module._probe_rocm,
        "metal": backends_module._probe_metal,
        "vulkan": backends_module._probe_vulkan,
        "directml": backends_module._probe_directml,
        "cpu": backends_module._probe_cpu,
    }
    for name, filed in BACKEND_OFFLOAD.items():
        info = probers[name]()
        if info.llama_cpp_kwargs.get("n_gpu_layers", 0) != filed:
            problems.append(f"backend {name}: live prober declares "
                            f"{info.llama_cpp_kwargs.get('n_gpu_layers')} "
                            f"!= filed {filed}")
    cpu_info = probers["cpu"]()
    if cpu_info.available is not True:
        problems.append("the cpu backend is no longer always available")

    for pid, row in PROFILE_GPU_ROWS.items():
        profile = load_hardware_profile(pid)
        hw = profile["hardware"]
        params = resolve_generation_params(profile)
        if row["declared_backend"] != hw["inference_backend"] \
                or row["declared_backend"] != params.inference_backend:
            problems.append(f"{pid}: declared backend drifted")
        if row["declared_fallbacks"] != hw.get("inference_backend_fallback", []) \
                or row["declared_fallbacks"] != params.inference_backend_fallbacks:
            problems.append(f"{pid}: declared fallbacks drifted")
        if row["declared_gpu"] != hw.get("gpu") \
                or row["declared_vram_gb"] != hw.get("vram_gb"):
            problems.append(f"{pid}: declared gpu/vram drifted")
        if row["resolved_n_gpu_layers"] != params.n_gpu_layers:
            problems.append(f"{pid}: resolved ladder drifted from "
                            f"the resolver")
        expect_cpu_path = "declared_cpu" if hw["inference_backend"] == "cpu" \
            else "declared_fallback_cpu"
        if row["cpu_path"] != expect_cpu_path:
            problems.append(f"{pid}: cpu_path classification drifted")
        if hw["inference_backend"] != "cpu" \
                and "cpu" not in hw.get("inference_backend_fallback", []):
            problems.append(f"{pid}: a non-CPU profile lost its cpu "
                            f"fallback — a secret GPU requirement")
        # The VM selection + effective kwargs, through the
        # engine's own selection path (TF-083's live host).
        selected = backends_module.select_backend(
            preferred=params.inference_backend,
            fallbacks=params.inference_backend_fallbacks)
        if selected.name != row["selected_on_vm"]:
            problems.append(f"{pid}: VM selection {selected.name} != "
                            f"filed {row['selected_on_vm']}")
        kwargs = backends_module.build_llama_kwargs(
            selected, "x.gguf", n_gpu_layers=params.n_gpu_layers)
        if kwargs["n_gpu_layers"] != row["kwargs_n_gpu_layers_on_vm"]:
            problems.append(f"{pid}: effective kwargs "
                            f"{kwargs['n_gpu_layers']} != filed "
                            f"{row['kwargs_n_gpu_layers_on_vm']}")
        if selected.llama_cpp_kwargs.get("n_gpu_layers", 0) == 0 \
                and kwargs["n_gpu_layers"] != 0:
            problems.append(f"{pid}: D-GPU-1 regressed — a no-offload "
                            f"backend was handed GPU layers")

    # Host records: coherent, and the two heads are one.
    if HOST_VM["tree_head"] != HOST_LAPTOP["tree_head"] \
            or len(HOST_VM["tree_head"]) != 40:
        problems.append("the two host records do not share one "
                        "commit-shaped tree head")
    if HOST_VM["vram_gb"] != 0.0 or HOST_VM["has_cuda"] is not False \
            or HOST_VM["gpu_name"] is not None:
        problems.append("the VM host record no longer reads GPU-less")
    if HOST_LAPTOP["has_cuda"] is not True \
            or "RTX 2060" not in (HOST_LAPTOP["gpu_name"] or ""):
        problems.append("the laptop host record drifted from the "
                        "measured RTX 2060 machine")

    problems.extend(_panels_problems())
    return problems


def gpu_live_problems() -> list[str]:
    """The clamp and the CPU paths, judged behaviourally
    on this machine's tree."""
    problems: list[str] = []
    from thoughtforge.inference.backends import (
        BackendInfo,
        build_llama_kwargs,
        select_backend,
    )
    from thoughtforge.inference.turboquant import TurboQuantEngine
    from thoughtforge.utils.config import list_hardware_profiles

    cpu = select_backend("cpu")
    if cpu.name != "cpu" or not cpu.available:
        problems.append("cpu is not selectable on this machine")
    if build_llama_kwargs(cpu, "x.gguf", n_gpu_layers=32)["n_gpu_layers"] != 0:
        problems.append("D-GPU-1 regressed: cpu + caller 32 layers "
                        "is not clamped to 0")
    cuda_shaped = BackendInfo(name="cuda", available=True, priority=1,
                              llama_cpp_kwargs={"n_gpu_layers": -1})
    if build_llama_kwargs(cuda_shaped, "x.gguf",
                          n_gpu_layers=32)["n_gpu_layers"] != 32:
        problems.append("a GPU-capable backend no longer keeps the "
                        "caller's layer count")
    unknown = select_backend("nonexistent_backend_tf083")
    if unknown.name != "cpu":
        problems.append("an unknown backend preference no longer "
                        "falls back to cpu")

    for pid in list_hardware_profiles():
        engine = TurboQuantEngine(model_path=None, profile_id=pid,
                                  backend="cpu")
        try:
            engine.load()
        finally:
            backend_name = engine.backend_name
            engine.unload()
        if backend_name != "cpu":
            problems.append(f"{pid}: forced-CPU engine load landed "
                            f"on {backend_name}")
    return problems


__all__ = [
    "ALL_PROFILES",
    "BACKEND_OFFLOAD",
    "CPU_PANEL_LAPTOP",
    "CPU_PANEL_VM",
    "GPU_FORMAT",
    "GPU_RULES",
    "HOST_LAPTOP",
    "HOST_VM",
    "PROFILE_GPU_ROWS",
    "gpu_as_data",
    "gpu_live_problems",
    "gpu_problems",
]
