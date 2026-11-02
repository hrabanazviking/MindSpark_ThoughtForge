"""
TF-083 — GPU-optional proof: no profile secretly
requires a GPU; CPU paths measured on the laptop
line.

The estate lives in
``src/thoughtforge/inference/gpu_optional.py``
(``GPU_RULES`` as data, the profile estate filed as
``PROFILE_GPU_ROWS`` / ``BACKEND_OFFLOAD``, the two
hosts as ``HOST_VM`` / ``HOST_LAPTOP``, the CPU
panels as ``CPU_PANEL_VM`` / ``CPU_PANEL_LAPTOP``,
format ``thoughtforge-gpu-optional/1``).
Write-up: ``docs/architecture/GPU_OPTIONAL.md``.

Measured start: four profiles declare cpu outright
and the two GPU profiles declare a cpu fallback —
but the machinery had one secret GPU requirement:
the resolved n_gpu_layers ladder (from the profile's
declared VRAM) was merged over the selected backend,
so on a GPU-less VM server_gpu selected cpu and
still carried -1 into its model-load kwargs
(D-GPU-1, fixed in ``build_llama_kwargs``).

Pinned here:

  * the filed estate: both checkers clean, rule ids
    unique, the export's proof grades stated,
  * the declarations: every profile's live JSON +
    resolver agree with its filed row; every non-CPU
    profile keeps cpu in its own fallback list,
  * D-GPU-1: the clamp — a no-offload backend (cpu,
    directml shape) is handed 0 whatever the caller
    resolved; a GPU-capable backend keeps the
    caller's count exactly; per profile, the
    effective VM kwargs equal the filed row,
  * selection on this GPU-less VM: cpu always
    selectable, an unknown preference falls back to
    cpu, server_gpu's declared path lands on cpu,
  * the hosts: the VM record reads GPU-less and
    auto-selects desktop_cpu; the laptop record is
    the RTX 2060 machine; both share one tree head,
  * the panels: all six profiles on both hosts —
    forced-CPU loads on cpu, core turns ok, walls
    positive; the GPU profiles' declared laptop
    loads bind cuda,
  * the live CPU paths: a forced-CPU no-model
    engine load succeeds for every profile here,
  * proof honesty: live-model CPU generation and
    GPU-less laptop hardware are filed not_proven.
"""

from __future__ import annotations

from thoughtforge.inference.backends import (
    BackendInfo,
    build_llama_kwargs,
    select_backend,
)
from thoughtforge.inference.gpu_optional import (
    ALL_PROFILES,
    BACKEND_OFFLOAD,
    CPU_PANEL_LAPTOP,
    CPU_PANEL_VM,
    GPU_FORMAT,
    GPU_RULES,
    HOST_LAPTOP,
    HOST_VM,
    PROFILE_GPU_ROWS,
    gpu_as_data,
    gpu_live_problems,
    gpu_problems,
)
from thoughtforge.inference.profiles import resolve_generation_params
from thoughtforge.utils.config import load_hardware_profile


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert gpu_problems() == []
        assert gpu_live_problems() == []

    def test_rules_and_format(self):
        ids = [r["id"] for r in GPU_RULES]
        assert len(ids) == len(set(ids)) == 10
        assert gpu_as_data()["format"] == GPU_FORMAT
        assert set(ALL_PROFILES) == set(PROFILE_GPU_ROWS)


# ── The declarations ───────────────────────────────────────────


class TestDeclarations:
    def test_four_profiles_declare_cpu_outright(self):
        for pid in ("desktop_cpu", "phone_low", "pi_5", "pi_zero"):
            row = PROFILE_GPU_ROWS[pid]
            assert row["cpu_path"] == "declared_cpu"
            assert row["declared_gpu"] is None
            assert row["declared_vram_gb"] == 0
            assert row["resolved_n_gpu_layers"] == 0

    def test_gpu_profiles_declare_a_cpu_fallback(self):
        for pid in ("desktop_gpu", "server_gpu"):
            row = PROFILE_GPU_ROWS[pid]
            assert row["cpu_path"] == "declared_fallback_cpu"
            assert "cpu" in row["declared_fallbacks"]
            live = load_hardware_profile(pid)
            assert "cpu" in live["hardware"]["inference_backend_fallback"]

    def test_rows_match_the_live_jsons_and_resolver(self):
        for pid, row in PROFILE_GPU_ROWS.items():
            profile = load_hardware_profile(pid)
            hw = profile["hardware"]
            assert row["declared_backend"] == hw["inference_backend"]
            assert row["declared_fallbacks"] == \
                hw.get("inference_backend_fallback", [])
            params = resolve_generation_params(profile)
            assert row["resolved_n_gpu_layers"] == params.n_gpu_layers


# ── D-GPU-1: the clamp ─────────────────────────────────────────


class TestOffloadClamp:
    def test_cpu_backend_is_clamped_to_zero(self):
        cpu = select_backend("cpu")
        assert build_llama_kwargs(
            cpu, "x.gguf", n_gpu_layers=32)["n_gpu_layers"] == 0
        assert build_llama_kwargs(
            cpu, "x.gguf", n_gpu_layers=-1)["n_gpu_layers"] == 0

    def test_gpu_capable_backend_keeps_the_callers_count(self):
        cuda = BackendInfo(name="cuda", available=True, priority=1,
                           llama_cpp_kwargs={"n_gpu_layers": -1})
        assert build_llama_kwargs(
            cuda, "x.gguf", n_gpu_layers=32)["n_gpu_layers"] == 32
        assert build_llama_kwargs(
            cuda, "x.gguf")["n_gpu_layers"] == -1

    def test_server_gpu_on_this_vm_is_cpu_in_fact(self):
        # The pre-fix defect, measured: server_gpu
        # selected cpu here and carried -1. Post-fix
        # the effective count under a cpu selection
        # is 0 — cpu in name and in fact.
        params = resolve_generation_params(load_hardware_profile("server_gpu"))
        assert params.n_gpu_layers == -1
        selected = select_backend(preferred=params.inference_backend,
                                  fallbacks=params.inference_backend_fallbacks)
        assert selected.name == "cpu"
        kwargs = build_llama_kwargs(selected, "x.gguf",
                                    n_gpu_layers=params.n_gpu_layers)
        assert kwargs["n_gpu_layers"] == 0

    def test_backend_offload_declarations(self):
        assert BACKEND_OFFLOAD["cpu"] == 0
        assert BACKEND_OFFLOAD["directml"] == 0
        for name in ("cuda", "rocm", "metal", "vulkan"):
            assert BACKEND_OFFLOAD[name] == -1


# ── Selection on this GPU-less VM ──────────────────────────────


class TestSelectionHere:
    def test_cpu_always_selectable(self):
        backend = select_backend("cpu")
        assert backend.name == "cpu" and backend.available is True

    def test_unknown_preference_falls_back_to_cpu(self):
        assert select_backend("nonexistent_backend_tf083").name == "cpu"


# ── The hosts and the panels ───────────────────────────────────


class TestHostsAndPanels:
    def test_the_vm_record_is_gpu_less(self):
        assert HOST_VM["vram_gb"] == 0.0
        assert HOST_VM["has_cuda"] is False
        assert HOST_VM["gpu_name"] is None
        assert HOST_VM["auto_selected"] == "desktop_cpu"

    def test_the_laptop_record_and_shared_head(self):
        assert HOST_LAPTOP["hostname"] == "purplelongship"
        assert "RTX 2060" in HOST_LAPTOP["gpu_name"]
        assert HOST_LAPTOP["has_cuda"] is True
        assert HOST_LAPTOP["auto_selected"] == "desktop_gpu"
        assert HOST_VM["tree_head"] == HOST_LAPTOP["tree_head"]
        assert len(HOST_LAPTOP["tree_head"]) == 40
        # Contention filed, not hidden.
        assert HOST_LAPTOP["load_average"][0] > 10

    def test_panels_cover_every_profile_on_cpu(self):
        for panel in (CPU_PANEL_VM, CPU_PANEL_LAPTOP):
            assert set(panel) == set(ALL_PROFILES)
            for pid, row in panel.items():
                assert row["engine_cpu_backend"] == "cpu", pid
                assert row["turn_ok"] is True, pid
                assert row["engine_load_cpu_ms"] > 0, pid
                assert row["core_plus_turn_ms"] > 0, pid

    def test_declared_loads_land_where_filed(self):
        assert CPU_PANEL_VM["server_gpu"]["engine_declared_backend"] == "cpu"
        assert CPU_PANEL_VM["desktop_gpu"]["engine_declared_backend"] == "vulkan"
        assert CPU_PANEL_LAPTOP["desktop_gpu"]["engine_declared_backend"] == "cuda"
        assert CPU_PANEL_LAPTOP["server_gpu"]["engine_declared_backend"] == "cuda"


# ── Proof honesty ──────────────────────────────────────────────


class TestProofHonesty:
    def test_not_proven_grades_stated(self):
        proof = gpu_as_data()["proof"]
        assert proof["live_model_cpu_generation"] == "not_proven"
        assert proof["gpu_less_laptop_hardware"] == "not_proven"
        rules = {r["id"]: r["rule"] for r in GPU_RULES}
        assert "NOT PROVEN" in rules["live_model_cpu_generation_not_proven"]
        assert "NOT PROVEN" in rules["gpu_less_laptop_not_proven"]
        assert "D-GPU-2" in rules["vulkan_probe_is_library_grade"]
