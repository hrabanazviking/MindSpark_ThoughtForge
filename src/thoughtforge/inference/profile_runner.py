"""TF-069 — Profile runner: every hardware profile boots its
stack *in simulation*, with its RAM ceiling enforced as a
hard limit.

Roadmap: "each hardware profile boots its stack in
simulation (budgets enforced: RAM ceilings as hard test
limits)".

Measured start (probed this tick, VM):

- No runner existed. The pieces booted separately — a
  no-model ``TurboQuantEngine`` load per profile and a
  knowledge-only ``ThoughtForgeCore`` per profile both
  succeed — but nothing booted a profile end-to-end, and
  the engine leg is *host*-shaped, not profile-shaped:
  on this VM a desktop_gpu load selected vulkan and a
  server_gpu load fell back to cpu, whatever the profile
  declares. Simulation must therefore build its hardware
  from the profile's own JSON, never from host detection.
- The profile JSONs' ``deployment`` section,
  ``model.max_params_b`` and ``model.recommended_models``
  were still read by no code (D-PROF-1's remaining share,
  TF-056 having closed the memory/retrieval share).
- The selector's floors were never tested against the
  profiles' own declared machines (D-PROF-2's TF-069
  share).

THE RUNNER (``run_profile_boot`` / ``run_all_boots``,
format ``thoughtforge-profile-runner/1``): for one
profile, in order —

1. ``profile`` — the shipped JSON loads live through the
   tree's own loader; the id matches.
2. ``params`` — ``resolve_generation_params`` resolves;
   the resolved quant is TF-068's policy default.
3. ``selector`` — a ``HardwareSnapshot`` built from the
   profile's *declared* hardware (ram/vram/arch from the
   JSON; cuda when the declared backend is cuda;
   raspberry-pi for the pi profiles, mobile for
   phone_low — simulation inputs, stated as such) is fed
   to the selector's own rule: it must select the profile
   itself.
4. ``memory_budget`` — TF-056's budget resolves live and
   a fresh store under the booted core reports within
   budget; the retrieval plan only caps.
5. ``core`` — a real ``ThoughtForgeCore`` boots in temp
   dirs on the profile (its lifecycle caps are the
   profile's budget caps, read live) and answers one
   knowledge-only turn.
6. ``engine`` — a real no-model ``TurboQuantEngine``
   load resolves the same params and records TF-068's
   ``no_model`` quant verdict. The backend it selects is
   the *host's* truth, recorded as ``host_backend`` and
   never claimed as the profile's declared backend.
7. ``ram_ceiling`` — the HARD leg (below).
8. ``recommended_models`` — each recommended name judged
   against the profile's own ``max_params_b`` where the
   name carries a parameter count; a name carrying none
   is ``unprovable``, never guessed.
9. ``deployment`` — the deployment section read live and
   reported (platforms, runtime, onnx_export, and every
   extra key — pi_zero's knowledge_subset, server_gpu's
   multi_gpu). Reported as data only: a ``docker`` entry
   in a shipped runtime list is the profile's claim, and
   this runner never runs Docker.

THE RAM CEILING (hard, not advisory):

- Accounting: a model's accounted size is its parameter
  count × a GB-per-billion-parameters figure. For the
  catalogue's quants the figure is *calibrated from the
  tree itself*: ``catalogue_gb_per_b`` parses each
  curated catalogue entry's parameter count out of its
  name/description and takes, per declared quant, the
  MAXIMUM size/params ratio in the catalogue (the
  conservative end — a ceiling is judged at the worst
  in-tree figure, not the average). Entries whose names
  carry no count (the Phi-3 rows, Mistral-Nemo's filename)
  are counted as unparseable and never used. fp16 needs
  no calibration: 16 bits = 2.0 GB per billion parameters
  by definition. **q2_k has no catalogue entry at all** —
  its accounted size is ``unprovable`` in-tree, filed as
  a gap, never estimated from a neighbouring quant.
- A boot may carry a *requested model* (params + quant):
  if its accounted size exceeds the profile's total
  declared memory (ram_gb + vram_gb), the boot is
  REFUSED before any stack leg runs — the ceiling is a
  gate, not a warning. An unprovable requested quant is
  likewise refused (a ceiling that cannot be computed
  cannot be passed silently).
- With no requested model, the leg judges the profile's
  own cap: ``max_params_b`` at the profile's default
  quant against the same ceiling, filed as data
  (``cap_fit``: pass / fail / unprovable).

MEASURED CEILING RESULTS (filed as ``RAM_ROWS`` /
``RECOMMENDED_FAILURES`` below, recomputed by the
checker on every run; D-PROF-5 in the TF-016 docket):

- server_gpu's cap FAILS its own ceiling: 70B at its
  fp16 default accounts 140.0 GB against 64 + 24 = 88 GB
  declared. Its q4_k_m recommendation would fit; the
  fp16-at-the-cap promise does not.
- pi_5 (2.0 / 4 GB), desktop_cpu (4.67 / 8 GB) and
  desktop_gpu (14.3 / 28 GB) caps pass at the catalogue's
  worst Q4_K_M / Q8_0 figures.
- phone_low + pi_zero caps are UNPROVABLE (q2_k
  uncalibrated in-tree) — the two TF-068 constrained
  exceptions keep their stated debt; TF-127's live panel
  or a shipped q2_k size figure is the road that proves
  them.
- Two recommended names exceed their own profile's cap:
  pi_zero recommends TinyLlama-1.1B against a 0.5B cap,
  desktop_cpu recommends LLaMA-3-8B against a 7.0B cap.
  The Phi-3 recommendations carry no count in their
  names and are filed unprovable (pi_5, desktop_cpu).

VERDICTS: ``boots`` (every leg passes),
``boots_with_ceiling_gaps`` (the stack boots; a ceiling
or recommendation leg fails or is unprovable — gaps
named), ``refused`` (a requested model breaks the hard
ceiling — no stack leg ran), ``failed`` (a stack leg
failed). D-PROF-1 narrows again (deployment /
max_params / recommended_models now read live here;
TF-110 keeps the subset craft) and D-PROF-2 narrows
(self-selection on declared hardware proven for all
six; floor mismatches on *other* machines stay with
TF-086's truth table v2).

Proof grade FIXTURE/SIMULATION (real core + engine boots
in temp dirs on this VM, simulated hardware from the
profiles' own declarations) — a boot on real phone/Pi/
server hardware is NOT PROVEN (TF-082 owns the real-Pi
road), a loaded model's real footprint is NOT PROVEN
(the accounting is catalogue-calibrated, not weighed),
never claimed.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thoughtforge.inference.model_browser import ModelBrowser
from thoughtforge.inference.profiles import (
    HardwareSnapshot,
    ProfileSelector,
    resolve_generation_params,
)
from thoughtforge.inference.quant_policy import default_for_profile
from thoughtforge.utils.config import (
    list_hardware_profiles,
    load_hardware_profile,
)

RUNNER_FORMAT = "thoughtforge-profile-runner/1"

BOOT_LEGS: tuple[str, ...] = (
    "profile", "params", "selector", "memory_budget", "core",
    "engine", "ram_ceiling", "recommended_models", "deployment",
)

# ── Catalogue calibration ───────────────────────────────────────────────────
# GB per billion parameters, per quant, derived LIVE from
# the curated catalogue: for each entry whose name or
# description carries a parameter count, ratio = size_gb /
# params_b; the filed figure is the MAXIMUM ratio (the
# conservative end for a ceiling). fp16 is definitional,
# not calibrated. Any other quant (q2_k above all) has no
# figure and is unprovable — never borrowed from a
# neighbour.

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*B\b")

#: Definitional GB-per-B figures (no calibration needed).
DEFINITIONAL_GB_PER_B: dict[str, float] = {"fp16": 2.0}


def params_from_name(name: str) -> float | None:
    """The parameter count a model name carries, in
    billions — the largest ``<number>B`` token, or None
    when the name carries none (never guessed)."""
    hits = [float(h) for h in _PARAM_RE.findall(name)]
    return max(hits) if hits else None


def catalogue_gb_per_b() -> dict[str, dict[str, Any]]:
    """Per-quant calibration from the curated catalogue,
    computed live: {quant_lower: {max_gb_per_b, entries,
    unparseable}}."""
    out: dict[str, dict[str, Any]] = {}
    for entry in ModelBrowser.CURATED_MODELS:
        quant = entry.quantization.lower()
        row = out.setdefault(
            quant, {"max_gb_per_b": None, "entries": 0, "unparseable": 0}
        )
        params = params_from_name(f"{entry.model_id} {entry.filename}") \
            or params_from_name(entry.description)
        if params is None:
            row["unparseable"] += 1
            continue
        row["entries"] += 1
        ratio = entry.size_gb / params
        if row["max_gb_per_b"] is None or ratio > row["max_gb_per_b"]:
            row["max_gb_per_b"] = ratio
    return out


def gb_per_b(quant: str) -> tuple[float | None, str]:
    """(GB per billion params, basis) for a quant:
    'definition' for fp16, 'catalogue_max' when the
    catalogue calibrates it, (None, 'uncalibrated') when
    the tree states no figure."""
    quant = quant.lower()
    if quant in DEFINITIONAL_GB_PER_B:
        return DEFINITIONAL_GB_PER_B[quant], "definition"
    row = catalogue_gb_per_b().get(quant)
    if row and row["max_gb_per_b"] is not None:
        return row["max_gb_per_b"], "catalogue_max"
    return None, "uncalibrated"


def accounted_model_gb(params_b: float, quant: str) -> tuple[float | None, str]:
    """A model's accounted size in GB under the runner's
    accounting, or (None, basis) when unprovable."""
    per_b, basis = gb_per_b(quant)
    if per_b is None:
        return None, basis
    return round(params_b * per_b, 4), basis


# ── The ceiling verdict ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class FitVerdict:
    """One model judged against one profile's RAM ceiling."""

    profile_id: str
    params_b: float
    quant: str
    accounted_gb: float | None    # None = unprovable in-tree
    ceiling_gb: float             # ram_gb + vram_gb, as declared
    status: str                   # fit | exceeds_ceiling | unprovable
    basis: str                    # definition | catalogue_max | uncalibrated

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id, "params_b": self.params_b,
            "quant": self.quant, "accounted_gb": self.accounted_gb,
            "ceiling_gb": self.ceiling_gb, "status": self.status,
            "basis": self.basis,
        }


def _ceiling_gb(profile: dict[str, Any]) -> float:
    hw = profile["hardware"]
    return float(hw["ram_gb"]) + float(hw["vram_gb"])


def check_model_fit(
    profile_id: str, params_b: float, quant: str
) -> FitVerdict:
    """Judge a model (params + quant) against a profile's
    declared RAM ceiling. The ceiling is hard: an
    accounted size above it is ``exceeds_ceiling``, and a
    quant the tree cannot account is ``unprovable`` —
    neither is a fit."""
    profile = load_hardware_profile(profile_id)
    accounted, basis = accounted_model_gb(params_b, quant)
    ceiling = _ceiling_gb(profile)
    if accounted is None:
        status = "unprovable"
    elif accounted <= ceiling:
        status = "fit"
    else:
        status = "exceeds_ceiling"
    return FitVerdict(
        profile_id=profile_id, params_b=params_b, quant=quant.lower(),
        accounted_gb=accounted, ceiling_gb=ceiling, status=status,
        basis=basis,
    )


# ── Filed measured rows (recomputed by the checker) ─────────────────────────

#: The cap row per profile: max_params_b at the profile's
#: default quant against its ceiling — the measured
#: outcome of the ram_ceiling leg with no requested model.
RAM_ROWS: dict[str, str] = {
    "phone_low": "unprovable",     # q2_k uncalibrated in-tree
    "pi_zero": "unprovable",       # q2_k uncalibrated in-tree
    "pi_5": "fit",
    "desktop_cpu": "fit",
    "desktop_gpu": "fit",
    "server_gpu": "exceeds_ceiling",  # 70B fp16 = 140 GB > 88 GB
}

#: Recommended names that exceed their own profile's
#: max_params_b (D-PROF-5), as (profile, name) pairs.
RECOMMENDED_FAILURES: frozenset[tuple[str, str]] = frozenset({
    ("pi_zero", "TinyLlama-1.1B-Chat-q2_k_subset"),
    ("desktop_cpu", "LLaMA-3-8B-Instruct-q4_k_m"),
})

#: Overall boot verdict per shipped profile (no requested
#: model): the stack boots everywhere; only desktop_gpu
#: is gap-free.
BOOT_VERDICTS: dict[str, str] = {
    "phone_low": "boots_with_ceiling_gaps",
    "pi_zero": "boots_with_ceiling_gaps",
    "pi_5": "boots_with_ceiling_gaps",
    "desktop_cpu": "boots_with_ceiling_gaps",
    "desktop_gpu": "boots",
    "server_gpu": "boots_with_ceiling_gaps",
}


# ── The boot ────────────────────────────────────────────────────────────────

@dataclass
class BootReport:
    """One profile's simulated boot, as data."""

    profile_id: str
    verdict: str = "failed"   # boots | boots_with_ceiling_gaps | refused | failed
    legs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": RUNNER_FORMAT,
            "profile_id": self.profile_id,
            "verdict": self.verdict,
            "legs": {k: dict(v) for k, v in self.legs.items()},
        }


def _simulated_snapshot(profile: dict[str, Any]) -> HardwareSnapshot:
    """The profile's declared machine, as a snapshot —
    simulation inputs derived from the JSON alone (arch =
    the first declared arch; cuda iff the declared
    backend is cuda; the pi/phone class flags from the
    profile id, as the JSONs' own descriptions state)."""
    hw = profile["hardware"]
    pid = profile["profile_id"]
    arch = hw["cpu_arch"][0]
    return HardwareSnapshot(
        ram_gb=float(hw["ram_gb"]),
        vram_gb=float(hw["vram_gb"]),
        cpu_arch=arch,
        os_name="linux",
        cpu_count=4,
        gpu_name=hw.get("gpu"),
        has_cuda=hw.get("inference_backend") == "cuda",
        is_arm=arch.startswith("arm"),
        is_raspberry_pi=pid.startswith("pi_"),
        is_mobile=pid == "phone_low",
    )


def run_profile_boot(
    profile_id: str,
    requested_model: tuple[float, str] | None = None,
) -> BootReport:
    """Boot one profile's stack in simulation.

    ``requested_model`` is an optional (params_b, quant)
    pair the boot must carry: when its accounted size
    exceeds the profile's RAM ceiling — or cannot be
    accounted in-tree — the boot is REFUSED before any
    stack leg runs (the ceiling is hard). Unknown profiles
    raise ValueError, as everywhere in this tree."""
    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile for profile runner: {profile_id!r} "
            f"(valid: {valid})"
        )
    profile = load_hardware_profile(profile_id)
    report = BootReport(profile_id=profile_id)

    # The hard ceiling gates first.
    if requested_model is not None:
        params_b, quant = requested_model
        fit = check_model_fit(profile_id, params_b, quant)
        report.legs["ram_ceiling"] = {
            "status": "pass" if fit.status == "fit" else "fail",
            "requested": fit.to_dict(),
        }
        if fit.status != "fit":
            report.verdict = "refused"
            for leg in BOOT_LEGS:
                report.legs.setdefault(leg, {"status": "not_run"})
            return report

    gaps = False

    # 1 — profile
    report.legs["profile"] = {
        "status": "pass" if profile["profile_id"] == profile_id else "fail",
        "ram_gb": profile["hardware"]["ram_gb"],
        "vram_gb": profile["hardware"]["vram_gb"],
    }

    # 2 — params + quant policy
    params = resolve_generation_params(profile)
    policy_quant = default_for_profile(profile_id)
    report.legs["params"] = {
        "status": "pass" if params.quantization == policy_quant else "fail",
        "quantization": params.quantization,
        "policy_quant": policy_quant,
        "inference_backend": params.inference_backend,
        "max_context_tokens": params.max_context_tokens,
        "n_gpu_layers": params.n_gpu_layers,
    }

    # 3 — selector on the declared machine
    selected = ProfileSelector._select_id(_simulated_snapshot(profile))
    report.legs["selector"] = {
        "status": "pass" if selected == profile_id else "fail",
        "selected": selected,
    }

    # 7 (filed before the boots it judges) — cap row
    cap = check_model_fit(
        profile_id,
        float(profile["model"]["max_params_b"]),
        profile["model"]["quantization"],
    )
    if requested_model is None:
        report.legs["ram_ceiling"] = {
            "status": "pass" if cap.status == "fit" else "fail",
            "cap": cap.to_dict(),
        }
    else:
        report.legs["ram_ceiling"]["cap"] = cap.to_dict()
    if cap.status != "fit":
        gaps = True

    # 8 — recommended models vs the profile's own cap
    max_params = float(profile["model"]["max_params_b"])
    rec_rows = []
    for name in profile["model"]["recommended_models"]:
        carried = params_from_name(name)
        if carried is None:
            status = "unprovable"
            gaps = True
        elif carried <= max_params:
            status = "within_cap"
        else:
            status = "exceeds_cap"
            gaps = True
        rec_rows.append({
            "name": name, "params_b": carried,
            "max_params_b": max_params, "status": status,
        })
    report.legs["recommended_models"] = {
        "status": "fail" if any(r["status"] == "exceeds_cap" for r in rec_rows)
                  else "pass",
        "rows": rec_rows,
    }

    # 9 — deployment, read live
    deployment = profile["deployment"]
    known = {"platforms", "runtime", "onnx_export"}
    report.legs["deployment"] = {
        "status": "pass",
        "platforms": list(deployment["platforms"]),
        "runtime": list(deployment["runtime"]),
        "onnx_export": bool(deployment["onnx_export"]),
        "extras": {k: deployment[k] for k in sorted(deployment) if k not in known},
    }

    # 4–6 — the stack itself, in temp dirs
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.inference.turboquant import TurboQuantEngine
    from thoughtforge.knowledge.memory_budgets import (
        budget_for_profile,
        budget_report,
        retrieval_plan,
    )

    with tempfile.TemporaryDirectory(prefix="tf069-") as tmp:
        root = Path(tmp)
        try:
            core = ThoughtForgeCore(
                config={"conversation": {"profile_id": profile_id}},
                memory_dir=root / "memory",
                db_path=root / "knowledge.db",
            )
            result = core.think("What is the ferry?")
            core_ok = bool(result.text)
            lifecycle_caps = core._lifecycle.caps
            budget = budget_for_profile(profile_id)
            caps_match = all(
                lifecycle_caps[kind] == budget.caps[kind]
                for kind in budget.caps
            )
            report_leg = budget_report(core._store, profile_id)
            plan = retrieval_plan(profile_id)
            report.legs["core"] = {
                "status": "pass" if core_ok else "fail",
                "max_response_tokens":
                    core._conversation_config.max_response_tokens,
                "max_refinement_passes":
                    core._conversation_config.max_refinement_passes,
                "response_chars": len(result.text or ""),
            }
            report.legs["memory_budget"] = {
                "status": "pass" if (caps_match and report_leg.within_budget
                                     and plan.bundle_top_k <=
                                     budget.max_memory_bundle_facts)
                          else "fail",
                "caps_from_profile": caps_match,
                "fresh_store_within_budget": report_leg.within_budget,
                "bundle_top_k": plan.bundle_top_k,
            }
        except Exception as exc:  # a stack that cannot boot is data
            report.legs["core"] = {
                "status": "fail", "error": f"{type(exc).__name__}: {exc}",
            }
            report.legs["memory_budget"] = {"status": "fail"}

        engine = TurboQuantEngine(profile_id=profile_id)
        try:
            engine.load()
            verdict = engine.quant_verdict
            report.legs["engine"] = {
                "status": "pass" if (
                    verdict is not None and verdict.status == "no_model"
                    and verdict.expected == policy_quant
                    and engine._params is not None
                    and engine._params.quantization == params.quantization
                ) else "fail",
                "host_backend": engine._backend.name if engine._backend else None,
                "quant_verdict": verdict.status if verdict else None,
            }
        except Exception as exc:
            report.legs["engine"] = {
                "status": "fail", "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            engine.unload()

    stack_legs = ("profile", "params", "selector", "memory_budget",
                  "core", "engine", "deployment")
    if any(report.legs[leg]["status"] != "pass" for leg in stack_legs):
        report.verdict = "failed"
    elif gaps or report.legs["ram_ceiling"]["status"] != "pass" \
            or report.legs["recommended_models"]["status"] != "pass":
        report.verdict = "boots_with_ceiling_gaps"
    else:
        report.verdict = "boots"
    return report


def run_all_boots() -> dict[str, BootReport]:
    """Every shipped profile's simulated boot, in the
    loader's profile order."""
    return {pid: run_profile_boot(pid) for pid in list_hardware_profiles()}


def runner_as_data() -> dict[str, Any]:
    """The whole runner estate as JSON-safe data: the
    calibration, the filed rows, and every boot."""
    return {
        "format": RUNNER_FORMAT,
        "legs": list(BOOT_LEGS),
        "calibration_gb_per_b": catalogue_gb_per_b(),
        "definitional_gb_per_b": dict(DEFINITIONAL_GB_PER_B),
        "ram_rows": dict(RAM_ROWS),
        "recommended_failures": sorted(RECOMMENDED_FAILURES),
        "boots": {pid: r.to_dict() for pid, r in run_all_boots().items()},
    }


# ── Live checker ────────────────────────────────────────────────────────────

def profile_runner_problems() -> list[str]:
    """Re-derive the runner's claims live; [] means true."""
    problems: list[str] = []

    # Calibration pins (hand-checked against the catalogue):
    calib = catalogue_gb_per_b()
    expected_calib = {
        "q4_k_m": (2.0 / 3.0, 8),     # Llama-3.2-3B is the worst ratio
        "q5_k_m": (2.5 / 3.0, 1),
        "q8_0": (7.7 / 7.0, 2),       # Mistral-7B Q8 is the worst ratio
        "iq2_xs": (19.5 / 70.0, 2),
    }
    for quant, (want, entries) in expected_calib.items():
        row = calib.get(quant)
        if row is None or row["max_gb_per_b"] is None \
                or abs(row["max_gb_per_b"] - want) > 1e-9:
            problems.append(f"calibration {quant}: {row} != max {want}")
        elif row["entries"] != entries:
            problems.append(f"calibration {quant}: entries {row['entries']} "
                            f"!= {entries}")
    if gb_per_b("q2_k") != (None, "uncalibrated"):
        problems.append("q2_k must stay uncalibrated in-tree")
    if gb_per_b("fp16") != (2.0, "definition"):
        problems.append("fp16 must stay definitional at 2.0 GB/B")

    # Name parsing pins:
    for name, want in (
        ("TinyLlama-1.1B-Chat-q2_k_subset", 1.1),
        ("LLaMA-3-8B-Instruct-q4_k_m", 8.0),
        ("Mixtral-8x7B-Instruct-v0.1-fp16", 7.0),  # largest token; stated limit
        ("Phi-3-mini-4k-instruct-q4_k_m", None),
        ("Gemma-2B-it-q4_k_m", 2.0),
    ):
        if params_from_name(name) != want:
            problems.append(f"params_from_name({name!r}) != {want}")

    # The filed rows recompute:
    boots = run_all_boots()
    if set(boots) != set(list_hardware_profiles()):
        problems.append("boots do not cover exactly the shipped profiles")
    measured_failures: set[tuple[str, str]] = set()
    for pid, report in boots.items():
        if report.verdict != BOOT_VERDICTS[pid]:
            problems.append(f"{pid}: verdict {report.verdict} != filed "
                            f"{BOOT_VERDICTS[pid]}")
        cap_status = report.legs["ram_ceiling"]["cap"]["status"]
        if cap_status != RAM_ROWS[pid]:
            problems.append(f"{pid}: cap row {cap_status} != filed "
                            f"{RAM_ROWS[pid]}")
        for row in report.legs["recommended_models"]["rows"]:
            if row["status"] == "exceeds_cap":
                measured_failures.add((pid, row["name"]))
        for leg in ("profile", "params", "selector", "memory_budget",
                    "core", "engine", "deployment"):
            if report.legs[leg]["status"] != "pass":
                problems.append(f"{pid}: stack leg {leg} not pass: "
                                f"{report.legs[leg]}")
    if measured_failures != set(RECOMMENDED_FAILURES):
        problems.append(f"recommended failures {sorted(measured_failures)} "
                        f"!= filed {sorted(RECOMMENDED_FAILURES)}")

    # The ceiling is hard: an oversized request is refused
    # before any stack leg runs, an unprovable one likewise,
    # and a fitting one boots.
    refused = run_profile_boot("desktop_cpu", requested_model=(70.0, "q4_k_m"))
    if refused.verdict != "refused" \
            or refused.legs["core"]["status"] != "not_run":
        problems.append(f"oversized request not refused: {refused.verdict}")
    unprovable = run_profile_boot("pi_5", requested_model=(1.1, "q2_k"))
    if unprovable.verdict != "refused":
        problems.append(f"unprovable request not refused: {unprovable.verdict}")
    fitting = run_profile_boot("desktop_cpu", requested_model=(7.0, "q4_k_m"))
    if fitting.verdict == "refused":
        problems.append("a fitting request was refused")

    return problems
