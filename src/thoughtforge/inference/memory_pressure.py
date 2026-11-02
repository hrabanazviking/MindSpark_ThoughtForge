"""TF-080 — Memory-pressure behaviour: at the profile's
ceiling the engine sheds load by rule (context first),
never crashes mute.

Roadmap: "Memory-pressure behaviour: at the profile's
ceiling the engine sheds load by rule (context first),
never crashes mute."

Measured start (probed live this tick, VM):

- **No memory-pressure behaviour existed in inference
  or cognition at all.** Grep for pressure / shed /
  OOM over src finds exactly one pressure response in
  the whole tree, and it is not the engine's: the
  Wikidata ETL commits early when psutil reports host
  memory over 85% (``etl/wikidata.py._check_memory``).
  That is the tree's sole precedent threshold, and it
  protects a database import, not a turn.
- **The engine's only cap is static and pressure-
  blind.** ``TurboQuantEngine._enforce_token_budget``
  clips a generation's max_tokens to
  ``min(requested, token_budget_per_turn,
  max_response_tokens)`` — the same clip at 1% memory
  and at 99%. Nothing reads any memory signal, and
  ``max_context_tokens`` is used exactly once: as the
  ``n_ctx`` passed to llama at load. No prompt or
  context ask is ever checked against it before a
  generation runs.
- **A memory failure therefore arrives whole.**
  ``TurboQuantEngine.generate`` wraps a llama failure
  (an OOM, a context overflow) as a raised
  ``RuntimeError``; probed live this tick, a backend
  raising "llama_decode: out of memory" through a
  real core turn becomes TF-031's whole-turn
  degradation ("the model failed during this turn") —
  announced, to TF-031's credit, but total: nothing
  first tries the turn with less context, fewer
  drafts, or no refinement. The graduated middle —
  shed load by rule, keep the turn alive smaller —
  did not exist. D-PRESS-1 in the TF-016 docket.
- **The sheddable pieces all existed, ungoverned.**
  TF-024 windows history under flat, profile-blind
  budgets (20 turns / 1000 word-tokens on every
  profile, pi_zero's 256-token context and server's
  32768 alike); TF-056 caps retrieval per profile
  (bundle / sql / vector) but nothing ever lowers
  those caps under pressure; the profiles carry
  draft_count and max_refinement_passes that nothing
  reduces. Each piece had a cap; no rule ever moved
  one because memory was tight.

THE GOVERNOR (``PRESSURE_RULES`` as data, format
``thoughtforge-memory-pressure/1``):

- **Pressure is a fraction of a named ceiling, in
  bands.** ``band_for_fraction``: below 0.75 normal,
  [0.75, 0.90) elevated, [0.90, 1.00) high, at or
  above 1.00 at_ceiling. The thresholds are
  campaign-set, stated as such; they bracket the
  tree's one precedent (the ETL's 85%) rather than
  copying it, because a turn can shed in stages
  where an import can only commit. Three gauges
  supply the fraction, each honest about what it
  measures: the **context gauge** (a turn's total
  context ask — protected prompt + sheddable
  context — against the profile's
  ``max_context_tokens``; the gauge this governor
  can recompute after shedding, and the only one it
  pronounces refusal on), an **explicit GB reading**
  (``assess_pressure``: a caller-measured used_gb
  against the profile's declared ram+vram ceiling —
  the measurement is the caller's, never invented
  here), and the **model gauge** (TF-069's accounted
  model size against the same ceiling — unprovable
  for q2_k exactly as TF-069 files it, never
  borrowed from a neighbour). A **store gauge**
  (TF-056's per-kind live/hard ratios) is readable
  through ``store_pressure`` as an input signal;
  store eviction itself stays TF-056's lifecycle
  work and is not re-done here.
- **Shedding is ordered, context first, and the
  order is the rule** (``SHED_STEPS``): (1) halve the
  sheddable context, (2) halve retrieval breadth
  (bundle / sql / vector, floor 1 each), (3) drafts
  to one, (4) refinement passes off, (5) halve the
  response-token cap (floor 32), (6) shed the
  remaining sheddable context entirely. Bands take
  cumulative prefixes: elevated takes step 1, high
  takes steps 1–4, at_ceiling takes all six. A step
  that would change nothing on a given load (pi_zero
  already runs one draft and zero refinement) is not
  reported as applied — the report names only steps
  that actually shed something.
- **What is never shed** (``protected_never_shed``):
  the current user prompt (``prompt_tokens`` is
  carried on the load so the gauge counts it, and no
  step touches it), the resident model (its fit is
  admission, TF-069's road, not runtime shedding),
  and protected store records (TF-056's protection
  stands; retrieval breadth is shed, records are
  not evicted by this governor).
- **Never crashes mute.** The governor never raises
  on pressure: its outcomes are data
  (``PressureReport``) — ``no_pressure``, ``shed``
  (with the steps named and the after-gauge
  recomputed), or ``refused_named``: when even the
  fully shed load cannot fit — by construction of
  the context gauge that means the protected prompt
  alone exceeds the profile's context ceiling — the
  report refuses BY NAME, carrying the profile, the
  prompt size, the ceiling, and the steps already
  shed. A caller pronounces that refusal to the
  user; nothing is silently truncated and nothing
  dies with a bare exception. (Unknown profiles and
  malformed loads still raise ValueError, as
  everywhere in this tree — a programming error is
  not pressure.)
- **Composition is a named later road.** Like
  TF-073's chain and TF-074's backend, the governor
  is conformed as an estate here; wiring it into
  ``TurboQuantEngine.generate`` / the core's turn
  path is TF-153's service-shape work, and TF-086's
  truth table must carry the line below.

Proof grade FIXTURE (the governor judged against the
live profile JSONs, TF-056's budgets and TF-069's
accounting, real no-model engine loads, and a fixture
store, VM) — a loaded model's real memory footprint
and its behaviour under genuine host memory pressure
are NOT PROVEN (no GGUF in-tree, llama_cpp absent,
no per-process RSS of a resident model has ever been
measured here), never claimed.
"""

from __future__ import annotations

import inspect
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thoughtforge.inference.profiles import resolve_generation_params
from thoughtforge.utils.config import (
    list_hardware_profiles,
    load_hardware_profile,
)

PRESSURE_FORMAT = "thoughtforge-memory-pressure/1"

# ── Bands (campaign-set thresholds, stated) ─────────────────────────────────
# The tree's sole precedent is the Wikidata ETL's 85%
# commit-early line (etl/wikidata.py). These bands
# bracket it: shedding starts earlier (0.75) because a
# turn can shed in stages, and the last stage must
# already be running before the ceiling itself (1.00).

BANDS: tuple[dict[str, Any], ...] = (
    {"id": "normal", "below": 0.75},
    {"id": "elevated", "below": 0.90},
    {"id": "high", "below": 1.00},
    {"id": "at_ceiling", "below": None},
)

# ── The shed steps, in order (context first) ────────────────────────────────

SHED_STEPS: tuple[dict[str, str], ...] = (
    {"id": "context_halved", "component": "context",
     "action": "the sheddable context is halved (floor 0)"},
    {"id": "retrieval_halved", "component": "retrieval",
     "action": "bundle / sql / vector caps are halved (floor 1 each)"},
    {"id": "drafts_to_one", "component": "drafts",
     "action": "the draft count is cut to one"},
    {"id": "refinement_off", "component": "refinement",
     "action": "refinement passes are cut to zero"},
    {"id": "response_halved", "component": "response",
     "action": "the response-token cap is halved (floor 32)"},
    {"id": "context_shed_entire", "component": "context",
     "action": "the remaining sheddable context is shed entirely"},
)

#: Cumulative step prefixes per band, in SHED_STEPS order.
_BAND_STEP_COUNTS: dict[str, int] = {
    "normal": 0, "elevated": 1, "high": 4, "at_ceiling": 6,
}

RESPONSE_FLOOR_TOKENS = 32

# ── The rules ───────────────────────────────────────────────────────────────

PRESSURE_RULES: tuple[dict, ...] = (
    {"id": "pressure_is_fraction_in_bands",
     "rule": "pressure is a fraction of a named ceiling, judged "
             "in four bands (normal < 0.75, elevated < 0.90, "
             "high < 1.00, at_ceiling >= 1.00); the thresholds "
             "are campaign-set, bracketing the tree's one "
             "precedent (the Wikidata ETL's 85% commit-early)"},
    {"id": "context_shed_first",
     "rule": "shedding is ordered and the order is the rule: "
             "context first (halved, then — at the ceiling — "
             "shed entire), then retrieval breadth, drafts, "
             "refinement, and the response cap; bands take "
             "cumulative prefixes of the one step ladder"},
    {"id": "protected_never_shed",
     "rule": "the current user prompt is never shed (it is "
             "carried on the load so the gauge counts it, and "
             "no step touches it); the resident model is never "
             "shed (its fit is TF-069 admission, not runtime); "
             "protected store records are never shed (TF-056's "
             "protection stands — breadth is shed, records are "
             "not evicted by this governor)"},
    {"id": "only_real_sheds_reported",
     "rule": "a step that would change nothing on a load "
             "(one draft already, refinement already off) is "
             "not reported as applied; the report names only "
             "steps that actually shed something"},
    {"id": "never_crashes_mute",
     "rule": "the governor never raises on pressure: outcomes "
             "are data — no_pressure, shed, or refused_named "
             "carrying the profile, the prompt size, the "
             "ceiling, and the steps already shed; nothing is "
             "silently truncated and nothing dies mute"},
    {"id": "refusal_only_on_the_context_gauge",
     "rule": "refusal is pronounced only on the context gauge, "
             "the gauge the governor can recompute after "
             "shedding; a fully shed load that still exceeds it "
             "means the protected prompt alone exceeds the "
             "profile's context ceiling, and the reason says so"},
    {"id": "external_fractions_are_the_callers_measurement",
     "rule": "an explicit used_gb or a supplied fraction is the "
             "caller's measurement, never invented here; with a "
             "supplied fraction the after-gauge is reported as "
             "None (not recomputed) and the caller re-measures"},
    {"id": "model_gauge_unprovable_stays_unprovable",
     "rule": "the model gauge is TF-069's accounting: a quant "
             "the tree cannot account (q2_k) reads unprovable "
             "with fraction None — never borrowed from a "
             "neighbouring quant, never estimated"},
    {"id": "store_gauge_is_input_not_eviction",
     "rule": "the store gauge (TF-056's live/hard ratios, read "
             "live) is an input signal; eviction stays TF-056's "
             "lifecycle work and is not re-done by this slice"},
    {"id": "live_pressure_not_proven",
     "rule": "a loaded model's real footprint and its behaviour "
             "under genuine host memory pressure are NOT PROVEN "
             "(no GGUF in-tree, llama_cpp absent, no resident "
             "model's RSS ever measured here); TF-082/TF-175 own "
             "the hardware roads, TF-153 owns composing the "
             "governor into the engine/core call paths, and "
             "TF-086's truth table must carry this line"},
)


# ── Bands ───────────────────────────────────────────────────────────────────

def band_for_fraction(fraction: float) -> str:
    """The pressure band for ``fraction`` of a ceiling.
    Pure; a negative fraction raises ValueError."""
    if fraction < 0:
        raise ValueError(f"pressure fraction must be >= 0, got {fraction}")
    for band in BANDS:
        if band["below"] is None or fraction < band["below"]:
            return str(band["id"])
    raise AssertionError("unreachable: the last band is unbounded")


def ceiling_gb(profile_id: str) -> float:
    """One shipped profile's declared memory ceiling:
    ram_gb + vram_gb from its JSON, read live (TF-069's
    ceiling definition). Unknown profiles raise
    ValueError naming the valid ids."""
    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile: {profile_id!r} "
            f"(valid: {', '.join(valid)})"
        )
    hw = load_hardware_profile(profile_id)["hardware"]
    return float(hw["ram_gb"]) + float(hw["vram_gb"])


@dataclass(frozen=True)
class PressureVerdict:
    """One explicit memory reading judged against a
    profile's ceiling, as data."""

    profile_id: str
    used_gb: float | None      # None = unprovable in-tree
    ceiling_gb: float
    status: str                  # measured | unprovable

    @property
    def fraction(self) -> float | None:
        if self.used_gb is None:
            return None
        return self.used_gb / self.ceiling_gb

    @property
    def band(self) -> str | None:
        if self.fraction is None:
            return None
        return band_for_fraction(self.fraction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id, "used_gb": self.used_gb,
            "ceiling_gb": self.ceiling_gb, "fraction": self.fraction,
            "band": self.band, "status": self.status,
        }


def assess_pressure(profile_id: str, used_gb: float) -> PressureVerdict:
    """Judge a caller-measured ``used_gb`` against the
    profile's declared ceiling. The measurement is the
    caller's; this function only names its band."""
    if used_gb < 0:
        raise ValueError(f"used_gb must be >= 0, got {used_gb}")
    return PressureVerdict(
        profile_id=profile_id, used_gb=float(used_gb),
        ceiling_gb=ceiling_gb(profile_id), status="measured")


def model_pressure(
    profile_id: str, params_b: float, quant: str,
) -> PressureVerdict:
    """The model gauge: TF-069's accounted size for
    (params_b, quant) against the profile's ceiling.
    A quant the tree cannot account reads unprovable
    with used None — never estimated."""
    from thoughtforge.inference.profile_runner import accounted_model_gb

    accounted, _basis = accounted_model_gb(params_b, quant)
    return PressureVerdict(
        profile_id=profile_id, used_gb=accounted,
        ceiling_gb=ceiling_gb(profile_id),
        status="measured" if accounted is not None else "unprovable")


def store_pressure(store: Any, profile_id: str) -> dict[str, Any]:
    """The store gauge, as data: TF-056's per-kind
    live/hard ratios read live through its budget
    report, the maximum ratio as the gauge fraction,
    and its band. An input signal only — nothing is
    evicted here."""
    from thoughtforge.knowledge.memory_budgets import budget_report

    report = budget_report(store, profile_id)
    ratios = {
        kind: (row["live"] / row["hard"] if row["hard"] else 0.0)
        for kind, row in report.rows.items()
    }
    fraction = max(ratios.values()) if ratios else 0.0
    return {
        "profile_id": profile_id, "ratios": ratios,
        "fraction": fraction, "band": band_for_fraction(fraction),
    }


# ── The sheddable turn load ─────────────────────────────────────────────────

@dataclass(frozen=True)
class TurnLoad:
    """One turn's sheddable load on one profile.

    ``prompt_tokens`` is the current user prompt —
    protected, counted by the gauge, touched by no
    step. Everything else is sheddable, in SHED_STEPS
    order: the context (history + retrieved context
    the turn carries), retrieval breadth, drafts,
    refinement, and the response cap."""

    profile_id: str
    prompt_tokens: int = 0
    context_tokens: int = 0
    bundle_facts: int = 0
    sql_results: int = 0
    vector_top_k: int = 0
    draft_count: int = 1
    refinement_passes: int = 0
    max_response_tokens: int = 0

    @property
    def total_context(self) -> int:
        """The turn's whole context ask: protected prompt
        + sheddable context."""
        return self.prompt_tokens + self.context_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "prompt_tokens": self.prompt_tokens,
            "context_tokens": self.context_tokens,
            "bundle_facts": self.bundle_facts,
            "sql_results": self.sql_results,
            "vector_top_k": self.vector_top_k,
            "draft_count": self.draft_count,
            "refinement_passes": self.refinement_passes,
            "max_response_tokens": self.max_response_tokens,
            "total_context": self.total_context,
        }


def profile_turn_load(profile_id: str) -> TurnLoad:
    """A turn asking for everything its profile allows:
    the context allocation at the profile's
    ``max_context_tokens`` (the engine allocates exactly
    that as ``n_ctx`` at load), retrieval breadth at
    TF-056's budget caps, and drafts / refinement /
    response at the resolved generation params. The
    ceiling reference point — its context gauge is
    exactly 1.0 on every profile by construction."""
    from thoughtforge.knowledge.memory_budgets import budget_for_profile

    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile: {profile_id!r} "
            f"(valid: {', '.join(valid)})"
        )
    profile = load_hardware_profile(profile_id)
    params = resolve_generation_params(profile)
    budget = budget_for_profile(profile_id)
    return TurnLoad(
        profile_id=profile_id,
        prompt_tokens=0,
        context_tokens=params.max_context_tokens,
        bundle_facts=budget.max_memory_bundle_facts,
        sql_results=budget.sql_max_results,
        vector_top_k=budget.vector_top_k,
        draft_count=params.draft_count,
        refinement_passes=int(profile["generation"]["max_refinement_passes"]),
        max_response_tokens=params.max_response_tokens,
    )


def max_context_tokens(profile_id: str) -> int:
    """The profile's context ceiling, read live from
    its shipped JSON."""
    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile: {profile_id!r} "
            f"(valid: {', '.join(valid)})"
        )
    return int(load_hardware_profile(profile_id)["model"]["max_context_tokens"])


def context_gauge(profile_id: str, load: TurnLoad) -> float:
    """The context gauge: the turn's total context ask
    against the profile's context ceiling. The gauge
    this governor recomputes after shedding and the
    only one it pronounces refusal on."""
    if load.profile_id != profile_id:
        raise ValueError(
            f"load profile {load.profile_id!r} != {profile_id!r}")
    return load.total_context / max_context_tokens(profile_id)


# ── Shedding ────────────────────────────────────────────────────────────────

def _replace(load: TurnLoad, **changes: Any) -> TurnLoad:
    from dataclasses import replace

    return replace(load, **changes)


def _shed_context_halved(load: TurnLoad) -> TurnLoad:
    return _replace(load, context_tokens=load.context_tokens // 2)


def _shed_retrieval_halved(load: TurnLoad) -> TurnLoad:
    return _replace(
        load,
        bundle_facts=max(1, load.bundle_facts // 2) if load.bundle_facts else 0,
        sql_results=max(1, load.sql_results // 2) if load.sql_results else 0,
        vector_top_k=max(1, load.vector_top_k // 2) if load.vector_top_k else 0,
    )


def _shed_drafts_to_one(load: TurnLoad) -> TurnLoad:
    return _replace(load, draft_count=1)


def _shed_refinement_off(load: TurnLoad) -> TurnLoad:
    return _replace(load, refinement_passes=0)


def _shed_response_halved(load: TurnLoad) -> TurnLoad:
    if load.max_response_tokens <= RESPONSE_FLOOR_TOKENS:
        return load
    return _replace(
        load, max_response_tokens=max(
            RESPONSE_FLOOR_TOKENS, load.max_response_tokens // 2))


def _shed_context_entire(load: TurnLoad) -> TurnLoad:
    return _replace(load, context_tokens=0)


_STEP_FUNCS = {
    "context_halved": _shed_context_halved,
    "retrieval_halved": _shed_retrieval_halved,
    "drafts_to_one": _shed_drafts_to_one,
    "refinement_off": _shed_refinement_off,
    "response_halved": _shed_response_halved,
    "context_shed_entire": _shed_context_entire,
}


@dataclass(frozen=True)
class PressureReport:
    """The governor's outcome, as data — never an
    exception, never silence."""

    profile_id: str
    band_before: str
    fraction_before: float
    gauge_derived: bool
    steps_applied: tuple[str, ...]
    load_before: TurnLoad
    load_after: TurnLoad
    fraction_after: float | None   # None = not recomputed (supplied fraction)
    band_after: str | None
    status: str                    # no_pressure | shed | refused_named
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "band_before": self.band_before,
            "fraction_before": self.fraction_before,
            "gauge_derived": self.gauge_derived,
            "steps_applied": list(self.steps_applied),
            "load_before": self.load_before.to_dict(),
            "load_after": self.load_after.to_dict(),
            "fraction_after": self.fraction_after,
            "band_after": self.band_after,
            "status": self.status, "reason": self.reason,
        }


def apply_pressure(
    profile_id: str, load: TurnLoad, *, fraction: float | None = None,
) -> PressureReport:
    """Govern one turn's load under pressure.

    ``fraction`` defaults to the load's context gauge;
    a caller may supply an external fraction (e.g. the
    model gauge's or a measured GB reading's) — the
    band then sheds by that fraction, the after-gauge
    is reported as None, and refusal is not pronounced
    (the caller re-measures; refusal belongs to the
    context gauge alone). Pure and deterministic: the
    same profile, load and fraction always give the
    same report."""
    if load.profile_id != profile_id:
        raise ValueError(
            f"load profile {load.profile_id!r} != {profile_id!r}")
    gauge_derived = fraction is None
    fraction_before = context_gauge(profile_id, load) if gauge_derived \
        else float(fraction)
    band_before = band_for_fraction(fraction_before)

    step_ids = [s["id"] for s in SHED_STEPS][:_BAND_STEP_COUNTS[band_before]]
    after = load
    applied: list[str] = []
    for step_id in step_ids:
        nxt = _STEP_FUNCS[step_id](after)
        if nxt != after:
            applied.append(step_id)
            after = nxt

    if band_before == "normal":
        status, reason = "no_pressure", ""
        fraction_after = fraction_before if gauge_derived else None
    elif gauge_derived:
        fraction_after = context_gauge(profile_id, after)
        if fraction_after > 1.0:
            status = "refused_named"
            reason = (
                f"prompt alone ({load.prompt_tokens} tokens) exceeds "
                f"{profile_id}'s context ceiling "
                f"({max_context_tokens(profile_id)} tokens) after all "
                f"shedding (steps: {', '.join(applied) or 'none'}); the "
                f"turn is refused by name, not crashed or silently "
                f"truncated"
            )
        else:
            status, reason = "shed", ""
    else:
        fraction_after = None
        status, reason = "shed", ""

    return PressureReport(
        profile_id=profile_id, band_before=band_before,
        fraction_before=fraction_before, gauge_derived=gauge_derived,
        steps_applied=tuple(applied), load_before=load, load_after=after,
        fraction_after=fraction_after,
        band_after=(band_for_fraction(fraction_after)
                    if fraction_after is not None else None),
        status=status, reason=reason)


# ── Export ──────────────────────────────────────────────────────────────────

def pressure_as_data() -> dict[str, Any]:
    return {
        "format": PRESSURE_FORMAT,
        "bands": [dict(b) for b in BANDS],
        "steps": [dict(s) for s in SHED_STEPS],
        "band_step_counts": dict(_BAND_STEP_COUNTS),
        "rules": [dict(r) for r in PRESSURE_RULES],
        "ceilings_gb": {pid: ceiling_gb(pid)
                        for pid in list_hardware_profiles()},
        "loads": {pid: profile_turn_load(pid).to_dict()
                  for pid in list_hardware_profiles()},
        "live_proof": "not_proven",
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def pressure_problems() -> list[str]:
    """The filed estate judged against the live tree."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in PRESSURE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("PRESSURE_RULES ids are not unique")
    step_ids = [s["id"] for s in SHED_STEPS]
    if step_ids != ["context_halved", "retrieval_halved", "drafts_to_one",
                    "refinement_off", "response_halved",
                    "context_shed_entire"]:
        problems.append(f"SHED_STEPS order drifted: {step_ids}")
    if step_ids[0] != "context_halved" or SHED_STEPS[0]["component"] != "context":
        problems.append("context is not shed first")
    if [b["below"] for b in BANDS] != [0.75, 0.90, 1.00, None]:
        problems.append("BANDS thresholds drifted")
    if _BAND_STEP_COUNTS != {"normal": 0, "elevated": 1,
                             "high": 4, "at_ceiling": 6}:
        problems.append("band step counts drifted")

    # The maximum turn loads recompute live from the
    # profile JSONs, TF-056's budgets and the resolved
    # generation params; each one's context gauge is
    # exactly 1.0 (the ceiling reference point).
    from thoughtforge.knowledge.memory_budgets import budget_for_profile

    for pid in list_hardware_profiles():
        profile = load_hardware_profile(pid)
        params = resolve_generation_params(profile)
        budget = budget_for_profile(pid)
        load = profile_turn_load(pid)
        want = {
            "context_tokens": params.max_context_tokens,
            "bundle_facts": budget.max_memory_bundle_facts,
            "sql_results": budget.sql_max_results,
            "vector_top_k": budget.vector_top_k,
            "draft_count": params.draft_count,
            "refinement_passes":
                int(profile["generation"]["max_refinement_passes"]),
            "max_response_tokens": params.max_response_tokens,
        }
        for field_name, value in want.items():
            if getattr(load, field_name) != value:
                problems.append(f"{pid}: load {field_name} "
                                f"{getattr(load, field_name)} != live {value}")
        if context_gauge(pid, load) != 1.0:
            problems.append(f"{pid}: maximum load's gauge is not 1.0")
        if ceiling_gb(pid) != float(profile["hardware"]["ram_gb"]
                                     + profile["hardware"]["vram_gb"]):
            problems.append(f"{pid}: ceiling not read live")

    # Measured-start pins: the engine's cap is the
    # static min-clip this slice's docstring names, and
    # the tree's one precedent threshold stands.
    from thoughtforge.inference import turboquant as turboquant_module

    engine_src = inspect.getsource(turboquant_module.TurboQuantEngine)
    if "min(requested_tokens, budget, max_resp)" not in engine_src:
        problems.append("the engine's static token clip drifted — the "
                        "measured start must be re-read")
    etl_src = Path(turboquant_module.__file__).resolve().parents[1] \
        .joinpath("etl", "wikidata.py").read_text(encoding="utf-8")
    if "virtual_memory().percent > 85" not in etl_src:
        problems.append("the ETL's 85% precedent drifted — the bands' "
                        "stated derivation must be re-read")

    # The model gauge keeps TF-069's unprovable rows
    # unprovable (q2_k), and proves the calibrated ones.
    if model_pressure("pi_zero", 0.5, "q2_k").status != "unprovable":
        problems.append("q2_k model pressure must stay unprovable")
    if model_pressure("desktop_cpu", 7.0, "q4_k_m").fraction is None:
        problems.append("q4_k_m model pressure must be provable")
    return problems


def pressure_live_problems() -> list[str]:
    """The governor re-derived behaviourally on named cases."""
    problems: list[str] = []

    # Elevated: only the context halves.
    load = TurnLoad(profile_id="desktop_cpu", prompt_tokens=100,
                    context_tokens=3172, bundle_facts=15, sql_results=15,
                    vector_top_k=8, draft_count=3, refinement_passes=2,
                    max_response_tokens=400)
    report = apply_pressure("desktop_cpu", load)
    if report.band_before != "elevated" \
            or report.steps_applied != ("context_halved",) \
            or report.load_after.context_tokens != 1586 \
            or report.status != "shed":
        problems.append(f"elevated case wrong: {report.to_dict()}")

    # At the ceiling on pi_zero: context shed in two
    # stages to zero, the no-op steps (drafts already
    # one, refinement already off) not reported, the
    # protected prompt untouched, the turn survives.
    load = TurnLoad(profile_id="pi_zero", prompt_tokens=20,
                    context_tokens=492, bundle_facts=3, sql_results=3,
                    vector_top_k=2, draft_count=1, refinement_passes=0,
                    max_response_tokens=120)
    report = apply_pressure("pi_zero", load)
    if report.band_before != "at_ceiling" \
            or report.steps_applied != (
                "context_halved", "retrieval_halved",
                "response_halved", "context_shed_entire") \
            or report.load_after.context_tokens != 0 \
            or report.load_after.prompt_tokens != 20 \
            or report.fraction_after is None \
            or report.fraction_after > 1.0 \
            or report.status != "shed":
        problems.append(f"pi_zero ceiling case wrong: {report.to_dict()}")

    # Refused by name: a prompt alone over the ceiling
    # cannot be saved by shedding, and the refusal
    # names the profile, the prompt, and the ceiling.
    load = TurnLoad(profile_id="desktop_cpu", prompt_tokens=20_000,
                    context_tokens=100, bundle_facts=15, sql_results=15,
                    vector_top_k=8, draft_count=3, refinement_passes=2,
                    max_response_tokens=400)
    report = apply_pressure("desktop_cpu", load)
    if report.status != "refused_named" \
            or "desktop_cpu" not in report.reason \
            or "20000" not in report.reason \
            or "4096" not in report.reason:
        problems.append(f"refusal case wrong: {report.to_dict()}")

    # The real engine leg: a no-model TurboQuantEngine's
    # resolved params are the maximum load's fields, on
    # a constrained profile and the VM profile.
    from thoughtforge.inference.turboquant import TurboQuantEngine

    for pid in ("pi_zero", "desktop_cpu"):
        engine = TurboQuantEngine(profile_id=pid)
        try:
            engine.load()
            want = profile_turn_load(pid)
            params = engine._params
            if params is None \
                    or params.max_context_tokens != want.context_tokens \
                    or params.draft_count != want.draft_count \
                    or params.max_response_tokens != want.max_response_tokens:
                problems.append(f"{pid}: engine params != maximum load")
        finally:
            engine.unload()

    # The store gauge, live over a fixture store: 70
    # facts against phone_low's hard cap of 64 reads
    # at_ceiling; a fresh store reads 0 / normal.
    from thoughtforge.knowledge.models import UserFactRecord
    from thoughtforge.knowledge.store import MemoryStore

    with tempfile.TemporaryDirectory(prefix="tf080-") as tmp:
        store = MemoryStore(memory_dir=Path(tmp))
        fresh = store_pressure(store, "phone_low")
        if fresh["fraction"] != 0.0 or fresh["band"] != "normal":
            problems.append(f"fresh store gauge wrong: {fresh}")
        for i in range(70):
            store.append_profile_record(UserFactRecord(
                record_id=f"fct_tf080_{i:03d}", summary=f"pressure fact {i}",
                tags=[f"t{i}"], confidence=0.5))
        over = store_pressure(store, "phone_low")
        if abs(over["fraction"] - 70 / 64) > 1e-9 \
                or over["band"] != "at_ceiling":
            problems.append(f"over-cap store gauge wrong: {over}")
    return problems


__all__ = [
    "BANDS",
    "PRESSURE_FORMAT",
    "PRESSURE_RULES",
    "RESPONSE_FLOOR_TOKENS",
    "SHED_STEPS",
    "PressureReport",
    "PressureVerdict",
    "TurnLoad",
    "apply_pressure",
    "assess_pressure",
    "band_for_fraction",
    "ceiling_gb",
    "context_gauge",
    "max_context_tokens",
    "model_pressure",
    "pressure_as_data",
    "pressure_live_problems",
    "pressure_problems",
    "profile_turn_load",
    "store_pressure",
]
