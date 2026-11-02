"""
Profile panels, as data (TF-128).

The roadmap's rule: *profile panels: suites per
hardware profile with the profile's budgets
enforced during the run.* TF-121's harness
narrowed **D-BENCH-1** to exactly this share: a
bogus profile is refused by name (fixed there),
but the profile was never *applied* —
``ProfileBenchmark`` measured the same ephemeral
empty-store core (built with no profile at all)
for every profile id it was handed. TF-032 /
TF-033 / TF-037 / TF-056 / TF-098 have since
narrowed the parent gap D-PROF-3 in the core
itself: a conversation can select a profile, and
the selected profile's ``max_response_tokens``,
latency budgets, refinement-pass cap and
retrieval / store caps are live in
``ThoughtForgeCore`` when it does. TF-069's
profile runner boots every shipped profile's
stack in simulation with its RAM ceiling
enforced as a hard limit. Nothing had ever run
the Part VII suites *per profile*, with those
budgets enforced during the run, or published
the result as a panel. That is this slice.

The measured starting state, probed live this
tick on this tree before anything was built:

  * **The core applies a selected profile,
    exactly as filed.** For every shipped
    profile, a core built with
    ``config={"conversation": {"profile_id": …}}``
    resolved ``max_response_tokens`` and
    refinement passes to the profile JSON's own
    generation values (pi_zero 120 / 0 through
    server_gpu 2000 / 3) and its lifecycle caps
    equalled TF-056's ``budget_for_profile`` caps
    for that profile — measured live, all six.
  * **The benchmark still did not.** Probing
    ``ProfileBenchmark`` per profile measured
    the identical empty-store shape everywhere
    because its ephemeral core is built with no
    profile config (``benchmark_profiles``' own
    filing, TF-121). Run *under an applied core*
    (the runner's existing ``core=`` parameter),
    the same benchmark files the identical
    empty-store metrics on every profile —
    10/10 successful, citation 0.0, enforcement
    0.0, 9.0 avg words, confidence 0.408,
    ``{"weak": 10}`` — because an empty store has
    nothing for the budgets to bite on. That
    identity is measured and filed below, not
    dressed as per-profile quality.
  * **TF-069's ceiling rows stand.** Re-measured
    live through ``check_model_fit``: desktop_cpu
    / desktop_gpu / pi_5 caps fit; server_gpu's
    cap exceeds its own ceiling (70B at fp16
    accounts 140.0 GB against 88 GB declared);
    phone_low / pi_zero caps are unprovable
    in-tree (q2_k uncalibrated, TF-069's filing).

What this module files, as verifiable data:

  * ``PANEL_ROWS`` — one literal row per shipped
    profile (the harness's ``KNOWN_PROFILES``
    six), recomputed live by the checker: the
    profile's filed budgets (read live from its
    JSON — RAM / VRAM / ceiling, generation
    tokens + refinement passes + context,
    retrieval caps, bundle cap), its TF-069 cap
    verdict + boot verdict, the applied core's
    measured tokens / passes, the profile
    benchmark's metrics under that applied core,
    and the four Part VII suites' scores run per
    profile through their real runners.
  * **Budgets enforced during the run**, by real
    machinery only: the benchmark cell runs on a
    core built with the profile selected, and
    the cell counts only when that core's
    resolved tokens / passes and lifecycle caps
    equal the profile's filed budgets (asserted
    live per profile in the checker, beside
    TF-069's own boot of the same profile); the
    RAM ceiling is TF-069's ``check_model_fit``
    cap row, filed per profile — a cap that
    exceeds its ceiling or cannot be accounted
    is named in the row, never passed silently;
    and the depth suite's scaffold (ON) road is
    gated by the profile's own
    ``max_refinement_passes``: pi_zero files 0,
    so its ON road is **refused-by-budget**
    (the budget named in the row) and the
    profile's depth score is its single-pass
    OFF score, filed as exactly that.
  * ``FINDING`` — computed from the rows, filed
    whatever the numbers are: the budgets apply
    exactly on all six profiles; the empty-store
    benchmark metrics do not vary by profile and
    the scripted suite scores vary only through
    the one refused cell (the measured identity,
    §limits); three profiles carry a ceiling gap
    (one exceeds, two unprovable).

The panel verdict is filed plainly: PASS only
when all six profiles are measured, every
applied budget matches its profile's filing
exactly, every boot verdict reproduces TF-069's
filed ``BOOT_VERDICTS`` and every cap row
reproduces TF-069's ``RAM_ROWS``, and the rows
reproduce ``PANEL_ROWS``. An empty panel or
selection scores 0.0 / REVIEW — no vacuous
pass; a partial selection files REVIEW however
its measured rows score.

D-BENCH-1 is FIXED by this slice in the
measuring sense: the Part VII suites and the
profile benchmark now run per shipped profile
with the profile applied and its budgets
enforced during the run, published as literal
rows recomputed live with a pinned harness
baseline. What is NOT fixed, and is stated in
the filed limits: ``benchmark_profiles``' own
default ephemeral core still applies no profile
(unchanged — this panel supplies the applied
core through its existing ``core=`` parameter);
per-profile behaviour on real hardware is NOT
PROVEN — every run here is on this VM, the
pi / phone profiles in TF-069's simulation
sense; and a seeded store under the budgets is
NOT PROVEN (TF-134's baseline book / TF-173's
real-hardware road).

Filed limits, stated plainly:

  * The benchmark cell measures an ephemeral
    core over an EMPTY store (``model_path=None``)
    — the empty-store refusal on every profile,
    never inference, never a loaded model. The
    budgets are proven live in the core; their
    effect on seeded-store quality is not
    measured here.
  * The four suites score scripted synthetic
    corpora (TF-017; ``data/`` never opened)
    through their real runners. Their scorers
    take no profile input, so their per-profile
    scores are identical by construction of the
    estate except where a budget refuses a road
    — the panel files that shape as the finding,
    and the only suite cell a budget changes is
    the one it refuses (pi_zero's depth ON road).
  * The RAM ceiling accounting is TF-069's:
    catalogue-calibrated GB-per-billion (worst
    in-tree ratio) or definitional for fp16;
    q2_k is uncalibrated in-tree, so phone_low /
    pi_zero cap rows are unprovable, never
    estimated from a neighbour.
  * Live per-profile hardware behaviour is NOT
    PROVEN: no phone, Pi or server GPU ran —
    TF-082 / TF-173 own the real-hardware road.

The human-readable write-up is
``docs/architecture/PROFILE_PANEL.md``.
Proof grade: FIXTURE/SIMULATION — synthetic
scripted corpora and an empty-store core through
the real machinery on this VM, profiles applied
and budgets enforced as filed; no live model,
never claimed.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILE_PANEL_FORMAT = "thoughtforge-profile-panel/1"
PROFILE_PANEL_DOC = "docs/architecture/PROFILE_PANEL.md"

# The real machinery, reused — never copied: the
# shipped profiles + their loader, the resolved
# generation params, TF-056's memory budgets,
# TF-069's runner (boot + RAM ceiling), the
# harness's profile filing, the profile benchmark
# and the four suites' real runners.
from benchmarks.benchmark_profiles import ProfileBenchmark  # noqa: E402
from benchmarks.harness import KNOWN_PROFILES  # noqa: E402
from thoughtforge.inference.profile_runner import (  # noqa: E402
    BOOT_VERDICTS,
    RAM_ROWS,
    check_model_fit,
    profile_runner_problems,
    run_profile_boot,
)
from thoughtforge.inference.profiles import (  # noqa: E402
    resolve_generation_params,
)
from thoughtforge.knowledge.memory_budgets import (  # noqa: E402
    budget_for_profile,
)
from thoughtforge.utils.config import load_hardware_profile  # noqa: E402

#: The panel's profiles: the shipped six, in the
#: harness's filed order.
PANEL_PROFILES: tuple[str, ...] = KNOWN_PROFILES

#: The panel's suites, in panel order (the four
#: Part VII suites whose runners accept corpora).
PANEL_SUITES: tuple[str, ...] = (
    "persona", "memory", "depth", "presence")

#: Filed cells per profile: the applied-core
#: benchmark plus the four suites.
CELLS_PER_PROFILE = 1 + len(PANEL_SUITES)
TOTAL_CELLS = len(PANEL_PROFILES) * CELLS_PER_PROFILE


# ── The panel measurement ───────────────────────────────────────────────────

def filed_budgets(profile_id: str) -> dict[str, Any]:
    """One profile's budgets, read live from its
    shipped JSON + TF-056's budget resolution —
    never invented, never copied from a filing."""
    profile = load_hardware_profile(profile_id)
    params = resolve_generation_params(profile)
    budget = budget_for_profile(profile_id)
    cap = check_model_fit(
        profile_id, float(profile["model"]["max_params_b"]),
        profile["model"]["quantization"])
    return {
        "ram_gb": float(profile["hardware"]["ram_gb"]),
        "vram_gb": float(profile["hardware"]["vram_gb"]),
        "ceiling_gb": cap.ceiling_gb,
        "max_params_b": float(profile["model"]["max_params_b"]),
        "quantization": profile["model"]["quantization"],
        "cap_status": cap.status,
        "cap_accounted_gb": cap.accounted_gb,
        "cap_basis": cap.basis,
        "max_response_tokens": params.max_response_tokens,
        "max_refinement_passes": params.max_refinement_passes,
        "max_context_tokens": params.max_context_tokens,
        "bundle_facts": budget.max_memory_bundle_facts,
        "sql_max_results": int(profile["retrieval"]["sql_max_results"]),
        "vector_top_k": int(profile["retrieval"]["vector_top_k"]),
    }


def _applied_core_benchmark(profile_id: str) -> dict[str, Any]:
    """The profile benchmark run DURING which the
    profile's budgets are enforced: the core is
    built with the profile selected, its resolved
    tokens / passes + lifecycle caps are read
    back, and the benchmark runs on that core
    through ProfileBenchmark's own ``core=``
    parameter over its filed default queries."""
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.knowledge.memory_budgets import budget_for_profile

    budgets = filed_budgets(profile_id)
    with tempfile.TemporaryDirectory(prefix="tf128-") as tmp:
        root = Path(tmp)
        core = ThoughtForgeCore(
            config={"conversation": {"profile_id": profile_id}},
            memory_dir=root / "memory", db_path=root / "knowledge.db")
        config = core._conversation_config
        budget = budget_for_profile(profile_id)
        caps_match = all(
            core._lifecycle.caps[kind] == budget.caps[kind]
            for kind in budget.caps)
        result = ProfileBenchmark().run(profile_id, core=core)
    return {
        "core_max_response_tokens": config.max_response_tokens,
        "core_max_refinement_passes": config.refinement_max_passes(),
        "core_caps_match_budget": caps_match,
        "budgets_applied": bool(
            config.max_response_tokens == budgets["max_response_tokens"]
            and config.refinement_max_passes()
            == budgets["max_refinement_passes"] and caps_match),
        "bench_successful_turns": result.successful_turns,
        "bench_total_turns": result.total_turns,
        "bench_citation_accuracy": result.citation_accuracy,
        "bench_enforcement_pass_rate": result.enforcement_pass_rate,
        "bench_avg_response_words": result.avg_response_words,
        "bench_avg_confidence": result.avg_confidence,
    }


def _suite_scores(profile_id: str,
                  budgets: dict[str, Any]) -> dict[str, Any]:
    """The four suites run per profile through
    their real runners. The depth ON road is
    gated by the profile's filed refinement
    budget: a profile allowing no refinement
    pass (pi_zero, 0) refuses that road by
    budget — its depth score is the OFF score,
    filed as exactly that."""
    from benchmarks.depth_suite import run_depth_suite
    from benchmarks.memory_suite import run_memory_suite
    from benchmarks.persona_suite import run_persona_suite
    from benchmarks.presence_suite import run_presence_suite

    persona = run_persona_suite().overall_score
    memory = run_memory_suite().overall_score
    depth = run_depth_suite()
    presence = run_presence_suite().overall_score
    if budgets["max_refinement_passes"] >= 1:
        depth_on: float | None = depth.overall_on
        depth_status = "measured"
        depth_headline = depth.overall_on
    else:
        depth_on = None
        depth_status = "refused_by_budget"
        depth_headline = depth.overall_off
    return {
        "persona_score": persona,
        "memory_score": memory,
        "depth_off_score": depth.overall_off,
        "depth_on_score": depth_on,
        "depth_on_status": depth_status,
        "presence_score": presence,
        "suite_score": (persona + memory + depth_headline + presence) / 4,
    }


def measure_profile_row(profile_id: str) -> dict[str, Any]:
    """One profile's full panel row, measured live:
    filed budgets, TF-069's boot + cap verdicts,
    the applied-core benchmark, the suites."""
    budgets = filed_budgets(profile_id)
    boot = run_profile_boot(profile_id)
    bench = _applied_core_benchmark(profile_id)
    suites = _suite_scores(profile_id, budgets)
    return {
        "profile": profile_id,
        **budgets,
        "boot_verdict": boot.verdict,
        **bench,
        **suites,
    }


def compute_panel_rows(
    profiles: Any = None,
) -> tuple[dict[str, Any], ...]:
    """The whole panel, measured live (profile order filed)."""
    profiles = PANEL_PROFILES if profiles is None else profiles
    return tuple(measure_profile_row(p) for p in profiles)


def finding_for(rows: Any) -> dict[str, Any]:
    """The panel's finding, computed from the rows:
    whether the budgets applied exactly, whether
    any score varies by profile, the refused
    cells, and the ceiling gaps."""
    def _varies(key: str) -> bool:
        return len({r[key] for r in rows}) > 1

    refused = [
        [r["profile"], "depth_on",
         f"max_refinement_passes={r['max_refinement_passes']}"]
        for r in rows if r["depth_on_status"] == "refused_by_budget"]
    return {
        "budgets_applied_on_all_profiles": bool(
            rows and all(r["budgets_applied"] for r in rows)),
        "bench_scores_vary_by_profile": bool(rows and (
            _varies("bench_citation_accuracy")
            or _varies("bench_avg_response_words")
            or _varies("bench_avg_confidence"))),
        "suite_scores_vary_by_profile": bool(
            rows and _varies("suite_score")),
        "suite_scores_vary_without_the_refused_cell": bool(
            rows and len({
                r["suite_score"] for r in rows
                if r["depth_on_status"] == "measured"}) > 1),
        "refused_cells": refused,
        "ceiling_gap_profiles": [
            r["profile"] for r in rows if r["cap_status"] != "fit"],
        "widest_profile": (
            max(rows, key=lambda r: r["max_response_tokens"])["profile"]
            if rows else None),
        "narrowest_profile": (
            min(rows, key=lambda r: r["max_response_tokens"])["profile"]
            if rows else None),
    }


# ── The published panel (literal data, TF-067's CURVE precedent) ────────────
#
# Measured live on this VM at banking (2026-11-01):
# every profile's budgets read from its JSON, its
# stack booted by TF-069, the benchmark run on a
# profile-applied core, the four suites run per
# profile — twice, identical. The checker
# recomputes every row live and names any drift.
# Filed whatever the numbers are: see the doc for
# the honest shapes (identical empty-store +
# scripted scores across profiles; pi_zero's
# refused depth ON road; server_gpu's cap over
# its own ceiling).

PANEL_ROWS: tuple[dict[str, Any], ...] = (
    {"profile": "desktop_cpu", "ram_gb": 8.0, "vram_gb": 0.0,
     "ceiling_gb": 8.0, "max_params_b": 7.0,
     "quantization": "q4_k_m", "cap_status": "fit",
     "cap_accounted_gb": 4.6667, "cap_basis": "catalogue_max",
     "max_response_tokens": 400, "max_refinement_passes": 2,
     "max_context_tokens": 4096, "bundle_facts": 15,
     "sql_max_results": 15, "vector_top_k": 8,
     "boot_verdict": "boots_with_ceiling_gaps",
     "core_max_response_tokens": 400,
     "core_max_refinement_passes": 2,
     "core_caps_match_budget": True, "budgets_applied": True,
     "bench_successful_turns": 10, "bench_total_turns": 10,
     "bench_citation_accuracy": 0.0,
     "bench_enforcement_pass_rate": 0.0,
     "bench_avg_response_words": 9, "bench_avg_confidence": 0.408,
     "persona_score": 0.8453703703703704,
     "memory_score": 0.8055555555555557,
     "depth_off_score": 0.47222222222222227,
     "depth_on_score": 0.8055555555555557,
     "depth_on_status": "measured",
     "presence_score": 0.7777777777777778,
     "suite_score": 0.8085648148148149},
    {"profile": "desktop_gpu", "ram_gb": 16.0, "vram_gb": 12.0,
     "ceiling_gb": 28.0, "max_params_b": 13.0,
     "quantization": "q8_0", "cap_status": "fit",
     "cap_accounted_gb": 14.3, "cap_basis": "catalogue_max",
     "max_response_tokens": 600, "max_refinement_passes": 3,
     "max_context_tokens": 8192, "bundle_facts": 15,
     "sql_max_results": 15, "vector_top_k": 10,
     "boot_verdict": "boots",
     "core_max_response_tokens": 600,
     "core_max_refinement_passes": 3,
     "core_caps_match_budget": True, "budgets_applied": True,
     "bench_successful_turns": 10, "bench_total_turns": 10,
     "bench_citation_accuracy": 0.0,
     "bench_enforcement_pass_rate": 0.0,
     "bench_avg_response_words": 9, "bench_avg_confidence": 0.408,
     "persona_score": 0.8453703703703704,
     "memory_score": 0.8055555555555557,
     "depth_off_score": 0.47222222222222227,
     "depth_on_score": 0.8055555555555557,
     "depth_on_status": "measured",
     "presence_score": 0.7777777777777778,
     "suite_score": 0.8085648148148149},
    {"profile": "phone_low", "ram_gb": 2.0, "vram_gb": 0.0,
     "ceiling_gb": 2.0, "max_params_b": 1.1,
     "quantization": "q2_k", "cap_status": "unprovable",
     "cap_accounted_gb": None, "cap_basis": "uncalibrated",
     "max_response_tokens": 180, "max_refinement_passes": 1,
     "max_context_tokens": 512, "bundle_facts": 5,
     "sql_max_results": 5, "vector_top_k": 3,
     "boot_verdict": "boots_with_ceiling_gaps",
     "core_max_response_tokens": 180,
     "core_max_refinement_passes": 1,
     "core_caps_match_budget": True, "budgets_applied": True,
     "bench_successful_turns": 10, "bench_total_turns": 10,
     "bench_citation_accuracy": 0.0,
     "bench_enforcement_pass_rate": 0.0,
     "bench_avg_response_words": 9, "bench_avg_confidence": 0.408,
     "persona_score": 0.8453703703703704,
     "memory_score": 0.8055555555555557,
     "depth_off_score": 0.47222222222222227,
     "depth_on_score": 0.8055555555555557,
     "depth_on_status": "measured",
     "presence_score": 0.7777777777777778,
     "suite_score": 0.8085648148148149},
    {"profile": "pi_5", "ram_gb": 4.0, "vram_gb": 0.0,
     "ceiling_gb": 4.0, "max_params_b": 3.0,
     "quantization": "q4_k_m", "cap_status": "fit",
     "cap_accounted_gb": 2.0, "cap_basis": "catalogue_max",
     "max_response_tokens": 250, "max_refinement_passes": 2,
     "max_context_tokens": 2048, "bundle_facts": 10,
     "sql_max_results": 10, "vector_top_k": 5,
     "boot_verdict": "boots_with_ceiling_gaps",
     "core_max_response_tokens": 250,
     "core_max_refinement_passes": 2,
     "core_caps_match_budget": True, "budgets_applied": True,
     "bench_successful_turns": 10, "bench_total_turns": 10,
     "bench_citation_accuracy": 0.0,
     "bench_enforcement_pass_rate": 0.0,
     "bench_avg_response_words": 9, "bench_avg_confidence": 0.408,
     "persona_score": 0.8453703703703704,
     "memory_score": 0.8055555555555557,
     "depth_off_score": 0.47222222222222227,
     "depth_on_score": 0.8055555555555557,
     "depth_on_status": "measured",
     "presence_score": 0.7777777777777778,
     "suite_score": 0.8085648148148149},
    {"profile": "pi_zero", "ram_gb": 0.5, "vram_gb": 0.0,
     "ceiling_gb": 0.5, "max_params_b": 0.5,
     "quantization": "q2_k", "cap_status": "unprovable",
     "cap_accounted_gb": None, "cap_basis": "uncalibrated",
     "max_response_tokens": 120, "max_refinement_passes": 0,
     "max_context_tokens": 256, "bundle_facts": 3,
     "sql_max_results": 3, "vector_top_k": 2,
     "boot_verdict": "boots_with_ceiling_gaps",
     "core_max_response_tokens": 120,
     "core_max_refinement_passes": 0,
     "core_caps_match_budget": True, "budgets_applied": True,
     "bench_successful_turns": 10, "bench_total_turns": 10,
     "bench_citation_accuracy": 0.0,
     "bench_enforcement_pass_rate": 0.0,
     "bench_avg_response_words": 9, "bench_avg_confidence": 0.408,
     "persona_score": 0.8453703703703704,
     "memory_score": 0.8055555555555557,
     "depth_off_score": 0.47222222222222227,
     "depth_on_score": None,
     "depth_on_status": "refused_by_budget",
     "presence_score": 0.7777777777777778,
     "suite_score": 0.7252314814814815},
    {"profile": "server_gpu", "ram_gb": 64.0, "vram_gb": 24.0,
     "ceiling_gb": 88.0, "max_params_b": 70.0,
     "quantization": "fp16", "cap_status": "exceeds_ceiling",
     "cap_accounted_gb": 140.0, "cap_basis": "definition",
     "max_response_tokens": 2000, "max_refinement_passes": 3,
     "max_context_tokens": 32768, "bundle_facts": 30,
     "sql_max_results": 30, "vector_top_k": 20,
     "boot_verdict": "boots_with_ceiling_gaps",
     "core_max_response_tokens": 2000,
     "core_max_refinement_passes": 3,
     "core_caps_match_budget": True, "budgets_applied": True,
     "bench_successful_turns": 10, "bench_total_turns": 10,
     "bench_citation_accuracy": 0.0,
     "bench_enforcement_pass_rate": 0.0,
     "bench_avg_response_words": 9, "bench_avg_confidence": 0.408,
     "persona_score": 0.8453703703703704,
     "memory_score": 0.8055555555555557,
     "depth_off_score": 0.47222222222222227,
     "depth_on_score": 0.8055555555555557,
     "depth_on_status": "measured",
     "presence_score": 0.7777777777777778,
     "suite_score": 0.8085648148148149},
)

FINDING: dict[str, Any] = {
    "budgets_applied_on_all_profiles": True,
    "bench_scores_vary_by_profile": False,
    "suite_scores_vary_by_profile": True,
    "suite_scores_vary_without_the_refused_cell": False,
    "refused_cells": [
        ["pi_zero", "depth_on", "max_refinement_passes=0"]],
    "ceiling_gap_profiles": ["phone_low", "pi_zero", "server_gpu"],
    "widest_profile": "server_gpu",
    "narrowest_profile": "pi_zero",
}

PANEL_RULES: tuple[dict[str, Any], ...] = (
    {"id": "profiles_are_the_shipped_six",
     "rule": "the panel runs exactly the shipped "
             "hardware profiles (the harness's "
             "KNOWN_PROFILES, hardware_profiles/*.json) "
             "— an unknown profile id is refused by "
             "the estate's own validators before "
             "anything runs (D-BENCH-1's TF-121 share)"},
    {"id": "budgets_read_live_from_the_profile_jsons",
     "rule": "every budget a row files — RAM / VRAM "
             "/ ceiling, max params + quant, "
             "generation tokens / refinement passes / "
             "context, bundle + retrieval caps — is "
             "read live from the profile's JSON "
             "through the estate's own loaders and "
             "resolvers (resolve_generation_params, "
             "budget_for_profile); no number is "
             "invented or copied from prose"},
    {"id": "budgets_enforced_during_the_run",
     "rule": "the benchmark cell runs on a "
             "ThoughtForgeCore built with the profile "
             "selected (TF-032's config surface), and "
             "the cell counts only when that core's "
             "resolved max_response_tokens + "
             "refinement passes and its lifecycle "
             "caps equal the profile's filed budgets "
             "— budgets_applied is measured per "
             "profile, and TF-069's boot of the same "
             "profile is re-run beside it"},
    {"id": "ram_ceiling_is_tf069s_machinery",
     "rule": "each profile's cap row is TF-069's "
             "check_model_fit over the profile's own "
             "max_params_b at its default quant "
             "against ram+vram — fit / "
             "exceeds_ceiling / unprovable filed as "
             "data and reproduced against RAM_ROWS; "
             "a gap is named in the row, never "
             "passed silently"},
    {"id": "depth_on_refused_where_no_refinement_is_budgeted",
     "rule": "the depth suite's scaffold (ON) road "
             "runs only where the profile budgets at "
             "least one refinement pass; pi_zero "
             "budgets 0, so its ON cell files "
             "refused_by_budget with the budget named "
             "and its depth score is the single-pass "
             "OFF score — a refusal, never a silent "
             "pass or a borrowed ON score"},
    {"id": "suites_scored_by_their_real_runners",
     "rule": "every suite cell is scored by the "
             "suite's own run_*_suite over its filed "
             "corpus, invoked per profile — the "
             "panel reuses the four suites' corpora, "
             "scorers and runners and copies none of "
             "them"},
    {"id": "panel_rows_are_literal_and_recomputed",
     "rule": "PANEL_ROWS / FINDING are filed as "
             "literal data measured at banking and "
             "recomputed live by the checker — a row "
             "that moves is a finding, never silently "
             "re-filed (TF-067's CURVE precedent)"},
    {"id": "no_vacuous_pass_anywhere",
     "rule": "an empty panel or selection scores "
             "0.0 and files REVIEW; PASS requires all "
             "six profiles measured with budgets "
             "applied exactly and TF-069's boot + "
             "cap rows reproduced"},
    {"id": "identical_scores_are_filed_not_dressed",
     "rule": "the empty-store benchmark metrics do "
             "not vary by profile and the scripted "
             "suite scores vary only through the "
             "refused cell on this estate — an empty "
             "store gives the budgets nothing to bite "
             "on and the suite scorers take no "
             "profile input; the panel files that "
             "shape as its finding, and the one "
             "cell a budget changes is the cell it "
             "refuses (pi_zero's depth ON road)"},
    {"id": "live_per_profile_hardware_is_not_proven",
     "rule": "every row ran on this VM — the pi / "
             "phone profiles in TF-069's simulation "
             "sense (declared hardware, host "
             "backends recorded as the host's truth); "
             "a seeded store under the budgets and "
             "behaviour on real phone / Pi / server "
             "hardware are NOT PROVEN (TF-082 / "
             "TF-134 / TF-173)"},
)


# ── The panel run ───────────────────────────────────────────────────────────

@dataclass
class ProfilePanelResult:
    """One full panel run: rows, finding, verdict data."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    finding: dict[str, Any] = field(default_factory=dict)
    total_cells: int = 0
    refused_cells: int = 0
    overall_score: float = 0.0

    @property
    def budgets_applied_everywhere(self) -> bool:
        return bool(self.rows) and all(
            r["budgets_applied"] for r in self.rows)

    @property
    def boots_reproduce(self) -> bool:
        return all(
            r["boot_verdict"] == BOOT_VERDICTS[r["profile"]]
            for r in self.rows)

    @property
    def caps_reproduce(self) -> bool:
        return all(
            r["cap_status"] == RAM_ROWS[r["profile"]]
            for r in self.rows)

    @property
    def rows_reproduce(self) -> bool:
        return [dict(r) for r in self.rows] == \
            [dict(r) for r in PANEL_ROWS]

    @property
    def verdict(self) -> str:
        """PASS only on the full six-profile panel
        reproducing the filed rows, with budgets
        applied exactly and TF-069's rows standing."""
        if self.total_cells != TOTAL_CELLS:
            return "REVIEW"
        return "PASS" if (
            self.rows_reproduce and self.budgets_applied_everywhere
            and self.boots_reproduce and self.caps_reproduce
        ) else "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_profile_panel(
    profiles: Any = None,
) -> ProfilePanelResult:
    """Run the whole panel (filed profiles default).

    Any explicit profile selection may be given —
    including an empty one, which scores 0.0 and
    keeps the panel at REVIEW (no vacuous pass).
    """
    profiles = PANEL_PROFILES if profiles is None else profiles
    if not profiles:
        return ProfilePanelResult(
            rows=[], finding=finding_for([]),
            total_cells=0, refused_cells=0, overall_score=0.0)
    rows = compute_panel_rows(profiles)
    refused = sum(
        1 for r in rows if r["depth_on_status"] == "refused_by_budget")
    overall = (
        sum(r["suite_score"] for r in rows) / len(rows) if rows else 0.0)
    return ProfilePanelResult(
        rows=[dict(r) for r in rows], finding=finding_for(rows),
        total_cells=len(rows) * CELLS_PER_PROFILE,
        refused_cells=refused, overall_score=overall)


# ── The live checker ────────────────────────────────────────────────────────

def profile_panel_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # TF-069's own estate, re-judged live: the
    # panel stands on its boots + ceilings.
    upstream = profile_runner_problems()
    if upstream:
        problems.append(f"TF-069's estate is not clean: {upstream}")

    # Shape: profiles, cells, rules.
    if tuple(PANEL_PROFILES) != tuple(KNOWN_PROFILES) or len(
            PANEL_PROFILES) != 6:
        problems.append(
            f"PANEL_PROFILES drifted from the shipped six: "
            f"{PANEL_PROFILES}")
    if TOTAL_CELLS != 30:
        problems.append(f"TOTAL_CELLS is {TOTAL_CELLS}, not 30")
    rule_ids = [r["id"] for r in PANEL_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate PANEL_RULES ids")
    for needed in ("profiles_are_the_shipped_six",
                   "budgets_read_live_from_the_profile_jsons",
                   "budgets_enforced_during_the_run",
                   "ram_ceiling_is_tf069s_machinery",
                   "depth_on_refused_where_no_refinement_is_budgeted",
                   "suites_scored_by_their_real_runners",
                   "panel_rows_are_literal_and_recomputed",
                   "no_vacuous_pass_anywhere",
                   "identical_scores_are_filed_not_dressed",
                   "live_per_profile_hardware_is_not_proven"):
        if needed not in rule_ids:
            problems.append(f"PANEL_RULES lost {needed}")

    # The budgets in the filed rows are the JSONs'
    # own values, read live — no invented numbers.
    for row in PANEL_ROWS:
        live = filed_budgets(row["profile"])
        for key, value in live.items():
            if row[key] != value:
                problems.append(
                    f"{row['profile']}: filed budget {key} "
                    f"{row[key]!r} != the profile JSON's {value!r}")
        if row["boot_verdict"] != BOOT_VERDICTS[row["profile"]]:
            problems.append(
                f"{row['profile']}: filed boot verdict "
                f"{row['boot_verdict']} != TF-069's "
                f"{BOOT_VERDICTS[row['profile']]}")
        if row["cap_status"] != RAM_ROWS[row["profile"]]:
            problems.append(
                f"{row['profile']}: filed cap status "
                f"{row['cap_status']} != TF-069's "
                f"{RAM_ROWS[row['profile']]}")

    # The refusal leg, filed exactly: only pi_zero,
    # only because its refinement budget is 0.
    refused = [r["profile"] for r in PANEL_ROWS
               if r["depth_on_status"] == "refused_by_budget"]
    if refused != ["pi_zero"]:
        problems.append(f"refused-by-budget profiles {refused} != "
                        f"['pi_zero']")
    for row in PANEL_ROWS:
        if row["depth_on_status"] == "refused_by_budget":
            if row["max_refinement_passes"] != 0 \
                    or row["depth_on_score"] is not None:
                problems.append(
                    f"{row['profile']}: a refused depth ON road "
                    f"without a 0 refinement budget or with an "
                    f"ON score filed")
        elif row["max_refinement_passes"] < 1 \
                or row["depth_on_score"] is None:
            problems.append(
                f"{row['profile']}: a measured depth ON road "
                f"without a refinement budget")

    # The published panel: literal rows recomputed
    # live, twice — filing, recomputation and
    # determinism in one pass.
    live_rows = compute_panel_rows()
    if [dict(r) for r in live_rows] != [dict(r) for r in PANEL_ROWS]:
        problems.append(
            "PANEL_ROWS drifted from the live recomputation")
    if finding_for(live_rows) != FINDING:
        problems.append("FINDING drifted from the filed rows")
    second_rows = compute_panel_rows()
    if second_rows != live_rows:
        problems.append("the panel is not deterministic: two "
                        "live runs differ")

    # A fresh full run files PASS; an empty panel
    # files 0.0 / REVIEW; a partial selection
    # measures its rows and files REVIEW.
    result = run_profile_panel()
    if result.total_cells != TOTAL_CELLS or result.verdict != "PASS" \
            or not result.passes:
        problems.append(
            f"the live panel run does not file {TOTAL_CELLS} cells "
            f"/ PASS: {result.total_cells} / {result.verdict}")
    if result.refused_cells != 1:
        problems.append(
            f"the live panel files {result.refused_cells} refused "
            f"cells, not 1")
    empty = run_profile_panel(profiles=())
    if empty.total_cells != 0 or empty.overall_score != 0.0 \
            or empty.verdict != "REVIEW" or empty.passes:
        problems.append("an empty panel does not file 0.0 / REVIEW")
    subset = run_profile_panel(profiles=("desktop_cpu",))
    if subset.total_cells != CELLS_PER_PROFILE \
            or subset.verdict != "REVIEW":
        problems.append(
            "a single-profile selection does not measure its "
            "cells / REVIEW mechanically")

    # Every earlier pinned baseline preserved exactly.
    from benchmarks.harness import baseline_problems

    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"earlier harness baselines moved: {earlier}")

    doc = Path(__file__).resolve().parent.parent / PROFILE_PANEL_DOC
    if not doc.exists():
        problems.append(f"the write-up {PROFILE_PANEL_DOC} is missing")
    return problems


def profile_panel_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_profile_panel()
    return {
        "format": PROFILE_PANEL_FORMAT,
        "doc": PROFILE_PANEL_DOC,
        "profiles": list(PANEL_PROFILES),
        "suites": list(PANEL_SUITES),
        "rows": [dict(r) for r in PANEL_ROWS],
        "finding": dict(FINDING),
        "measured": {
            "total_cells": result.total_cells,
            "refused_cells": result.refused_cells,
            "overall_score": result.overall_score,
            "verdict": result.verdict,
        },
        "rules": [dict(r) for r in PANEL_RULES],
    }


__all__ = [
    "CELLS_PER_PROFILE",
    "FINDING",
    "PANEL_PROFILES",
    "PANEL_ROWS",
    "PANEL_RULES",
    "PANEL_SUITES",
    "PROFILE_PANEL_DOC",
    "PROFILE_PANEL_FORMAT",
    "TOTAL_CELLS",
    "ProfilePanelResult",
    "compute_panel_rows",
    "filed_budgets",
    "finding_for",
    "measure_profile_row",
    "profile_panel_as_data",
    "profile_panel_problems",
    "run_profile_panel",
]
