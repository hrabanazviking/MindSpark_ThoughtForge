"""TF-098 — Cost discipline: refinement passes
capped per profile (a second small-model call is
a budget, not a habit).

Roadmap: "Cost discipline: refinement passes
capped per profile (a second small-model call is
a budget, not a habit)."

Measured start (probed live this tick, through
the real ``FragmentSalvage`` with a counting stub
engine whose every output improves):

- **Salvage itself honoured any number it was
  handed.** ``forge(max_passes=5)`` made 5 engine
  calls and returned ``refine_pass_5`` — beyond
  every shipped profile's cap (the largest is 3).
  A negative ``max_passes`` silently meant 0.
  Nothing inside ``salvage.py`` knew a profile
  existed, so every caller except the core
  bypassed the caps entirely — and the core leg
  was the only enforcement in the tree.
- **The caps themselves are real, and were
  already live in exactly one place.** Every
  shipped profile JSON files
  ``generation.max_refinement_passes`` (pi_zero 0,
  phone_low 1, pi_5 2, desktop_cpu 2, desktop_gpu
  3, server_gpu 3), and TF-032/TF-037's config
  surface resolves the selected profile's value
  for the core's compose step (D-PROF-4's floor
  share, fixed there). TF-098's assigned share —
  stated in the docket — is the general
  discipline: enforcement at the engine-call site
  itself, and measured refine cost across all
  profiles.
- **Refine cost was counted nowhere.** TF-097's
  rescue record carries each pass's outcome and
  its two latencies, but nothing summed them:
  no attempted / completed / adopted /
  not-adopted counts, no wall or reported totals,
  and token counts are read nowhere in
  ``refinement/`` (grepped — the engine's token
  fields stop at the core). A caller could not
  answer "what did refinement cost this turn?"
  from anything the tree filed. This is filed
  here as D-COST-1, found by this slice's probes
  and fixed in scope.

What this slice builds:

- **THE CAPS ARE THE PROFILES' OWN, READ LIVE.**
  ``profile_refine_cap`` reads a profile's
  ``generation.max_refinement_passes`` through
  ``load_hardware_profile`` at call time (the
  TF-033 shape) — there is no copied cap table in
  this module that could drift from the JSONs.
  An unknown profile raises ``ValueError`` naming
  the valid ids: a bogus profile never silently
  inherits a cap.
- **A CAP IS A CEILING, NOT A TARGET.**
  ``resolve_refine_budget(requested, profile_id)``
  returns the effective pass count as
  ``min(requested, cap)`` when a profile is
  named — a caller asking for less than the cap
  gets what it asked for, never the cap. When no
  request is named, the standing default (2,
  salvage's own ``_MAX_REFINE_PASSES``) is the
  request. A negative or non-integer request
  raises ``ValueError`` by name through this
  resolver — through ``forge()`` the pre-slice
  silent-zero leg is unchanged for callers who
  name no profile (stated, not hidden).
- **ZERO MEANS ZERO.** pi_zero's cap is 0: with
  that profile named, ``forge()`` makes no engine
  call at all and returns the best draft — the
  profile's promise kept at the call site, not
  only through the core.
- **NO PROFILE, NO INVENTED CAP.** With no
  profile named there is no cap to apply and
  this slice invents none: the caller's request
  (or the standing default) stands, exactly as
  before. Through the core this leg is already
  disciplined — TF-032's surface hands compose
  the standing 2 when no profile is selected.
  The filed fixtures pin both legs so the
  boundary is tested, not assumed.
- **SALVAGE ENFORCES THE CAP ITSELF**
  (D-COST-1's fix). ``FragmentSalvage.forge``
  gains a defaulted ``profile_id`` keyword: when
  one is supplied, the effective pass count is
  resolved through this module before any engine
  call is made, so a direct caller naming a
  profile cannot run past its cap. The core's
  ``_compose_final`` now passes its selected
  profile id through beside the already-resolved
  ``max_passes`` — defence in depth at the call
  site; the effective number through the core is
  unchanged on every profile, so core behaviour
  is byte-identical. With no profile id, every
  leg of ``forge()`` is byte-identical to before.
- **COST IS COUNTED IN ENGINE CALLS.**
  ``refine_cost`` reads a TF-097 rescue record
  (judged by TF-097's own checker first — an
  unsound record raises, it is never totalled)
  and counts what refinement spent: passes
  attempted, completed, adopted, completed-but-
  not-adopted (the wasted spend — a second call
  that bought nothing is named as such), and
  failed; and the two latency sums, wall and
  reported, kept apart exactly as TF-097 files
  them (``reported_ms`` is ``None`` when no
  completed pass reported a duration — never a
  fabricated zero). One pass is one
  ``engine.generate`` call: that call is the
  expensive unit on small hardware, and it is
  the unit the profiles budget. Token counts per
  pass are NOT carried in TF-097's record and
  are not invented here (filed limit; TF-133's
  cost panel owns tokens + time per suite).
- **THE SURFACES ARE UNCHANGED.** The scoring
  formula, TF-088's and TF-096's guards, the
  adoption rule, ``passes_used`` semantics and
  ``salvage_path`` naming all stand; the profile
  JSONs are read, never edited; the config
  surface's resolution is reused as the core's
  request, never duplicated.

Proof grade FIXTURE — every shipped profile's
cap is exercised live through the real
``FragmentSalvage`` with a counting stub engine
on this VM (requested 5, calls land exactly on
the cap), and the cost accounting is proven
against TF-097's filed records. A live model's
refine cost is NOT PROVEN (no model in tree;
TF-101 scores the stack over TF-100, TF-133 owns
the token/time panel), never claimed.

The human-readable write-up is
``docs/architecture/COST_DISCIPLINE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: This module's estate format.
COST_FORMAT = "thoughtforge-cost-discipline/1"

COST_DOC = "docs/architecture/COST_DISCIPLINE.md"

#: The standing refine-pass default when neither a
#: caller request nor a profile says otherwise —
#: salvage's own ``_MAX_REFINE_PASSES``, also filed
#: as TF-032's ``DEFAULT_REFINE_PASSES``. The data
#: checker verifies all three agree, so this number
#: cannot drift from either surface it names.
DEFAULT_REFINE_CAP = 2

# ── The rules ───────────────────────────────────────────────────────────────

COST_RULES: tuple[dict[str, Any], ...] = (
    {"id": "caps_are_the_profiles_own_read_live",
     "rule": "a profile's refine cap is its own "
             "generation.max_refinement_passes, read live "
             "from the shipped profile JSON through "
             "load_hardware_profile at call time — this "
             "module files no copied cap table that could "
             "drift from the profiles"},
    {"id": "a_cap_is_a_ceiling_not_a_target",
     "rule": "the effective pass count under a profile is "
             "min(requested, cap): a caller asking for less "
             "than the cap gets what it asked for, never "
             "the cap; a caller naming no request gets the "
             "standing default as its request"},
    {"id": "salvage_enforces_the_cap_itself",
     "rule": "D-COST-1 (FIXED by TF-098): "
             "FragmentSalvage.forge(profile_id=...) resolves "
             "the effective pass count through this module "
             "before any engine call, so a caller naming a "
             "profile cannot run past its cap at the call "
             "site — before this slice the core's resolved "
             "max_passes was the only enforcement and direct "
             "forge() callers bypassed the caps entirely"},
    {"id": "zero_means_zero",
     "rule": "a profile whose cap is 0 (pi_zero) makes no "
             "engine call at all under that profile: forge() "
             "returns the best draft, the profile's promise "
             "kept at the call site"},
    {"id": "no_profile_no_invented_cap",
     "rule": "with no profile named there is no cap to "
             "apply and none is invented: the caller's "
             "request (or the standing default) stands "
             "exactly as before this slice — through the "
             "core that request is TF-032's standing 2 when "
             "no profile is selected"},
    {"id": "unknown_profile_raises_by_name",
     "rule": "an unknown profile id raises ValueError "
             "naming the value and the valid ids; a bogus "
             "profile never silently inherits a cap"},
    {"id": "bad_request_raises_by_name",
     "rule": "a negative or non-integer pass request "
             "raises ValueError naming the value through "
             "this resolver — a caller bug is never "
             "silently read as zero (forge()'s no-profile "
             "leg is unchanged, stated in the docstring)"},
    {"id": "cost_is_counted_in_engine_calls",
     "rule": "refine_cost counts a TF-097 rescue record in "
             "engine calls — attempted, completed, adopted, "
             "completed-but-not-adopted (the wasted spend), "
             "failed — after judging the record by TF-097's "
             "own checker; an unsound record raises and is "
             "never totalled. Token counts are not carried "
             "in the record and are never invented here"},
    {"id": "latencies_summed_named_never_conflated",
     "rule": "the cost's wall_ms sums TF-097's measured "
             "wall latencies and reported_ms sums only the "
             "durations engines reported (None when none "
             "did) — the two are filed apart, exactly as "
             "TF-097 records them, neither substituted for "
             "the other"},
    {"id": "the_surfaces_are_unchanged",
     "rule": "the scoring formula, TF-088's and TF-096's "
             "guards, the adoption rule, passes_used "
             "semantics and salvage_path naming are "
             "unchanged; the profile JSONs are read never "
             "edited; the core passes its selected profile "
             "id beside its already-resolved max_passes, so "
             "the effective number through the core is "
             "unchanged on every profile"},
)

# ── Cap resolution ──────────────────────────────────────────────────────────


def _valid_profile_ids() -> list[str]:
    from thoughtforge.utils.config import list_hardware_profiles

    return list_hardware_profiles()


def profile_refine_cap(profile_id: str) -> int:
    """One shipped profile's refine-pass cap, read
    live from its JSON (``generation.
    max_refinement_passes``). An unknown profile
    raises ``ValueError`` naming the valid ids."""
    from thoughtforge.utils.config import load_hardware_profile

    valid = _valid_profile_ids()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile {profile_id!r}; "
            f"valid profiles: {', '.join(valid)}")
    profile = load_hardware_profile(profile_id)
    generation = profile.get("generation") or {}
    cap = generation.get("max_refinement_passes")
    if cap is None:
        raise ValueError(
            f"hardware profile {profile_id!r} files no "
            "generation.max_refinement_passes")
    return int(cap)


def profile_refine_caps() -> dict[str, int]:
    """Every shipped profile's cap, read live."""
    return {pid: profile_refine_cap(pid)
            for pid in _valid_profile_ids()}


@dataclass(frozen=True)
class RefineBudget:
    """One forge() request resolved against a cap."""

    profile_id: str | None      # the profile named, or None
    requested_passes: int       # what the caller asked for
    profile_cap: int | None     # the profile's cap, or None
    effective_passes: int       # what may actually run
    governed_by: str            # profile | caller_request | standing_default

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "requested_passes": self.requested_passes,
            "profile_cap": self.profile_cap,
            "effective_passes": self.effective_passes,
            "governed_by": self.governed_by,
        }


def resolve_refine_budget(
    requested_passes: int | None = None,
    *,
    profile_id: str | None = None,
) -> RefineBudget:
    """Resolve one refine request against a profile's
    cap. ``requested_passes=None`` is the standing
    default. Under a profile the effective count is
    ``min(requested, cap)``; with no profile the
    request stands (no cap is invented). A negative
    or non-integer request, or an unknown profile,
    raises ``ValueError`` naming the value."""
    if requested_passes is None:
        requested = DEFAULT_REFINE_CAP
        defaulted = True
    else:
        if isinstance(requested_passes, bool) or \
                not isinstance(requested_passes, int):
            raise ValueError(
                f"refine pass request {requested_passes!r} is "
                "not an integer count")
        if requested_passes < 0:
            raise ValueError(
                f"refine pass request {requested_passes} is "
                "negative: a pass count cannot be below zero")
        requested = requested_passes
        defaulted = False
    if profile_id is None:
        return RefineBudget(
            profile_id=None, requested_passes=requested,
            profile_cap=None, effective_passes=requested,
            governed_by="standing_default" if defaulted
            else "caller_request")
    cap = profile_refine_cap(profile_id)
    effective = min(requested, cap)
    return RefineBudget(
        profile_id=profile_id, requested_passes=requested,
        profile_cap=cap, effective_passes=effective,
        governed_by="profile" if effective < requested
        else "caller_request")


# ── Cost accounting (over TF-097's rescue record) ───────────────────────────


@dataclass(frozen=True)
class RefineCost:
    """What one forge()'s refinement spent, counted
    from its TF-097 rescue record."""

    passes_attempted: int
    passes_completed: int       # counted in passes_used
    passes_adopted: int
    passes_not_adopted: int     # completed, bought nothing
    passes_failed: int
    wall_ms: float              # salvage's measured wall sum
    reported_ms: float | None   # engines' reported sum; None if none reported
    reported_passes: int        # completed passes that reported a duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes_attempted": self.passes_attempted,
            "passes_completed": self.passes_completed,
            "passes_adopted": self.passes_adopted,
            "passes_not_adopted": self.passes_not_adopted,
            "passes_failed": self.passes_failed,
            "wall_ms": self.wall_ms,
            "reported_ms": self.reported_ms,
            "reported_passes": self.reported_passes,
        }


def refine_cost(record: Any) -> RefineCost:
    """Count one rescue record's refine cost. The
    record is judged by TF-097's own checker first —
    an unsound record raises ``ValueError`` with its
    problems and is never totalled."""
    from thoughtforge.refinement.rescue_provenance import (
        rescue_record_problems,
    )

    problems = rescue_record_problems(record)
    if problems:
        raise ValueError(
            "unsound rescue record, not totalled: "
            + "; ".join(problems))
    passes = record["refine_passes"]
    completed = [p for p in passes
                 if p["counted_in_passes_used"]]
    adopted = [p for p in passes if p["outcome"] == "adopted"]
    reported = [p["reported_latency_ms"] for p in completed
                if p["reported_latency_ms"] is not None]
    return RefineCost(
        passes_attempted=len(passes),
        passes_completed=len(completed),
        passes_adopted=len(adopted),
        passes_not_adopted=len(completed) - len(adopted),
        passes_failed=sum(
            1 for p in passes if p["outcome"] == "failed"),
        wall_ms=round(sum(
            p["latency_ms"] or 0.0 for p in passes), 3),
        reported_ms=(round(sum(reported), 3) if reported
                     else None),
        reported_passes=len(reported),
    )


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic throughout (TF-017's Lantern Hollow,
# never canon). Each cap fixture asks for 5 passes
# — beyond every shipped cap — through the real
# FragmentSalvage under one named profile, with a
# counting stub engine whose every output improves
# (pool sentences only, so TF-088's guard passes
# it), and files the engine calls that may land.

_BASE = (
    "According to Q123, the ferry crossing takes eleven "
    "minutes in calm water. Wrenna records the river level "
    "each morning in chalk on the mill door.")

#: The counting stub's reported duration per call.
STUB_DURATION_MS = 7.0

COST_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "cap_pi_zero", "profile_id": "pi_zero",
     "requested": 5, "expected_effective": 0,
     "expected_calls": 0, "expected_path": "best_draft"},
    {"id": "cap_phone_low", "profile_id": "phone_low",
     "requested": 5, "expected_effective": 1,
     "expected_calls": 1, "expected_path": "refine_pass_1"},
    {"id": "cap_pi_5", "profile_id": "pi_5",
     "requested": 5, "expected_effective": 2,
     "expected_calls": 2, "expected_path": "refine_pass_2"},
    {"id": "cap_desktop_cpu", "profile_id": "desktop_cpu",
     "requested": 5, "expected_effective": 2,
     "expected_calls": 2, "expected_path": "refine_pass_2"},
    {"id": "cap_desktop_gpu", "profile_id": "desktop_gpu",
     "requested": 5, "expected_effective": 3,
     "expected_calls": 3, "expected_path": "refine_pass_3"},
    {"id": "cap_server_gpu", "profile_id": "server_gpu",
     "requested": 5, "expected_effective": 3,
     "expected_calls": 3, "expected_path": "refine_pass_3"},
    {"id": "caller_below_cap", "profile_id": "desktop_gpu",
     "requested": 1, "expected_effective": 1,
     "expected_calls": 1, "expected_path": "refine_pass_1"},
    {"id": "standing_default_no_profile", "profile_id": None,
     "requested": None, "expected_effective": 2,
     "expected_calls": 2, "expected_path": "refine_pass_2"},
    {"id": "caller_request_no_profile", "profile_id": None,
     "requested": 5, "expected_effective": 5,
     "expected_calls": 5, "expected_path": "refine_pass_5",
     "note": "the boundary, filed not hidden: with no "
             "profile named there is no cap to apply and "
             "none is invented — the caller's request "
             "stands, exactly as before this slice"},
)

_FIXTURE_BY_ID = {f["id"]: f for f in COST_FIXTURES}

#: Cost-accounting fixtures over TF-097's filed
#: rescue fixtures: the record named, and the cost
#: it must total to.
COST_RECORD_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "cost_adopt_then_fabricate",
     "rescue_fixture": "refine_adopt_then_fabricate",
     "expected": {"passes_attempted": 2, "passes_completed": 2,
                  "passes_adopted": 1, "passes_not_adopted": 1,
                  "passes_failed": 0, "reported_ms": 7.0,
                  "reported_passes": 2}},
    {"id": "cost_single_adopted",
     "rescue_fixture": "refine_adopted",
     "expected": {"passes_attempted": 1, "passes_completed": 1,
                  "passes_adopted": 1, "passes_not_adopted": 0,
                  "passes_failed": 0, "reported_ms": 3.5,
                  "reported_passes": 1}},
    {"id": "cost_failed_attempt",
     "rescue_fixture": "refine_engine_failure_falls_back",
     "expected": {"passes_attempted": 1, "passes_completed": 0,
                  "passes_adopted": 0, "passes_not_adopted": 0,
                  "passes_failed": 1, "reported_ms": None,
                  "reported_passes": 0}},
    {"id": "cost_no_engine",
     "rescue_fixture": "best_draft_no_engine",
     "expected": {"passes_attempted": 0, "passes_completed": 0,
                  "passes_adopted": 0, "passes_not_adopted": 0,
                  "passes_failed": 0, "reported_ms": None,
                  "reported_passes": 0}},
)


class _CountingEngine:
    """A stub engine that counts its own calls and
    always improves: each output is the pool base
    repeated one more time (longer, same citations,
    no new checkable content for TF-088's guard)."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, params=None):
        from types import SimpleNamespace

        self.calls += 1
        text = " ".join([_BASE] * (self.calls + 1))
        return SimpleNamespace(
            text=text, duration_ms=STUB_DURATION_MS)


def run_cost_fixture(fixture_id: str):
    """Run one filed cap fixture through the real
    ``FragmentSalvage`` (its ``profile_id`` named,
    exactly as a disciplined caller now may) and
    return ``(result, engine, budget)``."""
    from thoughtforge.cognition.prompt_builder import PromptBuilder
    from thoughtforge.knowledge.models import (
        ActivatedRecord,
        CandidateRecord,
        CandidateScores,
        CognitionScaffold,
        InputSketch,
        MemoryActivationBundle,
    )
    from thoughtforge.refinement.salvage import FragmentSalvage

    fixture = _FIXTURE_BY_ID[fixture_id]
    budget = resolve_refine_budget(
        fixture["requested"], profile_id=fixture["profile_id"])
    candidates = [CandidateRecord(
        candidate_id="cand_0", mode="practical", text=_BASE,
        token_estimate=len(_BASE.split()), scores=CandidateScores())]
    bundle = MemoryActivationBundle(
        activated_records=[ActivatedRecord(
            "Q123", "entity", 0.8, "cue for Q123",
            raw={"qid": "Q123"})],
        total_retrieved=1, retrieval_confidence=0.8)
    sketch = InputSketch(
        raw_text="when does the ferry leave?",
        intent="factual_query", topic="ferry",
        response_mode="factual_direct")
    scaffold = CognitionScaffold(
        goal="give a clear direct answer",
        tone=["direct", "factual"], avoid=["filler"])
    engine = _CountingEngine()
    requested = fixture["requested"]
    result = FragmentSalvage().forge(
        candidates, bundle, engine=engine,
        prompt_builder=PromptBuilder(), sketch=sketch,
        scaffold=scaffold,
        max_passes=(requested if requested is not None
                    else DEFAULT_REFINE_CAP),
        profile_id=fixture["profile_id"])
    return result, engine, budget


# ── Checkers ────────────────────────────────────────────────────────────────


def cost_discipline_problems() -> list[str]:
    """Data-integrity checker: rule and fixture ids
    unique, fixtures well-formed, the standing
    default agreeing with both surfaces it names
    (salvage's ``_MAX_REFINE_PASSES`` and TF-032's
    ``DEFAULT_REFINE_PASSES``), every shipped
    profile's live cap matching its fixture's
    filed effective count, and the wiring anchors
    present (forge resolves a named profile; the
    core passes its profile id through)."""
    import inspect

    problems: list[str] = []
    rule_ids = [r["id"] for r in COST_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate COST_RULES ids")
    fixture_ids = [f["id"] for f in COST_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate COST_FIXTURES ids")
    record_ids = [f["id"] for f in COST_RECORD_FIXTURES]
    if len(record_ids) != len(set(record_ids)):
        problems.append("duplicate COST_RECORD_FIXTURES ids")

    from thoughtforge.cognition.config_surface import (
        DEFAULT_REFINE_PASSES,
    )
    from thoughtforge.refinement import salvage as salvage_module

    if DEFAULT_REFINE_CAP != salvage_module._MAX_REFINE_PASSES:
        problems.append("DEFAULT_REFINE_CAP disagrees with "
                        "salvage._MAX_REFINE_PASSES")
    if DEFAULT_REFINE_CAP != DEFAULT_REFINE_PASSES:
        problems.append("DEFAULT_REFINE_CAP disagrees with "
                        "TF-032's DEFAULT_REFINE_PASSES")

    caps = profile_refine_caps()
    filed_profiles = set()
    for fixture in COST_FIXTURES:
        pid = fixture["profile_id"]
        if pid is None:
            continue
        filed_profiles.add(pid)
        budget = resolve_refine_budget(
            fixture["requested"], profile_id=pid)
        if fixture["expected_effective"] != budget.effective_passes:
            problems.append(
                f"fixture {fixture['id']}: filed effective "
                f"{fixture['expected_effective']} != resolved "
                f"{budget.effective_passes}")
        if fixture["expected_calls"] != budget.effective_passes:
            problems.append(
                f"fixture {fixture['id']}: filed calls "
                f"{fixture['expected_calls']} != the effective "
                "pass count (one pass is one engine call)")
    if filed_profiles != set(caps):
        problems.append(
            "the cap fixtures do not cover every shipped "
            f"profile: filed {sorted(filed_profiles)} vs "
            f"shipped {sorted(caps)}")

    salvage_source = inspect.getsource(salvage_module)
    if "resolve_refine_budget" not in salvage_source:
        problems.append("salvage.py does not resolve a named "
                        "profile's budget")
    if "profile_id" not in salvage_source:
        problems.append("salvage.forge names no profile_id")
    from thoughtforge.cognition.core import ThoughtForgeCore

    if "profile_id" not in inspect.getsource(
            ThoughtForgeCore._compose_final):
        problems.append("core._compose_final does not pass its "
                        "profile id to salvage")
    return problems


def cost_discipline_live_problems() -> list[str]:
    """Live checker: every filed cap fixture is run
    through the real FragmentSalvage and must land
    exactly as filed — the counting engine's calls
    equal the effective count (never the uncapped
    request), the path names the passes that ran —
    and every cost-record fixture's TF-097 record
    totals to its filed cost; the resolver's error
    legs raise by name."""
    problems: list[str] = []
    for fixture in COST_FIXTURES:
        result, engine, budget = run_cost_fixture(fixture["id"])
        if budget.effective_passes != fixture["expected_effective"]:
            problems.append(
                f"fixture {fixture['id']}: effective "
                f"{budget.effective_passes} != "
                f"{fixture['expected_effective']}")
        if engine.calls != fixture["expected_calls"]:
            problems.append(
                f"fixture {fixture['id']}: engine calls "
                f"{engine.calls} != {fixture['expected_calls']}")
        if result.salvage_path != fixture["expected_path"]:
            problems.append(
                f"fixture {fixture['id']}: path "
                f"{result.salvage_path} != "
                f"{fixture['expected_path']}")
        if result.passes_used != fixture["expected_calls"]:
            problems.append(
                f"fixture {fixture['id']}: passes_used "
                f"{result.passes_used} != the calls that ran")
        cost = refine_cost(result.rescue_record)
        if cost.passes_completed != engine.calls:
            problems.append(
                f"fixture {fixture['id']}: the record's "
                "completed passes disagree with the engine's "
                "own call count")
    from thoughtforge.refinement.rescue_provenance import (
        run_rescue_fixture,
    )

    for fixture in COST_RECORD_FIXTURES:
        record = run_rescue_fixture(
            fixture["rescue_fixture"]).rescue_record
        cost = refine_cost(record).to_dict()
        for key, want in fixture["expected"].items():
            if cost[key] != want:
                problems.append(
                    f"record fixture {fixture['id']}: {key} "
                    f"{cost[key]} != {want}")
    try:
        profile_refine_cap("bogus_profile")
        problems.append("an unknown profile did not raise")
    except ValueError as exc:
        if "bogus_profile" not in str(exc):
            problems.append("the unknown-profile error does not "
                            "name its value")
    for bad in (-1, 1.5, "two"):
        try:
            resolve_refine_budget(bad)  # type: ignore[arg-type]
            problems.append(f"request {bad!r} did not raise")
        except ValueError as exc:
            if str(bad) not in str(exc) and \
                    repr(bad) not in str(exc):
                problems.append(f"the bad-request error for "
                                f"{bad!r} does not name its value")
    try:
        refine_cost({"format": "not-a-record"})
        problems.append("an unsound record was totalled")
    except ValueError:
        pass
    return problems


def cost_discipline_as_data() -> dict[str, Any]:
    return {
        "format": COST_FORMAT,
        "cost_doc": COST_DOC,
        "default_refine_cap": DEFAULT_REFINE_CAP,
        "profile_caps": profile_refine_caps(),
        "rules": [dict(r) for r in COST_RULES],
        "fixtures": [
            {k: v for k, v in f.items()} for f in COST_FIXTURES],
        "record_fixtures": [
            {k: v for k, v in f.items()}
            for f in COST_RECORD_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_refine_cost": "not_proven",
            "d_cost_1": "fixed",
            "d_prof_4": "fixed_tf037_floor_share_tf098_general",
            "tokens_per_pass": "not_carried_in_tf097_record",
            "salvage_scoring_and_guards": "unchanged",
        },
    }
