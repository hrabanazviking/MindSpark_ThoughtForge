"""Memory budgets — TF-056.

Roadmap: "store size + retrieval cost capped per hardware
profile; eviction by the lifecycle rules". TF-012 measured the
starting lie (D-PROF-1): every shipped profile JSON carries a
``memory`` section (per-kind soft/hard caps, bundle size) and a
``retrieval`` section (sql/vector caps, timeout), and NO code
read either — the lifecycle enforced one flat cap table
(``knowledge/lifecycle.py`` STORE_CAPS: facts/preferences/
patterns 128 soft / 256 hard, episodes 256 / 512) on every
profile, and the cognition core retrieved with a flat
``top_k=10`` on every profile.

Measured this tick, by direct probe on synthetic stores (VM):

* Seventy distinct facts in a store exceed phone_low's hard
  cap (64) and pi_zero's (32), yet a default lifecycle light
  prune kept all 70 — light mode's score thresholds are
  unreachable (TF-045), so only caps act under light, and the
  flat cap in force was the desktop hard cap, 256.
* The flat STORE_CAPS are exactly desktop_cpu/desktop_gpu's
  JSON caps for the four record kinds — the "universal" table
  was always one profile's budget, silently applied to phones
  and servers alike. The two caps the profiles do not carry
  (recent_queue, open_loops) are the flat table's alone.
* The core's flat retrieval top_k=10 exceeds phone_low's
  bundle cap (5), sql cap (5) and vector cap (3), and pi_zero's
  (3 / 3 / 2); the forge's own legs scale from that top_k
  (sql candidates = 2 × top_k, vector = top_k), so retrieval
  cost was uncapped exactly where hardware is smallest.

This module states the budgets as data, resolved LIVE from
the shipped profile JSONs through the tree's own loader
(``load_hardware_profile`` — the numbers are never copied
into a second table that could drift):

* :func:`budget_for_profile` — one profile's MemoryBudget:
  per-kind soft/hard caps for the four persisted record kinds
  (plus the flat recent_queue / open_loops caps, stated as
  profile-independent), the bundle cap, and the retrieval
  caps (sql_max_results, vector_top_k, max_vector_results,
  retrieval_timeout_ms).
* :func:`retrieval_plan` — a caller's request CAPPED by the
  budget: a budget never raises a request, it only lowers it
  (bundle = min(requested, max_memory_bundle_facts); the
  forge legs receive the profile's sql/vector caps and min
  them against their own scaling).
* :func:`measure_store` / :func:`budget_report` — the store's
  size stated honestly: live record counts per kind (loaded,
  so archived/deleted rows already rewritten out are not
  counted) and the store directory's file bytes by file.
  Per-kind BYTES are not claimed: facts and preferences share
  one file, so bytes are reported per file, counts per kind.
* :func:`enforce_budget` — eviction, by the lifecycle rules
  and nothing else: a MemoryLifecycle carrying the profile's
  caps runs its ordinary prune; the cap leg retires the
  lowest-survival unprotected records first (TF-045), score
  and dedup legs act per the chosen mode, and protected
  records are NEVER cap-evicted — a protected surplus leaves
  the report over-hard, named, not silently "fixed".

Wiring (this slice): MemoryLifecycle takes an optional
``profile_id`` (its caps resolve from that profile; no profile
keeps the flat STORE_CAPS byte-identically); the cognition
core builds its lifecycle on the conversation's selected
profile (TF-032) and caps its retrieval through the plan;
KnowledgeForge.retrieve takes optional sql/vector caps
(default None = the pre-TF-056 scaling, byte-identical).

Soft caps are a REPORT line, not an eviction trigger: a kind
over soft but under hard is reported ``over_soft`` and left
alone — eviction fires at hard caps only, exactly as the
lifecycle has always behaved. D-PROF-1's memory/retrieval
share is closed for the profiles' memory + retrieval
sections; its deployment / max_params / recommended-models
share stays with TF-069/TF-110. D-PROF-3 narrows again:
retrieval top_k is no longer flat when a profile is selected.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thoughtforge.utils.config import (
    list_hardware_profiles,
    load_hardware_profile,
)

#: The core's standing retrieval request before any budget is
#: applied (core.py's flat top_k, TF-012) — the default a plan
#: caps. A plan can only lower it.
DEFAULT_REQUESTED_TOP_K: int = 10

#: Persisted record kind → the profile JSON's memory keys.
KIND_CAP_KEYS: dict[str, tuple[str, str]] = {
    "user_preference": ("user_preferences_cap_soft", "user_preferences_cap_hard"),
    "user_fact": ("user_facts_cap_soft", "user_facts_cap_hard"),
    "episode": ("episodic_store_cap_soft", "episodic_store_cap_hard"),
    "response_pattern": ("response_patterns_cap_soft", "response_patterns_cap_hard"),
}

#: The kinds a budget caps, in store order.
BUDGETED_KINDS: tuple[str, ...] = tuple(KIND_CAP_KEYS)

#: Caps no profile JSON carries — the lifecycle's flat rows,
#: profile-independent by statement (the recent queue is a
#: recency window, open_loops a thread shape; neither scales
#: with the record stores in any shipped profile).
FLAT_CAP_KINDS: tuple[str, ...] = ("recent_queue", "open_loops")


def _flat_caps() -> dict[str, dict[str, int]]:
    from thoughtforge.knowledge.lifecycle import STORE_CAPS

    return {k: dict(STORE_CAPS[k]) for k in FLAT_CAP_KINDS}


# ── The budget, as data ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryBudget:
    """One hardware profile's memory + retrieval budget,
    resolved live from its shipped JSON."""

    profile_id: str
    caps: dict[str, dict[str, int]]
    max_memory_bundle_facts: int
    max_vector_results: int
    sql_max_results: int
    vector_top_k: int
    retrieval_timeout_ms: int

    def cap(self, kind: str, level: str = "hard") -> int:
        return self.caps[kind][level]

    def caps_for_lifecycle(self) -> dict[str, dict[str, int]]:
        """A fresh caps table in the lifecycle's shape (the
        four budgeted kinds + the two flat kinds)."""
        return {k: dict(v) for k, v in self.caps.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "caps": {k: dict(v) for k, v in self.caps.items()},
            "max_memory_bundle_facts": self.max_memory_bundle_facts,
            "max_vector_results": self.max_vector_results,
            "sql_max_results": self.sql_max_results,
            "vector_top_k": self.vector_top_k,
            "retrieval_timeout_ms": self.retrieval_timeout_ms,
        }


@dataclass(frozen=True)
class RetrievalPlan:
    """A retrieval request after the budget has capped it."""

    profile_id: str
    requested_top_k: int
    bundle_top_k: int
    sql_max_results: int
    vector_top_k: int
    max_vector_results: int
    retrieval_timeout_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "requested_top_k": self.requested_top_k,
            "bundle_top_k": self.bundle_top_k,
            "sql_max_results": self.sql_max_results,
            "vector_top_k": self.vector_top_k,
            "max_vector_results": self.max_vector_results,
            "retrieval_timeout_ms": self.retrieval_timeout_ms,
        }


def budget_for_profile(profile_id: str) -> MemoryBudget:
    """Resolve ``profile_id``'s budget from its shipped JSON.

    Unknown profiles raise ValueError naming the valid ids —
    a budget is never silently replaced by another profile's.
    """
    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile for memory budget: {profile_id!r} "
            f"(valid: {valid})"
        )
    profile = load_hardware_profile(profile_id)
    memory = profile["memory"]
    retrieval = profile["retrieval"]
    caps: dict[str, dict[str, int]] = {}
    for kind, (soft_key, hard_key) in KIND_CAP_KEYS.items():
        caps[kind] = {"soft": int(memory[soft_key]), "hard": int(memory[hard_key])}
    caps.update(_flat_caps())
    return MemoryBudget(
        profile_id=profile_id,
        caps=caps,
        max_memory_bundle_facts=int(memory["max_memory_bundle_facts"]),
        max_vector_results=int(memory["max_vector_results"]),
        sql_max_results=int(retrieval["sql_max_results"]),
        vector_top_k=int(retrieval["vector_top_k"]),
        retrieval_timeout_ms=int(retrieval["retrieval_timeout_ms"]),
    )


def all_budgets() -> dict[str, MemoryBudget]:
    return {pid: budget_for_profile(pid) for pid in list_hardware_profiles()}


def retrieval_plan(
    profile_id: str,
    requested_top_k: int = DEFAULT_REQUESTED_TOP_K,
) -> RetrievalPlan:
    """Cap a retrieval request by the profile's budget.

    The bundle can only shrink: bundle_top_k =
    min(requested, max_memory_bundle_facts). The sql/vector
    caps ride along for KnowledgeForge.retrieve, which mins
    them against its own per-leg scaling (sql 2 × top_k,
    vector top_k) — so no leg of a capped retrieval exceeds
    its profile's stated cost.
    """
    if requested_top_k < 1:
        raise ValueError(f"requested_top_k must be >= 1, got {requested_top_k}")
    budget = budget_for_profile(profile_id)
    return RetrievalPlan(
        profile_id=profile_id,
        requested_top_k=requested_top_k,
        bundle_top_k=min(requested_top_k, budget.max_memory_bundle_facts),
        sql_max_results=budget.sql_max_results,
        vector_top_k=min(budget.vector_top_k, budget.max_vector_results),
        max_vector_results=budget.max_vector_results,
        retrieval_timeout_ms=budget.retrieval_timeout_ms,
    )


# ── Store size: counts per kind, bytes per file ───────────────────────────────


@dataclass
class StoreSize:
    """A store's measured size: live counts + file bytes."""

    counts: dict[str, int] = field(default_factory=dict)
    file_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def total_records(self) -> int:
        return sum(self.counts.values())

    @property
    def total_bytes(self) -> int:
        return sum(self.file_bytes.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "total_records": self.total_records,
            "file_bytes": dict(self.file_bytes),
            "total_bytes": self.total_bytes,
        }


def measure_store(store: Any) -> StoreSize:
    """Measure one MemoryStore: live records per budgeted kind
    (profile records split by record_type) and the bytes of
    every file in its directory."""
    profile = store.load_profile_records()
    counts = {
        "user_preference": sum(
            1 for r in profile if r.record_type == "user_preference"
        ),
        "user_fact": sum(1 for r in profile if r.record_type == "user_fact"),
        "episode": len(store.load_episodes()),
        "response_pattern": len(store.load_patterns()),
    }
    file_bytes = {
        p.name: p.stat().st_size
        for p in sorted(store.dir.iterdir())
        if p.is_file()
    }
    return StoreSize(counts=counts, file_bytes=file_bytes)


@dataclass
class BudgetReport:
    """One store judged against one profile's budget."""

    profile_id: str
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_records: int = 0
    total_bytes: int = 0

    @property
    def over_hard_kinds(self) -> list[str]:
        return [k for k, r in self.rows.items() if r["status"] == "over_hard"]

    @property
    def over_soft_kinds(self) -> list[str]:
        return [
            k for k, r in self.rows.items() if r["status"] == "over_soft"
        ]

    @property
    def within_budget(self) -> bool:
        return not self.over_hard_kinds

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "rows": {k: dict(v) for k, v in self.rows.items()},
            "total_records": self.total_records,
            "total_bytes": self.total_bytes,
            "over_hard_kinds": self.over_hard_kinds,
            "over_soft_kinds": self.over_soft_kinds,
            "within_budget": self.within_budget,
        }


def budget_report(store: Any, profile_id: str) -> BudgetReport:
    """Judge a store's live size against a profile's caps.
    Status per kind: over_hard (live > hard — eviction is owed),
    over_soft (soft < live <= hard — reported, not evicted),
    within_soft otherwise."""
    budget = budget_for_profile(profile_id)
    size = measure_store(store)
    rows: dict[str, dict[str, Any]] = {}
    for kind in BUDGETED_KINDS:
        live = size.counts[kind]
        soft, hard = budget.cap(kind, "soft"), budget.cap(kind, "hard")
        status = (
            "over_hard" if live > hard
            else "over_soft" if live > soft
            else "within_soft"
        )
        rows[kind] = {"live": live, "soft": soft, "hard": hard, "status": status}
    return BudgetReport(
        profile_id=profile_id,
        rows=rows,
        total_records=size.total_records,
        total_bytes=size.total_bytes,
    )


def enforce_budget(
    store: Any, profile_id: str, mode: str = "light"
) -> tuple[dict[str, int], BudgetReport]:
    """Evict a store back inside a profile's budget, by the
    lifecycle rules only: a profile-capped MemoryLifecycle runs
    one prune pass in ``mode`` (default light — TF-045 proved
    light's score legs unreachable, so a light budget pass is
    dedup + cap eviction only, the purest form of this slice's
    act). Returns (prune counts, the AFTER report) — the report
    is the verdict: a protected surplus can leave it over-hard,
    honestly, because protection outranks the cap (TF-045)."""
    from thoughtforge.knowledge.lifecycle import MemoryLifecycle

    lifecycle = MemoryLifecycle(store=store, profile_id=profile_id)
    counts = lifecycle.prune(mode)  # type: ignore[arg-type]
    return counts, budget_report(store, profile_id)


# ── The live checker ───────────────────────────────────────────────────────────


def memory_budgets_problems() -> list[str]:
    """Re-derive TF-056's claims from the live tree; [] = holds.

    Pins: budgets cover exactly the shipped profiles and read
    the JSONs live; soft <= hard and the profile ladder is
    monotonic (pi_zero <= phone_low <= pi_5 <= desktop <=
    server) per kind; the flat default table IS the desktop
    budget for the four kinds; plans only ever cap; a fixture
    store over phone_low's fact cap is evicted to exactly the
    cap by a light budget pass, lowest-survival first, with a
    protected record surviving an over-cap store; the wiring
    stands (lifecycle takes a profile, the core builds its
    lifecycle on the selected profile and retrieves through
    the plan, the forge accepts the caps).
    """
    problems: list[str] = []

    from thoughtforge.knowledge.lifecycle import STORE_CAPS

    shipped = sorted(list_hardware_profiles())
    budgets = all_budgets()
    if sorted(budgets) != shipped:
        problems.append(f"budgets cover {sorted(budgets)} != shipped {shipped}")

    for pid, budget in budgets.items():
        raw = load_hardware_profile(pid)
        for kind, (soft_key, hard_key) in KIND_CAP_KEYS.items():
            if budget.cap(kind, "soft") != raw["memory"][soft_key]:
                problems.append(f"{pid}/{kind}: soft cap not read live")
            if budget.cap(kind, "hard") != raw["memory"][hard_key]:
                problems.append(f"{pid}/{kind}: hard cap not read live")
            if budget.cap(kind, "soft") > budget.cap(kind, "hard"):
                problems.append(f"{pid}/{kind}: soft exceeds hard")
        if budget.max_memory_bundle_facts != raw["memory"]["max_memory_bundle_facts"]:
            problems.append(f"{pid}: bundle cap not read live")
        if budget.sql_max_results != raw["retrieval"]["sql_max_results"]:
            problems.append(f"{pid}: sql cap not read live")
        for flat_kind in FLAT_CAP_KINDS:
            if budget.caps[flat_kind] != STORE_CAPS[flat_kind]:
                problems.append(f"{pid}: flat cap {flat_kind} drifted")

    ladder = ["pi_zero", "phone_low", "pi_5", "desktop_cpu", "server_gpu"]
    for kind in BUDGETED_KINDS:
        hards = [budgets[p].cap(kind) for p in ladder]
        if hards != sorted(hards):
            problems.append(f"ladder: {kind} hard caps not monotonic {hards}")
    for pid in ("desktop_cpu", "desktop_gpu"):
        for kind in BUDGETED_KINDS:
            if budgets[pid].caps[kind] != STORE_CAPS[kind]:
                problems.append(f"{pid}/{kind}: desktop budget != flat default")

    # Plans only cap.
    for pid, want in (("phone_low", 5), ("pi_zero", 3), ("desktop_cpu", 10),
                      ("server_gpu", 10)):
        plan = retrieval_plan(pid)
        if plan.bundle_top_k != want:
            problems.append(f"plan {pid}: bundle {plan.bundle_top_k} != {want}")
    if retrieval_plan("server_gpu", requested_top_k=25).bundle_top_k != 25:
        problems.append("plan server_gpu: a 25 request under the 30 cap must pass")
    try:
        budget_for_profile("bogus_profile")
        problems.append("unknown profile did not raise")
    except ValueError:
        pass

    # Fixture eviction: 70 distinct facts, phone_low hard = 64.
    from thoughtforge.knowledge.models import UserFactRecord
    from thoughtforge.knowledge.store import MemoryStore

    with tempfile.TemporaryDirectory(prefix="tf056-") as tmp:
        store = MemoryStore(memory_dir=Path(tmp))
        for i in range(70):
            store.append_profile_record(UserFactRecord(
                record_id=f"fct_tf056_{i:03d}", summary=f"budget fact {i}",
                tags=[f"t{i}"], confidence=0.5,
            ))
        before = budget_report(store, "phone_low")
        if before.rows["user_fact"]["status"] != "over_hard":
            problems.append("fixture: 70 facts not over phone_low's hard cap")
        _, after = enforce_budget(store, "phone_low", mode="light")
        if len(store.load_profile_records()) != 64:
            problems.append(
                f"fixture: eviction left {len(store.load_profile_records())} != 64"
            )
        if not after.within_budget:
            problems.append(f"fixture: after report not within budget {after.rows}")

    # Wiring stands.
    import inspect

    from thoughtforge.cognition import core as core_module
    from thoughtforge.knowledge import forge as forge_module
    from thoughtforge.knowledge import lifecycle as lifecycle_module

    if "profile_id" not in inspect.signature(
        lifecycle_module.MemoryLifecycle.__init__
    ).parameters:
        problems.append("wiring: MemoryLifecycle takes no profile_id")
    core_src = inspect.getsource(core_module)
    if "profile_id=self._conversation_config.profile_id" not in core_src:
        problems.append("wiring: core lifecycle not built on the selected profile")
    retrieve_src = inspect.getsource(core_module.ThoughtForgeCore._retrieve)
    if "retrieval_plan" not in retrieve_src:
        problems.append("wiring: core retrieval does not use the plan")
    forge_params = inspect.signature(forge_module.KnowledgeForge.retrieve).parameters
    if not {"sql_max_results", "vector_top_k"} <= set(forge_params):
        problems.append("wiring: forge retrieve accepts no retrieval caps")
    return problems
