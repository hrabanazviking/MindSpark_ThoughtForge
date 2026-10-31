"""
The concurrency contract (TF-035): two conversations at once
without state bleed.

TF-032 already ruled the shape — one core = one conversation
(one will ledger, one persona expectation, one resolved config
per ``ThoughtForgeCore`` instance) — but nothing had ever run
two conversations at the same time and *proven* they stay
separate, and one real bleed was hiding in the shape itself.
This slice measured both, fixed the bleed, and pins the proof
in ``tests/test_tf035_concurrency.py``:

**The bleed (found by this slice's probe, fixed in core.py).**
Two threads calling ``think()`` on ONE core raced on
``_last_turn_state``: the pipeline body is per-call, but
``think()`` reads the retained state back after the turn, so
both callers received whichever turn assigned it last — in the
probe, 3 of 3 trials handed both callers the *same* final text,
one of them the other conversation's answer. The fix is a
per-core turn lock (``ThoughtForgeCore._turn_lock``): whole
turns on one instance serialise — ``think()``, ``think_stream()``
(lock held from first event until the stream is exhausted or
closed), and ``resume_turn()`` all take it, and ``think()``
reads back its own thread's turn (a thread-local set by the
pipeline) rather than the shared retained state, so a queued
turn starting the instant the lock frees cannot cross the
read-back either. After the fix the same probe gives each
caller its own answer, 3 of 3. One core
still holds only ONE conversation's will / persona /
last-turn state — the lock makes sharing a core *safe*
(queued, correct), never *parallel*. Parallel conversations
use parallel cores; that is the contract TF-166 consumes at
service scale.

**Two cores in parallel: isolated, measured.** With a
barrier-synchronised backend forcing both generations to be
in flight at the same instant, two cores showed no bleed in
anything the probe or the tests could reach: distinct turn
ids, distinct retained turn states, each caller's history and
scaffold carrying only its own conversation, one core's
standing goal steering its own next turn and absent from the
other's, per-conversation config (depth / candidate modes)
and persona each holding on their own core. The surfaces and
their owners are the table below (``CONCURRENCY_SURFACES``),
stated as data so TF-055 (persona stores) and TF-166 (service
scale) inherit rules, not folklore:

  * **conversation** — held on the core instance, private by
    construction: the resolved config (TF-032, frozen), the
    will ledger (TF-029), the persona expectation (TF-030,
    a construction-time snapshot), the retained turn state
    (TF-025), and the generation source (the core's own engine
    or unified backend instance). The pipeline helpers the core
    holds (router, scaffold builder, prompt builder, salvage,
    enforcement) are per-core instances whose methods keep all
    state in locals — no per-turn field exists on them to bleed.
  * **caller** — the chat history: the core never owns or
    stores it; the caller hands one ``ChatHistory`` per
    conversation in per turn. Sharing one history object
    between conversations is a caller error the checker below
    names.
  * **shared-read** — the knowledge database and the memory
    directory: during a turn the core loop only *reads* them
    (TF-019's verdict: the write-back leg is invoked by
    nothing; retrieval opens a fresh SQLite connection per
    call in ``KnowledgeForge``). Two cores may therefore share
    one ``db_path``, and may share a ``memory_dir`` for reads
    (same persona, same stored thread state) — write-side
    separation of persona stores is TF-055's subject, not
    this slice's claim.
  * **shared-aggregate** — the perf tracker and logging:
    ``get_perf_tracker()`` is a thread-safe singleton by
    design, so both conversations' timings land in ONE
    aggregate. That is accounting, not conversation state —
    but it means perf numbers are never per-conversation
    evidence, and no test may read one conversation's timing
    out of the shared tracker.
  * **serialised** — turns on a single shared core instance,
    under the turn lock described above.

**The checker.** ``isolation_problems(core_a, core_b, ...)``
reads two cores' retained state (plus their histories when
handed in) and returns every bleed it can name: same core
instance on both sides, a missing completed turn, shared or
colliding turn state / turn ids / scaffolds / finals, a shared
will ledger or persona expectation object, a standing goal
from one conversation found steering the other's scaffold, or
one history object (or its messages) serving both. An empty
list means the retained state is exactly as separate as the
contract promises. ``run_concurrent_turns`` runs one turn per
core on two threads and returns both finals, any thread
errors, and the checker's verdict.

Proof grade: FIXTURE — stub backends and the synthetic
TF-017 world on this VM. Two *live models* generating
concurrently (and what that does to latency budgets) is
TF-081's measurement, not this slice's claim.

The human-readable write-up is
``docs/architecture/CONCURRENCY.md``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# ── Owners ────────────────────────────────────────────────────────────────────

OWNER_CONVERSATION = "conversation"
OWNER_CALLER = "caller"
OWNER_SHARED_READ = "shared-read"
OWNER_SHARED_AGGREGATE = "shared-aggregate"
OWNER_SERIALISED = "serialised"

VALID_OWNERS: frozenset[str] = frozenset(
    {
        OWNER_CONVERSATION,
        OWNER_CALLER,
        OWNER_SHARED_READ,
        OWNER_SHARED_AGGREGATE,
        OWNER_SERIALISED,
    }
)


# ── The surface table ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SurfaceRule:
    """One state surface, its owner, and the rule that keeps it
    from bleeding between conversations."""

    surface: str
    owner: str
    rule: str


CONCURRENCY_SURFACES: tuple[SurfaceRule, ...] = (
    SurfaceRule(
        "conversation_config",
        OWNER_CONVERSATION,
        "Resolved once at core construction and frozen (TF-032); "
        "a second conversation's selections live on its own core "
        "and cannot reach this one's scaffolds.",
    ),
    SurfaceRule(
        "will_ledger",
        OWNER_CONVERSATION,
        "One GoalLedger per core instance (TF-029); a standing "
        "goal steers only its own conversation's fact_block.",
    ),
    SurfaceRule(
        "persona_expectation",
        OWNER_CONVERSATION,
        "A construction-time snapshot held per core (TF-030); "
        "two cores built from different stores carry different "
        "personae, and neither snapshot mutates the other's.",
    ),
    SurfaceRule(
        "last_turn_state",
        OWNER_CONVERSATION,
        "The retained RuntimeTurnState is per core (TF-025); "
        "each core's holds its own conversation's latest turn, "
        "under its own turn_id.",
    ),
    SurfaceRule(
        "generation_source",
        OWNER_CONVERSATION,
        "The engine / unified backend instance belongs to the "
        "core that holds it; concurrent generation on separate "
        "instances shares no per-turn state.",
    ),
    SurfaceRule(
        "chat_history",
        OWNER_CALLER,
        "The core never owns history: the caller hands one "
        "ChatHistory per conversation in per turn. One history "
        "object serving two conversations is a caller error.",
    ),
    SurfaceRule(
        "knowledge_database",
        OWNER_SHARED_READ,
        "Retrieval is read-only during a turn and opens a fresh "
        "SQLite connection per call, so two cores may share one "
        "db_path; the core loop writes nothing to it (TF-019).",
    ),
    SurfaceRule(
        "memory_dir",
        OWNER_SHARED_READ,
        "During a turn the core reads the persona core and the "
        "stored thread state and writes nothing, so two cores "
        "may share a memory_dir for reads; write-side persona "
        "store separation is TF-055's subject.",
    ),
    SurfaceRule(
        "perf_tracker",
        OWNER_SHARED_AGGREGATE,
        "get_perf_tracker() is one thread-safe singleton: both "
        "conversations' timings land in a single aggregate. "
        "Accounting, not conversation state — never read as "
        "per-conversation evidence.",
    ),
    SurfaceRule(
        "same_core_turns",
        OWNER_SERIALISED,
        "One core = one conversation. Turns attempted on one "
        "shared instance serialise under the core's turn lock "
        "(TF-035): queued and correct, never parallel and "
        "never bleeding. Parallel conversations use parallel "
        "cores.",
    ),
)


def surface_rule(surface: str) -> SurfaceRule:
    """The rule for one named surface.

    Raises:
        ValueError: the surface is not in the contract table;
            the message names the valid surfaces.
    """
    for rule in CONCURRENCY_SURFACES:
        if rule.surface == surface:
            return rule
    valid = ", ".join(r.surface for r in CONCURRENCY_SURFACES)
    raise ValueError(f"unknown concurrency surface {surface!r}; valid: {valid}")


# ── The isolation checker ─────────────────────────────────────────────────────


def isolation_problems(
    core_a: Any,
    core_b: Any,
    history_a: Any | None = None,
    history_b: Any | None = None,
) -> list[str]:
    """
    Name every bleed visible in two conversations' retained
    state. Returns a list of problems — empty when the two
    conversations are exactly as separate as the contract
    promises. Pure inspection: nothing here runs a turn or
    mutates either core.

    Checks: distinct core instances; each side has a completed
    turn; the retained turn states, their turn ids, scaffolds
    and finals are not shared or colliding; the will ledgers
    and persona expectations are not the same object; neither
    conversation's active standing goal is found steering the
    other's scaffold (unless it is also the other's own goal);
    and, when histories are handed in, neither the history
    objects nor any of their message objects are shared.
    """
    problems: list[str] = []

    if core_a is core_b:
        problems.append(
            "both conversations are the same core instance — turns "
            "serialise under its turn lock and its will / persona / "
            "last-turn state are one conversation's; parallel "
            "conversations need one core each"
        )

    turn_a = getattr(core_a, "last_turn_state", None)
    turn_b = getattr(core_b, "last_turn_state", None)
    if turn_a is None:
        problems.append("conversation A has no completed turn to inspect")
    if turn_b is None:
        problems.append("conversation B has no completed turn to inspect")

    if turn_a is not None and turn_b is not None:
        if turn_a is turn_b:
            problems.append("both cores retain the SAME turn state object")
        if turn_a.turn_id == turn_b.turn_id:
            problems.append(
                f"turn_id collision: both conversations' last turns "
                f"are {turn_a.turn_id!r}"
            )
        if (
            turn_a.scaffold is not None
            and turn_a.scaffold is turn_b.scaffold
        ):
            problems.append("both turns share the same scaffold object")
        if (
            turn_a.final_response is not None
            and turn_a.final_response is turn_b.final_response
        ):
            problems.append("both turns share the same final response object")

    ledger_a = getattr(core_a, "will_ledger", None)
    ledger_b = getattr(core_b, "will_ledger", None)
    if ledger_a is not None and ledger_a is ledger_b:
        problems.append("both cores hold the SAME will ledger object")
    if turn_a is not None and turn_b is not None:
        # Will bleed, seen where it would do harm: one
        # conversation's active standing goal found steering the
        # OTHER conversation's scaffold fact_block.
        goals = []
        for ledger in (ledger_a, ledger_b):
            active = getattr(ledger, "active", None)
            goals.append(active.statement if active is not None else None)
        for idx, (turn, own_goal, other_goal) in enumerate(
            (
                (turn_a, goals[0], goals[1]),
                (turn_b, goals[1], goals[0]),
            )
        ):
            if not other_goal or other_goal == own_goal:
                continue
            scaffold = turn.scaffold
            fact_block = scaffold.fact_block if scaffold is not None else ""
            if other_goal in (fact_block or ""):
                side = "A" if idx == 0 else "B"
                problems.append(
                    f"will bleed: the other conversation's standing "
                    f"goal {other_goal!r} is steering conversation "
                    f"{side}'s scaffold"
                )

    persona_a = getattr(core_a, "persona_expectation", None)
    persona_b = getattr(core_b, "persona_expectation", None)
    if persona_a is not None and persona_a is persona_b:
        problems.append("both cores hold the SAME persona expectation object")

    if history_a is not None and history_b is not None:
        if history_a is history_b:
            problems.append(
                "both conversations were handed the SAME chat "
                "history object — history is caller-owned, one "
                "per conversation"
            )
        else:
            ids_a = {id(m) for m in history_a.messages}
            shared = [m for m in history_b.messages if id(m) in ids_a]
            if shared:
                problems.append(
                    "the two histories share message objects — a "
                    "message written in one conversation is present "
                    "in the other"
                )

    return problems


# ── The concurrent runner ─────────────────────────────────────────────────────


@dataclass
class ConcurrentTurnRun:
    """The outcome of one pair of concurrent turns: each side's
    final (None when its thread failed), each side's thread
    error text, and the isolation checker's problems."""

    finals: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def run_concurrent_turns(
    core_a: Any,
    text_a: str,
    core_b: Any,
    text_b: str,
    history_a: Any | None = None,
    history_b: Any | None = None,
) -> ConcurrentTurnRun:
    """
    Run one turn on each core, on two threads, at the same
    time — the fixture shape the isolation proof uses. Thread
    exceptions are captured into ``errors`` (never swallowed
    silently, never raised into the caller), and the finished
    pair is judged by ``isolation_problems``.
    """
    run = ConcurrentTurnRun()

    def _drive(label: str, core: Any, text: str, history: Any | None) -> None:
        try:
            run.finals[label] = core.think(text, history=history)
        except Exception as exc:  # a failed thread is data, not a crash
            run.errors[label] = f"{type(exc).__name__}: {exc}"

    threads = [
        threading.Thread(target=_drive, args=("a", core_a, text_a, history_a)),
        threading.Thread(target=_drive, args=("b", core_b, text_b, history_b)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    run.problems = isolation_problems(core_a, core_b, history_a, history_b)
    return run


__all__ = [
    "CONCURRENCY_SURFACES",
    "OWNER_CALLER",
    "OWNER_CONVERSATION",
    "OWNER_SERIALISED",
    "OWNER_SHARED_AGGREGATE",
    "OWNER_SHARED_READ",
    "VALID_OWNERS",
    "ConcurrentTurnRun",
    "SurfaceRule",
    "isolation_problems",
    "run_concurrent_turns",
    "surface_rule",
]
