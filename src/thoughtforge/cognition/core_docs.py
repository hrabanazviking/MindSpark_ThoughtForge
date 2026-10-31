"""
Core docs reconciliation, as data (TF-039).

TF-019 mapped the core loop from the code as it stood and filed
eight findings (CORE_LOOP §8, F1–F8) where the code and its own
story disagreed. TF-020–TF-038 then shipped the behaviour those
findings proposed owners for — but the architecture doc, the
``core.py`` module docstring, and the record-type comments were
never brought back into line, and the TF-034 API freeze gained
minor additions (TF-036's ``determinism_mode``, TF-038's
``last_turn_trace``) that no slice had yet recorded in one place.

This module is that reconciliation, kept as *verifiable data*
rather than prose alone:

  * ``RECONCILED_FINDINGS`` — every CORE_LOOP §8 finding with its
    status after Part II (closed / ruled / still open), the slice
    that owns the verdict, and the evidence. ``findings_problems``
    re-checks the machine-checkable evidence against the LIVE
    tree, so a future change that silently un-reconciles a
    finding (the second mode returning to brainstorming, the
    wrong-path retrieval gate being fixed without updating F2's
    status, the docstring drifting again) is caught by name.
  * ``POST_FREEZE_ADDITIONS`` — the minor additions to the
    TF-034 v1 freeze made by TF-035–TF-038, recorded here and in
    CORE_API_FREEZE_V1.md §7. The frozen tables and
    ``API_VERSION`` are deliberately NOT changed by this slice:
    additions are minor under the freeze's own rules, and the
    fold-in + version bump is the next freeze reconciliation
    (TF-192 verifies none is left unrecorded).
    ``additions_problems`` checks each recorded addition exists
    live, exactly as named.
  * ``EXAMPLES`` — the core-docs examples as runnable source.
    Each example is a small self-contained program (synthetic
    data only, temp dirs only) defining ``run() -> dict``;
    ``run_example`` executes one and returns its result. The
    TF-039 doc prints these sources verbatim and the tests run
    every one — an example that stops working fails the suite,
    which is what "examples runnable" means here.

The human-readable write-up is
``docs/architecture/CORE_DOCS_RECONCILED.md``; the reconciled
architecture doc itself is ``docs/architecture/CORE_LOOP.md``.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

# ── CORE_LOOP §8 findings, reconciled ────────────────────────────────────────
#
# status: "closed" (fixed, evidence in the tree), "ruled"
# (decided not-a-defect / contract ruled, behaviour stands),
# "open" (still a defect, owner named). TF-192 verifies no
# finding is left "open" without a reason at campaign end.

RECONCILED_FINDINGS: tuple[dict[str, Any], ...] = (
    {
        "finding": "F1",
        "subject": "num_drafts is a dead parameter",
        "status": "closed",
        "owner": "TF-034",
        "evidence": (
            "wired as a real draft cap in think()/think_stream() "
            "(first-N candidate modes; <1 raises); the false "
            "'default from hardware profile' docstring claim is gone"
        ),
    },
    {
        "finding": "F2",
        "subject": "retrieval gate checks the wrong (default) DB path",
        "status": "open",
        "owner": "TF-041",
        "evidence": (
            "still true at TF-039: _retrieve tests "
            "get_knowledge_db_path().exists(), not "
            "self._knowledge.db_path; Part III's first slice owns "
            "the fix — recorded open, not hidden"
        ),
    },
    {
        "finding": "F3",
        "subject": "turn state is built, timed, and thrown away",
        "status": "closed",
        "owner": "TF-025 + TF-038",
        "evidence": (
            "the state is retained on core.last_turn_state, "
            "salvage_ms is measured around compose, and the full "
            "trace exports via core.last_turn_trace() / "
            "cognition.traces"
        ),
    },
    {
        "finding": "F4",
        "subject": "the core module docstring's 9 steps are not the code's steps",
        "status": "closed",
        "owner": "TF-039",
        "evidence": (
            "core.py's module docstring now states the reconciled "
            "7-step loop (route / retrieve / scaffold / generate / "
            "fragment / compose / quality gate), names the three "
            "drivers of the one pipeline, and places prompt-"
            "building inside generation and enforcement inside "
            "compose, as the code does"
        ),
    },
    {
        "finding": "F5",
        "subject": "memory triggers are computed but not consumed",
        "status": "ruled",
        "owner": "TF-020",
        "evidence": (
            "ruled a contract, not a defect: the router's job is "
            "trigger extraction only; retrieval consumes "
            "sketch.raw_text by design and prompt cues come from "
            "the bundle (ROUTER_DECISION_TABLE §7)"
        ),
    },
    {
        "finding": "F6",
        "subject": "two fragment thresholds, two fragment systems",
        "status": "ruled",
        "owner": "TF-022",
        "evidence": (
            "ruled not-a-defect: core's 0.25 gates the FragmentScores "
            "composite for turn-record provenance; salvage's 0.54 "
            "gates a different (length+citation) formula for "
            "refine-worthiness; both constants cross-documented "
            "in code — do not unify them"
        ),
    },
    {
        "finding": "F7",
        "subject": "CognitionScaffold.prompt_text is never filled",
        "status": "closed",
        "owner": "TF-023",
        "evidence": (
            "the field was removed (a turn builds many prompts; "
            "the builders return strings and never mutate the "
            "scaffold); CognitionScaffold has no prompt_text field"
        ),
    },
    {
        "finding": "F8",
        "subject": "brainstorming generates the same mode twice",
        "status": "closed",
        "owner": "TF-021",
        "evidence": (
            "brainstorming's candidate modes are now "
            "[brainstorm_distinct, practical] and the registry "
            "carries a no-duplicate rule per intent"
        ),
    },
)


def findings_problems() -> list[str]:
    """
    Re-check the machine-checkable evidence in
    ``RECONCILED_FINDINGS`` against the live tree. Returns
    problems by finding id — ``[]`` when every reconciled status
    still matches the code. A status that has gone stale in
    either direction is a problem: a "closed" finding whose fix
    vanished, and equally F2's "open" once TF-041 lands its fix
    without flipping the record.
    """
    from thoughtforge.cognition import core as core_module
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.cognition.scaffold_registry import scaffold_registry
    from thoughtforge.knowledge.models import (
        CognitionScaffold,
        InputSketch,
        RuntimeTurnState,
    )
    from thoughtforge.refinement import salvage as salvage_module

    problems: list[str] = []
    by_id = {row["finding"]: row for row in RECONCILED_FINDINGS}
    if set(by_id) != {f"F{i}" for i in range(1, 9)}:
        problems.append("RECONCILED_FINDINGS must cover F1–F8 exactly")

    # F1 — num_drafts is a live parameter of both drivers.
    for driver in ("think", "think_stream"):
        params = inspect.signature(getattr(ThoughtForgeCore, driver)).parameters
        if "num_drafts" not in params:
            problems.append(f"F1: {driver} lost its num_drafts parameter")

    # F2 — still open: the gate still reads the default path.
    retrieve_src = inspect.getsource(ThoughtForgeCore._retrieve)
    if "get_knowledge_db_path" not in retrieve_src:
        problems.append(
            "F2: _retrieve no longer checks the default DB path — "
            "if TF-041 fixed the gate, flip F2's status with evidence"
        )

    # F3 — state retained, salvage measured, trace exportable.
    if not isinstance(
        getattr(ThoughtForgeCore, "last_turn_state", None), property
    ):
        problems.append("F3: last_turn_state is not a property on the core")
    if not callable(getattr(ThoughtForgeCore, "last_turn_trace", None)):
        problems.append("F3: last_turn_trace is missing from the core")
    if "salvage_ms" not in {
        f.name for f in dataclasses.fields(RuntimeTurnState)
    }:
        problems.append("F3: RuntimeTurnState lost its salvage_ms field")

    # F4 — the module docstring states the reconciled loop.
    doc = core_module.__doc__ or ""
    for step in ("ROUTE", "RETRIEVE", "SCAFFOLD", "GENERATE",
                 "FRAGMENT", "COMPOSE", "QUALITY GATE"):
        if step not in doc:
            problems.append(f"F4: core docstring no longer names step {step}")
    if "FinalResponseRecord assembly" in doc:
        problems.append("F4: the old 9-step docstring tail is back")

    # F5 — the sketch still carries the extracted triggers.
    if "memory_triggers" not in {
        f.name for f in dataclasses.fields(InputSketch)
    }:
        problems.append("F5: InputSketch lost its memory_triggers field")

    # F6 — both thresholds stand, at their ruled values.
    if core_module._MIN_FRAGMENT_SCORE != 0.25:
        problems.append("F6: core fragment threshold moved from 0.25")
    if salvage_module._MIN_FRAGMENT_SCORE != 0.54:
        problems.append("F6: salvage fragment threshold moved from 0.54")

    # F7 — prompt_text stays removed.
    if "prompt_text" in {
        f.name for f in dataclasses.fields(CognitionScaffold)
    }:
        problems.append("F7: CognitionScaffold.prompt_text is back")

    # F8 — no intent in the registry repeats a candidate mode.
    for intent, entry in scaffold_registry().items():
        modes = list(entry.candidate_modes)
        if len(modes) != len(set(modes)):
            problems.append(
                f"F8: intent {intent} repeats a candidate mode: {modes}"
            )

    return problems


# ── Post-freeze minor additions (TF-034 v1), recorded ────────────────────────
#
# kind: "core-method" / "core-property" (checked on
# ThoughtForgeCore), "module" (importable, with headline names),
# "record-field" (checked on the named record type).

POST_FREEZE_ADDITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "determinism_mode",
        "kind": "core-property",
        "added_by": "TF-036",
        "note": "the conversation's determinism selection (seed, or off)",
    },
    {
        "name": "last_turn_trace",
        "kind": "core-method",
        "added_by": "TF-038",
        "note": "the last turn's full cognition trace as detached data",
    },
    {
        "name": "thoughtforge.cognition.determinism",
        "kind": "module",
        "added_by": "TF-036",
        "names": ("DeterminismMode", "turn_fingerprint", "run_seeded_script"),
        "note": "determinism mode: seeded streams, fingerprints, replay",
    },
    {
        "name": "thoughtforge.cognition.concurrency",
        "kind": "module",
        "added_by": "TF-035",
        "names": ("CONCURRENCY_SURFACES", "isolation_problems",
                  "run_concurrent_turns"),
        "note": "the concurrency contract and its isolation checker",
    },
    {
        "name": "thoughtforge.cognition.small_model_floor",
        "kind": "module",
        "added_by": "TF-037",
        "names": ("FloorBudget", "check_floor_turn", "run_floor_conversation"),
        "note": "the small-model floor: budgets, checker, 20-turn runner",
    },
    {
        "name": "thoughtforge.cognition.traces",
        "kind": "module",
        "added_by": "TF-038",
        "names": ("TRACE_FORMAT", "export_turn_trace", "trace_problems",
                  "write_turn_trace"),
        "note": "the turn-trace format, exporter, and checker",
    },
    {
        "name": "ConversationConfig.max_refinement_passes",
        "kind": "record-field",
        "added_by": "TF-037",
        "note": (
            "the profile's refinement cap, resolved live (the "
            "D-PROF-4 fix); refinement_max_passes() reads it"
        ),
    },
)


def additions_problems() -> list[str]:
    """
    Check every recorded post-freeze addition exists live,
    exactly as named, and that the freeze itself is untouched
    (``verify_api_surface`` clean, version still 1.0.0 — the
    fold-in is a later reconciliation's act, not a silent one).
    """
    import importlib

    from thoughtforge.cognition.api_surface import (
        API_VERSION,
        verify_api_surface,
    )
    from thoughtforge.cognition.config_surface import ConversationConfig
    from thoughtforge.cognition.core import ThoughtForgeCore

    problems: list[str] = []
    if API_VERSION != "1.0.0":
        problems.append(
            f"freeze version moved to {API_VERSION} without a "
            "recorded fold-in (TF-039 records additions; it does "
            "not bump the freeze)"
        )
    problems.extend(
        f"freeze drift: {p}" for p in verify_api_surface()
    )
    for row in POST_FREEZE_ADDITIONS:
        name, kind = row["name"], row["kind"]
        if kind == "core-property":
            if not isinstance(getattr(ThoughtForgeCore, name, None), property):
                problems.append(f"addition {name} is not a core property")
        elif kind == "core-method":
            if not callable(getattr(ThoughtForgeCore, name, None)):
                problems.append(f"addition {name} is not a core method")
        elif kind == "module":
            try:
                module = importlib.import_module(name)
            except Exception as exc:
                problems.append(f"addition module {name} not importable: {exc}")
                continue
            for headline in row.get("names", ()):
                if not hasattr(module, headline):
                    problems.append(f"addition module {name} lost {headline}")
        elif kind == "record-field":
            field = name.rpartition(".")[2]
            if field not in {
                f.name for f in dataclasses.fields(ConversationConfig)
            }:
                problems.append(f"addition {name} missing from ConversationConfig")
        else:
            problems.append(f"addition {name} has unknown kind {kind}")
    return problems


# ── Runnable examples ────────────────────────────────────────────────────────
#
# Each source is a complete small program defining run() -> dict.
# Synthetic data only (the River Sarn ferry is invented), temp
# dirs only, no model and no knowledge DB — they run anywhere
# the package imports. The TF-039 doc prints them verbatim.

_EXAMPLE_PREAMBLE = '''\
import tempfile
from pathlib import Path

from thoughtforge.cognition.core import ThoughtForgeCore

FERRY_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)


def make_core(**kwargs):
    """A persona-free core on a temp memory dir and no DB file."""
    tmp = Path(tempfile.mkdtemp())
    return ThoughtForgeCore(
        config={"cognition": {}},
        memory_dir=tmp / "memory",
        db_path=tmp / "none.db",
        **kwargs,
    )
'''

EXAMPLES: dict[str, str] = {
    "knowledge_only_turn": _EXAMPLE_PREAMBLE + '''

def run():
    core = make_core()
    final = core.think("What time does the ferry leave?")
    return {
        "mode": final.mode,
        "text": final.text,
        "tokens": final.token_count,
    }
''',
    "stub_backend_turn": _EXAMPLE_PREAMBLE + '''
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)


class StubBackend(UnifiedBackend):
    """One completed-text candidate per turn (the unified path)."""

    def generate(self, request):
        return GenerationResponse(
            text=FERRY_TEXT, tokens_generated=len(FERRY_TEXT.split())
        )

    def health_check(self):
        return True

    def backend_name(self):
        return "example-stub"


def run():
    core = make_core(backend=StubBackend())
    final = core.think("What time does the ferry leave?")
    state = core.last_turn_state
    return {
        "mode": final.mode,
        "candidates": len(state.candidates),
        "salvage_path": final.salvage_path,
        "text": final.text,
    }
''',
    "streaming_turn": _EXAMPLE_PREAMBLE + '''
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)


class StubBackend(UnifiedBackend):
    """One completed-text candidate per turn (the unified path)."""

    def generate(self, request):
        return GenerationResponse(
            text=FERRY_TEXT, tokens_generated=len(FERRY_TEXT.split())
        )

    def health_check(self):
        return True

    def backend_name(self):
        return "example-stub"


def run():
    core = make_core(backend=StubBackend())
    events = list(core.think_stream("What time does the ferry leave?"))
    return {
        "kinds": [event.kind for event in events],
        "final_mode": events[-1].final.mode,
    }
''',
    "trace_export": _EXAMPLE_PREAMBLE + '''
from thoughtforge.cognition.traces import trace_problems


def run():
    core = make_core()
    core.think("What time does the ferry leave?")
    trace = core.last_turn_trace()
    return {
        "format": trace["format"],
        "sections": list(trace),
        "problems": trace_problems(trace),
        "final_mode": trace["final"]["mode"],
    }
''',
    "seeded_replay": _EXAMPLE_PREAMBLE + '''

def run():
    config = {"cognition": {}, "determinism": {"seed": 42}}
    turn_ids = []
    for _ in range(2):
        tmp = Path(tempfile.mkdtemp())
        core = ThoughtForgeCore(
            config=config,
            memory_dir=tmp / "memory",
            db_path=tmp / "none.db",
        )
        core.think("What time does the ferry leave?")
        turn_ids.append(core.last_turn_state.turn_id)
    return {"same_turn_id": turn_ids[0] == turn_ids[1],
            "turn_id": turn_ids[0]}
''',
}


def example_names() -> tuple[str, ...]:
    """The runnable examples, in doc order."""
    return tuple(EXAMPLES)


def run_example(name: str) -> dict[str, Any]:
    """
    Execute one example's source in a fresh namespace and return
    its ``run()`` result. Raises ``KeyError`` for an unknown
    example. The source is this module's own constant — never
    caller-supplied text.
    """
    if name not in EXAMPLES:
        raise KeyError(f"unknown example {name!r}; valid: {sorted(EXAMPLES)}")
    namespace: dict[str, Any] = {}
    exec(compile(EXAMPLES[name], f"<tf039-example:{name}>", "exec"), namespace)
    result = namespace["run"]()
    if not isinstance(result, dict):
        raise TypeError(f"example {name} returned {type(result)}, not dict")
    return result


__all__ = [
    "EXAMPLES",
    "POST_FREEZE_ADDITIONS",
    "RECONCILED_FINDINGS",
    "additions_problems",
    "example_names",
    "findings_problems",
    "run_example",
]
