"""
The cognition layer's public API, frozen at v1 (TF-034).

Parts II's slices TF-019–TF-033 each grew the cognition layer's
surface one honest piece at a time — the core's turn drivers, the
streaming contract, interruption/resume, the will ledger, the
persona expectation, the config surface, the latency budgets —
but nothing ever *named* that surface as one interface, stated
its version, or said what a consumer may rely on. The package
``__init__`` still exported only the six names it was born with.
Part IX's bindings (Rúnmál, Samtal, Hermes, Ember) bind HERE, so
this slice freezes the surface before they do:

  * ``API_VERSION`` — semantic version of this freeze ("1.0.0").
  * The frozen tables below — the core's constructor, methods
    (with parameter names, in order), and properties; the
    package-level exports; and each cognition module's frozen
    public names. These tables are the freeze: they are written
    by hand from the measured live surface, never derived from
    it, so a silent change to the real surface cannot rewrite
    the contract to match itself.
  * ``api_manifest()`` — the freeze as plain data (JSON-safe),
    for consumers and docs to carry without importing the layer.
  * ``verify_api_surface()`` — the drift check: compares a
    manifest (the frozen one by default, or one a consumer
    carries) against the LIVE objects and returns every problem
    it finds — a missing name, a renamed parameter, a property
    that stopped being a property. An empty list means the live
    surface is exactly the frozen one.

Versioning rules (the freeze's own law, binding on later slices):

  * Removing or renaming a frozen name, removing or renaming a
    frozen parameter, or changing a frozen return contract is a
    BREAKING change — it needs a major version bump and a note
    in the freeze doc.
  * Adding a name, or a parameter with a default, is MINOR.
  * Behaviour fixes that keep names, parameters, and return
    contracts are PATCH.
  * Anything underscore-prefixed is NOT public and may change
    without a bump: ``ThoughtForgeCore._run_turn`` and kin are
    the pipeline's private machinery, however visible they are
    in the file.
  * A consumer on an incompatible major version must get a
    truthful refusal, never silent corruption — TF-162 owns
    enforcing that at the service shape; this module gives it
    the version and the manifest to check against.

TF-034 also settles CORE_LOOP §8-F1, the docket item this slice
owns: ``num_drafts`` was a documented-but-dead parameter of
``think()`` / ``think_stream()`` ("Override draft count (default
from hardware profile)" — no code read it, and no hardware
profile was ever consulted). Ruling: WIRE it, honestly, rather
than remove it — pre-freeze callers already pass it (the phase-6
adversarial tests pass 1 and 5), and a draft cap is a real need
on small hardware. Its frozen v1 semantics, implemented in
``core.py`` with this slice:

  * ``num_drafts`` caps the number of candidate drafts generated
    in the turn. The local engine generates one draft per
    candidate mode (at most 2 by construction), so the cap
    selects the first ``num_drafts`` modes. The unified backend
    generates exactly one candidate by construction, so the cap
    cannot raise it — and for ``num_drafts >= 1`` does not lower
    it either.
  * ``None`` (the default) means no cap beyond construction.
  * A value below 1 raises ``ValueError`` — a turn that is asked
    for zero drafts is a caller error, not a knowledge-only turn
    (knowledge-only is what no model already means).

The human-readable write-up is
``docs/architecture/CORE_API_FREEZE_V1.md``.
"""

from __future__ import annotations

import copy
import importlib
import inspect
from typing import Any

# ── The version ──────────────────────────────────────────────────────────────

API_NAME = "thoughtforge.cognition"
API_VERSION = "1.0.0"
API_VERSION_TUPLE: tuple[int, int, int] = (1, 0, 0)

# ── The frozen core surface ──────────────────────────────────────────────────
#
# Parameter names in declaration order, ``self`` excluded. These
# are pinned by hand from the live signatures at freeze time;
# verify_api_surface() re-reads the live signatures and reports
# any drift from these tables.

CORE_CLASS = "thoughtforge.cognition.core.ThoughtForgeCore"

CORE_CONSTRUCTOR_PARAMS: tuple[str, ...] = (
    "config",
    "model_path",
    "memory_dir",
    "db_path",
    "backend",
)

CORE_METHODS: dict[str, tuple[str, ...]] = {
    "think": ("user_text", "retrieval_path", "num_drafts", "history", "interrupt"),
    "think_stream": (
        "user_text", "retrieval_path", "num_drafts", "history", "interrupt",
    ),
    "resume_turn": ("interrupt",),
    "last_turn_latency": ("total_ms",),
    "load_model": ("model_path",),
    "unload_model": (),
}

CORE_PROPERTIES: tuple[str, ...] = (
    "last_turn_state",
    "conversation_config",
    "selected_strategy",
    "latency_budget",
    "persona_expectation",
    "will_ledger",
)

# The record types a consumer receives across the surface. They
# live in thoughtforge.knowledge.models (the shared record home);
# the freeze names them so a consumer knows what think() hands
# back and what last_turn_state carries.
CORE_RECORD_TYPES: dict[str, str] = {
    "think_return": "thoughtforge.knowledge.models.FinalResponseRecord",
    "resume_turn_return": "thoughtforge.knowledge.models.FinalResponseRecord",
    "last_turn_state_type": "thoughtforge.knowledge.models.RuntimeTurnState",
    "think_stream_event": "thoughtforge.cognition.streaming.StreamEvent",
}

# ── The frozen package exports ───────────────────────────────────────────────
#
# What ``from thoughtforge.cognition import X`` guarantees in v1:
# the six original names plus the surface TF-019–TF-033 added.
# The package __init__ re-exports exactly these and its __all__
# must match this tuple (verify checks both directions).

PACKAGE_EXPORTS: tuple[str, ...] = (
    # The original six.
    "ChatHistory",
    "ChatMessage",
    "InputRouter",
    "PromptBuilder",
    "ScaffoldBuilder",
    "ThoughtForgeCore",
    # The freeze machinery itself.
    "API_VERSION",
    "api_manifest",
    "verify_api_surface",
    # TF-026 streaming + TF-027 interruption.
    "InterruptToken",
    "StreamEvent",
    "TurnInterrupted",
    # TF-029 will, TF-030 persona, TF-032 config, TF-033 latency.
    "ConversationConfig",
    "GoalLedger",
    "LatencyBudget",
    "LatencyVerdict",
    "PersonaExpectation",
    "StandingGoal",
)

# Where each package export lives (module path -> name), for the
# verifier and for consumers resolving a name to its home.
_EXPORT_HOMES: dict[str, str] = {
    "ChatHistory": "thoughtforge.cognition.chat_history",
    "ChatMessage": "thoughtforge.cognition.chat_history",
    "InputRouter": "thoughtforge.cognition.router",
    "PromptBuilder": "thoughtforge.cognition.prompt_builder",
    "ScaffoldBuilder": "thoughtforge.cognition.scaffold",
    "ThoughtForgeCore": "thoughtforge.cognition.core",
    "API_VERSION": "thoughtforge.cognition.api_surface",
    "api_manifest": "thoughtforge.cognition.api_surface",
    "verify_api_surface": "thoughtforge.cognition.api_surface",
    "InterruptToken": "thoughtforge.cognition.interruption",
    "StreamEvent": "thoughtforge.cognition.streaming",
    "TurnInterrupted": "thoughtforge.cognition.interruption",
    "ConversationConfig": "thoughtforge.cognition.config_surface",
    "GoalLedger": "thoughtforge.cognition.will",
    "LatencyBudget": "thoughtforge.cognition.latency_budgets",
    "LatencyVerdict": "thoughtforge.cognition.latency_budgets",
    "PersonaExpectation": "thoughtforge.cognition.persona",
    "StandingGoal": "thoughtforge.cognition.will",
}

# ── The frozen per-module surfaces ───────────────────────────────────────────
#
# The public names each cognition module guarantees in v1. A
# module may expose more (helpers other slices consume inside
# the layer); these are the names a Part IX consumer may rely
# on. Modules whose whole content is a fixture/scorer for the
# campaign's own harness (coherence) are frozen as such — their
# consumers are Part VII's suites, not Part IX's bindings.

MODULE_SURFACES: dict[str, tuple[str, ...]] = {
    "thoughtforge.cognition.chat_history": ("ChatHistory", "ChatMessage"),
    "thoughtforge.cognition.coherence": (
        "ANCHORS", "COHERENCE_SCRIPT", "DriftReport", "score_conversation",
    ),
    "thoughtforge.cognition.config_surface": (
        "ConversationConfig", "DEFAULT_MAX_TOKENS", "VALID_DEPTHS",
        "resolve_conversation_config",
    ),
    "thoughtforge.cognition.core": ("ThoughtForgeCore",),
    "thoughtforge.cognition.error_honesty": (
        "GenerationFailure", "generation_failures", "turn_failures",
    ),
    "thoughtforge.cognition.history_discipline": (
        "HistoryBudget", "estimate_tokens", "select_window",
    ),
    "thoughtforge.cognition.interruption": ("InterruptToken", "TurnInterrupted"),
    "thoughtforge.cognition.latency_budgets": (
        "LatencyBudget", "LatencyVerdict", "StageVerdict",
        "all_budgets", "budget_for_profile", "check_turn_latency",
    ),
    "thoughtforge.cognition.persona": (
        "PersonaExpectation", "check_persona_response", "score_persona_turns",
    ),
    "thoughtforge.cognition.prompt_builder": ("PromptBuilder",),
    "thoughtforge.cognition.prompt_contract": (
        "PROMPT_SECTIONS", "check_prompt_order",
    ),
    "thoughtforge.cognition.router": ("InputRouter",),
    "thoughtforge.cognition.scaffold": ("ScaffoldBuilder",),
    "thoughtforge.cognition.scaffold_registry": (
        "candidate_mode_registry", "scaffold_registry", "scaffold_shape",
    ),
    "thoughtforge.cognition.scaffold_strategies": (
        "ScaffoldStrategy", "StrategyStep", "select_strategy", "strategy_registry",
    ),
    "thoughtforge.cognition.streaming": ("EVENT_KINDS", "StreamEvent"),
    "thoughtforge.cognition.turn_state": ("PHASE_ORDER",),
    "thoughtforge.cognition.will": ("GoalLedger", "StandingGoal"),
}


# ── The manifest ─────────────────────────────────────────────────────────────


def api_manifest() -> dict[str, Any]:
    """
    The v1 freeze as plain JSON-safe data, built from the frozen
    tables above (never from live inspection — the tables ARE
    the contract). Returns a fresh deep copy each call: mutating
    the result cannot rewrite the freeze.
    """
    return {
        "api": API_NAME,
        "version": API_VERSION,
        "core": {
            "class": CORE_CLASS,
            "constructor_params": list(CORE_CONSTRUCTOR_PARAMS),
            "methods": {name: list(params) for name, params in CORE_METHODS.items()},
            "properties": list(CORE_PROPERTIES),
            "record_types": dict(CORE_RECORD_TYPES),
        },
        "package_exports": list(PACKAGE_EXPORTS),
        "modules": {
            module: list(names) for module, names in MODULE_SURFACES.items()
        },
    }


# ── The drift check ──────────────────────────────────────────────────────────


def _params_of(callable_obj: Any) -> list[str]:
    params = list(inspect.signature(callable_obj).parameters)
    return params[1:] if params and params[0] == "self" else params


def verify_api_surface(
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    """
    Compare a manifest (the frozen one by default) against the
    LIVE cognition surface. Returns a list of problems — empty
    when the live surface is exactly what the manifest promises.

    Checks: the core class exists; its constructor and every
    frozen method exist, are callable, and carry exactly the
    frozen parameter names in order; every frozen property is
    still a property; every package export is importable from
    ``thoughtforge.cognition``, listed in its ``__all__``, and
    is the same object as in its home module; every frozen
    per-module name exists in its module. Extra live names are
    NOT problems (additions are minor-version business, judged
    by humans, not by this check).
    """
    manifest = manifest if manifest is not None else api_manifest()
    problems: list[str] = []

    # ── Core class, constructor, methods, properties ──
    core_spec = manifest.get("core") or {}
    class_path = core_spec.get("class", CORE_CLASS)
    module_path, _, class_name = class_path.rpartition(".")
    try:
        core_cls = getattr(importlib.import_module(module_path), class_name)
    except Exception as exc:  # missing class: everything else is moot
        return [f"core class {class_path} not importable: {exc}"]

    live_ctor = _params_of(core_cls.__init__)
    if live_ctor != list(core_spec.get("constructor_params") or []):
        problems.append(
            f"ThoughtForgeCore.__init__ params drifted: live {live_ctor}, "
            f"frozen {list(core_spec.get('constructor_params') or [])}"
        )
    for name, frozen_params in (core_spec.get("methods") or {}).items():
        member = getattr(core_cls, name, None)
        if member is None or not callable(member):
            problems.append(f"core method {name} missing or not callable")
            continue
        live_params = _params_of(member)
        if live_params != list(frozen_params):
            problems.append(
                f"core method {name} params drifted: live {live_params}, "
                f"frozen {list(frozen_params)}"
            )
    for name in core_spec.get("properties") or []:
        if not isinstance(getattr(core_cls, name, None), property):
            problems.append(f"core property {name} missing or not a property")

    # ── Package exports ──
    package = importlib.import_module("thoughtforge.cognition")
    package_all = list(getattr(package, "__all__", []))
    for name in manifest.get("package_exports") or []:
        if not hasattr(package, name):
            problems.append(f"package export {name} not importable")
            continue
        if name not in package_all:
            problems.append(f"package export {name} missing from cognition.__all__")
        home = _EXPORT_HOMES.get(name)
        if home is not None:
            home_obj = getattr(importlib.import_module(home), name, None)
            if home_obj is not None and getattr(package, name) is not home_obj:
                problems.append(
                    f"package export {name} is not the object from {home}"
                )
    for name in package_all:
        if name not in (manifest.get("package_exports") or []):
            problems.append(
                f"cognition.__all__ carries {name}, which the freeze does "
                "not list — an unrecorded addition (minor bump + freeze "
                "update, or remove it)"
            )

    # ── Per-module surfaces ──
    for module_name, names in (manifest.get("modules") or {}).items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            problems.append(f"module {module_name} not importable: {exc}")
            continue
        for name in names:
            if not hasattr(module, name):
                problems.append(f"{module_name}.{name} missing")

    return problems


__all__ = [
    "API_NAME",
    "API_VERSION",
    "API_VERSION_TUPLE",
    "CORE_CLASS",
    "CORE_CONSTRUCTOR_PARAMS",
    "CORE_METHODS",
    "CORE_PROPERTIES",
    "CORE_RECORD_TYPES",
    "MODULE_SURFACES",
    "PACKAGE_EXPORTS",
    "api_manifest",
    "verify_api_surface",
]
