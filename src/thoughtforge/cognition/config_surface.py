"""
ConversationConfig — the cognition config surface (TF-032).

Before this slice a conversation could not be configured at all:
``ThoughtForgeCore`` read exactly one key from its config dict
(``cognition.personality_core_path``, the one live cognition key in
TF-009's census), had no profile parameter (TF-012's D-PROF-3: flat
constants on every profile), and TF-022's strategies were a library
nothing could select. TF-009's brief for this slice was explicit —
**live keys only** — and this module honours it as a rule, not a
slogan:

  * The one live file key, ``cognition.personality_core_path``, is
    carried through unchanged (the core already loads the persona
    from it at construction; this surface names it in the merged
    view, it does not re-implement the load).
  * Everything selectable here is a NEW per-conversation key, in a
    ``conversation`` section of the config dict handed to one core
    instance (one core = one conversation, the campaign's standing
    shape: one will ledger, one persona expectation per instance).
    These keys existed in no shipped file at TF-009's census, so
    none of them is a shadowed or dead key being quietly revived.
  * The shadowed/dead file keys are NOT read — pinned by tests:
    ``hardware.profile``, ``cognition.scaffold_depth`` (the file
    key; the per-conversation key is ``conversation.depth``),
    and the user_config ``profile`` / ``chat_*`` keys change
    nothing here. Wiring them would repeat D-CFG-1's mirage.
  * Profiles are resolved through the live mechanism that already
    exists — ``load_hardware_profile`` / ``list_hardware_profiles``
    (the selector's own loaders) — never through a config file's
    ``profile`` string. An unknown profile id is a ``ValueError``
    naming the valid ids: a bogus profile never silently passes
    (the benchmark-side half of that defect stays TF-121/TF-128's).
  * The live ``user_config.yaml`` backend keys (TF-009's other ten)
    are NOT part of this surface: the backend factory already reads
    them live, and TF-065/TF-153 own their shape. This surface is
    cognition only.

What a selection DOES, honestly stated:

  * ``profile_id`` — the profile is loaded and held on the core;
    its ``generation.max_response_tokens`` replaces the unified
    backend's hard-coded 512 (plain and streamed paths alike).
    With no profile selected the 512 stands, byte-identical to
    before this slice. The local-engine path takes no token cap
    from anywhere today (TF-021 recorded it) and still doesn't;
    retrieval top_k, salvage passes, history budgets and mode
    temperatures stay flat — TF-033/TF-037/TF-098 own those, and
    D-PROF-3 stays open for them, narrowed not closed.
  * ``depth`` — replaces the scaffold's derived depth for every
    turn of the conversation (the prompt builder's length hint is
    keyed by depth, so this steers prompt shape directly).
  * ``candidate_modes`` — replaces the scaffold's intent-derived
    candidate modes for every turn: 1–2 modes from the TF-021
    registry, no duplicates (TF-021's no-duplicate rule).
  * ``strategy`` — a TF-022 strategy name, resolved against the
    strategy registry and held on the core for consumers to read
    (``core.conversation_config`` / ``core.selected_strategy``).
    Strategy *execution* is not wired by this slice — TF-022
    recorded that the core loop does not run strategies, and that
    has not changed; selection is the config surface's job, and
    claiming execution would be false.

Invalid selections raise ``ValueError`` at resolution (core
construction) — never a silent fallback to a default the caller
did not ask for. That is also this slice's verdict on its share of
D-DEAD-1: ``utils.validators.validate_config`` validates a
*user_config backend* dict (it requires the ``backend`` key), so it
is the wrong shape to guard a cognition config and is NOT wired
here; its wire-or-retire verdict belongs to TF-094/TF-065, where
the backend surface lives.

The human-readable write-up is
``docs/architecture/COGNITION_CONFIG_SURFACE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The scaffold depths the prompt builder has length hints for
# (prompt_builder._DEPTH_TO_LENGTH_HINT) — the only legal values.
VALID_DEPTHS: tuple[str, ...] = ("light", "medium", "expert")

# The core generates at most two candidate passes per turn
# (candidate_modes[:2] in the generation paths) — a selection
# longer than that would be a lie about what runs.
MAX_CANDIDATE_MODES = 2

# The unified backend's pre-TF-032 hard-coded token cap. It stands
# unchanged when no profile is selected.
DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True)
class ConversationConfig:
    """One conversation's resolved cognition selections.

    Every field is ``None`` when the conversation did not select
    it — the pre-TF-032 behaviour then stands exactly. Instances
    are frozen: a conversation's selections do not mutate mid-turn
    or mid-conversation.
    """

    personality_core_path: str | None = None
    profile_id: str | None = None
    max_response_tokens: int | None = None
    strategy: str | None = None
    depth: str | None = None
    candidate_modes: tuple[str, ...] | None = None

    @property
    def is_default(self) -> bool:
        """True when nothing was selected (the old behaviour)."""
        return (
            self.profile_id is None
            and self.strategy is None
            and self.depth is None
            and self.candidate_modes is None
        )

    def generation_max_tokens(self) -> int:
        """The unified-backend token cap for this conversation:
        the selected profile's, or the standing 512."""
        if self.max_response_tokens is not None:
            return self.max_response_tokens
        return DEFAULT_MAX_TOKENS

    def to_dict(self) -> dict[str, Any]:
        return {
            "personality_core_path": self.personality_core_path,
            "profile_id": self.profile_id,
            "max_response_tokens": self.max_response_tokens,
            "strategy": self.strategy,
            "depth": self.depth,
            "candidate_modes": (
                list(self.candidate_modes)
                if self.candidate_modes is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationConfig":
        modes = data.get("candidate_modes")
        return cls(
            personality_core_path=data.get("personality_core_path"),
            profile_id=data.get("profile_id"),
            max_response_tokens=data.get("max_response_tokens"),
            strategy=data.get("strategy"),
            depth=data.get("depth"),
            candidate_modes=tuple(modes) if modes is not None else None,
        )


def _valid_profile_ids() -> list[str]:
    from thoughtforge.utils.config import list_hardware_profiles

    return list_hardware_profiles()


def _load_profile(profile_id: str) -> dict[str, Any]:
    from thoughtforge.utils.config import load_hardware_profile

    return load_hardware_profile(profile_id)


def _candidate_mode_names() -> list[str]:
    # Lazy: scaffold_registry imports core (mode temperatures), and
    # core imports this module — a top-level import would cycle.
    from thoughtforge.cognition.scaffold_registry import (
        candidate_mode_registry,
    )

    return sorted(candidate_mode_registry())


def _strategy_names() -> list[str]:
    from thoughtforge.cognition.scaffold_strategies import (
        strategy_registry,
    )

    return sorted(strategy_registry())


def resolve_conversation_config(
    config: dict[str, Any] | None,
) -> ConversationConfig:
    """
    Resolve one conversation's selections from a core config dict.

    Reads exactly two places: ``cognition.personality_core_path``
    (TF-009's one live cognition file key, carried through) and the
    NEW ``conversation`` section (``profile_id``, ``strategy``,
    ``depth``, ``candidate_modes``). Nothing else in the dict is
    consulted — the shadowed/dead keys of TF-009's census stay
    unread by design.

    Raises:
        ValueError: an unknown profile id, strategy, depth, or
            candidate mode; a duplicate candidate mode; or a
            candidate-mode selection outside 1–2 entries. The
            message names the valid options.
    """
    config = config or {}
    cognition = config.get("cognition") or {}
    conversation = config.get("conversation") or {}
    if not isinstance(cognition, dict) or not isinstance(conversation, dict):
        raise ValueError("config 'cognition' and 'conversation' must be mappings")

    personality_path = cognition.get("personality_core_path") or None

    profile_id = conversation.get("profile_id") or None
    max_response_tokens: int | None = None
    if profile_id is not None:
        valid = _valid_profile_ids()
        if profile_id not in valid:
            raise ValueError(
                f"unknown hardware profile {profile_id!r}; "
                f"valid profiles: {', '.join(valid)}"
            )
        profile = _load_profile(profile_id)
        tokens = (profile.get("generation") or {}).get("max_response_tokens")
        if tokens is not None:
            max_response_tokens = int(tokens)

    strategy = conversation.get("strategy") or None
    if strategy is not None:
        valid_strategies = _strategy_names()
        if strategy not in valid_strategies:
            raise ValueError(
                f"unknown scaffold strategy {strategy!r}; "
                f"valid strategies: {', '.join(valid_strategies)}"
            )

    depth = conversation.get("depth") or None
    if depth is not None and depth not in VALID_DEPTHS:
        raise ValueError(
            f"unknown scaffold depth {depth!r}; "
            f"valid depths: {', '.join(VALID_DEPTHS)}"
        )

    modes_raw = conversation.get("candidate_modes")
    candidate_modes: tuple[str, ...] | None = None
    if modes_raw is not None:
        if isinstance(modes_raw, str) or not isinstance(modes_raw, (list, tuple)):
            raise ValueError("conversation.candidate_modes must be a list of mode names")
        modes = [str(m) for m in modes_raw]
        if not modes or len(modes) > MAX_CANDIDATE_MODES:
            raise ValueError(
                f"conversation.candidate_modes must name 1–{MAX_CANDIDATE_MODES} "
                f"modes (the core runs at most {MAX_CANDIDATE_MODES} candidate "
                f"passes per turn); got {len(modes)}"
            )
        if len(set(modes)) != len(modes):
            raise ValueError(
                "conversation.candidate_modes repeats a mode — TF-021's "
                "no-duplicate rule: a repeated mode buys a second sample, "
                "not a second scaffold"
            )
        valid_modes = _candidate_mode_names()
        unknown = [m for m in modes if m not in valid_modes]
        if unknown:
            raise ValueError(
                f"unknown candidate mode(s): {', '.join(unknown)}; "
                f"valid modes: {', '.join(valid_modes)}"
            )
        candidate_modes = tuple(modes)

    return ConversationConfig(
        personality_core_path=personality_path,
        profile_id=profile_id,
        max_response_tokens=max_response_tokens,
        strategy=strategy,
        depth=depth,
        candidate_modes=candidate_modes,
    )


def apply_conversation_config(scaffold: Any, config: ConversationConfig) -> Any:
    """
    Apply a conversation's scaffold selections to one turn's
    scaffold: depth and candidate modes, when selected, replace
    the builder's derived values. Unselected fields are untouched,
    so a default conversation's scaffold is byte-identical to the
    pre-TF-032 builder output. Returns the scaffold for chaining.
    """
    if config.depth is not None:
        scaffold.depth = config.depth
    if config.candidate_modes is not None:
        scaffold.candidate_modes = list(config.candidate_modes)
    return scaffold


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "MAX_CANDIDATE_MODES",
    "VALID_DEPTHS",
    "ConversationConfig",
    "apply_conversation_config",
    "resolve_conversation_config",
]
