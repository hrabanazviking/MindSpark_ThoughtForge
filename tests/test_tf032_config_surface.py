"""
TF-032 — Cognition config surface: scaffolds/profiles selectable
per conversation via config (TF-009's live keys only).

The discipline lives in
``src/thoughtforge/cognition/config_surface.py``; the write-up is
``docs/architecture/COGNITION_CONFIG_SURFACE.md``.

Pinned here:

  * resolution defaults: an empty / missing config selects nothing
    and the standing token cap is the pre-TF-032 512,
  * live keys only: the one live cognition file key
    (``cognition.personality_core_path``) is carried through, while
    the shadowed/dead file keys TF-009 named — ``hardware.profile``,
    the *file* key ``cognition.scaffold_depth``, the user_config
    ``profile`` / ``chat_*`` keys — change nothing,
  * profiles resolve through the live profile loaders: every
    shipped profile's ``generation.max_response_tokens`` is the
    conversation's cap; a bogus profile raises ValueError naming
    the valid ids (never a silent pass),
  * strategy / depth / candidate-mode selections validate against
    the TF-022 / TF-021 registries and the depth table: unknown,
    duplicate, empty, and over-long selections all raise,
  * the frozen record's dict round-trip is exact,
  * apply_conversation_config replaces depth / candidate modes on
    a built scaffold and leaves a default scaffold untouched,
  * core wiring end-to-end: construction with a bogus profile
    raises; a configured knowledge-only core's turn scaffold
    carries the selected depth + modes; a selected strategy is
    exposed resolved (selection, not execution); the unified
    backend receives the profile's max_tokens on the plain AND
    streamed paths, and the standing 512 when none is selected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from thoughtforge.cognition.config_surface import (
    DEFAULT_MAX_TOKENS,
    VALID_DEPTHS,
    ConversationConfig,
    apply_conversation_config,
    resolve_conversation_config,
)
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.scaffold import ScaffoldBuilder
from thoughtforge.cognition.scaffold_registry import canonical_sketch
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.knowledge.models import MemoryActivationBundle

GOOD_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water. Wrenna the lantern-keeper checks the river level each "
    "morning and chalk-marks it on the mill door."
)


# ── Resolution: defaults and the live-keys-only rule ─────────────────────────


class TestDefaults:
    def test_empty_config_selects_nothing(self) -> None:
        cfg = resolve_conversation_config({})
        assert cfg.is_default
        assert cfg.profile_id is None
        assert cfg.generation_max_tokens() == DEFAULT_MAX_TOKENS == 512

    def test_none_config_selects_nothing(self) -> None:
        assert resolve_conversation_config(None).is_default

    def test_shipped_shape_selects_nothing(self) -> None:
        # The shape load_config() returns for the shipped
        # default.yaml: no "conversation" section exists in it.
        cfg = resolve_conversation_config(
            {"cognition": {"personality_core_path": "configs/personality_core.yaml",
                           "scaffold_depth": "medium"},
             "hardware": {"profile": "auto"}}
        )
        assert cfg.is_default


class TestLiveKeysOnly:
    def test_personality_core_path_carried_through(self) -> None:
        cfg = resolve_conversation_config(
            {"cognition": {"personality_core_path": "some/where.yaml"}}
        )
        assert cfg.personality_core_path == "some/where.yaml"
        assert cfg.is_default  # carrying the live key selects nothing else

    def test_shadowed_hardware_profile_key_ignored(self) -> None:
        cfg = resolve_conversation_config({"hardware": {"profile": "phone_low"}})
        assert cfg.profile_id is None
        assert cfg.generation_max_tokens() == 512

    def test_dead_file_scaffold_depth_key_ignored(self) -> None:
        # cognition.scaffold_depth is TF-009's DEAD file key; the
        # per-conversation key is conversation.depth.
        cfg = resolve_conversation_config({"cognition": {"scaffold_depth": "light"}})
        assert cfg.depth is None

    def test_dead_user_config_keys_ignored(self) -> None:
        cfg = resolve_conversation_config(
            {"profile": "phone_low", "chat_max_turns": 5,
             "chat_system_prompt": "x", "knowledge_db_built": True}
        )
        assert cfg.is_default

    def test_non_mapping_sections_raise(self) -> None:
        with pytest.raises(ValueError):
            resolve_conversation_config({"conversation": "phone_low"})


# ── Profiles ──────────────────────────────────────────────────────────────────


class TestProfiles:
    @pytest.mark.parametrize(
        "profile_id,tokens",
        [
            ("desktop_cpu", 400),
            ("desktop_gpu", 600),
            ("phone_low", 180),
            ("pi_5", 250),
            ("pi_zero", 120),
            ("server_gpu", 2000),
        ],
    )
    def test_profile_max_tokens_from_the_profile_json(
        self, profile_id: str, tokens: int
    ) -> None:
        cfg = resolve_conversation_config(
            {"conversation": {"profile_id": profile_id}}
        )
        assert cfg.profile_id == profile_id
        assert cfg.max_response_tokens == tokens
        assert cfg.generation_max_tokens() == tokens
        assert not cfg.is_default

    def test_bogus_profile_raises_and_names_valid_ids(self) -> None:
        with pytest.raises(ValueError) as exc:
            resolve_conversation_config(
                {"conversation": {"profile_id": "quantum_phone"}}
            )
        assert "quantum_phone" in str(exc.value)
        assert "phone_low" in str(exc.value)


# ── Strategy, depth, candidate modes ──────────────────────────────────────────


class TestSelections:
    @pytest.mark.parametrize(
        "strategy", ["chain_of_thought", "plan_then_answer", "self_check"]
    )
    def test_valid_strategies(self, strategy: str) -> None:
        cfg = resolve_conversation_config({"conversation": {"strategy": strategy}})
        assert cfg.strategy == strategy

    def test_bogus_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_conversation_config({"conversation": {"strategy": "wing_it"}})

    @pytest.mark.parametrize("depth", list(VALID_DEPTHS))
    def test_valid_depths(self, depth: str) -> None:
        cfg = resolve_conversation_config({"conversation": {"depth": depth}})
        assert cfg.depth == depth

    def test_bogus_depth_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_conversation_config({"conversation": {"depth": "abyssal"}})

    def test_candidate_modes_valid(self) -> None:
        cfg = resolve_conversation_config(
            {"conversation": {"candidate_modes": ["strict_spec", "practical"]}}
        )
        assert cfg.candidate_modes == ("strict_spec", "practical")

    def test_candidate_modes_single(self) -> None:
        cfg = resolve_conversation_config(
            {"conversation": {"candidate_modes": ["practical"]}}
        )
        assert cfg.candidate_modes == ("practical",)

    def test_candidate_modes_duplicate_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_conversation_config(
                {"conversation": {"candidate_modes": ["practical", "practical"]}}
            )

    def test_candidate_modes_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_conversation_config(
                {"conversation": {"candidate_modes": ["telepathic"]}}
            )

    def test_candidate_modes_too_many_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_conversation_config(
                {"conversation": {"candidate_modes": ["practical", "playful", "empathic"]}}
            )

    def test_candidate_modes_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_conversation_config({"conversation": {"candidate_modes": []}})

    def test_candidate_modes_bare_string_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_conversation_config(
                {"conversation": {"candidate_modes": "practical"}}
            )


# ── The record ────────────────────────────────────────────────────────────────


class TestRecord:
    def test_dict_round_trip_exact(self) -> None:
        cfg = resolve_conversation_config(
            {"cognition": {"personality_core_path": "p.yaml"},
             "conversation": {"profile_id": "phone_low", "strategy": "self_check",
                              "depth": "light", "candidate_modes": ["practical"]}}
        )
        assert ConversationConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()

    def test_default_dict_round_trip(self) -> None:
        cfg = ConversationConfig()
        assert ConversationConfig.from_dict(cfg.to_dict()) == cfg

    def test_frozen(self) -> None:
        cfg = ConversationConfig()
        with pytest.raises(Exception):
            cfg.depth = "light"  # type: ignore[misc]


# ── Applying to a scaffold ────────────────────────────────────────────────────


def _scaffold(intent: str = "factual_query"):
    return ScaffoldBuilder().build(canonical_sketch(intent), MemoryActivationBundle())


class TestApply:
    def test_selected_depth_and_modes_replace_derived(self) -> None:
        scaffold = _scaffold()
        cfg = ConversationConfig(depth="expert", candidate_modes=("practical",))
        apply_conversation_config(scaffold, cfg)
        assert scaffold.depth == "expert"
        assert scaffold.candidate_modes == ["practical"]

    def test_default_leaves_scaffold_untouched(self) -> None:
        scaffold = _scaffold()
        depth, modes = scaffold.depth, list(scaffold.candidate_modes)
        apply_conversation_config(scaffold, ConversationConfig())
        assert scaffold.depth == depth
        assert scaffold.candidate_modes == modes


# ── Core wiring ───────────────────────────────────────────────────────────────


def _core(config: dict | None = None, backend=None) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    kwargs: dict = dict(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    if config is not None:
        kwargs["config"] = config
    if backend is not None:
        kwargs["backend"] = backend
    return ThoughtForgeCore(**kwargs)


class _CapturingBackend(UnifiedBackend):
    def __init__(self) -> None:
        self.requests: list = []

    def generate(self, request):
        self.requests.append(request)
        return GenerationResponse(
            text=GOOD_TEXT, tokens_generated=len(GOOD_TEXT.split())
        )

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "stub-capturing"


class TestCoreWiring:
    def test_bogus_profile_fails_at_construction(self) -> None:
        with pytest.raises(ValueError):
            _core(config={"conversation": {"profile_id": "nope"}})

    def test_default_core_config_is_default(self) -> None:
        core = _core(config={"cognition": {}})
        assert core.conversation_config.is_default
        assert core.selected_strategy is None

    def test_turn_scaffold_carries_selections(self) -> None:
        core = _core(
            config={"cognition": {},
                    "conversation": {"depth": "light",
                                     "candidate_modes": ["practical"]}}
        )
        core.think("How long does the River Sarn ferry crossing take?")
        turn = core.last_turn_state
        assert turn is not None and turn.scaffold is not None
        assert turn.scaffold.depth == "light"
        assert turn.scaffold.candidate_modes == ["practical"]

    def test_selected_strategy_resolves(self) -> None:
        core = _core(
            config={"cognition": {},
                    "conversation": {"strategy": "plan_then_answer"}}
        )
        assert core.conversation_config.strategy == "plan_then_answer"
        strategy = core.selected_strategy
        assert strategy is not None and strategy.name == "plan_then_answer"

    def test_backend_gets_profile_max_tokens_plain(self) -> None:
        backend = _CapturingBackend()
        core = _core(
            config={"cognition": {},
                    "conversation": {"profile_id": "phone_low"}},
            backend=backend,
        )
        core.think("How long does the River Sarn ferry crossing take?")
        assert [r.max_tokens for r in backend.requests] == [180]

    def test_backend_gets_profile_max_tokens_streamed(self) -> None:
        backend = _CapturingBackend()
        core = _core(
            config={"cognition": {},
                    "conversation": {"profile_id": "phone_low"}},
            backend=backend,
        )
        for _ in core.think_stream("How long does the ferry crossing take?"):
            pass
        assert [r.max_tokens for r in backend.requests] == [180]

    def test_backend_gets_standing_512_without_profile(self) -> None:
        backend = _CapturingBackend()
        core = _core(config={"cognition": {}}, backend=backend)
        core.think("How long does the River Sarn ferry crossing take?")
        assert [r.max_tokens for r in backend.requests] == [512]
