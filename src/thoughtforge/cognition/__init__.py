"""
Cognition layer — scaffolds, intent routing, and the orchestration loop.

Deterministic YAML control objects steer any model toward grounded,
consistent, cite-verified responses.

TF-034: this package's public surface is frozen at API v1 — see
``thoughtforge.cognition.api_surface`` (``API_VERSION``,
``api_manifest()``, ``verify_api_surface()``) and
``docs/architecture/CORE_API_FREEZE_V1.md``. The names in
``__all__`` below are exactly the freeze's PACKAGE_EXPORTS; the
verifier fails if the two ever drift apart.
"""

from thoughtforge.cognition.api_surface import (
    API_VERSION,
    api_manifest,
    verify_api_surface,
)
from thoughtforge.cognition.chat_history import ChatHistory, ChatMessage
from thoughtforge.cognition.config_surface import ConversationConfig
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.interruption import InterruptToken, TurnInterrupted
from thoughtforge.cognition.latency_budgets import LatencyBudget, LatencyVerdict
from thoughtforge.cognition.persona import PersonaExpectation
from thoughtforge.cognition.prompt_builder import PromptBuilder
from thoughtforge.cognition.router import InputRouter
from thoughtforge.cognition.scaffold import ScaffoldBuilder
from thoughtforge.cognition.streaming import StreamEvent
from thoughtforge.cognition.will import GoalLedger, StandingGoal

__all__ = [
    "API_VERSION",
    "ChatHistory",
    "ChatMessage",
    "ConversationConfig",
    "GoalLedger",
    "InputRouter",
    "InterruptToken",
    "LatencyBudget",
    "LatencyVerdict",
    "PersonaExpectation",
    "PromptBuilder",
    "ScaffoldBuilder",
    "StandingGoal",
    "StreamEvent",
    "ThoughtForgeCore",
    "TurnInterrupted",
    "api_manifest",
    "verify_api_surface",
]
