"""
InputRouter — classifies raw user input into a structured InputSketch.

Heuristic-first classification: keyword/pattern matching with no model calls.
Intentionally lean — wrong classifications cost little; the scaffold corrects.

Intent taxonomy (keep small):
  technical_spec_request | technical_debugging | emotional_support |
  brainstorming | light_conversation | naming_or_branding |
  editing_refinement | planning_request | factual_query

Tone taxonomy:
  neutral | focused | frustrated | tired | curious | playful | urgent

Response modes:
  structured_technical | calm_supportive | concise_editorial |
  creative_brainstorm | light_natural_chat | factual_direct

Decision procedure (TF-020 — the full table, with a worked example for
every row, lives in docs/architecture/ROUTER_DECISION_TABLE.md):

  1. Text is stripped and lower-cased. Every signal pattern is matched
     against that lower-cased text, so patterns must be lower-case
     literals — an upper-case pattern can never match (five patterns
     of exactly that kind were found and fixed in TF-020: the api,
     how-do-i, i-don't-know, and i've-been-thinking signals were dead
     outright; \bTraceback\b was a dead duplicate of its own
     lower-case twin and was removed).
  2. Intent and tone are each scored by counting DISTINCT matching
     patterns (repeating one signal never raises a score). The highest
     count wins; ties break by the precedence order of the signal
     tables below (table order IS the precedence order). No intent
     signal at all → light_conversation; no tone signal → neutral.
  3. Response mode is the fixed _INTENT_TO_MODE lookup for the intent.
  4. Urgency = _URGENCY_BASELINE + one tone bonus + one intent bonus,
     capped at _URGENCY_CAP (see the bonus tables below).
  5. Retrieval path is set membership: _SQL_HEAVY_INTENTS → "sql",
     _VECTOR_HEAVY_INTENTS → "vector", everything else → "hybrid".
  6. Memory triggers are an extraction contract only (see
     _extract_memory_triggers): the router publishes them on the
     sketch; whether retrieval consumes them is TF-041's ruling
     (CORE_LOOP §8-F5).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from thoughtforge.knowledge.models import ActiveThreadStateRecord, InputSketch

if TYPE_CHECKING:
    pass

import logging

logger = logging.getLogger(__name__)


# ── Intent signal tables ───────────────────────────────────────────────────────

_INTENT_SIGNALS: list[tuple[str, list[str]]] = [
    ("technical_debugging", [
        r"\berror\b", r"\bexception\b", r"\bfails?\b", r"\bcrash(es|ing)?\b",
        r"\btraceback\b", r"\bbug\b", r"\bfix\b",
        r"\bnot work(ing)?\b", r"\bbroken\b", r"\bwrong output\b",
    ]),
    ("technical_spec_request", [
        r"\bspec\b", r"\bdesign\b", r"\barchitect(ure)?\b", r"\bschema\b",
        r"\bapi\b", r"\binterface\b", r"\bimplements?\b", r"\bmodule\b",
        r"\bclass\b", r"\bwrite.*code\b", r"\bbuild\b", r"\bcreate.*function\b",
        r"\brefactor\b", r"\bstructure\b",
    ]),
    ("editing_refinement", [
        r"\brewrite\b", r"\bedit\b", r"\bimprove\b", r"\bpolish\b",
        r"\bcleaner\b", r"\bcleaner version\b", r"\bshorter\b",
        r"\bmore concise\b", r"\bfix.*wording\b", r"\btighten\b",
    ]),
    ("brainstorming", [
        r"\bideas?\b", r"\bbrainstorm\b", r"\boptions?\b", r"\balternatives?\b",
        r"\bsuggests?\b", r"\bwhat could\b", r"\bwhat if\b", r"\bcreative\b",
        r"\bpossibilities\b",
    ]),
    ("naming_or_branding", [
        r"\bname\b.*\bfor\b", r"\bnames?\b", r"\bbranding\b", r"\bslogan\b",
        r"\btitle\b.*\bfor\b", r"\bwhat.*call\b",
    ]),
    ("planning_request", [
        r"\bplan\b", r"\broadmap\b", r"\bsteps?\b", r"\bphases?\b",
        r"\bhow (do|should|can) (i|we)\b", r"\bwhat.*order\b", r"\bschedule\b",
        r"\bprioritize\b",
    ]),
    ("emotional_support", [
        r"\btired\b", r"\bexhausted\b", r"\bfrustrated\b", r"\bstressed\b",
        r"\banxious\b", r"\boverwhelmed\b", r"\bdepressed\b", r"\bworried\b",
        r"\bscared\b", r"\bfeel(ing)?\b.*\b(bad|awful|terrible|lost)\b",
        r"\bi don'?t know (what|how)\b", r"\bnot ok(ay)?\b",
    ]),
    ("factual_query", [
        r"\bwhat is\b", r"\bwho is\b", r"\bwhen (did|was|is)\b",
        r"\bwhere (is|was|did)\b", r"\bhow (does|do|did)\b",
        r"\bdefine\b", r"\bexplain\b", r"\btell me about\b",
        r"\bwhat.*mean\b",
    ]),
]

_TONE_SIGNALS: list[tuple[str, list[str]]] = [
    ("frustrated", [
        r"\bugh\b", r"\bwhy (is|does|won'?t|can'?t)\b", r"\bstill\b.*\bbroken\b",
        r"\bdoesn'?t work\b", r"\bnot working\b", r"\bfed up\b",
        r"\bwasted\b", r"\bfailing\b.*\bagain\b",
    ]),
    ("tired", [
        r"\btired\b", r"\bexhausted\b", r"\bdrained\b", r"\bburnt?[ -]out\b",
        r"\bcan'?t keep\b", r"\bcan'?t focus\b",
    ]),
    ("urgent", [
        r"\basap\b", r"\burgent\b", r"\bquickly\b", r"\bfast\b",
        r"\bright now\b", r"\bdeadline\b", r"\btoday\b",
    ]),
    ("curious", [
        r"\bwonder\b", r"\bcurious\b", r"\binteresting\b", r"\bfascinating\b",
        r"\bi'?ve been thinking\b", r"\bwhat if\b",
    ]),
    ("playful", [
        r"\blol\b", r"\bhaha\b", r"\bfunny\b", r"\bjoke\b",
        r"\b:\)\b", r"\b;[\)\-]\b", r"\blmao\b", r"\bxd\b",
    ]),
    ("focused", [
        r"\bneed to\b", r"\bmust\b", r"\brequired?\b", r"\blet'?s (get|do|start)\b",
        r"\bfocus\b", r"\bspecifically\b", r"\bexactly\b",
    ]),
]

# ── Precedence (TF-020: explicit, was implicit in dict/max order) ─────────────
# When two intents (or tones) score the same count of distinct matching
# patterns, the one listed EARLIER in its signal table wins. These lists
# are derived from the tables so the precedence can never drift from them.

_INTENT_PRECEDENCE: list[str] = [intent for intent, _ in _INTENT_SIGNALS]
_TONE_PRECEDENCE: list[str] = [tone for tone, _ in _TONE_SIGNALS]

# ── Urgency table (TF-020: explicit, was inline conditionals) ─────────────────
# urgency = baseline + tone bonus + intent bonus, capped. The highest
# reachable value today is 0.85 (urgent tone + technical_debugging);
# the cap is defensive, not a reachable row.

_URGENCY_BASELINE = 0.3
_URGENCY_CAP = 1.0
_TONE_URGENCY_BONUS: dict[str, float] = {
    "urgent": 0.4,
    "frustrated": 0.2,
    "tired": 0.1,
}
_INTENT_URGENCY_BONUS: dict[str, float] = {
    "technical_debugging": 0.15,
    "emotional_support": 0.1,
}

# ── Mode derivation table ──────────────────────────────────────────────────────

_INTENT_TO_MODE: dict[str, str] = {
    "technical_spec_request": "structured_technical",
    "technical_debugging":    "structured_technical",
    "editing_refinement":     "concise_editorial",
    "brainstorming":          "creative_brainstorm",
    "naming_or_branding":     "creative_brainstorm",
    "planning_request":       "structured_technical",
    "emotional_support":      "calm_supportive",
    "factual_query":          "factual_direct",
    "light_conversation":     "light_natural_chat",
}

# ── Retrieval path heuristic ───────────────────────────────────────────────────

_SQL_HEAVY_INTENTS = {"factual_query", "technical_spec_request", "planning_request"}
_VECTOR_HEAVY_INTENTS = {"emotional_support", "brainstorming", "light_conversation"}


class InputRouter:
    """
    Classifies raw user text into a structured InputSketch.
    Uses keyword/pattern heuristics — no model call required.
    Fast and deterministic.
    """

    def route(
        self,
        raw_text: str,
        thread_state: ActiveThreadStateRecord | None = None,
    ) -> InputSketch:
        """
        Classify raw_text into an InputSketch.

        Args:
            raw_text:     The user's message.
            thread_state: Current active thread state for context carry-forward.

        Returns:
            InputSketch with intent, topic, tone_in, response_mode, memory_triggers,
            urgency, and retrieval_path populated.
        """
        normalized = raw_text.strip().lower()

        intent = self._classify_intent(normalized)
        tone_in = self._classify_tone(normalized)
        response_mode = _INTENT_TO_MODE.get(intent, "light_natural_chat")
        memory_triggers = self._extract_memory_triggers(raw_text, thread_state)
        topic = self._extract_topic(raw_text, intent)
        urgency = self._estimate_urgency(normalized, tone_in, intent)
        retrieval_path = self._select_retrieval_path(intent, urgency)

        sketch = InputSketch(
            raw_text=raw_text,
            intent=intent,
            topic=topic,
            tone_in=tone_in,
            response_mode=response_mode,
            memory_triggers=memory_triggers,
            urgency=urgency,
            retrieval_path=retrieval_path,
        )

        logger.debug(
            "InputRouter: intent=%s tone=%s mode=%s urgency=%.2f path=%s",
            intent, tone_in, response_mode, urgency, retrieval_path,
        )
        return sketch

    # ── Intent classification ──────────────────────────────────────────────────

    def _classify_intent(self, normalized: str) -> str:
        scores: dict[str, int] = {}
        for intent, patterns in _INTENT_SIGNALS:
            count = sum(1 for p in patterns if re.search(p, normalized))
            if count:
                scores[intent] = count

        if not scores:
            return "light_conversation"

        # Highest distinct-pattern count wins; ties break by signal-table
        # precedence (technical_debugging first — so it beats
        # technical_spec_request on a tie, not only when it dominates).
        return max(
            scores,
            key=lambda k: (scores[k], -_INTENT_PRECEDENCE.index(k)),
        )

    # ── Tone classification ────────────────────────────────────────────────────

    def _classify_tone(self, normalized: str) -> str:
        scores: dict[str, int] = {}
        for tone, patterns in _TONE_SIGNALS:
            count = sum(1 for p in patterns if re.search(p, normalized))
            if count:
                scores[tone] = count

        if not scores:
            return "neutral"

        # Same rule as intent: highest count, ties by table precedence.
        return max(
            scores,
            key=lambda k: (scores[k], -_TONE_PRECEDENCE.index(k)),
        )

    # ── Topic extraction ───────────────────────────────────────────────────────

    def _extract_topic(self, raw_text: str, intent: str) -> str:
        """
        Extract a short topic label from the raw text.
        Best-effort: grab the first meaningful noun phrase or fall back to intent.
        """
        # Strip common query prefixes
        cleaned = re.sub(
            r"^(what is|who is|how does|tell me about|explain|define|write|build|create|fix|help with)\s+",
            "", raw_text.strip().lower(), flags=re.IGNORECASE,
        )
        # Take first 6 words
        words = cleaned.split()[:6]
        if words:
            return " ".join(words)
        return intent.replace("_", " ")

    # ── Memory triggers ────────────────────────────────────────────────────────

    def _extract_memory_triggers(
        self,
        raw_text: str,
        thread_state: ActiveThreadStateRecord | None,
    ) -> list[str]:
        """
        Extract keyword triggers for memory retrieval lookup.
        Uses stopword-filtered terms from the user message + thread tags.
        """
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "to", "of", "and",
            "or", "but", "in", "on", "at", "by", "for", "with", "about",
            "what", "who", "how", "when", "where", "why", "i", "you", "my",
            "your", "me", "it", "its", "this", "that", "can", "just", "please",
        }
        clean = re.sub(r"[^\w\s]", " ", raw_text.lower())
        terms = [w for w in clean.split() if w not in stopwords and len(w) > 2]

        # Cap BEFORE de-duplication (pinned by TF-020 tests): a repeated
        # term consumes one of the 8 slots, so the unique message
        # triggers can number fewer than 8. Thread tags then append
        # (max 3, skipping ones already present) and the whole list is
        # capped at 10 — a tag can be cut by that final cap.
        triggers = list(dict.fromkeys(terms[:8]))   # deduplicated, ordered, max 8

        if thread_state and thread_state.tags:
            for tag in thread_state.tags[:3]:
                if tag not in triggers:
                    triggers.append(tag)

        return triggers[:10]

    # ── Urgency estimation ─────────────────────────────────────────────────────

    def _estimate_urgency(self, normalized: str, tone_in: str, intent: str) -> float:
        urgency = _URGENCY_BASELINE
        urgency += _TONE_URGENCY_BONUS.get(tone_in, 0.0)
        urgency += _INTENT_URGENCY_BONUS.get(intent, 0.0)
        return min(urgency, _URGENCY_CAP)

    # ── Retrieval path selection ───────────────────────────────────────────────

    def _select_retrieval_path(self, intent: str, urgency: float) -> str:
        if intent in _SQL_HEAVY_INTENTS:
            return "sql"
        if intent in _VECTOR_HEAVY_INTENTS:
            return "vector"
        return "hybrid"
