"""
All retrieval, pruning, and writeback scoring formulas for ThoughtForge.

Every formula is sourced directly from the spec documents.
No magic numbers — all weights are named constants.

TF-044 (scoring honesty): that claim is now *checked*, not just
asserted — ``knowledge/scoring_honesty.py`` censuses every
constant in this module (and the fusion constants the knowledge
forge consumes from here) and its checker fails by name if a
scoring function body holds a numeric literal that is not a
named constant. The writeback and genericness weights that used
to live as bare literals inside the functions below are named
tables now (values unchanged), and every scoring function takes
an optional ``weights`` table so a caller — e.g. a per-profile
resolution from ``scoring_honesty`` — can tune without editing
this module; the default is always the shipped table.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Retrieval scoring weights (base) ──────────────────────────────────────────

RETRIEVAL_WEIGHTS_BASE = {
    "semantic_similarity": 0.35,
    "tone_similarity": 0.20,
    "preference_relevance": 0.20,
    "recency": 0.10,
    "importance": 0.15,
}

RETRIEVAL_WEIGHTS_BY_TYPE: dict[str, dict[str, float]] = {
    "user_preference": {
        "semantic_similarity": 0.25,
        "tone_similarity": 0.20,
        "preference_relevance": 0.30,
        "recency": 0.05,
        "importance": 0.20,
        "thread_boost": 0.00,
        "pattern_quality": 0.00,
        "priority": 0.00,
    },
    "user_fact": {
        "semantic_similarity": 0.40,
        "tone_similarity": 0.05,
        "preference_relevance": 0.15,
        "recency": 0.10,
        "importance": 0.20,
        "thread_boost": 0.10,
        "pattern_quality": 0.00,
        "priority": 0.00,
    },
    "episode": {
        "semantic_similarity": 0.30,
        "tone_similarity": 0.25,
        "preference_relevance": 0.10,
        "recency": 0.20,
        "importance": 0.15,
        "thread_boost": 0.00,
        "pattern_quality": 0.00,
        "priority": 0.00,
    },
    "response_pattern": {
        "semantic_similarity": 0.25,
        "tone_similarity": 0.15,
        "preference_relevance": 0.30,
        "recency": 0.05,
        "importance": 0.10,
        "thread_boost": 0.00,
        "pattern_quality": 0.15,
        "priority": 0.00,
    },
    "active_thread_state": {
        "semantic_similarity": 0.30,
        "tone_similarity": 0.10,
        "preference_relevance": 0.15,
        "recency": 0.10,
        "importance": 0.05,
        "thread_boost": 0.15,
        "pattern_quality": 0.00,
        "priority": 0.15,
    },
}

# ── Thresholds ─────────────────────────────────────────────────────────────────

MIN_MEMORY_SCORE: float = 0.45
MIN_FRAGMENT_SCORE: float = 0.54
MIN_FINAL_RESPONSE_SCORE: float = 0.68
MIN_PREFERENCE_WRITEBACK_SCORE: float = 0.72
MIN_EPISODE_WRITEBACK_SCORE: float = 0.58
MIN_PATTERN_LEARNING_SCORE: float = 0.78

# ── Writeback + pattern-learning weights (named by TF-044; previously
# bare literals inside the compute_* functions — values unchanged) ──

PREFERENCE_WRITEBACK_WEIGHTS = {
    "durability": 0.30,
    "novelty": 0.20,
    "confidence": 0.25,
    "importance": 0.10,
    "future_relevance": 0.15,
}

EPISODE_WRITEBACK_WEIGHTS = {
    "novelty": 0.20,
    "confidence": 0.15,
    "importance": 0.25,
    "future_relevance": 0.20,
    "thread_relevance": 0.20,
}

PATTERN_LEARNING_WEIGHTS = {
    "final_response_quality": 0.45,
    "pattern_reusability": 0.30,
    "future_relevance": 0.25,
}

# ── Genericness penalty constants (named by TF-044; values unchanged) ──

GENERICNESS_PHRASE_PENALTY: float = 0.15     # per matched generic phrase
GENERICNESS_PHRASE_PENALTY_CAP: float = 0.6  # phrase share never exceeds this
GENERICNESS_REPETITION_WEIGHT: float = 0.4   # repetition ratio's share

# ── Recency fallback (named by TF-044; value unchanged) ──
# Half-life for a record type with no HALF_LIFE_DAYS row.
DEFAULT_HALF_LIFE_DAYS: float = 30.0

# ── Knowledge-forge fusion constants (named by TF-044) ──────────────────────
# The forge's SQL/vector fusion inputs, consumed by knowledge/forge.py.
# Values unchanged from the literals the forge carried inline; TF-041
# mapped this arithmetic as a proxy and TF-043 pinned its consequences
# (a GeoNames record's composite is always exactly 0.285 at these
# values — see scoring_honesty.GEONAMES_COMPOSITE).

FTS_SCORE_DIVISOR: float = 20.0        # semantic proxy = min(fts / this, 1)
SOURCE_RECENCY_PROXIES: dict[str, float] = {
    "reference": 0.9,
    "wikidata": 0.7,
    "conceptnet": 0.8,
    "geonames": 0.75,
    "dbpedia": 0.65,
}
DEFAULT_SOURCE_RECENCY: float = 0.7    # a source with no proxy row
NEUTRAL_TONE_SIMILARITY: float = 0.5   # no tone metadata at DB level
NEUTRAL_PREFERENCE_RELEVANCE: float = 0.5
IMPORTANCE_WEIGHT_DEFAULT: float = 0.5  # record.get("weight", this)
GEONAMES_FIXED_FTS_SCORE: float = 0.5   # the GeoNames pass's fixed score
VECTOR_MIN_SCORE: float = 0.3           # vector pass's similarity floor

# ── Pruning score weights ──────────────────────────────────────────────────────

PRUNE_SCORE_WEIGHTS = {
    "staleness": 0.25,
    "redundancy": 0.20,
    "low_confidence": 0.15,
    "low_importance": 0.15,
    "low_quality": 0.10,
    "low_future_relevance": 0.10,
    "low_reinforcement": 0.05,
}

SURVIVAL_SCORE_WEIGHTS = {
    "importance": 0.25,
    "confidence": 0.20,
    "quality": 0.15,
    "future_relevance": 0.20,
    "recency": 0.10,
    "reinforcement_strength": 0.10,
}

# ── Recency decay half-lives (in days) ────────────────────────────────────────

HALF_LIFE_DAYS: dict[str, float] = {
    "active_thread_state": 0.07,   # ~2 turns at 10 min/turn
    "episode": 7.0,
    "response_pattern": 30.0,
    "user_preference": 90.0,
    "user_fact": 180.0,
    "personality_core": 9999.0,
}

# ── Reinforcement increments ───────────────────────────────────────────────────

REINFORCE_CONFIDENCE_DELTA: float = 0.05
REINFORCE_IMPORTANCE_DELTA: float = 0.03
REINFORCE_QUALITY_DELTA: float = 0.02


# ── Scoring functions ──────────────────────────────────────────────────────────

@dataclass
class RetrievalDimensions:
    semantic_similarity: float = 0.0
    tone_similarity: float = 0.0
    preference_relevance: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    thread_boost: float = 0.0
    pattern_quality: float = 0.0
    priority: float = 0.0


def score_retrieval(
    dims: RetrievalDimensions,
    record_type: str,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute retrieval score for a memory record using type-specific weights.

    ``weights`` (TF-044) overrides the shipped table for this call —
    the per-profile tuning surface; ``None`` is the shipped table,
    byte-identical to the pre-TF-044 behaviour.
    """
    if weights is None:
        weights = RETRIEVAL_WEIGHTS_BY_TYPE.get(record_type, RETRIEVAL_WEIGHTS_BASE)
    d = {
        "semantic_similarity": dims.semantic_similarity,
        "tone_similarity": dims.tone_similarity,
        "preference_relevance": dims.preference_relevance,
        "recency": dims.recency,
        "importance": dims.importance,
        "thread_boost": dims.thread_boost,
        "pattern_quality": dims.pattern_quality,
        "priority": dims.priority,
    }
    return sum(d.get(k, 0.0) * w for k, w in weights.items())


def compute_recency_score(age_days: float, record_type: str) -> float:
    """Half-life recency decay: score = 0.5 ^ (age / half_life)."""
    half_life = HALF_LIFE_DAYS.get(record_type, DEFAULT_HALF_LIFE_DAYS)
    if half_life <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life)


@dataclass
class PruneComponents:
    staleness: float = 0.0
    redundancy: float = 0.0
    low_confidence: float = 0.0
    low_importance: float = 0.0
    low_quality: float = 0.0
    low_future_relevance: float = 0.0
    low_reinforcement: float = 0.0


def compute_prune_score(
    c: PruneComponents,
    weights: dict[str, float] | None = None,
) -> float:
    """Higher score = more likely to be pruned."""
    w = PRUNE_SCORE_WEIGHTS if weights is None else weights
    return (
        c.staleness * w["staleness"]
        + c.redundancy * w["redundancy"]
        + c.low_confidence * w["low_confidence"]
        + c.low_importance * w["low_importance"]
        + c.low_quality * w["low_quality"]
        + c.low_future_relevance * w["low_future_relevance"]
        + c.low_reinforcement * w["low_reinforcement"]
    )


@dataclass
class SurvivalComponents:
    importance: float = 0.0
    confidence: float = 0.0
    quality: float = 0.0
    future_relevance: float = 0.0
    recency: float = 0.0
    reinforcement_strength: float = 0.0


def compute_survival_score(
    c: SurvivalComponents,
    weights: dict[str, float] | None = None,
) -> float:
    """Higher score = more likely to survive pruning."""
    w = SURVIVAL_SCORE_WEIGHTS if weights is None else weights
    return (
        c.importance * w["importance"]
        + c.confidence * w["confidence"]
        + c.quality * w["quality"]
        + c.future_relevance * w["future_relevance"]
        + c.recency * w["recency"]
        + c.reinforcement_strength * w["reinforcement_strength"]
    )


@dataclass
class WritebackComponents:
    durability: float = 0.0
    novelty: float = 0.0
    confidence: float = 0.0
    importance: float = 0.0
    future_relevance: float = 0.0
    thread_relevance: float = 0.0
    final_response_quality: float = 0.0
    pattern_reusability: float = 0.0


def compute_preference_writeback_score(
    c: WritebackComponents,
    weights: dict[str, float] | None = None,
) -> float:
    w = PREFERENCE_WRITEBACK_WEIGHTS if weights is None else weights
    return (
        c.durability * w["durability"]
        + c.novelty * w["novelty"]
        + c.confidence * w["confidence"]
        + c.importance * w["importance"]
        + c.future_relevance * w["future_relevance"]
    )


def compute_episode_writeback_score(
    c: WritebackComponents,
    weights: dict[str, float] | None = None,
) -> float:
    w = EPISODE_WRITEBACK_WEIGHTS if weights is None else weights
    return (
        c.novelty * w["novelty"]
        + c.confidence * w["confidence"]
        + c.importance * w["importance"]
        + c.future_relevance * w["future_relevance"]
        + c.thread_relevance * w["thread_relevance"]
    )


def compute_pattern_learning_score(
    c: WritebackComponents,
    weights: dict[str, float] | None = None,
) -> float:
    w = PATTERN_LEARNING_WEIGHTS if weights is None else weights
    return (
        c.final_response_quality * w["final_response_quality"]
        + c.pattern_reusability * w["pattern_reusability"]
        + c.future_relevance * w["future_relevance"]
    )


GENERICNESS_PHRASES: frozenset[str] = frozenset({
    "a lot of people", "it's normal", "as an ai", "everything will be okay",
    "in today's world", "many people feel", "you're not alone", "it is what it is",
    "at the end of the day", "there are many ways", "it depends on the situation",
})


def compute_genericness_penalty(text: str) -> float:
    """Heuristic genericness penalty (0.0–1.0). Higher = more generic."""
    text_lower = text.lower()
    hits = sum(1 for phrase in GENERICNESS_PHRASES if phrase in text_lower)
    words = text.split()
    if not words:
        return 0.0
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))
    repetition_ratio = 1.0 - (unique_words / word_count) if word_count > 0 else 0.0
    phrase_penalty = min(hits * GENERICNESS_PHRASE_PENALTY, GENERICNESS_PHRASE_PENALTY_CAP)
    return min(phrase_penalty + repetition_ratio * GENERICNESS_REPETITION_WEIGHT, 1.0)
