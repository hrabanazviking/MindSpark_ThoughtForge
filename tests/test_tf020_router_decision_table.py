"""
TF-020 — Router decision table: one test (or more) for every row.

The table itself is docs/architecture/ROUTER_DECISION_TABLE.md; the
router's implementing tables live in
``src/thoughtforge/cognition/router.py``. These tests pin:

  * every intent row (8 signalled + the light_conversation fallback),
  * every tone row (6 signalled + the neutral fallback),
  * the scoring rule (DISTINCT patterns counted, repetitions ignored),
  * tie-breaking by signal-table precedence (intent and tone),
  * every response-mode row and every retrieval-path row,
  * every urgency row (baseline, each tone bonus, each intent bonus,
    the reachable combinations, the 0.85 reachable maximum),
  * topic extraction rows and memory-trigger rows (including the
    cap-before-de-duplication rule and thread-tag rules),
  * the table-completeness guards — including the regression guard for
    the dead-pattern class TF-020 fixed (an upper-case pattern can
    never match the lower-cased input text).
"""

import pytest

from thoughtforge.cognition.router import (
    InputRouter,
    _INTENT_PRECEDENCE,
    _INTENT_SIGNALS,
    _INTENT_TO_MODE,
    _INTENT_URGENCY_BONUS,
    _SQL_HEAVY_INTENTS,
    _TONE_PRECEDENCE,
    _TONE_SIGNALS,
    _TONE_URGENCY_BONUS,
    _URGENCY_BASELINE,
    _URGENCY_CAP,
    _VECTOR_HEAVY_INTENTS,
)
from thoughtforge.knowledge.models import ActiveThreadStateRecord


@pytest.fixture()
def router():
    return InputRouter()


# ── Intent rows ────────────────────────────────────────────────────────────────

INTENT_PROBES = [
    ("My function throws an error: AttributeError on line 12", "technical_debugging"),
    ("write a module that implements the KnowledgeForge class", "technical_spec_request"),
    ("Rewrite this paragraph to be more concise", "editing_refinement"),
    ("I need some ideas and alternatives for the project name", "brainstorming"),
    ("give me a name for my boat", "naming_or_branding"),
    ("How should I plan out the development phases?", "planning_request"),
    ("I'm exhausted and overwhelmed by this project", "emotional_support"),
    ("What is the Yggdrasil in Norse mythology?", "factual_query"),
    ("hey there", "light_conversation"),
]


@pytest.mark.parametrize("text,expected", INTENT_PROBES)
def test_intent_rows(router, text, expected):
    assert router.route(text).intent == expected


@pytest.mark.parametrize("text", ["", "   "])
def test_intent_fallback_on_empty_input(router, text):
    sketch = router.route(text)
    assert sketch.intent == "light_conversation"
    assert sketch.tone_in == "neutral"
    assert sketch.topic == "light conversation"
    assert sketch.memory_triggers == []


# ── Fixed dead patterns (TF-020 bug fix: upper-case patterns never matched) ──

def test_fixed_pattern_api_is_spec_signal(router):
    # \bAPI\b could never match lower-cased text; now \bapi\b can.
    # Tie with factual's "tell me about" breaks to spec by precedence.
    assert router.route("tell me about the api").intent == "technical_spec_request"


def test_fixed_pattern_how_do_i_is_planning_signal(router):
    assert router.route("how do i plan the phases").intent == "planning_request"


def test_fixed_pattern_i_dont_know_is_emotional_signal(router):
    assert router.route("i don't know what to do anymore").intent == "emotional_support"


def test_fixed_pattern_ive_been_thinking_is_curious_signal(router):
    assert router.route("i've been thinking about the old hall").tone_in == "curious"


def test_all_signal_patterns_are_lower_case():
    """Regression guard for the dead-pattern class: route() lower-cases
    the input, so any pattern containing an upper-case letter is dead."""
    for table in (_INTENT_SIGNALS, _TONE_SIGNALS):
        for _name, patterns in table:
            for pattern in patterns:
                assert pattern == pattern.lower(), pattern


# ── Scoring rule + tie-breaking ────────────────────────────────────────────────

def test_scoring_counts_distinct_patterns_not_repetitions(router):
    # "error" x3 is ONE distinct debugging pattern; spec has three
    # distinct patterns — spec wins despite debugging's precedence.
    sketch = router.route("error error error, schema design architecture")
    assert sketch.intent == "technical_spec_request"


def test_higher_count_beats_precedence(router):
    # factual (last in precedence) with 3 distinct patterns beats
    # debugging (first) with 1.
    sketch = router.route("error, what is it, who is it, when was it")
    assert sketch.intent == "factual_query"


INTENT_TIES = [
    # (probe, winner) — each probe scores exactly one distinct pattern
    # for two intents; the earlier in _INTENT_PRECEDENCE wins.
    ("error in the schema", "technical_debugging"),          # debugging > spec
    ("design a rewrite", "technical_spec_request"),          # spec > editing
    ("rewrite with ideas", "editing_refinement"),            # editing > brainstorming
    ("suggest names for a coffee shop", "brainstorming"),    # brainstorming > naming
    ("a name for the plan", "naming_or_branding"),           # naming > planning
    ("plan, I am stressed", "planning_request"),             # planning > emotional
    ("I am worried, what is stress", "emotional_support"),   # emotional > factual
    ("what is the plan", "planning_request"),                # planning > factual
]


@pytest.mark.parametrize("text,expected", INTENT_TIES)
def test_intent_tie_breaks_by_precedence(router, text, expected):
    assert router.route(text).intent == expected


TONE_TIES = [
    ("ugh I am tired", "frustrated"),        # frustrated > tired
    ("I am tired, need this asap", "tired"),  # tired > urgent
    ("asap, I wonder", "urgent"),            # urgent > curious
    ("haha, focus!", "playful"),             # playful > focused
]


@pytest.mark.parametrize("text,expected", TONE_TIES)
def test_tone_tie_breaks_by_precedence(router, text, expected):
    assert router.route(text).tone_in == expected


# ── Tone rows ──────────────────────────────────────────────────────────────────

TONE_PROBES = [
    ("ugh why doesn't this work, it's been broken for hours", "frustrated"),
    ("I am drained", "tired"),
    ("I need this fixed ASAP, deadline today", "urgent"),
    ("I'm curious, I wonder how stars form", "curious"),
    ("haha that's funny, tell me a joke lol", "playful"),
    ("I need to focus, let's start, specifically on this", "focused"),
    ("What time is it?", "neutral"),
]


@pytest.mark.parametrize("text,expected", TONE_PROBES)
def test_tone_rows(router, text, expected):
    assert router.route(text).tone_in == expected


# ── Response-mode rows (one per intent, via its probe) ────────────────────────

MODE_BY_INTENT = {
    "technical_debugging": "structured_technical",
    "technical_spec_request": "structured_technical",
    "planning_request": "structured_technical",
    "editing_refinement": "concise_editorial",
    "brainstorming": "creative_brainstorm",
    "naming_or_branding": "creative_brainstorm",
    "emotional_support": "calm_supportive",
    "factual_query": "factual_direct",
    "light_conversation": "light_natural_chat",
}


@pytest.mark.parametrize("text,intent", INTENT_PROBES)
def test_response_mode_rows(router, text, intent):
    sketch = router.route(text)
    assert sketch.intent == intent
    assert sketch.response_mode == MODE_BY_INTENT[intent]
    assert _INTENT_TO_MODE[intent] == MODE_BY_INTENT[intent]


# ── Retrieval-path rows (all 9 intents, direct + via probes) ──────────────────

RETRIEVAL_BY_INTENT = {
    "factual_query": "sql",
    "technical_spec_request": "sql",
    "planning_request": "sql",
    "emotional_support": "vector",
    "brainstorming": "vector",
    "light_conversation": "vector",
    "technical_debugging": "hybrid",
    "editing_refinement": "hybrid",
    "naming_or_branding": "hybrid",
}


@pytest.mark.parametrize("intent,expected", sorted(RETRIEVAL_BY_INTENT.items()))
def test_retrieval_path_rows_direct(router, intent, expected):
    assert router._select_retrieval_path(intent, urgency=0.3) == expected


@pytest.mark.parametrize("text,intent", INTENT_PROBES)
def test_retrieval_path_rows_via_route(router, text, intent):
    assert router.route(text).retrieval_path == RETRIEVAL_BY_INTENT[intent]


# ── Urgency rows (exact arithmetic) ────────────────────────────────────────────

URGENCY_PROBES = [
    # baseline only
    ("What is Python?", 0.30),
    # tone bonuses alone (light intent carries no intent bonus)
    ("I need this fixed ASAP, deadline today", 0.70),   # urgent +0.4
    ("ugh, fed up with waiting", 0.50),                 # frustrated +0.2
    ("I am drained", 0.40),                             # tired +0.1
    ("I'm curious, I wonder how stars form", 0.30),     # curious +0
    ("haha that's funny, tell me a joke lol", 0.30),    # playful +0
    ("I need to focus, let's start, specifically on this", 0.30),  # focused +0
    # intent bonuses alone (neutral tone)
    ("Exception: TypeError on line 5", 0.45),           # debugging +0.15
    ("I feel lost", 0.40),                              # emotional +0.1
    # combinations
    ("ugh why doesn't this work, it's been broken for hours", 0.65),  # frustrated+debugging
    ("I'm exhausted and overwhelmed by this project", 0.50),          # tired+emotional
    ("I feel bad and lost, help asap", 0.80),           # urgent+emotional
    ("this crash is urgent, fix it asap", 0.85),        # urgent+debugging = reachable max
]


@pytest.mark.parametrize("text,expected", URGENCY_PROBES)
def test_urgency_rows(router, text, expected):
    assert router.route(text).urgency == pytest.approx(expected)


def test_urgency_never_exceeds_cap(router):
    for text, _expected in URGENCY_PROBES + INTENT_PROBES + TONE_PROBES:
        assert router.route(text).urgency <= _URGENCY_CAP


# ── Topic rows ─────────────────────────────────────────────────────────────────

def test_topic_strips_question_prefix(router):
    assert router.route("What is a ferry?").topic == "a ferry?"


def test_topic_strips_command_prefix(router):
    assert router.route("Explain the Sovereign RAG architecture").topic == (
        "the sovereign rag architecture"
    )


def test_topic_truncates_to_six_words(router):
    sketch = router.route("one two three four five six seven eight")
    assert sketch.topic == "one two three four five six"


def test_topic_falls_back_to_intent_name(router):
    assert router.route("").topic == "light conversation"


# ── Memory-trigger rows ────────────────────────────────────────────────────────

def test_triggers_drop_stopwords_and_short_words(router):
    assert router.route("what is the api for a ui").memory_triggers == ["api"]


def test_triggers_cap_applies_before_dedup(router):
    # The repeated "lantern" consumes one of the 8 pre-dedup slots, so
    # only 7 unique message triggers survive; 3 thread tags then fill
    # the list to the overall cap of 10 (the 4th tag is never read).
    thread = ActiveThreadStateRecord(
        record_id="thr_tf020",
        tags=["alpha_tag", "beta_tag", "gamma_tag", "delta_tag"],
    )
    sketch = router.route(
        "lantern hollow lantern keeper wrenna sails harbor mist fjord rune stone",
        thread_state=thread,
    )
    assert sketch.memory_triggers == [
        "lantern", "hollow", "keeper", "wrenna", "sails", "harbor", "mist",
        "alpha_tag", "beta_tag", "gamma_tag",
    ]


def test_triggers_thread_tags_append_and_dedup(router):
    thread = ActiveThreadStateRecord(
        record_id="thr_tf020", tags=["lantern", "harbor"],
    )
    sketch = router.route("tell me about the lantern", thread_state=thread)
    assert sketch.memory_triggers == ["tell", "lantern", "harbor"]


def test_triggers_without_thread_state(router):
    sketch = router.route("Tell me about the ThoughtForge knowledge module")
    assert "thoughtforge" in sketch.memory_triggers
    assert len(sketch.memory_triggers) <= 10


# ── Determinism ────────────────────────────────────────────────────────────────

def test_route_is_deterministic(router):
    text = "ugh, this error is broken again, why won't it work"
    assert router.route(text) == router.route(text)


# ── Table completeness guards ──────────────────────────────────────────────────

ALL_INTENTS = {
    "technical_debugging", "technical_spec_request", "editing_refinement",
    "brainstorming", "naming_or_branding", "planning_request",
    "emotional_support", "factual_query", "light_conversation",
}
ALL_TONES = {
    "frustrated", "tired", "urgent", "curious", "playful", "focused", "neutral",
}


def test_intent_taxonomy_complete():
    signalled = {intent for intent, _ in _INTENT_SIGNALS}
    assert signalled | {"light_conversation"} == ALL_INTENTS
    assert set(_INTENT_TO_MODE) == ALL_INTENTS


def test_tone_taxonomy_complete():
    signalled = {tone for tone, _ in _TONE_SIGNALS}
    assert signalled | {"neutral"} == ALL_TONES


def test_precedence_lists_match_signal_table_order():
    assert _INTENT_PRECEDENCE == [i for i, _ in _INTENT_SIGNALS]
    assert _TONE_PRECEDENCE == [t for t, _ in _TONE_SIGNALS]


def test_retrieval_sets_partition_the_taxonomy():
    assert not (_SQL_HEAVY_INTENTS & _VECTOR_HEAVY_INTENTS)
    hybrid = ALL_INTENTS - _SQL_HEAVY_INTENTS - _VECTOR_HEAVY_INTENTS
    assert hybrid == {"technical_debugging", "editing_refinement", "naming_or_branding"}


def test_urgency_tables_reference_real_rows():
    assert set(_TONE_URGENCY_BONUS) <= ALL_TONES
    assert set(_INTENT_URGENCY_BONUS) <= ALL_INTENTS
    assert _URGENCY_BASELINE == 0.3
    # Reachable maximum is baseline + largest tone bonus + largest intent bonus.
    assert _URGENCY_BASELINE + max(_TONE_URGENCY_BONUS.values()) + max(
        _INTENT_URGENCY_BONUS.values()
    ) == pytest.approx(0.85)
