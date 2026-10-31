"""
TF-023 — Prompt Builder Contract: the assembly order of every
PromptBuilder method, specified as data and golden-tested.

The contract lives in ``src/thoughtforge/cognition/prompt_contract.py``
(section order per prompt kind, optional sections, section markers)
and is written down in docs/architecture/PROMPT_BUILDER_CONTRACT.md.
These tests pin:

  * the contract data itself (exactly the six prompt kinds, their
    section tuples, and which sections are optional),
  * golden prompts — byte-exact — for candidate (with and without
    the optional knowledge-context section), strategy (first and
    final steps), refine, repair, episode-summary, and
    preference-inference builders,
  * section order of every real built prompt via check_prompt_order,
    and that the checker actually detects a missing / out-of-order
    section,
  * the TF-023 defect fix: scaffold steering is NEVER truncated in
    candidate / strategy / refine prompts — the old [:3] caps meant
    intent avoid extras and the personality core's avoids (the
    persona's only prompt channel) never reached the model, and the
    user-tone overlay (tone's 4th entry) never did either,
  * the repair prompt's narrowing as SPECIFIED behaviour (tone first
    2, cues max 2, no focus/avoid/knowledge sections),
  * every cap and fallback: cues [:3] candidate / [:2] repair,
    refine fragments kept ≤5 with top-3-by-composite fallback when
    none are kept, episode final-response truncated at 300 chars,
    unknown mode / response-mode / depth fallbacks,
  * memory-cue extraction: type priority beats score, score breaks
    ties inside a type, and max_cues is applied BEFORE empty cues
    are dropped (a cue-less record consumes its slot — specified),
  * the F7 ruling (CORE_LOOP §8): CognitionScaffold has no
    prompt_text field, and builders return strings without mutating
    the scaffold,
  * measured canonical costs at the new contract (independently
    recomputed), and the contract doc naming every kind and method.
"""

import dataclasses
from pathlib import Path

import pytest

from thoughtforge.cognition.prompt_builder import PromptBuilder
from thoughtforge.cognition.prompt_contract import (
    OPTIONAL_SECTIONS,
    PROMPT_SECTIONS,
    check_prompt_order,
    section_markers,
)
from thoughtforge.cognition.scaffold import ScaffoldBuilder
from thoughtforge.cognition.scaffold_registry import (
    canonical_sketch,
    measure_prompt_costs,
    scaffold_shape,
)
from thoughtforge.cognition.scaffold_strategies import (
    measure_strategy_costs,
    strategy_registry,
)
from thoughtforge.knowledge.models import (
    ActivatedRecord,
    CognitionScaffold,
    FragmentRecord,
    FragmentScores,
    InputSketch,
    MemoryActivationBundle,
    PersonalityCoreRecord,
)

# ── Canonical synthetic fixtures (TF-017 policy: invented only) ──────────────

SKETCH = InputSketch(
    raw_text="Where does Wrenna keep the lantern oil?",
    intent="factual_query",
    topic="lantern hollow stores",
    tone_in="curious",
    response_mode="factual_direct",
    urgency=0.3,
)

SCAFFOLD = CognitionScaffold(
    goal="give an accurate, direct answer with cited sources",
    tone=["direct", "factual", "clear", "curious"],
    focus=["accuracy", "brevity", "citation"],
    avoid=[
        "filler", "padding", "repetition of the user's question",
        "guessing", "unverified claims", "persona-avoid-marker",
    ],
    depth="medium",
    candidate_modes=["implementation_friendly"],
    fact_block="[synthetic] Lantern oil — kept in the hollow stores",
)

CUES = [
    "Wrenna trims the wick at dusk",
    "Lantern Hollow fact cue",
    "third cue",
    "fourth cue dropped",
]


@pytest.fixture()
def pb():
    return PromptBuilder()


# ── The contract data ─────────────────────────────────────────────────────────

def test_contract_covers_exactly_the_six_prompt_kinds():
    assert set(PROMPT_SECTIONS) == {
        "candidate", "strategy", "refine", "repair",
        "episode_summary", "preference_inference",
    }
    assert set(OPTIONAL_SECTIONS) == set(PROMPT_SECTIONS)


def test_candidate_sections_are_the_specified_order():
    assert PROMPT_SECTIONS["candidate"] == (
        "opener", "user_request", "intent", "topic", "tone", "goal",
        "focus", "avoid", "memory_cues", "knowledge_context", "output",
    )


def test_only_knowledge_and_working_sections_are_optional():
    assert OPTIONAL_SECTIONS["candidate"] == frozenset({"knowledge_context"})
    assert OPTIONAL_SECTIONS["strategy"] == frozenset(
        {"knowledge_context", "working_so_far"}
    )
    for kind in ("refine", "repair", "episode_summary", "preference_inference"):
        assert OPTIONAL_SECTIONS[kind] == frozenset()


def test_every_marked_section_follows_its_kind_order():
    for kind in PROMPT_SECTIONS:
        markers = section_markers(kind)
        positions = [PROMPT_SECTIONS[kind].index(s) for s in markers]
        assert positions == sorted(positions)


def test_checker_detects_missing_and_out_of_order_sections():
    good = "User request: x\nIntent: y\nTopic: z\nTone: t\nGoal: g\nFocus: f\nAvoid: a\nMemory cues:\n- none\n\nOutput: o"
    assert check_prompt_order(good, "candidate") == []
    missing = good.replace("Focus: f\n", "")
    assert any("focus" in v for v in check_prompt_order(missing, "candidate"))
    swapped = good.replace("Tone: t\nGoal: g", "Goal: g\nTone: t")
    assert any("out of order" in v for v in check_prompt_order(swapped, "candidate"))


# ── Golden prompts ────────────────────────────────────────────────────────────

def test_candidate_prompt_golden_with_facts_and_cues(pb):
    prompt = pb.build_candidate_prompt(
        SKETCH, SCAFFOLD, "implementation_friendly", CUES,
    )
    assert prompt == (
        "Write one clear, implementation-ready response.\n\n"
        "User request: Where does Wrenna keep the lantern oil?\n"
        "Intent: factual query\n"
        "Topic: lantern hollow stores\n"
        "Tone: direct, factual, clear, curious\n"
        "Goal: give an accurate, direct answer with cited sources\n"
        "Focus: accuracy, brevity, citation\n"
        "Avoid: filler, padding, repetition of the user's question, "
        "guessing, unverified claims, persona-avoid-marker\n"
        "Memory cues:\n"
        "- Wrenna trims the wick at dusk\n"
        "- Lantern Hollow fact cue\n"
        "- third cue\n\n"
        "Knowledge context:\n"
        "[synthetic] Lantern oil — kept in the hollow stores\n\n"
        "Output: 1-3 sentences with source citation Keep it focused."
    )
    assert check_prompt_order(prompt, "candidate") == []


def test_candidate_prompt_golden_bare_with_all_fallbacks(pb):
    # Unknown mode -> fallback opener; unknown response_mode ->
    # fallback shape; empty scaffold lists -> documented defaults;
    # no fact block -> the optional section is simply absent.
    sketch = InputSketch(
        raw_text="hello there", intent="fallback", topic="",
        tone_in="neutral", response_mode="conversational", urgency=0.4,
    )
    prompt = pb.build_candidate_prompt(
        sketch, CognitionScaffold(goal="produce a useful, direct response"),
        "no_such_mode", [],
    )
    assert prompt == (
        "Write one short reply.\n\n"
        "User request: hello there\n"
        "Intent: fallback\n"
        "Topic: \n"
        "Tone: direct, natural\n"
        "Goal: produce a useful, direct response\n"
        "Focus: clarity\n"
        "Avoid: filler\n"
        "Memory cues:\n"
        "- none\n\n"
        "Output: 2-4 sentences Keep it focused."
    )
    assert "Knowledge context:" not in prompt
    assert check_prompt_order(prompt, "candidate") == []


def test_candidate_prompt_unknown_depth_has_no_length_hint(pb):
    scaffold = dataclasses.replace(SCAFFOLD, depth="weird", fact_block="")
    prompt = pb.build_candidate_prompt(SKETCH, scaffold, "practical", [])
    assert prompt.endswith("Output: 1-3 sentences with source citation")


def test_strategy_prompt_goldens(pb):
    strategy = strategy_registry()["chain_of_thought"]
    first = pb.build_strategy_prompt(SKETCH, SCAFFOLD, strategy.steps[0], ["cue a"])
    assert first == (
        "Think step by step in short numbered steps. Reason only — "
        "do not give the final answer yet.\n\n"
        "User request: Where does Wrenna keep the lantern oil?\n"
        "Intent: factual query\n"
        "Topic: lantern hollow stores\n"
        "Tone: direct, factual, clear, curious\n"
        "Goal: give an accurate, direct answer with cited sources\n"
        "Focus: accuracy, brevity, citation\n"
        "Avoid: filler, padding, repetition of the user's question, "
        "guessing, unverified claims, persona-avoid-marker\n"
        "Memory cues:\n"
        "- cue a\n\n"
        "Knowledge context:\n"
        "[synthetic] Lantern oil — kept in the hollow stores\n\n"
        "Output: reasoning only"
    )
    final = pb.build_strategy_prompt(
        SKETCH, SCAFFOLD, strategy.steps[1], ["cue a"],
        prior_text="reasoning scratch",
    )
    assert final == (
        "Using the reasoning above, write the final answer. Do not "
        "repeat the reasoning steps.\n\n"
        "User request: Where does Wrenna keep the lantern oil?\n"
        "Intent: factual query\n"
        "Topic: lantern hollow stores\n"
        "Tone: direct, factual, clear, curious\n"
        "Goal: give an accurate, direct answer with cited sources\n"
        "Focus: accuracy, brevity, citation\n"
        "Avoid: filler, padding, repetition of the user's question, "
        "guessing, unverified claims, persona-avoid-marker\n"
        "Memory cues:\n"
        "- cue a\n\n"
        "Knowledge context:\n"
        "[synthetic] Lantern oil — kept in the hollow stores\n\n"
        "Working so far:\n"
        "reasoning scratch\n\n"
        "Output: 1-3 sentences with source citation Keep it focused."
    )
    assert check_prompt_order(first, "strategy") == []
    assert check_prompt_order(final, "strategy") == []


def test_refine_prompt_golden(pb):
    frags = [
        FragmentRecord("f1", "c1", text="Wrenna keeps the lantern oil in the hollow stores.", keep=True),
        FragmentRecord("f2", "c1", text="The stores sit behind the lantern hollow.", keep=True),
    ]
    prompt = pb.build_refine_prompt(frags, SKETCH, SCAFFOLD)
    assert prompt == (
        "Using these useful fragments:\n"
        "- Wrenna keeps the lantern oil in the hollow stores.\n"
        "- The stores sit behind the lantern hollow.\n\n"
        "Write one clean response.\n\n"
        "Intent: factual query\n"
        "Tone: direct, factual, clear, curious\n"
        "Goal: give an accurate, direct answer with cited sources\n"
        "Avoid: filler, padding, repetition of the user's question, "
        "guessing, unverified claims, persona-avoid-marker\n\n"
        "Output: 1-3 sentences with source citation"
    )
    assert check_prompt_order(prompt, "refine") == []


def test_repair_prompt_golden(pb):
    prompt = pb.build_repair_prompt(SKETCH, SCAFFOLD, ["cue one", "cue two", "cue three dropped"])
    assert prompt == (
        "Write one direct response.\n\n"
        "User request: Where does Wrenna keep the lantern oil?\n"
        "Intent: factual query\n"
        "Tone: direct, factual\n"
        "Goal: give an accurate, direct answer with cited sources\n"
        "Memory cues:\n"
        "- cue one\n"
        "- cue two\n\n"
        "Avoid generic wording. Keep it specific and useful.\n"
        "Output: 1-3 sentences with source citation"
    )
    assert check_prompt_order(prompt, "repair") == []


def test_writeback_prompt_goldens(pb):
    assert pb.build_episode_summary_prompt(
        "Where does Wrenna keep the lantern oil?",
        "Wrenna keeps it in the hollow stores.",
    ) == (
        "Summarize the conversation moment in one short memory line.\n\n"
        "User request: Where does Wrenna keep the lantern oil?\n"
        "Final response: Wrenna keeps it in the hollow stores.\n\n"
        "Output: one compact summary under 20 words"
    )
    assert pb.build_preference_inference_prompt(
        "I like short answers", "factual questions",
    ) == (
        "Infer one durable user preference only if strongly supported.\n\n"
        "User request: I like short answers\n"
        "Context: factual questions\n\n"
        "Output:\n"
        "- preference summary\n"
        "or\n"
        "- none"
    )


# ── The TF-023 defect fix: steering is never silently truncated ──────────────

def test_persona_and_intent_avoids_reach_the_candidate_prompt(pb):
    # End-to-end through the real ScaffoldBuilder: the personality
    # core's avoids and the intent's avoid extras must survive into
    # the prompt — before TF-023 the builder's [:3] cap cut them all.
    personality = PersonalityCoreRecord(
        record_id="synthetic_persona",
        avoid=["persona-avoid-alpha", "persona-avoid-beta"],
    )
    sketch = canonical_sketch("factual_query")
    scaffold = ScaffoldBuilder().build(
        sketch, MemoryActivationBundle(), personality,
    )
    assert "persona-avoid-alpha" in scaffold.avoid
    prompt = pb.build_candidate_prompt(sketch, scaffold, "practical", [])
    avoid_line = next(l for l in prompt.splitlines() if l.startswith("Avoid:"))
    assert "guessing" in avoid_line          # intent extra
    assert "unverified claims" in avoid_line  # intent extra
    assert "persona-avoid-alpha" in avoid_line
    assert "persona-avoid-beta" in avoid_line


def test_user_tone_overlay_reaches_candidate_and_refine_prompts(pb):
    # tone_in="curious" overlays a 4th tone entry; the old [:3] cap
    # dropped exactly that entry from candidate prompts and refine's
    # [:2] dropped even the base list's tail.
    sketch = canonical_sketch("factual_query")
    sketch.tone_in = "curious"
    scaffold = ScaffoldBuilder().build(sketch, MemoryActivationBundle())
    assert len(scaffold.tone) == 4
    candidate = pb.build_candidate_prompt(sketch, scaffold, "practical", [])
    tone_line = next(l for l in candidate.splitlines() if l.startswith("Tone:"))
    assert tone_line == "Tone: " + ", ".join(scaffold.tone)
    frags = [FragmentRecord("f1", "c1", text="A kept fragment line.", keep=True)]
    refine = pb.build_refine_prompt(frags, sketch, scaffold)
    refine_tone = next(l for l in refine.splitlines() if l.startswith("Tone:"))
    assert refine_tone == tone_line


def test_repair_narrowing_is_the_specified_one(pb):
    prompt = pb.build_repair_prompt(SKETCH, SCAFFOLD, CUES)
    tone_line = next(l for l in prompt.splitlines() if l.startswith("Tone:"))
    assert tone_line == "Tone: direct, factual"  # first 2 only, by spec
    assert "Focus:" not in prompt
    assert "Knowledge context:" not in prompt
    assert "persona-avoid-marker" not in prompt  # no Avoid section at all
    assert "third cue" not in prompt             # cues capped at 2, by spec


# ── Caps and fallbacks ────────────────────────────────────────────────────────

def test_refine_uses_at_most_five_kept_fragments(pb):
    frags = [
        FragmentRecord(f"f{i}", "c1", text=f"kept fragment {i}", keep=True)
        for i in range(8)
    ]
    prompt = pb.build_refine_prompt(frags, SKETCH, SCAFFOLD)
    assert "kept fragment 4" in prompt
    assert "kept fragment 5" not in prompt


def test_refine_falls_back_to_top_three_by_composite_when_none_kept(pb):
    def frag(i, relevance):
        return FragmentRecord(
            f"f{i}", "c1", text=f"fragment number {i} text",
            scores=FragmentScores(relevance=relevance), keep=False,
        )
    prompt = pb.build_refine_prompt(
        [frag(1, 0.1), frag(2, 0.9), frag(3, 0.5), frag(4, 0.7)],
        SKETCH, SCAFFOLD,
    )
    assert "- fragment number 2 text\n- fragment number 4 text\n- fragment number 3 text" in prompt
    assert "fragment number 1" not in prompt


def test_episode_prompt_truncates_final_response_at_300_chars(pb):
    prompt = pb.build_episode_summary_prompt("req", "x" * 350)
    assert "x" * 300 in prompt
    assert "x" * 301 not in prompt


def test_extract_memory_cues_priority_beats_score(pb):
    bundle = MemoryActivationBundle(activated_records=[
        ActivatedRecord("e1", "entity", score=0.99, cue="entity cue"),
        ActivatedRecord("p1", "user_preference", score=0.10, cue="preference cue"),
        ActivatedRecord("f1", "user_fact", score=0.50, cue="fact cue"),
    ])
    assert pb.extract_memory_cues(bundle, max_cues=3) == [
        "preference cue", "fact cue", "entity cue",
    ]


def test_extract_memory_cues_max_applies_before_empty_cue_drop(pb):
    # Specified quirk: sorting + slicing happen first; a cue-less
    # record in the top max_cues consumes its slot and yields nothing.
    bundle = MemoryActivationBundle(activated_records=[
        ActivatedRecord("p1", "user_preference", score=0.9, cue=""),
        ActivatedRecord("f1", "user_fact", score=0.8, cue="fact cue"),
    ])
    assert pb.extract_memory_cues(bundle, max_cues=1) == []
    assert pb.extract_memory_cues(bundle, max_cues=2) == ["fact cue"]


# ── The F7 ruling ─────────────────────────────────────────────────────────────

def test_scaffold_has_no_prompt_text_field():
    assert "prompt_text" not in scaffold_shape()
    assert not hasattr(CognitionScaffold(), "prompt_text")


def test_builders_return_strings_and_never_mutate_the_scaffold(pb):
    scaffold = dataclasses.replace(SCAFFOLD)
    before = dataclasses.asdict(scaffold)
    pb.build_candidate_prompt(SKETCH, scaffold, "practical", CUES)
    pb.build_candidate_prompt(SKETCH, scaffold, "strict_spec", [])
    pb.build_repair_prompt(SKETCH, scaffold, [])
    frags = [FragmentRecord("f1", "c1", text="A kept fragment line.", keep=True)]
    pb.build_refine_prompt(frags, SKETCH, scaffold)
    assert dataclasses.asdict(scaffold) == before


# ── Measured costs at the new contract ────────────────────────────────────────

def test_candidate_costs_at_the_tf023_contract():
    costs = measure_prompt_costs()
    words = [c.word_tokens for c in costs]
    # Recomputed independently: chars/words follow the live builder.
    assert min(words) == 45
    assert max(words) == 66  # was 60 at TF-021; full avoid lists cost +6 max


def test_strategy_totals_at_the_tf023_contract():
    totals: dict[str, int] = {}
    for cost in measure_strategy_costs():
        totals[cost.strategy] = totals.get(cost.strategy, 0) + cost.word_tokens
    # TF-022 totals were 138 / 122 / 204; full steering adds its words.
    assert totals == {
        "chain_of_thought": 150,
        "plan_then_answer": 128,
        "self_check": 213,
    }


# ── The contract doc ──────────────────────────────────────────────────────────

def test_contract_doc_names_every_kind_method_and_ruling():
    doc = (
        Path(__file__).resolve().parent.parent
        / "docs" / "architecture" / "PROMPT_BUILDER_CONTRACT.md"
    )
    text = doc.read_text(encoding="utf-8")
    for kind in PROMPT_SECTIONS:
        assert kind in text
    for method in (
        "build_candidate_prompt", "build_strategy_prompt",
        "build_refine_prompt", "build_repair_prompt",
        "build_episode_summary_prompt", "build_preference_inference_prompt",
        "extract_memory_cues",
    ):
        assert method in text
    assert "F7" in text
    assert "prompt_text" in text
