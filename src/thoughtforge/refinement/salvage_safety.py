"""TF-088 — Salvage safety: the line between repair
and fabrication, specified as data and enforced in
``refinement/salvage.py``'s refine adoption rule.

Roadmap: "Salvage safety: rescue never *invents*
content — the line between repair and fabrication
specified + tested with adversarial broken
outputs."

Measured start (probed live this tick, every leg
run through the real ``FragmentSalvage`` with a
stub engine and the real ``PromptBuilder``):

- **The no-engine paths cannot invent.** Selection
  returns a candidate verbatim and extraction
  returns verbatim source sentences (TF-087) — by
  construction their output introduces nothing.
  The empty path returns one fixed sentence.
- **The refine path could invent, and did.** Its
  adoption rule was score-only (composite =
  0.45*length + 0.55*citation, TF-087), so content
  the score cannot see was adopted whenever it
  arrived inside a longer, cited text. Five
  adversarial stub-engine outputs were ALL adopted
  as ``refine_pass_1`` over the same cited ferry
  draft: an invented price ("costs 50 silver
  coins"), an invented citation (Q999, silently
  dropped from the citations list while the text
  kept it), a negation flip ("does not leave",
  "never lights"), invented destinations (Bergen,
  Reykjavik), and a displaced time (dawn →
  midnight). Filed as D-SALV-2 and FIXED here: a
  refine pass is now adopted only when
  ``fabrication_problems`` finds no new checkable
  marker against the source pool.
- **The line, stated plainly.** Repair changes
  *form*: selection, deletion, reordering,
  verbatim reassembly, punctuation. Fabrication
  changes *checkable content*: a number, a
  citation, a named entity, a polarity, or a time
  marker that the sources do not carry — or the
  displacement of one they do. The guard judges
  exactly those marker families, mechanically,
  against the source pool (the best draft + the
  kept fragments + the bundle's record cues +
  the retrieved QIDs) — the material the refine
  prompt itself is built from.
- **The residual, stated plainly.** A refine
  output can still elaborate in common words
  (TF-087's own adopted fixture adds a keeper,
  a lantern and a winter timetable in plain
  nouns). No mechanical checker can separate
  that from legitimate rephrasing, so this slice
  does NOT claim it judged: it is filed as the
  residual class in ``SAFETY_RULES``, measured
  by TF-089's corruption rate and policed at
  TF-104's gate over the adversarial set this
  slice files. Sentence-initial capitalised
  words are likewise unjudged (capitalisation
  there is ambiguous).

Proof grade FIXTURE — adversarial outputs are
stub-engine texts run through the real
FragmentSalvage on this VM. A live model's
fabrications are NOT PROVEN here; TF-100's
chaos corpus owns real-shaped failures.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

SAFETY_FORMAT = "thoughtforge-salvage-safety/1"

SAFETY_DOC = "docs/architecture/SALVAGE_SAFETY.md"

# ── The line, as rules ──────────────────────────────────────────────────────

SAFETY_RULES: tuple[dict[str, Any], ...] = (
    {"id": "selection_and_extraction_never_invent",
     "rule": "the no-engine paths return source text verbatim "
             "(a whole draft, or verbatim fragments feeding a "
             "prompt); by construction they introduce no "
             "content, so the fabrication question lives only "
             "in the refine adoption rule"},
    {"id": "repair_is_form_only",
     "rule": "repair may select, delete, reorder, reassemble "
             "verbatim material and change punctuation, casing "
             "or whitespace; it may not add checkable content"},
    {"id": "new_number_is_fabrication",
     "rule": "a digit-form number or number word (three and "
             "up, hundred/thousand/million/billion/dozen) in a "
             "refine output that no source carries rejects "
             "the pass"},
    {"id": "new_citation_is_fabrication",
     "rule": "a QID in a refine output that was neither "
             "retrieved nor present in a source rejects the "
             "pass — TF-087's checker drops it from the "
             "citations list, but the text itself must not "
             "keep it either"},
    {"id": "new_named_entity_is_fabrication",
     "rule": "a capitalised name (mid-sentence) or all-caps "
             "token in a refine output that no source carries "
             "rejects the pass; sentence-initial capitals are "
             "unjudged (ambiguous) and filed as residual"},
    {"id": "polarity_is_content",
     "rule": "negation introduced where no source negates, "
             "or dropped where a source negates, is a meaning "
             "flip, not a repair — the pass is rejected"},
    {"id": "displaced_time_is_fabrication",
     "rule": "a time marker (dawn, midnight, weekdays, "
             "months, …) that appears in a refine output while "
             "a different source time marker disappears is a "
             "displacement and rejects the pass; a time word "
             "merely added alongside the source's own is "
             "common-word elaboration (residual)"},
    {"id": "source_invention_is_not_salvage_invention",
     "rule": "the guard is pool-relative: content already in "
             "a candidate draft is the draft's own (judged by "
             "generation and TF-049/TF-093 enforcement), never "
             "counted as invented by salvage for passing it "
             "through verbatim"},
    {"id": "common_word_elaboration_is_the_residual",
     "rule": "new factual claims phrased entirely in common "
             "words with no new marker are NOT mechanically "
             "decidable against legitimate rephrasing; this "
             "guard does not claim them — TF-089 measures "
             "corruption on the fixture corpus and TF-104's "
             "gate runs the marker-adversarial set at zero"},
    {"id": "rejection_falls_back_never_edits",
     "rule": "a rejected pass is simply not adopted: the "
             "current best text stands, the attempt still "
             "counts in passes_used / salvage_path (TF-087's "
             "attempts naming), and no text is silently "
             "rewritten to remove the invention"},
)

#: The marker families the guard can name.
PROBLEM_KINDS: tuple[str, ...] = (
    "new_number", "new_number_word", "new_citation",
    "new_named_entity", "negation_introduced",
    "negation_dropped", "time_marker_displaced",
)

# ── Marker vocabularies (closed classes, filed as data) ─────────────────────

NUMBER_WORDS: frozenset[str] = frozenset({
    "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "dozen",
})

TIME_MARKERS: frozenset[str] = frozenset({
    "dawn", "daybreak", "sunrise", "morning", "midday",
    "noon", "afternoon", "evening", "dusk", "sunset",
    "twilight", "night", "midnight", "tonight", "today",
    "tomorrow", "yesterday",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november",
    "december",
})

_NEGATION_RAW = {
    "not", "no", "never", "none", "nobody", "nothing",
    "nowhere", "neither", "cannot", "can't", "won't",
    "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "couldn't", "shouldn't", "wouldn't",
}
NEGATION_WORDS: frozenset[str] = frozenset(
    w.replace("'", "") for w in _NEGATION_RAW)

_QID_RE = re.compile(r"\bQ\d{3,}\b")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|Q\d{3,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _normal_number(token: str) -> str:
    return token.replace(",", "").rstrip(".")


# ── The guard ───────────────────────────────────────────────────────────────

def fabrication_problems(
    source_texts: Iterable[str],
    output_text: str,
    retrieved_qids: Iterable[str] = (),
) -> list[str]:
    """Judge ``output_text`` against the source pool.

    Returns named problems, one per invented marker,
    in output order grouped by family in
    ``PROBLEM_KINDS`` order of first appearance.
    An empty list means: no new checkable content —
    the output is a repair (or a verbatim
    pass-through) as far as any mechanical checker
    can tell. Common-word elaboration is the filed
    residual and is never reported here.
    """
    sources = [t or "" for t in source_texts]
    source_joined = "\n".join(sources)
    source_words = _words(source_joined)
    source_lower = {w.lower() for w in source_words}
    # QID digits are citation markers, not numbers: strip
    # QID tokens before either side's number scan.
    source_numbers = {
        _normal_number(n) for n in _NUMBER_RE.findall(
            _QID_RE.sub(" ", source_joined))}
    allowed_qids = set(retrieved_qids) | set(
        _QID_RE.findall(source_joined))

    problems: list[str] = []

    # Numbers, digit-form and word-form.
    for token in _NUMBER_RE.findall(
            _QID_RE.sub(" ", output_text or "")):
        if _normal_number(token) not in source_numbers:
            problems.append(f"new_number:{token}")
    for word in _words(output_text or ""):
        lowered = word.lower()
        if lowered in NUMBER_WORDS and lowered not in source_lower:
            problems.append(f"new_number_word:{lowered}")

    # Citations.
    for qid in _QID_RE.findall(output_text or ""):
        if qid not in allowed_qids:
            problems.append(f"new_citation:{qid}")

    # Named entities: capitalised mid-sentence words
    # and all-caps tokens, judged case-insensitively
    # against the pool. Sentence-initial capitals
    # are skipped (ambiguous, filed residual).
    source_upper = {w for w in source_words}
    for sentence in _SENTENCE_RE.split(output_text or ""):
        tokens = _words(sentence)
        for position, token in enumerate(tokens):
            if _QID_RE.fullmatch(token) or token.isdigit():
                continue
            if token == "I":
                continue
            is_all_caps = (
                len(token) >= 3 and token.isalpha()
                and token.isupper())
            is_capitalised = (
                token[:1].isupper() and any(
                    c.islower() for c in token))
            if is_all_caps:
                if token not in source_upper and \
                        token.lower() not in source_lower:
                    problems.append(f"new_named_entity:{token}")
            elif is_capitalised and position > 0:
                if token.lower() not in source_lower:
                    problems.append(f"new_named_entity:{token}")

    # Polarity: negation presence must match the pool's.
    output_lower = {w.lower().replace("'", "")
                    for w in _words(output_text or "")}
    source_neg = bool(source_lower & NEGATION_WORDS) or any(
        w.lower().replace("'", "") in NEGATION_WORDS
        for w in source_words)
    output_neg = bool(output_lower & NEGATION_WORDS)
    if output_neg and not source_neg:
        problems.append("negation_introduced")
    elif source_neg and not output_neg:
        problems.append("negation_dropped")

    # Time displacement: a new time marker arriving
    # while a source time marker vanishes.
    source_time = source_lower & TIME_MARKERS
    output_time = output_lower & TIME_MARKERS
    new_time = output_time - source_time
    lost_time = source_time - output_time
    if new_time and lost_time:
        problems.append(
            "time_marker_displaced:"
            f"{sorted(lost_time)[0]}->{sorted(new_time)[0]}")

    # Deterministic order, duplicates collapsed.
    seen: set[str] = set()
    ordered: list[str] = []
    for problem in problems:
        if problem not in seen:
            seen.add(problem)
            ordered.append(problem)
    return ordered


def classify_output(
    source_texts: Iterable[str],
    output_text: str,
    retrieved_qids: Iterable[str] = (),
) -> dict[str, Any]:
    """One output's safety verdict as data."""
    sources = list(source_texts)
    problems = fabrication_problems(sources, output_text,
                                     retrieved_qids)
    if output_text in sources:
        verdict = "verbatim_pass_through"
    elif problems:
        verdict = "fabrication"
    else:
        verdict = "repair_within_markers"
    return {
        "verdict": verdict,
        "problems": problems,
        "problem_kinds": sorted(
            {p.split(":", 1)[0] for p in problems}),
    }


# ── Adversarial fixtures ────────────────────────────────────────────────────
# Every fixture was probed live this tick BEFORE
# the guard landed: all five fabrication engine
# outputs were adopted as refine_pass_1. With the
# guard they must be rejected (the candidate text
# stands), and the clean legs must still adopt.

_BASE = (
    "According to Q123, the ferry leaves at dawn when the "
    "tide bell rings across the harbour mouth and the keeper "
    "lights the lantern.")
_NEGATED_BASE = (
    "According to Q123, the ferry does not run in winter "
    "because the harbour bell does not ring at dawn and "
    "the keeper does not light the lantern.")
_TF087_BETTER = (
    "According to Q123, the ferry leaves at dawn when the "
    "tide bell rings across the harbour mouth and the keeper "
    "lights the red lantern for the last crossing of the "
    "night before the winter timetable begins at first light.")
_TF087_BASE = (
    "According to Q123, the ferry leaves at dawn when the "
    "tide bell rings.")

ADVERSARIAL_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "invented_price",
     "candidate": _BASE,
     "engine_text": (
         "According to Q123, the ferry leaves at dawn when "
         "the tide bell rings across the harbour mouth and "
         "the keeper lights the lantern, and the crossing "
         "costs 50 silver coins for every passenger who "
         "boards before the bell."),
     "expected": "rejected",
     "expected_kinds": ("new_number",)},
    {"id": "invented_citation",
     "candidate": _BASE,
     "engine_text": (
         "According to Q123 and Q999, the ferry leaves at "
         "dawn when the tide bell rings across the harbour "
         "mouth and the keeper lights the lantern for the "
         "long crossing over the wide harbour water tonight."),
     "expected": "rejected",
     "expected_kinds": ("new_citation",)},
    {"id": "negation_flip",
     "candidate": _BASE,
     "engine_text": (
         "According to Q123, the ferry does not leave at "
         "dawn when the tide bell rings across the harbour "
         "mouth and the keeper never lights the lantern "
         "for any crossing at all tonight ever."),
     "expected": "rejected",
     "expected_kinds": ("negation_introduced",)},
    {"id": "invented_destinations",
     "candidate": _BASE,
     "engine_text": (
         "According to Q123, the ferry leaves at dawn when "
         "the tide bell rings across the harbour mouth and "
         "the keeper lights the lantern, sailing directly "
         "to Bergen and then onward to Reykjavik tonight."),
     "expected": "rejected",
     "expected_kinds": ("new_named_entity",)},
    {"id": "time_displaced",
     "candidate": _BASE,
     "engine_text": (
         "According to Q123, the ferry leaves at midnight "
         "when the tide bell rings across the harbour mouth "
         "and the keeper lights the lantern for the very "
         "last crossing of the whole long night tonight."),
     "expected": "rejected",
     "expected_kinds": ("time_marker_displaced",)},
    {"id": "negation_dropped",
     "candidate": _NEGATED_BASE,
     "engine_text": (
         "According to Q123, the ferry runs in winter "
         "because the harbour bell rings at dawn and the "
         "keeper lights the lantern for every crossing "
         "over the wide harbour water in the morning."),
     "expected": "rejected",
     "expected_kinds": ("negation_dropped",)},
    {"id": "benign_rephrase",
     "candidate": _BASE,
     "engine_text": (
         "According to Q123, when the tide bell rings "
         "across the harbour mouth, the ferry leaves at "
         "dawn and the keeper lights the lantern."),
     "expected": "adopted",
     "expected_kinds": ()},
    {"id": "common_word_elaboration_residual",
     "candidate": _TF087_BASE,
     "engine_text": _TF087_BETTER,
     "expected": "adopted",
     "expected_kinds": (),
     "note": "TF-087's own adopted refine fixture: the "
             "elaboration arrives in common words with no "
             "new marker, so the guard does not judge it — "
             "the filed residual, owned by TF-089/TF-104"},
    {"id": "source_invention_pass_through",
     "candidate": (
         "The ferry costs 999 coins and sails to Atlantis "
         "at dawn according to Q123 while the tide bell "
         "rings across the harbour mouth."),
     "engine_text": None,
     "expected": "pass_through",
     "expected_kinds": (),
     "note": "no engine: the draft's own inventions pass "
             "through verbatim — salvage did not invent "
             "them (generation/enforcement's judgement)"},
)

_FIXTURE_BY_ID = {f["id"]: f for f in ADVERSARIAL_FIXTURES}


class _StubEngine:
    def __init__(self, text: str):
        self._text = text

    def generate(self, prompt, params=None):
        from types import SimpleNamespace
        return SimpleNamespace(text=self._text)


def run_adversarial(fixture_id: str):
    """Run one filed fixture through the real
    ``FragmentSalvage`` (stub engine + the real
    ``PromptBuilder``; no engine for the
    pass-through fixture) and return its
    SalvageResult."""
    from thoughtforge.cognition.prompt_builder import (
        PromptBuilder)
    from thoughtforge.knowledge.models import (
        ActivatedRecord,
        CandidateRecord,
        CandidateScores,
        CognitionScaffold,
        InputSketch,
        MemoryActivationBundle,
    )
    from thoughtforge.refinement.salvage import FragmentSalvage

    fixture = _FIXTURE_BY_ID[fixture_id]
    candidates = [CandidateRecord(
        candidate_id="cand_0", mode="practical",
        text=fixture["candidate"],
        token_estimate=len(fixture["candidate"].split()),
        scores=CandidateScores())]
    bundle = MemoryActivationBundle(
        activated_records=[ActivatedRecord(
            "Q123", "entity", 0.8, "cue for Q123",
            raw={"qid": "Q123"})],
        total_retrieved=1, retrieval_confidence=0.8)
    if fixture["engine_text"] is None:
        return FragmentSalvage().forge(candidates, bundle)
    sketch = InputSketch(
        raw_text="when does the ferry leave?",
        intent="factual_query", topic="ferry",
        response_mode="factual_direct")
    scaffold = CognitionScaffold(
        goal="give a clear direct answer",
        tone=["direct", "factual"], avoid=["filler"])
    return FragmentSalvage().forge(
        candidates, bundle, engine=_StubEngine(
            fixture["engine_text"]),
        prompt_builder=PromptBuilder(), sketch=sketch,
        scaffold=scaffold, max_passes=1)


# ── Checkers ────────────────────────────────────────────────────────────────

def salvage_safety_problems() -> list[str]:
    """Data-integrity checker: ids unique, every
    fixture's expected kinds are known families,
    and the guard is actually wired into
    salvage.py's adoption rule (source anchor)."""
    import inspect

    problems: list[str] = []
    rule_ids = [r["id"] for r in SAFETY_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate SAFETY_RULES ids")
    fixture_ids = [f["id"] for f in ADVERSARIAL_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate ADVERSARIAL_FIXTURES ids")
    for fixture in ADVERSARIAL_FIXTURES:
        if fixture["expected"] not in (
                "rejected", "adopted", "pass_through"):
            problems.append(
                f"fixture {fixture['id']}: unexpected "
                f"expected {fixture['expected']!r}")
        for kind in fixture["expected_kinds"]:
            if kind not in PROBLEM_KINDS:
                problems.append(
                    f"fixture {fixture['id']}: unknown kind "
                    f"{kind!r}")
    from thoughtforge.refinement import salvage as salvage_module
    source = inspect.getsource(salvage_module)
    if "fabrication_problems" not in source:
        problems.append(
            "salvage.py does not call fabrication_problems — "
            "the guard is not wired into the adoption rule")
    return problems


def salvage_safety_live_problems() -> list[str]:
    """Live checker: every adversarial fixture is
    run through the real FragmentSalvage and must
    land as filed — fabrications rejected (the
    candidate text stands, the attempt still
    counted), clean legs adopted, the pass-through
    verbatim — and the guard's own verdict on each
    engine text must name the filed kinds."""
    problems: list[str] = []
    for fixture in ADVERSARIAL_FIXTURES:
        pool = [fixture["candidate"]]
        if fixture["engine_text"] is not None:
            verdict = classify_output(
                pool, fixture["engine_text"], ("Q123",))
            kinds = set(verdict["problem_kinds"])
            if set(fixture["expected_kinds"]) - kinds:
                problems.append(
                    f"fixture {fixture['id']}: guard kinds "
                    f"{sorted(kinds)} miss filed "
                    f"{fixture['expected_kinds']}")
            if fixture["expected"] == "adopted" and verdict[
                    "problems"]:
                problems.append(
                    f"fixture {fixture['id']}: filed adopted "
                    f"but guard reports {verdict['problems']}")
        result = run_adversarial(fixture["id"])
        if fixture["expected"] == "rejected":
            if result.text != fixture["candidate"]:
                problems.append(
                    f"fixture {fixture['id']}: fabricated "
                    "text was adopted")
            if result.passes_used != 1:
                problems.append(
                    f"fixture {fixture['id']}: rejected "
                    f"attempt not counted (passes_used="
                    f"{result.passes_used})")
        elif fixture["expected"] == "adopted":
            if result.text != fixture["engine_text"]:
                problems.append(
                    f"fixture {fixture['id']}: clean refine "
                    "output was not adopted")
        else:  # pass_through
            if result.text != fixture["candidate"]:
                problems.append(
                    f"fixture {fixture['id']}: no-engine "
                    "output is not the verbatim candidate")
            verdict = classify_output(pool, result.text,
                                      ("Q123",))
            if verdict["verdict"] != "verbatim_pass_through":
                problems.append(
                    f"fixture {fixture['id']}: pass-through "
                    f"classified {verdict['verdict']}")
    return problems


def safety_as_data() -> dict[str, Any]:
    return {
        "format": SAFETY_FORMAT,
        "safety_doc": SAFETY_DOC,
        "rules": [dict(r) for r in SAFETY_RULES],
        "problem_kinds": list(PROBLEM_KINDS),
        "number_words": sorted(NUMBER_WORDS),
        "time_markers": sorted(TIME_MARKERS),
        "fixtures": [dict(f) for f in ADVERSARIAL_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_fabrications": "not_proven",
            "common_word_elaboration": "residual_not_judged",
        },
    }
