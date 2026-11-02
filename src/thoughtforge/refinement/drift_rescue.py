"""TF-092 — Language drift rescue: answers
wandering off-language pulled back or flagged.

Roadmap: "Language drift rescue: answers
wandering off-language pulled back or flagged
(fixtures bilingual)." TF-016 assigns TF-092
D-SALV-1's language-drift share.

Measured start (probed live this tick):

- **No drift check existed anywhere in
  refinement.** TF-087 catalogued it: salvage.py
  has no language detection at all, and
  ``FragmentSalvage`` returns a drifted draft
  byte-identical by construction (its fallback
  is the best draft verbatim). A grep over
  ``refinement/`` + ``cognition/`` for
  ``detect_language`` returned nothing.
- **The tree already owns a detector** —
  TF-060's ``knowledge.multilingual.
  detect_language``, a heuristic hint over
  en / es / de (+ a ``cjk`` script label, ties
  and zero-evidence reported ``unknown``).
  This gate reuses it as-is; it does not invent
  a second detector, and it inherits TF-060's
  stated limits exactly (a hint, never a
  certainty; the world's other languages are
  out of scope until a later slice widens the
  hint table).
- **The rescue therefore lives here, as a
  caller-invoked gate** — the TF-090 / TF-091
  shape. The caller supplies the expected
  language (the language of the ask / the
  conversation — salvage has no such signal and
  must not guess one). Wiring a cut into
  ``FragmentSalvage.forge`` would silently
  rewrite TF-087's catalogue and TF-089's filed
  metrics mid-Part; ``FragmentSalvage`` is
  untouched, TF-093 owns the unified policy
  surface, TF-095 the user-facing telling.
- **Pulled back, by deletion only** (TF-088's
  line): when an on-language core remains and
  drift is not the majority of the judged
  sentences, the off-language sentences are
  *cut* — never translated. Translation would
  invent wording the model never produced in
  the expected language (TF-060's rule:
  translation is the model's job at answer
  time, not the store's — and not a rescue
  gate's). The kept sentences are the model's
  own, verbatim and in order.
- **Flagged, never silently cut**, when there
  is nothing honest to pull back to: the whole
  answer is off-language (no on-language
  sentence to keep), or drift is the majority
  of the judged sentences (cutting would leave
  a minority fragment dressed as the answer).
  A flagged verdict returns the text verbatim
  with the drift recorded, so the caller can
  regenerate or tell the user (TF-095).
- **Judgement is per sentence, by hint.**
  A sentence whose hint is the expected
  language is on-language; a sentence hinting
  a different known language (or the ``cjk``
  script label) is off-language; a sentence
  with no distinctive evidence (``unknown`` —
  a place name alone, a very short sentence)
  is neutral: kept, and counted on neither
  side. Neutral sentences are never cut.

Proof grade FIXTURE — every filed case is run
live through the real gate on this VM, over
bilingual (en<->es) fixtures plus de legs. A
live model's drift is NOT PROVEN (no model in
tree); TF-100's chaos corpus owns real-shaped
failures and TF-101 runs the full stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from thoughtforge.knowledge.multilingual import (
    KNOWN_LANGUAGES,
    detect_language,
)

DRIFT_FORMAT = "thoughtforge-drift-rescue/1"

DRIFT_DOC = "docs/architecture/DRIFT_RESCUE.md"

# ── The rules ───────────────────────────────────────────────────────────────

DRIFT_RULES: tuple[dict[str, Any], ...] = (
    {"id": "expected_language_is_caller_supplied",
     "rule": "the gate judges a draft against an expected "
             "language the caller supplies (the language of "
             "the ask / conversation); it never guesses the "
             "expected language from the draft itself"},
    {"id": "detector_is_tf060s_hint",
     "rule": "sentence languages are TF-060's detect_language "
             "hints (knowledge/multilingual.py), reused as-is — "
             "a heuristic hint, never a certainty; supported "
             "expected languages are exactly TF-060's known "
             "languages (en / es / de)"},
    {"id": "pull_back_by_deletion_only",
     "rule": "a rescue cuts off-language sentences and keeps "
             "the on-language sentences verbatim and in order; "
             "nothing is translated, reworded, or completed "
             "(TF-088's repair/fabrication line)"},
    {"id": "flagged_never_silently_cut",
     "rule": "when the whole answer is off-language or drift "
             "is the majority of the judged sentences, the "
             "verdict is flagged and the text is returned "
             "verbatim with the drift recorded — no minority "
             "fragment is dressed up as the answer"},
    {"id": "neutral_sentences_are_never_cut",
     "rule": "a sentence whose hint is unknown (no distinctive "
             "evidence — a name alone, a very short sentence) "
             "is neutral: kept, and counted on neither side"},
    {"id": "majority_is_of_judged_sentences",
     "rule": "the majority test counts judged sentences only "
             "(on-language + off-language); drift is flagged "
             "when off-language sentences strictly outnumber "
             "on-language ones"},
    {"id": "every_drift_is_recorded",
     "rule": "rescued and flagged verdicts alike carry every "
             "off-language segment in order — its detected "
             "language, its sentences verbatim, its first "
             "sentence index, and whether it was cut "
             "(TF-097 owns carrying the record into the trace)"},
    {"id": "the_gate_is_caller_invoked",
     "rule": "the gate runs when a caller asks it to judge a "
             "draft against an expected language; "
             "FragmentSalvage is unchanged — TF-093 owns the "
             "unified policy surface, TF-095 the user-facing "
             "telling, TF-101 the stack scoring"},
)

# ── Result types ────────────────────────────────────────────────────────────


@dataclass
class DriftSegment:
    """One recorded run of consecutive off-language
    sentences (one detected language)."""
    language: str                   # detected hint of the segment
    sentences: tuple[str, ...]      # verbatim, in order
    first_sentence: int             # index of its first sentence
    cut: bool                       # True when a rescue removed it


@dataclass
class DriftResult:
    """Verdict of the drift gate on one output."""
    verdict: str                    # "clean" | "rescued" | "flagged" | "refused"
    text: str | None = None         # usable / verbatim text; None when refused
    expected_language: str = ""
    segments: tuple[DriftSegment, ...] = ()
    flags: tuple[str, ...] = ()     # named, on flagged verdicts
    problems: tuple[str, ...] = ()  # named, on refused verdicts


# ── Sentence spans (the TF-091 shape) ───────────────────────────────────────

_SENTENCE_RE = re.compile(r"[^.!?\u2026\n]+(?:[.!?\u2026]+[\"'\u201d\u2019)\]]*)?")


@dataclass
class _Span:
    start: int
    end: int
    text: str
    hint: str


def _sentence_spans(text: str, expected: str) -> list[_Span]:
    spans: list[_Span] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group()
        stripped = raw.strip()
        if not stripped:
            continue
        start = match.start() + (len(raw) - len(raw.lstrip()))
        end = match.start() + len(raw.rstrip())
        sentence = text[start:end]
        spans.append(_Span(start, end, sentence,
                           detect_language(sentence).language))
    return spans


def _is_off(hint: str, expected: str) -> bool:
    return hint != expected and hint != "unknown"


# ── The gate ────────────────────────────────────────────────────────────────


def rescue_drift(text: str, expected_language: str) -> DriftResult:
    """Judge one draft for language drift against
    ``expected_language``.

    Verdicts: ``clean`` (no off-language sentence
    — the text returned byte-identical),
    ``rescued`` (off-language sentences cut by
    deletion only, an on-language core kept
    verbatim), ``flagged`` (the whole answer is
    off-language or drift is the majority — the
    text returned verbatim, the drift recorded,
    nothing cut), ``refused`` (nothing to judge:
    empty output, or an expected language the
    TF-060 hint cannot judge — problems named)."""
    expected = (expected_language or "").strip().lower()
    if expected not in KNOWN_LANGUAGES:
        return DriftResult(
            "refused", expected_language=expected,
            problems=("unsupported_expected_language",))
    raw = text or ""
    if not raw.strip():
        return DriftResult(
            "refused", expected_language=expected,
            problems=("empty_output",))

    spans = _sentence_spans(raw, expected)
    if not spans:
        return DriftResult(
            "refused", expected_language=expected,
            problems=("empty_output",))

    off_idx = [i for i, s in enumerate(spans) if _is_off(s.hint, expected)]
    on_count = sum(1 for s in spans if s.hint == expected)
    if not off_idx:
        return DriftResult("clean", text=raw, expected_language=expected)

    # Record the off-language segments (consecutive
    # runs, split when the detected language changes).
    segments: list[DriftSegment] = []

    def _record(cut: bool) -> tuple[DriftSegment, ...]:
        recorded: list[DriftSegment] = []
        run: list[int] = []

        def flush() -> None:
            if run:
                recorded.append(DriftSegment(
                    language=spans[run[0]].hint,
                    sentences=tuple(spans[i].text for i in run),
                    first_sentence=run[0], cut=cut))

        for i in off_idx:
            if run and (i != run[-1] + 1
                        or spans[i].hint != spans[run[0]].hint):
                flush()
                run = []
            run.append(i)
        flush()
        return tuple(recorded)

    if on_count == 0:
        return DriftResult(
            "flagged", text=raw, expected_language=expected,
            segments=_record(cut=False),
            flags=("whole_answer_off_language",))
    if len(off_idx) > on_count:
        return DriftResult(
            "flagged", text=raw, expected_language=expected,
            segments=_record(cut=False),
            flags=("drift_is_majority",))

    # Rescue: delete each off-language span (with the
    # separator that follows it), keep everything else
    # byte-identical.
    rescued = raw
    for i in sorted(off_idx, reverse=True):
        start = spans[i].start
        end = spans[i + 1].start if i + 1 < len(spans) else spans[i].end
        rescued = rescued[:start] + rescued[end:]
    return DriftResult(
        "rescued", text=rescued.strip(), expected_language=expected,
        segments=_record(cut=True))


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic Lantern Hollow material throughout
# (TF-017's world, never canon). Bilingual en<->es
# legs plus de legs, judged by TF-060's real hints.
# expected_text is the exact usable / verbatim text;
# segments are filed as (language, sentence count,
# cut) in order.

_EN1 = "The ferry leaves at dawn and the tide is low."
_EN2 = "The keeper lights the lantern when the ferry is late."
_EN3 = "The bell rings across the harbour and the water is calm."
_ES1 = "El faro está en la costa y la marea es alta."
_ES2 = "La campana suena cuando el barco llega a la costa."
_ES3 = "El puerto está tranquilo y las olas son suaves."
_DE1 = "Der Leuchtturm steht an der Küste und das Meer ist ruhig."

DRIFT_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "clean_english",
     "expected": "en",
     "text": f"{_EN1} {_EN2}",
     "verdict": "clean",
     "expected_text": f"{_EN1} {_EN2}",
     "segments": ()},
    {"id": "clean_spanish",
     "expected": "es",
     "text": f"{_ES1} {_ES2}",
     "verdict": "clean",
     "expected_text": f"{_ES1} {_ES2}",
     "segments": ()},
    {"id": "clean_german",
     "expected": "de",
     "text": f"{_DE1} Der Hafen ist still und die Fischer sind fort.",
     "verdict": "clean",
     "expected_text": f"{_DE1} Der Hafen ist still und die Fischer sind fort.",
     "segments": ()},
    {"id": "clean_neutral_sentence_kept",
     "expected": "en",
     "text": f"{_EN1} Lantern Hollow. {_EN2}",
     "verdict": "clean",
     "expected_text": f"{_EN1} Lantern Hollow. {_EN2}",
     "segments": ()},
    {"id": "trailing_spanish_drift",
     "expected": "en",
     "text": f"{_EN1} {_EN2} {_ES1} {_ES2}",
     "verdict": "rescued",
     "expected_text": f"{_EN1} {_EN2}",
     "segments": (("es", 2, True),)},
    {"id": "trailing_english_drift_spanish_answer",
     "expected": "es",
     "text": f"{_ES1} {_ES2} {_EN1}",
     "verdict": "rescued",
     "expected_text": f"{_ES1} {_ES2}",
     "segments": (("en", 1, True),)},
    {"id": "mid_answer_spanish_block",
     "expected": "en",
     "text": f"{_EN1} {_ES1} {_ES2} {_EN2} {_EN3}",
     "verdict": "rescued",
     "expected_text": f"{_EN1} {_EN2} {_EN3}",
     "segments": (("es", 2, True),)},
    {"id": "trailing_german_drift",
     "expected": "en",
     "text": f"{_EN1} {_EN2} {_EN3} {_DE1}",
     "verdict": "rescued",
     "expected_text": f"{_EN1} {_EN2} {_EN3}",
     "segments": (("de", 1, True),)},
    {"id": "single_sentence_drift",
     "expected": "en",
     "text": f"{_EN1} {_EN2} {_ES3}",
     "verdict": "rescued",
     "expected_text": f"{_EN1} {_EN2}",
     "segments": (("es", 1, True),)},
    {"id": "leading_drift_then_recovers",
     "expected": "en",
     "text": f"{_ES1} {_EN1} {_EN2} {_EN3}",
     "verdict": "rescued",
     "expected_text": f"{_EN1} {_EN2} {_EN3}",
     "segments": (("es", 1, True),)},
    {"id": "whole_answer_spanish_flagged",
     "expected": "en",
     "text": f"{_ES1} {_ES2}",
     "verdict": "flagged",
     "expected_text": f"{_ES1} {_ES2}",
     "segments": (("es", 2, False),),
     "flags": ("whole_answer_off_language",)},
    {"id": "whole_answer_english_flagged",
     "expected": "es",
     "text": f"{_EN1} {_EN2}",
     "verdict": "flagged",
     "expected_text": f"{_EN1} {_EN2}",
     "segments": (("en", 2, False),),
     "flags": ("whole_answer_off_language",)},
    {"id": "drift_majority_flagged",
     "expected": "en",
     "text": f"{_EN1} {_ES1} {_ES2} {_ES3}",
     "verdict": "flagged",
     "expected_text": f"{_EN1} {_ES1} {_ES2} {_ES3}",
     "segments": (("es", 3, False),),
     "flags": ("drift_is_majority",)},
    {"id": "refuse_empty",
     "expected": "en",
     "text": "   ",
     "verdict": "refused",
     "expected_text": None,
     "segments": (),
     "problems": ("empty_output",)},
    {"id": "refuse_unsupported_expected_language",
     "expected": "fr",
     "text": _EN1,
     "verdict": "refused",
     "expected_text": None,
     "segments": (),
     "problems": ("unsupported_expected_language",)},
    {"id": "refuse_unknown_expected_language",
     "expected": "unknown",
     "text": _EN1,
     "verdict": "refused",
     "expected_text": None,
     "segments": (),
     "problems": ("unsupported_expected_language",)},
)


# ── Checkers ────────────────────────────────────────────────────────────────


def drift_rescue_problems() -> list[str]:
    """Data-integrity checker: rule and fixture ids
    unique, every fixture well-formed, refusals
    naming their problems, flagged cases naming
    their flags, rescues naming cut segments and
    flagged cases naming uncut ones."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in DRIFT_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate DRIFT_RULES ids")
    fixture_ids = [f["id"] for f in DRIFT_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate DRIFT_FIXTURES ids")
    for fixture in DRIFT_FIXTURES:
        if fixture["verdict"] not in (
                "clean", "rescued", "flagged", "refused"):
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{fixture['verdict']!r} not in the vocabulary")
        if fixture["verdict"] == "refused" and \
                not fixture.get("problems"):
            problems.append(
                f"fixture {fixture['id']}: a refusal must "
                "name its problems")
        if fixture["verdict"] == "flagged" and \
                not fixture.get("flags"):
            problems.append(
                f"fixture {fixture['id']}: a flagged verdict "
                "must name its flags")
        if fixture["verdict"] in ("rescued", "flagged") and \
                not fixture["segments"]:
            problems.append(
                f"fixture {fixture['id']}: drift must be "
                "recorded in segments")
        if fixture["verdict"] == "clean" and fixture["segments"]:
            problems.append(
                f"fixture {fixture['id']}: a clean verdict "
                "carries drift segments")
        for language, _count, cut in fixture["segments"]:
            if fixture["verdict"] == "rescued" and not cut:
                problems.append(
                    f"fixture {fixture['id']}: a rescued "
                    "segment is filed uncut")
            if fixture["verdict"] == "flagged" and cut:
                problems.append(
                    f"fixture {fixture['id']}: a flagged "
                    "segment is filed cut")
    return problems


def drift_rescue_live_problems() -> list[str]:
    """Live checker: every filed fixture is run
    through the real gate and must land exactly as
    filed (verdict, usable text, segments in
    order); a rescued text must be a deletion of
    its source (every kept word is the source's
    own, in order) and must judge clean on a
    second pass; a flagged text must be
    byte-identical to its source."""
    problems: list[str] = []
    for fixture in DRIFT_FIXTURES:
        result = rescue_drift(fixture["text"], fixture["expected"])
        if result.verdict != fixture["verdict"]:
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{result.verdict} != {fixture['verdict']}")
            continue
        if result.text != fixture["expected_text"]:
            problems.append(
                f"fixture {fixture['id']}: text "
                f"{result.text!r} != {fixture['expected_text']!r}")
        filed_segments = tuple(
            (language, count, cut)
            for language, count, cut in fixture["segments"])
        live_segments = tuple(
            (seg.language, len(seg.sentences), seg.cut)
            for seg in result.segments)
        if live_segments != filed_segments:
            problems.append(
                f"fixture {fixture['id']}: segments "
                f"{live_segments} != {filed_segments}")
        for seg in result.segments:
            for sentence in seg.sentences:
                if sentence not in fixture["text"]:
                    problems.append(
                        f"fixture {fixture['id']}: a recorded "
                        "drift sentence is not verbatim in "
                        "the source")
        if fixture["verdict"] == "refused":
            if result.text is not None:
                problems.append(
                    f"fixture {fixture['id']}: a refusal "
                    "carries usable text")
            if tuple(result.problems) != tuple(fixture["problems"]):
                problems.append(
                    f"fixture {fixture['id']}: problems "
                    f"{result.problems} != {fixture['problems']}")
        if fixture["verdict"] == "flagged":
            if result.text != fixture["text"]:
                problems.append(
                    f"fixture {fixture['id']}: a flagged "
                    "verdict is not byte-identical to its source")
            if tuple(result.flags) != tuple(fixture["flags"]):
                problems.append(
                    f"fixture {fixture['id']}: flags "
                    f"{result.flags} != {fixture['flags']}")
        if fixture["verdict"] == "clean" and \
                result.text != fixture["text"]:
            problems.append(
                f"fixture {fixture['id']}: a clean verdict "
                "is not byte-identical to its source")
        if fixture["verdict"] == "rescued":
            second = rescue_drift(result.text or "",
                                  fixture["expected"])
            if second.verdict != "clean" or \
                    second.text != result.text:
                problems.append(
                    f"fixture {fixture['id']}: the rescued "
                    "text does not judge clean on a second pass")
    return problems


def drift_as_data() -> dict[str, Any]:
    return {
        "format": DRIFT_FORMAT,
        "drift_doc": DRIFT_DOC,
        "supported_expected_languages": list(KNOWN_LANGUAGES),
        "rules": [dict(r) for r in DRIFT_RULES],
        "fixtures": [dict(f) for f in DRIFT_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_drift": "not_proven",
            "salvage_py": "unchanged_by_design",
            "detector": "tf060_detect_language_hint",
        },
    }
