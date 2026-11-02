"""TF-091 — Repetition & loop rescue: degenerate
loops detected and cut by rule.

Roadmap: "Repetition & loop rescue: degenerate
loops detected and cut by rule (small models'
classic failure)." TF-016 assigns TF-091
D-SALV-1's repetition/loop share.

Measured start (probed live this tick):

- **No loop cut existed anywhere.**
  ``FragmentSalvage`` returned TF-087's filed
  20x loop fixture ("The bell rings." x20,
  319 chars) byte-identical as ``best_draft``
  — TF-087 catalogued exactly that (its dedup
  touches the fragment list, never the output
  text) and TF-089's metrics file the loop
  among the 5 unrescued verbatim. The other
  loop shapes probed this tick — a sentence
  cycle, a loop in mid-answer, a phrase loop
  with no punctuation — have no detector
  either: salvage's fallback returns the best
  draft verbatim by construction.
- **The rescue therefore lives here, as a
  caller-invoked gate** — the TF-090 shape.
  Loop detection needs no signal from the ask
  (unlike TF-090's structured-ask signal), but
  wiring a cut into ``FragmentSalvage.forge``
  would silently rewrite TF-087's catalogue
  and TF-089's filed metrics mid-Part; the
  gate is a separate surface until TF-093
  unifies the policy stack and TF-101 scores
  it over TF-100's corpus. ``FragmentSalvage``
  is untouched by this slice, so those filings
  stand exactly as banked.
- **The cut line is TF-088's line, applied to
  repetition**: a cut only *deletes* — excess
  copies of a unit the model itself produced.
  The first full occurrence of every loop unit
  is kept verbatim; nothing is reworded,
  completed, or summarised. A cut is recorded
  with its kind, its kept unit, its full-repeat
  count, and the dropped text verbatim.
- **Detection is exact, by stated thresholds.**
  Sentences are compared normalised (casefold,
  whitespace collapsed, terminal punctuation
  stripped — the kept text is the first
  occurrence verbatim): a run or cycle counts
  as a loop at >= 3 full repeats of a period of
  1–3 sentences. Inside the word stream, a
  block of 1–8 words counts at >= 4 full
  repeats. Below those thresholds repetition
  is left alone — a sentence said twice may be
  emphasis, and this gate does not judge it.
  A partial repetition trailing a proven loop
  (whole units matching the period's prefix,
  or one word that is a truncated copy of the
  expected next word) is dropped with the
  loop: it is an incomplete copy, and an
  incomplete copy is never completed.
- **Filed limits, not hidden**: near-duplicate
  loops (a cycle whose copies differ by a word
  each time, paraphrase loops) are NOT
  detected — exact normalised equality only.
  Non-consecutive repetition (a refrain
  returning after other material, outside a
  consecutive cycle) is NOT cut. Rambling that
  is not exact repetition is NOT compressed —
  TF-089's measured share stands; only the
  loop-shaped subset of the filed rambling
  fixture (one phrase repeated verbatim) is
  cut here, and what the cut leaves is a
  fragment, filed as such: loop rescue cuts
  repetition, it does not polish the remainder
  (TF-090's truncation gate judges endings).

Proof grade FIXTURE — every filed case is run
live through the real gate functions on this
VM. A live model's degenerate loops are NOT
PROVEN (no model in tree); TF-100's chaos
corpus owns real-shaped failures and TF-101
runs the full policy stack over them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

LOOP_FORMAT = "thoughtforge-loop-rescue/1"

LOOP_DOC = "docs/architecture/LOOP_RESCUE.md"

# ── The thresholds ──────────────────────────────────────────────────────────
#: The loop thresholds, filed as data and judged
#: live by the checker — a silent threshold change
#: in the detectors fails this slice's tests.

LOOP_CONSTANTS: dict[str, Any] = {
    "sentence_min_repeats": 3,
    "sentence_max_period": 3,
    "word_min_repeats": 4,
    "word_min_period": 1,
    "word_max_period": 8,
    "max_cuts": 32,
}

# ── The rules ───────────────────────────────────────────────────────────────

LOOP_RULES: tuple[dict[str, Any], ...] = (
    {"id": "cut_only_never_rewrite",
     "rule": "a cut deletes excess copies of a unit the model "
             "itself produced; the first full occurrence is kept "
             "verbatim and nothing is reworded, completed, or "
             "summarised (TF-088's repair/fabrication line)"},
    {"id": "sentence_loop_threshold",
     "rule": "a sentence-level loop is a period of 1-3 sentences "
             "repeated consecutively for >= 3 full repeats, "
             "compared normalised (casefold, whitespace collapsed, "
             "terminal punctuation stripped); it is cut to one "
             "period occurrence"},
    {"id": "word_loop_threshold",
     "rule": "a word-level loop is a block of 1-8 words repeated "
             "consecutively for >= 4 full repeats, compared "
             "normalised (casefold, surrounding punctuation "
             "stripped); it is cut to one block occurrence"},
    {"id": "below_threshold_is_left_alone",
     "rule": "repetition under the thresholds passes byte-"
             "identical with verdict 'clean' — a sentence said "
             "twice may be emphasis and this gate does not judge it"},
    {"id": "partial_tail_dropped_never_completed",
     "rule": "a partial repetition trailing a proven loop (whole "
             "units matching the period's prefix, or one word that "
             "is a truncated copy of the expected next word) is "
             "dropped with the loop and recorded; an incomplete "
             "copy is never completed"},
    {"id": "every_cut_is_recorded",
     "rule": "a rescued verdict carries every cut in order — kind, "
             "kept unit verbatim, full-repeat count, and the "
             "dropped text verbatim in the text as it stood when "
             "that cut fired (TF-097 owns carrying the record "
             "into the trace)"},
    {"id": "exact_detection_only",
     "rule": "detection is exact normalised equality only: "
             "near-duplicate or paraphrase loops, and non-"
             "consecutive repetition outside a consecutive cycle, "
             "are not detected and not cut — filed limits, "
             "never implied"},
    {"id": "the_gate_is_caller_invoked",
     "rule": "the gate runs when a caller asks it to judge a "
             "draft; FragmentSalvage is unchanged — TF-093 owns "
             "the unified policy surface, TF-095 the user-facing "
             "telling, TF-101 the stack scoring"},
)

# ── Result types ────────────────────────────────────────────────────────────


@dataclass
class LoopCut:
    """One recorded cut: what looped, what was kept,
    and the dropped text verbatim."""
    kind: str                       # "sentence_loop" | "sentence_cycle" | "word_loop"
    unit: str                       # the kept first period, verbatim
    occurrences: int                # full repeats found (>= threshold)
    dropped_text: str               # verbatim, contiguous in the stage text
    partial_units: int = 0          # partial-tail units dropped with the loop


@dataclass
class LoopResult:
    """Verdict of the loop gate on one output."""
    verdict: str                    # "clean" | "rescued" | "refused"
    text: str | None = None         # usable text; None when refused
    cuts: tuple[LoopCut, ...] = ()
    problems: tuple[str, ...] = ()


# ── Normalisation (comparison only — kept text is verbatim) ─────────────────

_WS_RE = re.compile(r"\s+")
_TERMINAL = ".!?\u2026"
_WORD_STRIP = ".,!?;:\"'()[]{}\u201c\u201d\u2018\u2019\u2026"


def _sentence_norm(text: str) -> str:
    folded = _WS_RE.sub(" ", text.casefold()).strip()
    folded = folded.strip("\"'\u201c\u201d\u2018\u2019")
    return folded.rstrip(_TERMINAL).strip()


def _word_norm(word: str) -> str:
    return word.casefold().strip(_WORD_STRIP)


# ── Sentence spans ──────────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r"[^.!?\u2026\n]+(?:[.!?\u2026]+[\"'\u201d\u2019)\]]*)?")


@dataclass
class _Span:
    start: int
    end: int
    text: str
    norm: str


def _sentence_spans(text: str) -> list[_Span]:
    spans: list[_Span] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group()
        stripped = raw.strip()
        if not stripped:
            continue
        start = match.start() + (len(raw) - len(raw.lstrip()))
        end = match.start() + len(raw.rstrip())
        spans.append(_Span(start, end, text[start:end],
                           _sentence_norm(stripped)))
    return spans


def _word_spans(text: str) -> list[_Span]:
    spans: list[_Span] = []
    for match in re.finditer(r"\S+", text):
        word = match.group()
        norm = _word_norm(word) or word.casefold()
        spans.append(_Span(match.start(), match.end(), word, norm))
    return spans


# ── The detectors ───────────────────────────────────────────────────────────
# Each finder returns (remove_start, remove_end, LoopCut)
# for the FIRST loop in the text (leftmost start,
# smallest period), or None. Removing
# text[remove_start:remove_end] deletes exactly the
# excess copies plus any partial tail, leaving the
# first period occurrence and everything after the
# loop byte-identical.


def _find_sentence_cut(text: str) -> tuple[int, int, LoopCut] | None:
    spans = _sentence_spans(text)
    norms = [s.norm for s in spans]
    n = len(spans)
    min_repeats = LOOP_CONSTANTS["sentence_min_repeats"]
    max_period = LOOP_CONSTANTS["sentence_max_period"]
    for s in range(n):
        for p in range(1, max_period + 1):
            if s + p > n:
                break
            period = norms[s:s + p]
            if any(not unit for unit in period):
                continue
            k = 0
            while s + (k + 1) * p <= n and \
                    norms[s + k * p:s + (k + 1) * p] == period:
                k += 1
            if k < min_repeats:
                continue
            base = s + k * p
            t = 0
            while t < p and base + t < n and norms[base + t] == period[t]:
                t += 1
            last = base + t - 1 if t else s + k * p - 1
            cut = LoopCut(
                kind="sentence_loop" if p == 1 else "sentence_cycle",
                unit=text[spans[s].start:spans[s + p - 1].end],
                occurrences=k,
                dropped_text=text[spans[s + p].start:spans[last].end],
                partial_units=t,
            )
            return spans[s + p - 1].end, spans[last].end, cut
    return None


def _find_word_cut(text: str) -> tuple[int, int, LoopCut] | None:
    spans = _word_spans(text)
    norms = [s.norm for s in spans]
    n = len(spans)
    min_repeats = LOOP_CONSTANTS["word_min_repeats"]
    min_period = LOOP_CONSTANTS["word_min_period"]
    max_period = LOOP_CONSTANTS["word_max_period"]
    for s in range(n):
        for p in range(min_period, max_period + 1):
            if s + p > n:
                break
            period = norms[s:s + p]
            if any(not unit for unit in period):
                continue
            k = 0
            while s + (k + 1) * p <= n and \
                    norms[s + k * p:s + (k + 1) * p] == period:
                k += 1
            if k < min_repeats:
                continue
            base = s + k * p
            t = 0
            while t < p and base + t < n and norms[base + t] == period[t]:
                t += 1
            last = base + t - 1 if t else s + k * p - 1
            partial = t
            # One further word may be a truncated copy of
            # the expected next period word (a cut word,
            # TF-090's dropped-tail shape): it is dropped
            # with the loop, never completed.
            extra = base + t
            if t < p and extra < n and norms[extra] != period[t] and \
                    period[t].startswith(norms[extra]) and norms[extra]:
                last = extra
                partial = t + 1
            cut = LoopCut(
                kind="word_loop",
                unit=text[spans[s].start:spans[s + p - 1].end],
                occurrences=k,
                dropped_text=text[spans[s + p].start:spans[last].end],
                partial_units=partial,
            )
            return spans[s + p - 1].end, spans[last].end, cut
    return None


# ── The gate ────────────────────────────────────────────────────────────────


def rescue_loops(text: str) -> LoopResult:
    """Judge one draft for degenerate loops.

    Verdicts: ``clean`` (no loop at the filed
    thresholds — the text returned byte-identical),
    ``rescued`` (one or more loops cut by deletion
    only; every cut recorded), ``refused`` (nothing
    to judge — empty output, problems named)."""
    raw = text or ""
    if not raw.strip():
        return LoopResult("refused", problems=("empty_output",))

    cuts: list[LoopCut] = []
    current = raw
    for finder in (_find_sentence_cut, _find_word_cut):
        for _ in range(LOOP_CONSTANTS["max_cuts"]):
            found = finder(current)
            if found is None:
                break
            remove_start, remove_end, cut = found
            current = current[:remove_start] + current[remove_end:]
            cuts.append(cut)
    if not cuts:
        return LoopResult("clean", text=raw)
    return LoopResult("rescued", text=current.strip(),
                      cuts=tuple(cuts))


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic Lantern Hollow material throughout
# (TF-017's world, never canon). expected_text is
# the exact usable text; cuts are filed as
# (kind, occurrences, partial_units) in order.

_RAMBLING = " ".join(
    ["The river is nice and the river is nice and the river "
     "is nice and"] * 8)

LOOP_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "clean_prose",
     "text": "The ferry leaves at dawn. The tide bell rings "
             "across the harbour. The keeper lights the red "
             "lantern for the last crossing.",
     "verdict": "clean",
     "expected_text": "The ferry leaves at dawn. The tide bell rings "
                      "across the harbour. The keeper lights the red "
                      "lantern for the last crossing.",
     "cuts": ()},
    {"id": "clean_double_sentence_is_emphasis_room",
     "text": "The bell rings. The bell rings.",
     "verdict": "clean",
     "expected_text": "The bell rings. The bell rings.",
     "cuts": ()},
    {"id": "clean_varied_bell_sentences",
     "text": "The bell rings at dawn. The bell rings at dusk. "
             "The bell rings at noon. The bell rings at dawn.",
     "verdict": "clean",
     "expected_text": "The bell rings at dawn. The bell rings at dusk. "
                      "The bell rings at noon. The bell rings at dawn.",
     "cuts": ()},
    {"id": "tf087_loop_fixture",
     "text": " ".join(["The bell rings."] * 20),
     "verdict": "rescued",
     "expected_text": "The bell rings.",
     "cuts": (("sentence_loop", 20, 0),)},
    {"id": "sentence_run_at_threshold",
     "text": "The tide is low. The tide is low. The tide is low.",
     "verdict": "rescued",
     "expected_text": "The tide is low.",
     "cuts": (("sentence_loop", 3, 0),)},
    {"id": "loop_in_the_middle",
     "text": "The ferry leaves at dawn. The bell rings. "
             "The bell rings. The bell rings. The bell rings. "
             "The keeper lights the lantern.",
     "verdict": "rescued",
     "expected_text": "The ferry leaves at dawn. The bell rings. "
                      "The keeper lights the lantern.",
     "cuts": (("sentence_loop", 4, 0),)},
    {"id": "sentence_cycle",
     "text": " ".join(
         ["The ferry leaves at dawn.", "The bell rings once."] * 3),
     "verdict": "rescued",
     "expected_text": "The ferry leaves at dawn. The bell rings once.",
     "cuts": (("sentence_cycle", 3, 0),)},
    {"id": "sentence_cycle_with_partial_tail",
     "text": " ".join(
         ["The ferry leaves at dawn.", "The bell rings once."] * 3
         + ["The ferry leaves at dawn."]),
     "verdict": "rescued",
     "expected_text": "The ferry leaves at dawn. The bell rings once.",
     "cuts": (("sentence_cycle", 3, 1),)},
    {"id": "case_and_punctuation_variants_still_a_loop",
     "text": "The bell rings. the bell rings! THE BELL RINGS.",
     "verdict": "rescued",
     "expected_text": "The bell rings.",
     "cuts": (("sentence_loop", 3, 0),)},
    {"id": "two_loops_in_one_draft",
     "text": "The bell rings. The bell rings. The bell rings. "
             "The ferry leaves at dawn. The tide is low. "
             "The tide is low. The tide is low. The tide is low.",
     "verdict": "rescued",
     "expected_text": "The bell rings. The ferry leaves at dawn. "
                      "The tide is low.",
     "cuts": (("sentence_loop", 3, 0), ("sentence_loop", 4, 0))},
    {"id": "word_loop_no_punctuation",
     "text": "The bell rings the bell rings the bell rings "
             "the bell rings",
     "verdict": "rescued",
     "expected_text": "The bell rings",
     "cuts": (("word_loop", 4, 0),)},
    {"id": "word_loop_mid_sentence",
     "text": "The keeper listened while the bell rings "
             "the bell rings the bell rings the bell rings "
             "echoed over the harbour.",
     "verdict": "rescued",
     "expected_text": "The keeper listened while the bell rings "
                      "echoed over the harbour.",
     "cuts": (("word_loop", 4, 0),)},
    {"id": "word_loop_with_truncated_tail_word",
     "text": "the bell rings the bell rings the bell rings "
             "the bell rings the bel",
     "verdict": "rescued",
     "expected_text": "the bell rings",
     "cuts": (("word_loop", 4, 2),)},
    {"id": "single_word_stutter",
     "text": "The bell rang and and and and and echoed.",
     "verdict": "rescued",
     "expected_text": "The bell rang and echoed.",
     "cuts": (("word_loop", 5, 0),)},
    {"id": "rambling_loop_shaped_subset",
     "text": _RAMBLING,
     "verdict": "rescued",
     "expected_text": "The river is nice and",
     "cuts": (("word_loop", 24, 0),)},
    {"id": "refuse_empty",
     "text": "   ",
     "verdict": "refused",
     "expected_text": None,
     "cuts": (),
     "problems": ("empty_output",)},
)


# ── Checkers ────────────────────────────────────────────────────────────────


def loop_rescue_problems() -> list[str]:
    """Data-integrity checker: rule and fixture ids
    unique, thresholds sane, every fixture
    well-formed, refusals naming their problems,
    rescues naming their cuts."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in LOOP_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate LOOP_RULES ids")
    fixture_ids = [f["id"] for f in LOOP_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate LOOP_FIXTURES ids")
    if LOOP_CONSTANTS["sentence_min_repeats"] < 2:
        problems.append("sentence_min_repeats below 2 would "
                        "cut ordinary emphasis")
    if LOOP_CONSTANTS["word_min_repeats"] < 3:
        problems.append("word_min_repeats below 3 would "
                        "cut ordinary phrasing")
    if LOOP_CONSTANTS["word_min_period"] < 1 or \
            LOOP_CONSTANTS["word_max_period"] < \
            LOOP_CONSTANTS["word_min_period"]:
        problems.append("word period bounds are incoherent")
    for fixture in LOOP_FIXTURES:
        if fixture["verdict"] not in ("clean", "rescued", "refused"):
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{fixture['verdict']!r} not in the vocabulary")
        if fixture["verdict"] == "refused" and \
                not fixture.get("problems"):
            problems.append(
                f"fixture {fixture['id']}: a refusal must "
                "name its problems")
        if fixture["verdict"] == "rescued" and not fixture["cuts"]:
            problems.append(
                f"fixture {fixture['id']}: a rescue must "
                "name its cuts")
        if fixture["verdict"] == "clean" and fixture["cuts"]:
            problems.append(
                f"fixture {fixture['id']}: a clean verdict "
                "carries cuts")
    return problems


def loop_rescue_live_problems() -> list[str]:
    """Live checker: every filed fixture is run
    through the real gate and must land exactly as
    filed (verdict, usable text, cut kinds / counts
    in order); a rescued text must be a deletion
    of its source (every kept word is the source's
    own, in order) and must judge clean on a
    second pass (the rescue is complete, not a
    first instalment)."""
    problems: list[str] = []
    for fixture in LOOP_FIXTURES:
        result = rescue_loops(fixture["text"])
        if result.verdict != fixture["verdict"]:
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{result.verdict} != {fixture['verdict']}")
            continue
        if result.text != fixture["expected_text"]:
            problems.append(
                f"fixture {fixture['id']}: text "
                f"{result.text!r} != {fixture['expected_text']!r}")
        filed_cuts = tuple(
            (kind, occurrences, partial)
            for kind, occurrences, partial in fixture["cuts"])
        live_cuts = tuple(
            (cut.kind, cut.occurrences, cut.partial_units)
            for cut in result.cuts)
        if live_cuts != filed_cuts:
            problems.append(
                f"fixture {fixture['id']}: cuts "
                f"{live_cuts} != {filed_cuts}")
        if fixture["verdict"] == "refused":
            if result.text is not None:
                problems.append(
                    f"fixture {fixture['id']}: a refusal "
                    "carries usable text")
            if tuple(result.problems) != tuple(fixture["problems"]):
                problems.append(
                    f"fixture {fixture['id']}: problems "
                    f"{result.problems} != {fixture['problems']}")
        if fixture["verdict"] == "clean" and \
                result.text != fixture["text"]:
            problems.append(
                f"fixture {fixture['id']}: a clean verdict "
                "is not byte-identical to its source")
        if fixture["verdict"] == "rescued":
            for cut in result.cuts:
                if cut.dropped_text not in fixture["text"] and \
                        cut.kind != "word_loop":
                    problems.append(
                        f"fixture {fixture['id']}: dropped text "
                        "is not a verbatim span of the source")
            second = rescue_loops(result.text or "")
            if second.verdict != "clean" or \
                    second.text != result.text:
                problems.append(
                    f"fixture {fixture['id']}: the rescued "
                    "text does not judge clean on a second pass")
    return problems


def loop_as_data() -> dict[str, Any]:
    return {
        "format": LOOP_FORMAT,
        "loop_doc": LOOP_DOC,
        "constants": dict(LOOP_CONSTANTS),
        "rules": [dict(r) for r in LOOP_RULES],
        "fixtures": [dict(f) for f in LOOP_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_loops": "not_proven",
            "salvage_py": "unchanged_by_design",
        },
    }
