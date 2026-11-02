"""TF-101 — Enforcement + chaos: the full policy
stack run over TF-100's chaos corpus, outcomes
scored (rescued / refused / passed-clean).

Roadmap: "Enforcement + chaos: the full policy
stack run over TF-100; outcomes scored
(rescued / refused / passed-clean)."

Measured start (probed live this tick, every
corpus entry run through the real
``enforce_policy`` before this module existed):

- **No scored run existed.** TF-100 filed the
  corpus as *demands* (38 entries, 13 families,
  expected 12 rescued / 21 refused /
  5 passed-clean) and proved only the cheap legs
  on the individual gates. It deliberately left
  three legs unjudged — draft-side hallucination
  beyond the signalled citation entries, generic
  filler, and the invented reference — for this
  slice's scored stack run to judge. TF-093
  filed the same debt from the other side: a
  live model's behaviour under the unified
  surface was NOT PROVEN, TF-101 owns the stack
  score over TF-100.
- **The stack is TF-093's surface, exactly.**
  ``enforce_policy`` (refinement/policy_surface)
  is the full policy stack: its seven checks
  (format, truncation, loop, drift, citation,
  length, genericness) in execution order, the
  real gates called, signals caller-supplied
  never guessed. This slice adds no check,
  retunes no gate, and edits neither the corpus
  nor any expectation — it *runs* that surface
  over every corpus entry and publishes the
  score it earns.
- **Signal resolution is TF-100's own pattern,
  reused.** A corpus ``records`` signal naming
  ``lantern_hollow_synthetic`` resolves to
  TF-093's filed synthetic record pair exactly
  as TF-100's live checker resolves it
  (``policy_surface._fixture_records()`` — the
  Odin Q123 entity and the lantern-oil chunk);
  a ``persona`` signal naming
  ``wrenna_synthetic`` resolves to TF-030's
  ``FIXTURE_EXPECTATION`` exactly as TF-100's
  checker resolves it for ``check_voice``. No
  parallel record or persona set is invented.
- **One signal has nowhere to go — measured,
  not assumed.** ``enforce_policy`` takes no
  persona signal and POLICY_CHECKS names no
  persona check: TF-096's voice guard lives in
  salvage's selection / refine-adoption rules,
  not in the policy surface. The persona signal
  is therefore resolved (and the voice guard's
  own verdict recorded per entry as a
  diagnostic, outside the score), but the stack
  itself cannot consume it. That is a named
  gap below, not a silent skip.

THE MEASURED SCORE (recomputed live by this
module's checker on every suite run; a filed
number the live run cannot reproduce fails by
name):

- **29 of 38 entries land as filed (76.3%).**
  By expected outcome: rescued 12/12, refused
  12/21, passed-clean 5/5. By stack verdict:
  rescued 12, refused 4, flagged 8, clean 14.
- **Every rescue demanded is delivered** (all
  truncation, loop, format-repair and minority-
  drift legs), **every clean control passes
  clean byte-identical** (5/5 — no over-rescue
  anywhere in the run), and the mechanically
  decidable refusal legs land: format refusals,
  drift majority/whole flagged, refusal-shaped
  hard phrases flagged, empty refused by name,
  the two-word answer flagged too short, and
  all three fabricated-citation entries flagged
  (foreign QID; uncited matched claims).
- **All 9 mismatches are one shape**: an entry
  filed *refused* that the stack passes *clean*,
  byte-identical — hallucination 4/4,
  persona-voice loss 3/3, genericness 2/2.
  Each family is a named gap with an owner in
  ``NAMED_GAPS``: no policy check judges a
  draft's invented facts beyond TF-049's
  citation matching (TF-088's fabrication guard
  is refine-adoption, pool-relative — it judges
  salvage's steps, not drafts); no policy check
  judges persona voice (above); and the
  genericness leg names only its hard phrase
  list, which fluent filler trips nowhere.
  Nothing was tuned to move these numbers: the
  corpus, its expectations, and every gate
  stand exactly as TF-100/TF-093 filed them.
- **Fabrication rescues: zero.** Of the 7
  hallucination + fabricated_citation entries,
  none earns a stack ``rescued`` verdict. The
  honest companion number is also filed: 4 of
  them (the whole hallucination family) pass
  *clean* — a pass, not a rescue, and scored
  as mismatches above, never dressed as
  enforcement success.
- **One matched entry carries a caveat, filed
  in ``ENTRY_NOTES``:** the invented-reference
  entry lands flagged, which satisfies
  *refused* under TF-100's scoring rule — but
  the flag is ``citation:uncited_claim`` raised
  through an incidental term match against the
  lantern-oil chunk (the claim shares
  *hollow* / *lantern*), not any detection of
  the invented almanac, which no check names.
  It is counted as filed; the mechanism is
  stated so nobody reads it as reference
  verification.

TF-016: no docket entry is owned by TF-101
(the docket names TF-101 only in D-PROF-4's
remainder text, as a future live panel) and
none is added — the totals stand unchanged
at 90. The three gaps above are scored gaps
for TF-104's Part V gate to inherit, filed
here with their owners; they are not docket
entries and no gate is fixed in this slice,
because each fix belongs to its owner's
surface (a draft-side fabrication check,
a persona check in the surface, a wider
genericness judgement), not to a scoring run.

Proof grade FIXTURE — the filed corpus run
live through the real policy surface on this
VM. A live model's score is NOT PROVEN (no
model in tree; the corpus is real-shaped, not
recorded), never claimed.

The human-readable write-up is
``docs/architecture/CHAOS_RUN.md``.
"""

from __future__ import annotations

import copy
from typing import Any

from thoughtforge.refinement.chaos_corpus import (
    CHAOS_ENTRIES,
    FAMILIES,
    KNOWN_OWNERS,
    SIGNAL_PERSONAS,
    SIGNAL_RECORD_SETS,
)
from thoughtforge.refinement.policy_surface import (
    _fixture_records,
    enforce_policy,
)

#: This run's estate format.
CHAOS_RUN_FORMAT = "thoughtforge-chaos-run/1"

CHAOS_RUN_DOC = "docs/architecture/CHAOS_RUN.md"

#: Stack verdicts TF-093's surface can return.
STACK_VERDICTS: tuple[str, ...] = (
    "clean", "rescued", "flagged", "refused")

#: The scored vocabulary (TF-100's outcomes).
SCORED_OUTCOMES: tuple[str, ...] = (
    "rescued", "refused", "passed-clean")

# ── The scoring rules ───────────────────────────────────────────────────────

SCORING_RULES: tuple[dict[str, Any], ...] = (
    {"id": "the_stack_is_tf093s_surface_exactly",
     "rule": "every corpus entry is run through "
             "TF-093's real enforce_policy with the "
             "entry's own signals resolved; no check "
             "is added, removed, reordered or retuned "
             "for this run, and no other gate's "
             "verdict is substituted for the stack's"},
    {"id": "signals_resolved_by_tf100s_pattern",
     "rule": "a records signal naming a filed "
             "synthetic set resolves to TF-093's "
             "_fixture_records() and a persona signal "
             "naming a filed synthetic persona resolves "
             "to TF-030's FIXTURE_EXPECTATION, exactly "
             "as TF-100's live checker resolves them; "
             "structured / required_keys / "
             "expected_format / truncation_suspect / "
             "expected_language / retrieved_qids pass "
             "through in TF-093's own parameter shapes"},
    {"id": "rescued_requires_a_stack_rescue",
     "rule": "expected rescued is satisfied only by "
             "a stack verdict of rescued — a repair or "
             "deletion-only cut was applied and every "
             "judgement on the final text passed"},
    {"id": "refused_satisfied_by_refused_or_blocking_flagged",
     "rule": "expected refused is satisfied by a "
             "stack verdict of refused or flagged "
             "(TF-100's filed scoring rule: a blocking "
             "flag — drift majority/whole, citation, "
             "length, genericness — means the surface "
             "will not pass the text clean); a clean "
             "verdict never satisfies refused"},
    {"id": "passed_clean_requires_clean_and_byte_identical",
     "rule": "expected passed-clean is satisfied "
             "only by a stack verdict of clean whose "
             "returned text is byte-identical to the "
             "corpus text; a rescue or annotation "
             "applied to a clean control is a "
             "mismatch (over-rescue is failure)"},
    {"id": "a_clean_stack_verdict_on_a_broken_entry_is_a_mismatch",
     "rule": "an entry whose expected outcome is "
             "rescued or refused but whose stack "
             "verdict is clean is scored mismatched "
             "with its actual outcome recorded as "
             "passed-clean — the mapping names what "
             "the stack did, never what was demanded"},
    {"id": "mismatches_are_named_gaps_never_tuned_away",
     "rule": "every mismatched entry is listed "
             "explicitly with its reason and belongs "
             "to exactly one NAMED_GAPS entry with an "
             "owner; neither the corpus, its "
             "expectations, nor any gate is edited "
             "to move the score in this slice"},
    {"id": "persona_signal_resolved_but_not_consumed_by_the_stack",
     "rule": "the persona signal is resolved to "
             "TF-030's expectation and TF-096's "
             "check_voice verdict on the entry (against "
             "its filed source_text, TF-100's exact "
             "call) is recorded as a diagnostic outside "
             "the score, because enforce_policy takes "
             "no persona signal and POLICY_CHECKS names "
             "no persona check; the diagnostic never "
             "changes the entry's scored outcome"},
)

# ── The filed score ─────────────────────────────────────────────────────────
# Every value is recomputed live by
# chaos_run_live_problems(); a filed value the
# live run cannot reproduce fails by name.

FILED_SCORE: dict[str, Any] = {
    "entries": 38,
    "matched": 29,
    "mismatched": 9,
    "score": 29 / 38,
    "by_family": {
        "truncation": {"entries": 3, "matched": 3},
        "loop_sentence": {"entries": 3, "matched": 3},
        "loop_cycle": {"entries": 2, "matched": 2},
        "loop_word": {"entries": 2, "matched": 2},
        "hallucination": {"entries": 4, "matched": 0},
        "fabricated_citation": {"entries": 3, "matched": 3},
        "format_collapse": {"entries": 4, "matched": 4},
        "language_drift": {"entries": 3, "matched": 3},
        "persona_voice_loss": {"entries": 3, "matched": 0},
        "genericness": {"entries": 2, "matched": 0},
        "refusal_shaped": {"entries": 2, "matched": 2},
        "empty_or_short": {"entries": 2, "matched": 2},
        "clean_control": {"entries": 5, "matched": 5},
    },
    "by_expected_outcome": {
        "rescued": {"entries": 12, "matched": 12},
        "refused": {"entries": 21, "matched": 12},
        "passed-clean": {"entries": 5, "matched": 5},
    },
    "stack_verdicts": {
        "clean": 14, "rescued": 12, "flagged": 8, "refused": 4},
    "fabrication_entries": 7,
    "fabrication_rescues": 0,
    "fabrication_passed_clean": 4,
    "clean_controls": 5,
    "clean_controls_byte_identical": 5,
}

#: The filed per-entry stack outcomes, in corpus
#: order: (entry id, stack verdict, scored
#: actual outcome, matched). The live checker
#: re-runs every entry and compares all four.
FILED_RESULTS: tuple[dict[str, Any], ...] = (
    {"id": "chaos_truncation_cut_mid_sentence",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_truncation_cut_mid_word",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_truncation_no_complete_sentence",
     "stack_verdict": "refused", "actual": "refused",
     "matched": True},
    {"id": "chaos_loop_sentence_run",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_loop_sentence_in_the_middle",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_loop_sentence_case_variants",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_loop_cycle_two_sentence",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_loop_cycle_with_partial_tail",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_loop_word_repeat",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_loop_word_stutter",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_hallucination_invented_price",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": False},
    {"id": "chaos_hallucination_invented_destinations",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": False},
    {"id": "chaos_hallucination_negation_flip",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": False},
    {"id": "chaos_hallucination_common_word_elaboration",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": False},
    {"id": "chaos_citation_foreign_qid",
     "stack_verdict": "flagged", "actual": "refused",
     "matched": True},
    {"id": "chaos_citation_invented_reference",
     "stack_verdict": "flagged", "actual": "refused",
     "matched": True},
    {"id": "chaos_citation_uncited_claim",
     "stack_verdict": "flagged", "actual": "refused",
     "matched": True},
    {"id": "chaos_format_trailing_comma",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_format_truncated_json",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_format_prose_when_json_asked",
     "stack_verdict": "refused", "actual": "refused",
     "matched": True},
    {"id": "chaos_format_missing_required_key",
     "stack_verdict": "refused", "actual": "refused",
     "matched": True},
    {"id": "chaos_drift_minority_spanish",
     "stack_verdict": "rescued", "actual": "rescued",
     "matched": True},
    {"id": "chaos_drift_majority_spanish",
     "stack_verdict": "flagged", "actual": "refused",
     "matched": True},
    {"id": "chaos_drift_whole_answer_spanish",
     "stack_verdict": "flagged", "actual": "refused",
     "matched": True},
    {"id": "chaos_voice_handwaving",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": False},
    {"id": "chaos_voice_repetition",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": False},
    {"id": "chaos_voice_invented_crossings",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": False},
    {"id": "chaos_generic_filler_non_answer",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": False},
    {"id": "chaos_generic_vague_restatement",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": False},
    {"id": "chaos_refusal_cannot_provide",
     "stack_verdict": "flagged", "actual": "refused",
     "matched": True},
    {"id": "chaos_refusal_unable_to_help",
     "stack_verdict": "flagged", "actual": "refused",
     "matched": True},
    {"id": "chaos_empty_output",
     "stack_verdict": "refused", "actual": "refused",
     "matched": True},
    {"id": "chaos_short_non_answer",
     "stack_verdict": "flagged", "actual": "refused",
     "matched": True},
    {"id": "chaos_clean_prose",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": True},
    {"id": "chaos_clean_structured_json",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": True},
    {"id": "chaos_clean_cited_claim",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": True},
    {"id": "chaos_clean_emphasis_double_repeat",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": True},
    {"id": "chaos_clean_spanish_answer",
     "stack_verdict": "clean", "actual": "passed-clean",
     "matched": True},
)

# ── The named gaps ──────────────────────────────────────────────────────────
# Every mismatched entry belongs to exactly one
# gap; each gap names the mechanism that is
# absent and the owner of the surface where a
# fix would belong. TF-104's gate inherits
# these as the scored gaps of Part V.

NAMED_GAPS: tuple[dict[str, Any], ...] = (
    {"id": "draft_side_hallucination_not_judged_by_the_policy_surface",
     "owner": "TF-088",
     "entries": (
         "chaos_hallucination_invented_price",
         "chaos_hallucination_invented_destinations",
         "chaos_hallucination_negation_flip",
         "chaos_hallucination_common_word_elaboration"),
     "gap": "no check in TF-093's surface judges a "
            "draft's invented facts as facts: the "
            "citation check (TF-049) flags only claims "
            "matching a supplied record uncited, or "
            "foreign markers — three of these entries "
            "supply no records signal at all, and the "
            "fourth (the negation flip) supplies "
            "records carrying no ferry fact for its "
            "claim to match, so its citation check "
            "passes. TF-088's fabrication guard, the "
            "entries' owner mechanism, judges salvage "
            "refine steps pool-relatively; it is not "
            "a draft gate and the surface never calls "
            "it. A draft-side fabrication judgement "
            "is the missing mechanism"},
    {"id": "persona_voice_not_a_policy_surface_check",
     "owner": "TF-096",
     "entries": (
         "chaos_voice_handwaving",
         "chaos_voice_repetition",
         "chaos_voice_invented_crossings"),
     "gap": "TF-093's POLICY_CHECKS names no persona "
            "check and enforce_policy takes no persona "
            "signal, so the resolved persona is never "
            "judged by the stack: all three outputs "
            "pass clean while TF-096's own check_voice "
            "(recorded here as a diagnostic, TF-100's "
            "exact call against each entry's "
            "source_text) names a new avoid violation "
            "in every one. TF-096's guard lives in "
            "salvage's selection / refine adoption; "
            "a persona leg in the policy surface is "
            "the missing mechanism"},
    {"id": "genericness_checks_hard_phrases_only",
     "owner": "TF-093",
     "entries": (
         "chaos_generic_filler_non_answer",
         "chaos_generic_vague_restatement"),
     "gap": "the surface's genericness leg is "
            "EnforcementGate's hard-phrase list and "
            "its length leg is a five-word floor: "
            "fluent filler that answers nothing and "
            "a question restated as its own answer "
            "trip no listed phrase and pass the "
            "floor, so both pass clean. A judgement "
            "of answerhood beyond the phrase list "
            "is the missing mechanism"},
)

#: Filed caveats on matched entries whose
#: mechanism is weaker than the outcome suggests.
ENTRY_NOTES: dict[str, str] = {
    "chaos_citation_invented_reference": (
        "counted matched per the filed scoring rule "
        "(a blocking flagged verdict satisfies "
        "refused), but the flag is "
        "citation:uncited_claim raised through an "
        "incidental term match against the "
        "lantern-oil reference chunk (shared terms "
        "hollow / lantern): the invented Hollow "
        "Almanac itself is named by no check — this "
        "is not reference verification"),
}

_GAP_BY_ENTRY = {
    entry_id: gap for gap in NAMED_GAPS
    for entry_id in gap["entries"]}

# ── Signal resolution (TF-100's pattern) ────────────────────────────────────


def resolve_entry_signals(
        entry: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Resolve one corpus entry's signals exactly
    as TF-100's checker/tests do: the returned
    dict is ``enforce_policy`` kwargs in TF-093's
    own parameter shapes, and the second value is
    the resolved persona expectation (TF-030's
    ``FIXTURE_EXPECTATION``) or None. The persona
    is resolved, never passed to the surface —
    the surface takes no persona signal (filed
    rule ``persona_signal_resolved_but_not_``
    ``consumed_by_the_stack``)."""
    signals = entry["signals"]
    kwargs: dict[str, Any] = {}
    if signals.get("structured"):
        kwargs["structured"] = True
    if "required_keys" in signals:
        kwargs["required_keys"] = tuple(signals["required_keys"])
    if "expected_format" in signals:
        kwargs["expected_format"] = signals["expected_format"]
    if signals.get("truncation_suspect"):
        kwargs["truncation_suspect"] = True
    if "expected_language" in signals:
        kwargs["expected_language"] = signals["expected_language"]
    if "records" in signals:
        if signals["records"] not in SIGNAL_RECORD_SETS:
            raise ValueError(
                f"{entry['id']}: records names a "
                f"non-synthetic set {signals['records']!r}")
        kwargs["records"] = _fixture_records()
    if "retrieved_qids" in signals:
        kwargs["retrieved_qids"] = list(signals["retrieved_qids"])
    persona = None
    if "persona" in signals:
        if signals["persona"] not in SIGNAL_PERSONAS:
            raise ValueError(
                f"{entry['id']}: persona names a "
                f"non-synthetic persona {signals['persona']!r}")
        from thoughtforge.cognition.persona import (
            FIXTURE_EXPECTATION,
        )

        persona = FIXTURE_EXPECTATION
    return kwargs, persona


def mapped_actual(stack_verdict: str,
                  output_text: str | None,
                  original_text: str) -> str:
    """Map a stack verdict into TF-100's scored
    vocabulary per the filed scoring rules: a
    flagged verdict maps to refused (blocking),
    a clean verdict maps to passed-clean only
    when the returned text is byte-identical —
    otherwise it maps to clean-modified, a
    diagnostic label outside the vocabulary
    that can satisfy nothing."""
    if stack_verdict == "rescued":
        return "rescued"
    if stack_verdict in ("refused", "flagged"):
        return "refused"
    if stack_verdict == "clean":
        return "passed-clean" if output_text == original_text \
            else "clean-modified"
    raise ValueError(f"unknown stack verdict {stack_verdict!r}")


def is_match(expected: str, stack_verdict: str,
             output_text: str | None,
             original_text: str) -> bool:
    """Whether a live stack outcome satisfies an
    expected outcome, per SCORING_RULES."""
    if expected == "rescued":
        return stack_verdict == "rescued"
    if expected == "refused":
        return stack_verdict in ("refused", "flagged")
    if expected == "passed-clean":
        return stack_verdict == "clean" and \
            output_text == original_text
    raise ValueError(f"unknown expected outcome {expected!r}")


def _reason(entry: dict[str, Any], stack_verdict: str,
            matched: bool, flags: tuple[str, ...],
            problems: tuple[str, ...]) -> str:
    expected = entry["expected_outcome"]
    if matched and expected == "rescued":
        return "stack verdict rescued satisfies expected rescued"
    if matched and expected == "refused" and stack_verdict == "refused":
        return ("stack verdict refused (problems: "
                f"{', '.join(problems) or 'none named'}) "
                "satisfies expected refused")
    if matched and expected == "refused" and stack_verdict == "flagged":
        return ("stack verdict flagged (flags: "
                f"{', '.join(flags) or 'none named'}) is "
                "blocking and satisfies expected refused "
                "per TF-100's scoring rule")
    if matched and expected == "passed-clean":
        return ("stack verdict clean with byte-identical "
                "text satisfies expected passed-clean")
    gap = _GAP_BY_ENTRY.get(entry["id"])
    gap_id = gap["id"] if gap is not None else "unnamed"
    return (f"expected {expected} but the stack verdict "
            f"is {stack_verdict} — gap {gap_id}: "
            f"{gap['gap'] if gap is not None else 'no gap filed'}")


# ── The run ─────────────────────────────────────────────────────────────────


def run_chaos_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Run one corpus entry through the real
    policy stack and score it. The persona
    diagnostic (persona entries only) is
    TF-100's exact ``check_voice`` call, recorded
    outside the score."""
    kwargs, persona = resolve_entry_signals(entry)
    result = enforce_policy(entry["text"], **kwargs)
    matched = is_match(
        entry["expected_outcome"], result.verdict,
        result.text, entry["text"])
    actual = mapped_actual(
        result.verdict, result.text, entry["text"])
    diagnostic = None
    if persona is not None:
        from thoughtforge.refinement.persona_voice import (
            check_voice,
        )

        voice = check_voice(
            persona, entry["source_text"], entry["text"])
        diagnostic = {
            "verdict": voice.verdict,
            "new_violations": list(voice.new_violations),
            "scored": False,
        }
    gap = _GAP_BY_ENTRY.get(entry["id"])
    return {
        "id": entry["id"],
        "family": entry["family"],
        "owner": entry["owner"],
        "expected": entry["expected_outcome"],
        "stack_verdict": result.verdict,
        "actual": actual,
        "matched": matched,
        "flags": list(result.flags),
        "problems": list(result.problems),
        "checks": [
            {"id": outcome.id, "status": outcome.status}
            for outcome in result.checks],
        "output_text": result.text,
        "byte_identical": result.text == entry["text"],
        "gap": None if matched else (gap["id"] if gap else None),
        "gap_owner": None if matched else (
            gap["owner"] if gap else None),
        "reason": _reason(
            entry, result.verdict, matched,
            result.flags, result.problems),
        "note": ENTRY_NOTES.get(entry["id"]),
        "persona_voice_diagnostic": diagnostic,
    }


def score_results(
        results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scored rows into the score shape
    FILED_SCORE files (per family in TF-100's
    FAMILIES order, per expected outcome, per
    stack verdict, and the fabrication counts)."""
    by_family: dict[str, dict[str, int]] = {
        family: {"entries": 0, "matched": 0} for family in FAMILIES}
    by_expected: dict[str, dict[str, int]] = {
        outcome: {"entries": 0, "matched": 0}
        for outcome in SCORED_OUTCOMES}
    verdicts = {verdict: 0 for verdict in STACK_VERDICTS}
    fabrication = [
        r for r in results
        if r["family"] in ("hallucination", "fabricated_citation")]
    controls = [r for r in results if r["family"] == "clean_control"]
    matched = sum(1 for r in results if r["matched"])
    aggregates = _fill_aggregates(
        results, by_family, by_expected, verdicts)
    return {
        "entries": len(results),
        "matched": matched,
        "mismatched": len(results) - matched,
        "score": matched / len(results) if results else 0.0,
        "by_family": {
            family: by_family[family] for family in FAMILIES},
        **aggregates,
        "fabrication_entries": len(fabrication),
        "fabrication_rescues": sum(
            1 for r in fabrication if r["stack_verdict"] == "rescued"),
        "fabrication_passed_clean": sum(
            1 for r in fabrication if r["stack_verdict"] == "clean"),
        "clean_controls": len(controls),
        "clean_controls_byte_identical": sum(
            1 for r in controls if r["byte_identical"]),
    }


def _fill_aggregates(results, by_family, by_expected, verdicts):
    for row in results:
        by_family[row["family"]]["entries"] += 1
        by_family[row["family"]]["matched"] += 1 if row["matched"] else 0
        by_expected[row["expected"]]["entries"] += 1
        by_expected[row["expected"]]["matched"] += \
            1 if row["matched"] else 0
        verdicts[row["stack_verdict"]] += 1
    return {
        "by_expected_outcome": by_expected,
        "stack_verdicts": verdicts,
    }


def run_chaos() -> dict[str, Any]:
    """The full scored run, live: every TF-100
    corpus entry (loaded by TF-100's module from
    its JSON file, never hardcoded here) through
    the real TF-093 stack, with the aggregate
    score and the explicit mismatch list."""
    results = [run_chaos_entry(entry) for entry in CHAOS_ENTRIES]
    return {
        "results": results,
        "score": score_results(results),
        "mismatches": [
            {"id": r["id"], "family": r["family"],
             "expected": r["expected"], "actual": r["actual"],
             "stack_verdict": r["stack_verdict"],
             "gap": r["gap"], "gap_owner": r["gap_owner"],
             "reason": r["reason"]}
            for r in results if not r["matched"]],
    }


# ── Checkers ────────────────────────────────────────────────────────────────


def chaos_run_problems(
        filed_score: dict[str, Any] | None = None,
        filed_results: tuple[dict[str, Any], ...] | None = None,
        gaps: tuple[dict[str, Any], ...] | None = None) -> list[str]:
    """Data-integrity checker: rule ids unique;
    the filed results cover the corpus exactly
    once, in corpus order, with verdicts and
    actuals in their vocabularies and matched
    flags consistent with the filed scoring
    rules; the filed score's totals, family and
    outcome splits, verdict counts and
    fabrication counts all agree with the filed
    results and the corpus; every filed mismatch
    belongs to exactly one named gap, every gap
    entry is a filed mismatch, and gap owners
    are known."""
    problems: list[str] = []
    score = FILED_SCORE if filed_score is None else filed_score
    rows = FILED_RESULTS if filed_results is None else filed_results
    gap_rows = NAMED_GAPS if gaps is None else gaps
    rule_ids = [r["id"] for r in SCORING_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate SCORING_RULES ids")
    corpus_ids = [e["id"] for e in CHAOS_ENTRIES]
    if [r["id"] for r in rows] != corpus_ids:
        problems.append(
            "FILED_RESULTS does not cover the corpus "
            "exactly once in corpus order")
    by_id = {e["id"]: e for e in CHAOS_ENTRIES}
    for row in rows:
        entry = by_id.get(row["id"])
        if entry is None:
            continue
        if row["stack_verdict"] not in STACK_VERDICTS:
            problems.append(
                f"{row['id']}: stack verdict "
                f"{row['stack_verdict']!r} not in vocabulary")
        if row["actual"] not in SCORED_OUTCOMES:
            problems.append(
                f"{row['id']}: actual {row['actual']!r} "
                "not in the scored vocabulary")
        want_actual = {
            "rescued": "rescued", "refused": "refused",
            "flagged": "refused", "clean": "passed-clean",
        }.get(row["stack_verdict"])
        if want_actual is not None and row["actual"] != want_actual:
            problems.append(
                f"{row['id']}: actual {row['actual']!r} does "
                f"not map from stack verdict "
                f"{row['stack_verdict']!r}")
        # For non-clean verdicts the matched flag is
        # fully derivable from the filed row (the
        # scoring rules consult only the verdict);
        # for a clean verdict the byte-identical
        # condition cannot be re-derived from filed
        # rows alone — the live checker proves it —
        # so the data checker pins only that a clean
        # match is claimed solely by a passed-clean
        # expectation, and a clean mismatch solely
        # by any other.
        if row["stack_verdict"] != "clean":
            want_match = is_match(
                entry["expected_outcome"], row["stack_verdict"],
                entry["text"], entry["text"])
            if bool(row["matched"]) != want_match:
                problems.append(
                    f"{row['id']}: filed matched flag "
                    "inconsistent with the scoring rules")
        elif bool(row["matched"]) != (
                entry["expected_outcome"] == "passed-clean"):
            problems.append(
                f"{row['id']}: a clean verdict matches "
                "exactly the passed-clean expectations")
    matched = sum(1 for r in rows if r["matched"])
    if score.get("entries") != len(rows):
        problems.append("filed score entries != filed results")
    if score.get("matched") != matched:
        problems.append("filed score matched != filed results")
    if score.get("mismatched") != len(rows) - matched:
        problems.append("filed score mismatched != filed results")
    if score.get("score") != (matched / len(rows) if rows else 0.0):
        problems.append("filed score rate != filed results")
    fam: dict[str, dict[str, int]] = {}
    verdicts = {v: 0 for v in STACK_VERDICTS}
    fabrication_rescues = 0
    fabrication_clean = 0
    fabrication_entries = 0
    for row in rows:
        entry = by_id.get(row["id"])
        if entry is None:
            continue
        slot = fam.setdefault(
            entry["family"], {"entries": 0, "matched": 0})
        slot["entries"] += 1
        slot["matched"] += 1 if row["matched"] else 0
        verdicts[row["stack_verdict"]] = \
            verdicts.get(row["stack_verdict"], 0) + 1
        if entry["family"] in ("hallucination", "fabricated_citation"):
            fabrication_entries += 1
            fabrication_rescues += \
                1 if row["stack_verdict"] == "rescued" else 0
            fabrication_clean += \
                1 if row["stack_verdict"] == "clean" else 0
    if score.get("by_family") != fam:
        problems.append("filed score by_family != filed results")
    if score.get("stack_verdicts") != verdicts:
        problems.append("filed score stack_verdicts != filed results")
    if score.get("fabrication_entries") != fabrication_entries or \
            score.get("fabrication_rescues") != fabrication_rescues or \
            score.get("fabrication_passed_clean") != fabrication_clean:
        problems.append("filed fabrication counts != filed results")
    gap_ids = [g["id"] for g in gap_rows]
    if len(gap_ids) != len(set(gap_ids)):
        problems.append("duplicate NAMED_GAPS ids")
    covered: list[str] = []
    for gap in gap_rows:
        if gap.get("owner") not in KNOWN_OWNERS:
            problems.append(
                f"gap {gap.get('id')}: unknown owner "
                f"{gap.get('owner')!r}")
        if not str(gap.get("gap", "")).strip():
            problems.append(f"gap {gap.get('id')}: empty gap text")
        covered.extend(gap.get("entries", ()))
    if len(covered) != len(set(covered)):
        problems.append("a mismatched entry is covered by two gaps")
    mismatched_ids = {r["id"] for r in rows if not r["matched"]}
    if set(covered) != mismatched_ids:
        problems.append(
            "named gaps do not cover exactly the filed mismatches")
    for entry_id in ENTRY_NOTES:
        if entry_id not in by_id:
            problems.append(f"ENTRY_NOTES names unknown {entry_id}")
    return problems


def chaos_run_live_problems() -> list[str]:
    """Live checker: the full run recomputed live
    must reproduce the filed score and every
    filed per-entry outcome exactly; every clean
    control clean byte-identical; fabrication
    rescues zero as filed; every persona entry's
    diagnostic voice_broken with new violations
    while its stack verdict stands as filed; and
    a second full run byte-identical to the
    first (the run is deterministic)."""
    problems: list[str] = []
    run = run_chaos()
    if run["score"] != FILED_SCORE:
        problems.append(
            f"live score {run['score']} != filed {FILED_SCORE}")
    filed_by_id = {r["id"]: r for r in FILED_RESULTS}
    for row in run["results"]:
        filed = filed_by_id.get(row["id"])
        if filed is None:
            problems.append(f"{row['id']}: no filed result")
            continue
        for key in ("stack_verdict", "actual", "matched"):
            if row[key] != filed[key]:
                problems.append(
                    f"{row['id']}: live {key} {row[key]!r} "
                    f"!= filed {filed[key]!r}")
        if not row["matched"] and not row["reason"]:
            problems.append(f"{row['id']}: mismatch without a reason")
        if row["family"] == "clean_control":
            if row["stack_verdict"] != "clean" or \
                    not row["byte_identical"]:
                problems.append(
                    f"{row['id']}: clean control not clean "
                    "byte-identical in the stack run")
        diagnostic = row["persona_voice_diagnostic"]
        if row["family"] == "persona_voice_loss":
            if diagnostic is None or \
                    diagnostic["verdict"] != "voice_broken" or \
                    not diagnostic["new_violations"]:
                problems.append(
                    f"{row['id']}: persona diagnostic does "
                    "not name a new violation")
        elif diagnostic is not None:
            problems.append(
                f"{row['id']}: persona diagnostic on a "
                "non-persona entry")
    if run["score"]["fabrication_rescues"] != 0:
        problems.append("a fabrication entry was rescued by the stack")
    second = run_chaos()
    if second != run:
        problems.append("the chaos run is not deterministic")
    return problems


def chaos_run_as_data() -> dict[str, Any]:
    run = run_chaos()
    return {
        "format": CHAOS_RUN_FORMAT,
        "chaos_run_doc": CHAOS_RUN_DOC,
        "rules": [dict(r) for r in SCORING_RULES],
        "filed_score": copy.deepcopy(FILED_SCORE),
        "filed_results": copy.deepcopy(list(FILED_RESULTS)),
        "named_gaps": [
            {**dict(g), "entries": list(g["entries"])}
            for g in NAMED_GAPS],
        "entry_notes": dict(ENTRY_NOTES),
        "live_score": run["score"],
        "results": run["results"],
        "mismatches": run["mismatches"],
        "proof": {
            "grade": "fixture",
            "live_model_score": "not_proven",
            "stack": "tf093_enforce_policy_unchanged",
            "corpus": "tf100_chaos_corpus_unchanged",
            "tf016_totals": "unchanged_90",
        },
    }
