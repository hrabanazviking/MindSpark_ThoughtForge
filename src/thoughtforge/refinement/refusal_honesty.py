"""TF-095 — Refusal honesty: when enforcement
blocks an answer, the user learns that and why,
with the salvage path offered.

Roadmap: "Refusal honesty: when enforcement
blocks an answer, the user learns *that and
why*, with the salvage path offered."

Measured start (probed live this tick):

- **A refused answer was silent.** TF-093's
  ``enforce_policy`` and TF-094's
  ``enforce_configured`` return a refused
  verdict as data — ``text=None`` plus a
  ``problems`` tuple of machine codes
  (``empty_output``, ``no_json_span``,
  ``no_complete_sentence``,
  ``missing_required_key: ferry``, …) — and a
  strict block under TF-094 returns the flags
  as problems (``citation:uncited_claim``,
  ``length:too_short``, …). Nothing anywhere
  in the tree turned that record into
  something a user could read: no function
  renders a refusal, no message says an answer
  was blocked, and the codes themselves are
  the only "why" that existed. (A grep for a
  refusal renderer over ``src/`` finds only
  the knowledge estate's internal ``_refuse``
  helper and TF-062's screening probe — neither
  speaks to a user about an answer.)
- **TF-031 is the precedent, not the answer.**
  Cognition's error-honesty module states a
  *generation* failure in words. An enforcement
  block is a different event — generation
  succeeded and enforcement refused to pass
  the result — and TF-090 through TF-094 each
  filed the telling of it to this slice.

What this slice builds — ``refusal_notice`` /
``refusal_message`` in this module, a telling
layer over TF-093's and TF-094's real results:

- **THAT: THE BLOCK IS STATED, FIRST.** Every
  notice opens with ``REFUSAL_MARKER`` and the
  plain sentence that the answer was blocked
  by enforcement and has not been shown. A
  strict block says so in its own words: the
  answer was flagged, and under the strict
  level a flagged answer does not leave.
- **WHY: EVERY PROBLEM, VERBATIM, WITH A
  PLAIN REASON.** Each problem the surface
  recorded is named in the message exactly as
  recorded, beside the plain-language reason
  catalogued for it in ``REASONS`` (as data).
  A problem carrying a detail after ``": "``
  (``missing_required_key: ferry``) is matched
  to its reason by its head, and the whole
  problem — detail included — is still named
  verbatim. A problem with **no catalogued
  reason** is named verbatim anyway and marked
  uncatalogued in the notice: a new code from
  a future gate is never dropped from the
  telling and never given an invented reason.
  A refusal that named **no problem at all**
  is told that way — the gap is stated, never
  filled with an invented reason.
- **THE SALVAGE PATH IS OFFERED, NOT RUN.**
  Each catalogued reason carries the honest
  next step for that problem (regenerate with
  the knowledge in view, ask again in the
  expected language, cut-back only where a
  complete sentence exists, …), and the
  message offers those paths, deduplicated in
  problem order. Offering is all this slice
  does: it never regenerates, never calls
  salvage, never lowers a strictness level,
  and never claims a path was taken.
- **THE BLOCKED TEXT NEVER LEAKS.** The
  telling takes the *result*, never the draft:
  there is no parameter through which blocked
  text could enter the message, so a notice
  cannot quote, excerpt, or paraphrase the
  answer it reports blocking. (Under TF-094 a
  refused / blocked result carries
  ``text=None`` already; this layer keeps the
  guarantee structural rather than trusting
  that.)
- **ONLY A BLOCK GETS A REFUSAL MESSAGE.**
  ``refusal_notice`` on a clean / rescued /
  flagged result raises ``ValueError`` (the
  TF-031 precedent): a refusal message with no
  block behind it would itself be a
  fabrication. A flagged answer under standard
  *leaves* — it is not blocked, and this slice
  does not dress it as one.
- **THE SURFACES ARE UNCHANGED.** TF-093's
  surface, TF-094's configured layer, every
  underlying gate and ``FragmentSalvage`` are
  untouched; this slice reads their results
  and writes words about them. TF-097 owns
  carrying the rescue record into the trace,
  TF-101 the stack score over TF-100, TF-102
  the user-visible quality marks — this slice
  owns only the telling of a block.

Proof grade FIXTURE — every filed case is run
live through the real TF-093 / TF-094 surfaces
on this VM and its notice judged against the
record. A live model's blocked answers are NOT
PROVEN (no model in tree); TF-101 runs the full
stack over TF-100's chaos corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thoughtforge.refinement.policy_surface import PolicyResult

REFUSAL_FORMAT = "thoughtforge-refusal-honesty/1"

REFUSAL_DOC = "docs/architecture/REFUSAL_HONESTY.md"

#: Marker that opens every user-visible refusal
#: message. Its presence is the told-exactly-once
#: witness, in TF-031's shape.
REFUSAL_MARKER = "Blocked by the forge (TF-095):"

# ── The rules ───────────────────────────────────────────────────────────────

REFUSAL_RULES: tuple[dict[str, Any], ...] = (
    {"id": "the_block_is_stated_first",
     "rule": "every refusal message opens with the marker "
             "and the plain statement that the answer was "
             "blocked by enforcement and has not been shown — "
             "the user learns THAT before any reason"},
    {"id": "every_problem_named_verbatim_with_a_reason",
     "rule": "every problem the surface recorded is named "
             "in the message exactly as recorded, beside the "
             "plain-language reason catalogued for it in "
             "REASONS; a problem's ': ' detail rides with it "
             "while the reason is matched by its head"},
    {"id": "uncatalogued_problems_named_never_invented",
     "rule": "a problem with no catalogued reason is named "
             "verbatim and marked uncatalogued, with the "
             "generic offered path — it is never dropped from "
             "the telling and never given an invented reason"},
    {"id": "no_problem_named_is_stated_not_filled",
     "rule": "a refusal that recorded no problem is told "
             "that way: the message states that the surface "
             "named no problem, rather than inventing one"},
    {"id": "the_salvage_path_is_offered_not_run",
     "rule": "the message offers each problem's catalogued "
             "next step, deduplicated in problem order; this "
             "layer never regenerates, never calls salvage, "
             "never changes a strictness level, and never "
             "claims a path was taken"},
    {"id": "blocked_text_never_enters_the_message",
     "rule": "the telling is built from the result record "
             "only — there is no draft parameter — so blocked "
             "text cannot be quoted, excerpted, or paraphrased "
             "in its own refusal message"},
    {"id": "only_a_block_gets_a_refusal_message",
     "rule": "a clean, rescued, or flagged result raises "
             "ValueError: a flagged answer under standard "
             "leaves and is not a block, and a refusal message "
             "with no block behind it would be a fabrication"},
    {"id": "a_strict_block_is_told_as_a_strict_block",
     "rule": "a TF-094 blocked_flagged result is told as "
             "what it is — the answer was flagged, and under "
             "the strict level a flagged answer does not leave "
             "— with the flags named verbatim as its problems"},
    {"id": "the_surfaces_are_unchanged",
     "rule": "TF-093's surface, TF-094's layer, every "
             "underlying gate and FragmentSalvage are "
             "unchanged by this slice; the telling reads "
             "results and writes words about them"},
)

# ── The reasons: every refusal code, in plain words ─────────────────────────
# Keyed by the problem code as recorded, or by its
# head where the recorded problem carries a ": "
# detail (missing_required_key, wrong_top_level_shape).
# ``path`` is the salvage path OFFERED for that
# problem — offered, never run by this slice.

REASONS: dict[str, dict[str, str]] = {
    "empty_output": {
        "reason": "the model produced no answer text at all, "
                  "so there was nothing honest to show",
        "path": "ask the question again so a new answer can "
                "be generated — an empty draft has nothing "
                "to salvage",
    },
    "no_json_span": {
        "reason": "the answer was asked for as structured "
                  "JSON, but the draft contained no JSON at all",
        "path": "ask again for structured output, or drop "
                "the structured requirement and ask in plain "
                "prose instead",
    },
    "multiple_json_spans": {
        "reason": "the draft contained more than one JSON "
                  "block, so no single structured answer "
                  "could be chosen honestly",
        "path": "ask again for one structured answer, or "
                "drop the structured requirement and ask in "
                "plain prose instead",
    },
    "unparseable_json": {
        "reason": "the draft's JSON could not be read, and "
                  "repairing it would have meant inventing "
                  "content",
        "path": "ask again for structured output — the "
                "format gate can repair broken shapes, but "
                "this draft was beyond honest repair",
    },
    "missing_required_key": {
        "reason": "the structured answer is missing a "
                  "required field — the field is named in "
                  "the problem, after the colon",
        "path": "ask again naming the required fields; a "
                "missing field is never invented to complete "
                "the shape",
    },
    "wrong_top_level_shape": {
        "reason": "the structured answer has the wrong "
                  "top-level shape — what was expected and "
                  "what arrived are named in the problem, "
                  "after the colon",
        "path": "ask again for the expected shape, or drop "
                "the structured requirement and ask in plain "
                "prose instead",
    },
    "no_complete_sentence": {
        "reason": "the draft appears cut off mid-sentence "
                  "and holds no complete sentence — "
                  "completing it would mean inventing its "
                  "ending",
        "path": "generate the answer again, or ask for a "
                "shorter one — a cut-off draft can be cut "
                "back only when it holds a complete sentence",
    },
    "unsupported_expected_language": {
        "reason": "the answer was to be judged in a language "
                  "the language checker cannot judge — it "
                  "covers English, Spanish, and German",
        "path": "ask in English, Spanish, or German, or "
                "drop the expected-language requirement for "
                "the answer",
    },
    "citation:uncited_claim": {
        "reason": "the answer states a fact drawn from "
                  "stored knowledge without citing its "
                  "source, and a claim from memory must "
                  "carry its citation",
        "path": "generate the answer again with the "
                "retrieved knowledge in view, so the claim "
                "can carry its citation (TF-093's hand-off)",
    },
    "citation:foreign_marker": {
        "reason": "the answer cites a source marker that "
                  "was not among the retrieved knowledge",
        "path": "generate the answer again with the "
                "retrieved knowledge in view, so only real "
                "sources are cited (TF-093's hand-off)",
    },
    "citation:no_retrieved_qids_cited": {
        "reason": "knowledge was retrieved for this answer "
                  "but the answer cited none of it",
        "path": "generate the answer again with the "
                "retrieved knowledge in view, so what was "
                "used can be cited",
    },
    "length:too_short": {
        "reason": "the answer is shorter than the minimum "
                  "the enforcement gate accepts — too short "
                  "to stand as an answer",
        "path": "generate the answer again, asking for a "
                "fuller one",
    },
    "genericness:hard_generic_phrase": {
        "reason": "the answer leans on a stock generic "
                  "phrase instead of giving a real answer",
        "path": "generate the answer again, asking for a "
                "specific one",
    },
    "drift:whole_answer_off_language": {
        "reason": "the whole answer is in a different "
                  "language from the one the question was "
                  "asked in",
        "path": "generate the answer again in the expected "
                "language — enforcement never translates, "
                "because translating would invent wording",
    },
    "drift:drift_is_majority": {
        "reason": "most of the answer's judged sentences "
                  "are in a different language from the one "
                  "the question was asked in",
        "path": "generate the answer again in the expected "
                "language — enforcement never translates, "
                "because translating would invent wording",
    },
}

UNCATALOGUED_REASON = (
    "no plain-language reason is catalogued for this "
    "problem — it is named exactly as the enforcement "
    "surface recorded it")

UNCATALOGUED_PATH = (
    "ask the question again, or have the caller inspect "
    "the named problem — no specific salvage path is "
    "catalogued for it")

NO_PROBLEM_REASON = (
    "the enforcement surface named no problem for this "
    "block — that gap is stated here rather than filled "
    "with an invented reason")

NO_PROBLEM_PATH = (
    "ask the question again, or have the caller inspect "
    "the block — no problem was recorded to point at a "
    "more specific path")


def _reason_key(problem: str) -> str:
    """The REASONS key for a recorded problem: the
    problem itself, or its head where it carries a
    ``": "`` detail (``missing_required_key: ferry``
    is keyed by ``missing_required_key``). A colon
    without the space is part of the code itself
    (``citation:uncited_claim``) and never splits."""
    return problem.split(": ", 1)[0]


# ── The notice ──────────────────────────────────────────────────────────────


@dataclass
class RefusalNotice:
    """One block, told: that, why, and what next."""
    verdict: str                    # always "refused" here
    problems: tuple[str, ...]       # verbatim, deduplicated, in order
    explanations: tuple[dict[str, Any], ...]  # per problem, in order
    message: str                    # the user-visible telling
    strictness: str | None = None   # TF-094's level, when its result
    disposition: str = "as_judged"  # TF-094's disposition, when its result
    base_verdict: str = "refused"   # the verdict before disposition


def refusal_notice(result: Any) -> RefusalNotice:
    """Build the telling of one blocked answer from
    its result record — a TF-093 ``PolicyResult`` or
    a TF-094 ``ConfiguredResult`` whose verdict is
    ``refused``. The result only: no draft text is
    accepted, so none can leak into the message.

    Raises ``ValueError`` on any non-refused result:
    a refusal message with no block behind it would
    itself be a fabrication."""
    verdict = getattr(result, "verdict", None)
    if verdict != "refused":
        raise ValueError(
            f"no enforcement block to tell: verdict is "
            f"{verdict!r}, not 'refused' — a flagged answer "
            f"under standard leaves and is not a block")
    strictness = getattr(result, "strictness", None)
    disposition = getattr(result, "disposition", "as_judged")
    base_verdict = getattr(result, "base_verdict", "refused")

    problems: list[str] = []
    for problem in getattr(result, "problems", ()) or ():
        if problem not in problems:
            problems.append(problem)

    explanations: list[dict[str, Any]] = []
    for problem in problems:
        entry = REASONS.get(_reason_key(problem))
        if entry is None:
            explanations.append({
                "problem": problem, "key": _reason_key(problem),
                "catalogued": False,
                "reason": UNCATALOGUED_REASON,
                "path": UNCATALOGUED_PATH})
        else:
            explanations.append({
                "problem": problem, "key": _reason_key(problem),
                "catalogued": True,
                "reason": entry["reason"], "path": entry["path"]})

    if disposition == "blocked_flagged":
        opening = (
            f"{REFUSAL_MARKER} this answer was flagged by "
            f"enforcement, and under the {strictness} level a "
            f"flagged answer does not leave — it has been "
            f"blocked and has not been shown.")
    else:
        opening = (
            f"{REFUSAL_MARKER} this answer was blocked by "
            f"enforcement and has not been shown.")
    if explanations:
        why = " Why: " + "; ".join(
            f"{e['problem']} — {e['reason']}" for e in explanations) + "."
        paths: list[str] = []
        for e in explanations:
            if e["path"] not in paths:
                paths.append(e["path"])
        offered = (" What can happen next: " +
                   "; ".join(paths) + ".")
    else:
        why = f" Why: {NO_PROBLEM_REASON}."
        offered = f" What can happen next: {NO_PROBLEM_PATH}."
    closing = (" No part of the blocked answer is included "
               "in this message.")
    return RefusalNotice(
        verdict="refused", problems=tuple(problems),
        explanations=tuple(explanations),
        message=opening + why + offered + closing,
        strictness=strictness, disposition=disposition,
        base_verdict=base_verdict)


def refusal_message(result: Any) -> str:
    """The user-visible telling of one block, as text."""
    return refusal_notice(result).message


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic material throughout (TF-093 / TF-094's
# filed cases, reused — never canon). ``source``
# names the surface whose live result the case
# builds its notice from; the two ``synthetic``
# cases construct a PolicyResult directly, to pin
# the uncatalogued-code and no-problem legs that
# no shipped gate currently produces.

_EN1 = "The ferry leaves at dawn and the tide is low."
_ES = ("El faro está en la costa y la marea es alta. "
       "La campana suena cuando el barco llega a la costa.")
_ODIN_UNCITED = ("Odin is the chief god of Norse mythology, "
                 "a god of wisdom and war.")

REFUSAL_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "empty_refused",
     "source": "policy",
     "text": "   ",
     "signals": {},
     "problems": ("empty_output",),
     "catalogued": True},
    {"id": "format_refused_no_json_span",
     "source": "policy",
     "text": "The ferry leaves at dawn.",
     "signals": {"structured": True},
     "problems": ("no_json_span",),
     "catalogued": True},
    {"id": "missing_required_key_refused",
     "source": "policy",
     "text": '{"tide": "low"}',
     "signals": {"structured": True, "required_keys": ("ferry",)},
     "problems": ("missing_required_key: ferry",),
     "catalogued": True},
    {"id": "truncation_refused_no_complete_sentence",
     "source": "policy",
     "text": "The keeper lights the lantern when",
     "signals": {"truncation_suspect": True},
     "problems": ("no_complete_sentence",),
     "catalogued": True},
    {"id": "unsupported_language_refused",
     "source": "policy",
     "text": _EN1,
     "signals": {"expected_language": "fr"},
     "problems": ("unsupported_expected_language",),
     "catalogued": True},
    {"id": "strict_blocked_citation",
     "source": "configured",
     "text": _ODIN_UNCITED,
     "signals": {"records": "lantern"},
     "strictness": "strict",
     "problems": ("citation:uncited_claim",),
     "catalogued": True},
    {"id": "strict_blocked_drift",
     "source": "configured",
     "text": _ES,
     "signals": {"expected_language": "en"},
     "strictness": "strict",
     "problems": ("drift:whole_answer_off_language",),
     "catalogued": True},
    {"id": "strict_blocked_two_flags",
     "source": "configured",
     "text": "Odin is chief.",
     "signals": {"records": "lantern"},
     "strictness": "strict",
     "problems": ("citation:uncited_claim", "length:too_short"),
     "catalogued": True},
    {"id": "unknown_problem_uncatalogued",
     "source": "synthetic",
     "problems": ("some_future_problem",),
     "catalogued": False},
    {"id": "refused_no_problem_named",
     "source": "synthetic",
     "problems": (),
     "catalogued": True},
)


def _fixture_records() -> list:
    from thoughtforge.knowledge.models import ActivatedRecord

    return [
        ActivatedRecord(
            "Q123", "entity", 0.9, "Odin chief god of Norse mythology",
            raw={"qid": "Q123", "label_en": "Odin",
                 "description_en": "chief god of Norse mythology",
                 "source": "wikidata"}),
        ActivatedRecord(
            "chunk_7", "reference_chunk", 0.8,
            "lantern oil kept in the hollow stores",
            raw={"chunk_id": "chunk_7", "source_file": "hollow.md",
                 "title": "The Hollow",
                 "content": "lantern oil is kept in the hollow stores",
                 "source": "reference"}),
    ]


def _run_fixture(fixture: dict[str, Any]) -> Any:
    if fixture["source"] == "synthetic":
        return PolicyResult(
            "refused", problems=tuple(fixture["problems"]))
    signals = dict(fixture["signals"])
    if signals.get("records") == "lantern":
        signals["records"] = _fixture_records()
    if fixture["source"] == "configured":
        from thoughtforge.refinement.policy_config import (
            enforce_configured,
        )
        return enforce_configured(
            fixture["text"], strictness=fixture["strictness"],
            **signals)
    from thoughtforge.refinement.policy_surface import enforce_policy

    return enforce_policy(fixture["text"], **signals)


# ── Checkers ────────────────────────────────────────────────────────────────


def _estate_problem_codes() -> set[str]:
    """Every problem / blocked-flag code the shipped
    estate can put on a refused result, collected
    from the filed fixtures of TF-090–TF-094 —
    the coverage the REASONS catalogue must hold."""
    from thoughtforge.refinement.drift_rescue import DRIFT_FIXTURES
    from thoughtforge.refinement.format_enforcement import (
        FORMAT_FIXTURES,
        TRUNCATION_FIXTURES,
    )
    from thoughtforge.refinement.loop_rescue import LOOP_FIXTURES
    from thoughtforge.refinement.policy_config import CONFIG_FIXTURES
    from thoughtforge.refinement.policy_surface import POLICY_FIXTURES

    codes: set[str] = set()
    for fixture in POLICY_FIXTURES:
        codes.update(fixture.get("problems") or ())
    for fixture in CONFIG_FIXTURES:
        codes.update(fixture.get("problems") or ())
        if fixture["base_verdict"] == "flagged":
            codes.update(fixture.get("flags") or ())
    for fixture in FORMAT_FIXTURES + TRUNCATION_FIXTURES:
        codes.update(fixture.get("problems") or ())
    for fixture in DRIFT_FIXTURES:
        codes.update(fixture.get("problems") or ())
        codes.update(
            f"drift:{flag}" for flag in fixture.get("flags") or ())
    for fixture in LOOP_FIXTURES:
        codes.update(fixture.get("problems") or ())
    return codes


def refusal_honesty_problems() -> list[str]:
    """Data-integrity checker: rule ids unique,
    every REASONS entry carrying a reason and an
    offered path, the catalogue covering every
    refusal code the shipped estate's filed
    fixtures can produce (by ``_reason_key``),
    fixtures well-formed and unique."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in REFUSAL_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate REFUSAL_RULES ids")
    for key, entry in REASONS.items():
        if not entry.get("reason") or not entry.get("path"):
            problems.append(
                f"REASONS[{key}]: a reason and an offered "
                "path are both required")
        if ": " in key:
            problems.append(
                f"REASONS[{key}]: keys are heads, never "
                "carrying a ': ' detail")
    for code in sorted(_estate_problem_codes()):
        if _reason_key(code) not in REASONS:
            problems.append(
                f"estate refusal code {code!r} has no "
                "catalogued reason")
    fixture_ids = [f["id"] for f in REFUSAL_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate REFUSAL_FIXTURES ids")
    for fixture in REFUSAL_FIXTURES:
        if fixture["source"] not in ("policy", "configured", "synthetic"):
            problems.append(
                f"fixture {fixture['id']}: unknown source "
                f"{fixture['source']!r}")
        for problem in fixture["problems"]:
            if fixture["catalogued"] and \
                    _reason_key(problem) not in REASONS:
                problems.append(
                    f"fixture {fixture['id']}: filed catalogued "
                    f"but {problem!r} has no REASONS entry")
    return problems


def refusal_honesty_live_problems() -> list[str]:
    """Live checker: every filed fixture's real
    result is refused exactly as filed, and its
    notice must tell it exactly — marker first,
    every problem verbatim with its catalogued
    (or marked-uncatalogued) reason and offered
    path, a strict block told as a strict block,
    no draft text anywhere in the message — and
    a non-refused result must raise."""
    problems: list[str] = []
    for fixture in REFUSAL_FIXTURES:
        result = _run_fixture(fixture)
        if result.verdict != "refused":
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{result.verdict} != refused")
            continue
        if tuple(result.problems) != tuple(fixture["problems"]):
            problems.append(
                f"fixture {fixture['id']}: problems "
                f"{result.problems} != {fixture['problems']}")
            continue
        notice = refusal_notice(result)
        if not notice.message.startswith(REFUSAL_MARKER):
            problems.append(
                f"fixture {fixture['id']}: the message does "
                "not open with the marker")
        if notice.problems != tuple(fixture["problems"]):
            problems.append(
                f"fixture {fixture['id']}: notice problems "
                "drifted from the record")
        for explanation in notice.explanations:
            if explanation["problem"] not in notice.message or \
                    explanation["reason"] not in notice.message or \
                    explanation["path"] not in notice.message:
                problems.append(
                    f"fixture {fixture['id']}: problem "
                    f"{explanation['problem']!r}, its reason, "
                    "or its offered path missing from the message")
            if explanation["catalogued"] != fixture["catalogued"]:
                problems.append(
                    f"fixture {fixture['id']}: problem "
                    f"{explanation['problem']!r} catalogued "
                    "status drifted from the filing")
        if fixture["source"] == "configured":
            if notice.strictness != "strict" or \
                    notice.disposition != "blocked_flagged" or \
                    "strict" not in notice.message or \
                    "flagged" not in notice.message:
                problems.append(
                    f"fixture {fixture['id']}: a strict block "
                    "is not told as a strict block")
        if fixture.get("text") and fixture["text"].strip() and \
                fixture["text"] in notice.message:
            problems.append(
                f"fixture {fixture['id']}: blocked draft text "
                "leaked into its refusal message")
        if not fixture["problems"] and \
                NO_PROBLEM_REASON not in notice.message:
            problems.append(
                f"fixture {fixture['id']}: the unnamed-problem "
                "gap is not stated")
    from thoughtforge.refinement.policy_surface import enforce_policy

    clean = enforce_policy(
        "The ferry leaves at dawn and the tide is low. "
        "The keeper lights the lantern when the ferry is late.")
    try:
        refusal_notice(clean)
        problems.append("a clean result did not raise")
    except ValueError:
        pass
    return problems


def refusal_honesty_as_data() -> dict[str, Any]:
    return {
        "format": REFUSAL_FORMAT,
        "refusal_doc": REFUSAL_DOC,
        "marker": REFUSAL_MARKER,
        "rules": [dict(r) for r in REFUSAL_RULES],
        "reasons": {k: dict(v) for k, v in REASONS.items()},
        "fixtures": [
            {k: v for k, v in f.items()} for f in REFUSAL_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_behaviour": "not_proven",
            "policy_surface": "unchanged",
            "policy_config": "unchanged",
            "fragment_salvage": "unchanged",
            "salvage_path": "offered_never_run_by_this_slice",
        },
    }
