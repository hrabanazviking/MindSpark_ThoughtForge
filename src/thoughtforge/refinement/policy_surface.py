"""TF-093 — Enforcement mapped + completed: the
enforcement module's checks unified with TF-049's
cite-or-explain into one policy surface.

Roadmap: "Enforcement mapped + completed:
enforcement module's checks unified with TF-049's
cite-or-explain into one policy surface."

Measured start (probed live this tick):

- **Enforcement was five separate surfaces.**
  ``EnforcementGate`` (refinement/enforcement.py)
  judged length, genericness, and a legacy
  any-QID citation floor, with TF-049's per-claim
  cite-or-explain ANDed into its citation leg when
  the caller passed ``records``. TF-090's format
  gate, TF-091's loop gate, and TF-092's drift
  gate were each caller-invoked beside it. No
  single place named every check, their order,
  the signal each one needs, or how their
  verdicts combine — a caller had to know all
  five surfaces and compose them by hand.
- **Two citation rules stood unreconciled.**
  TF-049 filed it twice (its module docstring and
  CITATION_POLICY): the per-claim rule passes an
  answer that claims no stored fact, while the
  legacy floor fails that same answer when QIDs
  were retrieved but none is cited — the floor
  is claim-blind, the very defect TF-049 was
  built to fix. TF-049 explicitly left the call
  to TF-093: "relaxing it is TF-093's call,
  recorded in the policy, not taken silently."
- **Regeneration was a named, unwired option**
  in CITATION_POLICY, for a stated reason: the
  core's only regeneration path (the quality-gate
  repair pass) builds its prompt with NO
  knowledge context (TF-023), so a regenerated
  answer cannot cite what it was never shown.
  TF-093 owns "wiring regeneration under the
  unified policy surface" (TF-049's words).

What this slice builds — ``enforce_policy`` in
this module, the one policy surface:

- **THE MAP (POLICY_CHECKS, as data).** Every
  enforcement check in the refinement estate is
  named in execution order, with its owner, its
  source function, the caller signal it needs,
  and what its failure does. The checks are the
  real gates, called, never re-implemented:
  TF-090 format + prose truncation, TF-091 loop
  rescue, TF-092 drift rescue, the citation
  check (TF-049 per-claim / legacy floor,
  reconciled below), and EnforcementGate's
  length + genericness legs.
- **SIGNALS ARE CALLER-SUPPLIED, NEVER GUESSED**
  (the TF-090 shape, generalised). A check whose
  signal the caller did not supply — a structured
  ask, a suspected truncation, an expected
  language, records / retrieved QIDs — is
  recorded ``skipped`` with its reason. A skipped
  check is never reported as passed, and the
  surface never invents a signal (salvage has no
  structured-ask signal; that is why TF-090's
  gate was caller-invoked, and the rule carries).
- **ORDER: REPAIR BEFORE JUDGEMENT.** Format and
  truncation act first (they can change the
  text's shape), loop and drift rescues second
  (deletion only), and citation / length /
  genericness judge LAST — on the exact text
  that would leave. A citation that survives
  only inside a cut segment does not count, and
  a rescued text is judged as rescued.
- **THE CITATION RECONCILIATION (TF-093's call,
  recorded here, per TF-049's filing).** When
  the caller supplies ``records``, the per-claim
  cite-or-explain rule is the citation law of
  this surface: an answer claiming no stored
  fact passes its citation check even when
  retrieved QIDs exist uncited — claim-blindness
  is not re-imposed by the back door. The legacy
  floor is recorded ``subsumed_by_per_claim``
  in that mode (the surface invokes
  EnforcementGate with its floor leg inert and
  the records leg live). When ``records`` are
  NOT supplied, the legacy floor is the only
  citation rule available and applies exactly
  as before. ``EnforcementGate`` itself is
  **unchanged**: its no-records behaviour is
  byte-identical, its with-records AND is its
  own standing behaviour for its own callers
  (the core's compose path among them); the
  reconciliation is this surface's policy,
  stated here, not a silent edit to the gate.
- **REGENERATION: RULED NOT WIRED, HERE.**
  The unified surface never calls a model (no
  gate in this estate does; regeneration is the
  caller's engine work). Under the shipped
  policy the citation action stays ``flag`` and
  ``regeneration_required`` stays False. The
  surface hands the caller the flagged verdict
  with the full per-claim record, so a caller
  who regenerates does so WITH the knowledge
  context TF-023's repair pass lacks — that
  hand-off, not a hidden model call inside a
  checker, is the honest wiring. TF-094 owns
  strictness configuration, TF-095 the
  user-facing telling of a flag.
- **SALVAGE SCORING: RULED NOT ENFORCEMENT.**
  TF-049 §5 left TF-093 the question whether
  salvage's internal QID-only draft scoring
  moves to markers. Ruling: it does not. That
  score ranks drafts; it enforces nothing, and
  moving it would rewrite TF-087's catalogue
  and TF-089's filed metrics mid-Part.
  ``FragmentSalvage`` is untouched by this
  slice, as it was by TF-090/091/092.
- **STRUCTURED ANSWERS ARE JUDGED AS
  STRUCTURE.** On a structured ask the prose
  checks (loop, drift, citation, length,
  genericness) are recorded
  ``skipped_not_applicable_structured`` — a JSON
  object's word count is not a prose length,
  and its values are not prose claims. The
  format gate alone judges it (its schema leg
  included).
- **VERDICTS COMBINE BY PRECEDENCE.**
  ``refused`` (a check refused — nothing honest
  to judge or to pass on; the surface returns
  no usable text and later checks are recorded
  ``skipped_after_refusal``) outranks
  ``flagged`` (drift flagged verbatim, a
  citation / length / genericness failure —
  the text returns, annotated by the gate
  where the gate annotates, with every flag
  named), which outranks ``rescued`` (a repair
  or cut was applied and every judgement
  passed), which outranks ``clean``.

Proof grade FIXTURE — every filed case is run
live through the real gates on this VM,
including the reconciliation contrast proven
against the real ``EnforcementGate``. A live
model's behaviour under the unified surface is
NOT PROVEN (no model in tree); TF-101 runs the
full policy stack over TF-100's chaos corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from thoughtforge.knowledge.models import ActivatedRecord
from thoughtforge.refinement.citation_enforcement import check_citations
from thoughtforge.refinement.drift_rescue import rescue_drift
from thoughtforge.refinement.enforcement import EnforcementGate
from thoughtforge.refinement.format_enforcement import (
    enforce_format,
    repair_truncation,
)
from thoughtforge.refinement.loop_rescue import rescue_loops

POLICY_FORMAT = "thoughtforge-policy-surface/1"

POLICY_DOC = "docs/architecture/ENFORCEMENT_POLICY_SURFACE.md"

# ── The rules ───────────────────────────────────────────────────────────────

POLICY_RULES: tuple[dict[str, Any], ...] = (
    {"id": "one_surface_every_check_named",
     "rule": "every enforcement check in the refinement "
             "estate runs through enforce_policy or is named "
             "in POLICY_CHECKS with its owner and source; "
             "no check hides outside the surface"},
    {"id": "signals_are_caller_supplied_never_guessed",
     "rule": "a check whose caller signal was not supplied "
             "(structured ask, suspected truncation, expected "
             "language, records / retrieved QIDs) is recorded "
             "skipped with its reason — never passed, never "
             "guessed from the draft"},
    {"id": "order_is_repair_before_judgement",
     "rule": "execution order is POLICY_CHECKS order: format "
             "and truncation repairs first, loop and drift "
             "rescues (deletion only) second, citation / "
             "length / genericness judging the exact final "
             "text last"},
    {"id": "citation_reconciled_per_claim_when_records",
     "rule": "when the caller supplies records, TF-049's "
             "per-claim cite-or-explain rule is the citation "
             "law of this surface and the legacy any-QID "
             "floor is recorded subsumed_by_per_claim; when "
             "records are not supplied, the legacy floor "
             "applies exactly as before; EnforcementGate "
             "itself is unchanged in both modes"},
    {"id": "regeneration_not_wired_by_ruling",
     "rule": "the surface never calls a model; the shipped "
             "citation action stays flag and "
             "regeneration_required stays False — the caller "
             "receives the flagged verdict with the full "
             "per-claim record and may regenerate WITH the "
             "knowledge context TF-023's repair pass lacks"},
    {"id": "salvage_scoring_is_not_enforcement",
     "rule": "FragmentSalvage's QID-only draft scoring ranks "
             "drafts and enforces nothing; ruled unmoved to "
             "markers (moving it would rewrite TF-087's "
             "catalogue and TF-089's metrics); FragmentSalvage "
             "is unchanged by this slice"},
    {"id": "structured_answers_judged_as_structure",
     "rule": "on a structured ask the prose checks (loop, "
             "drift, citation, length, genericness) are "
             "recorded skipped_not_applicable_structured and "
             "the format gate alone judges the output"},
    {"id": "refusal_stops_the_pipeline",
     "rule": "a refused check ends the run: the surface "
             "returns no usable text, the refusal's problems "
             "are named, and every later check is recorded "
             "skipped_after_refusal — never passed"},
    {"id": "verdict_precedence_is_filed",
     "rule": "unified verdicts combine refused > flagged > "
             "rescued > clean; every check outcome is "
             "recorded in order whatever the verdict"},
    {"id": "every_outcome_is_recorded",
     "rule": "a PolicyResult carries one CheckOutcome per "
             "POLICY_CHECKS entry, in order — status, the "
             "underlying gate's verdict, and its recorded "
             "detail (repairs, cuts, segments, uncited and "
             "foreign markers) — TF-097 owns carrying the "
             "record into the trace"},
)

# ── The map: every check, in execution order ────────────────────────────────

POLICY_CHECKS: tuple[dict[str, Any], ...] = (
    {"id": "format",
     "owner": "TF-090",
     "source": "refinement/format_enforcement.enforce_format",
     "signal": "structured_ask",
     "on_failure": "refused — no usable text; a repair that "
                   "would have to invent content is a refusal"},
    {"id": "truncation",
     "owner": "TF-090",
     "source": "refinement/format_enforcement.repair_truncation",
     "signal": "truncation_suspect",
     "on_failure": "refused — a draft with no complete "
                   "sentence is never given an invented ending"},
    {"id": "loop",
     "owner": "TF-091",
     "source": "refinement/loop_rescue.rescue_loops",
     "signal": "none",
     "on_failure": "rescued by deletion only; refused only "
                   "on empty output"},
    {"id": "drift",
     "owner": "TF-092",
     "source": "refinement/drift_rescue.rescue_drift",
     "signal": "expected_language",
     "on_failure": "rescued by deletion only, or flagged "
                   "verbatim when drift is the whole answer "
                   "or the majority; refused on an expected "
                   "language the TF-060 hint cannot judge"},
    {"id": "citation",
     "owner": "TF-049 (+ legacy floor, pre-TF-049)",
     "source": "refinement/citation_enforcement.check_citations "
               "/ refinement/enforcement.EnforcementGate",
     "signal": "records_or_retrieved_qids",
     "on_failure": "flagged — the gate fails the citation "
                   "check, annotates the text, and (through "
                   "TF-047) the turn's writeback gate closes"},
    {"id": "length",
     "owner": "pre-existing EnforcementGate",
     "source": "refinement/enforcement.EnforcementGate",
     "signal": "none",
     "on_failure": "flagged — the gate appends its "
                   "shorter-than-expected forge note"},
    {"id": "genericness",
     "owner": "pre-existing EnforcementGate",
     "source": "refinement/enforcement.EnforcementGate",
     "signal": "none",
     "on_failure": "flagged — a hard generic phrase is named "
                   "in the result notes"},
)

_CHECK_IDS = tuple(c["id"] for c in POLICY_CHECKS)

_STATUSES = ("passed", "rescued", "flagged", "refused", "skipped")

# ── Result types ────────────────────────────────────────────────────────────


@dataclass
class CheckOutcome:
    """One check's outcome under the unified surface."""
    id: str                         # a POLICY_CHECKS id
    status: str                     # passed|rescued|flagged|refused|skipped
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResult:
    """The unified verdict on one output."""
    verdict: str                    # "clean" | "rescued" | "flagged" | "refused"
    text: str | None = None         # usable / annotated text; None when refused
    checks: tuple[CheckOutcome, ...] = ()
    flags: tuple[str, ...] = ()     # named, on flagged verdicts
    problems: tuple[str, ...] = ()  # named, on refused verdicts
    notes: str = ""                 # the gate's notes, when the gate ran
    claim_check: Any = None         # TF-049 CitationCheck, when records ran

    def check(self, check_id: str) -> CheckOutcome:
        for outcome in self.checks:
            if outcome.id == check_id:
                return outcome
        raise KeyError(check_id)


# ── The surface ─────────────────────────────────────────────────────────────


def enforce_policy(
    text: str,
    *,
    citations: list[str] | None = None,
    retrieved_qids: set[str] | list[str] | tuple[str, ...] | None = None,
    records: list | None = None,
    expected_language: str | None = None,
    structured: bool = False,
    required_keys: tuple[str, ...] = (),
    expected_format: str = "object",
    truncation_suspect: bool = False,
) -> PolicyResult:
    """Judge one output under every enforcement check
    whose signal the caller supplied, in POLICY_CHECKS
    order, and combine the verdicts by the filed
    precedence (refused > flagged > rescued > clean).

    Signals: ``structured`` (+ ``required_keys`` /
    ``expected_format``) the ask was structured;
    ``truncation_suspect`` the caller suspects a
    truncated prose draft; ``expected_language`` the
    language of the ask / conversation; ``records``
    the retrieval bundle's activated records (TF-049
    per-claim law) and/or ``retrieved_qids`` (the
    legacy floor's set). A check without its signal
    is recorded skipped, never passed."""
    raw = text or ""
    outcomes: dict[str, CheckOutcome] = {}
    flags: list[str] = []
    problems: list[str] = []
    notes = ""
    claim_check = None

    def skip(check_id: str, reason: str) -> None:
        outcomes[check_id] = CheckOutcome(
            check_id, "skipped", {"reason": reason})

    if not raw.strip():
        for check_id in _CHECK_IDS:
            skip(check_id, "empty_output")
        return PolicyResult(
            "refused", checks=tuple(outcomes[c] for c in _CHECK_IDS),
            problems=("empty_output",))

    working = raw
    stopped = False

    # ── format (TF-090) ─────────────────────────────────────────────────
    if structured:
        result = enforce_format(
            working, required_keys=tuple(required_keys),
            expected=expected_format)
        if result.verdict == "refused":
            outcomes["format"] = CheckOutcome(
                "format", "refused",
                {"verdict": result.verdict,
                 "problems": list(result.problems)})
            problems.extend(result.problems)
            stopped = True
        elif result.verdict == "repaired":
            outcomes["format"] = CheckOutcome(
                "format", "rescued",
                {"verdict": result.verdict,
                 "repairs": list(result.repairs),
                 "dropped_tail": result.dropped_tail})
            working = result.text or working
        else:
            outcomes["format"] = CheckOutcome(
                "format", "passed", {"verdict": result.verdict})
    else:
        skip("format", "signal_not_supplied:not_structured_ask")

    # ── truncation (TF-090) ─────────────────────────────────────────────
    if stopped:
        skip("truncation", "skipped_after_refusal")
    elif structured:
        skip("truncation", "covered_by_format_check")
    elif not truncation_suspect:
        skip("truncation", "signal_not_supplied:truncation_suspect")
    else:
        result = repair_truncation(working)
        if result.verdict == "refused":
            outcomes["truncation"] = CheckOutcome(
                "truncation", "refused",
                {"verdict": result.verdict,
                 "problems": list(result.problems)})
            problems.extend(result.problems)
            stopped = True
        elif result.verdict == "repaired":
            outcomes["truncation"] = CheckOutcome(
                "truncation", "rescued",
                {"verdict": result.verdict,
                 "dropped_tail": result.dropped_tail})
            working = result.text or working
        else:
            outcomes["truncation"] = CheckOutcome(
                "truncation", "passed", {"verdict": result.verdict})

    # ── loop (TF-091) ───────────────────────────────────────────────────
    if stopped:
        skip("loop", "skipped_after_refusal")
    elif structured:
        skip("loop", "skipped_not_applicable_structured")
    else:
        result = rescue_loops(working)
        if result.verdict == "refused":
            outcomes["loop"] = CheckOutcome(
                "loop", "refused",
                {"verdict": result.verdict,
                 "problems": list(result.problems)})
            problems.extend(result.problems)
            stopped = True
        elif result.verdict == "rescued":
            outcomes["loop"] = CheckOutcome(
                "loop", "rescued",
                {"verdict": result.verdict,
                 "cuts": [{"kind": c.kind, "unit": c.unit,
                           "occurrences": c.occurrences,
                           "dropped_text": c.dropped_text}
                          for c in result.cuts]})
            working = result.text or working
        else:
            outcomes["loop"] = CheckOutcome(
                "loop", "passed", {"verdict": result.verdict})

    # ── drift (TF-092) ──────────────────────────────────────────────────
    if stopped:
        skip("drift", "skipped_after_refusal")
    elif structured:
        skip("drift", "skipped_not_applicable_structured")
    elif expected_language is None:
        skip("drift", "signal_not_supplied:expected_language")
    else:
        result = rescue_drift(working, expected_language)
        if result.verdict == "refused":
            outcomes["drift"] = CheckOutcome(
                "drift", "refused",
                {"verdict": result.verdict,
                 "problems": list(result.problems)})
            problems.extend(result.problems)
            stopped = True
        elif result.verdict == "flagged":
            outcomes["drift"] = CheckOutcome(
                "drift", "flagged",
                {"verdict": result.verdict,
                 "flags": list(result.flags),
                 "segments": [
                     {"language": s.language,
                      "sentences": list(s.sentences),
                      "first_sentence": s.first_sentence,
                      "cut": s.cut} for s in result.segments]})
            flags.extend(f"drift:{f}" for f in result.flags)
        elif result.verdict == "rescued":
            outcomes["drift"] = CheckOutcome(
                "drift", "rescued",
                {"verdict": result.verdict,
                 "segments": [
                     {"language": s.language,
                      "sentences": list(s.sentences),
                      "first_sentence": s.first_sentence,
                      "cut": s.cut} for s in result.segments]})
            working = result.text or working
        else:
            outcomes["drift"] = CheckOutcome(
                "drift", "passed", {"verdict": result.verdict})

    # ── citation / length / genericness (the gate, on the final text) ──
    if stopped:
        skip("citation", "skipped_after_refusal")
        skip("length", "skipped_after_refusal")
        skip("genericness", "skipped_after_refusal")
    elif structured:
        skip("citation", "skipped_not_applicable_structured")
        skip("length", "skipped_not_applicable_structured")
        skip("genericness", "skipped_not_applicable_structured")
    else:
        gate = EnforcementGate()
        qids = set(retrieved_qids or ())
        if records is not None:
            # TF-093 reconciliation: the per-claim rule is
            # the citation law; the gate is invoked with
            # its floor leg inert (empty retrieved set) and
            # its records leg live, and the floor's standing
            # is recorded, not silently dropped.
            gate_result = gate.check(
                working, list(citations or []), set(),
                records=list(records))
            claim_check = gate_result.claim_check
            citation_detail: dict[str, Any] = {
                "mode": "per_claim",
                "floor": "subsumed_by_per_claim",
                "regeneration_required": bool(
                    claim_check and claim_check.regeneration_required),
            }
            if claim_check is not None:
                citation_detail["uncited_markers"] = list(
                    claim_check.uncited_markers)
                citation_detail["foreign_markers"] = list(
                    claim_check.foreign_markers)
            if gate_result.citation_check:
                outcomes["citation"] = CheckOutcome(
                    "citation", "passed", citation_detail)
            else:
                outcomes["citation"] = CheckOutcome(
                    "citation", "flagged", citation_detail)
                if citation_detail.get("uncited_markers"):
                    flags.append("citation:uncited_claim")
                if citation_detail.get("foreign_markers"):
                    flags.append("citation:foreign_marker")
        elif qids:
            gate_result = gate.check(
                working, list(citations or []), qids)
            if gate_result.citation_check:
                outcomes["citation"] = CheckOutcome(
                    "citation", "passed",
                    {"mode": "legacy_floor", "floor": "applied"})
            else:
                outcomes["citation"] = CheckOutcome(
                    "citation", "flagged",
                    {"mode": "legacy_floor", "floor": "applied"})
                flags.append("citation:no_retrieved_qids_cited")
        else:
            gate_result = gate.check(
                working, list(citations or []), set())
            skip("citation", "signal_not_supplied:records_or_retrieved_qids")
        notes = gate_result.notes
        working = gate_result.text
        outcomes["length"] = CheckOutcome(
            "length", "passed" if gate_result.length_check else "flagged",
            {"verdict": "pass" if gate_result.length_check else "fail"})
        if not gate_result.length_check:
            flags.append("length:too_short")
        outcomes["genericness"] = CheckOutcome(
            "genericness",
            "passed" if gate_result.genericness_check else "flagged",
            {"verdict": "pass" if gate_result.genericness_check else "fail"})
        if not gate_result.genericness_check:
            flags.append("genericness:hard_generic_phrase")

    checks = tuple(outcomes[c] for c in _CHECK_IDS)
    if stopped:
        return PolicyResult(
            "refused", checks=checks, flags=tuple(flags),
            problems=tuple(problems), notes=notes,
            claim_check=claim_check)
    if flags:
        verdict = "flagged"
    elif any(o.status == "rescued" for o in checks):
        verdict = "rescued"
    else:
        verdict = "clean"
    return PolicyResult(
        verdict, text=working, checks=checks, flags=tuple(flags),
        notes=notes, claim_check=claim_check)


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic material throughout (TF-017's Lantern Hollow
# world + TF-049's synthetic Odin record, never canon).
# ``records`` names the filed record set built by
# _fixture_records(); ``checks`` files the expected
# status of the checks the case exercises.

_EN1 = "The ferry leaves at dawn and the tide is low."
_EN2 = "The keeper lights the lantern when the ferry is late."
_ES1 = "El faro está en la costa y la marea es alta."
_ES2 = "La campana suena cuando el barco llega a la costa."
_BELL = "The bell rings across the harbour at dawn."
_ODIN_UNCITED = "Odin is the chief god of Norse mythology, a god of wisdom and war."
_ODIN_CITED = "Odin [Q123] is the chief god of Norse mythology and wisdom."
_WEATHER = "The weather over the fjord is calm and bright today."
_NO_CITE_NOTE = "\n\n[Forge: limited verified citations available for this response.]"
_TOO_SHORT_NOTE = "\n\n[Forge: response is shorter than expected — knowledge may be sparse.]"


def _fixture_records() -> list[ActivatedRecord]:
    """TF-049's synthetic record pair, rebuilt fresh
    per call (an entity and a reference chunk)."""
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


POLICY_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "clean_prose_no_signals",
     "text": f"{_EN1} {_EN2}",
     "signals": {},
     "verdict": "clean",
     "expected_text": f"{_EN1} {_EN2}",
     "checks": {"format": "skipped", "truncation": "skipped",
                "loop": "passed", "drift": "skipped",
                "citation": "skipped", "length": "passed",
                "genericness": "passed"}},
    {"id": "loop_rescued_then_judged_clean",
     "text": " ".join([_BELL] * 4),
     "signals": {},
     "verdict": "rescued",
     "expected_text": _BELL,
     "checks": {"loop": "rescued", "length": "passed",
                "genericness": "passed"}},
    {"id": "drift_rescued_then_judged_clean",
     "text": f"{_EN1} {_EN2} {_ES1} {_ES2}",
     "signals": {"expected_language": "en"},
     "verdict": "rescued",
     "expected_text": f"{_EN1} {_EN2}",
     "checks": {"loop": "passed", "drift": "rescued",
                "length": "passed"}},
    {"id": "drift_flagged_verbatim",
     "text": f"{_ES1} {_ES2}",
     "signals": {"expected_language": "en"},
     "verdict": "flagged",
     "expected_text": f"{_ES1} {_ES2}",
     "checks": {"drift": "flagged", "length": "passed"},
     "flags": ("drift:whole_answer_off_language",)},
    {"id": "citation_uncited_claim_flagged",
     "text": _ODIN_UNCITED,
     "signals": {"records": "lantern"},
     "verdict": "flagged",
     "expected_text": _ODIN_UNCITED + _NO_CITE_NOTE,
     "checks": {"citation": "flagged", "length": "passed"},
     "flags": ("citation:uncited_claim",)},
    {"id": "citation_no_claim_passes_floor_subsumed",
     "text": _WEATHER,
     "signals": {"records": "lantern", "retrieved_qids": ["Q123"]},
     "verdict": "clean",
     "expected_text": _WEATHER,
     "checks": {"citation": "passed", "length": "passed",
                "genericness": "passed"}},
    {"id": "citation_cited_claim_clean",
     "text": _ODIN_CITED,
     "signals": {"records": "lantern"},
     "verdict": "clean",
     "expected_text": _ODIN_CITED,
     "checks": {"citation": "passed"}},
    {"id": "legacy_floor_flagged_without_records",
     "text": _WEATHER,
     "signals": {"retrieved_qids": ["Q123"]},
     "verdict": "flagged",
     "expected_text": _WEATHER + _NO_CITE_NOTE,
     "checks": {"citation": "flagged"},
     "flags": ("citation:no_retrieved_qids_cited",)},
    {"id": "format_repaired_structured",
     "text": '{"ferry": "dawn",}',
     "signals": {"structured": True},
     "verdict": "rescued",
     "expected_text": '{"ferry": "dawn"}',
     "checks": {"format": "rescued", "loop": "skipped",
                "drift": "skipped", "citation": "skipped",
                "length": "skipped", "genericness": "skipped"}},
    {"id": "format_refused_structured",
     "text": "The ferry leaves at dawn.",
     "signals": {"structured": True},
     "verdict": "refused",
     "expected_text": None,
     "checks": {"format": "refused", "loop": "skipped",
                "citation": "skipped", "length": "skipped"},
     "problems": ("no_json_span",)},
    {"id": "truncation_repaired",
     "text": "The ferry leaves at dawn and the tide is low. The keeper lights",
     "signals": {"truncation_suspect": True},
     "verdict": "rescued",
     "expected_text": "The ferry leaves at dawn and the tide is low.",
     "checks": {"truncation": "rescued", "loop": "passed",
                "length": "passed"}},
    {"id": "truncation_refused_no_complete_sentence",
     "text": "The keeper lights the lantern when",
     "signals": {"truncation_suspect": True},
     "verdict": "refused",
     "expected_text": None,
     "checks": {"truncation": "refused", "loop": "skipped",
                "citation": "skipped"},
     "problems": ("no_complete_sentence",)},
    {"id": "generic_phrase_flagged",
     "text": "As an AI language model I cannot provide that information today.",
     "signals": {},
     "verdict": "flagged",
     "expected_text": "As an AI language model I cannot provide that information today.",
     "checks": {"genericness": "flagged", "length": "passed",
                "citation": "skipped"},
     "flags": ("genericness:hard_generic_phrase",)},
    {"id": "too_short_flagged",
     "text": "Too short.",
     "signals": {},
     "verdict": "flagged",
     "expected_text": "Too short." + _TOO_SHORT_NOTE,
     "checks": {"length": "flagged"},
     "flags": ("length:too_short",)},
    {"id": "empty_refused",
     "text": "   ",
     "signals": {},
     "verdict": "refused",
     "expected_text": None,
     "checks": {"format": "skipped", "loop": "skipped",
                "citation": "skipped", "genericness": "skipped"},
     "problems": ("empty_output",)},
    {"id": "drift_unsupported_language_refused",
     "text": _EN1,
     "signals": {"expected_language": "fr"},
     "verdict": "refused",
     "expected_text": None,
     "checks": {"loop": "passed", "drift": "refused",
                "citation": "skipped", "length": "skipped"},
     "problems": ("unsupported_expected_language",)},
)


def _run_fixture(fixture: dict[str, Any]) -> PolicyResult:
    signals = dict(fixture["signals"])
    if signals.get("records") == "lantern":
        signals["records"] = _fixture_records()
    return enforce_policy(fixture["text"], **signals)


# ── Checkers ────────────────────────────────────────────────────────────────


def policy_surface_problems() -> list[str]:
    """Data-integrity checker: rule / check / fixture
    ids unique, the map covering every check exactly
    once, fixtures well-formed (refusals naming
    problems, flagged cases naming flags, statuses
    in the vocabulary)."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in POLICY_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate POLICY_RULES ids")
    check_ids = [c["id"] for c in POLICY_CHECKS]
    if len(check_ids) != len(set(check_ids)):
        problems.append("duplicate POLICY_CHECKS ids")
    if check_ids != list(_CHECK_IDS):
        problems.append("POLICY_CHECKS order drifted from _CHECK_IDS")
    for entry in POLICY_CHECKS:
        if not entry.get("owner") or not entry.get("source"):
            problems.append(
                f"check {entry['id']}: owner and source must be named")
    fixture_ids = [f["id"] for f in POLICY_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate POLICY_FIXTURES ids")
    for fixture in POLICY_FIXTURES:
        if fixture["verdict"] not in (
                "clean", "rescued", "flagged", "refused"):
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{fixture['verdict']!r} not in the vocabulary")
        if fixture["verdict"] == "refused" and not fixture.get("problems"):
            problems.append(
                f"fixture {fixture['id']}: a refusal must name "
                "its problems")
        if fixture["verdict"] == "flagged" and not fixture.get("flags"):
            problems.append(
                f"fixture {fixture['id']}: a flagged verdict "
                "must name its flags")
        for check_id, status in fixture["checks"].items():
            if check_id not in _CHECK_IDS:
                problems.append(
                    f"fixture {fixture['id']}: unknown check "
                    f"{check_id!r}")
            if status not in _STATUSES:
                problems.append(
                    f"fixture {fixture['id']}: status {status!r} "
                    "not in the vocabulary")
    return problems


def policy_surface_live_problems() -> list[str]:
    """Live checker: every filed fixture is run
    through the real surface and must land exactly
    as filed (verdict, usable text, the filed check
    statuses, flags and problems); the reconciliation
    is proven against the real EnforcementGate (the
    no-claim answer the legacy mode flags is clean
    here, and the gate's own no-records behaviour
    is untouched); a rescued text judges clean on
    a second pass under the same signals."""
    problems: list[str] = []
    for fixture in POLICY_FIXTURES:
        result = _run_fixture(fixture)
        if result.verdict != fixture["verdict"]:
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{result.verdict} != {fixture['verdict']}")
            continue
        if result.text != fixture["expected_text"]:
            problems.append(
                f"fixture {fixture['id']}: text "
                f"{result.text!r} != {fixture['expected_text']!r}")
        if tuple(o.id for o in result.checks) != _CHECK_IDS:
            problems.append(
                f"fixture {fixture['id']}: check outcomes are "
                "not the full map in order")
        for check_id, status in fixture["checks"].items():
            live = result.check(check_id).status
            if live != status:
                problems.append(
                    f"fixture {fixture['id']}: check {check_id} "
                    f"{live} != {status}")
        if fixture["verdict"] == "refused":
            if result.text is not None:
                problems.append(
                    f"fixture {fixture['id']}: a refusal carries "
                    "usable text")
            if tuple(result.problems) != tuple(fixture["problems"]):
                problems.append(
                    f"fixture {fixture['id']}: problems "
                    f"{result.problems} != {fixture['problems']}")
        if fixture["verdict"] == "flagged" and \
                tuple(result.flags) != tuple(fixture["flags"]):
            problems.append(
                f"fixture {fixture['id']}: flags "
                f"{result.flags} != {fixture['flags']}")
        if fixture["verdict"] == "rescued":
            second = _run_fixture(
                {**fixture, "text": result.text or ""})
            if second.verdict != "clean" or second.text != result.text:
                problems.append(
                    f"fixture {fixture['id']}: the rescued text "
                    "does not judge clean on a second pass")

    # The reconciliation, proven against the real gate:
    # with records, the legacy AND fails the no-claim
    # answer (its floor leg); the unified surface,
    # per-claim law, passes it — and the gate's own
    # no-records behaviour is byte-identical to before.
    gate = EnforcementGate()
    legacy = gate.check(_WEATHER, [], {"Q123"}, records=_fixture_records())
    if legacy.citation_check:
        problems.append(
            "reconciliation contrast lost: the gate's with-records "
            "floor no longer flags the no-claim answer")
    unified = enforce_policy(
        _WEATHER, records=_fixture_records(), retrieved_qids={"Q123"})
    if unified.verdict != "clean" or \
            unified.check("citation").detail.get("floor") != \
            "subsumed_by_per_claim":
        problems.append(
            "reconciliation lost: the unified surface does not "
            "subsume the floor under the per-claim rule")
    no_records = gate.check("No citations in this response.", [], {"Q123"})
    if no_records.citation_check or no_records.claim_check is not None:
        problems.append("EnforcementGate's no-records legacy behaviour drifted")
    per_claim = check_citations(_ODIN_UNCITED, _fixture_records())
    if per_claim.passed or per_claim.regeneration_required:
        problems.append(
            "TF-049's per-claim verdict or its flag-not-regenerate "
            "action drifted under the unified surface")
    return problems


def policy_as_data() -> dict[str, Any]:
    return {
        "format": POLICY_FORMAT,
        "policy_doc": POLICY_DOC,
        "rules": [dict(r) for r in POLICY_RULES],
        "checks": [dict(c) for c in POLICY_CHECKS],
        "fixtures": [
            {k: v for k, v in f.items()} for f in POLICY_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_behaviour": "not_proven",
            "enforcement_gate": "unchanged",
            "fragment_salvage": "unchanged_by_design",
            "citation_floor": "subsumed_by_per_claim_when_records_supplied",
            "regeneration": "not_wired_by_ruling_tf093",
        },
    }
