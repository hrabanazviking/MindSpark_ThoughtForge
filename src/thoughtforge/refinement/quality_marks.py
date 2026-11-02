"""TF-102 — User-visible quality marks: where the
product surfaces answers, a rescued/cited answer
can *show* its marks (data ready for consumers;
UI is the consumer's).

Roadmap: "User-visible quality marks: where the
product surfaces answers, a rescued/cited answer
can *show* its marks (data ready for consumers;
UI is the consumer's)."

Measured start (probed live this tick, through
the real TF-093 / TF-094 surfaces and the real
``FragmentSalvage``):

- **Marks existed only scattered, and most
  answers could show none of them.** A policy
  result carries its truth as data — verdict,
  per-check outcomes, flags — but nothing in the
  tree turned that record into a mark a product
  surface could show. The only user-visible
  traces were accidents of other layers: the
  EnforcementGate appends its ``[Forge: …]``
  notes into the answer text itself on a
  citation/length failure (so a genericness or
  drift flag leaves NO visible trace in the
  text at all), citation markers ride inside
  the answer's prose with no summary of what
  was cited, and TF-097's rescue record rides
  the trace (TF-038/TF-084) — inspector data,
  not a user-visible mark. TF-095 built the
  telling for the one case that does not leave
  (a block); the answers that DO leave had no
  equivalent.
- **A passed citation check is not a cited
  answer.** Probed live: the weather answer
  under records passes its citation check with
  zero claims matched and zero markers cited —
  the per-claim rule passes it because it claims
  no stored fact. Any mark layer that read
  "citation check passed" as "cited" would
  badge answers that cite nothing.
- **Salvage's path string cannot carry a mark.**
  Probed live (TF-097's filed debt, again): a
  fabrication-rejected refine returns
  ``salvage_path == "refine_pass_1"`` with the
  final text still the selected draft — the
  path names passes *attempted*. Only TF-097's
  rescue record states which pass, if any, the
  final text came from (``final_source`` /
  ``final_pass``), and a result built outside
  ``forge()`` carries no record at all.

What this slice builds — ``quality_marks`` in
this module, a data layer over TF-093's /
TF-094's real results and TF-097's real rescue
record:

- **FOUR MARK KINDS, AS DATA (MARK_KINDS).**
  ``cited`` (the answer cites retrieved sources
  and the citation check passed — the markers
  named), ``rescued`` (a repair or cut was
  applied and adopted — every mechanism named:
  the policy checks that rescued, and salvage's
  adopted refine pass), ``flagged`` (the answer
  leaves under standard with flags — every flag
  verbatim, with TF-095's catalogued plain
  reason), and ``warning`` (TF-094 relaxed:
  the flags downgraded to warnings, verbatim,
  the verdict earned underneath recorded).
  A consumer renders these however its UI
  chooses; ``marks_line`` is the one plain-text
  rendering filed here, for surfaces that want
  words.
- **CITED REQUIRES TWO TRUTHS AT ONCE** — the
  citation check *passed* AND at least one
  retrieved marker is actually cited. Under
  TF-049's per-claim mode the markers are the
  claim check's own: the known markers the
  answer's claims carry. Under the legacy floor
  the markers are the QIDs the result's own
  final text carries (read from the record's
  text, mode named in the mark). A skipped
  citation check earns no mark, and a flagged
  citation earns the *flagged* mark, never a
  cited one.
- **RESCUED NAMES ITS MECHANISMS, AND
  ATTEMPTED IS NOT ADOPTED.** Policy rescues
  name the checks that rescued (format /
  truncation / loop / drift, each with its
  plain label in RESCUE_LABELS). A salvage
  rescue is marked only when TF-097's record
  says the final text came from a refine pass
  (``final_source == "refine_pass"``), and the
  mark names the *adopted* pass — the
  adopt-then-fabricate fixture's path says
  pass 2 (attempted) while its mark says
  pass 1 (adopted). A rejected attempt earns
  no mark, and a result with no rescue record
  earns no salvage mark: the path string alone
  would lie.
- **ONLY A LEAVING ANSWER GETS MARKS.** A
  refused result (TF-093's refusal, or TF-094's
  strict block) raises ``ValueError`` — a
  blocked answer's telling is TF-095's refusal
  notice, and dressing a block in quality marks
  would be a fabrication. An empty salvage
  result (no candidates, no text) raises the
  same way. Calling with neither a result nor
  a salvage result raises too.
- **A PLAIN ANSWER HAS NO MARKS, STATED.** A
  clean, unrescued, uncited answer's mark set
  is empty — that emptiness is the filed,
  honest state (``has_marks`` False,
  ``marks_line`` "" so a consumer shows
  nothing), never dressed with an invented
  "checked" badge: checks that were skipped
  were not passed, and this layer claims
  nothing they did not earn.
- **MARKS NEVER ALTER THE ANSWER.** The layer
  reads result records and returns separate
  data; no result's text is touched, no gate
  is called, no check is re-judged. TF-093's
  surface, TF-094's layer, TF-095's catalogue
  (reused for flag reasons, never edited),
  TF-097's record and ``FragmentSalvage`` are
  all unchanged.

Filed limits: a citation's *label / source /
locator* (TF-048's payload) is not carried on
a policy result, so a cited mark names markers,
not titles — a consumer holding the bundle can
join them itself. Salvage contributes no cited
mark of its own: its ``citations`` list is a
draft-scoring count (TF-093's filed ruling —
scoring is not enforcement), so citation truth
has exactly one source, the policy surface.
The core does not yet route through the policy
surfaces (TF-093/TF-094/TF-095's standing
limit), so no shipped turn yet emits these
marks end-to-end; this slice builds the data
every leaving result can be marked with, and
wiring it into a consumer is that consumer's
slice (Part IX owns bindings).

Proof grade FIXTURE — every filed case is run
live through the real TF-093 / TF-094 surfaces
and TF-097's real salvage fixtures on this VM,
and its marks judged against the records. A
live model's marks are NOT PROVEN (no model
in tree).

The human-readable write-up is
``docs/architecture/QUALITY_MARKS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MARKS_FORMAT = "thoughtforge-quality-marks/1"

MARKS_DOC = "docs/architecture/QUALITY_MARKS.md"

#: The mark kinds, in the order marks are filed.
MARK_KINDS: tuple[str, ...] = (
    "cited", "rescued", "flagged", "warning")

# ── The rules ───────────────────────────────────────────────────────────────

MARKS_RULES: tuple[dict[str, Any], ...] = (
    {"id": "marks_are_data_for_consumers_ui_is_the_consumers",
     "rule": "quality_marks returns detached data (kinds, "
             "markers, mechanisms, flags verbatim) plus one "
             "filed plain-text rendering, marks_line; how a "
             "product surface displays them is the consumer's "
             "choice — this slice builds no UI and appends "
             "nothing to any answer"},
    {"id": "derived_from_records_never_invented",
     "rule": "every mark is derived from a result record — "
             "TF-093/TF-094's verdict, check outcomes, flags "
             "and warnings, the TF-049 claim check, TF-097's "
             "rescue record; no gate is called and no draft is "
             "re-judged by this layer"},
    {"id": "cited_requires_passed_check_and_a_real_citation",
     "rule": "a cited mark requires the citation check "
             "passed AND at least one retrieved marker "
             "actually cited — per-claim markers from the "
             "claim check, or, under the legacy floor, the "
             "QIDs the result's own final text carries; a "
             "passed check with nothing cited (no stored "
             "fact claimed) and a skipped check both earn "
             "no cited mark"},
    {"id": "a_flagged_citation_is_never_a_cited_mark",
     "rule": "when the citation check is flagged the answer "
             "earns the flagged mark with its flag verbatim — "
             "never a cited mark — and under TF-094 relaxed "
             "the same flag rides as a warning, never both"},
    {"id": "rescued_names_its_mechanisms",
     "rule": "a rescued mark names every mechanism that "
             "rescued: each policy check whose outcome is "
             "rescued (format / truncation / loop / drift, "
             "labelled in RESCUE_LABELS) and salvage's "
             "adopted refine pass when TF-097's record "
             "carries one"},
    {"id": "attempted_is_not_adopted_for_salvage",
     "rule": "a salvage rescue is marked only when the "
             "rescue record's final_source is refine_pass, "
             "and names final_pass — the adopted pass; a "
             "rejected or not-improved attempt earns no "
             "mark, salvage_path's attempted naming earns "
             "none, and a result carrying no rescue record "
             "earns no salvage mark at all"},
    {"id": "flags_and_warnings_verbatim_with_tf095_reasons",
     "rule": "flagged and warning marks carry the surface's "
             "flags / warnings verbatim, each beside the "
             "plain reason TF-095's REASONS catalogue files "
             "for it (an uncatalogued flag is named verbatim "
             "and marked uncatalogued, TF-095's rule)"},
    {"id": "only_a_leaving_answer_gets_marks",
     "rule": "a refused result — TF-093's refusal or "
             "TF-094's strict block — raises ValueError: a "
             "blocked answer is told by TF-095's refusal "
             "notice, never dressed in quality marks; an "
             "empty salvage result and a call with neither "
             "record raise the same way"},
    {"id": "a_plain_answer_has_no_marks_and_that_is_stated",
     "rule": "a clean unrescued uncited answer's mark set "
             "is empty — has_marks False and marks_line "
             "empty — with no invented badge for checks "
             "that passed or were skipped; emptiness is "
             "the honest display state"},
    {"id": "the_surfaces_are_unchanged",
     "rule": "TF-093's surface, TF-094's layer, TF-095's "
             "catalogue, TF-097's record, every gate and "
             "FragmentSalvage are unchanged; this layer "
             "reads their records and writes data about "
             "them, and never mutates a result it reads"},
)

# ── The plain labels ────────────────────────────────────────────────────────

#: One plain label per mechanism that can rescue,
#: keyed by the TF-093 check id, plus salvage's
#: refine adoption. The data checker proves this
#: covers every check that can return rescued.
RESCUE_LABELS: dict[str, str] = {
    "format": "structured format repaired",
    "truncation": "cut back to the last complete sentence",
    "loop": "repeated passages cut",
    "drift": "off-language sentences cut",
    "salvage_refine": "refined by an adopted salvage pass",
}

#: Checks whose TF-093 outcome can be "rescued".
_RESCUABLE_CHECKS = ("format", "truncation", "loop", "drift")


def _flag_reason(flag: str) -> dict[str, Any]:
    """One flag's catalogued plain reason, by
    TF-095's own catalogue and key rule. An
    uncatalogued flag is named verbatim and
    marked — never given an invented reason."""
    from thoughtforge.refinement.refusal_honesty import (
        REASONS,
        UNCATALOGUED_REASON,
        _reason_key,
    )

    entry = REASONS.get(_reason_key(flag))
    if entry is None:
        return {"flag": flag, "catalogued": False,
                "reason": UNCATALOGUED_REASON}
    return {"flag": flag, "catalogued": True,
            "reason": entry["reason"]}


# ── The marks ───────────────────────────────────────────────────────────────


@dataclass
class QualityMark:
    """One user-visible quality mark, as data."""
    kind: str                       # a MARK_KINDS entry
    label: str                      # its plain-text segment
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMarks:
    """Every mark one leaving answer earned."""
    marks: tuple[QualityMark, ...] = ()
    policy_verdict: str | None = None   # the leaving verdict, when a result was given
    strictness: str | None = None       # TF-094's level, when its result
    disposition: str | None = None      # TF-094's disposition, when its result
    salvage_final_source: str | None = None  # TF-097's final_source, when a record rode

    @property
    def has_marks(self) -> bool:
        return bool(self.marks)

    def kinds(self) -> tuple[str, ...]:
        return tuple(m.kind for m in self.marks)

    def mark(self, kind: str) -> QualityMark:
        for entry in self.marks:
            if entry.kind == kind:
                return entry
        raise KeyError(kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MARKS_FORMAT,
            "policy_verdict": self.policy_verdict,
            "strictness": self.strictness,
            "disposition": self.disposition,
            "salvage_final_source": self.salvage_final_source,
            "has_marks": self.has_marks,
            "marks": [
                {"kind": m.kind, "label": m.label,
                 "detail": m.detail} for m in self.marks],
        }


def _cited_mark(result: Any) -> QualityMark | None:
    """The cited mark a result earned, or None.
    Requires the citation outcome passed and a
    real citation: per-claim markers from the
    claim check, else (legacy floor) the QIDs in
    the result's own final text."""
    try:
        outcome = result.check("citation")
    except (KeyError, AttributeError):
        return None
    if outcome.status != "passed":
        return None
    claim_check = getattr(result, "claim_check", None)
    if claim_check is not None:
        markers: list[str] = []
        for claim in claim_check.claims:
            for marker in claim.cited_markers:
                if marker not in markers:
                    markers.append(marker)
        if not markers:
            return None
        return QualityMark(
            "cited",
            "cites retrieved sources: " + ", ".join(markers),
            {"markers": markers, "mode": "per_claim",
             "claims_cited": sum(
                 1 for c in claim_check.claims if c.cited_markers)})
    if outcome.detail.get("mode") == "legacy_floor":
        from thoughtforge.refinement.enforcement import _QID_RE

        markers = []
        for marker in _QID_RE.findall(result.text or ""):
            if marker not in markers:
                markers.append(marker)
        if not markers:
            return None
        return QualityMark(
            "cited",
            "cites retrieved sources: " + ", ".join(markers),
            {"markers": markers, "mode": "legacy_floor"})
    return None


def quality_marks(
    result: Any = None,
    *,
    salvage_result: Any = None,
) -> QualityMarks:
    """Build every quality mark one leaving answer
    earned, from its records: a TF-093
    ``PolicyResult`` or TF-094 ``ConfiguredResult``
    (``result``), and/or a ``SalvageResult``
    (``salvage_result``) whose TF-097 rescue record
    states the salvage leg. Marks are returned in
    MARK_KINDS order, at most one per kind.

    Raises ``ValueError`` on a refused result (a
    block is TF-095's telling, not a mark), on an
    empty salvage result, and when neither record
    is supplied."""
    if result is None and salvage_result is None:
        raise ValueError(
            "no record to mark: give a policy/configured "
            "result and/or a salvage result")
    verdict = getattr(result, "verdict", None) if result is not None else None
    if verdict == "refused":
        raise ValueError(
            "no quality marks for a blocked answer: verdict "
            "is 'refused' — TF-095's refusal notice tells a "
            "block, and a block is never dressed in marks")

    record = getattr(salvage_result, "rescue_record", None)
    final = (record or {}).get("final") or {}
    final_source = final.get("final_source")
    if salvage_result is not None and result is None:
        if final_source == "empty" or (
                record is None and not (salvage_result.text or "")):
            raise ValueError(
                "no quality marks for an empty salvage "
                "result: there is no answer to mark")

    by_kind: dict[str, QualityMark] = {}

    if result is not None:
        cited = _cited_mark(result)
        if cited is not None:
            by_kind["cited"] = cited

    # Rescued — policy mechanisms in check order,
    # then salvage's adopted pass.
    mechanisms: list[dict[str, Any]] = []
    if result is not None:
        for outcome in getattr(result, "checks", ()) or ():
            if outcome.status == "rescued" and \
                    outcome.id in RESCUE_LABELS:
                mechanisms.append({
                    "mechanism": outcome.id,
                    "label": RESCUE_LABELS[outcome.id],
                    "source": "policy_check"})
    if final_source == "refine_pass":
        mechanisms.append({
            "mechanism": "salvage_refine",
            "label": RESCUE_LABELS["salvage_refine"],
            "source": "salvage_rescue_record",
            "final_pass": final.get("final_pass")})
    if mechanisms:
        segments = []
        for entry in mechanisms:
            if entry["mechanism"] == "salvage_refine":
                segments.append(
                    f"{entry['label']} (pass {entry['final_pass']})")
            else:
                segments.append(entry["label"])
        by_kind["rescued"] = QualityMark(
            "rescued", "rescued: " + "; ".join(segments),
            {"mechanisms": mechanisms})

    if result is not None:
        flags = tuple(getattr(result, "flags", ()) or ())
        if verdict == "flagged" and flags:
            by_kind["flagged"] = QualityMark(
                "flagged",
                "flagged: " + ", ".join(flags),
                {"flags": list(flags),
                 "reasons": [_flag_reason(f) for f in flags]})
        warnings = tuple(getattr(result, "warnings", ()) or ())
        if warnings:
            by_kind["warning"] = QualityMark(
                "warning",
                "warnings: " + ", ".join(warnings),
                {"flags": list(warnings),
                 "reasons": [_flag_reason(f) for f in warnings],
                 "base_verdict": getattr(
                     result, "base_verdict", None)})

    return QualityMarks(
        marks=tuple(by_kind[k] for k in MARK_KINDS if k in by_kind),
        policy_verdict=verdict,
        strictness=getattr(result, "strictness", None)
        if result is not None else None,
        disposition=getattr(result, "disposition", None)
        if result is not None else None,
        salvage_final_source=final_source)


def marks_line(marks: QualityMarks) -> str:
    """The one filed plain-text rendering: a
    consumer surface that wants words shows this
    line beside the answer. An answer with no
    marks renders as the empty string — the
    consumer shows nothing, by design."""
    if not marks.has_marks:
        return ""
    return "Forge marks: " + "; ".join(
        m.label for m in marks.marks) + "."


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic material throughout (TF-093 / TF-094's
# filed cases and TF-097's filed salvage fixtures,
# reused — never canon). ``source`` names what the
# case is built from: policy (TF-093), configured
# (TF-094), salvage (TF-097's run_rescue_fixture),
# combined (a policy result + a salvage result),
# or synthetic_salvage (a SalvageResult built by
# hand, to pin the no-record leg no forge() result
# can produce). ``expected_error`` cases must
# raise ValueError.

_EN1 = "The ferry leaves at dawn and the tide is low."
_EN2 = "The keeper lights the lantern when the ferry is late."
_ES1 = "El faro está en la costa y la marea es alta."
_ES2 = "La campana suena cuando el barco llega a la costa."
_BELL = "The bell rings across the harbour at dawn."
_ODIN_UNCITED = ("Odin is the chief god of Norse mythology, "
                 "a god of wisdom and war.")
_ODIN_CITED = "Odin [Q123] is the chief god of Norse mythology and wisdom."
_CHUNK_CITED = "Lantern oil is kept in the hollow stores [ref:chunk_7]."
_WEATHER = "The weather over the fjord is calm and bright today."

MARKS_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "clean_plain_no_marks",
     "source": "policy",
     "text": f"{_EN1} {_EN2}",
     "signals": {},
     "kinds": (),
     "line": ""},
    {"id": "cited_entity_per_claim",
     "source": "policy",
     "text": _ODIN_CITED,
     "signals": {"records": "lantern"},
     "kinds": ("cited",),
     "cited_markers": ["Q123"],
     "cited_mode": "per_claim"},
    {"id": "cited_reference_chunk_per_claim",
     "source": "policy",
     "text": _CHUNK_CITED,
     "signals": {"records": "lantern"},
     "kinds": ("cited",),
     "cited_markers": ["ref:chunk_7"],
     "cited_mode": "per_claim"},
    {"id": "no_claim_pass_no_cited_mark",
     "source": "policy",
     "text": _WEATHER,
     "signals": {"records": "lantern", "retrieved_qids": ["Q123"]},
     "kinds": (),
     "line": ""},
    {"id": "legacy_floor_cited",
     "source": "policy",
     "text": _ODIN_CITED,
     "signals": {"retrieved_qids": ["Q123"]},
     "kinds": ("cited",),
     "cited_markers": ["Q123"],
     "cited_mode": "legacy_floor"},
    {"id": "loop_rescued",
     "source": "policy",
     "text": " ".join([_BELL] * 4),
     "signals": {},
     "kinds": ("rescued",),
     "mechanisms": ["loop"]},
    {"id": "format_rescued",
     "source": "policy",
     "text": '{"ferry": "dawn",}',
     "signals": {"structured": True},
     "kinds": ("rescued",),
     "mechanisms": ["format"]},
    {"id": "truncation_rescued",
     "source": "policy",
     "text": "The ferry leaves at dawn and the tide is low. The keeper lights",
     "signals": {"truncation_suspect": True},
     "kinds": ("rescued",),
     "mechanisms": ["truncation"]},
    {"id": "drift_rescued",
     "source": "policy",
     "text": f"{_EN1} {_EN2} {_ES1} {_ES2}",
     "signals": {"expected_language": "en"},
     "kinds": ("rescued",),
     "mechanisms": ["drift"]},
    {"id": "citation_flagged_not_cited",
     "source": "policy",
     "text": _ODIN_UNCITED,
     "signals": {"records": "lantern"},
     "kinds": ("flagged",),
     "flags": ["citation:uncited_claim"]},
    {"id": "too_short_flagged",
     "source": "policy",
     "text": "Too short.",
     "signals": {},
     "kinds": ("flagged",),
     "flags": ["length:too_short"]},
    {"id": "generic_flagged",
     "source": "policy",
     "text": "As an AI language model I cannot provide that information today.",
     "signals": {},
     "kinds": ("flagged",),
     "flags": ["genericness:hard_generic_phrase"]},
    {"id": "drift_flagged",
     "source": "policy",
     "text": f"{_ES1} {_ES2}",
     "signals": {"expected_language": "en"},
     "kinds": ("flagged",),
     "flags": ["drift:whole_answer_off_language"]},
    {"id": "rescued_and_flagged",
     "source": "policy",
     "text": " ".join([_BELL] * 3) + " " + _ODIN_UNCITED,
     "signals": {"records": "lantern"},
     "kinds": ("rescued", "flagged"),
     "mechanisms": ["loop"],
     "flags": ["citation:uncited_claim"]},
    {"id": "relaxed_citation_warning",
     "source": "configured",
     "text": _ODIN_UNCITED,
     "signals": {"records": "lantern"},
     "strictness": "relaxed",
     "kinds": ("warning",),
     "flags": ["citation:uncited_claim"]},
    {"id": "strict_blocked_raises",
     "source": "configured",
     "text": _ODIN_UNCITED,
     "signals": {"records": "lantern"},
     "strictness": "strict",
     "expected_error": True},
    {"id": "refused_empty_raises",
     "source": "policy",
     "text": "   ",
     "signals": {},
     "expected_error": True},
    {"id": "salvage_refine_adopted",
     "source": "salvage",
     "salvage_fixture": "refine_adopted",
     "kinds": ("rescued",),
     "mechanisms": ["salvage_refine"],
     "final_pass": 1},
    {"id": "salvage_best_draft_no_marks",
     "source": "salvage",
     "salvage_fixture": "best_draft_no_engine",
     "kinds": (),
     "line": ""},
    {"id": "salvage_rejected_attempt_no_mark",
     "source": "salvage",
     "salvage_fixture": "refine_rejected_fabrication",
     "kinds": (),
     "line": ""},
    {"id": "salvage_adopt_then_fabricate_names_adopted_pass",
     "source": "salvage",
     "salvage_fixture": "refine_adopt_then_fabricate",
     "kinds": ("rescued",),
     "mechanisms": ["salvage_refine"],
     "final_pass": 1},
    {"id": "salvage_empty_raises",
     "source": "salvage",
     "salvage_fixture": "empty_candidates",
     "expected_error": True},
    {"id": "combined_policy_and_salvage_rescued",
     "source": "combined",
     "text": " ".join([_BELL] * 4),
     "signals": {},
     "salvage_fixture": "refine_adopted",
     "kinds": ("rescued",),
     "mechanisms": ["loop", "salvage_refine"],
     "final_pass": 1},
    {"id": "salvage_no_record_no_mark",
     "source": "synthetic_salvage",
     "kinds": (),
     "line": ""},
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


def _build_fixture(fixture: dict[str, Any]) -> QualityMarks:
    """Build one filed fixture's marks from its
    real records (the live checker's road; the
    tests use the same builder)."""
    result = None
    salvage_result = None
    if fixture["source"] in ("policy", "configured", "combined"):
        signals = dict(fixture["signals"])
        if signals.get("records") == "lantern":
            signals["records"] = _fixture_records()
        if fixture["source"] == "configured":
            from thoughtforge.refinement.policy_config import (
                enforce_configured,
            )

            result = enforce_configured(
                fixture["text"],
                strictness=fixture["strictness"], **signals)
        else:
            from thoughtforge.refinement.policy_surface import (
                enforce_policy,
            )

            result = enforce_policy(fixture["text"], **signals)
    if fixture["source"] in ("salvage", "combined"):
        from thoughtforge.refinement.rescue_provenance import (
            run_rescue_fixture,
        )

        salvage_result = run_rescue_fixture(fixture["salvage_fixture"])
    if fixture["source"] == "synthetic_salvage":
        from thoughtforge.refinement.salvage import SalvageResult

        salvage_result = SalvageResult(
            text="The ferry leaves at dawn.", citations=["Q123"],
            confidence=0.5, passes_used=1, best_draft_score=0.5,
            fragments_kept=1, salvage_path="refine_pass_1",
            rescue_record=None)
    return quality_marks(result, salvage_result=salvage_result)


# ── Checkers ────────────────────────────────────────────────────────────────


def _estate_flag_codes() -> set[str]:
    """Every flag the shipped estate's filed
    fixtures can put on a leaving (flagged) or
    warning result — the coverage the flag
    reasons must hold, by TF-095's catalogue."""
    from thoughtforge.refinement.policy_config import CONFIG_FIXTURES
    from thoughtforge.refinement.policy_surface import POLICY_FIXTURES

    codes: set[str] = set()
    for fixture in POLICY_FIXTURES:
        codes.update(fixture.get("flags") or ())
    for fixture in CONFIG_FIXTURES:
        codes.update(fixture.get("flags") or ())
    return codes


def quality_marks_problems() -> list[str]:
    """Data-integrity checker: rule / kind /
    fixture ids unique and in their vocabularies,
    RESCUE_LABELS covering every rescuable policy
    check plus salvage, every estate flag carrying
    a TF-095 catalogued reason, fixtures
    well-formed (mark expectations naming kinds
    and their filed details)."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in MARKS_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate MARKS_RULES ids")
    if MARK_KINDS != ("cited", "rescued", "flagged", "warning"):
        problems.append("MARK_KINDS drifted from the four filed kinds")
    for check_id in _RESCUABLE_CHECKS:
        if check_id not in RESCUE_LABELS:
            problems.append(
                f"rescuable check {check_id} has no RESCUE_LABELS entry")
    if "salvage_refine" not in RESCUE_LABELS:
        problems.append("salvage_refine has no RESCUE_LABELS entry")
    from thoughtforge.refinement.refusal_honesty import (
        REASONS,
        _reason_key,
    )

    for code in sorted(_estate_flag_codes()):
        if _reason_key(code) not in REASONS:
            problems.append(
                f"estate flag {code!r} has no TF-095 catalogued reason")
    fixture_ids = [f["id"] for f in MARKS_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate MARKS_FIXTURES ids")
    for fixture in MARKS_FIXTURES:
        if fixture["source"] not in (
                "policy", "configured", "salvage", "combined",
                "synthetic_salvage"):
            problems.append(
                f"fixture {fixture['id']}: unknown source "
                f"{fixture['source']!r}")
        if fixture.get("expected_error"):
            if "kinds" in fixture:
                problems.append(
                    f"fixture {fixture['id']}: an error case "
                    "files mark kinds")
            continue
        for kind in fixture["kinds"]:
            if kind not in MARK_KINDS:
                problems.append(
                    f"fixture {fixture['id']}: kind {kind!r} "
                    "not in the vocabulary")
        if list(fixture["kinds"]) != [
                k for k in MARK_KINDS if k in fixture["kinds"]]:
            problems.append(
                f"fixture {fixture['id']}: kinds are not in "
                "MARK_KINDS order")
        if "cited" in fixture["kinds"] and \
                not fixture.get("cited_markers"):
            problems.append(
                f"fixture {fixture['id']}: a cited case files "
                "no markers")
        if "rescued" in fixture["kinds"] and \
                not fixture.get("mechanisms"):
            problems.append(
                f"fixture {fixture['id']}: a rescued case "
                "files no mechanisms")
        if ("flagged" in fixture["kinds"] or
                "warning" in fixture["kinds"]) and \
                not fixture.get("flags"):
            problems.append(
                f"fixture {fixture['id']}: a flagged/warning "
                "case files no flags")
    return problems


def quality_marks_live_problems() -> list[str]:
    """Live checker: every filed fixture is built
    from its real records and its marks must land
    exactly as filed — kinds in order, cited
    markers and mode, rescue mechanisms and the
    adopted pass, flags verbatim with catalogued
    reasons, the filed line for mark-less cases,
    ValueError on the filed error cases — and no
    result's text is altered by being marked."""
    problems: list[str] = []
    for fixture in MARKS_FIXTURES:
        if fixture.get("expected_error"):
            try:
                _build_fixture(fixture)
                problems.append(
                    f"fixture {fixture['id']}: a blocked or "
                    "empty record did not raise")
            except ValueError:
                pass
            continue
        built = _build_fixture(fixture)
        if built.kinds() != tuple(fixture["kinds"]):
            problems.append(
                f"fixture {fixture['id']}: kinds "
                f"{built.kinds()} != {fixture['kinds']}")
            continue
        if built.has_marks != bool(fixture["kinds"]):
            problems.append(
                f"fixture {fixture['id']}: has_marks drifted")
        if "line" in fixture and marks_line(built) != fixture["line"]:
            problems.append(
                f"fixture {fixture['id']}: line "
                f"{marks_line(built)!r} != {fixture['line']!r}")
        if built.has_marks and not marks_line(built).startswith(
                "Forge marks: "):
            problems.append(
                f"fixture {fixture['id']}: a marked line does "
                "not open with the filed prefix")
        if "cited" in fixture["kinds"]:
            detail = built.mark("cited").detail
            if detail["markers"] != fixture["cited_markers"] or \
                    detail["mode"] != fixture["cited_mode"]:
                problems.append(
                    f"fixture {fixture['id']}: cited detail "
                    f"{detail} != filed markers/mode")
        if "mechanisms" in fixture:
            detail = built.mark("rescued").detail
            names = [m["mechanism"] for m in detail["mechanisms"]]
            if names != fixture["mechanisms"]:
                problems.append(
                    f"fixture {fixture['id']}: mechanisms "
                    f"{names} != {fixture['mechanisms']}")
            if "final_pass" in fixture:
                salvage = [m for m in detail["mechanisms"]
                           if m["mechanism"] == "salvage_refine"]
                if not salvage or \
                        salvage[0]["final_pass"] != fixture["final_pass"]:
                    problems.append(
                        f"fixture {fixture['id']}: the adopted "
                        "pass is not named as filed")
        if "flags" in fixture:
            kind = "flagged" if "flagged" in fixture["kinds"] \
                else "warning"
            detail = built.mark(kind).detail
            if detail["flags"] != fixture["flags"]:
                problems.append(
                    f"fixture {fixture['id']}: flags "
                    f"{detail['flags']} != {fixture['flags']}")
            for reason in detail["reasons"]:
                if not reason["catalogued"]:
                    problems.append(
                        f"fixture {fixture['id']}: flag "
                        f"{reason['flag']!r} rides uncatalogued")
    return problems


def quality_marks_as_data() -> dict[str, Any]:
    return {
        "format": MARKS_FORMAT,
        "marks_doc": MARKS_DOC,
        "kinds": list(MARK_KINDS),
        "rules": [dict(r) for r in MARKS_RULES],
        "rescue_labels": dict(RESCUE_LABELS),
        "fixtures": [
            {k: v for k, v in f.items()} for f in MARKS_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_marks": "not_proven",
            "policy_surface": "unchanged",
            "policy_config": "unchanged",
            "refusal_honesty": "catalogue_reused_unchanged",
            "rescue_provenance": "unchanged",
            "fragment_salvage": "unchanged",
            "ui": "consumers_own_not_built_here",
        },
    }
