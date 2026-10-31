"""
Cite-or-explain II — the enforcement side (TF-049).

TF-048 guaranteed the retrieval side: every retrieved item
carries its citation payload (``CitationPayload`` — marker,
source, record identity) into the prompt by construction.
This slice is the other half of the roadmap's rule: *a
post-generation check — claims matching stored facts must
cite, or the answer is flagged/regenerated per policy (with
TF-093).*

The measured starting state, before this slice:

  * The only citation check in the estate was
    ``EnforcementGate.check``'s answer-level QID rule: if any
    Wikidata QIDs were retrieved, at least ONE of them had to
    appear somewhere in the answer. It was **claim-blind** (an
    answer could state a second stored fact uncited and pass
    on the strength of an unrelated citation), **kind-blind**
    (reference chunks, GeoNames places, ConceptNet edges and
    every fallback kind carry TF-048 markers but had no check
    at all — a bundle with no QIDs skipped citation
    enforcement entirely), and **identity-blind** (a citation
    to a QID that was never retrieved — a fabricated citation
    — was simply ignored).
  * The core carried a second, duplicate minimal gate
    (``_check_citation_gate``, the repair path) with the same
    any-one-QID logic, and ``_extract_citations`` could only
    ever extract QIDs.

What this slice builds:

  * ``check_citations(text, records)`` — the per-claim check.
    The answer is split into claims (sentences, the tree's own
    ``split_sentences``). A claim **matches** a retrieved
    record under the stated ``MATCH_RULES`` (the record's
    label phrase in the claim; a label term plus another
    shared distinctive term; or ≥ ``MIN_SHARED_TERMS``
    shared distinctive terms between the claim and the
    record's claim text — its label, description/content,
    and cue). A matched claim is **cited** only when that
    record's own TF-048 marker appears in that claim (a bare
    QID, or the bracketed ``[marker]`` the prompt taught).
    A citation appearing in a *different* sentence does not
    cover this claim — that is the difference between
    cite-or-explain and cite-somewhere.
  * The result is data (``CitationCheck`` / ``ClaimCheck``):
    every claim, what it matched, what it cited, what it left
    uncited, and any **foreign markers** — bracketed/QID
    citations in the answer that no retrieved record owns.
    ``passed`` is True exactly when no matched claim is
    uncited and no foreign marker appears.
  * ``CITATION_POLICY`` states the policy as data. The
    shipped default action on an uncited claim is **flag**:
    the EnforcementGate fails the answer's citation check,
    appends its forge note naming the uncited markers, and —
    through TF-047 — the turn's writeback gate closes, so an
    uncited answer teaches the store nothing. **Regeneration
    is a named policy option, not the default and not wired
    here**: the estate's only regeneration path (the core's
    quality-gate repair) builds its prompt with NO knowledge
    context (TF-023's stated narrowing), so a regenerated
    answer cannot cite what it was never shown; wiring
    regeneration under one policy surface is TF-093's
    unification, which this slice's policy table is shaped
    for. ``regeneration_required`` is therefore always False
    under the shipped policy, and the checker pins that.
  * Wiring: ``EnforcementGate.check`` takes the bundle's
    records as an optional argument and ANDs the per-claim
    verdict into its citation check (the legacy any-QID floor
    is unchanged when no records are supplied, and unchanged
    as a floor when they are — relaxing it is TF-093's call,
    recorded in the policy, not taken silently here). The
    result carries the ``CitationCheck`` as ``claim_check``.
    The core passes its bundle records in ``_compose_final``,
    retains the check on ``core.last_citation_check``
    (post-freeze minor addition, recorded like TF-047's), and
    the repair path's ``_check_citation_gate`` now delegates
    to the same checker plus the same floor — the duplicate
    logic is one logic. ``_extract_citations`` extracts every
    kind's marker, not only QIDs (entity markers ARE QIDs, so
    its pinned behaviour for entities is unchanged).

Honesty boundaries, stated: claim matching is a deterministic
term-overlap heuristic, not semantic entailment — a claim
that paraphrases a stored fact below the stated thresholds
is not caught, and a claim sharing terms by coincidence can
be flagged (flagging is the safe direction: the user sees the
note, and TF-095 owns refusal wording). The knowledge-only
path does not run this check: it presents retrieved records
as retrieved knowledge and cites record ids by construction
(TF-048) — there is no generated claim to police. Proof
grade is FIXTURE (synthetic records + the TF-043 corpus +
stub-backend core turns, VM); a live model's citation
behaviour is NOT PROVEN.

The human-readable write-up is
``docs/architecture/CITATION_ENFORCEMENT.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from thoughtforge.knowledge.citations import citation_for_record
from thoughtforge.knowledge.models import ActivatedRecord
from thoughtforge.utils.common import split_sentences as _split_sentences

# ── The matching rule, as data ───────────────────────────────────────────────

#: Distinctive-term overlap at or above this count, between a
#: claim and a record's claim text, is a match on its own (the
#: ``term_overlap`` rule). The other two rules in MATCH_RULES
#: are label-anchored and need fewer shared terms.
MIN_SHARED_TERMS: int = 3

#: Words that carry no claim identity — shared stopwords never
#: count toward a match.
STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "and", "or", "in", "on", "at", "by", "for",
    "with", "as", "it", "its", "this", "that", "these", "those",
    "has", "have", "had", "from", "into", "over", "under",
})

MATCH_RULES: tuple[dict[str, Any], ...] = (
    {"name": "label_phrase",
     "rule": "the record's full label phrase appears in the claim"},
    {"name": "label_plus_term",
     "rule": "a distinctive label term appears in the claim and at "
             "least one further distinctive term is shared with the "
             "record's claim text"},
    {"name": "term_overlap",
     "rule": f"at least {MIN_SHARED_TERMS} distinctive terms are "
             "shared between the claim and the record's claim text"},
)

# ── The policy, as data ──────────────────────────────────────────────────────

CITATION_POLICY: dict[str, Any] = {
    "on_uncited_claim": "flag",
    "actions": ("pass", "flag", "regenerate"),
    "flag_means": (
        "the EnforcementGate fails the answer's citation check and "
        "appends a forge note naming the uncited markers; through "
        "TF-047 the flagged turn's writeback gate closes, so the "
        "answer teaches the store nothing"
    ),
    "on_foreign_marker": "flag",
    "on_no_matching_claims": (
        "pass — nothing in the answer claims a stored fact, so "
        "there is nothing to cite; the legacy any-QID floor in "
        "EnforcementGate is a separate, older rule and is NOT "
        "relaxed by this slice (TF-093 owns reconciling the two)"
    ),
    "regeneration": (
        "a named option, not the shipped default and not wired: "
        "the core's only regeneration path (the quality-gate "
        "repair pass) builds its prompt with no knowledge context "
        "(TF-023), so a regenerated answer cannot cite what it "
        "was never shown; TF-093 owns wiring regeneration under "
        "the unified policy surface"
    ),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BRACKET_RE = re.compile(r"\[([^\]]{1,120})\]")
_QID_RE = re.compile(r"\bQ\d{3,}\b")


def _fold(token: str) -> str:
    """A crude singular fold (stores→store), stated once: a
    trailing 's' on a token longer than 3 chars, unless the
    token ends in 'ss'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def distinctive_terms(text: str) -> set[str]:
    """The claim-identity terms of a text: lowercase word
    tokens of length ≥ 3, singular-folded, minus STOPWORDS."""
    return {
        _fold(tok)
        for tok in _TOKEN_RE.findall((text or "").lower())
        if len(tok) >= 3 and tok not in STOPWORDS
    }


def _label_text(record: ActivatedRecord) -> str:
    payload = citation_for_record(record)
    if payload.label:
        return payload.label
    raw = getattr(record, "raw", None) or {}
    return str(raw.get("label_en") or raw.get("title") or "")


def record_claim_text(record: ActivatedRecord) -> str:
    """Everything a claim can match against, for one record:
    its label, its description/content, and its cue."""
    raw = getattr(record, "raw", None) or {}
    parts = [
        _label_text(record),
        str(raw.get("description_en") or ""),
        str(raw.get("content") or ""),
        str(getattr(record, "cue", "") or ""),
    ]
    return " ".join(p for p in parts if p)


def cited_markers_in(text: str) -> set[str]:
    """Every citation marker a text carries: bracketed
    ``[marker]`` contents plus bare QIDs (the entity shape
    predates brackets and still counts)."""
    markers = {m.strip() for m in _BRACKET_RE.findall(text or "") if m.strip()}
    markers.update(_QID_RE.findall(text or ""))
    return markers


def _foreign_markers(text: str, known: set[str]) -> list[str]:
    """Cited markers no retrieved record owns — fabricated
    citations. Marker-shaped only (a QID, or a bracketed
    ``prefix:rest`` token with no sentence punctuation); the
    EnforcementGate's own ``[Forge: …]`` notes are not
    citations and are excluded by name."""
    foreign: list[str] = []
    for marker in sorted(cited_markers_in(text)):
        if marker in known:
            continue
        if marker.lower().startswith("forge:"):
            continue
        if _QID_RE.fullmatch(marker):
            foreign.append(marker)
        elif ":" in marker and not marker.endswith(".") and " " not in marker:
            foreign.append(marker)
    return foreign


def match_claim_to_record(
    claim_text: str, record: ActivatedRecord
) -> dict[str, Any] | None:
    """Whether one claim matches one record, and by which
    rule in MATCH_RULES. None when it does not match."""
    claim_terms = distinctive_terms(claim_text)
    if not claim_terms:
        return None
    record_terms = distinctive_terms(record_claim_text(record))
    shared = claim_terms & record_terms
    label = _label_text(record)
    label_terms = distinctive_terms(label)
    marker = citation_for_record(record).marker

    label_phrase = " ".join(_TOKEN_RE.findall(label.lower()))
    claim_phrase = " ".join(_TOKEN_RE.findall((claim_text or "").lower()))
    if label_phrase and len(label_phrase.split()) >= 2 and label_phrase in claim_phrase:
        return {"record_id": record.record_id, "marker": marker,
                "rule": "label_phrase", "shared_terms": sorted(shared)}
    if label_terms & claim_terms and len(shared) >= 2:
        return {"record_id": record.record_id, "marker": marker,
                "rule": "label_plus_term", "shared_terms": sorted(shared)}
    if len(shared) >= MIN_SHARED_TERMS:
        return {"record_id": record.record_id, "marker": marker,
                "rule": "term_overlap", "shared_terms": sorted(shared)}
    return None


# ── The check, as data ───────────────────────────────────────────────────────


@dataclass
class ClaimCheck:
    """One claim (sentence) of an answer, checked."""

    text: str
    matched: list[dict[str, Any]] = field(default_factory=list)
    cited_markers: list[str] = field(default_factory=list)
    uncited_record_ids: list[str] = field(default_factory=list)

    @property
    def matched_record_ids(self) -> list[str]:
        return [m["record_id"] for m in self.matched]


@dataclass
class CitationCheck:
    """The whole answer's cite-or-explain verdict, as data."""

    claims: list[ClaimCheck] = field(default_factory=list)
    foreign_markers: list[str] = field(default_factory=list)
    action: str = "pass"

    @property
    def matched_record_ids(self) -> list[str]:
        seen: list[str] = []
        for claim in self.claims:
            for record_id in claim.matched_record_ids:
                if record_id not in seen:
                    seen.append(record_id)
        return seen

    @property
    def uncited_record_ids(self) -> list[str]:
        seen: list[str] = []
        for claim in self.claims:
            for record_id in claim.uncited_record_ids:
                if record_id not in seen:
                    seen.append(record_id)
        return seen

    @property
    def uncited_markers(self) -> list[str]:
        markers: list[str] = []
        for claim in self.claims:
            for match in claim.matched:
                if match["record_id"] in claim.uncited_record_ids \
                        and match["marker"] not in markers:
                    markers.append(match["marker"])
        return markers

    @property
    def passed(self) -> bool:
        """True exactly when no matched claim is uncited and
        the answer carries no foreign (fabricated) marker."""
        return not self.uncited_record_ids and not self.foreign_markers

    @property
    def regeneration_required(self) -> bool:
        """Under the shipped policy (flag) this is always
        False — see CITATION_POLICY['regeneration']."""
        return self.action == "regenerate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "action": self.action,
            "matched_record_ids": self.matched_record_ids,
            "uncited_record_ids": self.uncited_record_ids,
            "uncited_markers": self.uncited_markers,
            "foreign_markers": list(self.foreign_markers),
            "claims": [
                {
                    "text": c.text,
                    "matched": c.matched,
                    "cited_markers": list(c.cited_markers),
                    "uncited_record_ids": list(c.uncited_record_ids),
                }
                for c in self.claims
            ],
        }


def check_citations(
    text: str, records: list[ActivatedRecord]
) -> CitationCheck:
    """
    The post-generation cite-or-explain check: every claim in
    ``text`` that matches a retrieved record must cite that
    record's TF-048 marker inside the claim itself.
    """
    records = list(records or [])
    known_markers = {citation_for_record(r).marker for r in records}
    check = CitationCheck(
        foreign_markers=_foreign_markers(text or "", known_markers),
    )
    sentences = [s.strip() for s in _split_sentences(text or "") if s.strip()]
    if not sentences and (text or "").strip():
        sentences = [(text or "").strip()]
    for sentence in sentences:
        cited_here = cited_markers_in(sentence)
        claim = ClaimCheck(
            text=sentence,
            cited_markers=sorted(cited_here & known_markers),
        )
        for record in records:
            match = match_claim_to_record(sentence, record)
            if match is None:
                continue
            claim.matched.append(match)
            if match["marker"] not in cited_here:
                claim.uncited_record_ids.append(record.record_id)
        check.claims.append(claim)
    if not check.passed:
        check.action = str(CITATION_POLICY["on_uncited_claim"])
    return check


# ── Self-check ───────────────────────────────────────────────────────────────


def citation_enforcement_problems() -> list[str]:
    """
    Re-check the cite-or-explain enforcement leg against the
    live tree. Returns problems by name; ``[]`` when the
    per-claim check, its policy, and the gate/core wiring
    still match the code.
    """
    import inspect

    problems: list[str] = []

    # The policy table is well-formed and the shipped action
    # is the stated flag, with regeneration unwired under it.
    if CITATION_POLICY.get("on_uncited_claim") != "flag":
        problems.append("CITATION_POLICY default action is not the shipped 'flag'")
    if "regenerate" not in CITATION_POLICY.get("actions", ()):
        problems.append("CITATION_POLICY no longer names regeneration as an option")
    rule_names = [row["name"] for row in MATCH_RULES]
    if rule_names != ["label_phrase", "label_plus_term", "term_overlap"]:
        problems.append(f"MATCH_RULES drifted: {rule_names}")

    # Synthetic records, one per citation kind that the legacy
    # gate could not see at all.
    entity = ActivatedRecord(
        "Q123", "entity", 0.9, "Odin chief god of Norse mythology",
        raw={"qid": "Q123", "label_en": "Odin",
             "description_en": "chief god of Norse mythology",
             "source": "wikidata"},
    )
    chunk = ActivatedRecord(
        "chunk_7", "reference_chunk", 0.8, "lantern oil kept in the hollow stores",
        raw={"chunk_id": "chunk_7", "source_file": "hollow.md",
             "title": "The Hollow",
             "content": "lantern oil is kept in the hollow stores",
             "source": "reference"},
    )
    records = [entity, chunk]

    cited = check_citations(
        "Odin is the chief god of Norse mythology [Q123]. "
        "Lantern oil is kept in the hollow stores [ref:chunk_7].",
        records,
    )
    if not cited.passed or cited.uncited_record_ids:
        problems.append(f"cited synthetic answer does not pass: {cited.to_dict()}")
    uncited = check_citations(
        "Odin is the chief god of Norse mythology. "
        "Lantern oil is kept in the hollow stores.",
        records,
    )
    if uncited.passed or set(uncited.uncited_record_ids) != {"Q123", "chunk_7"}:
        problems.append(f"uncited synthetic answer not flagged per record: {uncited.to_dict()}")
    if uncited.action != "flag" or uncited.regeneration_required:
        problems.append("uncited answer's policy action is not the shipped flag")
    elsewhere = check_citations(
        "Lantern oil is kept in the hollow stores. Odin [Q123] is a god.",
        records,
    )
    if "chunk_7" not in elsewhere.uncited_record_ids:
        problems.append("a citation in another sentence covered a claim it does not cite")
    no_claim = check_citations(
        "The weather over the fjord is calm and bright today.", records,
    )
    if not no_claim.passed or no_claim.matched_record_ids:
        problems.append("an answer claiming no stored fact was flagged or matched")
    foreign = check_citations(
        "Odin is the chief god of Norse mythology [Q999].", records,
    )
    if foreign.passed or foreign.foreign_markers != ["Q999"]:
        problems.append(f"foreign citation not flagged: {foreign.to_dict()}")

    # The gate: with records, an uncited matched claim fails
    # the citation check and the note names the marker; without
    # records the legacy any-QID behaviour is byte-identical.
    from thoughtforge.refinement.enforcement import EnforcementGate

    gate = EnforcementGate()
    flagged = gate.check(
        "Odin is the chief god of Norse mythology, a god of wisdom.",
        [], {"Q123"}, records=records,
    )
    if flagged.citation_check or flagged.claim_check is None \
            or "Q123" not in flagged.notes:
        problems.append("EnforcementGate no longer flags an uncited matched claim by marker")
    legacy = gate.check("No citations in this response.", [], {"Q123"})
    if legacy.citation_check or legacy.claim_check is not None:
        problems.append("EnforcementGate's no-records legacy behaviour drifted")

    # Core wiring: compose passes the bundle records, the
    # repair gate delegates to this checker, and the check is
    # retained on the core.
    from thoughtforge.cognition import core as core_module

    compose_src = inspect.getsource(core_module.ThoughtForgeCore._compose_final)
    if "records=bundle.activated_records" not in compose_src:
        problems.append("core wiring: _compose_final does not pass the bundle records to enforcement")
    gate_src = inspect.getsource(core_module._check_citation_gate)
    if "check_citations" not in gate_src:
        problems.append("core wiring: _check_citation_gate no longer delegates to check_citations")
    if not isinstance(
        getattr(core_module.ThoughtForgeCore, "last_citation_check", None), property
    ):
        problems.append("core wiring: ThoughtForgeCore.last_citation_check is not a property")

    return problems


__all__ = [
    "CITATION_POLICY",
    "MATCH_RULES",
    "MIN_SHARED_TERMS",
    "STOPWORDS",
    "CitationCheck",
    "ClaimCheck",
    "check_citations",
    "citation_enforcement_problems",
    "cited_markers_in",
    "distinctive_terms",
    "match_claim_to_record",
    "record_claim_text",
]
