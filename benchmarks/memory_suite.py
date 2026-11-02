"""
Memory fidelity suite, as data (TF-123).

The roadmap's rule: *memory fidelity suite:
cite-or-explain scored mechanically (citations
present, citations true).* TF-121's harness filed the
architecture, TF-122 built the persona suite, and the
harness's ``retrieval_precision`` entry has said since
TF-121 that TF-123 consumes it. This slice is that
consumer.

The measured starting state, probed live this tick on
this tree before anything was built:

  * **No memory fidelity scorer existed anywhere.**
    The benchmarks estate held the persona scorer +
    suite and the profile runner; TF-043's retrieval
    fixture was bound in the harness as a neighbour
    with no suite consuming it, and TF-049's doc
    (§5) filed that "TF-123 will score this properly"
    — the scoring was assigned, never built.
  * **Citations true was scored by nothing — probed.**
    TF-049's ``check_citations`` judges presence only:
    the scripted claim "the lantern hollow ferry
    departs at midnight [ref:4301]" — citing TF-043's
    real timetable chunk, whose stored fact is a dawn
    departure — PASSED ``check_citations`` identically
    to the true dawn claim. A citation can be present,
    well-formed, and resolved, and the claim it backs
    can still be false; no checker in the estate
    could tell the two apart. Filed as D-BENCH-9 and
    fixed in scope by this suite's truth axis.
  * **The explain side was scored by nothing —
    probed.** TF-049's policy passes an answer that
    matches no stored fact ("nothing to cite"), and
    TF-093's unified surface carries that rule
    unchanged: an answer owning nothing — no citation,
    no statement that the reasoning is the model's
    own — passes every citation check in the estate.
    Cite-or-explain's second leg (answers not from
    memory own themselves) had no scorer.
  * What already worked, and is REUSED here, never
    copied: TF-048's citation payloads (every filed
    record carries its marker by construction),
    TF-049's per-claim checker (presence, per-sentence
    citation, foreign/fabricated markers — probed:
    ``[ref:9999]`` and ``[Q999]`` are both flagged
    foreign, and a claim citing the wrong record is
    flagged uncited for the record it actually
    matches), TF-093's policy surface (the per-claim
    rule as the citation law when records are
    supplied), and TF-043's synthetic corpus +
    pinned baseline (re-measured clean live this
    tick: precision 0.8889 / recall 0.9630 at the
    defaults, ``baseline_problems`` empty).

What this module files, as verifiable data:

  * ``RECORDS`` — the suite's retrieved-record set,
    built programmatically from TF-043's filed
    corpus rows (``FIXTURE_ENTITIES`` /
    ``FIXTURE_REFERENCE_CHUNKS`` /
    ``FIXTURE_CONCEPTNET_EDGES`` /
    ``FIXTURE_GEONAMES`` in
    ``thoughtforge.knowledge.retrieval_precision``,
    imported and reused — never copied, never canon):
    eight ``ActivatedRecord``s, each carrying its
    TF-048 citation payload by construction.
  * ``MEMORY_FACTS`` — six facts of that corpus,
    each filed against one record with the phrases
    a supporting claim must carry (``required``)
    and the phrases that contradict the stored fact
    (``breaking``) — the truth axis's filed rule.
  * ``PRESENT_CASES`` / ``TRUE_CASES`` /
    ``EXPLAIN_CASES`` — 36 scripted cases, 12 per
    axis, each with its filed expectation
    (``holds``), including the deliberately failing
    shapes the task names: a missing citation, a
    citation in the wrong sentence, a cited-but-
    false value, a fabricated citation to a
    non-existent fact, a citation resolving to a
    record that does not support the claim, an
    unowned non-memory assertion, and a memory
    claim disguised as the model's own reasoning.
  * ``run_memory_suite()`` — scores all three axes:
    **present** by TF-049's real ``check_citations``
    (a memory claim holds only when the checker
    passes it — cited in-claim, nothing uncited,
    nothing foreign); **true** by
    ``citation_truth`` (every cited marker resolves
    to a retrieved record, that record matches the
    citing claim under TF-049's real matcher, and
    the claim carries the cited fact's required
    phrases with none of its breaking phrases —
    presence alone is not truth); **explain** by
    the filed ownership rule (a non-memory answer
    holds only when it matches no stored record,
    carries no citation marker, and carries one of
    ``OWNERSHIP_PHRASES``). Per-axis scores plus
    the overall (their mean), PASS / REVIEW by the
    filed threshold (an axis passes at 0.75, the
    suite passes only when every axis passes); an
    empty axis or empty suite scores 0.0 / REVIEW
    — no vacuous pass (the TF-122 precedent).
  * ``memory_suite_problems()`` — the live checker:
    the filing's shapes, the corpora's counts and
    balance, the filed expectations agreeing with
    the scorers case by case, TF-048 / TF-049 /
    TF-093's own checkers clean, and TF-043's live
    leg (its fixture DB built fresh, its pinned
    baseline unregressed, and a real retrieval
    returning the filed records with their TF-048
    markers).

D-BENCH-9 is FIXED by this slice in the scoring
sense TF-049 assigned: citation truth and the
explain side now have a mechanical scorer with a
pinned harness baseline. Enforcement itself is
unchanged — TF-049 / TF-093 still flag presence,
and no gate in the estate yet blocks a cited-but-
false answer at generation time; that wiring is
not claimed here (see the filed limits).

Filed limits, stated plainly:

  * The cases are scripted synthetic fixtures
    (TF-017) in TF-043's invented Lantern Hollow
    world: invented answers judged against invented
    stored facts, never canon, never a model's live
    output, and ``data/`` was never opened. The
    suite proves the *scoring* end to end; a live
    small model's citation behaviour is NOT PROVEN
    here (TF-125 / TF-126's road).
  * Truth is judged by the filed phrase rules
    (required / breaking) plus TF-049's term-overlap
    matcher — the deterministic, checkable subset.
    A false claim phrased to carry the required
    phrases and dodge the breaking ones is not
    caught; semantic entailment is judgement, not
    smuggled into a matcher.
  * The explain axis's ownership phrases are filed
    HERE, as suite data: no estate module defines
    an own-reasoning marker (probed — the scaffold,
    TF-049 and TF-093 name none), so the suite
    files the vocabulary it scores and claims no
    more than that.
  * The synthetic QF-ids of TF-043's corpus do not
    match the QID shape (TF-048's fixture note), so
    entity citations in the scripted cases are
    bracketed — the shape ``cited_markers_in``
    extracts — exactly as TF-049's checker reads
    them.

The human-readable write-up is
``docs/architecture/MEMORY_FIDELITY_SUITE.md``.
Proof grade: FIXTURE — synthetic scripted cases
through the real TF-048 / TF-049 / TF-093 machinery
and TF-043's real retrieval on this VM; no live
model, never claimed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MEMORY_SUITE_FORMAT = "thoughtforge-memory-fidelity-suite/1"
MEMORY_SUITE_DOC = "docs/architecture/MEMORY_FIDELITY_SUITE.md"

#: Per-axis pass threshold (the estate's filed 0.75,
#: TF-122's shape) and the filed corpus scale:
#: 12 cases per axis, 36 in total.
AXIS_PASS_THRESHOLD = 0.75
CASES_PER_AXIS = 12

# The real machinery, reused — never copied: TF-048's
# payloads ride every record by construction; TF-049's
# checker / matcher / marker extraction judge presence
# and truth; TF-043's corpus is the stored-fact world.
from thoughtforge.knowledge.citations import citation_for_record  # noqa: E402
from thoughtforge.knowledge.models import ActivatedRecord  # noqa: E402
from thoughtforge.knowledge.retrieval_precision import (  # noqa: E402
    FIXTURE_CONCEPTNET_EDGES,
    FIXTURE_ENTITIES,
    FIXTURE_GEONAMES,
    FIXTURE_REFERENCE_CHUNKS,
)
from thoughtforge.refinement.citation_enforcement import (  # noqa: E402
    check_citations,
    cited_markers_in,
    match_claim_to_record,
)
from thoughtforge.utils.common import split_sentences  # noqa: E402


def _norm(text: str) -> str:
    """The suite's phrase normalisation (TF-122's shape)."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(clean.split())


def _contains(text_norm: str, phrase: str) -> bool:
    return f" {_norm(phrase)} " in f" {text_norm} "


# ── The retrieved-record set (TF-043's corpus, reused) ──────────────────────
#
# Built from TF-043's filed rows: entities by qid,
# reference chunks by chunk id, the ConceptNet ferry
# edge, and the GeoNames place. Every record carries
# its TF-048 citation payload by construction
# (ActivatedRecord.__post_init__).

_ENTITY_ROWS = {row[0]: row for row in FIXTURE_ENTITIES}
_CHUNK_ROWS = {row[0]: row for row in FIXTURE_REFERENCE_CHUNKS}


def _entity(qid: str) -> ActivatedRecord:
    _, label, description, popularity = _ENTITY_ROWS[qid]
    return ActivatedRecord(
        qid, "entity", popularity,
        f"{label} {description}",
        raw={"qid": qid, "label_en": label,
             "description_en": description, "source": "wikidata"})


def _chunk(chunk_id: int) -> ActivatedRecord:
    _, title, content = _CHUNK_ROWS[chunk_id]
    return ActivatedRecord(
        str(chunk_id), "reference_chunk", 0.8, f"{title} {content}",
        raw={"chunk_id": chunk_id, "source_file": "synthetic.md",
             "title": title, "content": content, "source": "reference"})


def _place() -> ActivatedRecord:
    geonames_id, name, population = FIXTURE_GEONAMES[0]
    return ActivatedRecord(
        name, "place", 0.7, f"{name} place population {population}",
        raw={"geonames_id": geonames_id, "label_en": name,
             "description_en": f"place, population {population}",
             "source": "geonames"})


def _edge() -> ActivatedRecord:
    concept1, relation, concept2, weight = FIXTURE_CONCEPTNET_EDGES[0]
    return ActivatedRecord(
        concept1, "concept_edge", 0.6, f"{concept1} {relation} {concept2}",
        raw={"concept1": concept1, "relation": relation,
             "concept2": concept2,
             "content": f"{concept1} {relation} {concept2}",
             "source": "conceptnet"})


RECORDS: dict[str, ActivatedRecord] = {
    "ferry_entity": _entity("QF4301"),
    "mill_entity": _entity("QF4302"),
    "observatory_entity": _entity("QF4303"),
    "lighthouse_entity": _entity("QF4304"),
    "ferry_chunk": _chunk(4301),
    "mill_chunk": _chunk(4302),
    "observatory_chunk": _chunk(4303),
    "place": _place(),
    "ferry_edge": _edge(),
}

ALL_RECORD_KEYS: tuple[str, ...] = tuple(RECORDS)

#: record key -> its TF-048 marker, derived live.
RECORD_MARKERS: dict[str, str] = {
    key: citation_for_record(record).marker
    for key, record in RECORDS.items()}


def _records(keys: tuple[str, ...]) -> list[ActivatedRecord]:
    return [RECORDS[key] for key in keys]


# ── The stored facts (the truth axis's filed rules) ─────────────────────────

MEMORY_FACTS: tuple[dict[str, Any], ...] = (
    {"id": "ferry_departure", "record": "ferry_chunk",
     "fact": "the lantern hollow ferry departs at dawn",
     "required": ("departs", "dawn"),
     "breaking": ("midnight", "departs at noon")},
    {"id": "ferry_carries", "record": "ferry_entity",
     "fact": "the lantern hollow ferry carries passengers and cargo",
     "required": ("carries passengers", "cargo"),
     "breaking": ("carries only mail", "carries no passengers")},
    {"id": "mill_grain", "record": "mill_chunk",
     "fact": "the bracken mill chunk lists grain and flour prices",
     "required": ("grain", "flour"),
     "breaking": ("timber prices", "iron prices")},
    {"id": "observatory_stars", "record": "observatory_entity",
     "fact": "the alder observatory's astronomers watch stars "
             "through a telescope",
     "required": ("telescope", "stars"),
     "breaking": ("lighthouse beacon", "ferry timetable")},
    {"id": "hollow_place", "record": "place",
     "fact": "lantern hollow is a stored place",
     "required": ("lantern hollow",),
     "breaking": ("a desert port", "an island fortress")},
    {"id": "ferry_crosses", "record": "ferry_edge",
     "fact": "the ferry crosses the river",
     "required": ("ferry", "crosses", "river"),
     "breaking": ("crosses the desert", "crosses the sea")},
)

_FACT_BY_ID = {f["id"]: f for f in MEMORY_FACTS}
_FACT_BY_RECORD = {f["record"]: f for f in MEMORY_FACTS}


# ── The present axis ────────────────────────────────────────────────────────
#
# Judged by TF-049's real checker alone: a case holds
# when check_citations passes the answer against the
# case's retrieved records. Ten holding answers and
# two deliberately failing ones (a missing citation;
# a citation filed in a different sentence than the
# claim it should back — TF-049's cite-somewhere
# distinction).

PRESENT_CASES: tuple[dict[str, Any], ...] = (
    {"id": "p01_ferry_chunk_cited", "fact": "ferry_departure",
     "records": ("ferry_chunk", "mill_chunk"),
     "text": "The lantern hollow ferry departs dawn "
             "[ref:4301].",
     "holds": True},
    {"id": "p02_ferry_entity_cited", "fact": "ferry_carries",
     "records": ("ferry_entity",),
     "text": "The Lantern Hollow Ferry carries passengers "
             "and cargo over the river [QF4301].",
     "holds": True},
    {"id": "p03_mill_chunk_cited", "fact": "mill_grain",
     "records": ("mill_chunk",),
     "text": "Bracken mill grain flour prices are filed "
             "[ref:4302].",
     "holds": True},
    {"id": "p04_observatory_cited", "fact": "observatory_stars",
     "records": ("observatory_entity",),
     "text": "The Alder Observatory astronomers watch "
             "stars through a telescope dome [QF4303].",
     "holds": True},
    {"id": "p05_place_cited", "fact": "hollow_place",
     "records": ("place",),
     "text": "Lantern Hollow is the stored place on the "
             "river [geonames:430001].",
     "holds": True},
    {"id": "p06_edge_cited", "fact": "ferry_crosses",
     "records": ("ferry_edge",),
     "text": "The ferry crosses the river "
             "[conceptnet:ferry|crosses|river].",
     "holds": True},
    {"id": "p07_two_facts_both_cited", "fact": "ferry_departure",
     "records": ("ferry_chunk", "mill_chunk"),
     "text": "The lantern hollow ferry departs dawn "
             "[ref:4301]. Bracken mill grain flour prices "
             "are filed [ref:4302].",
     "holds": True},
    {"id": "p08_mill_entity_cited", "fact": "mill_grain",
     "records": ("mill_entity",),
     "text": "Bracken Mill grinds grain flour at the "
             "waterwheel [QF4302].",
     "holds": True},
    {"id": "p09_ferry_cited_with_distractor_present",
     "fact": "ferry_departure",
     "records": ("ferry_chunk", "lighthouse_entity"),
     "text": "The ferry timetable lantern hollow ferry "
             "departs dawn daily [ref:4301].",
     "holds": True},
    {"id": "p10_observatory_chunk_cited", "fact": "observatory_stars",
     "records": ("observatory_chunk",),
     "text": "Alder observatory visiting telescope stars "
             "evening observatory [ref:4303].",
     "holds": True},
    {"id": "p11_missing_citation", "fact": "ferry_departure",
     "records": ("ferry_chunk",),
     "text": "The lantern hollow ferry departs dawn.",
     "holds": False},
    {"id": "p12_citation_in_wrong_sentence",
     "fact": "ferry_departure",
     "records": ("ferry_chunk",),
     "text": "The lantern hollow ferry departs dawn. "
             "The timetable is posted at the landing "
             "[ref:4301].",
     "holds": False},
)


# ── The true axis ───────────────────────────────────────────────────────────
#
# Judged by citation_truth(): a case holds only when
# every cited marker resolves to a retrieved record,
# that record matches the citing claim under TF-049's
# matcher, and the claim carries the cited fact's
# required phrases with none of its breaking phrases.
# Nine holding answers and three deliberately failing
# ones (a cited-but-false value — the D-BENCH-9 probe;
# a fabricated citation; a wrong-record citation).

TRUE_CASES: tuple[dict[str, Any], ...] = (
    {"id": "t01_ferry_dawn_true", "fact": "ferry_departure",
     "records": ("ferry_chunk",),
     "text": "The lantern hollow ferry departs dawn "
             "[ref:4301].",
     "holds": True},
    {"id": "t02_ferry_carries_true", "fact": "ferry_carries",
     "records": ("ferry_entity",),
     "text": "The Lantern Hollow Ferry carries passengers "
             "and cargo over the river [QF4301].",
     "holds": True},
    {"id": "t03_mill_grain_true", "fact": "mill_grain",
     "records": ("mill_chunk",),
     "text": "Bracken mill grain flour prices are filed "
             "[ref:4302].",
     "holds": True},
    {"id": "t04_observatory_true", "fact": "observatory_stars",
     "records": ("observatory_entity",),
     "text": "The Alder Observatory astronomers watch "
             "stars through a telescope dome [QF4303].",
     "holds": True},
    {"id": "t05_place_true", "fact": "hollow_place",
     "records": ("place",),
     "text": "Lantern Hollow is the stored place on the "
             "river [geonames:430001].",
     "holds": True},
    {"id": "t06_edge_true", "fact": "ferry_crosses",
     "records": ("ferry_edge",),
     "text": "The ferry crosses the river "
             "[conceptnet:ferry|crosses|river].",
     "holds": True},
    {"id": "t07_mill_entity_true", "fact": "mill_grain",
     "records": ("mill_entity",),
     "text": "Bracken Mill grinds grain flour at the "
             "waterwheel [QF4302].",
     "holds": True},
    {"id": "t08_ferry_timetable_true", "fact": "ferry_departure",
     "records": ("ferry_chunk", "mill_chunk"),
     "text": "The ferry timetable lantern hollow ferry "
             "departs dawn, crossings daily [ref:4301].",
     "holds": True},
    {"id": "t09_two_citations_both_true",
     "fact": "ferry_departure",
     "records": ("ferry_chunk", "mill_chunk"),
     "text": "The lantern hollow ferry departs dawn "
             "[ref:4301]. Bracken mill grain flour prices "
             "are filed [ref:4302].",
     "holds": True},
    {"id": "t10_cited_but_false_value", "fact": "ferry_departure",
     "records": ("ferry_chunk",),
     "text": "The lantern hollow ferry departs at "
             "midnight [ref:4301].",
     "holds": False},
    {"id": "t11_fabricated_citation", "fact": "ferry_departure",
     "records": ("ferry_chunk",),
     "text": "The lantern hollow ferry departs dawn "
             "[ref:9999].",
     "holds": False},
    {"id": "t12_wrong_record_citation", "fact": "mill_grain",
     "records": ("ferry_chunk", "mill_chunk"),
     "text": "Bracken mill grain flour prices are filed "
             "[ref:4301].",
     "holds": False},
)


# ── The explain axis ────────────────────────────────────────────────────────
#
# Answers NOT from memory must own themselves. The
# ownership vocabulary is filed here as suite data
# (no estate module defines one — see the docstring).
# A case holds only when TF-049's checker matches no
# stored record, the answer carries no citation
# marker, and an ownership phrase is present. Judged
# against the whole filed record set. Ten holding
# answers and two deliberately failing ones (an
# unowned assertion; a memory claim disguised as
# own reasoning).

OWNERSHIP_PHRASES: tuple[str, ...] = (
    "in my own reasoning",
    "my own reasoning",
    "my own inference",
    "i reason that",
    "i infer that",
    "speaking for myself",
)

EXPLAIN_CASES: tuple[dict[str, Any], ...] = (
    {"id": "e01_owned_weather", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "In my own reasoning, spring rain makes the "
             "harbour path muddy.",
     "holds": True},
    {"id": "e02_owned_season", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "My own inference is that autumn evenings "
             "suit quiet harbour walks.",
     "holds": True},
    {"id": "e03_owned_market", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "I reason that a weekly market would draw "
             "the villages together.",
     "holds": True},
    {"id": "e04_owned_colour", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "Speaking for myself, a blue door would "
             "cheer the harbour cottages.",
     "holds": True},
    {"id": "e05_owned_infer_path", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "I infer that the cliff path is steepest "
             "beyond the pine grove.",
     "holds": True},
    {"id": "e06_owned_reasoning_festival", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "In my own reasoning, a harvest festival "
             "belongs in the village square.",
     "holds": True},
    {"id": "e07_owned_winter", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "My own reasoning: winter guests would want "
             "a warm common room.",
     "holds": True},
    {"id": "e08_owned_garden", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "I reason that herb gardens thrive on the "
             "sunny slope above the village.",
     "holds": True},
    {"id": "e09_owned_music", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "My own inference is that fiddle music "
             "suits the village hall.",
     "holds": True},
    {"id": "e10_owned_boats_opinion", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "Speaking for myself, painted boats look "
             "finest in morning light.",
     "holds": True},
    {"id": "e11_unowned_assertion", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "Spring rain makes the harbour path muddy.",
     "holds": False},
    {"id": "e12_memory_claim_disguised", "fact": None,
     "records": ALL_RECORD_KEYS,
     "text": "My own reasoning is that the lantern "
             "hollow ferry departs dawn.",
     "holds": False},
)

ALL_SUITE_CASES: tuple[dict[str, Any], ...] = (
    PRESENT_CASES + TRUE_CASES + EXPLAIN_CASES
)

SUITE_AXES: tuple[dict[str, Any], ...] = (
    {"id": "citations_present", "cases": len(PRESENT_CASES),
     "scorer": "thoughtforge.refinement.citation_enforcement."
               "check_citations (TF-049, reused)",
     "computes": "fraction of scripted memory-claim cases "
                 "TF-049's checker passes: every matched "
                 "claim cited in-claim, nothing uncited, "
                 "no foreign marker"},
    {"id": "citations_true", "cases": len(TRUE_CASES),
     "scorer": "citation_truth (this module: TF-049's "
               "matcher + the filed fact rules)",
     "computes": "fraction of scripted cited cases where "
                 "every citation resolves to a retrieved "
                 "record that matches the citing claim and "
                 "the claim carries the cited fact's "
                 "required phrases with no breaking phrase"},
    {"id": "explain", "cases": len(EXPLAIN_CASES),
     "scorer": "the filed ownership rule + TF-049's "
               "checker (no match leg)",
     "computes": "fraction of scripted non-memory cases "
                 "that own themselves: no stored record "
                 "matched, no citation marker carried, "
                 "and a filed ownership phrase present"},
)

SUITE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "three_axes_one_rule",
     "rule": "the suite scores exactly three axes — "
             "citations present, citations true, explain "
             "— the two legs of cite-or-explain (TF-048 / "
             "TF-049 / TF-093) plus the truth leg TF-049's "
             "doc assigned to this slice"},
    {"id": "presence_is_never_truth",
     "rule": "D-BENCH-9's remedy: a citation passing "
             "TF-049's presence check scores the present "
             "axis only; the true axis additionally "
             "requires resolution, a matcher match "
             "against the cited record, and the filed "
             "required / breaking phrase rules — the "
             "probed cited-but-false claim (a midnight "
             "departure citing the dawn chunk) holds on "
             "present and fails on true, by design"},
    {"id": "machinery_reused_never_copied",
     "rule": "presence is TF-049's check_citations, "
             "truth's match leg is TF-049's "
             "match_claim_to_record, markers are TF-048's "
             "payloads, and the stored-fact world is "
             "TF-043's corpus imported in place — this "
             "module files cases and rules, not a second "
             "checker"},
    {"id": "no_vacuous_pass_anywhere",
     "rule": "an empty axis or an empty suite scores "
             "0.0 and files REVIEW; PASS requires a "
             "non-empty run with every axis at or above "
             "AXIS_PASS_THRESHOLD (the TF-122 precedent)"},
    {"id": "fixtures_are_synthetic_only",
     "rule": "every case is invented for this suite in "
             "TF-043's synthetic world (TF-017): canon is "
             "never a benchmark input and data/ was "
             "never opened"},
    {"id": "failing_shapes_are_filed_not_hidden",
     "rule": "the corpora file the failing shapes by "
             "name (missing citation, wrong-sentence "
             "citation, cited-but-false, fabricated, "
             "wrong-record, unowned, disguised) with "
             "holds=False — the suite scores failures, "
             "not just passes"},
)


# ── The truth judge ─────────────────────────────────────────────────────────

def citation_truth(
    text: str, records: list[ActivatedRecord],
) -> dict[str, Any]:
    """Judge the citations in one answer for truth.

    True exactly when the answer carries at least one
    citation, every cited marker resolves to one of
    ``records``, no foreign marker appears (TF-049's
    leg), the cited record matches the citing claim
    under TF-049's real matcher, and — where the cited
    record carries a filed fact — the claim carries
    that fact's required phrases and none of its
    breaking phrases.
    """
    records = list(records or [])
    by_marker = {citation_for_record(r).marker: r for r in records}
    key_by_record = {id(r): k for k, r in RECORDS.items()}
    check = check_citations(text or "", records)
    cited = cited_markers_in(text or "")
    sentences = [s.strip() for s in split_sentences(text or "")
                 if s.strip()]
    if not sentences and (text or "").strip():
        sentences = [(text or "").strip()]
    verdicts: list[dict[str, Any]] = []
    true = bool(cited) and not check.foreign_markers
    for marker in sorted(cited):
        record = by_marker.get(marker)
        entry: dict[str, Any] = {"marker": marker,
                                 "resolves": record is not None}
        if record is None:
            true = False
            verdicts.append(entry)
            continue
        claim = next(
            (s for s in sentences if marker in cited_markers_in(s)),
            "")
        match = match_claim_to_record(claim, record) if claim else None
        entry["matches_claim"] = match is not None
        fact = _FACT_BY_RECORD.get(key_by_record.get(id(record), ""))
        if fact is not None:
            claim_norm = _norm(claim)
            entry["fact"] = fact["id"]
            entry["required_present"] = all(
                _contains(claim_norm, p) for p in fact["required"])
            entry["breaking_present"] = any(
                _contains(claim_norm, p) for p in fact["breaking"])
        else:
            entry["required_present"] = True
            entry["breaking_present"] = False
        if not (entry["matches_claim"] and entry["required_present"]
                and not entry["breaking_present"]):
            true = False
        verdicts.append(entry)
    return {"true": true, "cited_markers": sorted(cited),
            "foreign_markers": list(check.foreign_markers),
            "verdicts": verdicts}


# ── The axis scorers ────────────────────────────────────────────────────────

def score_present_cases(
    cases: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the present axis by TF-049's real checker."""
    cases = PRESENT_CASES if cases is None else cases
    held = 0
    for case in cases:
        check = check_citations(case["text"], _records(case["records"]))
        held += int(check.passed and not check.uncited_record_ids
                    and not check.foreign_markers)
    total = len(cases)
    return {"axis": "citations_present", "cases": total, "held": held,
            "score": (held / total) if total else 0.0}


def score_true_cases(
    cases: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the true axis by citation_truth()."""
    cases = TRUE_CASES if cases is None else cases
    held = 0
    for case in cases:
        held += int(citation_truth(
            case["text"], _records(case["records"]))["true"])
    total = len(cases)
    return {"axis": "citations_true", "cases": total, "held": held,
            "score": (held / total) if total else 0.0}


def score_explain_cases(
    cases: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the explain axis by the filed ownership rule."""
    cases = EXPLAIN_CASES if cases is None else cases
    held = 0
    for case in cases:
        records = _records(case["records"])
        check = check_citations(case["text"], records)
        text_norm = _norm(case["text"])
        owned = any(_contains(text_norm, p) for p in OWNERSHIP_PHRASES)
        held += int(not check.matched_record_ids
                    and not cited_markers_in(case["text"]) and owned)
    total = len(cases)
    return {"axis": "explain", "cases": total, "held": held,
            "score": (held / total) if total else 0.0}


# ── The suite run ───────────────────────────────────────────────────────────

@dataclass
class MemorySuiteResult:
    """One full suite run: per-axis scores, overall, verdict."""

    present_score: float
    true_score: float
    explain_score: float
    overall_score: float
    total_cases: int
    per_axis: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """PASS only on a non-empty run with every axis passing."""
        if self.total_cases == 0:
            return "REVIEW"
        if min(self.present_score, self.true_score,
               self.explain_score) >= AXIS_PASS_THRESHOLD:
            return "PASS"
        return "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_memory_suite(
    present_cases: Any = None,
    true_cases: Any = None,
    explain_cases: Any = None,
) -> MemorySuiteResult:
    """Run the whole suite (filed corpora by default).

    Any axis may be given an explicit corpus — including
    an empty one, which scores that axis 0.0 and keeps
    the suite at REVIEW (no vacuous pass anywhere).
    """
    present = score_present_cases(present_cases)
    true_axis = score_true_cases(true_cases)
    explain = score_explain_cases(explain_cases)
    per_axis = [present, true_axis, explain]
    total = sum(a["cases"] for a in per_axis)
    overall = (
        sum(a["score"] for a in per_axis) / len(per_axis)) if total else 0.0
    return MemorySuiteResult(
        present_score=present["score"], true_score=true_axis["score"],
        explain_score=explain["score"], overall_score=overall,
        total_cases=total, per_axis=per_axis)


# ── The live checker ────────────────────────────────────────────────────────

def memory_suite_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    axis_ids = [a["id"] for a in SUITE_AXES]
    if axis_ids != ["citations_present", "citations_true", "explain"]:
        problems.append(f"SUITE_AXES changed shape: {axis_ids}")
    rule_ids = [r["id"] for r in SUITE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate SUITE_RULES ids")
    for needed in ("three_axes_one_rule", "presence_is_never_truth",
                   "machinery_reused_never_copied",
                   "no_vacuous_pass_anywhere",
                   "fixtures_are_synthetic_only",
                   "failing_shapes_are_filed_not_hidden"):
        if needed not in rule_ids:
            problems.append(f"SUITE_RULES lost {needed}")

    # The record set is TF-043's corpus, bound live, and
    # every record carries a distinct TF-048 marker.
    if set(RECORD_MARKERS.values()) != {
            "QF4301", "QF4302", "QF4303", "QF4304", "ref:4301",
            "ref:4302", "ref:4303", "geonames:430001",
            "conceptnet:ferry|crosses|river"}:
        problems.append(f"RECORD_MARKERS drifted: {RECORD_MARKERS}")
    for fact in MEMORY_FACTS:
        if fact["record"] not in RECORDS:
            problems.append(f"fact {fact['id']} names no filed record")
        if not fact["required"]:
            problems.append(f"fact {fact['id']} files no required phrases")

    # Corpora: counts, balance, and filed expectations true.
    for name, cases, held_want in (
            ("PRESENT_CASES", PRESENT_CASES, 10),
            ("TRUE_CASES", TRUE_CASES, 9),
            ("EXPLAIN_CASES", EXPLAIN_CASES, 10)):
        if len(cases) != CASES_PER_AXIS:
            problems.append(f"{name} {len(cases)} != {CASES_PER_AXIS}")
        if sum(c["holds"] for c in cases) != held_want:
            problems.append(
                f"{name} files {sum(c['holds'] for c in cases)} "
                f"holding (want {held_want})")
    case_ids = [c["id"] for c in ALL_SUITE_CASES]
    if len(case_ids) != len(set(case_ids)):
        problems.append("duplicate case ids across the corpora")
    for axis_name, scorer, cases in (
            ("present", score_present_cases, PRESENT_CASES),
            ("true", score_true_cases, TRUE_CASES),
            ("explain", score_explain_cases, EXPLAIN_CASES)):
        for case in cases:
            single = scorer([case])
            if bool(single["held"]) != bool(case["holds"]):
                problems.append(
                    f"{axis_name} case filed holds={case['holds']} "
                    f"scores held={single['held']}: {case['id']}")

    # The D-BENCH-9 probe, pinned: the cited-but-false
    # claim passes TF-049's presence check and fails
    # this suite's truth judge.
    false_case = _FACT_BY_ID and [c for c in TRUE_CASES
                                  if c["id"] == "t10_cited_but_false_value"]
    if false_case:
        case = false_case[0]
        records = _records(case["records"])
        if not check_citations(case["text"], records).passed:
            problems.append(
                "the cited-but-false probe no longer passes TF-049 "
                "(the presence/truth contrast this suite files is gone)")
        if citation_truth(case["text"], records)["true"]:
            problems.append("the cited-but-false probe scores true")

    # The neighbours' own checkers stay clean.
    from thoughtforge.knowledge.citations import citation_problems
    from thoughtforge.refinement.citation_enforcement import (
        citation_enforcement_problems,
    )
    from thoughtforge.refinement.policy_surface import (
        policy_surface_live_problems,
        policy_surface_problems,
    )

    if citation_problems():
        problems.append("TF-048's citation_problems is not clean")
    if citation_enforcement_problems():
        problems.append("TF-049's checker is not clean")
    if policy_surface_problems() or policy_surface_live_problems():
        problems.append("TF-093's policy surface checkers are not clean")

    # TF-043's live leg: the fixture DB built fresh, its
    # pinned baseline unregressed, and a real retrieval
    # returning the filed records with their markers.
    import tempfile
    from pathlib import Path

    from thoughtforge.knowledge.forge import KnowledgeForge
    from thoughtforge.knowledge.retrieval_precision import (
        baseline_problems as retrieval_baseline_problems,
    )
    from thoughtforge.knowledge.retrieval_precision import (
        build_fixture_db,
        evaluate_retrieval,
    )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = build_fixture_db(Path(tmp) / "tf123_fixture.db")
            report = evaluate_retrieval(db_path)
            moved = retrieval_baseline_problems(report, setting="default")
            if moved:
                problems.append(f"TF-043's baseline regressed: {moved}")
            bundle = KnowledgeForge(db_path=db_path).retrieve(
                "Lantern Hollow ferry", path="sql", top_k=10,
                min_score=0.0)
            got = {r.record_id: citation_for_record(r).marker
                   for r in bundle.activated_records}
            if got.get("QF4301") != "QF4301" \
                    or got.get("4301") != "ref:4301":
                problems.append(
                    f"TF-043 retrieval does not return the filed "
                    f"records with their TF-048 markers: {got}")
    except Exception as exc:  # a leg that cannot run is a finding
        problems.append(f"TF-043's live leg failed: {exc!r}")

    # A fresh live run has the filed shape; empty is 0.0 / REVIEW.
    result = run_memory_suite()
    if result.total_cases != len(ALL_SUITE_CASES):
        problems.append(
            f"live run totals {result.total_cases} != corpus "
            f"{len(ALL_SUITE_CASES)}")
    if result.total_cases == 0 or result.overall_score <= 0.0:
        problems.append("the live suite run scores nothing")
    empty = run_memory_suite([], [], [])
    if empty.total_cases != 0 or empty.overall_score != 0.0 \
            or empty.verdict != "REVIEW" or empty.passes:
        problems.append("an empty suite does not file 0.0 / REVIEW")

    from pathlib import Path

    doc = Path(__file__).resolve().parent.parent / MEMORY_SUITE_DOC
    if not doc.exists():
        problems.append(f"the write-up {MEMORY_SUITE_DOC} is missing")
    return problems


def memory_suite_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_memory_suite()
    return {
        "format": MEMORY_SUITE_FORMAT,
        "doc": MEMORY_SUITE_DOC,
        "axis_pass_threshold": AXIS_PASS_THRESHOLD,
        "axes": [dict(a) for a in SUITE_AXES],
        "record_markers": dict(RECORD_MARKERS),
        "memory_facts": [
            {**dict(f), "required": list(f["required"]),
             "breaking": list(f["breaking"])} for f in MEMORY_FACTS],
        "ownership_phrases": list(OWNERSHIP_PHRASES),
        "case_counts": {
            "citations_present": len(PRESENT_CASES),
            "citations_true": len(TRUE_CASES),
            "explain": len(EXPLAIN_CASES),
            "total": result.total_cases,
        },
        "measured": {
            "present_score": result.present_score,
            "true_score": result.true_score,
            "explain_score": result.explain_score,
            "overall_score": result.overall_score,
            "verdict": result.verdict,
        },
        "rules": [dict(r) for r in SUITE_RULES],
    }


__all__ = [
    "ALL_RECORD_KEYS",
    "ALL_SUITE_CASES",
    "AXIS_PASS_THRESHOLD",
    "CASES_PER_AXIS",
    "EXPLAIN_CASES",
    "MEMORY_FACTS",
    "MEMORY_SUITE_DOC",
    "MEMORY_SUITE_FORMAT",
    "OWNERSHIP_PHRASES",
    "PRESENT_CASES",
    "RECORDS",
    "RECORD_MARKERS",
    "SUITE_AXES",
    "SUITE_RULES",
    "TRUE_CASES",
    "MemorySuiteResult",
    "citation_truth",
    "memory_suite_as_data",
    "memory_suite_problems",
    "run_memory_suite",
    "score_explain_cases",
    "score_present_cases",
    "score_true_cases",
]
