"""
Cite-or-explain I — the retrieval side (TF-048).

The roadmap's rule: *every retrieved item carries its citation
payload into the prompt by construction*. Before this slice the
estate had citations only as an after-the-fact extraction: the
enforcement/salvage side regex-hunted QIDs in generated text
(TF-049's side), while on the way IN the scaffold improvised a
bracket from raw fields — a QID when the raw happened to carry
one, a bare source name otherwise — and a record with neither a
qid nor a label fell through to its bare cue and entered the
prompt with no citation at all. Provenance that exists only
sometimes is not a payload.

What this slice builds:

  * ``CitationPayload`` + ``derive_citation`` (knowledge/models.py,
    beside the record it belongs to): one payload per retrieved
    item — marker, source, record_id, record_type, locator,
    label — derived deterministically from the record's own raw
    retrieval dict, using TF-042's identity fields per kind
    (entity → qid, reference chunk → chunk_id + source_file,
    GeoNames place → geonames_id, ConceptNet edge → its triple).
    ``ActivatedRecord.__post_init__`` derives it whenever the
    caller did not supply one, so a retrieved item cannot be
    constructed without its payload — the forge's one bundle
    builder, hand-built fixtures, and any future retrieval path
    all inherit the guarantee from the record itself.
  * The marker shapes are the prompt's citation tokens:
    a bare QID for entities (so the QID checks in refinement/
    keep matching the token the prompt taught the model to use),
    ``ref:<chunk_id>``, ``geonames:<id>``,
    ``conceptnet:<c1>|<relation>|<c2>``, and the total fallback
    ``<source>:<record_id>`` / ``<source>:unidentified`` — the
    marker is never empty, so "every line cited" has no silent
    exception.
  * The scaffold's fact block (cognition/scaffold.py) renders
    every line as ``[<marker>] …`` from the payload — the old
    qid/source improvisation and the uncited cue fall-through
    are gone. The 8-line budget is unchanged: the bundle may
    hold more records, each still carrying its payload; the
    lines that enter the prompt are the top-scored ones.
  * Two retrieval-side defects the payload exposed, fixed in
    scope: the forge's ConceptNet query selected
    concept1/relation/concept2 and then dropped them (the
    edge's TF-042 identity never reached the raw dict), and
    vector reference results — which carry no record_type —
    were labelled "entity" by the bundle builder's default;
    the fallback now derives "reference_chunk" from the chunk
    identity.

Stated boundaries (not silent): the prompt's *memory cues*
channel keeps TF-023's pinned verbatim shape and carries no
markers — the citation-carrying channel is the fact block
("Knowledge context"), present in candidate and strategy
prompts; the repair prompt carries no knowledge context by
TF-023's stated narrowing. The trace export (TF-038) does not
carry the payload yet — TF-061 owns inspector data.
(TF-061, landed: the trace's retrieval entries now carry
each record's payload, and ``knowledge/inspector.py``
surfaces it per turn.) Whether a generated answer
actually *uses* the citations is TF-049 (enforcement
side), not this slice.

``CITATION_FORMATS`` states the marker shape per retrieval kind
as data, and ``citation_problems()`` re-proves the whole leg
against the live tree (derivation per kind, the forge's bundle
behaviour on synthetic scored records of every source shape,
the scaffold's every-line-cited rendering, and the prompt
end-to-end), so drift is caught by name.
"""

from __future__ import annotations

from typing import Any

from thoughtforge.knowledge.models import (
    ActivatedRecord,
    CitationPayload,
    MemoryActivationBundle,
    derive_citation,
)

# ── The marker shapes, as data ───────────────────────────────────────────────
#
# One row per retrieval kind the forge can produce. "identity" names
# the raw fields the marker is built from (TF-042's identity fields
# for the kind); "example" is a literal marker of that shape.

CITATION_FORMATS: tuple[dict[str, Any], ...] = (
    {"kind": "entity", "identity": ("qid",),
     "marker_shape": "<qid>", "example": "Q123"},
    {"kind": "reference_chunk", "identity": ("chunk_id", "source_file"),
     "marker_shape": "ref:<chunk_id>", "example": "ref:chunk_7"},
    {"kind": "geonames_place", "identity": ("geonames_id",),
     "marker_shape": "geonames:<geonames_id>", "example": "geonames:5128581"},
    {"kind": "conceptnet_edge", "identity": ("concept1", "relation", "concept2"),
     "marker_shape": "conceptnet:<concept1>|<relation>|<concept2>",
     "example": "conceptnet:ferry|UsedFor|crossing rivers"},
    {"kind": "fallback", "identity": ("record_id",),
     "marker_shape": "<source>:<record_id>", "example": "user_fact:fct_1"},
)

def citation_for_record(record: ActivatedRecord) -> CitationPayload:
    """The record's citation payload (deriving one if it has none).

    Records built through ``ActivatedRecord`` always carry one
    (``__post_init__``); the derive-here path covers duck-typed
    records a caller assembled without the constructor.
    """
    payload = getattr(record, "citation", None)
    if isinstance(payload, CitationPayload):
        return payload
    return derive_citation(record.record_id, record.record_type, getattr(record, "raw", None))


def markers_for_bundle(bundle: MemoryActivationBundle) -> list[str]:
    """Every retrieved item's citation marker, in bundle order."""
    return [citation_for_record(r).marker for r in bundle.activated_records]


def payloads_for_bundle(bundle: MemoryActivationBundle) -> list[CitationPayload]:
    """Every retrieved item's citation payload, in bundle order."""
    return [citation_for_record(r) for r in bundle.activated_records]


# ── Self-check ───────────────────────────────────────────────────────────────

# Synthetic scored records in the exact shapes the forge's SQL and
# vector passes produce (measured from forge.py / embeddings.py).
_SYNTHETIC_SCORED: tuple[dict[str, Any], ...] = (
    {"qid": "Q123", "label_en": "Odin", "description_en": "chief god",
     "popularity_score": 0.9, "fts_score": 3.0, "source": "wikidata",
     "record_type": "entity", "_composite_score": 0.9},
    {"chunk_id": "chunk_7", "source_file": "hollow.md", "title": "The Hollow",
     "content": "lantern oil is kept in the hollow stores", "tags": "",
     "fts_score": 2.0, "source": "reference", "record_type": "reference_chunk",
     "_composite_score": 0.8},
    {"chunk_id": "chunk_9", "source_file": "ferries.md", "title": "Ferries",
     "content": "the ferry crosses at dawn", "similarity": 0.7,
     "source": "reference_chunk", "_composite_score": 0.7},  # vector shape: no record_type
    {"content": "ferry UsedFor crossing", "label_en": "ferry",
     "concept1": "ferry", "relation": "UsedFor", "concept2": "crossing",
     "weight": 1.0, "fts_score": 1.0, "source": "conceptnet",
     "record_type": "concept_edge", "_composite_score": 0.6},
    {"geonames_id": 5128581, "label_en": "Hollow Creek",
     "description_en": "P in US, pop 120", "fts_score": 0.5,
     "source": "geonames", "record_type": "place", "_composite_score": 0.5},
)

_EXPECTED_MARKERS: tuple[str, ...] = (
    "Q123", "ref:chunk_7", "ref:chunk_9",
    "conceptnet:ferry|UsedFor|crossing", "geonames:5128581",
)


def citation_problems() -> list[str]:
    """
    Re-check the cite-or-explain retrieval leg against the live
    tree. Returns problems by name; ``[]`` when every retrieved
    item still carries its payload into the prompt by
    construction.
    """
    problems: list[str] = []

    # The formats table is well-formed and its examples derive.
    if len({row["kind"] for row in CITATION_FORMATS}) != len(CITATION_FORMATS):
        problems.append("CITATION_FORMATS holds duplicate kinds")
    for row in CITATION_FORMATS:
        if not row["marker_shape"] or not row["example"]:
            problems.append(f"CITATION_FORMATS[{row['kind']}] has an empty shape/example")

    # Derivation is total: even an empty raw yields a non-empty marker.
    bare = derive_citation("rec_x", "user_fact", {})
    if bare.marker != "user_fact:rec_x":
        problems.append(f"fallback derivation drifted: {bare.marker!r}")
    ghost = derive_citation("", "entity", {})
    if not ghost.marker.endswith(":unidentified"):
        problems.append(f"identity-less derivation drifted: {ghost.marker!r}")

    # Construction carries the payload: a bare ActivatedRecord
    # derives one in __post_init__.
    auto = ActivatedRecord("Q9", "entity", 0.5, "cue", raw={"qid": "Q9"})
    if auto.citation is None or auto.citation.marker != "Q9":
        problems.append("ActivatedRecord no longer derives its citation at construction")

    # The entity marker is a bare QID the refinement QID regex finds —
    # the prompt token and TF-049's check must speak the same shape.
    from thoughtforge.refinement.enforcement import _QID_RE

    if not _QID_RE.fullmatch(_EXPECTED_MARKERS[0]):
        problems.append("entity marker no longer matches the enforcement QID shape")

    # The forge's bundle builder: synthetic records of every source
    # shape come out carrying the expected payloads, and the vector
    # reference shape is typed reference_chunk, not entity.
    from thoughtforge.knowledge.forge import KnowledgeForge

    forge = KnowledgeForge.__new__(KnowledgeForge)
    bundle = forge._build_bundle([dict(r) for r in _SYNTHETIC_SCORED])
    markers = markers_for_bundle(bundle)
    if markers != list(_EXPECTED_MARKERS):
        problems.append(f"bundle markers drifted: {markers}")
    ref_types = [r.record_type for r in bundle.activated_records if r.record_id == "chunk_9"]
    if ref_types != ["reference_chunk"]:
        problems.append(f"vector reference record typed {ref_types}, not reference_chunk")
    if any(r.citation is None for r in bundle.activated_records):
        problems.append("a bundled record carries no citation payload")

    # The scaffold renders every fact-block line cited: one bracketed
    # marker per line, including a label-less record that previously
    # fell through to its bare cue.
    from thoughtforge.cognition.scaffold import _build_fact_block

    labelless = ActivatedRecord("", "entity", 0.95, "orphan cue with no label", raw={})
    block = _build_fact_block([labelless, *bundle.activated_records])
    lines = block.splitlines()
    if len(lines) != 6:
        problems.append(f"fact block holds {len(lines)} lines for 6 records")
    for line, record in zip(lines, sorted(
        [labelless, *bundle.activated_records], key=lambda r: r.score, reverse=True
    )):
        marker = citation_for_record(record).marker
        if not line.startswith(f"[{marker}]"):
            problems.append(f"fact line for {record.record_id} opens without its marker: {line!r}")

    # End to end: the candidate prompt carries every fact-block marker.
    from thoughtforge.cognition.prompt_builder import PromptBuilder
    from thoughtforge.cognition.scaffold import ScaffoldBuilder
    from thoughtforge.knowledge.models import InputSketch

    sketch = InputSketch(raw_text="what crosses at dawn?", intent="factual_query",
                         topic="ferries", response_mode="factual_direct")
    scaffold = ScaffoldBuilder().build(sketch, bundle)
    prompt = PromptBuilder().build_candidate_prompt(sketch, scaffold, "practical", [])
    for marker in markers_for_bundle(bundle):
        if f"[{marker}]" not in prompt:
            problems.append(f"candidate prompt is missing citation [{marker}]")

    return problems


__all__ = [
    "CITATION_FORMATS",
    "citation_for_record",
    "citation_problems",
    "markers_for_bundle",
    "payloads_for_bundle",
]
