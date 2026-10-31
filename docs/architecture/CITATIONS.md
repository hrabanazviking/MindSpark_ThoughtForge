# TF-048 — Cite-or-Explain I: the Retrieval Side

*Part III, slice TF-048. Code:
`src/thoughtforge/knowledge/citations.py` (the formats table,
the bundle APIs, and `citation_problems()` re-deriving every
claim below from the live tree), `CitationPayload` +
`derive_citation` in `src/thoughtforge/knowledge/models.py`,
the fact-block rendering in
`src/thoughtforge/cognition/scaffold.py`. Tests:
`tests/test_tf048_citations.py`. Builds on TF-042's identity
fields per kind and TF-043's fixture corpus; TF-049 owns the
enforcement side (whether a generated answer uses the
citations this slice guarantees it was shown).*

## 1. The question and the measured starting state

The roadmap asks: does every retrieved item carry its
citation payload into the prompt **by construction**?
Measured before this slice, the honest answer was **no** —
citations existed only on the way out (the refinement side
regex-hunts QIDs in generated text), while on the way in:

- `ActivatedRecord` carried no citation field at all. What
  provenance a retrieved item had lived scattered in its
  `raw` dict, in a different shape per source.
- The scaffold's fact block improvised the bracket per line:
  `[qid]` when the raw carried a QID, `[source]` (a bare
  source name, not an identity) otherwise — and a record
  with neither a qid nor a label fell through to its bare
  cue, entering the prompt with **no citation at all**.
- The forge's ConceptNet query selected
  `concept1, relation, concept2` and then dropped them, so
  the edge's identity (TF-042: the triple) never reached
  the raw dict a citation could be built from.
- Vector reference results carry no `record_type`, and the
  bundle builder's default labelled them `"entity"` — a
  reference chunk is not an entity.

## 2. The payload

`CitationPayload` (frozen, in `knowledge/models.py` beside
the record it belongs to): **marker, source, record_id,
record_type, locator, label**. `derive_citation(record_id,
record_type, raw)` builds it deterministically from the
record's own raw retrieval dict, using TF-042's identity
fields per kind:

| kind | marker | locator |
|---|---|---|
| entity | the bare QID (`Q123`) | the QID |
| reference chunk | `ref:<chunk_id>` | the `source_file` (TF-042: that file IS the chunk's provenance) |
| GeoNames place | `geonames:<geonames_id>` | the id |
| ConceptNet edge | `conceptnet:<c1>\|<relation>\|<c2>` | the triple text |
| anything else | `<source>:<record_id>` | the record id |

The entity marker stays a **bare QID** on purpose: the
prompt token and the refinement side's QID checks
(`_QID_RE`, `\bQ\d{3,}\b`) must speak the same shape, and
`citation_problems()` pins that they do.

Derivation is **total**: an item whose raw carries no
stronger identity falls back to `<source>:<record_id>`,
and even an empty id yields `<source>:unidentified`. The
marker is never empty, so "every line cited" has no silent
exception.

`ActivatedRecord.__post_init__` derives the payload
whenever the caller did not supply one. That placement is
the "by construction": the forge's one bundle builder,
hand-built fixtures, and any future retrieval path all
inherit the guarantee from the record itself — there is no
construction site left to forget it.

## 3. Into the prompt

The scaffold's `_build_fact_block` renders every line as
`[<marker>] <label> — <description>` from the payload
(with the cue as the line's text when the record has no
label — still marker-first). The old qid/source
improvisation and the uncited cue fall-through are gone.
Entity lines are byte-identical to before (`[Q123] Odin —
…`); reference, place, and edge lines gain their real
identities in place of the bare source name.

The 8-line budget is unchanged: the core retrieves up to
10, the fact block shows the top-scored 8, and **every**
record in the bundle — shown or not — carries its payload
for the consumers that read the bundle (TF-049's gate,
TF-061's inspector).

## 4. Defects fixed in scope

1. **ConceptNet triple dropped** (forge SQL pass): the
   query selected the triple and stored only a flattened
   `content` string. The three fields now ride the raw
   dict, as TF-042's identity for the kind requires.
2. **Vector reference chunks typed "entity"** (bundle
   builder default): the fallback now derives
   `reference_chunk` from the chunk identity. SQL shapes
   are unaffected (they always carried an explicit type).

## 5. Stated boundaries

- The prompt's **memory-cues** channel keeps TF-023's
  pinned verbatim shape and carries no markers; the
  citation-carrying channel is the fact block ("Knowledge
  context"), present in candidate and strategy prompts.
  The repair prompt carries no knowledge context at all,
  by TF-023's stated narrowing.
- The turn trace (TF-038) does not export the payload
  yet — TF-061 owns inspector data. *(TF-061, landed:
  trace retrieval entries carry the payload, and the
  inspector export surfaces it per turn.)*
- Whether a generated answer actually cites what it was
  shown is **TF-049** (enforcement side, with TF-093).
  This slice guarantees the citations were *in the
  prompt*, nothing more.

## 6. Proof

Fixture grade: the TF-043 synthetic corpus through the
real forge (SQL path), synthetic scored records of every
source shape (including the vector shapes) through the
real bundle builder, and a real knowledge-only core turn
on the fixture DB whose scaffold fact block is checked
line by line against its bundle's markers.
`citation_problems() == []` re-derives the whole leg from
the live tree. Vector/hybrid retrieval itself remains
NOT PROVEN on this machine (D-ENV-1) — the vector *shapes*
are proven at the bundle builder, which is where the
payload attaches.
