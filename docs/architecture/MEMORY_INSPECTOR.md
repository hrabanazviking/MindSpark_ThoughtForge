# TF-061 — Memory Inspector Data: Holdings + Why a Retrieval Fired

*Part III, slice TF-061. Code:
`src/thoughtforge/knowledge/inspector.py` (the format,
the dimension explanation + its mirror, the per-turn
export, both checkers), plus wiring in
`knowledge/forge.py` (D-INSP-1: retained `_explanation`
+ `last_retrieval_report`), `cognition/traces.py`
(citation payloads in the trace), and
`cognition/core.py` (`last_turn_inspector()`, recorded
in `core_docs.POST_FREEZE_ADDITIONS` + the freeze doc
§7). Tests: `tests/test_tf061_inspector.py` (41 tests).
Consumes TF-038 (the turn trace), TF-048 (citation
payloads), TF-049 (the retained claim check), TF-057
(the store report).*

## 1. The question and the measured starting state

The roadmap asks: can an inspector say **what the store
holds** and **why a retrieval fired**, as data exported
per turn? Measured this tick by direct probe:

- **Why a record fired was discarded at scoring time
  (D-INSP-1).** `KnowledgeForge._score_records` computed
  the five `RetrievalDimensions` per candidate, folded
  them into one composite — and threw the dimensions
  away. Probe P1: a scored record's raw carried only
  `_composite_score`; no dimension, semantic-source, or
  recency-source key survived. "Why this record, at
  this score" was unanswerable.
- **Records that did not fire were unexplainable.**
  `retrieve()` filtered below-threshold candidates and
  cut beyond `top_k` in place. Afterwards nothing
  recorded that a candidate had been considered and
  rejected, or why — indistinguishable from a record
  the legs never returned.
- **The TF-038 trace carried no citation payload.**
  Probe P2: trace retrieval entries held exactly
  `{cue, raw, record_id, record_type, score}` — the
  record itself carried a TF-048 payload by
  construction, but the trace dropped it. TF-048
  assigned this leg to TF-061.
- **"What the store holds" existed only split.**
  TF-057's `inspect_store` reports the personal
  `MemoryStore`; `forge_memory status` /
  `etl/db_integrity` report the knowledge DB; the trace
  reports the turn. Probe P3/P4: no inspector module
  existed, and the core had no inspector surface — the
  join TF-038's doc named ("TF-061's inspector data …
  consume it as data") was unbuilt.

## 2. The export

One named format — `INSPECTOR_FORMAT =
"thoughtforge-inspector/1"` — with sections
(`INSPECTOR_SECTIONS`) in reading order:

| Section | Carries |
|---|---|
| `store` | TF-057's `inspect_store` report over the personal store (counts per kind, bytes per file, counts by status, core/thread presence, queue length, tag count) |
| `knowledge_db` | the DB estate described read-only: path, bytes, per-table row counts. A missing DB is a verdict (`present: False`); an unreadable one a verdict with the error named. Content is never copied out — canon described, not carried (TF-058's rule) |
| `turn` | the TF-038 trace reduced to identity: turn id, phase, completion, profile/strategy/persona, the query, the final's response id + enforcement verdict + citations |
| `retrieval` | the why (§3) |
| `citations` | the core's retained TF-049 per-claim check as data, or `present: False` with the reason named (a knowledge-only turn runs no claim check) |

Every input is independently optional: a missing input
is a stated verdict inside its section, never a raise
and never a fabricated section. The export is detached
and JSON-safe; `write_inspector_report` writes it as
one JSON document (TF-038's writer discipline).
`inspector_problems(report)` judges an export as data:
format, sections present, every fired record carrying
its citation payload, dimension contributions summing
to their composite, rejection reasons inside the
stated vocabulary.

## 3. Why a retrieval fired

**Retained at retrieval time (D-INSP-1, FIXED).**
`_score_records` now retains, beside
`_composite_score`, an `_explanation` per record: the
five dimension values, each dimension's contribution
under the effective weights row (the shipped
`user_fact` row, or the forge's explicit TF-044 table,
named `custom`), the semantic score's source
(`vector_similarity` / `fts_proxy` / `none`), and the
recency score's source (`record_timestamp` — TF-059's
own-timestamp decay — / `source_proxy`). Retention is
at the retrieval moment on purpose: a timestamp's
recency decays, so an explanation recomputed later
would explain a different moment. The arithmetic itself
did not move — it stays in `_score_records`, where
TF-044's census pins the fusion constants, and the
pinned composites (TF-041's 0.49, TF-059's 0.57) are
unchanged.

**Rejections are data.** `retrieve()` retains
`last_retrieval_report`: query, path, `top_k`,
`min_score`, sources, how many candidates were
considered, the fired ids in bundle order, and every
rejected candidate named (by the same identity
`_build_bundle` derives) with its composite and its
reason — `below_min_score` or `beyond_top_k`. The
property returns a detached deep copy.

**Per record (`explain_record`).** Score, cue, source,
the TF-048 citation payload, and one explanation,
its source stated in `explanation_source`:

- `retained_at_retrieval` — the forge's `_explanation`
  (the normal case);
- `recomputed_at_inspection` — `dimension_explanation`
  (the mirror in `inspector.py`) applied to a
  forge-shaped record that predates retention. The
  mirror is pinned against the retained explanation on
  every record shape by the live checker and the tests
  (a timestamped record within a stated tolerance —
  its recency decays between the two calls), so the
  two cannot drift apart silently;
- `carried_evidence` — a record scored by a
  personal-store search's own rule (TF-058/059/060)
  surfaces the evidence that search carried in its raw
  (`CARRIED_EVIDENCE_KEYS`: `matched_terms`,
  `freshness`, `language`). The inspector never
  fabricates forge dimensions for another rule's score.

**Unknown is stated, not hidden.** A bundle with no
retained forge report (hand-built, or a personal-store
search) reports `rejections_known: False` and
`rejected: None` with a note: which candidates were
considered cannot be reconstructed after the fact, and
is reported unknown — not as none.

**The core join (`last_turn_inspector()`).** Joins the
core's own store, its forge's DB path, its TF-038
trace, the turn's bundle, the forge's retrieval report,
and the retained citation check. One guard: the forge
report is passed only when its fired ids are exactly
this turn's bundle records in order — a turn whose
retrieval never reached the forge (DB missing, leg
failed) leaves an older report behind, and pairing it
with this turn would explain the wrong retrieval; the
inspector then reports rejections as unknown instead.

**The trace leg (TF-048's assignment, landed).**
`cognition/traces.py` retrieval entries now carry each
record's citation payload beside its score and cue.

## 4. Boundaries

- The inspector explains retrieval and surfaces
  evidence; it does not re-rank, re-score for verdicts,
  or judge answer quality.
- The knowledge-DB section counts rows; it does not
  read content out of the DB.
- Proof grade: FIXTURE (synthetic records, TF-043's
  fixture DB, synthetic stores, real core turns on this
  VM) — inspection over a real lived-in store is NOT
  PROVEN (no real store on this machine, TF-041), never
  claimed. Live-model turns differ only in who wrote
  the candidates (TF-038's grade), not in the bundle's
  shape.
