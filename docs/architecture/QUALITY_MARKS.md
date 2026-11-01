# TF-102 — User-Visible Quality Marks: a Leaving Answer Can Show Its Marks

*Where the product surfaces answers, a rescued or
cited answer can now **show its marks**: detached
data a consumer renders however its UI chooses.
The estate lives in
`src/thoughtforge/refinement/quality_marks.py`
(`MARKS_RULES` / `MARK_KINDS` / `RESCUE_LABELS` /
`MARKS_FIXTURES` as data, `quality_marks` /
`marks_line`, the checkers
`quality_marks_problems` /
`quality_marks_live_problems`, format
`thoughtforge-quality-marks/1`). No surface,
gate, catalogue, or salvage code is changed;
the UI itself is the consumer's, and none is
built here.*

## The plain truth first

Until this slice, an answer that left the forge
carried its quality truth only as internal data.
A policy result knew its verdict, its per-check
outcomes, and its flags — but nothing turned
that record into a mark a product surface could
show. The only visible traces were accidents:
the enforcement gate appends its `[Forge: …]`
notes into the answer text itself on a citation
or length failure (a genericness or drift flag
leaves no visible trace at all), citation
markers ride inside the prose with no summary
of what was cited, and TF-097's rescue record
rides the trace — inspector data, not a mark.
TF-095 built the telling for the answer that
does *not* leave (a block); this slice is the
equivalent for every answer that does.

Two probed truths shape the whole slice:

- **A passed citation check is not a cited
  answer.** The weather answer under records
  passes its citation check claiming no stored
  fact and citing nothing. A mark layer reading
  "check passed" as "cited" would badge answers
  that cite nothing.
- **Salvage's path string names attempts, not
  adoptions** (TF-097's debt, probed again): a
  fabrication-rejected refine returns
  `refine_pass_1` with the draft's text intact.
  Only TF-097's rescue record says which pass
  the final text actually came from.

## The four marks

`quality_marks(result, salvage_result=…)` reads
the records — TF-093's or TF-094's result,
and/or a salvage result carrying TF-097's
record — and returns at most one mark per kind,
in this order:

1. **cited** — the citation check *passed* and
   at least one retrieved marker is actually
   cited. Per-claim mode names the claim
   check's own markers (`Q123`,
   `ref:chunk_7`); legacy-floor mode names the
   QIDs in the result's own final text, with
   the mode stated in the mark. A skipped
   check earns nothing; a flagged citation
   earns the flagged mark, never this one.
2. **rescued** — a repair or cut was applied
   and adopted, with **every mechanism named**:
   the policy checks that rescued (structured
   format repaired; cut back to the last
   complete sentence; repeated passages cut;
   off-language sentences cut) and, when
   TF-097's record says the final text came
   from a refine pass, salvage's **adopted**
   pass by number. The adopt-then-fabricate
   fixture pins the distinction: its path says
   pass 2 (attempted), its mark says pass 1
   (adopted). A rejected attempt, and a result
   with no rescue record at all, earn no
   salvage mark — the path string alone would
   lie.
3. **flagged** — the answer leaves under
   standard carrying flags: every flag
   verbatim, each beside the plain reason
   TF-095's `REASONS` catalogue files for it
   (the catalogue is reused, never edited; the
   data checker proves it covers every flag
   the estate's filed fixtures can produce).
4. **warning** — TF-094 relaxed: the same
   flags, downgraded, ride as warnings verbatim
   with their reasons, and the verdict earned
   underneath them is recorded. A flag is a
   warning or a flag, never both.

`marks_line(marks)` is the one filed plain-text
rendering (`Forge marks: … .`) for surfaces
that want words. Everything else about display
— badges, colours, placement — is the
consumer's UI, deliberately not built here.

## The honesty edges

- **Only a leaving answer gets marks.** A
  refused result — TF-093's refusal or TF-094's
  strict block — raises `ValueError`: a block
  is told by TF-095's refusal notice, and
  dressing it in quality marks would be a
  fabrication. An empty salvage result, and a
  call with neither record, raise the same way.
- **A plain answer has no marks, stated.** A
  clean, unrescued, uncited answer's mark set
  is empty: `has_marks` is False and
  `marks_line` is the empty string, so a
  consumer shows nothing. There is no invented
  "checked" badge — checks that were skipped
  were not passed, and this layer claims only
  what the records earn.
- **Marks never alter the answer.** The layer
  calls no gate, re-judges no draft, and
  mutates nothing it reads; a result's text is
  byte-identical after being marked (the tests
  pin this).

## Filed limits, not hidden

- **A cited mark names markers, not titles.**
  A policy result does not carry TF-048's
  payload (label / source / locator), so the
  mark names the markers the answer cites; a
  consumer holding the retrieval bundle can
  join them to titles itself.
- **Salvage contributes no cited mark.** Its
  `citations` list is a draft-scoring count,
  and TF-093 ruled scoring is not enforcement —
  citation truth has exactly one source, the
  policy surface, so a salvage-only answer can
  earn a rescued mark but never a cited one.
- **No shipped turn emits these marks yet.**
  The core does not route through the policy
  surfaces (the standing TF-093/TF-094/TF-095
  limit), so this slice builds the data every
  leaving result can be marked with; wiring it
  into a consumer is that consumer's slice —
  Part IX owns the bindings.

TF-016 totals are unchanged: 90 entries —
43 OPEN / 1 OPEN-ENVIRONMENT / 43 FIXED /
3 CLOSED-BY-VERDICT (no docket entry is owned
by TF-102; the slice closes a gap the roadmap
itself assigned, and adds no entry).

## Proof grade

FIXTURE — every filed case (fifteen live
policy/configured cases: clean, cited by entity
and by reference chunk in per-claim mode, cited
under the legacy floor, the no-claim pass that
earns no cited mark, four single-mechanism
rescues, four flagged shapes, a rescued-and-
flagged answer, a relaxed warning; two blocked
or empty cases that must raise; five salvage
cases through TF-097's real fixtures including
the rejected attempt and the adopt-then-
fabricate pass naming; one combined
policy-plus-salvage rescue; and one hand-built
record-less salvage result) is recomputed live
on this VM and judged against the records —
kinds in order, markers and modes, mechanisms
and the adopted pass, flags verbatim with
catalogued reasons. A live model's marks are
**NOT PROVEN** (no model in tree).
