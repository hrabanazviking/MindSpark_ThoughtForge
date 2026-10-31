# TF-050 — Contradiction Handling: New Fact vs Stored Fact

*Part III, slice TF-050. Code:
`src/thoughtforge/knowledge/contradictions.py` (the
detection rules, the scoring, the resolution rule,
`assess_contradiction()`,
`apply_contradiction_resolution()`, and
`contradiction_problems()` re-deriving every claim below
from the live tree), wired into TF-047's writeback leg in
`src/thoughtforge/knowledge/writeback.py`. Tests:
`tests/test_tf050_contradictions.py`. The lifecycle
spec's §27 named the strategies; this slice is the first
code that implements any of them.*

## 1. The question and the measured starting state

The roadmap asks: when a new fact contradicts a stored
fact, is the conflict surfaced, scored, and resolved by a
stated rule — or silently overwritten (or silently piled
up)? Measured before this slice, the honest answer was
**nothing handled it at all**:

- TF-047's writeback leg appended every contract-valid
  fact and preference blind. A fact stating the ferry
  leaves at 09:00 landed beside a stored fact stating
  07:30 — both live, neither flagged, no record anywhere
  that a conflict had happened.
- The only contradiction machinery in the tree was
  TF-028's answer-side coherence anchors (a generated
  answer contradicting a probed fact scores 0 — it
  polices output, not ingestion) and the lifecycle spec's
  §27, which listed strategies (mark superseded / reduce
  confidence / archive / preserve both if
  context-dependent) that no code implemented.
- The research wing's `contradiction_record` schema
  (claim_a / claim_b / reason / preferred /
  resolution_state) described the shape a conflict record
  should carry, but bound nothing.

## 2. Detection

`assess_contradiction(new_record, stored_records)` is
pure. A new `UserFactRecord` or `UserPreferenceRecord`
is compared against stored records **of its own kind**
only. A pair contradicts under exactly the rules in
`CONTRADICTION_RULES`:

| rule | strength | fires when |
|---|---|---|
| `fact_value_slot` | 1.0 | both facts carry the same normalised `fact_value` slot key with different normalised values |
| `preference_value` | 0.9 | both preferences share a normalised category and carry different non-empty values |
| `negated_claim` | 0.7 | the summaries' distinctive-term Jaccard overlap ≥ `MIN_SUMMARY_OVERLAP` (0.5), with ≥ 2 shared terms, and exactly one summary is negated (`NEGATION_MARKERS`) |

Normalisation is stated, never model-judged: slot keys
lowercased with spaces/dashes folded to underscores;
values compared as normalised text, with numbers
compared numerically (`30` == `30.0` == `"30"`) and
booleans as words; summary terms are lowercase tokens of
length ≥ 3 with a stated singular fold, minus stopwords
and the negation markers themselves.

**Scope guard (spec §27.3).** When both records carry
tags and the tag sets are disjoint, the pair is *not* a
contradiction: concise replies generally and detailed
answers for project docs are different scopes holding
different values, legitimately. One shared tag, or
either side untagged, leaves the pair in scope.

**Duplicates are not contradictions.** A new record
that carries nothing the stored one does not already
carry (same summary and slot values; same category and
value) is a duplicate. A record carrying a slot the
stored one lacks *expands* it (spec §10.2) and is
neither duplicate nor contradiction — an early draft of
this slice classified expansions as duplicates, which
would have reinforced the new slot away; the slice's
own tests caught it and the rule now requires the new
record's slots to be a subset of the stored one's. A
pair contradicting on any slot is never also a
duplicate.

## 3. Scoring

Every conflict carries its arithmetic
(`ConflictRecord`): each side's **strength** is
`0.6·confidence + 0.4·provenance authority`, where
authority comes from `SOURCE_AUTHORITY` over TF-042's
`MEMORY_PROVENANCE` (`user_stated` 1.0, `manual_entry`
0.9, `conversation_writeback` 0.6, `import` 0.5,
`synthetic_fixture` 0.4; an unstamped legacy record
rates 0.3), plus a stored-side reinforcement bonus of
0.02 per `times_reinforced`, capped at 5 — a fact
confirmed five times is harder to move than one stated
once. The **conflict score** is `0.5·rule strength +
0.25·new strength + 0.25·stored strength`: a conflict
matters most when the rule is hard and both sides are
credible.

## 4. The resolution rule

`RESOLUTION_RULE`, driven by the strength margin
`new − stored`, with `WIN_MARGIN` = 0.15:

- **margin ≥ 0.15 — `new_supersedes_stored`.** The
  stored record is archived: it leaves the live store
  through the store's own rewrite shape (the same
  removal TF-047's archive leg performs), and the new
  record is appended. The old claim is never edited in
  place; the `ConflictRecord` naming both claims, both
  values, the score, and the margin is the surface.
- **margin ≤ −0.15 — `stored_stands`.** The new record
  is not written; the stored record is left
  byte-identical. The writeback leg refuses the entry
  with the conflict named as the reason.
- **in between — `preserve_both_flagged`.** Neither
  side earned the win: both stay live, both records'
  confidence is multiplied by
  `CLOSE_CALL_CONFIDENCE_FACTOR` (0.8 — spec §27.2's
  "reduce confidence", applied to both because the
  conflict itself is evidence against each), and the
  conflict rides the outcome.
- **duplicate — reinforce.** The stored record's
  `times_reinforced` increments and its status becomes
  `reinforced`; no second copy is appended.
- If one new record meets several stored records, any
  `stored_stands` blocks the whole write (the strongest
  stored defence decides); supersedes and downgrades
  otherwise apply per conflict.

`apply_contradiction_resolution(store, new_record,
report)` performs a non-blocked report: the new record
is appended through the store's ingestion gate, and
stored-side mutations are applied to a list reloaded
*after* the append, so a rewrite can never erase the
new record. Applying a blocked report raises
`ValueError` — it is a programming error, not an
outcome.

## 5. Wiring

TF-047's writeback leg routes its fact and preference
writes through one landing helper
(`writeback._land_record`): construct → contract/score
gates (unchanged) → assess → blocked refuses by name,
otherwise apply. Every conflict found rides the
outcome as data (`WritebackOutcome.contradictions`, a
list of `ConflictRecord` dicts) on *every* branch —
superseded, stood, preserved, or refused. The
enforcement gate still fires first: a failed turn is
assessed against nothing and writes nothing (pinned).
Episodes and patterns are out of scope by ruling: an
episode is a moment (two moments cannot contradict) and
a pattern is a procedure, not a claim.

## 6. Stated boundaries

- Detection is structural, not semantic entailment.
  Two prose summaries that contradict without a shared
  slot, a shared category, or a negation marker over a
  shared subject are **not caught**; a coincidental
  subject overlap with a stray negation can flag a pair
  a human would call compatible. Flagging with both
  claims preserved is the safe direction; the only
  removal path (archival) requires the winning margin,
  and every removal is surfaced.
- `RecordStatus` has no "superseded" value, and TF-042's
  validator closes that vocabulary — so supersession is
  expressed with the estate's existing archive shape
  rather than an invented status that would fail
  ingestion elsewhere.
- Cross-kind conflicts (a preference against a fact)
  are not assessed: the kinds' claim shapes differ, and
  no rule in the spec joins them.
- The knowledge DB (Wikidata/ConceptNet/…) is not
  assessed: it is a read-only reference estate here,
  not user memory being written.

## 7. Proof

Fixture grade: synthetic record pairs through the pure
assessor (each rule, the scope guard both ways,
duplicates, expansions, both-negated prose, margins at
exactly ±0.15), synthetic stores through the applier
(archive-out, byte-identical stand, dual downgrade,
reinforce-without-append), and real writeback turns
through TF-047's leg (blocked fact, superseding fact,
close-call fact, duplicate fact, blocked preference,
enforcement-gate precedence).
`contradiction_problems() == []` re-derives the whole
leg from the live tree, and `writeback_problems()` now
includes it. Contradiction behaviour over a real,
lived-in store is NOT PROVEN — no real store exists on
this machine (TF-041).
