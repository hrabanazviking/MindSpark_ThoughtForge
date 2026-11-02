# TF-088 — Salvage Safety: Where Repair Ends and Fabrication Begins

*Rescue never invents content — the line specified
as data and enforced in the refine adoption rule.
The estate lives in
`src/thoughtforge/refinement/salvage_safety.py`
(`SAFETY_RULES` / `PROBLEM_KINDS` /
`ADVERSARIAL_FIXTURES` as data, the guard
`fabrication_problems` / `classify_output`, the
checkers `salvage_safety_problems` /
`salvage_safety_live_problems`, format
`thoughtforge-salvage-safety/1`), wired into
`src/thoughtforge/refinement/salvage.py`.*

## The plain truth first

Most of salvage could never invent anything: picking
the best draft returns it word for word, and pulling
fragments out of drafts returns the model's own
sentences. The danger lived in exactly one place —
the **refine pass**, where an engine writes a fresh
answer from the fragments and salvage decides whether
to keep it. That decision was made by score alone
(length and citation share), and a score cannot see
new facts.

Probed live before the fix, with a stub engine over
one cited ferry draft, **five invented answers were
all adopted**: one that added a price of 50 silver
coins, one that cited a record Q999 that was never
retrieved, one that flipped the answer to "does not
leave … never lights", one that sailed the ferry to
Bergen and Reykjavik, and one that moved dawn to
midnight. Each scored better than the honest draft
because each was longer and still carried Q123. That
is D-SALV-2, and it is fixed in this slice.

## The line

**Repair changes form. Fabrication changes checkable
content.**

Repair may: select a draft, delete material, reorder
it, reassemble verbatim fragments, and change
punctuation, casing or whitespace.

Fabrication is any of these arriving in a refine
output without a source carrying them — each one is
a named problem kind, judged mechanically against
the **source pool** (the current best text, the kept
fragments, and the retrieved records' cues — exactly
the material the refine prompt is built from, plus
the retrieved QIDs):

| Kind | What it catches |
|---|---|
| `new_number` | a digit-form number no source carries (QID digits are citations, never numbers) |
| `new_number_word` | a number word (three and up, hundred/thousand/million/billion/dozen) no source carries |
| `new_citation` | a QID neither retrieved nor present in a source |
| `new_named_entity` | a capitalised name mid-sentence, or an all-caps token, no source carries |
| `negation_introduced` / `negation_dropped` | polarity flipped either way — a meaning change, not a repair |
| `time_marker_displaced` | a new time word (dawn, midnight, weekdays, months…) arriving while a different source time word disappears |

The enforcement is one gate in
`FragmentSalvage._refine_with_engine`, placed
**before** the score gate: a pass with any problem
is not adopted, however well it scores. Nothing is
silently rewritten to remove the invention — the
pass is simply refused, the current best text
stands, and the attempt still counts in
`passes_used` / `salvage_path` (TF-087's rule that
paths name attempts). A later pass may still
produce a clean text and be adopted on score.

## The adversarial set

Nine fixtures are filed in `ADVERSARIAL_FIXTURES`
(synthetic Lantern Hollow material, never canon)
and run through the real `FragmentSalvage` by the
live checker: the five probed fabrications above,
a negation dropped from a negated draft, a benign
rephrase (adopted), the residual case below, and a
draft whose own inventions pass through verbatim
without an engine — salvage did not invent those;
generation and enforcement judge drafts, salvage is
pool-relative by design (`source_invention_is_not_
salvage_invention`).

## The residual — stated, not hidden

- **Common-word elaboration is not judged.** A
  refine output can add factual-sounding claims in
  plain nouns with no new marker — TF-087's own
  adopted fixture adds a keeper, a red lantern and
  a winter timetable to a one-sentence draft, and
  this guard returns no problem for it (filed as
  the `common_word_elaboration_residual` fixture).
  No mechanical checker can separate that from a
  legitimate rephrase, so this slice does not claim
  it: TF-089's corruption rate measures it on the
  fixture corpus, and TF-104's gate runs this
  slice's marker-adversarial set at zero.
- **Sentence-initial capitals are unjudged.** A name
  opening a sentence is indistinguishable from an
  ordinary sentence start, so the entity check
  judges mid-sentence capitals and all-caps tokens
  only.
- **Time words merely added** (the source's own time
  kept, another added alongside) count as common-word
  elaboration; only displacement — new time in, source
  time out — is a named fabrication.

## Not proven — the honest remainder

- **A live model's fabrications: NOT PROVEN.** Every
  adversarial output here is a deterministic stub;
  no model ships in this tree. TF-100's chaos corpus
  owns real-shaped failures and TF-101 runs the full
  policy stack over them.
- **Semantic contradiction in common words** (a
  swapped harbour for a river, say) is the residual
  class above — measured later, never claimed here.

## Docket

- **D-SALV-2 added + FIXED by TF-088**: the refine
  adoption rule was score-only, and all five probed
  adversarial inventions were adopted as
  `refine_pass_1`. The fix is the
  `fabrication_problems` gate above, wired into
  `salvage.py` before the score comparison.
- TF-016 totals move to 88 entries — 45 OPEN / 1
  OPEN-ENVIRONMENT / 39 FIXED / 3
  CLOSED-BY-VERDICT (Part V 3 → 4).

## Proof grade

FIXTURE — the five fabrications were adopted live
before the fix and are rejected live after it,
through the real `FragmentSalvage` with a stub
engine and the real `PromptBuilder` on this VM;
clean refine legs (benign rephrase, TF-087's
adoption fixture) still adopt. A live model's
broken or inventive outputs are **NOT PROVEN**,
never claimed.
