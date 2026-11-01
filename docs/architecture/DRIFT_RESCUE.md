# TF-092 — Language Drift Rescue: Pulled Back or Flagged

*An answer that wanders off-language is pulled
back by deletion only — the off-language
sentences are cut, the on-language sentences
kept verbatim — or, when there is nothing
honest to pull back to, flagged with the text
returned verbatim and the drift recorded. The
estate lives in
`src/thoughtforge/refinement/drift_rescue.py`
(`DRIFT_RULES` / `DRIFT_FIXTURES` as data,
`rescue_drift`, the checkers
`drift_rescue_problems` /
`drift_rescue_live_problems`, format
`thoughtforge-drift-rescue/1`).*

## The plain truth first

Until this slice, no language-drift check
existed anywhere in refinement. TF-087
catalogued it plainly: salvage.py has no
language detection at all, and a drifted draft
comes back byte-identical from
`FragmentSalvage` — its fallback is the best
draft verbatim by construction. A grep over
`refinement/` and `cognition/` for
`detect_language` returned nothing this tick.

The tree does own a detector, though — TF-060's
`detect_language` in
`knowledge/multilingual.py`, a heuristic hint
over English, Spanish and German (plus a `cjk`
script label; ties and zero-evidence texts
report `unknown`). This gate **reuses that
hint as-is**. It invents no second detector,
and it inherits TF-060's stated limits exactly:
a hint is never a certainty, and languages
outside the hint table cannot be judged.

That rescue does not change inside salvage —
deliberately. A cut wired into
`FragmentSalvage.forge` would silently rewrite
TF-087's catalogue and TF-089's filed metrics
mid-Part. The rescue is a **caller-invoked
gate**, the shape TF-090's format gate and
TF-091's loop gate took: the caller who
receives a draft calls `rescue_drift(draft,
expected_language)`. The expected language is
the caller's to supply — the language of the
ask or the conversation; salvage has no such
signal and this gate does not guess one from
the draft itself. `FragmentSalvage` is
untouched, so the catalogue and the metrics
stand exactly as banked. TF-093 owns unifying
this gate with the policy surface; TF-095 owns
telling the user; TF-101 scores the full stack
over TF-100's corpus.

## Pulled back — by deletion only, never translated

TF-088 drew the line for refine passes — rescue
never invents content. This gate holds the
same line for language. When an on-language
core remains and drift is not the majority of
the judged sentences, the off-language
sentences are **cut**. Nothing is translated:
a translation would put words in the expected
language that the model never produced there —
fabrication wearing a rescue's clothes (and
TF-060's standing rule: translation is the
model's job at answer time, not a store's or a
gate's). The kept sentences are the model's
own, verbatim and in order, and every cut
segment is recorded — detected language,
sentences verbatim, first sentence index — so
a consumer can see exactly what the gate
removed (TF-097 owns carrying that record into
the turn trace). A rescued text must judge
`clean` on a second pass; the live checker
proves this for every filed rescue.

## Flagged — when cutting would lie

Two shapes are flagged, never silently cut,
and the text comes back **byte-identical**
with the drift recorded and the flag named:

- **The whole answer is off-language**
  (`whole_answer_off_language`) — there is no
  on-language sentence to pull back to.
- **Drift is the majority**
  (`drift_is_majority`) — off-language
  sentences strictly outnumber on-language
  ones among the judged sentences; cutting
  would leave a minority fragment dressed up
  as the answer.

Flagging is the honest outcome for both: the
caller can regenerate, or tell the user what
happened (TF-095 owns that telling).

## How a sentence is judged

Judgement is per sentence, by TF-060's hint:

| Hint vs expected | Standing |
|---|---|
| hint == expected language | on-language — kept, counted |
| hint == another known language, or the `cjk` script label | off-language — cut or flagged, recorded |
| hint == `unknown` (no distinctive evidence — a place name alone, a very short sentence) | neutral — kept, counted on neither side |

Neutral sentences are never cut: a sentence
the hint cannot read is not evidence of drift,
and this gate does not guess. An expected
language outside TF-060's known languages
(`en` / `es` / `de`) cannot be judged at all —
the gate refuses by name
(`unsupported_expected_language`), as it
refuses empty output (`empty_output`).

## The fixtures

16 filed cases, all synthetic Lantern Hollow
material, every one run live through the real
gate by the slice's checker: 4 clean (English,
Spanish, and German answers in their expected
languages; an English answer whose middle
sentence is a bare place name — neutral, kept),
6 rescued (trailing Spanish drift from an
English answer; trailing English drift from a
Spanish answer — the bilingual pair; a Spanish
block in mid-answer with the English around it
preserved; trailing German drift; a single
drifted sentence; drift that opens the answer
and is cut when the answer recovers), 3 flagged
(a whole Spanish answer to an English ask, a
whole English answer to a Spanish ask, and a
majority-drift answer), and 3 refusals (empty
output; an unsupported expected language; an
`unknown` expected language).

## Filed limits, not hidden

- **The hint's languages only.** Expected
  languages, and detected drift, are limited
  to TF-060's table — English, Spanish, German,
  and the `cjk` script label. Drift into or
  out of any other language is NOT detected.
  Widening the table is a TF-060-line slice,
  not this gate's silent claim.
- **Sentence grain only.** A drifted clause
  inside an otherwise on-language sentence is
  not judged — the hint reads sentences, and a
  mixed sentence usually hints as its majority
  or as `unknown` (neutral, kept).
- **A hint is not a certainty.** Short or
  proper-name-heavy sentences can hint wrong
  or `unknown`; TF-060 files the same limit
  for memory records. The gate's thresholds
  (majority of judged sentences) keep one
  mis-hinted sentence from flagging an answer,
  but a mis-hinted sentence in a short answer
  can still be cut — filed, not hidden.

## Docket

- **D-SALV-1's language-drift share is FIXED
  at gate grade by TF-092**: answers wandering
  off-language are pulled back by deletion
  only or flagged verbatim in
  `refinement/drift_rescue.py` — per-sentence
  judgement by TF-060's real hints, no
  translation ever, whole-answer and
  majority drift flagged never cut. The entry
  stays **OPEN** for its one remaining share:
  rambling compression in general (TF-089
  measured it; TF-091 cut only its loop-shaped
  subset, and no general mechanism exists —
  none is claimed here either). TF-016 totals
  unchanged: 88 entries — 45 OPEN /
  1 OPEN-ENVIRONMENT / 39 FIXED /
  3 CLOSED-BY-VERDICT.

## Proof grade

FIXTURE — every filed case recomputed live
through the real gate functions on this VM,
each rescue proven a deletion of its source
and complete on a second pass, each flag
proven byte-identical. A live model's drift
is **NOT PROVEN** (no model in tree); TF-100's
chaos corpus owns real-shaped failures and
TF-101 runs the full policy stack — this gate
included — over them.
