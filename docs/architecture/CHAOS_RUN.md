# TF-101 — Enforcement + Chaos: The Scored Stack Run

*The full policy stack has now been run over
TF-100's chaos corpus and the score is
published: **29 of 38 entries land exactly as
filed (76.3%)** — every demanded rescue
delivered, every clean control passed clean
byte-identical, and 9 entries missed, all of
one shape (filed *refused*, passed *clean*),
each filed below as a named gap with an
owner. The estate lives in
`src/thoughtforge/refinement/chaos_run.py`
(`SCORING_RULES` / `FILED_SCORE` /
`FILED_RESULTS` / `NAMED_GAPS` as data, the
runner `run_chaos`, the checkers
`chaos_run_problems` /
`chaos_run_live_problems`, format
`thoughtforge-chaos-run/1`). No docket entry
was assigned to TF-101 and none is added:
TF-016's totals stand unchanged at 90.*

## The plain truth first

Before this slice the corpus was a set of
demands nobody had scored. TF-100 proved the
cheap legs gate by gate and deliberately left
three legs unjudged — draft-side hallucination
beyond the signalled citation entries, generic
filler, and the invented reference. TF-093
likewise filed a live model's behaviour under
its unified surface as NOT PROVEN, with TF-101
owning the stack score. This slice is that
run, and nothing else: no check was added to
the stack, no gate was retuned, and neither
the corpus nor any expectation in it was
edited to move the number.

## How the run works

Every one of the 38 corpus entries is run
through TF-093's real `enforce_policy` — the
seven checks (format, truncation, loop, drift,
citation, length, genericness) in their filed
execution order, the real gates called, never
re-implemented. Signals are the entry's own,
resolved exactly as TF-100's checker resolves
them:

- `records` naming `lantern_hollow_synthetic`
  resolves to TF-093's filed synthetic record
  pair (`policy_surface._fixture_records()` —
  the Odin Q123 entity and the lantern-oil
  reference chunk);
- `persona` naming `wrenna_synthetic` resolves
  to TF-030's `FIXTURE_EXPECTATION`;
- `structured` / `required_keys` /
  `expected_format` / `truncation_suspect` /
  `expected_language` / `retrieved_qids` pass
  through in `enforce_policy`'s own shapes.

Scoring follows TF-100's filed rules, stated
in `SCORING_RULES` as data: *rescued* requires
a stack verdict of `rescued`; *refused* is
satisfied by `refused` **or** a blocking
`flagged`; *passed-clean* requires `clean`
**and** byte-identical text. Each entry's row
carries its family, owner, expected and actual
outcomes, the stack verdict, flags, problems,
per-check statuses, and — for a mismatch — an
explicit reason and its named gap.

One signal has nowhere to go, measured rather
than assumed: `enforce_policy` takes no
persona signal and `POLICY_CHECKS` names no
persona check (TF-096's voice guard lives in
salvage's selection and refine adoption, not
in the policy surface). The persona signal is
resolved, and TF-096's own `check_voice`
verdict on each persona entry — TF-100's
exact call, against the entry's filed
`source_text` — is recorded as a diagnostic
**outside** the score. It never changes an
entry's scored outcome.

## The score

| measure | result |
|---|---|
| overall | **29 / 38 matched (76.3%)** |
| expected rescued | 12 / 12 |
| expected refused | 12 / 21 |
| expected passed-clean | 5 / 5 |
| stack verdicts | rescued 12 · refused 4 · flagged 8 · clean 14 |
| clean controls byte-identical | 5 / 5 |
| fabrication rescues (hallucination + fabricated_citation) | **0 / 7** |
| fabrication entries passed clean | 4 / 7 (the hallucination family) |

| family | entries | matched |
|---|---|---|
| truncation | 3 | 3 |
| loop_sentence | 3 | 3 |
| loop_cycle | 2 | 2 |
| loop_word | 2 | 2 |
| hallucination | 4 | **0** |
| fabricated_citation | 3 | 3 |
| format_collapse | 4 | 4 |
| language_drift | 3 | 3 |
| persona_voice_loss | 3 | **0** |
| genericness | 2 | **0** |
| refusal_shaped | 2 | 2 |
| empty_or_short | 2 | 2 |
| clean_control | 5 | 5 |

What the stack does well, it does completely:
both truncation rescues and the no-complete-
sentence refusal; all seven loop legs; both
format repairs and both format refusals;
minority drift rescued, majority and whole
drift flagged; both refusal shapes flagged on
the gate's hard phrases; the empty output
refused by name and the two-word answer
flagged too short; and all three fabricated-
citation entries flagged (the foreign QID, and
uncited matched claims). No clean control was
rescued, annotated, or cut — over-rescue is
zero in this run.

## The 9 mismatches, as named gaps

Every mismatch is the same shape — filed
*refused*, stack *clean*, text passed
byte-identical — and belongs to exactly one
gap in `NAMED_GAPS`:

1. **`draft_side_hallucination_not_judged_by_the_policy_surface`
   (owner TF-088; 4 entries — invented price,
   invented destinations, negation flip,
   common-word elaboration).** No policy check
   judges a draft's invented facts as facts.
   The citation check flags only claims
   matching a supplied record, uncited, or
   foreign markers: three entries supply no
   records signal at all, and the negation
   flip's supplied records carry no ferry fact
   for its claim to match, so its citation
   check passes. TF-088's fabrication guard —
   the entries' owner mechanism — judges
   salvage refine steps pool-relatively; it is
   not a draft gate and the surface never
   calls it. A draft-side fabrication
   judgement is the missing mechanism.
2. **`persona_voice_not_a_policy_surface_check`
   (owner TF-096; 3 entries — handwaving,
   repetition, invented crossings).** The
   surface has no persona leg (above), so all
   three pass clean — while the diagnostic
   shows TF-096's own guard naming a new avoid
   violation in every one (`vague handwaving`,
   `repetition`, `invented crossings after the
   second tide bell`). The guard exists; it
   does not live in this stack. A persona leg
   in the policy surface is the missing
   mechanism.
3. **`genericness_checks_hard_phrases_only`
   (owner TF-093; 2 entries — fluent filler,
   vague restatement).** The genericness leg
   is the gate's hard-phrase list and the
   length leg a five-word floor. Both entries
   trip no listed phrase and pass the floor,
   so both pass clean. A judgement of
   answerhood beyond the phrase list is the
   missing mechanism.

These gaps are TF-104's Part V gate
inheritance: they are published here as the
score, not repaired here — each fix belongs
to its owner's surface, and a scoring run
that quietly grew a new gate would falsify
the very number it publishes.

## One caveat on a matched entry

`chaos_citation_invented_reference` lands
flagged, which satisfies *refused* under the
filed scoring rule, and it is counted. But
the flag is `citation:uncited_claim`, raised
through an incidental term match against the
lantern-oil chunk (the claim shares *hollow*
and *lantern* with that record) — the
invented Hollow Almanac itself is named by
no check. Counted as filed; not reference
verification. The caveat is filed in the
module as `ENTRY_NOTES` so it travels with
the score.

## What did not change

TF-093's surface, every Part V gate,
TF-100's corpus and its JSON expectations,
TF-096's guard, and TF-016's docket (no entry
owned by TF-101 — the docket names it only in
D-PROF-4's remainder text, as a future live
panel — and none added; totals unchanged at
90: 43 OPEN / 1 OPEN-ENVIRONMENT / 43 FIXED /
3 CLOSED-BY-VERDICT).

## Proof grade

FIXTURE — the filed corpus run live through
the real policy surface on this VM, the filed
score recomputed on every suite run. A live
model's score is NOT PROVEN: the corpus is
shaped like real failures, it is not a
recording of one, and no number here is a
live enforcement rate. TF-104's gate inherits
this score and its named gaps.
