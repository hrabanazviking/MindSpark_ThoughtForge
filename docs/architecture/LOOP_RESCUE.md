# TF-091 — Repetition & Loop Rescue: Detected and Cut by Rule

*Degenerate loops — a small model's classic
failure — are detected mechanically and cut by
deletion only: the first full occurrence of the
loop unit is kept verbatim, the excess copies
are dropped and recorded. The estate lives in
`src/thoughtforge/refinement/loop_rescue.py`
(`LOOP_RULES` / `LOOP_CONSTANTS` /
`LOOP_FIXTURES` as data, `rescue_loops`, the
checkers `loop_rescue_problems` /
`loop_rescue_live_problems`, format
`thoughtforge-loop-rescue/1`).*

## The plain truth first

Until this slice, no loop cut existed anywhere.
TF-087 probed its filed 20× loop fixture —
*"The bell rings."* twenty times, 319 chars —
coming back byte-identical from
`FragmentSalvage` (its dedup touches the
fragment list, never the output text), and
TF-089's metrics file the loop among the five
unrescued verbatim. The other loop shapes
probed this tick — a sentence cycle, a loop in
mid-answer, a phrase loop with no punctuation —
have no detector either: salvage's fallback
returns the best draft verbatim by
construction.

That does not change inside salvage —
deliberately. A cut wired into
`FragmentSalvage.forge` would silently rewrite
TF-087's catalogue and TF-089's filed metrics
mid-Part. The rescue is a **caller-invoked
gate**, the shape TF-090's format gate took:
the caller who receives a draft calls
`rescue_loops`. `FragmentSalvage` is untouched,
so the catalogue (loop fixture still verbatim
through salvage) and the metrics (rescue 2/7,
unrescued verbatim 5) stand exactly as banked.
TF-093 owns unifying this gate with the policy
surface; TF-095 owns telling the user; TF-101
scores the full stack over TF-100's corpus.

## The line: a cut deletes, never rewrites

TF-088 drew the line for refine passes — rescue
never invents content. This gate holds the same
line for repetition. A cut may only delete
excess copies of a unit the model itself
produced. The first full occurrence of every
loop unit is kept verbatim; nothing is
reworded, completed, or summarised; and every
cut is recorded — kind, kept unit, full-repeat
count, and the dropped text verbatim — so a
consumer can see exactly what the gate removed
(TF-097 owns carrying that record into the
turn trace).

## Detection: exact, by stated thresholds

Two detectors, both comparing *normalised*
forms (the kept text is always the first
occurrence verbatim):

| Detector | Unit | Normalisation | Loop at | Cut to |
|---|---|---|---|---|
| sentence | a period of 1–3 sentences | casefold, whitespace collapsed, terminal punctuation stripped | ≥ 3 full consecutive repeats | one period occurrence (`sentence_loop` for a 1-sentence period, `sentence_cycle` above that) |
| word | a block of 1–8 words | casefold, surrounding punctuation stripped | ≥ 4 full consecutive repeats | one block occurrence (`word_loop`) |

Because comparison is normalised, *"The bell
rings. the bell rings! THE BELL RINGS."* is a
loop of three and is cut to its first,
verbatim, occurrence. Because the thresholds
are the rule, repetition below them passes
byte-identical with verdict `clean`: a sentence
said twice may be emphasis, and this gate does
not judge it; varied sentences that merely
share a stem (*"…at dawn / at dusk / at noon"*)
are not a cycle and pass untouched.

A **partial tail** trailing a proven loop — a
whole sentence matching the cycle's next
expected sentence, or a word that is a
truncated copy of the expected next word
(*"…the bell rings the bel"*) — is dropped
with the loop and recorded in the cut's
`partial_units`. An incomplete copy is never
completed: *"the bel"* does not become
*"the bell"*, it disappears with the loop that
spawned it, exactly as TF-090's truncation
closure drops a cut value rather than guessing
it.

A rescued text must judge `clean` on a second
pass through the gate — the live checker proves
this for every filed rescue, so a rescue is
complete, not a first instalment.

## The fixtures

16 filed cases, all synthetic Lantern Hollow
material, every one run live through the real
gate by the slice's checker: 3 clean (ordinary
prose; a doubled sentence; varied bell
sentences), 12 rescued (TF-087's own 20× loop
fixture, cut to one *"The bell rings."*; a run
at the exact threshold of 3; a loop in
mid-answer with the prose around it preserved;
a two-sentence cycle, and the same cycle with
a partial tail; case/punctuation variants; two
loops in one draft, cut in order; a phrase loop
with no punctuation; a phrase loop mid-sentence;
a loop ending in a truncated word; a
single-word stutter; and the loop-shaped subset
of TF-087's rambling fixture), and 1 refusal
(empty output).

## Filed limits, not hidden

- **Exact detection only.** A cycle whose
  copies differ by a word each time, and
  paraphrase loops generally, are NOT detected.
  TF-100's chaos corpus may widen this; it is
  not claimed here.
- **Consecutive only.** A refrain returning
  after other material, outside a consecutive
  cycle, is not cut — loop and refrain cannot
  be told apart mechanically at that distance,
  and this gate does not guess.
- **Rambling is not compressed.** Only the
  loop-shaped subset of TF-087's rambling
  fixture is cut — one phrase repeated verbatim
  24 times collapses to its first occurrence,
  *"The river is nice and"* — and what the cut
  leaves is a fragment, filed as such. Loop
  rescue cuts repetition; it does not polish
  the remainder (TF-090's truncation gate
  judges endings). Non-repetitive rambling
  still has no mechanism: TF-089's measured
  share stands.

## Docket

- **D-SALV-1's repetition/loop share is FIXED
  at gate grade by TF-091**: degenerate loops
  are detected and cut by rule in
  `refinement/loop_rescue.py` — sentence runs
  and cycles at ≥ 3 full repeats, word blocks
  at ≥ 4, partial tails dropped never
  completed, every cut recorded, and
  TF-087's loop fixture now rescues to exactly
  one occurrence. The entry stays **OPEN** —
  its remaining shares keep their owners:
  rambling compression in general (TF-089
  measured, still no mechanism beyond the
  loop-shaped subset cut here) and language
  drift (TF-092). TF-016 totals unchanged:
  88 entries — 45 OPEN / 1 OPEN-ENVIRONMENT /
  39 FIXED / 3 CLOSED-BY-VERDICT.

## Proof grade

FIXTURE — every filed case recomputed live
through the real gate functions on this VM,
each rescue proven a deletion of its source
and complete on a second pass. A live model's
degenerate loops are **NOT PROVEN** (no model
in tree); TF-100's chaos corpus owns
real-shaped failures and TF-101 runs the full
policy stack — this gate included — over them.
