# TF-096 — Persona-Voice Preservation Under Salvage

*Refinement no longer sands a character's voice
flat. The estate lives in
`src/thoughtforge/refinement/persona_voice.py`
(`VOICE_RULES` / `VOICE_FIXTURES` as data,
`persona_violations` / `voice_problems` /
`check_voice`, the checkers
`persona_voice_problems` /
`persona_voice_live_problems`, format
`thoughtforge-persona-voice/1`), with the guard
wired into `refinement/salvage.py`'s selection
and refine adoption rules and the core passing
its TF-030 expectation in `_compose_final`.
Docket: D-SALV-3, found and FIXED by this slice.*

## The plain truth first

Salvage's two choosing rules could not see voice.
Selection ranks candidates by the composite
(0.45·length + 0.55·citation), and refine adoption
added TF-088's fabrication guard on top of the
same score — neither looks at the persona. That
matters because a filed voice is often *short*:
TF-030's Wrenna speaks in short, concrete
sentences, and the length-weighted score rewards
exactly the padding that flattens her.

Probed live this tick, through the real
`FragmentSalvage` with a stub engine and the real
`PromptBuilder`, against TF-030's filed
expectation:

- handed the in-voice cited draft (144 chars, no
  violations) beside a longer cited rival tripping
  `vague handwaving` (337 chars), **selection
  returned the rival**;
- three stub-engine refine outputs built on
  TF-030's three out-of-voice fixture turns — each
  tripping exactly one avoid entry (`invented
  crossings after the second tide bell`, `vague
  handwaving`, `repetition`), each padded longer
  than the in-voice draft — were **all adopted**
  as `refine_pass_1` over it. A benign in-voice
  refine output adopted too (the control leg).

## What counts as breaking a voice

Only what TF-030 can check. Traits, speech style
and behavior rules are *steering, not scoring*
(TF-030's ruling — scoring a response's fidelity
to "plain-spoken" is judgement, owned by TF-122's
persona suite). The mechanically provable subset
is the persona's `avoid` list, judged by TF-030's
own `check_persona_response`. Voice breakage is
therefore defined **relative to the source**
(TF-088's pool-relative shape): a violation is
salvage's only when the output carries it and the
text it came from does not. A violation already
in a draft is generation's, and salvage passing
it through verbatim — or rephrasing within the
same violation — introduces nothing.

## The guard

Active only when a persona is supplied
(`FragmentSalvage.forge(...,
persona_expectation=...)`; the core now passes
its own expectation, so a shipped turn is guarded
whenever a persona is in force, and unguarded —
byte-identical to before — when none is):

1. **Selection.** When at least one candidate is
   persona-consistent, only the consistent drafts
   are ranked (score order kept within them), so
   an in-voice draft at hand is never passed over
   for a flattened rival. When *no* candidate is
   consistent, score order over all stands — the
   guard restricts choice among what generation
   offered; it invents no in-voice text.
2. **Refine adoption.** Beside TF-088's
   fabrication gate, a pass whose output
   introduces a new avoid violation against the
   current best text is never adopted, however it
   scores. The attempt still counts in
   `passes_used` / `salvage_path` (TF-087's
   naming), and a later clean pass may still
   adopt.

## Filed limits, not hidden

- **The refine prompt still does not carry the
  persona block.** TF-030's block (traits /
  speech style / behavior) rides the scaffold's
  `fact_block`, and `build_refine_prompt` never
  reads `fact_block` — refine generation is
  steered by tone / goal / `scaffold.avoid` (which
  in core flow does carry the persona's avoids,
  TF-023/TF-030), not by the block itself. The
  prompt's assembly is TF-023's frozen contract,
  so this slice enforces at the adoption gate,
  where a flattened output can be refused
  whatever the prompt asked for. Whether the
  refine prompt should carry the block is a
  PromptBuilder-contract question, not taken here.
- **Flattening that trips no avoid entry is the
  residual.** A longer, blander rephrase in common
  words is not mechanically decidable against
  legitimate rephrasing (TF-088's residual, in
  voice's clothing). This guard does not claim
  it: TF-122's persona suite scores voice
  fidelity, and TF-104's gate owns the estate.
- **Selection's preference is consistency-first,
  not score-first.** Within the consistent drafts
  the composite still decides; a consistent draft
  with a weaker score beats an inconsistent one
  with a stronger score by design — that is the
  slice's whole point, filed plainly so no later
  reader mistakes it for a scoring bug.

TF-016 totals updated: 89 entries — 45 OPEN /
1 OPEN-ENVIRONMENT / 40 FIXED / 3
CLOSED-BY-VERDICT (D-SALV-3 added + FIXED by
TF-096; primary-owner Part V 4 → 5).

## Proof grade

FIXTURE — ten filed cases run live through the
real `FragmentSalvage` on this VM, judged by
TF-030's real checker against its filed
expectation: the three flattened refine outputs
rejected with the in-voice candidate standing,
the benign and same-violation refine legs
adopted, selection preferring the in-voice draft
only under a persona, both no-persona contrast
legs byte-identical to before this slice, and
the all-out-of-voice and lone-draft fallbacks
verbatim. A live model's voice drift is
**NOT PROVEN** (no model in tree); TF-122's
persona consistency suite owns the full
judgement.
