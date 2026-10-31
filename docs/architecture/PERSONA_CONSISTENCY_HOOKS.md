# TF-030 — Persona Consistency Hooks: the Cognition Layer's Half

*Part II, slice TF-030. Code: `src/thoughtforge/cognition/persona.py`
(`PersonaExpectation`, `inject_persona`, `check_persona_response`,
`score_persona_turns`, the synthetic fixtures), wired minimally
into `ThoughtForgeCore._run_turn`. Tests:
`tests/test_tf030_persona_hooks.py` (29 tests). The benchmark half
is TF-122's persona consistency suite (the roadmap's TF-030 note
names TF-128); this slice builds everything in the cognition layer
that such a suite needs to bind to.*

## 1. The gap, as found

The persona_consistency benchmark scores responses against a
hard-coded Skald phrase list (21 generic phrases, 42 Norse markers,
9 Skald quality markers — TF-010). Before this slice the cognition
layer offered it nothing better, because the persona the forge
actually loads barely steered anything:

- **The expressive surface was dormant (TF-009, D-CFG-2).**
  `PersonalityCoreRecord` carries `traits`, `speech_style`,
  `behavior_rules`, `avoid` and five tuning weights, but the only
  consumers were the scaffold's `avoid[:3]` fold and a `record_id`
  in a startup log line. Traits, style and behaviour rules were
  parsed, carried, and silently dropped — a persona whose voice
  never reaches a prompt cannot drift; it was never present.
- **The core's own `personality_fit` never looks at the persona.**
  The fragment/final scorers in `core.py` derive it from a
  genericness penalty on the response text alone (0.7/0.75/0.8 −
  penalty). It measures "not generic", not "in this persona".
- **The benchmark's empty-list vacuous PASS (D-BENCH-4)** stood as
  the estate's consistency arithmetic: `score([])` returned 1.0
  PASS. TF-028 applied the lesson in the coherence scorer; the
  cognition half applies it here too (§4).

One measurement this slice adds: with the **default config**, a
core built on an *empty* memory dir still has a persona in force —
the construction fallback loads the shipped
`configs/personality_core.yaml` (`configs/default.yaml:46`,
`personality_core_path`). Verified live: the expectation's
`record_id` is the shipped persona's (`pers_001`). A genuinely
persona-free core requires an explicit config with no
`personality_core_path` — the tests build one that way, and it is
untouched by everything in this slice.

## 2. The specification (the hooks)

| Hook | Contract |
|---|---|
| `PersonaExpectation.from_record(record)` | A construction-time **snapshot** of the persona in force: record_id, traits, speech_style, behavior_rules, avoid, weights — copied, so later edits to the record object do not re-persona a live conversation. `None` record → `None` expectation. Dict serialisation round-trips exactly. |
| `expectation.steering_lines()` | One line per non-empty expressive list, fixed order: `Persona traits:` / `Persona speech style:` / `Persona behavior:`, entries joined `"; "`, each list capped at `MAX_STEERING_ENTRIES = 6`. |
| `inject_persona(scaffold, expectation)` | Prepends the steering block to `scaffold.fact_block` (the TF-019/024 history channel, the TF-029 will channel). Never duplicates (first-line guard), never touches the per-turn fields (goal/tone/focus/avoid/depth/modes), no-ops on no persona or no expressive lines. |
| `check_persona_response(expectation, text)` | The mechanically checkable subset: the persona's `avoid` entries literally present in the response, after one shared normalisation (lower-case, punctuation stripped, whitespace collapsed). Multi-word entries match as phrases; single-word entries as **whole words only** (`repetition` does not fire inside `repetitions`). Pure function. No persona in force → `ValueError`, not a clean sheet. |
| `score_persona_turns(expectation, responses)` | Aggregate `PersonaConsistencyReport`: per-turn checks, consistent-turn fraction as the score, `PASS` only when ≥1 turn was checked **and** every turn was consistent; an empty batch is 0.0 / `REVIEW`. |
| `FIXTURE_EXPECTATION` + `PERSONA_FIXTURE_TURNS` | Synthetic ground truth (§5) for TF-096 and TF-122. |

**Deliberate non-goals, stated plainly:**

- The `avoid` list gets **no steering line** — it already steers
  through `scaffold.avoid` (universal + intent extras + personality
  `avoid[:3]`, rendered as the prompt's own Avoid line, TF-023).
  Repeating it in the block would spend prompt budget to say one
  thing twice.
- The five tuning **weights get no prompt effect**. They are
  carried on the expectation for scorers and for TF-141's
  persona-compiler verdict; numbers are not prose, and inventing
  prose from them would be a persona compiler — exactly the
  question TF-141 exists to decide. This is the honest remainder
  of D-CFG-2, and the docket now says so (§6).
- Traits/style/behavior are **steering, not scoring** in this
  layer: their consistency is verified by their presence in the
  prompt. Scoring a response's fidelity to "plain-spoken" is
  judgement, owned by TF-122's suite — not smuggled into a phrase
  matcher here.

## 3. How it steers (measured)

The core derives one expectation at construction (from whichever
source loaded the personality — memory store first, config
fallback second) and exposes it as `core.persona_expectation`.
Injection runs in `_run_turn` after history injection and before
the will layer observes the turn, so the fact_block order is:
**will line, persona block, history, knowledge facts**.

Persona steers from the **first** turn — unlike the will layer,
identity is not persistence; there is nothing to wait for.

Measured through the real knowledge-only core over the TF-017
synthetic store (Wrenna's persona saved in the store, empty
knowledge DB), one factual ferry turn:

```
Persona traits: steady; observant; plain-spoken
Persona speech style: short, concrete sentences; names the river level before giving advice; no filler
Persona behavior: answer what was asked; cite the synthetic reference when using it; say plainly when the fixture store is silent
```

— the whole fact_block (no knowledge records were retrieved to
follow it), exactly once. The same turn's `scaffold.avoid` shows
the older channel still working beside it: the 3 universal entries,
the light-conversation intent extras, then the persona's three
avoids in full TF-023 form.

## 4. The check, and the no-vacuous-pass rule

`score_persona_turns` over the five fixture turns (§5) measures:
2 consistent of 5, score 0.4, `REVIEW`, 3 violations — one per
out-of-voice turn, each naming the exact avoid entry tripped.
Over the two in-voice turns alone: 1.0 / `PASS`. Over an empty
list: 0 turns, 0.0, `REVIEW` — the D-BENCH-4 lesson applied in
the cognition half from the start (the benchmark module's own
fix stays TF-122's).

## 5. Fixtures (for TF-096 / TF-122)

`FIXTURE_EXPECTATION` mirrors
`tests/fixtures/synthetic_personality_core.yaml` field for field
(a test asserts the two `to_dict()`s are equal, so the src-side
and test-side copies cannot drift apart silently) — Wrenna the
Lantern-Keeper, invented, no canon. `PERSONA_FIXTURE_TURNS` pairs
five responses with ground truth: two in-voice (violate nothing)
and three out-of-voice, each tripping **exactly one** avoid entry
(the forbidden crossing offered, the vague-handwaving shrug, the
repetition aside) — the shape TF-096's salvage fixtures need: a
voice, and three named ways to fall out of it.

## 6. Docket

- **D-CFG-2 — narrowed, still OPEN.** Traits, speech_style and
  behavior_rules now steer every prompt through the persona block;
  `avoid` already steered (TF-023) and is now also checkable
  through the hook. What remains dormant is exactly the five
  tuning weights: carried on the expectation, given no steering
  effect by deliberate ruling (§2), with the compiler question
  owned by TF-141 as the docket always said. The TF-016 row is
  updated to state this, with this slice as evidence.
- D-BENCH-3 / D-BENCH-4 (the benchmark module's dead `weights`
  argument, lying formula, vacuous PASS) are **not** touched —
  they are TF-122's, and this slice's report arithmetic is the
  contract that suite inherits, not a patch to the old scorer.
