# TF-100 — The Chaos Corpus

*Part V's proving ground now exists as one
published, versioned fixture set: 38 real-shaped
small-model failures — each with the prompt that
produced it, the outcome the stack must reach,
and the strategy that owns it — ready for
TF-101's full-stack scoring run, TF-142 and
TF-104's gate. The estate lives in
`src/thoughtforge/refinement/chaos_corpus.py`
(`CHAOS_RULES` / `FAMILIES` / `FAMILY_MINIMUMS`
as data, the loader `load_chaos_corpus`, the
checkers `chaos_corpus_problems` /
`chaos_corpus_live_problems`, format
`thoughtforge-chaos-corpus/1`), and the corpus
itself is data, filed at
`tests/fixtures/chaos_corpus.json` — TF-017's
fixture home — loaded by the module, never
hardcoded in tests. No docket entry was
assigned to TF-100 and none is added: TF-016's
totals stand unchanged at 90.*

## The plain truth first

Before this slice there was no chaos corpus.
Each Part V gate had filed deep fixtures of its
own mechanism, and two older corpora scored
neighbouring questions — but nothing collected
the failure modes in one place, in one shape a
whole policy stack could be run over and
scored against.

Measured at the start of this tick:

- **TF-089's corpus is 18 entries with a
  different job.** TF-087's nine broken fixtures
  and TF-088's nine adversarial refine outputs,
  held as tuples inside their modules and shaped
  for their own runners (candidate lists, stub
  engine texts), exist to measure salvage's
  rescue and corruption *rates*. No entry
  carries a prompt, an owner, or an expected
  outcome in TF-101's vocabulary.
- **TF-062 is not an output corpus at all.** Its
  adversarial set is poisoned *memory records*,
  screened at the store boundary — what a model
  is fed, not what a model produced.
- **Whole failure modes had no fixture home.**
  Draft-side hallucination (as distinct from
  refine-side fabrication), generic filler
  non-answers, refusal-shaped non-answers, and
  empty or two-word outputs appeared in gate
  internals at most, never as named, scored
  corpus entries.

## The corpus

Thirty-eight entries in thirteen families, all
invented in TF-017's Lantern Hollow world,
all marked synthetic in the data file:

| family | entries | what it proves |
|---|---|---|
| truncation | 3 | cut mid-sentence and mid-word (rescued by cutting to the last complete sentence); no complete sentence at all (refused — an ending is never invented) |
| loop_sentence | 3 | a five-fold run, a loop in the middle of a sound answer, casing/punctuation variants of one loop |
| loop_cycle | 2 | a period-2 cycle, and a cycle with a truncated partial tail |
| loop_word | 2 | a two-word loop inside a sentence; a single-word stutter |
| hallucination | 4 | an invented price, invented destinations, a negation flip against a stored fact, and TF-088's common-word elaboration residual |
| fabricated_citation | 3 | a foreign QID no record owns, an invented almanac reference, a matched claim carrying no citation |
| format_collapse | 4 | trailing comma and truncated JSON (repaired without invention); prose answering a JSON ask and a missing required key (refused) |
| language_drift | 3 | minority drift (rescued by deletion), majority drift and a whole off-language answer (flagged, scored refused-side) |
| persona_voice_loss | 3 | handwaving, repetition and invented crossings — each judged against its in-voice source text |
| genericness | 2 | fluent filler and a question restated as its own answer |
| refusal_shaped | 2 | polite refusals of benign asks the reference answers plainly |
| empty_or_short | 2 | the empty output (the corpus's one empty text, by design) and a two-word non-answer |
| clean_control | 5 | plain prose, valid JSON, a properly cited claim, a double repeat that is emphasis not a loop, and a wholly Spanish answer to a Spanish ask |

Outcomes: 12 rescued, 21 refused, 5
passed-clean. Every family carries at least
two entries, filed as `FAMILY_MINIMUMS` data —
no family rests on a single example — and the
roadmap-named families carry more, because
each must show both sides of its line where a
line exists (repair vs refusal, marker tier vs
residual, loop vs emphasis).

## The outcome vocabulary

TF-101 scores `rescued` / `refused` /
`passed-clean`, and the corpus files exactly
what those words demand:

- **rescued** — an honest repair (deletion or
  re-delimiting only, TF-088's line) leaves a
  usable answer from text already present.
- **refused** — the output must not be
  delivered as an answer. In TF-101's scoring a
  `refused` verdict *or* a blocking `flagged`
  verdict satisfies the class; a pass-clean
  never does.
- **passed-clean** — reserved for the clean
  controls: already sound, and every gate must
  leave the text byte-identical. Over-rescue is
  a scored failure, so the corpus guards both
  directions.

An expected outcome is a **demand, not a
claim**. Where an existing gate already decides
the filed way, TF-100's live checker proves it
on this corpus: both truncation legs, every
loop leg, the format legs, the drift legs, the
persona legs' new violations, the foreign-QID
and uncited-claim legs, the refusal and
empty/short legs, and every clean control under
its own gate. Where no cheap gate decides —
draft-side hallucination beyond the signalled
citation entries, generic filler, the invented
reference — the entry is filed with its
rationale and TF-101's scored run owns the
judgement. If the stack passes one of those
clean, that is a scored gap for TF-101 and
TF-104's gate to face, not a corpus error to
quietly re-file.

## Signals and owners

Each entry names its owner from a filed
vocabulary (`TF-090` format/truncation, `TF-091`
loops, `TF-092` drift, `TF-088` fabrication,
`TF-049/TF-093` citations, `TF-096` voice,
`TF-093` policy legs, `TF-093/TF-095` refusal
shapes) and may carry caller signals in TF-093's
vocabulary (`structured` + `required_keys`,
`truncation_suspect`, `expected_language`,
`records`, `retrieved_qids`, `persona`). Signal
targets are named synthetic sets only —
`lantern_hollow_synthetic` records and the
`wrenna_synthetic` persona (TF-017's Wrenna) —
which TF-101 resolves to the real synthetic
fixtures when it runs the stack. Persona
entries additionally carry their `source_text`:
voice loss is a relative judgement (TF-096),
meaningless without the in-voice text the
output flattened.

## What did not change

TF-087/TF-088's fixtures, TF-089's filed rates
and thresholds, TF-062's screening estate, and
every Part V gate are untouched. This corpus
neither merges nor recounts them; where a
failure shape resembles an older fixture, the
entry is a new row in a new shape (prompt,
owner, outcome), not a copy — the checker's
prior-corpora rule files that boundary.

## Proof grade

FIXTURE — 38 synthetic entries loaded from the
filed JSON and judged by the real gates on this
VM. A live model's failures are NOT PROVEN:
the corpus is *shaped like* the failures small
models make, it is not a recording of any
model, and no number here is a live rescue or
refusal rate. TF-101 runs the full policy stack
over this corpus and publishes the score;
TF-104's gate inherits it.
