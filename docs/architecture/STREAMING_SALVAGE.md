# TF-099 — Streaming Salvage

*What can be rescued mid-stream is now specified
per strategy, and the two strategies that are
honestly mid-stream-safe are built: a
sentence-boundary streaming rescuer that cuts
period-1 sentence loops and drops off-language
sentences while a candidate is still arriving.
The estate lives in
`src/thoughtforge/refinement/streaming_salvage.py`
(`STREAM_RULES` / `STRATEGY_PLACEMENTS` /
`STREAM_FIXTURES` as data, `StreamingSalvage` /
`run_streaming_salvage` / `salvage_stream`, the
checkers `streaming_salvage_problems` /
`streaming_salvage_live_problems`, format
`thoughtforge-streaming-salvage/1`). No docket
entry was assigned to TF-099 and none is added:
the gates it composes (TF-091, TF-092) stand as
filed, and the core wiring it deliberately does
not take is TF-164's.*

## The plain truth first

Before this slice, nothing in refinement could
act while text was still arriving. Every surface
takes complete text — `forge` a completed
candidate list, the TF-090/091/092 gates a
complete string — and TF-026's streamed turn
matches: candidate chunks go out raw, salvage and
enforcement run only on the completed candidate,
and their answer arrives as the single `final`
event. What *could* honestly be rescued
mid-stream was unspecified.

Probing settled where the line is:

- **A loop is only provable at its threshold.**
  Through TF-091's real gate, one or two repeats
  of "The bell rings." judge clean; the third
  judges rescued. A streaming rescuer that
  delivered the second repeat at once could
  never cut the loop — delivered text cannot be
  un-delivered.
- **Partial-text judgements corrupt.** TF-090's
  format gate already "repairs" the first 12
  characters of a JSON answer — its truncation
  closure would close a value still being
  written. Its truncation repair, run on a
  mid-stream prefix, cuts a sentence the model
  had not finished. Format and truncation are
  properties of an ending.
- **Completed-sentence judgements do not
  change.** TF-092 judges drift per sentence;
  TF-091's period-1 run is a fact about completed
  sentences. Those two, at a sentence boundary,
  by deletion only, are the whole mid-stream set.

## The placement table

`STRATEGY_PLACEMENTS` files every refinement
strategy with its placement and its reason —
seventeen strategies in all. `mid_stream` holds
exactly two: period-1 sentence-loop cutting
(TF-091's share) and per-sentence drift dropping
(TF-092's share). `stream_end` holds format
enforcement and truncation repair (TF-090) —
judged when the stream closes, never before.
Everything else is `post_hoc`: draft scoring and
selection, fragment extraction, refine passes,
the fabrication guard (TF-088), the voice guard
(TF-096), loop cycles and word loops (TF-091's
other share), drift *majority* flagging
(TF-092's other share), citation and policy
enforcement (TF-049/TF-093), the length and
genericness gate, policy disposition (TF-094),
refusal honesty (TF-095), rescue provenance and
cost (TF-097/TF-098), and salvage metrics
(TF-089). The data checker pins the mid-stream
set to exactly those two, so a future slice that
moves a strategy must amend its filed reason,
not drift past it.

## The streaming rescuer

`StreamingSalvage` is fed chunk by chunk and is
indifferent to chunk boundaries — the fixtures
split mid-word and mid-sentence on purpose. A
sentence becomes judgeable only when complete:
its terminator seen with following whitespace,
or, at `finish()`, a buffer ending in a
terminator. Each completed sentence is judged
drift first (a dropped sentence never enters the
loop state), then for the loop holdback:

- **Loops.** The first occurrence is delivered.
  A second identical sentence (TF-091's
  normalisation) is *held*, undelivered. A third
  proves the loop: held and third and every
  further repeat are dropped, each recorded as a
  `sentence_loop` cut. Any different sentence —
  or the stream ending — releases the held
  sentence as what TF-091 says twice may be:
  emphasis. On the filed loop fixtures the
  delivered text is exactly what TF-091's
  post-hoc gate returns on the same whole text,
  proven live by the checker.
- **Drift.** Only with a caller-supplied expected
  language (TF-092's rule — none named, no
  judgement; an unsupported one raises
  `ValueError` by name). A completed sentence
  whose TF-060 hint is a known different language
  is dropped before delivery and recorded with
  its detected language; an unknown-hint sentence
  is neutral and kept. Nothing is translated. On
  the minority-drift fixture the delivered text
  equals TF-092's post-hoc rescue of the same
  text, proven live.

## The end of the stream

`finish()` releases a held repeat, judges a
final terminated sentence, and records any
unterminated tail verbatim as an `incomplete_tail`
cut — not delivered, never completed; TF-090's
truncation gate owns the ending's verdict. If no
complete sentence ever arrived, the result names
it (`no_complete_sentence`) instead of dressing
the tail as an answer. If every sentence is
dropped, the result is empty and its cuts say
why, cut by cut — never a silent empty, never a
minority fragment promoted to hide one.

## Mid-stream is not final

The rescued text is a *candidate*, not an
answer: every result carries
`post_hoc_still_required = True`. TF-093's
policy surface still owns the verdict over the
completed text, and TF-092's majority and
whole-answer flagging stays post-hoc even where
its per-sentence share ran mid-stream. The
rescuer is pull-based (TF-072): it releases a
sentence only from chunks already fed, and
`salvage_stream` pulls one chunk at a time.

## What did not change

TF-091's and TF-092's gates, `FragmentSalvage`,
the core, and TF-026's event contract are
untouched — the checker verifies no other
refinement module grew a streaming surface.
Wiring this rescuer into `think_stream` would
change TF-026's filed chunk contract (chunks
today are the source's own, `native`-marked);
that is a core slice's call, and TF-164 owns
streaming across bindings.

## Proof grade

FIXTURE — twelve chunked synthetic streams run
through the real rescuer on this VM, with the
loop legs proven equal to TF-091's gate and the
drift leg equal to TF-092's rescue on the same
texts. A live model's streamed output is NOT
PROVEN — no model exists in the tree — and
never claimed. TF-100's chaos corpus and
TF-101's stack score own the real-shaped
failures.
