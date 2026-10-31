# TF-072 — Streaming Backpressure & Clean Cancellation

**The plain truth first: the streaming chain in this tree
was already pull-based end to end — no queue, thread, or
buffer anywhere — so a slow consumer could never make a
producer run ahead or corrupt a generation; what was
missing was proof, and one real cleanup at the very
bottom of the chain.** `TurboQuantEngine.generate_stream`
held the llama `stream=True` iterator in a local variable
and abandoned it un-closed on two of its three exits: a
consumer closing the stream early, and a mid-stream
error. This slice files the discipline as data
(`src/thoughtforge/inference/streaming_discipline.py`,
format `thoughtforge-streaming-discipline/1`), measures
every layer, fixes that one defect (D-STREAM-1), and
states each backend's true cancellation point instead of
letting "cancellation clean on every backend" imply
powers the tree does not have.

## The layers, as measured

```
consumer → core think_stream (TF-026/TF-027 events)
         → backend generate_stream
           → (native) TurboQuantBackend — yield from
             → TurboQuantEngine.generate_stream
               → llama stream=True iterator
           → (simulated) ollama / lmstudio / huggingface —
             the TF-065 default: one chunk = one completed
             generate() response
```

Every arrow is a Python generator. A generator advances
only when its consumer pulls, so backpressure is not a
mechanism that had to be built — it is the shape of the
chain, and this slice's job was to *pin* it: at every
pause, a counting producer had produced exactly as many
items as had been pulled, at the engine layer and through
the backend's `yield from` alike. If a future layer ever
introduces a queue, a thread, or a prefetch buffer, the
produced == pulled pins fail first.

## What a slow consumer can and cannot do

- **Native (turboquant):** deltas are produced one pull
  at a time. Pausing between pulls — for a second or a
  day — changes nothing: the joined deltas are
  byte-identical to a quickly consumed stream and to the
  raw completion (unstripped, in order; the engine's
  TF-026 contract).
- **Simulated (ollama / lmstudio / huggingface):** the
  single chunk is a completed `generate()` response —
  the text exists whole before the first pull returns
  it, so there is nothing in flight for slowness to
  corrupt. The core re-chunks it for delivery (TF-026)
  under the consumer's own pace.

## Cancellation, layer by layer

- **Before the first pull, cancellation is free** —
  and this is pinned, not assumed. Every stream surface
  is a generator *function*, so creating a stream starts
  no work: closing an unpulled default stream runs
  `generate()` zero times, and closing an unpulled
  TurboQuantBackend stream loads no engine.
- **Mid-stream, a close closes every layer.** The
  backend's `yield from` forwards `close()` into the
  engine generator (measured: the stub engine's
  `finally` runs). The engine now forwards it to the
  model's own iterator — D-STREAM-1's fix, below.
- **What was already yielded stays yielded.** No layer
  replays, patches, or completes a cancelled stream's
  text. What a partial *means* is the caller's call —
  the cognition core's standing rule (TF-027) discards
  the partial candidate whole and records the cut; a
  consumer closing `think_stream` mid-turn was probed
  this tick: the stub engine's generator was closed, no
  candidate landed, the turn recorded not-complete.
- **A cancelled stream poisons nothing.** The next
  stream from the same backend/engine instance produces
  its full text (measured at both layers).
- **The simulated backends' cancellation point is
  stated, not implied:** between chunks — in practice
  before their single chunk. Nothing in this tree aborts
  an in-flight HTTP `generate()`, and this slice does
  not claim one; a caller who needs the generation not
  to happen must cancel before the first pull, which
  the free-cancel guarantee makes reliable.

## D-STREAM-1 — the one defect

**Found:** `TurboQuantEngine.generate_stream` never
closed the llama stream iterator. On plain exhaustion
the iterator drained and the omission was invisible;
on a consumer's early close (GeneratorExit raised at
the `yield`) and on a mid-stream error (the iterator
left part-consumed), the model's streaming state was
simply abandoned to garbage collection. Probes with a
closable stub stream recorded `closed == False` on both
paths.

**Fixed:** the iteration now sits in a `try / finally`
that closes the stream on *every* exit — exhaustion,
error, and consumer close alike. One exit discipline,
not three. GeneratorExit is not an `Exception` and is
never converted to the RuntimeError the error path
raises; it propagates untouched after the `finally`
runs. A stream iterator without `close()` (a plain
iterator) needs nothing and gets nothing; a `close()`
that itself fails is logged at debug and never masks
the stream's own outcome.

## The rules (filed as data)

`STREAMING_RULES` in `streaming_discipline.py`:
`pull_based_no_run_ahead`, `slow_consumer_integrity`,
`cancel_before_first_pull_is_free`,
`cancel_closes_every_layer`, `cancel_leaves_no_partial`,
`reuse_after_cancel`,
`simulated_cancellation_point_is_stated` — each with the
measurement that judges it, re-derived live by
`streaming_live_problems()`; the per-backend shapes are
`BACKEND_STREAM_SHAPES`, judged against each class's
override status by `streaming_problems()`.

## Proof grade

FIXTURE — the real engine and backends on this VM over
closable stub llama streams and counting stub backends,
plus the core leg over a TF-026-shaped stub engine. A
loaded GGUF model's stream-close behaviour is NOT
PROVEN (no llama_cpp, no GGUF on this VM; the engine's
close is defensive — `getattr` + callable — so an
iterator without `close()` is still disciplined), and a
live HTTP backend's in-flight abort is NOT PROVEN and
not claimed — the simulated shape's stated cancellation
point is the whole claim.
