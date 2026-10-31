# TF-081 — Batch & Concurrency per Profile

*Batch & concurrency per profile, measured honestly.
The estate lives in
`src/thoughtforge/inference/batch_concurrency.py`
(`CONCURRENCY_SHAPES` / `BATCH_RULES` as data,
`measure_batch_concurrency()` /
`median_batch_concurrency()`, format
`thoughtforge-batch-concurrency/1`).*

## The plain truth first

There is no batched generation in ThoughtForge.
Every backend answers one request at a time; the
one multi-generation the engine offers — drafts —
runs them one after another in a plain loop; and
the one way two generations ever overlap is the
shape TF-035 already proved: separate conversations
on separate cores, in parallel threads. This slice
measured all of that per profile instead of leaving
it as folklore — TF-035 explicitly deferred the
timing to here.

And the limit, stated first: no real model exists
on this VM (no GGUF in-tree, `llama_cpp` absent),
so what is measured is fixture generations with a
scripted 50 ms delay through real cores and the
real engine. A live model's concurrent throughput
is **NOT PROVEN** for every profile — TF-175's
laptop line and TF-082's Pi own that road, and
TF-086's truth table must carry the line.

## Measured start

Probed live this tick:

- **No batch entry exists on the contract.**
  `UnifiedBackend`'s methods are generate /
  generate_stream / health_check / backend_name /
  count_tokens / capabilities — one request in, one
  response out. The inference tree's one true batch
  is `onnx_export`'s tokenisation loop
  (`batch_size` chunks over texts) — an
  embedding-export step, not generation.
- **Drafts are a sequential loop in an unguarded
  engine.** `generate_drafts` is
  `for i in range(count): self.generate(...)`;
  `turboquant.py` imports no threading and holds no
  lock, so concurrent calls on ONE engine are an
  unproven shape, not a supported one.
- **No profile declares a concurrency knob.** No
  shipped profile JSON carries a concurrency,
  batch, or thread key (checker-pinned over all
  six); `n_threads` defaults to auto everywhere.
  The only per-profile number shaping
  multi-generation is `draft_count`: pi_zero 1,
  phone_low 2, pi_5 / desktop_cpu / desktop_gpu 3,
  server_gpu 5. D-BATCH-2 (TF-016) files this for
  TF-153/TF-086.
- **The server backends differ.** Aesir serialises
  server-side behind one resident model (TF-074).
  The HTTP backends are stateless per call on the
  client (own `requests` call per generate, no
  session, no lock) — concurrent client calls are
  possible by construction; what a server does with
  them is NOT PROVEN (no live servers here).

## The panel

The same batch on every profile — 4 distinct
synthetic prompts, each generation carrying a
scripted 50 ms delay in an instrumented backend
that records the most generations ever
simultaneously in flight — run four ways:
sequentially on one core; one core per prompt in
parallel threads; all threads on one core; and
drafts through the real `TurboQuantEngine` (stub
llama, same delay + recorder) at the profile's own
draft count. Rows are medians of 3 panels, measured
live on this VM this tick:

| profile | drafts | sequential | parallel | same-core | drafts wall | speedup | same/seq |
|---|---|---|---|---|---|---|---|
| desktop_cpu | 3 | 390.031 | 110.311 | 381.038 | 278.434 | 3.536× | 0.977 |
| desktop_gpu | 3 | 387.886 | 163.314 | 407.450 | 264.458 | 2.375× | 1.050 |
| phone_low | 2 | 354.313 | 81.367 | 379.604 | 182.156 | 4.355× | 1.071 |
| pi_5 | 3 | 392.487 | 152.691 | 382.569 | 277.758 | 2.570× | 0.975 |
| pi_zero | 1 | 377.347 | 70.487 | 395.110 | 65.773 | 5.353× | 1.047 |
| server_gpu | 5 | 389.931 | 84.234 | 395.236 | 458.807 | 4.629× | 1.014 |

*(Walls in ms. Witnesses, every profile: max in
flight parallel = 4 of 4; same-core = 1;
drafts = 1.)*

## What the numbers prove — and what they don't

- **Parallel cores really overlap.** Max in flight
  4 of 4 on every profile, with speedups of
  2.375–5.353× over the sequential wall. The shape
  TF-035 proved for isolation is also the tree's
  only speed shape.
- **One core serialises, measured.** Same-core max
  in flight is 1 on every profile and its wall ratio
  to plain sequential is 0.975–1.071 — the turn lock
  doing exactly what TF-035 said, now at batch
  scale: safe, correct, never parallel.
- **Drafts cost their count, serially.** Max in
  flight 1 through the real engine on every
  profile: a draft_count of 5 (server_gpu) costs
  ~5 scripted delays plus engine overhead
  (458.807 ms), a count of 1 (pi_zero) costs one
  (65.773 ms). Draft count is a wall-time budget,
  not a parallelism.
- **The walls themselves are fixture walls.** Core
  turn overhead (~50 ms a turn beyond the scripted
  delay on this shared VM) dominates them; they
  vary under contention and are filed for shape and
  ordering, never as throughput predictions.
  Per-profile differences in the rows are the
  profiles' resolved configurations riding along —
  no profile declares, or gets, a different
  parallelism.

## The shapes, filed

Seven surfaces, each with its shape and evidence in
`CONCURRENCY_SHAPES`: the contract
(single_request_only), engine drafts
(sequential_loop), one core's turns
(serialised_by_turn_lock), parallel cores
(parallel_isolated), Aesir (serialised_server_side),
the HTTP backends (stateless_per_call_client), and
ONNX tokenisation (batched_not_generation). The
rules in `BATCH_RULES` draw the lines consumers may
rely on: parallel only across cores; one engine's
concurrent calls unproven; Aesir batches are queues
by the server's design; HTTP server concurrency
unproven; and the whole panel comparable by
construction — same prompts, same delay, medians of
3, on every profile.

D-BATCH-1 (TF-016) closes **FIXED at fixture
grade** by this slice: the measurement and the
shape table now stand where none did. D-BATCH-2
stands OPEN: profiles still declare no concurrency
parameter — TF-153's service shape is where a
deployment's concurrency gets decided, and TF-086's
truth table carries both lines.
