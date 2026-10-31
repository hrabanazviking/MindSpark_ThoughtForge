# TF-073 — Backend Health & Fallback

**The plain truth first: before this slice there was no
fallback at all.** The cognition core holds exactly one
backend; when it answered with an error, the turn degraded
to knowledge-only (TF-031 records that failure honestly —
but no second backend was ever tried, because nothing in
the tree could hold a second one). Health checks existed
on all four backends but meant four different unstated
things, and one of them lied: HuggingFace's counted a
nonexistent model (404) and rejected credentials (401) as
"healthy". And the circuit breaker built for exactly this
job protected nothing — `utils/retry.py` was test-only
(D-ORPHAN-2). This slice files the policy as data
(`src/thoughtforge/inference/backend_fallback.py`, format
`thoughtforge-backend-fallback/1`), implements it as a
backend that wraps an ordered chain, fixes the health lie,
and wires the breaker into production.

## The policy, stated

`FallbackBackend` is itself a `UnifiedBackend`, so any
consumer that can hold one backend — the core included,
unchanged — can hold the chain. The rules:

1. **Order is the policy.** The chain is tried strictly in
   the caller's order. Nothing reorders it by capability,
   cost, or history. The first backend that serves, serves.
2. **Health gate before generate.** A backend is asked to
   generate only after its `health_check()` returns True.
   A False (or defensively, a raising) health check skips
   it without a generate call.
3. **Error is failure.** A response with finish_reason
   `error` or an error string (TF-065's failure-as-data)
   is passed over — its text, if any, is never the answer.
4. **Empty is failure.** A response with no usable text
   answered nothing (TF-031's ruling) and is passed over.
5. **An exception is failure data.** A backend whose
   `generate()` raises — a contract breach — is caught and
   passed over; the wrapper's own `generate()` never raises.
6. **The breaker skips the known-dead.** Each member
   carries a `utils/retry.CircuitBreaker` fed by its
   generate outcomes. At the threshold (default 3) the
   breaker opens and the member is skipped *without* a
   health or generate call, until the recovery timeout
   (default 60 s) lets the breaker's half-open probe
   through.
7. **Announced, never silent.** Every skip and failure,
   and which backend served, lands in the wrapper's
   `last_report` and the log. A serve by a fallback is
   marked `degraded` — a successful answer, honestly
   labelled as not the primary's.
8. **All-failed is stated whole.** If every member fails,
   the returned error names every backend and its reason
   (`all backends failed: a: …; b: …`), the text is empty,
   and `backend_used` is the wrapper — never one backend's
   error dressed as the whole story, never invented text.
9. **Streaming falls back only before the first chunk.**
   Same gates, but once a chunk is delivered the stream
   belongs to that backend (TF-072: a later failure
   propagates; the caller decides what a partial means).
   A stream that yields nothing, or raises before its
   first chunk, is a failure and the next backend is tried.
10. **Chain health is any-member.** The wrapper's
    `health_check()` is True exactly when at least one
    non-circuit-open member reports healthy, in order,
    short-circuiting. It never raises.

## What a health check actually proves (per backend)

Filed as `HEALTH_SEMANTICS`, judged live against each
class's source:

| Backend | Check | Proves | Does not prove |
|---|---|---|---|
| huggingface | per-model info endpoint | the named model exists and the host answers **200** for it | that inference will succeed (a loading model is the backend's own 503-retry leg) |
| ollama | server `/api/tags` | the server answers 200 | that the configured model is pulled |
| lmstudio | server `/v1/models` | the server answers 200 | that the configured model is loaded |
| turboquant | file exists + `llama_cpp` importable | the file is present and the library imports | that the model loads (loading is lazy) |

The two server-shaped gaps — health that cannot see
whether the *configured model* is there — are filed as
**D-HEALTH-2**, owned by TF-077, whose routing table is
the first consumer that needs per-model health.

## Defects found and fixed in scope

- **D-HEALTH-1 (FIXED)** — `HuggingFaceBackend.health_check`
  returned `status_code < 500`: measured live, 401 and 404
  both returned True. The endpoint is per-model, so
  "healthy" claimed a model that does not exist could
  serve. It now returns True exactly on 200.
- **D-ORPHAN-2 (FIXED)** — `utils/retry.py` protected no
  production call. The `CircuitBreaker` now does: one per
  chain member inside `FallbackBackend`, fed by every
  generate outcome. The `retry` decorator itself stays
  unused in production, and the reason is stated rather
  than hidden: it retries *exceptions*, and under the
  TF-065 contract backend failure is data, not an
  exception — wiring the decorator here would have meant
  converting data back into raises for no benefit.

## What this slice does not claim

- **Which chain a deployment should run is TF-077's**
  (local-first routing: privacy, cost, capability). This
  slice ships the mechanism and the policy; no shipped
  config yet names a chain, and the factory still loads
  the single configured backend (TF-153 owns the service
  shape that will compose them).
- **A live server's death is NOT PROVEN**: no backend
  servers run on this VM, so every degradation here is
  proven over stub members and mocked HTTP (proof grade
  FIXTURE) — the policy, the gates, the breaker and the
  announcements are the shipped behaviour a live chain
  will execute.
- The core's own degradation (knowledge-only with the
  failure stated, TF-031) is unchanged and still the last
  resort when a whole chain — or a single backend — fails.
