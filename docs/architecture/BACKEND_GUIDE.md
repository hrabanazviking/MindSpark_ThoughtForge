# TF-085 — Backend Docs + the New-Backend Guide

**Backend docs as one estate, and the TF-065 contract
as a recipe: fourteen ordered steps from subclass to
filed doc, proven by a worked example backend that
TF-065's own checker judges clean.**
`src/thoughtforge/inference/backend_guide.py`
(`BACKEND_DOCS` / `RECIPE_STEPS` / `GUIDE_RULES` as
data, `GuideExampleBackend`, format
`thoughtforge-backend-guide/1`). Tests:
`tests/test_tf085_backend_guide.py`.

## Measured start

The backend docs were an estate, not a guide. Six
core docs and three supporting docs stood in
`docs/architecture/` (catalogue below), each good
on its own slice — but no file indexed them as one
estate, and no file anywhere gave the steps for
writing a new backend. Greps for a recipe found only
TF-153's admission ownership repeated in the
TF-074/TF-076 docs: the gate was named, the road to
it was not written. And the public entry docs are
silent on backends: `docs/index.md` and `docs/api.md`
contain no mention of "backend" at all, so a reader
arriving through the front door never learns a
backend contract exists. That gap is D-GUIDE-1,
added to the TF-016 docket and FIXED by this slice
at guide grade (this doc is the index; reconciling
`docs/index.md` / `docs/api.md` is TF-182's wider
work, and wiring any README is forbidden by
standing law).

Meanwhile the contract itself was already
recipe-shaped, unassembled: TF-065's
`CONTRACT_DUTIES` names every duty, `UnifiedBackend`
supplies the `count_tokens` / `capabilities` /
`generate_stream` defaults, and
`backend_contract_problems()` judges any candidate
live. This slice assembles those pieces; it changes
no contract rule.

## The catalogue

Filed as `BACKEND_DOCS`, judged live by
`backend_guide_problems()` (each doc exists and
names its slice in its head; each module beside it
exists; a core backend doc on disk but absent from
the catalogue fails the checker):

| Doc | Slice | Role | Covers |
|---|---|---|---|
| docs/architecture/BACKEND_CONTRACT.md | TF-065 | core | the contract itself: duties, capability vocabularies, dependency rulings, factory keys |
| docs/architecture/BACKEND_HEALTH_AND_FALLBACK.md | TF-073 | core | health semantics + FallbackBackend |
| docs/architecture/AESIR_BACKEND.md | TF-074 | core | a worked binding of an external server, conformed directly |
| docs/architecture/AESIR_TURBOQUANT_INTERPLAY.md | TF-075 | core | quant truth for a served backend |
| docs/architecture/OPENAI_COMPATIBLE_BACKEND.md | TF-076 | core | a worked binding of a common HTTP shape |
| docs/architecture/LOCAL_FIRST_ROUTING.md | TF-077 | core | placement, local-first order, remote opt-in |
| docs/architecture/TOKENIZER_DISCIPLINE.md | TF-071 | supporting | the counting tolerance a new backend must land inside |
| docs/architecture/STREAMING_BACKPRESSURE.md | TF-072 | supporting | streaming duties: backpressure, clean cancellation |
| docs/architecture/JOINED_TRACES.md | TF-084 | supporting | what the core records about a backend's calls per turn |

## The recipe

Filed as `RECIPE_STEPS`; the checker recomputes
that their duties cover every duty in TF-065's
`CONTRACT_DUTIES`, plus dependencies, conformance,
admission, and docs.

1. **subclass_and_name** — subclass `UnifiedBackend`;
   `backend_name()` returns `family:model`, and that
   string rides every response's `backend_used`.
2. **generate_returns_data_never_raises** —
   `generate()` returns a `GenerationResponse` for
   every outcome; failure is `finish_reason "error"`
   plus the error string.
3. **finish_vocabulary_only** — `finish_reason` is
   stop / length / error only; dialect reasons
   (content_filter, tool_calls) are mapped or named
   error data, never passed through.
4. **health_check_bool_never_raises** — returns a
   bool, never raises; server health and per-model
   listing are separate surfaces.
5. **capabilities_honest_record** — the truth in the
   contract vocabularies, read live; placement never
   `unknown` for a backend offered for admission.
6. **count_tokens_declared_basis** — empty counts 0;
   the model's tokeniser when actually loaded
   (`model`), the contract estimate otherwise
   (`estimate`); land inside TF-071's tolerance.
7. **streaming_decision_stated** — keep the base
   default (simulated; the TF-084 stash carries the
   error) or override and yield in order (native),
   closing what the stream opened (TF-072). The
   record must match the choice.
8. **seed_forward_or_declare** — forward
   `request.seed` when carried and declare
   `honoured`, or declare `not_supported`; silently
   dropping a seed is a breach.
9. **placement_and_local_first** — placement honest,
   resolved per instance for local_or_remote; a
   remote backend is opt-in in routing and its cost
   declared, never guessed.
10. **dependencies_declared** — a new top-level
    `requests` consumer amends `REQUESTS_CONSUMERS`
    on purpose; heavier imports are optional extras,
    lazy or guarded.
11. **conformance_before_admission** —
    `backend_contract_problems([your_backend]) == []`
    against the real class is the gate.
12. **tests_in_the_estate_shape** — pin the success
    payload, error-as-data, counting, health, seed
    presence/absence, and the capability record
    against the class.
13. **admission_is_tf153s** — do not add the backend
    to `SHIPPED_BACKENDS` or the factory yourself;
    conform directly first (the TF-074/TF-076
    pattern). TF-153 owns admission.
14. **doc_filed_with_proof_grade** — file the
    backend's own architecture doc with a measured
    start and a proof grade stating what is
    NOT PROVEN until it has actually been faced.

## The worked example

`GuideExampleBackend` (`guide_example:lantern-1`)
is the recipe cooked: the smallest honest backend —
a deterministic synthetic responder, no model, no
network. It conforms under
`backend_contract_problems`, serves a plain turn,
streams simulated through the base default (its
empty-prompt streamed error survives in the TF-084
stash with the exact error string), and states its
failure as data on an empty prompt. It joins
neither `SHIPPED_BACKENDS` nor the factory.

## Proof grade

**Fixture.** The catalogue and recipe are judged
live against the tree; the example backend is judged
by the contract's own checker and exercised plain
and streamed on this VM. A real author following
this guide to a live third-party backend is
NOT PROVEN — no such backend has been built from it
yet; TF-086's gate and TF-153's admission own that
road, and TF-086's truth table must carry the line.
