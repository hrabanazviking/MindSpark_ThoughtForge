# TF-065 — Backend Contract v1

**The contract every inference backend owes ThoughtForge —
generate, stream, tokenise, capability truth — with the four
shipped backends (ollama / lmstudio / huggingface /
turboquant) conformed and tested against it.**
`src/thoughtforge/inference/backend_contract.py` (the
contract as data + live checkers), conformance wiring in
`inference/unified_backend.py` (the base duties) and the
four backend modules; dependency rulings in
`pyproject.toml`. Tests:
`tests/test_tf065_backend_contract.py`.

## Measured start (probed this slice)

The four backends shared only the thin `UnifiedBackend`
ABC. Probing the classes directly:

- no `capabilities()` and no `count_tokens()` existed on
  the base or on any backend — nothing anywhere stated
  what a backend can or cannot do;
- `GenerationRequest.seed` (TF-036) was forwarded by
  exactly one backend (TurboQuant, through
  `GenerationParams`); the ollama / lmstudio / hf
  `generate()` sources contained no seed at all, so a
  deterministic-mode turn ran silently unseeded on three
  of four backends;
- no backend overrode `generate_stream` — every backend
  streamed *simulated* under TF-026's labelling, including
  TurboQuant, whose engine owns a native streaming
  generator the wrapper never exposed;
- no backend could count tokens; HF reported
  `tokens_generated` as `len(text.split())`, a word count
  wearing a token count's name, declared nowhere;
- `requests` was a top-level import in four inference
  modules and declared nowhere in `pyproject.toml`
  (D-DEP-1); transformers / huggingface_hub /
  onnxruntime / optimum / psutil were declared nowhere
  at all (D-DEP-5), though every import site of theirs
  is lazy or guarded.

## The contract (CONTRACT_DUTIES, as data)

| Duty | Required | The rule |
|---|---|---|
| generate | yes | Returns a `GenerationResponse`; failures are data (`finish_reason "error"` + the error string), never a raise |
| generate_stream | yes | Yields chunks in order; the base default yields the completed text as one chunk (nothing on error) and is *simulated* — an override is *native* |
| count_tokens | yes | The model's tokeniser when the backend has one loaded (`model`), otherwise the contract estimate, declared `estimate`; empty text counts 0 |
| capabilities | yes | The backend's truth in the contract vocabularies, read live; a claim the code contradicts is a breach the checkers catch |
| seed | no | A backend that *can* seed MUST forward `request.seed` when set (TF-036's duty, assigned here); one that cannot declares `not_supported` — silently dropping a carried seed is a breach |
| health_check | yes | Returns a bool, never raises |
| backend_name | yes | `family:model`; carried as `backend_used` on every response |
| finish_reason | yes | One of `stop` / `length` / `error` |

The capability record (`BackendCapabilities`, format
`thoughtforge-backend-contract/1`) states: `streaming`
∈ native / simulated / none, `token_counting` ∈ model /
estimate / none, `seed` ∈ honoured / not_supported,
`placement` ∈ local / remote / local_or_remote / unknown.
A shipped backend may not leave its placement `unknown`.

**Claims are checked against code, not trusted:**
`backend_contract_problems()` re-derives the contract
live — a backend declaring `native` must actually
override `generate_stream` (the same test the cognition
core uses to label chunks); one declaring `honoured`
seed must mention seed forwarding in its class source;
every backend's `count_tokens` must map "" → 0 and be
positive and non-decreasing on prose; every
`health_check` must return a bool without raising.

## The base duties

`UnifiedBackend` gains the two defaults a backend gets
for declaring nothing:

- `count_tokens` — the contract estimate
  (`estimate_tokens`: ~4 chars/token, the heuristic
  `TurboQuantEngine.estimate_tokens` falls back to,
  named an estimate wherever used);
- `capabilities` — the conservative record
  (`default_capabilities`): simulated streaming unless
  the class overrides `generate_stream`, estimate
  counting, seed `not_supported`, placement `unknown`
  (which a *shipped* backend must then replace with its
  own record — all four do).

## Conformance, per backend

| Backend | Streaming | Counting | Seed | Placement |
|---|---|---|---|---|
| ollama | simulated | estimate | **honoured — wired this slice** (`options.seed`) | local_or_remote |
| lmstudio | simulated | estimate | **honoured — wired this slice** (the OpenAI-shaped `seed` field) | local_or_remote |
| huggingface | simulated | estimate | **honoured — wired this slice** (client kwarg + raw-request parameters) | remote |
| turboquant | **native — wired this slice** (the wrapper now delegates to `TurboQuantEngine.generate_stream`) | model when loaded, estimate until then (read live) | honoured (already, via `GenerationParams`) | local |

Seed forwarding is measured in the actual payloads in
tests, and its absence when the request carries no seed
is pinned too — pre-slice payloads are byte-identical
for unseeded requests. TurboQuant's stream: a load
failure yields nothing (the standing streamed-error
behaviour; the error *channel* for streamed turns is
TF-084's road, D-HON-1), a mid-stream engine failure
raises exactly as the engine documents. HF's record
carries the note that its `tokens_generated` is a
word-split approximation — the count is now declared
for what it is.

## Dependencies (D-DEP-1 FIXED, D-DEP-5 FIXED)

The contract slice's rulings, held in
`DEPENDENCY_RULINGS` as data and judged by
`dependency_problems()` against `pyproject.toml` + an
AST scan of `src/`:

- **`requests` → core.** It is imported unconditionally
  at module top level, so a minimal install must carry
  it: declared `requests>=2.31` in `[project]
  dependencies`. The consumer set (hf, lmstudio,
  model_browser, ollama) is pinned — a fifth top-level
  consumer fails the checker until the contract is
  amended on purpose.
- **The other five → optional extras.** huggingface_hub
  + transformers under `hub`; onnxruntime + optimum +
  transformers under `onnx`; psutil under `system`.
  Every import site stays lazy or guarded — the checker
  fails if any of the five is ever imported at module
  top level.

## Config keys (D-CFG-3, backend share narrowed)

`BACKEND_CONFIG_KEYS` pins the factory's live key set —
the 11 `cfg.get(...)` names in
`load_backend_from_config`, `openai_model` included —
judged by AST against the factory's own source
(`factory_problems()`). The factory can neither lose a
live key nor grow a read of a dead key (`profile` /
`chat_*` / `knowledge_db_built`) unnoticed. Striking
the dead keys from the shipped file stays TF-153's.

DETERMINISM_MODE.md's TF-036 remainder is updated in
place: honouring the seed is now a landed duty, not a
road. TF-016 totals updated: 58 entries — 38 OPEN /
1 OPEN-ENVIRONMENT / 17 FIXED / 2 CLOSED-BY-VERDICT.

## Proof grade and limits

FIXTURE proof, on this VM: payloads captured from
mocked HTTP legs, a fake InferenceClient, and a fake
TurboQuant engine; the live checkers re-derive the
contract from the real classes, `pyproject.toml`, and
the factory source. What is **NOT PROVEN** here:

- a *live* server's behaviour under the forwarded seed
  (a server may accept and ignore `seed` — reproducibility
  of a live model is TF-067 / TF-126's measurement);
- native streaming from a *loaded* TurboQuant engine
  (delegation is pinned against the fake engine; no
  GGUF model is loaded on this machine);
- token-count *agreement* between backends — TF-071
  owns the tolerance discipline; this slice only makes
  each backend's counting mode declared and reachable;
- the OpenAI-compatible *family* beyond LM Studio's
  shape — TF-076 binds arbitrary servers under this
  same contract; Aesir joins at TF-074.
