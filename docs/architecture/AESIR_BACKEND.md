# TF-074 — The Aesir Backend

**Project Aesir bound as a ThoughtForge backend under the
TF-065 contract — claiming only what Aesir's own capability
ledger proves, no more.**
`src/thoughtforge/inference/aesir_backend.py`
(`AesirBackend`, plus `AESIR_LEDGER` / `AESIR_RULES` as
data, format `thoughtforge-aesir-backend/1`). Tests:
`tests/test_tf074_aesir_backend.py`.

## Measured start (probed this slice)

- No Aesir backend existed anywhere in the tree — no
  module, no factory entry, no config key. Aesir was
  named only as a road (TF-065's doc, TF-071's note).
- No live Aesir server answers on this VM
  (`127.0.0.1:18434` refuses): all proof here is fixture
  grade, and a live serve is **NOT PROVEN** (D-AESIR-1,
  below — the same standing Rúnmál's P-009/F-02 recorded;
  the owed live smoke is a purplelongship leg).
- The core's requests carry prompt / temperature /
  max_tokens / seed only — `stop` is never set by the
  core, which matters because Aesir serves stop nowhere.

## The ledger respected (AESIR_LEDGER)

Sources are Aesir's own versioned statements, read
2026-10-31 in Aesir's repo checkout on this VM (HEAD
`1eed9706` — the pin Rúnmál's P-009 bound to):
`docs/API_SUPPORT_MATRIX_V1.md`,
`CAPABILITY_LEDGER.md` (AES-SRV-006/008),
`docs/NATIVE_SERVICE.md`; cross-checked against
Rúnmál's P-009 ledger and F-02 repair.

| Surface | Aesir's truth | This binding |
|---|---|---|
| `POST /v1/chat/completions` | both modes; strict subset {model, messages, stream, max_tokens, temperature, top_p, seed} | **bound** — the chat paradigm ThoughtForge's requests already have |
| `POST /v1/generate` | native only; the FULL sampler set (top_k/min_p/repeat_penalty) | not bound — a text paradigm; recorded alternative |
| `GET /v1/models` | the catalog (ids only); visibility ≠ runnable | `list_models()` — never used as readiness |
| `GET /health/ready` | native only; the resident model's readiness | native `health_check()` + `readiness_check()` |
| Embeddings | **no route in either mode** ("embeddings remain Ollama") | never bound to Aesir |
| Vision/multimodal | rejected on every route | non-string content refused client-side |
| Tools / structured output | bounded Gemma single-function + Boolean-only subsets only | neither is sent |
| One resident model | serialized; non-resident → 404 | configured model sent every request; 404 is error data, never retried elsewhere |
| Ports/auth | native 18434 + mandatory bearer; `--ollama` 11434, no bearer; loopback only | defaults match; placement declared `local` |

Admission bounds (checked client-side before sending):
128 messages, 64 KiB per text field, 131072-byte body,
roles system/user/assistant, final message from the user.
Context/generation admission failures are Aesir's 422.

## The binding rules (AESIR_RULES)

- **Served subset only.** The payload is exactly model
  (when set), messages, `stream`, `max_tokens`,
  `temperature`, and `seed` when the request carries one
  (the chat route honours it — TF-065's seed duty).
  `stream_options` is never sent: on Rúnmál's adapter it
  broke every request against the real server (D1).
- **Stop is refused, never dropped.** Aesir serves
  `stop` on no route. A stop-carrying request returns an
  error response naming the refusal, before any HTTP
  call — the TF-065 seed lesson applied to stop.
- **Health is resident readiness** (native): True
  exactly on `/health/ready` 200 with `ready` true.
  In `--ollama` mode the health routes 404, so health
  is `/v1/models` reachability, labelled the weaker
  claim. A catalog listing is never readiness.
- **Incomplete stream is failure.** The SSE leg is a
  native `generate_stream` override: deltas in order,
  and the stream must end with a finish reason or
  `[DONE]` — EOF without either raises (Aesir's own
  S18/S19 instruction; Rúnmál's D6 repair), as does an
  SSE error event.
- **Errors are data with status.** Native error bodies
  (`{"error":{"code","message"}}`) are parsed; the
  response names the HTTP status and the server's
  message, 422 as context/generation admission, 404 as
  model-not-resident. `generate()` never raises.
- **Token counting stays estimate.** Aesir v1 serves
  no tokenise route, so `count_tokens` is the contract
  estimate (it agrees with every other backend on
  TF-071's corpus, trivially). `tokens_generated` is
  Aesir's own `usage.completion_tokens` when the
  response reports it, the declared estimate otherwise.
  The SSE stream carries no usage at all.

## Estate placement + the one contract amendment

Like TF-073's `FallbackBackend`, `AesirBackend` is a
`UnifiedBackend` conformed and tested against the
TF-065 contract directly — judged clean by
`backend_contract_problems(backends=[...])` itself —
but it does **not** join `SHIPPED_BACKENDS` or the
config factory: TF-153 owns the service/factory shape
that admits new backends, and TF-065's shipped set
stays the four it pinned.

One deliberate amendment in scope: this module imports
`requests` at top level like its siblings, so
`REQUESTS_CONSUMERS` in `backend_contract.py` grew
from four to five (the D-DEP-1 set — a consumer must
be announced, and this one is), its ruling text records
the growth, and the TF-065 test pin was updated to
match, exactly as TF-071 updated a TF-065 pin in scope.

## Docket

- **D-AESIR-1 filed OPEN, owned by TF-086** — no live
  Aesir serve has ever been on the other end of this
  backend; the binding is fixture-proven only. The
  gate's truth table must carry the Aesir backend as
  NOT PROVEN live until the owed smoke (one real Aesir
  serve on the laptop line + one ThoughtForge turn
  through this backend) has run. TF-016 totals:
  70 entries — 41 OPEN / 1 OPEN-ENVIRONMENT / 26 FIXED /
  2 CLOSED-BY-VERDICT (primary-owner Part IV 19 → 20).

## Proof grade and limits

FIXTURE proof, on this VM: mocked HTTP legs in Aesir's
own v1 shapes (chat payloads, SSE streams, native error
bodies, readiness payloads), the TF-065 contract checker
run over the real class, a real cognition-core turn over
the fake transport. **NOT PROVEN**: a live Aesir serve
(D-AESIR-1); Aesir + TurboQuant interplay (TF-075);
per-model health beyond resident readiness (the
resident model IS the health signal here — D-HEALTH-2's
server-shaped gap does not apply to a one-resident
engine, and no broader claim is made).
