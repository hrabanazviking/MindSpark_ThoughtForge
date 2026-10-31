# TF-076 — The OpenAI-Compatible Backend

*Any server speaking the common shape binds under the
TF-065 contract. The estate lives in
`src/thoughtforge/inference/openai_compatible.py`
(`OpenAICompatibleBackend` + `OPENAI_SHAPE` /
`OPENAI_RULES` / `KNOWN_SERVERS` as data, the shared
family parser `chat_completion_outcome`, format
`thoughtforge-openai-compatible-backend/1`).*

## Measured start

The common shape was already reachable in fact, but only
as a side effect: `LMStudioBackend`'s docstring claimed
it works with "LM Studio, vLLM, text-generation-webui,
Oobabooga, LocalAI, and any server that implements the
/v1/chat/completions endpoint", and the config factory
maps the `openai_compatible` backend type onto that
class. There was no generic binding of its own — no
family identity (`backend_name` said `lmstudio:...`
whatever the server was), no filed shape, no rules for
the dialects. Mocked probes of the LM Studio leg this
tick found three contract defects (all fixed in scope,
in both legs, through one shared parser):

- **D-OAI-1** — response parsing breached the contract:
  `finish_reason` values `content_filter` / `tool_calls`
  passed straight through, outside TF-065's vocabulary
  (stop / length / error); a `content: null` message
  landed as `text=None`; an empty `choices` list
  surfaced only as `'list index out of range'`.
- **D-OAI-2** — a conventional base URL ending in `/v1`
  (the shape's own convention, e.g. the hosted API's
  base) produced `/v1/v1/chat/completions`.
- **D-OAI-3** — a response without `usage` reported
  `tokens_generated` 0 over real text.

Server error bodies were also discarded: a 401 carrying
`{"error": {"message": "..."}}` surfaced as the bare
string `HTTP 401`.

## The shape

`POST /v1/chat/completions` (non-streamed in this
binding) and `GET /v1/models`. Sent: `messages`,
`temperature`, `max_tokens`, `stream: false`, `model`
when configured, `stop` when carried (the shape serves
stop — the opposite ruling from Aesir's, each from its
own server's truth), `seed` when carried. Auth: an
Authorization header only when a key is configured — no
placeholder credential is ever sent to a validating
server. Not bound: the shape's SSE streaming (claimed
`simulated` through the contract's base default; the
dialects vary per server and none is assumed), any
tokenise route (the shape guarantees none; counting is
the declared estimate), and `/v1/embeddings`
(ThoughtForge binds embeddings in its knowledge layer,
TF-053).

## The rules

Filed as `OPENAI_RULES`, each measured in
`tests/test_tf076_openai_compatible.py`:

- `base_url_v1_normalised` — routes are built from the
  root; bare, `/v1`, and `/v1/` bases produce one
  identical route URL.
- `finish_vocabulary_enforced` — stop/length pass; any
  other finish reason is error data naming it. This
  backend sends no tools, so such a finish is never
  dressed as a completion.
- `content_is_text_or_error` — content must be a
  string; null/non-string is named error data.
  `GenerationResponse.text` is never `None`.
- `no_choices_is_named_error`.
- `tokens_usage_or_declared_estimate` — the server's
  usage count when reported, the contract estimate
  otherwise, never a silent 0.
- `errors_are_data_with_the_servers_reason` — status +
  the server's own message, both error shapes parsed;
  `generate()` never raises.
- `auth_only_when_configured`.
- `health_is_server_level_model_listed_separately` —
  health is `/v1/models` at exactly 200 (the D-HEALTH-1
  standard); `model_listed()` answers the configured
  model's listing separately (True / False / None when
  unasked or unanswerable). D-HEALTH-2's per-model
  routing road stays TF-077's; this surface is its data.
- `stop_is_served_by_the_shape`.
- `live_proof_not_claimed` — see below.

## The LM Studio preset

`LMStudioBackend` stays exactly where it was — its
defaults, name, capabilities, and factory seat
unchanged — and is now the family's LM Studio preset:
it consumes the shared parser and normaliser (the
checker pins the identity, D-DUP-1's rule), so D-OAI-1,
D-OAI-2, and D-OAI-3 are fixed in the shipped leg too,
not just the new one. Payload parity between the two
legs on a shared request is pinned in the tests.

## Placement

Like TF-073's FallbackBackend and TF-074's
AesirBackend, the new backend is conformed against the
TF-065 contract directly (`backend_contract_problems`
over the real class is clean) but does NOT join
`SHIPPED_BACKENDS` or the factory — TF-153 owns
admitting new backends there, and the factory's
`openai_compatible` seat keeps returning the LM Studio
preset until then. One deliberate contract amendment:
`REQUESTS_CONSUMERS` grew five → six (announced in
`backend_contract.py`, the TF-065 length pin updated).

## Proof grade

**Fixture + live-HTTP-loopback.** Mocked transports in
the shape's own payload forms, the contract's own
checker, payload parity with the LM Studio leg, one
real core turn over a fake transport, and a real HTTP
round trip against a stdlib fixture server on loopback
(the wire, headers, and parsing exercised end to end).
A real third-party server — LM Studio, vLLM, LocalAI,
llama.cpp server, the hosted OpenAI API — on the other
end is **NOT PROVEN**: filed as **D-OAI-4**, owned by
TF-086's gate, whose truth table must carry the line
until a real server faces this backend. The named
families in `KNOWN_SERVERS` are shape claims, each
filed `shape_bound_not_live_proven`.
