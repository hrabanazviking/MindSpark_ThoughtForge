# Rúnmál Binding — Rúnmál's Calls Through ThoughtForge (TF-156)

GENERATED from the filed data in `src/thoughtforge/runmal_binding.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.runmal_binding`. The checker (`runmal_binding_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-runmal-binding/1`
Contract version: `1.0.0` (TF-034's `API_VERSION` — consumed, not reinvented)
Backend key: `thoughtforge`
Default enabled: False — the binding is OFF by default; enabling is an explicit act (`RunmalBinding(enabled=True)` or `binding.enable()`)
Calls: 9 (7 routed); field rows: 31; rules: 7
Probes: 7/7

## 1. One binding

`RunmalBinding` wraps TF-154's `CognitionAPI` and TF-155's `MemoryAPI` — each injectable directly, by factory, or via a `turn_callable` / `memory_store` passthrough, and both built lazily: constructing the binding builds neither API. While disabled, every routed call returns a truthful refusal as data (`ok=False`, `status="refused"`, the `error` naming the binding as disabled) and touches neither API. When enabled, cognition calls route through the CognitionAPI and memory calls through the MemoryAPI — called, never reimplemented. An invalid request raises `ValueError` naming the problem(s), judged by the pure validators before any API is built or called. Every response is a plain JSON-safe detached dict: `call`, `ok`, `status`, `backend_key`, `contract_version` (TF-034's `API_VERSION`), `error`, plus the call's payload (`turn` + `events` for chat, `models` for list_models, `health` for health_check, the TF-155 response verbatim in `memory` for memory calls). No Rúnmál code is imported or copied anywhere: Rúnmál's shapes are filed below as literal data read from its public backend contract.

## 2. The backend-table row

| Row | Backend key | Registry | Interface | Abstract methods | Default model | Enabled by default | What it is |
|---|---|---|---|---|---|---|---|
| thoughtforge_backend | thoughtforge | BackendRegistry (runmal/backends/registry.py) | ChatBackend (runmal/backends/base.py) — the abstract chat backend | `chat_stream`, `list_models`, `health_check` | None | no | one ThoughtForge entry in Rúnmál's backend table: it answers the ChatBackend interface under the key "thoughtforge"; the registry's aggregations (list_all_models / health_all / token_count) reach it through the same three abstract methods every backend answers; it carries no default model, so Rúnmál's own filter_params default changes nothing |

Registry semantics, filed from `runmal/backends/registry.py`: `register(backend, name=None)` returns `name or backend.backend_key` — a ThoughtForge backend registers under `thoughtforge` with no name argument; `get(key)` returns the backend; `keys()` is sorted. The registry's aggregations (`list_all_models`, `list_models_for`, `health_all`, `health_for`, `health_dicts`, `token_count`, `token_count_from_usage`) reach the entry through the same interface.

## 3. The calls

| Call | Surface | Rúnmál side | ThoughtForge route | Binding method | Routed |
|---|---|---|---|---|---|
| chat_stream | cognition | ChatBackend.chat_stream(messages: list[ChatMessage], params: SamplingParams, cancel: asyncio.Event) -> AsyncIterator[Event] | TF-154 CognitionAPI.turn — the turn-in assembled by the filed rule in FIELD_ROWS (last user message as user_text, every other message as history); the turn-out returns verbatim in the response's turn and as the filed reply events | chat(messages, params=None) | yes |
| list_models | cognition | ChatBackend.list_models() -> list[ModelInfo] (the registry aggregates it as list_all_models / list_models_for) | the filed FILED_MODEL row — an honest answer from TF-154's CognitionAPI contract: the one model is the ThoughtForge cognition API itself; no model process is queried and none is invented | list_models() | yes |
| health_check | cognition | ChatBackend.health_check() -> HealthStatus (the registry aggregates it as health_all / health_for / health_dicts) | the filed FILED_HEALTH row — reports the binding's own enabled state against the TF-154 CognitionAPI / TF-155 MemoryAPI contract; claims no live model health | health_check() | yes |
| count_tokens | cognition | ChatBackend.count_tokens(messages, model_id) -> TokenCount — a concrete default on the ABC (the shared runmal.tokens counter over concatenated content) | not routed: the method is Rúnmál's own concrete default; the binding files no counter and invents none — FILED_MODEL's token_count_quality "estimated" agrees with Rúnmál's default quality | — | no |
| filter_params | cognition | ChatBackend.filter_params(params) -> (SamplingParams, warnings) — a concrete default on the ABC, driven by the backend's default_model | not routed: the filed backend row's default_model is None, so Rúnmál's own default returns the params unchanged with no warnings — and FILED_MODEL's supported_samplers is empty, so no sampler is ever claimed supported | — | no |
| memory_add | memory | runmal.memory.service.create_memory(...) — Rúnmál's memory write call | TF-155 MemoryAPI.call("add", request) — TF-057's add_record through TF-155, called, never reimplemented: the TF-042 gate, the manual_entry stamp and the duplicate refusal are TF-057's by construction | memory_call("add", request) / memory_add(kind, ...) | yes |
| memory_inspect | memory | runmal.memory.service.get_memory(...) / list_memories(...) — Rúnmál's memory read calls | TF-155 MemoryAPI.call("inspect", request) — TF-057's inspect_store / inspect_records / find_record through TF-155, called, never reimplemented; read-only | memory_call("inspect", request) / memory_inspect(...) | yes |
| memory_prune | memory | runmal.memory.service.delete_memory(...) — Rúnmál's memory removal call | TF-155 MemoryAPI.call("prune", request) — PARTIAL by filing: TF-057's prune is a MemoryLifecycle pass over the whole store, not a targeted single-record delete; a targeted delete has no ThoughtForge home and is NOT routed — the prune is what this row honestly offers | memory_call("prune", request) / memory_prune(...) | yes |
| memory_export | memory | no counterpart call in runmal.memory.service — Rúnmál's rebuild / backup paths are its own | TF-155 MemoryAPI.call("export", request) — TF-057's export_store_bundle through TF-155, the verified TF-052 bundle returned as detached data; filed so all four TF-155 ops have a row and the parity campaign can consume the whole surface | memory_call("export", request) / memory_export(...) | yes |

## 4. Field mapping — in (Rúnmál -> turn-in)

| Field | Source | Target | Disposition | What it is |
|---|---|---|---|---|
| messages.last_user.content | ChatMessage.content of the last message with role "user" | turn-in user_text | mapped | the filed assembly rule's first half: the last user message is the turn's question — TF-154 requires user_text as a non-empty string, so an empty last user content is an invalid request, named by chat_request_problems() |
| messages.others | every message that is not the last user message, in original order | turn-in history | mapped_as_history | the filed assembly rule's second half: earlier user turns, assistant replies, system and tool messages all ride as history entries — each a plain {"role", "content"} dict, plus "name" when the message carries one; an empty history is not passed at all (TF-154 passes only set kwargs) |
| message.role | ChatMessage.role | the history entry's role; for the user_text source message, the role selects it and is not itself passed | mapped | roles are preserved verbatim in history — the binding rewrites no role and invents none |
| message.name | ChatMessage.name (optional) | the history entry's "name" when the message rides in history; no target when the message is the user_text source | partial | TF-154's turn-in files no name field: a name rides only inside its history entry; the user_text source message's name has no home and is filed here as dropped-by-filing, never silently |
| params.temperature | SamplingParams.temperature | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.top_p | SamplingParams.top_p | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.top_k | SamplingParams.top_k | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.min_p | SamplingParams.min_p | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.repetition_penalty | SamplingParams.repetition_penalty | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.presence_penalty | SamplingParams.presence_penalty | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.frequency_penalty | SamplingParams.frequency_penalty | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.max_output_tokens | SamplingParams.max_output_tokens | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.stop | SamplingParams.stop | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.seed | SamplingParams.seed | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| params.context_length | SamplingParams.context_length | no target — TURN_IN_FIELDS files no sampling field | no_home_not_passed | accepted in the call shape, never passed to TF-154, never claimed honored |
| cancel | chat_stream's cancel (asyncio.Event) | no target | no_home_not_wired | the binding is synchronous library data shaped on Rúnmál's async interface; cancellation is not wired through this filing — a Rúnmál caller cancelling mid-stream has no routed call here to cancel (any live async shape is TF-170's gate business, filed as a limit) |
| turn_in.retrieval_path | nothing on the Rúnmál side files a retrieval path | turn-in retrieval_path | no_source_not_set | the binding never sets it; the core's own default retrieval applies — no Rúnmál value is invented for it |
| turn_in.num_drafts | nothing on the Rúnmál side files a draft count | turn-in num_drafts | no_source_not_set | the binding never sets it; the core's own default draft count applies |
| turn_in.want_trace | nothing on the Rúnmál side files a trace request | turn-in want_trace | no_source_default_applies | the binding never sets it; TF-154's filed default (True) applies, so a routed turn carries its trace status honestly in the returned turn |

## 5. Field mapping — out (turn-out -> reply)

| Field | Source | Target | Disposition | What it is |
|---|---|---|---|---|
| turn.text | turn-out text | the response's turn (verbatim) and events[0] — a TextDelta-shaped dict {"type": "text_delta", "text": ...} | mapped | ONE delta carrying the whole text: the binding is not a streamer and FILED_MODEL's streaming False agrees — no token-by-token stream is pretended |
| turn.token_count | turn-out token_count | the response's turn (verbatim) and events[1] — a UsageUpdate-shaped dict with total_tokens set to it | partial | TF-154 files ONE count, not a prompt / completion split: the usage event files prompt_tokens and completion_tokens as None rather than inventing a split |
| turn.response_id | turn-out response_id | the response's turn (verbatim); no event home | mapped | Rúnmál's events file no response id; the id rides the binding response's turn, never dropped |
| turn.turn_id | turn-out turn_id | the response's turn (verbatim); no event home | mapped | rides the turn verbatim, beside its trace |
| turn.mode | turn-out mode | the response's turn (verbatim); no event home | mapped | the turn's mode (knowledge_only when no model served) rides the turn — a Rúnmál consumer can see honestly how the answer was produced |
| turn.enforcement_passed | turn-out enforcement_passed | the response's turn (verbatim); no event home | mapped | the enforcement verdict rides the turn verbatim |
| turn.citations | turn-out citations | the response's turn (verbatim); no event home | mapped | the turn's citations ride the turn verbatim |
| turn.contract_version | turn-out contract_version | the response's turn (verbatim) and the binding envelope's contract_version — the same TF-034 value | mapped | one version, TF-034's, in both places; no competing version is filed |
| turn.trace | turn-out trace | the response's turn (verbatim); no event home | mapped | TF-084's joined trace rides the turn when TF-154 includes it; events carry no trace |
| turn.trace_format | turn-out trace_format | the response's turn (verbatim); no event home | mapped | rides the turn verbatim beside the trace |
| turn.trace_status | turn-out trace_status | the response's turn (verbatim); no event home | mapped | rides the turn verbatim — included / not_requested / unavailable with its reason named, exactly as TF-154 filed it |
| events.done | this binding — the filed constant DONE_REASON | events[-1] — a Done-shaped dict {"type": "done", "reason": DONE_REASON, "usage": the usage event} | binding_filed | Rúnmál's Done.reason is a free string and TF-154 files no finish reason: the binding files its own constant "complete" rather than inventing a provider reason |

## 6. The rules

- `off_by_default`: the binding is off by default: DEFAULT_ENABLED is False, filed as data; enabling is an explicit act — the constructor flag enabled=True or the enable() method — and disable() returns the binding to refusal
- `disabled_refusal_is_data_touches_nothing`: while disabled, every routed call returns a truthful refusal as data (ok=False, status="refused", an error naming the binding as disabled) and touches neither API — no API is built and no call is routed, proven by the probes with counting factories
- `called_never_reimplemented`: cognition routes only through TF-154's CognitionAPI and memory only through TF-155's MemoryAPI (which itself calls TF-057's toolset) — this module implements no cognition, store or enforcement logic of its own
- `no_runmal_code_imported_or_copied`: no Rúnmál code is imported or copied: Rúnmál's shapes are filed as literal data read from its public backend contract (backends/base.py, registry.py, messages.py, models.py, params.py, events.py, memory/service.py), and nothing under the Rúnmál tree is edited by this slice
- `every_field_has_a_disposition`: every field on either side has a FIELD_ROWS row with a disposition — none is silently dropped and none is invented; the params and the cancel event are filed no-home in the open
- `params_never_claimed_honored`: sampling params are accepted in the call shape, never passed to TF-154 and never claimed honored; FILED_MODEL's empty supported_samplers and streaming False agree with that filing
- `responses_detached_json_safe_versioned`: every response is detached JSON-safe data (a JSON round-trip copy) whose contract_version is TF-034's API_VERSION, consumed from thoughtforge.cognition.api_surface — no competing version is invented here

## 7. Docket

TF-156 has **no share** of TF-016 and found no new defect in existing code, so this slice does not edit the docket and adds no entry. TF-016 totals stand unchanged (137 entries — 32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4 CLOSED-BY-VERDICT), verified live against the docket header and filed in the open here.

## 8. The probes

- `disabled_refusal_touches_nothing`: a default binding refuses every routed call as data (ok=False, the error naming the binding disabled), and counting factories prove neither API is ever built
- `enabled_cognition_routes_by_the_filed_rule`: an enabled binding routes a chat call through the injected CognitionAPI by the filed assembly rule (last user message as user_text, the rest as history in order, names riding) and returns the turn with the filed text_delta / usage / done events
- `enabled_memory_round_trip_temp_store`: an enabled binding adds through TF-155 (stamped manual_entry), finds the record through TF-155's inspect, and exports the verified bundle — all through the binding, on a temp store
- `backend_table_row_consistency`: a fake registry shaped on the filed row registers the fake ThoughtForge backend under "thoughtforge" with no name argument, answers get / keys as filed, and the row's abstract methods and FILED_MODEL's backend_key agree
- `models_and_health_are_honest`: list_models returns exactly the filed model (chat only, streaming False, no samplers claimed, token counts estimated) and health_check returns the filed health — reporting the binding, claiming no live model
- `params_not_passed_and_invalid_raises_named`: filed sampling params reach no turn (never passed, never claimed); an invalid enabled request (no user message, an unfiled params key, an empty list) raises ValueError naming the problem; the same garbage to a disabled binding returns the refusal as data instead
- `doc_matches_render_and_version_is_tf034`: the filed doc is the render verbatim, the contract version is TF-034's API_VERSION with its surface clean, and DEFAULT_ENABLED is False

## 9. Filed limits

- Proof grade FIXTURE: the checker and probes run a fake Rúnmál consumer shaped on the filed ChatBackend / registry data. NO live Rúnmál process was routed and NO Rúnmál code was imported or copied; live end-to-end routing is NOT PROVEN — TF-170's Part IX gate owns the at-least-one-live proof, never claimed here.
- The binding is synchronous library data shaped on Rúnmál's async interface (`chat_stream` is an async iterator in Rúnmál): a routed chat returns its filed events as one list with a single text delta, and cancellation is filed no-home. No streaming is pretended — the filed model's `streaming` is False.
- The memory delete is partial by filing: Rúnmál's `delete_memory` is a targeted single-record delete and TF-155's prune is a lifecycle pass; the binding routes only what prune honestly is and pretends no targeted delete.
- Sampling params have no turn-in home: they are accepted in the call shape, never passed to TF-154 and never claimed honored; the filed model's `supported_samplers` is empty.
- TF-157 owns the Samtal binding; TF-161 owns binding fixtures in depth; TF-162 owns versioning enforcement in depth; TF-163 owns the local-only proof for bindings. TF-156 files the Rúnmál binding shape, its off-by-default routing and this doc for the parity campaign's P-slices.
- No canon data was touched (TF-017); no README.md anywhere was touched; no Docker was used; nothing under the Rúnmál tree was edited.
