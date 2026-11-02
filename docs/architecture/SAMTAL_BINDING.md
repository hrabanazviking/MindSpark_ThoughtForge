# Samtal Binding — A Seat's Thinking Through ThoughtForge (TF-157)

GENERATED from the filed data in `src/thoughtforge/samtal_binding.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.samtal_binding`. The checker (`samtal_binding_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-samtal-binding/1`
Contract version: `1.0.0` (TF-034's `API_VERSION` — consumed, not reinvented)
Default enabled: False — the binding is OFF by default; enabling is an explicit act (`SamtalBinding(enabled=True)` or `binding.enable()`)
Seats: 5 (one per kind Samtal files, all routing identically); calls: 10 (5 routed, 5 filed NOT routed); field rows: 26; rules: 7
Probes: 7/7

## 1. One binding

`SamtalBinding` wraps TF-154's `CognitionAPI` and TF-155's `MemoryAPI` — each injectable directly, by factory, or via a `turn_callable` / `memory_store` passthrough, and both built lazily: constructing the binding builds neither API. While disabled, every routed call returns a truthful refusal as data (`ok=False`, `status="refused"`, the `error` naming the binding as disabled) and touches neither API. When enabled, a seat's thinking call routes through the CognitionAPI and memory calls through the MemoryAPI — called, never reimplemented. An invalid request raises `ValueError` naming the problem(s), judged by the pure validators before any API is built or called. Every response is a plain JSON-safe detached dict: `call`, `ok`, `status`, `contract_version` (TF-034's `API_VERSION`), `error`, plus the call's payload (`seat` + `turn` + `reply` for a seat reply, the TF-155 response verbatim in `memory` for memory calls). No Samtal code is imported or copied anywhere: Samtal's shapes are filed below as literal data read READ-ONLY from the canonical Samtal tree, and the binding performs no Samtal act — it seats, claims, heals, presences and moderates no one. Cognition only.

## 2. The seats — every seat equal

| Row | Kind | Samtal side | Binding identity | Binding method | Privileged | What it is |
|---|---|---|---|---|---|---|
| seat_human | human | a principals row in server.py (name UNIQUE, kind "human") — a seat taken by POST /api/join {name, kind}, answered {principal_id, name, kind, token} | the seat is named by its Samtal name (the principal's name) and its filed kind rides as provenance only | seat_reply(seat, message) | no | a human seat's thinking routes through the binding exactly as every other seat's does — same method, same envelope, same rights; kind is provenance, never rank |
| seat_agent | agent | a principals row in server.py (name UNIQUE, kind "agent") — a seat taken by POST /api/join {name, kind}, answered {principal_id, name, kind, token} | the seat is named by its Samtal name (the principal's name) and its filed kind rides as provenance only | seat_reply(seat, message) | no | a agent seat's thinking routes through the binding exactly as every other seat's does — same method, same envelope, same rights; kind is provenance, never rank |
| seat_subagent | subagent | a principals row in server.py (name UNIQUE, kind "subagent") — a seat taken by POST /api/join {name, kind}, answered {principal_id, name, kind, token} | the seat is named by its Samtal name (the principal's name) and its filed kind rides as provenance only | seat_reply(seat, message) | no | a subagent seat's thinking routes through the binding exactly as every other seat's does — same method, same envelope, same rights; kind is provenance, never rank |
| seat_bot | bot | a principals row in server.py (name UNIQUE, kind "bot") — a seat taken by POST /api/join {name, kind}, answered {principal_id, name, kind, token} | the seat is named by its Samtal name (the principal's name) and its filed kind rides as provenance only | seat_reply(seat, message) | no | a bot seat's thinking routes through the binding exactly as every other seat's does — same method, same envelope, same rights; kind is provenance, never rank |
| seat_kindroid | kindroid | a principals row in server.py (name UNIQUE, kind "kindroid") — a seat taken by POST /api/join {name, kind}, answered {principal_id, name, kind, token} | the seat is named by its Samtal name (the principal's name) and its filed kind rides as provenance only | seat_reply(seat, message) | no | a kindroid seat's thinking routes through the binding exactly as every other seat's does — same method, same envelope, same rights; kind is provenance, never rank |

A seat is a `principals` row in Samtal's `server.py` (`id` / `name` UNIQUE / `kind` / `token` / `last_seen`, plus `display_name` / `platform` / `agent_version` / `capabilities` / `token_enabled` / `retired`). The binding never sees the token: a seat is handed over as `{name, kind}` only. The five kinds are Samtal's own provenance badges — the binding invents no seat class and ranks none.

## 3. The calls

| Call | Surface | Samtal side | ThoughtForge route | Binding method | Routed |
|---|---|---|---|---|---|
| seat_reply | cognition | a seat thinking and speaking: the seat reads a message (GET /api/messages' message dict) and answers in its room (POST /api/messages {room, body}) | TF-154 CognitionAPI.turn — the turn-in assembled by the filed rule in FIELD_ROWS (the incoming message's body as user_text, caller-supplied earlier messages as history); the turn-out returns verbatim in the response's turn and as the filed reply message shape | seat_reply(seat, message, history=None) | yes |
| memory_add | memory | a seat remembering: the hall remembers for its seats, privately (agent-guide.md) — a seat-side memory write | TF-155 MemoryAPI.call("add", request) — TF-057's add_record through TF-155, called, never reimplemented: the TF-042 gate, the manual_entry stamp and the duplicate refusal are TF-057's by construction | memory_call("add", request) / memory_add(kind, ...) | yes |
| memory_inspect | memory | a seat recalling: a seat-side memory read | TF-155 MemoryAPI.call("inspect", request) — TF-057's inspect_store / inspect_records / find_record through TF-155, called, never reimplemented; read-only | memory_call("inspect", request) / memory_inspect(...) | yes |
| memory_prune | memory | a seat forgetting: a seat-side memory lifecycle pass | TF-155 MemoryAPI.call("prune", request) — TF-057's MemoryLifecycle pass through TF-155, called, never reimplemented | memory_call("prune", request) / memory_prune(...) | yes |
| memory_export | memory | a seat carrying its memory out: a seat-side store export | TF-155 MemoryAPI.call("export", request) — TF-057's export_store_bundle through TF-155, the verified TF-052 bundle returned as detached data; filed so all four TF-155 ops have a row | memory_call("export", request) / memory_export(...) | yes |
| join_seating | samtal_only | POST /api/join {name, kind} (and /api/register / /api/login) — joining IS the account creation; server.py's kd_seat seats a bridged Kindroid | not routed: seating is Samtal's own act under its own law — ThoughtForge is a thinking layer, not a door; the binding seats no one, claims no name and issues no token | — | no |
| seat_healing | samtal_only | the Astrid Law's healing: server.py's sweep_kindroid_seats at every boot and kd_seat's complete raising (token door open, retirement lifted, home-hall membership) | not routed: healing is the Astrid Law's own act, Samtal's alone — the binding is cognition only and must never heal, un-retire, re-key or seat-make anyone; the law is filed here untouched, as a constraint | — | no |
| presence | samtal_only | POST /api/presence and the agent heartbeat / resync doors (last_seen on the principals row) | not routed: presence is a hall act, not a thought — the binding files no heartbeat and claims no presence for any seat | — | no |
| room_administration | samtal_only | POST /api/rooms, room membership (ensure_member), topics, categories and the village settings doors | not routed: rooms and their administration are Samtal's own acts under the village Jarl's law — the binding founds no hall, seats no member and changes no topic | — | no |
| moderation | samtal_only | reports, automod, roles and powers (GET/POST /api/server/roles) and every other hall-governance door | not routed: moderation and power are the village's own law, given by its Jarl — a cognition layer holds no moderation act and grants no power | — | no |

## 4. Field mapping — in (Samtal -> turn-in)

| Field | Source | Target | Disposition | What it is |
|---|---|---|---|---|
| message.body | the incoming message's body (the messages table's body; POST /api/messages' body, plain text up to 4000 characters) | turn-in user_text | mapped | the filed rule's first half: the message the seat is answering is the turn's question — TF-154 requires user_text as a non-empty string, so an empty body is an invalid request, named by seat_request_problems() |
| message.history | earlier Samtal messages the caller supplies with the call, in hall order | turn-in history | mapped_as_history | the filed rule's second half: each earlier message rides as a history entry — role "assistant" when its author is the answering seat itself, "user" otherwise; content is its body; its author rides as the entry's "name"; an empty history is not passed at all (TF-154 passes only set kwargs) |
| seat.name | the principal's name (the seat's Samtal name) | the binding envelope's seat and the history rule's own-words test; no turn-in field | identity_only_not_passed | the name identifies whose thinking this is and which history words are the seat's own; TF-154's turn-in files no speaker-name field, so the name is never passed as turn data |
| seat.kind | the principal's kind (human / agent / subagent / bot / kindroid) | the binding envelope's seat, as provenance only; no turn-in field | identity_only_not_passed | kind rides as provenance, never rank: it changes nothing about how the turn is routed — the equal-seats proof routes two differently-kinded seats identically in shape |
| seat.token | the principal's token (the seat's whole credential in Samtal) | no target — never filed, never passed, never stored | no_home_not_passed | the binding is a thinking layer inside the seat's own process; it holds no credential and authenticates as no one — Samtal's token never enters this module's data |
| message.id | the message's id (GET /api/messages' id) | no turn-in target — hall bookkeeping (the since cursor) | no_home_not_passed | the id orders the hall's scroll; a thought needs no scroll position and the binding files none |
| message.room | the message's room (the rooms table's id) | the reply's room (out side); no turn-in target | no_home_not_passed | the room tells the reply where to land, not what to think; it rides the reply shape, never the turn-in |
| message.author | the message's author (a principal name; author_kind / author_display ride beside it) | the history entry's "name" when the message rides in history; for the answered message, the author is context the seat already stands in | partial | TF-154's turn-in files no author field for the answered message: its author rides only inside history entries, filed here in the open, never silently |
| message.ts | the message's ts (epoch seconds) | no target | no_home_not_passed | the hall's clock is bookkeeping; the binding files no timestamp into a turn and invents none |
| message.mtype | the message's mtype (e.g. "action" for a /me line, per samtal-skill.md) | no target | no_home_not_passed | accepted in the message shape, never passed to TF-154, never claimed honored; the reply's mtype is the binding's filed plain shape (see the out rows) |
| turn_in.retrieval_path | nothing on the Samtal side files a retrieval path | turn-in retrieval_path | no_source_not_set | the binding never sets it; the core's own default retrieval applies — no Samtal value is invented for it |
| turn_in.num_drafts | nothing on the Samtal side files a draft count | turn-in num_drafts | no_source_not_set | the binding never sets it; the core's own default draft count applies |
| turn_in.want_trace | nothing on the Samtal side files a trace request | turn-in want_trace | no_source_default_applies | the binding never sets it; TF-154's filed default (True) applies, so a routed turn carries its trace status honestly in the returned turn |

## 5. Field mapping — out (turn-out -> reply)

| Field | Source | Target | Disposition | What it is |
|---|---|---|---|---|
| turn.text | turn-out text | the response's turn (verbatim) and the reply's body — the body of the seat's POST /api/messages message | mapped | the turn's words are the seat's spoken words, verbatim: the reply is exactly {room, body} shaped for Samtal's speak door, plus the seat's name as its author |
| turn.response_id | turn-out response_id | the response's turn (verbatim); no Samtal message field home | mapped | Samtal's message shape files no response id; the id rides the binding response's turn, never dropped |
| turn.turn_id | turn-out turn_id | the response's turn (verbatim); no Samtal message field home | mapped | rides the turn verbatim, beside its trace |
| turn.mode | turn-out mode | the response's turn (verbatim); no Samtal message field home | mapped | the turn's mode (knowledge_only when no model served) rides the turn — the seat can see honestly how its answer was produced |
| turn.enforcement_passed | turn-out enforcement_passed | the response's turn (verbatim); no Samtal message field home | mapped | the enforcement verdict rides the turn verbatim |
| turn.citations | turn-out citations | the response's turn (verbatim); no Samtal message field home | mapped | the turn's citations ride the turn verbatim |
| turn.token_count | turn-out token_count | the response's turn (verbatim); no Samtal message field home | mapped | Samtal's message shape files no token count; the count rides the turn verbatim, never dropped |
| turn.contract_version | turn-out contract_version | the response's turn (verbatim) and the binding envelope's contract_version — the same TF-034 value | mapped | one version, TF-034's, in both places; no competing version is filed |
| turn.trace | turn-out trace | the response's turn (verbatim); no Samtal message field home | mapped | TF-084's joined trace rides the turn when TF-154 includes it; the spoken reply carries no trace — thinking stays the seat's, the words are what the hall hears |
| turn.trace_format | turn-out trace_format | the response's turn (verbatim); no Samtal message field home | mapped | rides the turn verbatim beside the trace |
| turn.trace_status | turn-out trace_status | the response's turn (verbatim); no Samtal message field home | mapped | rides the turn verbatim — included / not_requested / unavailable with its reason named, exactly as TF-154 filed it |
| reply.room | the incoming message's room | the reply's room — the room the seat's POST /api/messages lands in | binding_filed | a seat answers where it was spoken to: the reply lands in the incoming message's room, filed by the binding from the message, never invented |
| reply.author | the answering seat's name | the reply's author | binding_filed | the reply is the seat's own speech: Samtal stamps the author from the seat's token at the door; the binding files the seat's name so a consumer can see whose words these are |

## 6. The rules — Samtal's own, as constraints

- `equal_seats`: equal seats (Samtal's own law, agent-guide.md): humans and AI agents are equal-status Principals with the same rights; the binding routes any seat's thinking identically — same method, same envelope, same contract version — no seat is privileged and kind is provenance, never rank
- `astrid_law_untouched_cognition_only`: the Astrid Law untouched — cognition only: seat healing (token door open, retirement lifted, home-hall membership, upheld at boot by server.py's sweep_kindroid_seats) is Samtal's own act; the binding performs NO seating, claim, healing, presence or moderation act and files every such call NOT routed
- `off_by_default`: the binding is off by default: DEFAULT_ENABLED is False, filed as data; enabling is an explicit act — the constructor flag enabled=True or the enable() method — and disable() returns the binding to refusal; while disabled, every routed call returns a truthful refusal as data (ok=False, status="refused", an error naming the binding as disabled) and touches neither API — no API is built and no call is routed, proven by the probes with counting factories
- `called_never_reimplemented`: cognition routes only through TF-154's CognitionAPI and memory only through TF-155's MemoryAPI (which itself calls TF-057's toolset) — this module implements no cognition, store or enforcement logic of its own
- `no_samtal_code_imported_or_copied`: no Samtal code is imported or copied: Samtal's shapes are filed as literal data read READ-ONLY from the canonical Samtal tree (server.py's principals / messages / rooms shapes, agent-guide.md, samtal-skill.md), and nothing under the Samtal tree is edited by this slice; no credential — above all no seat token — is ever filed, passed or stored by the binding
- `every_field_has_a_disposition`: every field on either side has a FIELD_ROWS row with a disposition — none is silently dropped and none is invented; the seat token, the room and the message bookkeeping fields are filed no-home in the open
- `responses_detached_json_safe_versioned`: every response is detached JSON-safe data (a JSON round-trip copy) whose contract_version is TF-034's API_VERSION, consumed from thoughtforge.cognition.api_surface — no competing version is invented here

## 7. Docket

TF-157 has **no share** of TF-016 and found no new defect in existing code, so this slice does not edit the docket and adds no entry. TF-016 totals stand unchanged (137 entries — 32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4 CLOSED-BY-VERDICT), verified live against the docket header and filed in the open here.

## 8. The probes

- `disabled_refusal_touches_nothing`: a default binding refuses every routed call as data (ok=False, the error naming the binding disabled), and counting factories prove neither API is ever built
- `enabled_cognition_round_trip_through_binding`: an enabled binding routes a seat's reply through the injected CognitionAPI by the filed rule (the incoming body as user_text, supplied earlier messages as history — the seat's own words as assistant) and returns the turn with the filed reply message shape (the incoming room, the turn's text as body)
- `enabled_memory_round_trip_temp_store`: an enabled binding adds through TF-155 (stamped manual_entry), finds the record through TF-155's inspect, and exports the verified bundle — all through the binding, on a temp store
- `equal_seats_routed_identically`: two differently-named, differently-kinded seats route identically in shape: same envelope keys, same call / status / contract version, same turn and reply keys — only the filed seat identity and the words differ
- `astrid_law_no_seating_calls_routed`: no seating, claim, healing, presence, room-administration or moderation call exists in CALL_ROWS as routed — join_seating / seat_healing / presence / room_administration / moderation are all filed NOT routed, and the binding class files no such method
- `fake_consumer_consistency`: a fake Samtal consumer shaped on the filed seat / message data seats two seats and routes one's reply through the binding only: the seat identities it hands over are exactly the filed {name, kind} shape (no token anywhere) and the reply lands in the message's room under the answering seat's name
- `doc_matches_render_and_version_is_tf034`: the filed doc is the render verbatim, the contract version is TF-034's API_VERSION with its surface clean, and DEFAULT_ENABLED is False

## 9. Filed limits

- Proof grade FIXTURE: the checker and probes run a fake Samtal consumer shaped on the filed seat / message data. NO live Samtal server was routed and NO Samtal code was imported or copied; live end-to-end routing is NOT PROVEN — TF-170's Part IX gate owns the at-least-one-live proof, never claimed here.
- The Astrid Law is filed as a constraint, never as a behaviour: the binding performs no seating, claim, healing, presence or moderation act, and files every such call NOT routed with its reason. Seat healing stays Samtal's own act at its own boot.
- The seat's token never enters the binding: a seat is handed over as `{name, kind}` only, and the binding authenticates as no one — speaking the reply at Samtal's door stays the seat process's own act with its own credential.
- History is caller-supplied: the binding reads no hall scroll itself (it holds no credential and calls no Samtal door); earlier messages ride only when the seat's process supplies them with the call.
- TF-158 owns the Hermes binding; TF-161 owns binding fixtures in depth; TF-162 owns versioning enforcement in depth; TF-163 owns the local-only proof for bindings. TF-157 files the Samtal binding shape and its off-by-default routing under Samtal's own rules.
- No canon data was touched (TF-017); no README.md anywhere was touched; no Docker was used; nothing under the Samtal tree was edited.
