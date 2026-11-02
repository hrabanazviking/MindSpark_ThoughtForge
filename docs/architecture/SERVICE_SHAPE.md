# Service Shape — Library AND Local Service, One Contract (TF-153)

GENERATED from the filed data in `src/thoughtforge/service_shape.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.service_shape`. The checker (`service_shape_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-service-shape/1`
Request format: `thoughtforge-service-request/1`
Response format: `thoughtforge-service-response/1`
Contract version: `1.0.0` (TF-034's `API_VERSION` — consumed, not reinvented)
Default bind: `127.0.0.1` (loopback only)
Ops: 3; routes: 3; rules: 6
Probes: 7/7

## 1. One contract

A request is a plain JSON-safe dict: `{"format": "thoughtforge-service-request/1", "contract_version": "<TF-034 API_VERSION>", "op": "<op>", ...}`. A response is a plain JSON-safe dict: `{"format": "thoughtforge-service-response/1", "contract_version": ..., "op": ..., "ok": true, ...}` — and a refusal is the same dict with `"ok": false` and an `"error"` naming the problem. The library caller and the HTTP caller send and receive these exact dicts: `ThoughtForgeService.handle(request)` is the one entry point, and the HTTP routes call it unchanged.

## 2. The ops

| Op | Route | What it does |
|---|---|---|
| health | GET /health | the service is up and files its contract version and op ids |
| contract | GET /contract | the service contract data plus the TF-034 cognition api manifest it is versioned by |
| turn | POST /turn | one cognition turn in / out through ThoughtForgeCore.think(); the response fields are the FinalResponseRecord's |

The `turn` op's response fields are the real `FinalResponseRecord`'s: `text`, `response_id`, `turn_id`, `mode`, `enforcement_passed`, `citations`, `token_count`. TF-154 deepens the cognition API (traces, TF-084); TF-153 files only this shape.

## 3. The two shapes

**Library**: `ThoughtForgeService(core)` — or an injected core factory / turn callable — and `handle(request) -> response`. The core is built lazily, so health and contract never build one.

**Local service**: `make_server(service, host="127.0.0.1", port=0)` — stdlib `http.server` only, no new dependency. Routes: `GET /health`, `GET /contract`, `POST /turn` (the JSON body is the same request dict; if `op` is omitted in the body, the route supplies it). HTTP status: 200 on ok, 400 on a refusal, 404 on an unknown route, 405 on a wrong method — errors still arrive as the contract response dict where applicable.

## 4. The rules

- `one_contract_both_shapes`: the library handle() and the HTTP routes speak the SAME request and response dicts — one contract, two shapes; the HTTP layer adds no field and removes none
- `loopback_by_default`: DEFAULT_BIND is 127.0.0.1: the service binds the loopback interface unless a caller explicitly says otherwise
- `non_loopback_is_explicit`: binding a non-loopback host requires allow_non_loopback=True, judged before any socket is bound, and the act is recorded on the server object as non_loopback_acknowledged (TF-163 owns the logged proof in depth)
- `refusals_are_data_never_raises`: a malformed request returns a refusal response dict (ok=False, error naming the problem); handle() never raises on a malformed request and never silently corrupts one into a different op
- `contract_version_is_tf034`: the contract_version is TF-034's API_VERSION from thoughtforge.cognition.api_surface — no competing version is invented here; a request on an incompatible contract major version is refused (TF-162 owns versioning enforcement in depth)
- `turn_fields_from_final_response`: the turn response's fields are the real FinalResponseRecord's: text, response_id, turn_id, mode, enforcement_passed, citations and token_count — sizes and fields, filed from the record the core actually returns

## 5. Docket

TF-153 has **no share** of TF-016 and found no new defect in existing code, so this slice does not edit the docket and adds no entry. TF-016 totals stand unchanged (137 entries — 32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4 CLOSED-BY-VERDICT), filed in the open here.

## 6. The probes

- `library_health_ok`: a library handle() health request returns ok with the filed op ids
- `library_contract_is_tf034`: a library contract request returns TF-034's api_manifest() verbatim and this service contract at TF-034's API_VERSION
- `library_turn_knowledge_only`: a library turn against a knowledge-only core returns ok with the FinalResponseRecord fields (text / ids / mode / enforcement / citations)
- `http_equals_library`: a real loopback HTTP round-trip returns response dicts identical to library handle() for health and contract, status 200
- `refusal_on_unknown_op`: an unknown op returns a refusal dict naming it — handle() does not raise
- `refusal_on_bad_version_major`: a request on an incompatible contract major version returns a refusal dict naming the mismatch
- `non_loopback_requires_explicit_flag`: bind_problems() names a non-loopback host without the flag, clears it with the flag, and make_server() raises ValueError naming the gate before binding anything

## 7. Filed limits

- Proof grade FIXTURE: the checker and probes run a knowledge-only core (the TF-084 fixture shape) plus a real loopback HTTP round-trip on an ephemeral port on this VM. A live model turn through the service is NOT PROVEN, never claimed.
- Versioning enforcement in depth is TF-162's; local-only proof (the logged non-loopback record) is TF-163's. TF-153 files the shape, the major-mismatch refusal, the loopback default and the explicit gate.
- No canon data was touched (TF-017); no README.md anywhere was touched; no Docker was used.
