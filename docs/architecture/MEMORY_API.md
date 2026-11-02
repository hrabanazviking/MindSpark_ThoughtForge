# Memory API — Store Operations as Service Calls (TF-155)

GENERATED from the filed data in `src/thoughtforge/memory_api.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.memory_api`. The checker (`memory_api_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-memory-api/1`
Contract version: `1.0.0` (TF-034's `API_VERSION` — consumed, not reinvented)
Bundle format: `thoughtforge-store-bundle` (TF-052's `BUNDLE_FORMAT` — consumed, not reinvented)
Ops: 4; request fields: 21; response fields: 5; rules: 6
Probes: 7/7

## 1. One call

`MemoryAPI(store).call(op, request)` — or the kwargs convenience methods `add(kind, ...)`, `inspect(...)`, `prune(...)`, `export(...)` — runs one of TF-057's store operations and returns the response as a plain JSON-safe detached dict. The API calls TF-057's own functions in `thoughtforge.knowledge.forge_toolset`; it reimplements no store logic, so the enforcement is TF-057's by construction. An invalid request raises `ValueError` naming the problem(s), judged by the pure request validators before any store is built or called — a library-layer caller error, never silently coerced. An enforcement refusal from the toolset returns an `ok=False`, `status="refused"` response whose `refusal` names the toolset's own message — data, never a silent success. The store is built lazily: constructing `MemoryAPI()` builds no store until a call, and a store / store factory / toolset callables can be injected (TF-154's injection shape).

## 2. The ops (TF-057's toolset)

| Op | TF-057 operation | Toolset function(s) | Writes the store | What it does |
|---|---|---|---|---|
| add | add | `add_record` | yes | add one memory record by hand through TF-057's add_record — stamped source='manual_entry' (TF-042) and landed through the store's ingestion gate |
| inspect | inspect | `inspect_store`, `inspect_records`, `find_record` | no | report one store, list its records, or find one record by id through TF-057's inspect_store / inspect_records / find_record — read-only |
| prune | prune | `prune_store` | yes | one MemoryLifecycle prune pass in a named mode through TF-057's prune_store, reported before/after |
| export | export | `export_store_bundle` | no | the whole store as one verified TF-052 bundle through TF-057's export_store_bundle, returned as detached JSON-safe data |

## 3. Request fields

| Op | Field | Required | What it is |
|---|---|---|---|
| add | kind | yes | the record kind to add — one of TF-042's stamped kinds in TF-057's ADD_KINDS (user_preference / user_fact / episode / response_pattern); an unknown kind is refused by the toolset as data |
| add | summary | no | the record's summary text; the store gate judges its validity (an empty or over-long summary is refused by the gate as data, with nothing written) |
| add | tags | no | optional list of tag strings |
| add | category | no | optional category (user_preference) |
| add | value | no | optional value (user_preference) |
| add | fact_value | no | optional mapping of fact values (user_fact) |
| add | pattern_shape | no | optional list of shape steps (response_pattern) |
| add | tone | no | optional tone (episode) |
| add | importance | no | optional importance score; the gate judges its range |
| add | confidence | no | optional confidence score; the gate judges its range |
| add | quality | no | optional quality score; the gate judges its range |
| add | record_id | no | optional explicit record id; an id that already lives in the store is refused by the toolset as data — add never duplicates or overwrites |
| inspect | record_id | no | when set, find this one record id across every surface (find_record); a miss is data (found: False), not a refusal |
| inspect | kind | no | optional record-kind filter for the listing (inspect_records); an unknown kind is refused by the toolset as data |
| inspect | tag | no | optional tag filter for the listing |
| inspect | status | no | optional status filter for the listing |
| inspect | text | no | optional text needle for the listing, matched over summaries as TF-057 matches it (folded, TF-060) |
| inspect | profile_id | no | optional hardware profile: the store report then carries TF-056's budget report for the same store |
| prune | mode | no | the MemoryLifecycle prune mode, default "routine"; an unknown mode is refused by the lifecycle through the toolset as data, before any store is rewritten |
| prune | profile_id | no | optional hardware profile whose caps the same prune pass enforces (TF-056) |
| export | destination | no | optional path the bundle file is written to; when omitted the bundle is written to a temporary file, read back and verified, and returned in the response |

Request validation is shape only: which kinds exist, whether a record passes the TF-042 gate, which prune modes exist — those are TF-057's enforcement and surface as refusal data, never as shape verdicts.

## 4. The response

| Field | Source | What it is |
|---|---|---|
| op | this API | the op that was called |
| ok | this API | True when the toolset call landed; False when the toolset refused — a refusal is data, never a silent success |
| status | TF-057 toolset report / this API | the toolset report's own status on success (added / inspected / listed / found / not_found / pruned / exported), or "refused" when ok is False |
| contract_version | TF-034 API_VERSION | the contract version — TF-034's API_VERSION, consumed, not reinvented |
| refusal | this API | None on success; on refusal, the toolset's own refusal message naming the problem (the gate's verdict, the duplicate id, the unknown kind or mode) |

On success the toolset report's own fields ride the envelope flattened in (an add's `kind` / `record_id` / `source` / `record`, a store report's counts and bytes, a listing's `records` and `count`, a prune's `counts` / `before` / `after`). The export response additionally carries `path`, `format`, `version`, `counts`, `total_records`, `verified`, `problems` and `bundle` — the TF-052 bundle itself, written through TF-057's `export_store_bundle`, read back with `read_bundle` and re-verified with `verify_bundle`, exported as detached JSON-safe data (a JSON round-trip copy): mutating a response's bundle cannot touch the store.

## 5. The rules

- `ops_are_tf057s_toolset`: the four ops are TF-057's toolset operations (add / inspect / prune / export) and each call goes to TF-057's own functions in thoughtforge.knowledge.forge_toolset — this module reimplements no store logic
- `same_enforcement_by_construction`: because the calls are TF-057's functions, the enforcement is the same by construction: adds land through the TF-042 ingestion gate stamped manual_entry, duplicate ids are refused, prune is the TF-045 lifecycle's act, and export is a verified TF-052 bundle
- `invalid_request_raises_named`: an invalid request (unknown op, unfiled field, missing required field, wrong-typed value) raises ValueError naming the problem(s) judged by the pure request validators — a library-layer caller error, never silently coerced
- `enforcement_refusal_is_data`: an enforcement refusal from the toolset surfaces as data — an ok=False, status="refused" response whose refusal names the toolset's own message — never as a silent success and never as an unfiled raise
- `export_is_verified_detached_bundle`: the export op returns the TF-052 bundle as detached JSON-safe data (a JSON round-trip copy), written through TF-057's export_store_bundle, read back with read_bundle and re-verified with verify_bundle; mutating a response's bundle cannot touch the store
- `version_is_tf034`: the contract_version on every response is TF-034's API_VERSION from thoughtforge.cognition.api_surface — no competing version is invented here (TF-162 owns versioning enforcement in depth)

## 6. Docket

TF-155 has **no share** of TF-016 and found no new defect in existing code, so this slice does not edit the docket and adds no entry. TF-016 totals stand unchanged (137 entries — 32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4 CLOSED-BY-VERDICT), filed in the open here.

## 7. The probes

- `add_lands_stamped_and_findable`: an add through the API lands stamped source='manual_entry', is findable through the API, and sits in the tag index and the recent queue
- `inspect_is_read_only`: the store report, the listing and the find through the API leave the store byte-for-byte unchanged
- `gate_refusal_surfaces_as_data`: a gate-refused add (empty summary) and a duplicate-id add return ok=False refusal dicts naming the refusal, with nothing written
- `export_verified_and_detached`: an export through the API verifies with verify_bundle() clean and mutating the returned bundle leaves a re-export unchanged
- `prune_reports_and_refuses_as_data`: a routine prune through the API reports before/after counts, and an unknown mode returns an ok=False refusal dict before any write
- `version_is_tf034_and_toolset_clean`: the contract version is TF-034's API_VERSION, verify_api_surface() is clean, and TF-057's forge_toolset_problems() is clean
- `invalid_request_raises_named`: an invalid request (unknown op, unfiled field, missing kind) raises ValueError naming the problem

## 8. Filed limits

- Proof grade FIXTURE: the checker and probes run synthetic temp stores on this VM. Behaviour over a real, lived-in store is NOT PROVEN (no real store exists on this machine, TF-041) and any live model behaviour is NOT PROVEN (TF-173 class), never claimed.
- TF-156 owns the Rúnmál binding; TF-162 owns versioning enforcement in depth. TF-155 files the four store ops as service calls over TF-057's toolset, with TF-034's version consumed.
- TF-153's service shape can carry this API, but wiring the service ops to it is NOT done here: `service_shape.py` is not modified by this slice, so the service and this API stand as separate filed shapes until a later slice wires them.
- No canon data was touched (TF-017); no README.md anywhere was touched; no Docker was used.
