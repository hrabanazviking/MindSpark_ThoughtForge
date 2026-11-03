# Cognition API — Turn-In / Turn-Out with Traces (TF-154)

GENERATED from the filed data in `src/thoughtforge/cognition_api.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.cognition_api`. The checker (`cognition_api_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-cognition-api/1`
Contract version: `1.0.0` (TF-034's `API_VERSION` — consumed, not reinvented)
Trace format: `thoughtforge-joined-trace/1` (TF-084's `JOINED_FORMAT` — consumed, not reinvented)
Turn-in fields: 5; turn-out fields: 11; rules: 6
Probes: 7/7

## 1. One turn

`CognitionAPI.turn(user_text, retrieval_path=None, num_drafts=None, history=None, want_trace=True)` runs one turn through `ThoughtForgeCore.think()` and returns the turn-out as a plain JSON-safe dict. Only the kwargs the caller actually set (a non-None value) are passed to `think()`, whose frozen params are TF-034's. An invalid turn-in raises `ValueError` naming the problem(s), judged by the pure validator `turn_in_problems()` before any core is built or called — a library-layer caller error, never silently coerced. The core is built lazily: constructing `CognitionAPI()` builds no core until a turn, and a core / core factory / turn callable plus a trace callable can be injected (TF-153's injection shape).

## 2. Turn-in

| Field | Required | What it is |
|---|---|---|
| user_text | yes | the user's text for this turn — a non-empty string, required |
| retrieval_path | no | optional retrieval path override ("sql" | "vector" | "hybrid"), passed to think() only when set |
| num_drafts | no | optional cap on candidate drafts; >= 1 when present, per TF-034's ruling (a value below 1 is a caller error) |
| history | no | optional conversation history, passed to think() only when set |
| want_trace | no | optional bool, default True: whether the turn-out should carry TF-084's joined trace for the turn |

## 3. Turn-out

| Field | Source | What it is |
|---|---|---|
| text | FinalResponseRecord.text | the turn's response text |
| response_id | FinalResponseRecord.response_id | the final response's id |
| turn_id | FinalResponseRecord.turn_id | the turn's id — the included trace's turn_id must agree with it |
| mode | FinalResponseRecord.mode | the turn's mode (knowledge_only when no model served) |
| enforcement_passed | FinalResponseRecord.enforcement_passed | the enforcement verdict on the final response |
| citations | FinalResponseRecord.citations | the final response's citations, as a list of strings |
| token_count | FinalResponseRecord.token_count | the final response's token count |
| contract_version | TF-034 API_VERSION | the contract version — TF-034's API_VERSION, consumed, not reinvented |
| trace | TF-084 joined trace | TF-084's joined trace for the turn as detached JSON-safe data, or None when trace_status is not "included" |
| trace_format | TF-084 JOINED_FORMAT | the included trace's format (TF-084's JOINED_FORMAT), or None when no trace is included |
| trace_status | this API | "included", "not_requested" (want_trace=False), or "unavailable: <reason>" naming why no trace is filed |

The response fields are the real `FinalResponseRecord`'s: `text`, `response_id`, `turn_id`, `mode`, `enforcement_passed`, `citations`, `token_count`.

## 4. The trace (TF-084)

After a turn on a real core, the trace is obtained from the core's `last_turn_joined_trace()` — TF-084's joined trace in `thoughtforge-joined-trace/1`, validated live with `joined_trace_problems()`. `trace_status` is `"included"` when the trace rides the turn-out, `"not_requested"` when `want_trace=False`, or `"unavailable: <reason>"` when there is no trace source, the core returned none, or the trace failed its checker — the reason is named, a trace that fails its checker is never filed as good, and no trace is ever faked. With an injected turn callable and no trace callable, the status files unavailable honestly. The exported trace is detached JSON-safe data (a JSON round-trip copy): mutating a turn-out's trace cannot touch the core's retained state, and the trace's `turn_id` must agree with the turn-out's.

## 5. The rules

- `version_is_tf034`: the contract_version on every turn-out is TF-034's API_VERSION from thoughtforge.cognition.api_surface — no competing version is invented here (TF-162 owns versioning enforcement in depth)
- `turn_out_fields_from_final_response`: the turn-out's response fields are the real FinalResponseRecord's: text, response_id, turn_id, mode, enforcement_passed, citations and token_count — filed from the record the core actually returns
- `trace_is_tf084_joined`: the trace is TF-084's joined trace, in its JOINED_FORMAT, validated live by its own checker joined_trace_problems(); the trace's turn_id must agree with the turn-out's turn_id
- `trace_is_detached_json_safe`: the exported trace is detached JSON-safe data (a JSON round-trip copy): mutating a turn-out's trace cannot touch the core's retained state
- `no_trace_is_filed_never_faked`: when no trace is filed, trace_status names why — "not_requested" when want_trace=False, "unavailable: <reason>" when there is no trace source, the core returned none, or the trace failed its checker; a bad trace is never filed as good and no trace is ever faked
- `invalid_turn_in_raises_named`: an invalid turn-in raises ValueError naming the problem(s) judged by turn_in_problems() — a library-layer caller error, never silently coerced

## 6. Docket

TF-154 has **no share** of TF-016 and found no new defect in existing code, so this slice does not edit the docket and adds no entry. TF-016 totals stand unchanged (137 entries — 31 OPEN / 1 OPEN-ENVIRONMENT / 101 FIXED / 4 CLOSED-BY-VERDICT), filed in the open here.

## 7. The probes

- `knowledge_turn_with_trace_included`: a knowledge-only core turn through the API returns a clean turn-out with mode knowledge_only and its TF-084 joined trace included and clean
- `trace_turn_id_agrees_and_sections_present`: the included trace's turn_id agrees with the turn-out's and its cognition / inference / reconciliation sections are present in TF-084's JOINED_FORMAT
- `trace_is_detached_from_core`: mutating the returned trace leaves the core's re-exported joined trace unchanged and clean
- `want_trace_false_files_not_requested`: want_trace=False files trace_status not_requested and no trace
- `injected_turn_without_trace_source_unavailable`: an injected turn callable with no trace source files trace_status unavailable naming why, and no trace
- `version_is_tf034`: the contract version is TF-034's API_VERSION, verify_api_surface() is clean, and the api manifest's version matches
- `invalid_turn_in_raises_named`: an invalid turn-in (empty user_text, num_drafts=0) raises ValueError naming the problem

## 8. Filed limits

- Proof grade FIXTURE: the checker and probes run a knowledge-only core (the TF-084 fixture shape) on this VM. A live model turn through the cognition API is NOT PROVEN, never claimed.
- TF-155 owns the memory API; TF-162 owns versioning enforcement in depth. TF-154 files the turn-in / turn-out, the TF-034 version consumed, and the TF-084 trace carried.
- TF-153's service shape can carry this API, but wiring the service turn op to it is NOT done here: `service_shape.py` is not modified by this slice, so the service and this API stand as separate filed shapes until a later slice wires them.
- No canon data was touched (TF-017); no README.md anywhere was touched; no Docker was used.
