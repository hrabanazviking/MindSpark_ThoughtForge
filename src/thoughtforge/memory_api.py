"""
TF-155 — the memory API, filed: store
operations (TF-057's toolset) as service
calls with the same enforcement.

TF-155's roadmap text: *The memory API: store
operations (TF-057's toolset) as service calls
with the same enforcement.* It is Part IX
slice 3, following TF-154's cognition API and
TF-153's service shape.

**What this slice files**

  * ``MemoryAPI`` — TF-057's four store
    operations (``add`` / ``inspect`` /
    ``prune`` / ``export``) as service calls
    over one ``MemoryStore``. The API calls
    TF-057's own functions in
    ``thoughtforge.knowledge.forge_toolset``
    (``add_record`` / ``inspect_store`` /
    ``inspect_records`` / ``find_record`` /
    ``prune_store`` / ``export_store_bundle``)
    — it reimplements no store logic, so the
    enforcement is TF-057's *by construction*:
    the TF-042 ingestion gate's refusals,
    duplicate-id refusal, prune's lifecycle
    rules, and export as a verified TF-052
    bundle are the toolset's own acts.
  * ``MEMORY_OPS`` — the four ops as literal
    data, each row mapping the op to its
    TF-057 ``OPERATIONS`` row (name / CLI /
    ``writes_store``) and to the underlying
    toolset function(s) it calls. Per-op
    request-field tables as literal data
    (``MEMORY_REQUEST_FIELDS``, grouped by
    ``request_fields_for(op)``) and the
    response envelope as ``MEMORY_RESPONSE_FIELDS``.
  * Requests are plain dicts or kwargs;
    responses are plain JSON-safe detached
    dicts: ``{"op", "ok", "status",
    "contract_version", "refusal", ...}``
    with the toolset report's own fields
    flattened in on success. The contract
    version is consumed from TF-034
    (``thoughtforge.cognition.api_surface``'s
    ``API_VERSION`` / ``api_manifest()`` /
    ``verify_api_surface()``) — no competing
    version is invented here.
  * Two failure shapes, kept distinct. An
    invalid *request* (an unknown op, an
    unfiled field, a missing required field,
    a wrong-typed value — judged by the pure
    validators ``request_problems()`` /
    ``add_request_problems()`` /
    ``inspect_request_problems()`` /
    ``prune_request_problems()`` /
    ``export_request_problems()``) raises
    ``ValueError`` naming the problem(s) —
    a library-layer caller error, never
    silently coerced, judged before any
    store is built or called. An
    *enforcement* outcome from the toolset
    (the gate refusing an invalid record,
    a duplicate id, an unknown add kind, an
    unknown prune mode, an unknown inspect
    filter kind) surfaces as DATA: an
    ``ok=False``, ``status="refused"``
    response naming the refusal in
    ``refusal`` — never a silent success
    and never a raise the caller cannot
    file.
  * The export op returns the TF-052 bundle
    itself as detached JSON-safe data: the
    bundle is written through TF-057's
    ``export_store_bundle``, read back with
    TF-052's ``read_bundle`` and re-verified
    with ``verify_bundle``; the response
    files ``verified`` and ``problems``
    from that verification, and the bundle
    rides as a JSON round-trip copy, so
    mutating a response's bundle cannot
    touch the store.
  * ``MemoryAPI`` mirrors TF-154's injection
    shape: a store / store factory /
    toolset callables (``add_callable`` /
    ``inspect_store_callable`` /
    ``inspect_records_callable`` /
    ``find_callable`` / ``prune_callable`` /
    ``export_callable``, each taking the
    store first, exactly as TF-057's
    functions do). The store is built
    lazily: constructing ``MemoryAPI()``
    builds no store until a call.
  * ``MEMORY_RULES`` — the API's law as
    literal data, and
    ``memory_api_problems()`` — a checker
    that re-derives every claim live: the
    filings, the TF-034 surface, TF-057's
    ``forge_toolset_problems()``, a real
    round-trip on a temp store through the
    API only (add → inspect / find / list →
    export verified → prune), the filed
    probes, the docket header and the filed
    doc.

**Docket**: TF-155 has no share of TF-016 and
finds no new defect in existing code; it does
NOT edit ``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md``.
TF-016 totals stand unchanged (137 entries —
32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4
CLOSED-BY-VERDICT), filed in the open here and
in the render.

The human-readable doc is
``docs/architecture/MEMORY_API.md``, filed
verbatim as this module's render (regenerate
with ``python3 -m thoughtforge.memory_api``).
Proof grade: FIXTURE — synthetic temp stores
on this VM; behaviour over a real, lived-in
store and any live model behaviour are NOT
PROVEN (TF-173 class), never claimed.
TF-156 owns the Rúnmál binding; TF-162 owns
versioning enforcement in depth. TF-153's
service shape can carry this API, but wiring
the service ops to it is NOT done here —
``service_shape.py`` is not modified by this
slice.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from thoughtforge.cognition.api_surface import (
    API_VERSION,
    api_manifest,
    verify_api_surface,
)
from thoughtforge.knowledge.forge_toolset import (
    OPERATIONS as TOOLSET_OPERATIONS,
)
from thoughtforge.knowledge.forge_toolset import (
    add_record,
    export_store_bundle,
    find_record,
    forge_toolset_problems,
    inspect_records,
    inspect_store,
    prune_store,
)
from thoughtforge.knowledge.portability import (
    BUNDLE_FORMAT,
    read_bundle,
    verify_bundle,
)

MEMORY_API_FORMAT = "thoughtforge-memory-api/1"
MEMORY_API_DOC = "docs/architecture/MEMORY_API.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"


# ── The ops, as literal data ─────────────────────────────────────────────────

MEMORY_OPS: tuple[dict[str, Any], ...] = (
    {"id": "add",
     "toolset_operation": "add",
     "functions": ["add_record"],
     "writes_store": True,
     "summary": "add one memory record by hand through "
                "TF-057's add_record — stamped "
                "source='manual_entry' (TF-042) and landed "
                "through the store's ingestion gate"},
    {"id": "inspect",
     "toolset_operation": "inspect",
     "functions": ["inspect_store", "inspect_records",
                   "find_record"],
     "writes_store": False,
     "summary": "report one store, list its records, or "
                "find one record by id through TF-057's "
                "inspect_store / inspect_records / "
                "find_record — read-only"},
    {"id": "prune",
     "toolset_operation": "prune",
     "functions": ["prune_store"],
     "writes_store": True,
     "summary": "one MemoryLifecycle prune pass in a "
                "named mode through TF-057's prune_store, "
                "reported before/after"},
    {"id": "export",
     "toolset_operation": "export",
     "functions": ["export_store_bundle"],
     "writes_store": False,
     "summary": "the whole store as one verified TF-052 "
                "bundle through TF-057's "
                "export_store_bundle, returned as detached "
                "JSON-safe data"},
)

OP_IDS: tuple[str, ...] = tuple(op["id"] for op in MEMORY_OPS)


# ── Request fields, as literal data (per op) ─────────────────────────────────

MEMORY_REQUEST_FIELDS: tuple[dict[str, Any], ...] = (
    # add — TF-057 add_record's own parameters.
    {"op": "add", "id": "kind", "required": True,
     "summary": "the record kind to add — one of TF-042's "
                "stamped kinds in TF-057's ADD_KINDS "
                "(user_preference / user_fact / episode / "
                "response_pattern); an unknown kind is "
                "refused by the toolset as data"},
    {"op": "add", "id": "summary", "required": False,
     "summary": "the record's summary text; the store "
                "gate judges its validity (an empty or "
                "over-long summary is refused by the gate "
                "as data, with nothing written)"},
    {"op": "add", "id": "tags", "required": False,
     "summary": "optional list of tag strings"},
    {"op": "add", "id": "category", "required": False,
     "summary": "optional category (user_preference)"},
    {"op": "add", "id": "value", "required": False,
     "summary": "optional value (user_preference)"},
    {"op": "add", "id": "fact_value", "required": False,
     "summary": "optional mapping of fact values "
                "(user_fact)"},
    {"op": "add", "id": "pattern_shape", "required": False,
     "summary": "optional list of shape steps "
                "(response_pattern)"},
    {"op": "add", "id": "tone", "required": False,
     "summary": "optional tone (episode)"},
    {"op": "add", "id": "importance", "required": False,
     "summary": "optional importance score; the gate "
                "judges its range"},
    {"op": "add", "id": "confidence", "required": False,
     "summary": "optional confidence score; the gate "
                "judges its range"},
    {"op": "add", "id": "quality", "required": False,
     "summary": "optional quality score; the gate judges "
                "its range"},
    {"op": "add", "id": "record_id", "required": False,
     "summary": "optional explicit record id; an id that "
                "already lives in the store is refused by "
                "the toolset as data — add never duplicates "
                "or overwrites"},
    # inspect — TF-057's three inspect functions' parameters.
    {"op": "inspect", "id": "record_id", "required": False,
     "summary": "when set, find this one record id across "
                "every surface (find_record); a miss is "
                "data (found: False), not a refusal"},
    {"op": "inspect", "id": "kind", "required": False,
     "summary": "optional record-kind filter for the "
                "listing (inspect_records); an unknown kind "
                "is refused by the toolset as data"},
    {"op": "inspect", "id": "tag", "required": False,
     "summary": "optional tag filter for the listing"},
    {"op": "inspect", "id": "status", "required": False,
     "summary": "optional status filter for the listing"},
    {"op": "inspect", "id": "text", "required": False,
     "summary": "optional text needle for the listing, "
                "matched over summaries as TF-057 matches "
                "it (folded, TF-060)"},
    {"op": "inspect", "id": "profile_id", "required": False,
     "summary": "optional hardware profile: the store "
                "report then carries TF-056's budget report "
                "for the same store"},
    # prune — TF-057 prune_store's own parameters.
    {"op": "prune", "id": "mode", "required": False,
     "default": "routine",
     "summary": "the MemoryLifecycle prune mode, default "
                "\"routine\"; an unknown mode is refused by "
                "the lifecycle through the toolset as data, "
                "before any store is rewritten"},
    {"op": "prune", "id": "profile_id", "required": False,
     "summary": "optional hardware profile whose caps the "
                "same prune pass enforces (TF-056)"},
    # export — TF-057 export_store_bundle's destination.
    {"op": "export", "id": "destination", "required": False,
     "summary": "optional path the bundle file is written "
                "to; when omitted the bundle is written to "
                "a temporary file, read back and verified, "
                "and returned in the response"},
)


def request_fields_for(op: str) -> tuple[dict[str, Any], ...]:
    """One op's filed request fields, in filing order."""
    return tuple(f for f in MEMORY_REQUEST_FIELDS if f["op"] == op)


def request_field_ids(op: str) -> tuple[str, ...]:
    return tuple(f["id"] for f in request_fields_for(op))


# ── Response fields, as literal data ─────────────────────────────────────────

MEMORY_RESPONSE_FIELDS: tuple[dict[str, Any], ...] = (
    {"id": "op",
     "source": "this API",
     "summary": "the op that was called"},
    {"id": "ok",
     "source": "this API",
     "summary": "True when the toolset call landed; False "
                "when the toolset refused — a refusal is "
                "data, never a silent success"},
    {"id": "status",
     "source": "TF-057 toolset report / this API",
     "summary": "the toolset report's own status on "
                "success (added / inspected / listed / "
                "found / not_found / pruned / exported), "
                "or \"refused\" when ok is False"},
    {"id": "contract_version",
     "source": "TF-034 API_VERSION",
     "summary": "the contract version — TF-034's "
                "API_VERSION, consumed, not reinvented"},
    {"id": "refusal",
     "source": "this API",
     "summary": "None on success; on refusal, the "
                "toolset's own refusal message naming the "
                "problem (the gate's verdict, the duplicate "
                "id, the unknown kind or mode)"},
)

RESPONSE_IDS: tuple[str, ...] = tuple(
    f["id"] for f in MEMORY_RESPONSE_FIELDS)


# ── The rules, as literal data ───────────────────────────────────────────────

MEMORY_RULES: tuple[dict[str, Any], ...] = (
    {"id": "ops_are_tf057s_toolset",
     "rule": "the four ops are TF-057's toolset operations "
             "(add / inspect / prune / export) and each call "
             "goes to TF-057's own functions in "
             "thoughtforge.knowledge.forge_toolset — this "
             "module reimplements no store logic"},
    {"id": "same_enforcement_by_construction",
     "rule": "because the calls are TF-057's functions, "
             "the enforcement is the same by construction: "
             "adds land through the TF-042 ingestion gate "
             "stamped manual_entry, duplicate ids are "
             "refused, prune is the TF-045 lifecycle's act, "
             "and export is a verified TF-052 bundle"},
    {"id": "invalid_request_raises_named",
     "rule": "an invalid request (unknown op, unfiled "
             "field, missing required field, wrong-typed "
             "value) raises ValueError naming the "
             "problem(s) judged by the pure request "
             "validators — a library-layer caller error, "
             "never silently coerced"},
    {"id": "enforcement_refusal_is_data",
     "rule": "an enforcement refusal from the toolset "
             "surfaces as data — an ok=False, "
             "status=\"refused\" response whose refusal "
             "names the toolset's own message — never as a "
             "silent success and never as an unfiled raise"},
    {"id": "export_is_verified_detached_bundle",
     "rule": "the export op returns the TF-052 bundle as "
             "detached JSON-safe data (a JSON round-trip "
             "copy), written through TF-057's "
             "export_store_bundle, read back with "
             "read_bundle and re-verified with "
             "verify_bundle; mutating a response's bundle "
             "cannot touch the store"},
    {"id": "version_is_tf034",
     "rule": "the contract_version on every response is "
             "TF-034's API_VERSION from "
             "thoughtforge.cognition.api_surface — no "
             "competing version is invented here "
             "(TF-162 owns versioning enforcement in "
             "depth)"},
)


# ── The contract as plain data ───────────────────────────────────────────────

def memory_contract() -> dict[str, Any]:
    """The memory API contract as plain JSON-safe data,
    built from the filed literals above and TF-034's
    version. A fresh copy each call."""
    return {
        "format": MEMORY_API_FORMAT,
        "contract_version": API_VERSION,
        "bundle_format": BUNDLE_FORMAT,
        "ops": [dict(op) for op in MEMORY_OPS],
        "request_fields": [dict(f) for f in MEMORY_REQUEST_FIELDS],
        "response_fields": [dict(f) for f in MEMORY_RESPONSE_FIELDS],
        "rules": [dict(r) for r in MEMORY_RULES],
    }


# ── Pure validators ──────────────────────────────────────────────────────────

def memory_op_problems(row: Any) -> list[str]:
    """Judge one op filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["op: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("op: the row files no id")
    if not row.get("toolset_operation"):
        problems.append(f"op {row.get('id')}: no toolset_operation filed")
    if not row.get("functions") \
            or not isinstance(row.get("functions"), list):
        problems.append(f"op {row.get('id')}: no functions filed as a list")
    if not isinstance(row.get("writes_store"), bool):
        problems.append(f"op {row.get('id')}: writes_store is not a boolean")
    if not row.get("summary"):
        problems.append(f"op {row.get('id')}: no summary filed")
    return problems


def request_field_problems(row: Any) -> list[str]:
    """Judge one request-field filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["field: the row is empty"]
    problems: list[str] = []
    if row.get("op") not in OP_IDS:
        problems.append(
            f"field {row.get('id')}: op is {row.get('op')!r}, "
            f"not one of {OP_IDS}")
    if not row.get("id"):
        problems.append("field: the row files no id")
    if not isinstance(row.get("required"), bool):
        problems.append(
            f"field {row.get('id')}: required is not a boolean")
    if not row.get("summary"):
        problems.append(f"field {row.get('id')}: no summary filed")
    return problems


def memory_rule_problems(row: Any) -> list[str]:
    """Judge one rule filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["rule: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("rule: the row files no id")
    if not row.get("rule"):
        problems.append(f"rule {row.get('id')}: no rule filed")
    return problems


def _unfiled_problems(op: str, mapping: dict[str, Any]) -> list[str]:
    filed = request_field_ids(op)
    return [
        f"{op} request: {key!r} is not a filed {op} request "
        f"field — the filed fields are {list(filed)}"
        for key in mapping if key not in filed
    ]


def _string_problems(
    op: str, mapping: dict[str, Any], keys: tuple[str, ...],
    *, non_empty: tuple[str, ...] = (),
) -> list[str]:
    problems: list[str] = []
    for key in keys:
        if key in mapping and mapping[key] is not None:
            value = mapping[key]
            if not isinstance(value, str):
                problems.append(
                    f"{op} request: {key} must be a string when "
                    f"present, not {type(value).__name__}")
            elif key in non_empty and not value:
                problems.append(
                    f"{op} request: {key} must be a non-empty "
                    "string when present")
    return problems


def add_request_problems(mapping: Any) -> list[str]:
    """Judge one add request mapping, purely. Empty-safe.

    Shape only: ``kind`` is required as a non-empty
    string and every present field must carry its filed
    type. The *values* (which kinds exist, whether the
    record passes the gate) are TF-057's enforcement,
    surfaced as refusal data — never judged here."""
    if not isinstance(mapping, dict):
        return ["add request: the mapping is empty"]
    problems = _unfiled_problems("add", mapping)
    kind = mapping.get("kind")
    if not isinstance(kind, str) or not kind:
        problems.append(
            "add request: kind is required as a non-empty string")
    problems.extend(_string_problems(
        "add", mapping,
        ("summary", "category", "value", "tone"),
        non_empty=("record_id",)))
    if "record_id" in mapping and mapping["record_id"] is not None:
        if not isinstance(mapping["record_id"], str) \
                or not mapping["record_id"]:
            problems.append(
                "add request: record_id must be a non-empty "
                "string when present")
    if "tags" in mapping and mapping["tags"] is not None:
        tags = mapping["tags"]
        if not isinstance(tags, list) \
                or not all(isinstance(t, str) for t in tags):
            problems.append(
                "add request: tags must be a list of strings "
                "when present")
    if "pattern_shape" in mapping \
            and mapping["pattern_shape"] is not None:
        shape = mapping["pattern_shape"]
        if not isinstance(shape, list) \
                or not all(isinstance(s, str) for s in shape):
            problems.append(
                "add request: pattern_shape must be a list of "
                "strings when present")
    if "fact_value" in mapping and mapping["fact_value"] is not None \
            and not isinstance(mapping["fact_value"], dict):
        problems.append(
            "add request: fact_value must be a mapping when present")
    for key in ("importance", "confidence", "quality"):
        if key in mapping and mapping[key] is not None:
            value = mapping[key]
            if isinstance(value, bool) \
                    or not isinstance(value, (int, float)):
                problems.append(
                    f"add request: {key} must be a number when "
                    f"present, not {value!r}")
    return problems


def inspect_request_problems(mapping: Any) -> list[str]:
    """Judge one inspect request mapping, purely.
    Empty-safe: an empty mapping is the whole-store
    report request and is valid."""
    if mapping is None:
        return []
    if not isinstance(mapping, dict):
        return ["inspect request: the mapping is not a dict"]
    problems = _unfiled_problems("inspect", mapping)
    problems.extend(_string_problems(
        "inspect", mapping,
        ("record_id", "kind", "tag", "status", "text", "profile_id"),
        non_empty=("record_id",)))
    return problems


def prune_request_problems(mapping: Any) -> list[str]:
    """Judge one prune request mapping, purely.
    Empty-safe: an empty mapping is the default
    routine prune and is valid. Shape only — which
    modes exist is the lifecycle's enforcement,
    surfaced as refusal data."""
    if mapping is None:
        return []
    if not isinstance(mapping, dict):
        return ["prune request: the mapping is not a dict"]
    problems = _unfiled_problems("prune", mapping)
    problems.extend(_string_problems(
        "prune", mapping, ("mode", "profile_id"),
        non_empty=("mode",)))
    return problems


def export_request_problems(mapping: Any) -> list[str]:
    """Judge one export request mapping, purely.
    Empty-safe: an empty mapping exports to a
    temporary file and returns the bundle."""
    if mapping is None:
        return []
    if not isinstance(mapping, dict):
        return ["export request: the mapping is not a dict"]
    problems = _unfiled_problems("export", mapping)
    problems.extend(_string_problems(
        "export", mapping, ("destination",)))
    return problems


_REQUEST_JUDGES: dict[str, Callable[[Any], list[str]]] = {
    "add": add_request_problems,
    "inspect": inspect_request_problems,
    "prune": prune_request_problems,
    "export": export_request_problems,
}


def request_problems(op: Any, mapping: Any) -> list[str]:
    """Judge one (op, request) pair, purely. Empty-safe:
    an unknown op is named, never crashed on."""
    if op not in OP_IDS:
        return [f"request: op is {op!r}, not one of {list(OP_IDS)}"]
    return _REQUEST_JUDGES[op](mapping)


def response_problems(row: Any) -> list[str]:
    """Judge one response dict against the envelope,
    purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["response: the row is empty"]
    problems: list[str] = []
    for field in RESPONSE_IDS:
        if field not in row:
            problems.append(f"response: no {field} filed")
    if problems:
        return problems
    if row.get("op") not in OP_IDS:
        problems.append(
            f"response: op is {row.get('op')!r}, not one of "
            f"{list(OP_IDS)}")
    if not isinstance(row.get("ok"), bool):
        problems.append("response: ok is not a boolean")
    if not isinstance(row.get("status"), str) \
            or not row.get("status"):
        problems.append("response: status is not a non-empty string")
    if row.get("contract_version") != API_VERSION:
        problems.append(
            f"response: contract_version is "
            f"{row.get('contract_version')!r}, not TF-034's "
            f"{API_VERSION!r}")
    if row.get("ok") is True and row.get("refusal") is not None:
        problems.append(
            "response: ok is True but a refusal is filed — a "
            "success files refusal None")
    if row.get("ok") is False:
        if row.get("status") != "refused":
            problems.append(
                "response: ok is False but status is "
                f"{row.get('status')!r}, not \"refused\"")
        if not isinstance(row.get("refusal"), str) \
                or not row.get("refusal"):
            problems.append(
                "response: ok is False but no refusal names "
                "the problem")
    try:
        json.dumps(row)
    except (TypeError, ValueError) as exc:
        problems.append(f"response: not JSON-safe: {exc}")
    return problems


# ── The API ──────────────────────────────────────────────────────────────────

def _detached(data: Any) -> Any:
    """A detached JSON-safe copy (a JSON round-trip)."""
    return json.loads(json.dumps(data))


def _envelope(op: str, ok: bool, status: str, **fields: Any) -> dict[str, Any]:
    return _detached({
        "op": op, "ok": ok, "status": status,
        "contract_version": API_VERSION,
        "refusal": None if ok else fields.pop("refusal", ""),
        **fields,
    })


def _refused(op: str, exc: ValueError) -> dict[str, Any]:
    return _envelope(op, False, "refused", refusal=str(exc))


class MemoryAPI:
    """TF-057's store toolset as service calls.

    Wraps one ``MemoryStore``: pass ``store`` to inject
    one, ``store_factory`` to build one lazily, or the
    toolset callables to inject TF-057's functions
    themselves (each takes the store first, exactly as
    ``forge_toolset``'s functions do). With none of
    those, a default ``MemoryStore`` is constructed
    lazily on the first call — constructing this API
    builds no store.
    """

    def __init__(
        self,
        store: Any = None,
        *,
        store_factory: Callable[[], Any] | None = None,
        add_callable: Callable[..., Any] | None = None,
        inspect_store_callable: Callable[..., Any] | None = None,
        inspect_records_callable: Callable[..., Any] | None = None,
        find_callable: Callable[..., Any] | None = None,
        prune_callable: Callable[..., Any] | None = None,
        export_callable: Callable[..., Any] | None = None,
    ) -> None:
        self._store = store
        self._store_factory = store_factory
        self._add = add_callable or add_record
        self._inspect_store = inspect_store_callable or inspect_store
        self._inspect_records = \
            inspect_records_callable or inspect_records
        self._find = find_callable or find_record
        self._prune = prune_callable or prune_store
        self._export = export_callable or export_store_bundle

    def _get_store(self) -> Any:
        if self._store is None:
            if self._store_factory is not None:
                self._store = self._store_factory()
            else:
                from thoughtforge.knowledge.store import MemoryStore

                self._store = MemoryStore()
        return self._store

    # ── The one entry point ──

    def call(
        self,
        op: str,
        request: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """One memory service call: a request as a plain
        dict and/or kwargs, a response as a plain
        JSON-safe detached dict.

        An invalid request raises ValueError naming the
        problem(s), judged by ``request_problems()``
        before any store is built or called. An
        enforcement refusal from TF-057's toolset
        returns an ok=False refusal dict naming it."""
        mapping: dict[str, Any] = {}
        if request is not None:
            if not isinstance(request, dict):
                raise ValueError(
                    "request: the request is not a dict — a "
                    "memory request is a plain mapping of the "
                    "op's filed fields")
            mapping.update(request)
        mapping.update(kwargs)
        judged: Any = mapping if op == "add" or mapping else None
        problems = request_problems(op, judged)
        if problems:
            raise ValueError("; ".join(problems))
        handler = {
            "add": self._call_add,
            "inspect": self._call_inspect,
            "prune": self._call_prune,
            "export": self._call_export,
        }[op]
        return handler(mapping)

    # ── The four ops ──

    def _call_add(self, mapping: dict[str, Any]) -> dict[str, Any]:
        store = self._get_store()
        kwargs = {k: mapping[k] for k in (
            "summary", "tags", "category", "value", "fact_value",
            "pattern_shape", "tone", "importance", "confidence",
            "quality", "record_id") if mapping.get(k) is not None}
        try:
            report = self._add(store, mapping["kind"], **kwargs)
        except ValueError as exc:
            return _refused("add", exc)
        return _envelope("add", True, str(report.get("status", "added")),
                         **{k: v for k, v in report.items()
                            if k != "status"})

    def _call_inspect(self, mapping: dict[str, Any]) -> dict[str, Any]:
        store = self._get_store()
        try:
            if mapping.get("record_id") is not None:
                found = self._find(store, mapping["record_id"])
                status = "found" if found.get("found") else "not_found"
                return _envelope("inspect", True, status, **found)
            filters = {k: mapping[k] for k in (
                "kind", "tag", "status", "text")
                if mapping.get(k) is not None}
            if filters:
                records = self._inspect_records(store, **filters)
                return _envelope("inspect", True, "listed",
                                 records=records, count=len(records))
            report = self._inspect_store(
                store, profile_id=mapping.get("profile_id"))
        except ValueError as exc:
            return _refused("inspect", exc)
        return _envelope("inspect", True, "inspected", **report)

    def _call_prune(self, mapping: dict[str, Any]) -> dict[str, Any]:
        store = self._get_store()
        try:
            report = self._prune(
                store, mode=mapping.get("mode") or "routine",
                profile_id=mapping.get("profile_id"))
        except ValueError as exc:
            return _refused("prune", exc)
        return _envelope("prune", True, str(report.get("status", "pruned")),
                         **{k: v for k, v in report.items()
                            if k != "status"})

    def _call_export(self, mapping: dict[str, Any]) -> dict[str, Any]:
        store = self._get_store()
        destination = mapping.get("destination")
        tmp: tempfile.TemporaryDirectory | None = None
        if destination is None:
            tmp = tempfile.TemporaryDirectory(prefix="tf155-")
            destination = str(Path(tmp.name) / "bundle.json")
        try:
            report = self._export(store, destination)
            bundle = read_bundle(Path(destination))
            problems = verify_bundle(bundle)
            verified = problems == [] and bool(report.get("verified"))
            filed = dict(report)
            filed["verified"] = verified
            filed["problems"] = problems if problems \
                else list(report.get("problems") or [])
            filed["bundle"] = bundle
            return _envelope(
                "export", True, str(report.get("status", "exported")),
                **{k: v for k, v in filed.items() if k != "status"})
        except ValueError as exc:
            return _refused("export", exc)
        finally:
            if tmp is not None:
                tmp.cleanup()

    # ── Convenience methods (kwargs in, dict out) ──

    def add(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Add one record through TF-057's add_record."""
        return self.call("add", {"kind": kind, **fields})

    def inspect(self, **fields: Any) -> dict[str, Any]:
        """Inspect the store / list records / find one
        record, through TF-057's inspect functions."""
        return self.call("inspect", fields or None)

    def prune(self, **fields: Any) -> dict[str, Any]:
        """One prune pass through TF-057's prune_store."""
        return self.call("prune", fields or None)

    def export(self, **fields: Any) -> dict[str, Any]:
        """Export the store through TF-057's
        export_store_bundle; the verified TF-052 bundle
        rides the response as detached data."""
        return self.call("export", fields or None)


# ── Temp-store fixture (checker + probes) ────────────────────────────────────

def _temp_api() -> tuple[MemoryAPI, Any, str]:
    """An API over a fresh temp store. Returns the API,
    the store and the temp home (kept alive by the
    store's directory existing on disk)."""
    from thoughtforge.knowledge.store import MemoryStore

    tmp = tempfile.mkdtemp(prefix="tf155-")
    store = MemoryStore(memory_dir=Path(tmp) / "store")
    return MemoryAPI(store=store), store, tmp


def _seeded_api() -> tuple[MemoryAPI, Any, str]:
    api, store, tmp = _temp_api()
    added = api.add("user_fact", summary="ferry at 07:30",
                    tags=["ferry"], record_id="fct_tf155")
    assert added["ok"] is True, added
    return api, store, tmp


# ── The filed probes ─────────────────────────────────────────────────────────

MEMORY_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "add_lands_stamped_and_findable",
     "kind": "add_round_trip",
     "detail": "an add through the API lands stamped "
               "source='manual_entry', is findable through "
               "the API, and sits in the tag index and the "
               "recent queue"},
    {"id": "inspect_is_read_only",
     "kind": "inspect_read_only",
     "detail": "the store report, the listing and the find "
               "through the API leave the store byte-for-byte "
               "unchanged"},
    {"id": "gate_refusal_surfaces_as_data",
     "kind": "gate_refusal",
     "detail": "a gate-refused add (empty summary) and a "
               "duplicate-id add return ok=False refusal "
               "dicts naming the refusal, with nothing "
               "written"},
    {"id": "export_verified_and_detached",
     "kind": "export_detached",
     "detail": "an export through the API verifies with "
               "verify_bundle() clean and mutating the "
               "returned bundle leaves a re-export unchanged"},
    {"id": "prune_reports_and_refuses_as_data",
     "kind": "prune_round_trip",
     "detail": "a routine prune through the API reports "
               "before/after counts, and an unknown mode "
               "returns an ok=False refusal dict before any "
               "write"},
    {"id": "version_is_tf034_and_toolset_clean",
     "kind": "version_tf034",
     "detail": "the contract version is TF-034's "
               "API_VERSION, verify_api_surface() is clean, "
               "and TF-057's forge_toolset_problems() is "
               "clean"},
    {"id": "invalid_request_raises_named",
     "kind": "invalid_request",
     "detail": "an invalid request (unknown op, unfiled "
               "field, missing kind) raises ValueError "
               "naming the problem"},
)


def run_memory_probes() -> list[dict[str, Any]]:
    """Run the filed probes; each returns its verdict."""
    results: list[dict[str, Any]] = []
    for probe in MEMORY_PROBES:
        kind = probe["kind"]
        try:
            if kind == "add_round_trip":
                api, store, _tmp = _temp_api()
                out = api.add("user_fact", summary="ferry at 07:30",
                              tags=["ferry"], record_id="fct_probe")
                found = api.inspect(record_id="fct_probe")
                passed = out.get("ok") is True \
                    and out.get("status") == "added" \
                    and out.get("record", {}).get("source") \
                    == "manual_entry" \
                    and found.get("status") == "found" \
                    and store.load_tag_index().get("ferry") \
                    == ["fct_probe"] \
                    and "fct_probe" in store.load_recent_queue() \
                    and response_problems(out) == []
            elif kind == "inspect_read_only":
                api, store, _tmp = _seeded_api()
                before = {p.name: p.read_bytes()
                          for p in sorted(store.dir.iterdir())
                          if p.is_file()}
                report = api.inspect()
                listing = api.inspect(tag="ferry")
                found = api.inspect(record_id="fct_tf155")
                after = {p.name: p.read_bytes()
                         for p in sorted(store.dir.iterdir())
                         if p.is_file()}
                passed = before == after \
                    and report.get("total_records") == 1 \
                    and listing.get("count") == 1 \
                    and found.get("status") == "found"
            elif kind == "gate_refusal":
                api, store, _tmp = _temp_api()
                refused = api.add("user_fact", summary="")
                landed = api.add("user_fact", summary="ferry at 07:30",
                                 record_id="fct_probe")
                duplicate = api.add("user_fact", summary="again",
                                    record_id="fct_probe")
                passed = refused.get("ok") is False \
                    and refused.get("status") == "refused" \
                    and bool(refused.get("refusal")) \
                    and landed.get("ok") is True \
                    and duplicate.get("ok") is False \
                    and "already exists" in str(
                        duplicate.get("refusal")) \
                    and len(store.load_profile_records()) == 1 \
                    and response_problems(refused) == [] \
                    and response_problems(duplicate) == []
            elif kind == "export_detached":
                api, _store, tmp = _seeded_api()
                out = api.export(destination=str(
                    Path(tmp) / "bundle.json"))
                bundle = out.get("bundle") or {}
                passed = out.get("ok") is True \
                    and out.get("verified") is True \
                    and verify_bundle(bundle) == [] \
                    and bundle.get("format") == BUNDLE_FORMAT
                if passed:
                    bundle["sections"]["profile_records"][0][
                        "summary"] = "mutated"
                    again = api.export(destination=str(
                        Path(tmp) / "bundle2.json"))
                    passed = again.get("verified") is True \
                        and again["bundle"]["sections"][
                            "profile_records"][0]["summary"] \
                        == "ferry at 07:30"
            elif kind == "prune_round_trip":
                api, store, _tmp = _seeded_api()
                before = {p.name: p.read_bytes()
                          for p in sorted(store.dir.iterdir())
                          if p.is_file()}
                refused = api.prune(mode="bogus_mode")
                after_refusal = {p.name: p.read_bytes()
                                 for p in sorted(store.dir.iterdir())
                                 if p.is_file()}
                out = api.prune(mode="routine")
                passed = refused.get("ok") is False \
                    and "valid modes" in str(refused.get("refusal")) \
                    and before == after_refusal \
                    and out.get("ok") is True \
                    and out.get("status") == "pruned" \
                    and out.get("before") == out.get("after") \
                    and out.get("after", {}).get("user_fact") == 1
            elif kind == "version_tf034":
                passed = memory_contract()["contract_version"] \
                        == API_VERSION \
                    and api_manifest().get("version") == API_VERSION \
                    and verify_api_surface() == [] \
                    and forge_toolset_problems() == []
            elif kind == "invalid_request":
                api, _store, _tmp = _temp_api()
                named: list[bool] = []
                for call, needle in (
                        (lambda: api.call("explode"), "explode"),
                        (lambda: api.add("user_fact", summary="x",
                                         surprise=1), "surprise"),
                        (lambda: api.call("add", {}), "kind")):
                    try:
                        call()
                        named.append(False)
                    except ValueError as exc:
                        named.append(needle in str(exc))
                passed = all(named)
            else:  # pragma: no cover — the checker names unknown kinds
                passed = False
        except Exception:
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the memory-api doc, verbatim) ───────────────────────

def render_memory_api_markdown() -> str:
    """Render the memory-api doc deterministically
    — the exact text filed as
    docs/architecture/MEMORY_API.md."""
    lines: list[str] = []
    lines.append("# Memory API — Store Operations as "
                 "Service Calls (TF-155)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/memory_api.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.memory_api`. The "
        "checker (`memory_api_problems()`) re-derives "
        "every claim live from the tree and fails on any "
        "drift.")
    lines.append("")
    lines.append(f"Format: `{MEMORY_API_FORMAT}`")
    lines.append(
        f"Contract version: `{API_VERSION}` (TF-034's "
        "`API_VERSION` — consumed, not reinvented)")
    lines.append(
        f"Bundle format: `{BUNDLE_FORMAT}` (TF-052's "
        "`BUNDLE_FORMAT` — consumed, not reinvented)")
    lines.append(f"Ops: {len(MEMORY_OPS)}; request fields: "
                 f"{len(MEMORY_REQUEST_FIELDS)}; response "
                 f"fields: {len(MEMORY_RESPONSE_FIELDS)}; "
                 f"rules: {len(MEMORY_RULES)}")
    lines.append(f"Probes: {len(MEMORY_PROBES)}/{len(MEMORY_PROBES)}")
    lines.append("")
    lines.append("## 1. One call")
    lines.append("")
    lines.append(
        "`MemoryAPI(store).call(op, request)` — or the "
        "kwargs convenience methods `add(kind, ...)`, "
        "`inspect(...)`, `prune(...)`, `export(...)` — "
        "runs one of TF-057's store operations and "
        "returns the response as a plain JSON-safe "
        "detached dict. The API calls TF-057's own "
        "functions in "
        "`thoughtforge.knowledge.forge_toolset`; it "
        "reimplements no store logic, so the enforcement "
        "is TF-057's by construction. An invalid request "
        "raises `ValueError` naming the problem(s), "
        "judged by the pure request validators before "
        "any store is built or called — a library-layer "
        "caller error, never silently coerced. An "
        "enforcement refusal from the toolset returns "
        "an `ok=False`, `status=\"refused\"` response "
        "whose `refusal` names the toolset's own "
        "message — data, never a silent success. The "
        "store is built lazily: constructing "
        "`MemoryAPI()` builds no store until a call, "
        "and a store / store factory / toolset callables "
        "can be injected (TF-154's injection shape).")
    lines.append("")
    lines.append("## 2. The ops (TF-057's toolset)")
    lines.append("")
    lines.append("| Op | TF-057 operation | Toolset function(s) | "
                 "Writes the store | What it does |")
    lines.append("|---|---|---|---|---|")
    for op in MEMORY_OPS:
        writes = "yes" if op["writes_store"] else "no"
        functions = ", ".join(f"`{fn}`" for fn in op["functions"])
        lines.append(f"| {op['id']} | {op['toolset_operation']} | "
                     f"{functions} | {writes} | {op['summary']} |")
    lines.append("")
    lines.append("## 3. Request fields")
    lines.append("")
    lines.append("| Op | Field | Required | What it is |")
    lines.append("|---|---|---|---|")
    for field in MEMORY_REQUEST_FIELDS:
        required = "yes" if field["required"] else "no"
        lines.append(f"| {field['op']} | {field['id']} | "
                     f"{required} | {field['summary']} |")
    lines.append("")
    lines.append(
        "Request validation is shape only: which kinds "
        "exist, whether a record passes the TF-042 gate, "
        "which prune modes exist — those are TF-057's "
        "enforcement and surface as refusal data, never "
        "as shape verdicts.")
    lines.append("")
    lines.append("## 4. The response")
    lines.append("")
    lines.append("| Field | Source | What it is |")
    lines.append("|---|---|---|")
    for field in MEMORY_RESPONSE_FIELDS:
        lines.append(f"| {field['id']} | {field['source']} | "
                     f"{field['summary']} |")
    lines.append("")
    lines.append(
        "On success the toolset report's own fields ride "
        "the envelope flattened in (an add's `kind` / "
        "`record_id` / `source` / `record`, a store "
        "report's counts and bytes, a listing's `records` "
        "and `count`, a prune's `counts` / `before` / "
        "`after`). The export response additionally "
        "carries `path`, `format`, `version`, `counts`, "
        "`total_records`, `verified`, `problems` and "
        "`bundle` — the TF-052 bundle itself, written "
        "through TF-057's `export_store_bundle`, read "
        "back with `read_bundle` and re-verified with "
        "`verify_bundle`, exported as detached JSON-safe "
        "data (a JSON round-trip copy): mutating a "
        "response's bundle cannot touch the store.")
    lines.append("")
    lines.append("## 5. The rules")
    lines.append("")
    for rule in MEMORY_RULES:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## 6. Docket")
    lines.append("")
    lines.append(
        "TF-155 has **no share** of TF-016 and found no "
        "new defect in existing code, so this slice does "
        "not edit the docket and adds no entry. TF-016 "
        "totals stand unchanged (137 entries — 32 OPEN / "
        "1 OPEN-ENVIRONMENT / 100 FIXED / 4 "
        "CLOSED-BY-VERDICT), filed in the open here.")
    lines.append("")
    lines.append("## 7. The probes")
    lines.append("")
    for probe in MEMORY_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 8. Filed limits")
    lines.append("")
    lines.append(
        "- Proof grade FIXTURE: the checker and probes "
        "run synthetic temp stores on this VM. Behaviour "
        "over a real, lived-in store is NOT PROVEN (no "
        "real store exists on this machine, TF-041) and "
        "any live model behaviour is NOT PROVEN "
        "(TF-173 class), never claimed.")
    lines.append(
        "- TF-156 owns the Rúnmál binding; TF-162 owns "
        "versioning enforcement in depth. TF-155 files "
        "the four store ops as service calls over "
        "TF-057's toolset, with TF-034's version "
        "consumed.")
    lines.append(
        "- TF-153's service shape can carry this API, "
        "but wiring the service ops to it is NOT done "
        "here: `service_shape.py` is not modified by "
        "this slice, so the service and this API stand "
        "as separate filed shapes until a later slice "
        "wires them.")
    lines.append(
        "- No canon data was touched (TF-017); no "
        "README.md anywhere was touched; no Docker was "
        "used.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_memory_api_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / MEMORY_API_DOC
    doc.write_text(render_memory_api_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.memory_api`."""
    doc = write_memory_api_doc()
    print(f"memory api doc regenerated: -> {MEMORY_API_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def memory_api_problems() -> list[str]:
    """Re-derive TF-155's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    if MEMORY_API_FORMAT != "thoughtforge-memory-api/1":
        problems.append(
            f"the format is {MEMORY_API_FORMAT!r}, not "
            "thoughtforge-memory-api/1")
    op_ids = [op["id"] for op in MEMORY_OPS]
    if op_ids != ["add", "inspect", "prune", "export"] \
            or len(op_ids) != len(set(op_ids)):
        problems.append(
            f"the MEMORY_OPS filing is not add/inspect/prune/"
            f"export: {op_ids}")
    for op in MEMORY_OPS:
        problems.extend(memory_op_problems(op))
    toolset = {r["name"]: r for r in TOOLSET_OPERATIONS}
    for op in MEMORY_OPS:
        row = toolset.get(op["toolset_operation"])
        if row is None:
            problems.append(
                f"op {op['id']}: TF-057 files no operation "
                f"{op['toolset_operation']!r}")
        elif row["writes_store"] != op["writes_store"]:
            problems.append(
                f"op {op['id']}: writes_store is "
                f"{op['writes_store']}, not TF-057's "
                f"{row['writes_store']}")
    for field in MEMORY_REQUEST_FIELDS:
        problems.extend(request_field_problems(field))
    for op_id, expected in (
            ("add", ["kind", "summary", "tags", "category",
                     "value", "fact_value", "pattern_shape",
                     "tone", "importance", "confidence",
                     "quality", "record_id"]),
            ("inspect", ["record_id", "kind", "tag", "status",
                         "text", "profile_id"]),
            ("prune", ["mode", "profile_id"]),
            ("export", ["destination"])):
        if list(request_field_ids(op_id)) != expected:
            problems.append(
                f"the {op_id} request-field filing is not as "
                f"filed: {list(request_field_ids(op_id))}")
    kind_field = next(
        (f for f in MEMORY_REQUEST_FIELDS
         if f["op"] == "add" and f["id"] == "kind"), {})
    if kind_field.get("required") is not True:
        problems.append("add request: kind is not filed as required")
    for field in MEMORY_RESPONSE_FIELDS:
        if not field.get("id") or not field.get("source") \
                or not field.get("summary"):
            problems.append(f"response field filing broken: {field}")
    rule_ids = [r["id"] for r in MEMORY_RULES]
    if len(rule_ids) != len(set(rule_ids)) or len(MEMORY_RULES) != 6:
        problems.append("the MEMORY_RULES filing is not six unique rules")
    for rule in MEMORY_RULES:
        problems.extend(memory_rule_problems(rule))
    probe_ids = [p["id"] for p in MEMORY_PROBES]
    if len(probe_ids) != len(set(probe_ids)) or len(MEMORY_PROBES) != 7:
        problems.append("the MEMORY_PROBES filing is not seven unique probes")

    # The version is TF-034's, and its surface is clean.
    contract = memory_contract()
    if contract["contract_version"] != API_VERSION \
            or API_VERSION != api_manifest().get("version"):
        problems.append(
            "the filed contract_version is not TF-034's API_VERSION")
    if contract["bundle_format"] != BUNDLE_FORMAT:
        problems.append("the filed bundle_format is not TF-052's BUNDLE_FORMAT")
    surface = verify_api_surface()
    if surface:
        problems.append(
            f"TF-034: verify_api_surface() is not clean: {surface[:3]}")

    # TF-057's own checker is clean — the enforcement
    # this API borrows still holds underneath it.
    toolset_problems = forge_toolset_problems()
    if toolset_problems:
        problems.append(
            f"TF-057: forge_toolset_problems() is not clean: "
            f"{toolset_problems[:3]}")

    # A real round-trip on a temp store, through the
    # API only: add → inspect/find/list → export
    # verified → prune; refusals surface as data.
    try:
        api, store, tmp = _temp_api()
        added = api.add("user_fact", summary="ferry at 07:30",
                        tags=["ferry"], record_id="fct_tf155")
        problems.extend(
            f"round-trip add: {p}" for p in response_problems(added))
        if added.get("ok") is not True \
                or added.get("record", {}).get("source") != "manual_entry":
            problems.append(f"round-trip add is not a stamped add: {added}")
        refused = api.add("user_fact", summary="")
        if refused.get("ok") is not False \
                or refused.get("status") != "refused" \
                or len(store.load_profile_records()) != 1:
            problems.append(
                f"round-trip: the gate refusal did not surface "
                f"as data with nothing written: {refused}")
        report = api.inspect()
        if report.get("total_records") != 1:
            problems.append(f"round-trip inspect report: {report}")
        found = api.inspect(record_id="fct_tf155")
        if found.get("status") != "found" \
                or found.get("kind") != "user_fact":
            problems.append(f"round-trip find: {found}")
        listing = api.inspect(tag="ferry")
        if listing.get("count") != 1:
            problems.append(f"round-trip listing: {listing}")
        exported = api.export(
            destination=str(Path(tmp) / "bundle.json"))
        problems.extend(
            f"round-trip export: {p}"
            for p in response_problems(exported))
        if exported.get("verified") is not True \
                or verify_bundle(exported.get("bundle")) != []:
            problems.append(
                f"round-trip export is not verified: "
                f"{exported.get('verified')} / "
                f"{exported.get('problems')}")
        pruned = api.prune(mode="routine")
        if pruned.get("ok") is not True \
                or pruned.get("after", {}).get("user_fact") != 1:
            problems.append(f"round-trip prune: {pruned}")
    except Exception as exc:
        problems.append(f"round-trip failed: {exc}")

    # The probes all behave.
    probes = run_memory_probes()
    if len(probes) != len(MEMORY_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the memory probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The docket stands unchanged.
    import re

    docket_text = _DOCKET.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", docket_text)
    if "137 entries — 32 OPEN, 1 OPEN-ENVIRONMENT, 100 FIXED" \
            not in normalized:
        problems.append(
            "docket: the header totals no longer file 137 "
            "entries — 32 OPEN / 1 OPEN-ENVIRONMENT / 100 "
            "FIXED; TF-155 must not move them")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / MEMORY_API_DOC
    if not doc.exists():
        problems.append(f"the memory doc {MEMORY_API_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != render_memory_api_markdown():
        problems.append(
            f"the filed doc {MEMORY_API_DOC} is not the render "
            "verbatim (regenerate with python3 -m "
            "thoughtforge.memory_api)")
    return problems


def memory_api_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_memory_probes()
    return {
        "format": MEMORY_API_FORMAT,
        "doc": MEMORY_API_DOC,
        "contract": memory_contract(),
        "manifest": api_manifest(),
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "MEMORY_API_DOC",
    "MEMORY_API_FORMAT",
    "MEMORY_OPS",
    "MEMORY_PROBES",
    "MEMORY_REQUEST_FIELDS",
    "MEMORY_RESPONSE_FIELDS",
    "MEMORY_RULES",
    "OP_IDS",
    "RESPONSE_IDS",
    "MemoryAPI",
    "add_request_problems",
    "export_request_problems",
    "inspect_request_problems",
    "main",
    "memory_api_as_data",
    "memory_api_problems",
    "memory_contract",
    "memory_op_problems",
    "memory_rule_problems",
    "prune_request_problems",
    "render_memory_api_markdown",
    "request_field_ids",
    "request_field_problems",
    "request_fields_for",
    "request_problems",
    "response_problems",
    "run_memory_probes",
    "write_memory_api_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
