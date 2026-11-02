"""
TF-154 — the cognition API, filed: turn-in /
turn-out with traces (TF-084), versioned
(TF-034 consumed).

TF-154's roadmap text: *The cognition API:
turn-in/turn-out with traces (TF-084),
versioned (TF-034 consumed).* It is Part IX
slice 2, following TF-153's service shape.

**What this slice files**

  * ``CognitionAPI.turn()`` — one cognition
    turn as plain JSON-safe data. Turn-in is
    ``user_text`` (required, a non-empty
    string) plus the optional
    ``retrieval_path`` / ``num_drafts`` /
    ``history`` / ``want_trace`` (default
    True), filed as ``TURN_IN_FIELDS`` literal
    data. ``num_drafts`` follows TF-034's
    ruling: when present it must be >= 1.
    Only the kwargs a caller actually set
    (a non-None value) are passed to
    ``ThoughtForgeCore.think()``, whose frozen
    params are ``user_text`` /
    ``retrieval_path`` / ``num_drafts`` /
    ``history`` / ``interrupt`` (TF-034).
  * Turn-out, filed as ``TURN_OUT_FIELDS``
    literal data: ``text``, ``response_id``,
    ``turn_id``, ``mode``,
    ``enforcement_passed``, ``citations`` and
    ``token_count`` — read from the real
    ``FinalResponseRecord`` the core returns
    (TF-153's ``_record_fields`` precedent,
    replicated faithfully here) — plus
    ``contract_version``, ``trace``,
    ``trace_format`` and ``trace_status``.
  * The version is consumed from TF-034
    (``thoughtforge.cognition.api_surface``'s
    ``API_VERSION`` / ``api_manifest()`` /
    ``verify_api_surface()``) — this module
    invents no competing version.
    ``contract_version`` on every turn-out is
    TF-034's ``API_VERSION``.
  * The trace is TF-084's joined trace: after
    a turn on a real core it is obtained from
    the core's ``last_turn_joined_trace()``;
    its format is
    ``thoughtforge.inference.joined_trace``'s
    ``JOINED_FORMAT`` (imported, never a
    competing literal) and it is validated
    live with ``joined_trace_problems()``.
    A trace that fails its checker is never
    filed as good: the turn-out files
    ``trace_status`` naming the problem
    instead, and no trace. ``trace_status``
    is one of ``"included"``,
    ``"not_requested"`` (``want_trace=False``)
    or ``"unavailable: <reason>"`` (no trace
    source, the core returned None, or the
    trace failed its checker — the reason is
    named). The exported trace is detached
    JSON-safe data (a JSON round-trip copy),
    so mutating a turn-out's trace cannot
    touch the core's retained state.
  * ``CognitionAPI`` — an injectable core /
    core factory / turn callable plus a trace
    callable, mirroring
    ``ThoughtForgeService``'s injection shape
    in TF-153's ``service_shape.py``. With a
    real core, the trace comes from
    ``core.last_turn_joined_trace()``; with
    an injected turn callable and no trace
    callable, ``trace_status`` files
    unavailable honestly. The core is built
    lazily: constructing ``CognitionAPI()``
    builds no core until a turn.
  * An invalid turn-in (judged by the pure
    validator ``turn_in_problems()``) raises
    ``ValueError`` naming the problem(s) —
    a library-layer caller error, never
    silently coerced.
  * ``COGNITION_RULES`` — the API's law as
    literal data, and
    ``cognition_api_problems()`` — a checker
    that re-measures the live tree: the
    filings, the TF-034 surface, a
    knowledge-only core turn through the API
    (the TF-084 fixture shape:
    ``ThoughtForgeCore(memory_dir=tmp/memory,
    db_path=tmp/nonexistent.db,
    model_path=None)``) returning a clean
    turn-out with mode ``knowledge_only`` and
    its trace included and clean, the filed
    probes, the docket header and the filed
    doc.

**Docket**: TF-154 has no share of TF-016 and
finds no new defect in existing code; it does
NOT edit ``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md``.
TF-016 totals stand unchanged (137 entries —
32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4
CLOSED-BY-VERDICT), filed in the open here and
in the render.

The human-readable doc is
``docs/architecture/COGNITION_API.md``, filed
verbatim as this module's render (regenerate
with ``python3 -m thoughtforge.cognition_api``).
Proof grade: FIXTURE — a knowledge-only core
on this VM; a live model turn through the
cognition API is NOT PROVEN, never claimed.
TF-155 owns the memory API; TF-162 owns
versioning enforcement in depth. TF-153's
service shape can carry this API, but wiring
the service turn op to it is NOT done here —
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
from thoughtforge.inference.joined_trace import (
    JOINED_FORMAT,
    joined_trace_problems,
)

COGNITION_API_FORMAT = "thoughtforge-cognition-api/1"
COGNITION_API_DOC = "docs/architecture/COGNITION_API.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"


# ── Turn-in, as literal data ─────────────────────────────────────────────────

TURN_IN_FIELDS: tuple[dict[str, Any], ...] = (
    {"id": "user_text",
     "required": True,
     "summary": "the user's text for this turn — a "
                "non-empty string, required"},
    {"id": "retrieval_path",
     "required": False,
     "summary": "optional retrieval path override "
                "(\"sql\" | \"vector\" | \"hybrid\"), "
                "passed to think() only when set"},
    {"id": "num_drafts",
     "required": False,
     "summary": "optional cap on candidate drafts; "
                ">= 1 when present, per TF-034's ruling "
                "(a value below 1 is a caller error)"},
    {"id": "history",
     "required": False,
     "summary": "optional conversation history, passed "
                "to think() only when set"},
    {"id": "want_trace",
     "required": False,
     "default": True,
     "summary": "optional bool, default True: whether "
                "the turn-out should carry TF-084's "
                "joined trace for the turn"},
)

TURN_IN_IDS: tuple[str, ...] = tuple(f["id"] for f in TURN_IN_FIELDS)


# ── Turn-out, as literal data ────────────────────────────────────────────────

TURN_OUT_FIELDS: tuple[dict[str, Any], ...] = (
    {"id": "text",
     "source": "FinalResponseRecord.text",
     "summary": "the turn's response text"},
    {"id": "response_id",
     "source": "FinalResponseRecord.response_id",
     "summary": "the final response's id"},
    {"id": "turn_id",
     "source": "FinalResponseRecord.turn_id",
     "summary": "the turn's id — the included trace's "
                "turn_id must agree with it"},
    {"id": "mode",
     "source": "FinalResponseRecord.mode",
     "summary": "the turn's mode (knowledge_only when "
                "no model served)"},
    {"id": "enforcement_passed",
     "source": "FinalResponseRecord.enforcement_passed",
     "summary": "the enforcement verdict on the final "
                "response"},
    {"id": "citations",
     "source": "FinalResponseRecord.citations",
     "summary": "the final response's citations, as a "
                "list of strings"},
    {"id": "token_count",
     "source": "FinalResponseRecord.token_count",
     "summary": "the final response's token count"},
    {"id": "contract_version",
     "source": "TF-034 API_VERSION",
     "summary": "the contract version — TF-034's "
                "API_VERSION, consumed, not reinvented"},
    {"id": "trace",
     "source": "TF-084 joined trace",
     "summary": "TF-084's joined trace for the turn as "
                "detached JSON-safe data, or None when "
                "trace_status is not \"included\""},
    {"id": "trace_format",
     "source": "TF-084 JOINED_FORMAT",
     "summary": "the included trace's format (TF-084's "
                "JOINED_FORMAT), or None when no trace "
                "is included"},
    {"id": "trace_status",
     "source": "this API",
     "summary": "\"included\", \"not_requested\" "
                "(want_trace=False), or "
                "\"unavailable: <reason>\" naming why "
                "no trace is filed"},
)

TURN_OUT_IDS: tuple[str, ...] = tuple(f["id"] for f in TURN_OUT_FIELDS)

#: The trace_status values that file no reason.
TRACE_STATUSES: tuple[str, ...] = ("included", "not_requested")


# ── The rules, as literal data ───────────────────────────────────────────────

COGNITION_RULES: tuple[dict[str, Any], ...] = (
    {"id": "version_is_tf034",
     "rule": "the contract_version on every turn-out is "
             "TF-034's API_VERSION from "
             "thoughtforge.cognition.api_surface — no "
             "competing version is invented here "
             "(TF-162 owns versioning enforcement in "
             "depth)"},
    {"id": "turn_out_fields_from_final_response",
     "rule": "the turn-out's response fields are the "
             "real FinalResponseRecord's: text, "
             "response_id, turn_id, mode, "
             "enforcement_passed, citations and "
             "token_count — filed from the record the "
             "core actually returns"},
    {"id": "trace_is_tf084_joined",
     "rule": "the trace is TF-084's joined trace, in "
             "its JOINED_FORMAT, validated live by its "
             "own checker joined_trace_problems(); the "
             "trace's turn_id must agree with the "
             "turn-out's turn_id"},
    {"id": "trace_is_detached_json_safe",
     "rule": "the exported trace is detached JSON-safe "
             "data (a JSON round-trip copy): mutating a "
             "turn-out's trace cannot touch the core's "
             "retained state"},
    {"id": "no_trace_is_filed_never_faked",
     "rule": "when no trace is filed, trace_status "
             "names why — \"not_requested\" when "
             "want_trace=False, \"unavailable: "
             "<reason>\" when there is no trace source, "
             "the core returned none, or the trace "
             "failed its checker; a bad trace is never "
             "filed as good and no trace is ever faked"},
    {"id": "invalid_turn_in_raises_named",
     "rule": "an invalid turn-in raises ValueError "
             "naming the problem(s) judged by "
             "turn_in_problems() — a library-layer "
             "caller error, never silently coerced"},
)


# ── The contract as plain data ───────────────────────────────────────────────

def cognition_contract() -> dict[str, Any]:
    """The cognition API contract as plain JSON-safe
    data, built from the filed literals above and
    TF-034's version. A fresh copy each call."""
    return {
        "format": COGNITION_API_FORMAT,
        "contract_version": API_VERSION,
        "trace_format": JOINED_FORMAT,
        "turn_in_fields": [dict(f) for f in TURN_IN_FIELDS],
        "turn_out_fields": [dict(f) for f in TURN_OUT_FIELDS],
        "rules": [dict(r) for r in COGNITION_RULES],
    }


# ── Pure validators ──────────────────────────────────────────────────────────

def turn_field_problems(row: Any) -> list[str]:
    """Judge one turn field filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["field: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("field: the row files no id")
    if not row.get("summary"):
        problems.append(f"field {row.get('id')}: no summary filed")
    if "required" in row and not isinstance(row.get("required"), bool):
        problems.append(
            f"field {row.get('id')}: required is not a boolean")
    if "source" in row and not row.get("source"):
        problems.append(f"field {row.get('id')}: no source filed")
    return problems


def cognition_rule_problems(row: Any) -> list[str]:
    """Judge one rule filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["rule: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("rule: the row files no id")
    if not row.get("rule"):
        problems.append(f"rule {row.get('id')}: no rule filed")
    return problems


def turn_in_problems(mapping: Any) -> list[str]:
    """Judge one turn-in mapping, purely. Empty-safe.

    The judge ``CognitionAPI.turn()`` raises by:
    ``user_text`` is required as a non-empty string;
    ``num_drafts``, when present, must be an integer
    >= 1 (TF-034's ruling); ``want_trace``, when
    present, must be a bool; an unfiled field is
    named, never silently carried."""
    if not isinstance(mapping, dict) or not mapping:
        return ["turn-in: the mapping is empty"]
    problems: list[str] = []
    for key in mapping:
        if key not in TURN_IN_IDS:
            problems.append(
                f"turn-in: {key!r} is not a filed turn-in "
                f"field — the filed fields are {list(TURN_IN_IDS)}")
    user_text = mapping.get("user_text")
    if not isinstance(user_text, str) or not user_text:
        problems.append(
            "turn-in: user_text is required as a non-empty "
            "string")
    if "retrieval_path" in mapping \
            and mapping["retrieval_path"] is not None \
            and not isinstance(mapping["retrieval_path"], str):
        problems.append(
            "turn-in: retrieval_path must be a string when "
            f"present, not {type(mapping['retrieval_path']).__name__}")
    if "num_drafts" in mapping \
            and mapping["num_drafts"] is not None:
        num_drafts = mapping["num_drafts"]
        if isinstance(num_drafts, bool) \
                or not isinstance(num_drafts, int):
            problems.append(
                "turn-in: num_drafts must be an integer "
                f">= 1 when present, not {num_drafts!r}")
        elif num_drafts < 1:
            problems.append(
                "turn-in: num_drafts must be >= 1 when "
                f"present (TF-034's ruling), not {num_drafts}")
    if "want_trace" in mapping \
            and not isinstance(mapping["want_trace"], bool):
        problems.append(
            "turn-in: want_trace must be a bool when "
            f"present, not {mapping['want_trace']!r}")
    return problems


def turn_out_problems(row: Any) -> list[str]:
    """Judge one turn-out dict against the contract,
    purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["turn-out: the row is empty"]
    problems: list[str] = []
    for field in TURN_OUT_IDS:
        if field not in row:
            problems.append(f"turn-out: no {field} filed")
    if problems:
        return problems
    if row.get("contract_version") != API_VERSION:
        problems.append(
            f"turn-out: contract_version is "
            f"{row.get('contract_version')!r}, not TF-034's "
            f"{API_VERSION!r}")
    if not isinstance(row.get("text"), str):
        problems.append("turn-out: text is not a string")
    for key in ("response_id", "turn_id", "mode"):
        if not isinstance(row.get(key), str):
            problems.append(f"turn-out: {key} is not a string")
    if not isinstance(row.get("enforcement_passed"), bool):
        problems.append("turn-out: enforcement_passed is not a boolean")
    if not isinstance(row.get("citations"), list):
        problems.append("turn-out: citations is not a list")
    token_count = row.get("token_count")
    if isinstance(token_count, bool) \
            or not isinstance(token_count, int) or token_count < 0:
        problems.append("turn-out: token_count is not an integer >= 0")
    status = row.get("trace_status")
    if status not in TRACE_STATUSES \
            and not (isinstance(status, str)
                     and status.startswith("unavailable: ")
                     and status[len("unavailable: "):].strip()):
        problems.append(
            f"turn-out: trace_status is {status!r}, not one of "
            "\"included\" / \"not_requested\" / "
            "\"unavailable: <reason>\"")
    trace = row.get("trace")
    if status == "included" and trace is None:
        problems.append(
            "turn-out: trace_status is \"included\" but no "
            "trace is filed")
    if status != "included" and trace is not None:
        problems.append(
            f"turn-out: trace_status is {status!r} but a "
            "trace is filed — a trace is filed only as "
            "\"included\"")
    if trace is None:
        if row.get("trace_format") is not None:
            problems.append(
                "turn-out: no trace is filed but trace_format "
                f"is {row.get('trace_format')!r}, not None")
    else:
        if row.get("trace_format") != JOINED_FORMAT:
            problems.append(
                f"turn-out: trace_format is "
                f"{row.get('trace_format')!r}, not TF-084's "
                f"{JOINED_FORMAT!r}")
        if not isinstance(trace, dict):
            problems.append("turn-out: the trace is not a mapping")
        else:
            if trace.get("format") != JOINED_FORMAT:
                problems.append(
                    f"turn-out: the trace's format is "
                    f"{trace.get('format')!r}, not {JOINED_FORMAT!r}")
            problems.extend(
                f"turn-out trace: {p}"
                for p in joined_trace_problems(trace))
            if trace.get("turn_id") != row.get("turn_id"):
                problems.append(
                    f"turn-out: the trace's turn_id is "
                    f"{trace.get('turn_id')!r}, not the "
                    f"turn-out's turn_id {row.get('turn_id')!r}")
    try:
        json.dumps(row)
    except (TypeError, ValueError) as exc:
        problems.append(f"turn-out: not JSON-safe: {exc}")
    return problems


# ── The API ──────────────────────────────────────────────────────────────────

def _record_fields(record: Any) -> dict[str, Any]:
    """The turn-out response fields, read from a real
    FinalResponseRecord (or a mapping with the same
    fields, as an injected turn callable may return).
    TF-153's ``_record_fields`` precedent, replicated
    faithfully."""
    def get(name: str, default: Any = "") -> Any:
        if isinstance(record, dict):
            return record.get(name, default)
        return getattr(record, name, default)

    text = get("text", get("response_text", ""))
    citations = get("citations", [])
    if not isinstance(citations, list):
        citations = [str(citations)]
    return {
        "text": str(text),
        "response_id": str(get("response_id")),
        "turn_id": str(get("turn_id")),
        "mode": str(get("mode")),
        "enforcement_passed": bool(get("enforcement_passed", False)),
        "citations": [str(c) for c in citations],
        "token_count": int(get("token_count", 0) or 0),
    }


class CognitionAPI:
    """One cognition turn in / out, with TF-084 traces.

    Wraps a ThoughtForgeCore: pass ``core`` to inject
    one, ``core_factory`` to build one lazily, or
    ``turn_callable`` to inject the turn itself (a
    callable taking ``user_text`` and returning a
    FinalResponseRecord or a mapping with its fields).
    Pass ``trace_callable`` to inject the trace source
    (a callable taking no arguments and returning a
    TF-084 joined trace or None). With none of those,
    a default ThoughtForgeCore is constructed lazily
    on the first turn — constructing this API builds
    no core.
    """

    def __init__(
        self,
        core: Any = None,
        *,
        core_factory: Callable[[], Any] | None = None,
        turn_callable: Callable[..., Any] | None = None,
        trace_callable: Callable[[], Any] | None = None,
    ) -> None:
        self._core = core
        self._core_factory = core_factory
        self._turn_callable = turn_callable
        self._trace_callable = trace_callable

    def _get_core(self) -> Any:
        if self._core is None:
            if self._core_factory is not None:
                self._core = self._core_factory()
            else:
                from thoughtforge.cognition.core import ThoughtForgeCore

                self._core = ThoughtForgeCore()
        return self._core

    def _obtain_trace(
        self, used_callable: bool, turn_id: str,
    ) -> tuple[Any, Any, str]:
        """The turn's trace, its format and its status.

        Returns (trace, trace_format, trace_status):
        the trace is a detached JSON-safe copy only
        when the status is "included"; otherwise the
        trace and format are None and the status names
        why."""
        if self._trace_callable is not None:
            try:
                raw = self._trace_callable()
            except Exception as exc:
                return None, None, (
                    f"unavailable: the trace source raised "
                    f"{type(exc).__name__}: {exc}")
            if raw is None:
                return None, None, (
                    "unavailable: the trace source returned "
                    "no trace for this turn")
        elif used_callable:
            return None, None, (
                "unavailable: no trace source — an injected "
                "turn callable files no joined trace and no "
                "trace_callable was supplied")
        else:
            core = self._core
            getter = getattr(core, "last_turn_joined_trace", None)
            if not callable(getter):
                return None, None, (
                    "unavailable: the core provides no "
                    "last_turn_joined_trace() trace source")
            try:
                raw = getter()
            except Exception as exc:
                return None, None, (
                    f"unavailable: the core's "
                    f"last_turn_joined_trace() raised "
                    f"{type(exc).__name__}: {exc}")
            if raw is None:
                return None, None, (
                    "unavailable: the core returned no joined "
                    "trace for this turn "
                    "(last_turn_joined_trace() was None)")
        if not isinstance(raw, dict):
            return None, None, (
                f"unavailable: the trace source returned "
                f"{type(raw).__name__}, not a joined trace "
                "mapping")
        problems = list(joined_trace_problems(raw))
        if raw.get("turn_id") != turn_id:
            problems.append(
                f"trace turn_id {raw.get('turn_id')!r} disagrees "
                f"with the turn-out's turn_id {turn_id!r}")
        if problems:
            return None, None, (
                "unavailable: the joined trace failed its "
                f"checker: {'; '.join(problems)}")
        try:
            detached = json.loads(json.dumps(raw))
        except (TypeError, ValueError) as exc:
            return None, None, (
                f"unavailable: the joined trace is not "
                f"JSON-safe: {exc}")
        if joined_trace_problems(detached):
            return None, None, (
                "unavailable: the joined trace failed its "
                "checker after detachment")
        return detached, JOINED_FORMAT, "included"

    def turn(
        self,
        user_text: str,
        retrieval_path: str | None = None,
        num_drafts: int | None = None,
        history: Any = None,
        want_trace: bool = True,
    ) -> dict[str, Any]:
        """One cognition turn in / out.

        An invalid turn-in raises ValueError naming the
        problem(s) — a library-layer caller error,
        judged by ``turn_in_problems()`` before any
        core is built or called. Only the kwargs the
        caller actually set (a non-None value) are
        passed to ``core.think()``."""
        mapping: dict[str, Any] = {"user_text": user_text,
                                   "want_trace": want_trace}
        if retrieval_path is not None:
            mapping["retrieval_path"] = retrieval_path
        if num_drafts is not None:
            mapping["num_drafts"] = num_drafts
        if history is not None:
            mapping["history"] = history
        problems = turn_in_problems(mapping)
        if problems:
            raise ValueError("; ".join(problems))
        if self._turn_callable is not None:
            record = self._turn_callable(user_text)
            used_callable = True
        else:
            core = self._get_core()
            kwargs: dict[str, Any] = {}
            for key in ("retrieval_path", "num_drafts", "history"):
                if key in mapping:
                    kwargs[key] = mapping[key]
            record = core.think(user_text, **kwargs)
            used_callable = False
        fields = _record_fields(record)
        if not want_trace:
            trace, trace_format, trace_status = None, None, "not_requested"
        else:
            trace, trace_format, trace_status = self._obtain_trace(
                used_callable, fields["turn_id"])
        return {
            "text": fields["text"],
            "response_id": fields["response_id"],
            "turn_id": fields["turn_id"],
            "mode": fields["mode"],
            "enforcement_passed": fields["enforcement_passed"],
            "citations": fields["citations"],
            "token_count": fields["token_count"],
            "contract_version": API_VERSION,
            "trace": trace,
            "trace_format": trace_format,
            "trace_status": trace_status,
        }


# ── Knowledge-only fixture (checker + probes) ────────────────────────────────

def _knowledge_only_api() -> tuple[CognitionAPI, Any]:
    """An API over a knowledge-only core — the TF-084
    fixture shape, in a fresh temp home. Returns the
    API and its core (the core is kept for trace
    re-export checks)."""
    from thoughtforge.cognition.core import ThoughtForgeCore

    tmp = tempfile.mkdtemp(prefix="tf154-")
    core = ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    return CognitionAPI(core=core), core


def _stub_record(user_text: str) -> dict[str, Any]:
    return {
        "text": f"stub answer to: {user_text}",
        "response_id": "resp_stub", "turn_id": "turn_stub",
        "mode": "stub_mode", "enforcement_passed": True,
        "citations": ["frag-1"], "token_count": 4,
    }


# ── The filed probes ─────────────────────────────────────────────────────────

COGNITION_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "knowledge_turn_with_trace_included",
     "kind": "knowledge_trace",
     "detail": "a knowledge-only core turn through the "
               "API returns a clean turn-out with mode "
               "knowledge_only and its TF-084 joined "
               "trace included and clean"},
    {"id": "trace_turn_id_agrees_and_sections_present",
     "kind": "trace_agreement",
     "detail": "the included trace's turn_id agrees "
               "with the turn-out's and its cognition / "
               "inference / reconciliation sections are "
               "present in TF-084's JOINED_FORMAT"},
    {"id": "trace_is_detached_from_core",
     "kind": "trace_detachment",
     "detail": "mutating the returned trace leaves the "
               "core's re-exported joined trace unchanged "
               "and clean"},
    {"id": "want_trace_false_files_not_requested",
     "kind": "no_trace_requested",
     "detail": "want_trace=False files trace_status "
               "not_requested and no trace"},
    {"id": "injected_turn_without_trace_source_unavailable",
     "kind": "injected_no_trace",
     "detail": "an injected turn callable with no trace "
               "source files trace_status unavailable "
               "naming why, and no trace"},
    {"id": "version_is_tf034",
     "kind": "version_tf034",
     "detail": "the contract version is TF-034's "
               "API_VERSION, verify_api_surface() is "
               "clean, and the api manifest's version "
               "matches"},
    {"id": "invalid_turn_in_raises_named",
     "kind": "invalid_turn_in",
     "detail": "an invalid turn-in (empty user_text, "
               "num_drafts=0) raises ValueError naming "
               "the problem"},
)


def run_cognition_probes() -> list[dict[str, Any]]:
    """Run the filed probes; each returns its verdict."""
    results: list[dict[str, Any]] = []
    for probe in COGNITION_PROBES:
        kind = probe["kind"]
        try:
            if kind == "knowledge_trace":
                api, _core = _knowledge_only_api()
                out = api.turn("What time does the ferry leave?")
                passed = turn_out_problems(out) == [] \
                    and out.get("mode") == "knowledge_only" \
                    and out.get("trace_status") == "included" \
                    and out.get("trace") is not None \
                    and bool(out.get("text"))
            elif kind == "trace_agreement":
                api, _core = _knowledge_only_api()
                out = api.turn("What time does the ferry leave?")
                trace = out.get("trace") or {}
                passed = trace.get("format") == JOINED_FORMAT \
                    and out.get("trace_format") == JOINED_FORMAT \
                    and trace.get("turn_id") == out.get("turn_id") \
                    and all(section in trace for section in (
                        "cognition", "inference", "reconciliation")) \
                    and joined_trace_problems(trace) == []
            elif kind == "trace_detachment":
                api, core = _knowledge_only_api()
                out = api.turn("What time does the ferry leave?")
                original_turn_id = out["turn_id"]
                out["trace"]["turn_id"] = "turn_mutated"
                reexported = core.last_turn_joined_trace()
                passed = reexported is not None \
                    and reexported.get("turn_id") == original_turn_id \
                    and reexported.get("turn_id") != "turn_mutated" \
                    and joined_trace_problems(reexported) == []
            elif kind == "no_trace_requested":
                api, _core = _knowledge_only_api()
                out = api.turn("What time does the ferry leave?",
                               want_trace=False)
                passed = out.get("trace_status") == "not_requested" \
                    and out.get("trace") is None \
                    and out.get("trace_format") is None \
                    and turn_out_problems(out) == []
            elif kind == "injected_no_trace":
                api = CognitionAPI(turn_callable=_stub_record)
                out = api.turn("hello forge")
                passed = str(out.get("trace_status", "")).startswith(
                        "unavailable: ") \
                    and out.get("trace") is None \
                    and turn_out_problems(out) == []
            elif kind == "version_tf034":
                passed = cognition_contract()["contract_version"] \
                        == API_VERSION \
                    and api_manifest().get("version") == API_VERSION \
                    and verify_api_surface() == []
            elif kind == "invalid_turn_in":
                api = CognitionAPI(turn_callable=_stub_record)
                named: list[bool] = []
                for kwargs, needle in (
                        ({"user_text": ""}, "user_text"),
                        ({"user_text": "hi", "num_drafts": 0},
                         "num_drafts")):
                    try:
                        api.turn(**kwargs)  # type: ignore[arg-type]
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


# ── The readable render (the cognition-api doc, verbatim) ────────────────────

def render_cognition_api_markdown() -> str:
    """Render the cognition-api doc deterministically
    — the exact text filed as
    docs/architecture/COGNITION_API.md."""
    lines: list[str] = []
    lines.append("# Cognition API — Turn-In / Turn-Out "
                 "with Traces (TF-154)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/cognition_api.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.cognition_api`. The "
        "checker (`cognition_api_problems()`) re-derives "
        "every claim live from the tree and fails on any "
        "drift.")
    lines.append("")
    lines.append(f"Format: `{COGNITION_API_FORMAT}`")
    lines.append(
        f"Contract version: `{API_VERSION}` (TF-034's "
        "`API_VERSION` — consumed, not reinvented)")
    lines.append(
        f"Trace format: `{JOINED_FORMAT}` (TF-084's "
        "`JOINED_FORMAT` — consumed, not reinvented)")
    lines.append(f"Turn-in fields: {len(TURN_IN_FIELDS)}; "
                 f"turn-out fields: {len(TURN_OUT_FIELDS)}; "
                 f"rules: {len(COGNITION_RULES)}")
    lines.append(f"Probes: {len(COGNITION_PROBES)}/{len(COGNITION_PROBES)}")
    lines.append("")
    lines.append("## 1. One turn")
    lines.append("")
    lines.append(
        "`CognitionAPI.turn(user_text, retrieval_path=None, "
        "num_drafts=None, history=None, want_trace=True)` "
        "runs one turn through `ThoughtForgeCore.think()` "
        "and returns the turn-out as a plain JSON-safe "
        "dict. Only the kwargs the caller actually set (a "
        "non-None value) are passed to `think()`, whose "
        "frozen params are TF-034's. An invalid turn-in "
        "raises `ValueError` naming the problem(s), judged "
        "by the pure validator `turn_in_problems()` before "
        "any core is built or called — a library-layer "
        "caller error, never silently coerced. The core "
        "is built lazily: constructing `CognitionAPI()` "
        "builds no core until a turn, and a core / core "
        "factory / turn callable plus a trace callable "
        "can be injected (TF-153's injection shape).")
    lines.append("")
    lines.append("## 2. Turn-in")
    lines.append("")
    lines.append("| Field | Required | What it is |")
    lines.append("|---|---|---|")
    for field in TURN_IN_FIELDS:
        required = "yes" if field["required"] else "no"
        lines.append(f"| {field['id']} | {required} | {field['summary']} |")
    lines.append("")
    lines.append("## 3. Turn-out")
    lines.append("")
    lines.append("| Field | Source | What it is |")
    lines.append("|---|---|---|")
    for field in TURN_OUT_FIELDS:
        lines.append(f"| {field['id']} | {field['source']} | "
                     f"{field['summary']} |")
    lines.append("")
    lines.append(
        "The response fields are the real "
        "`FinalResponseRecord`'s: `text`, `response_id`, "
        "`turn_id`, `mode`, `enforcement_passed`, "
        "`citations`, `token_count`.")
    lines.append("")
    lines.append("## 4. The trace (TF-084)")
    lines.append("")
    lines.append(
        "After a turn on a real core, the trace is "
        "obtained from the core's "
        "`last_turn_joined_trace()` — TF-084's joined "
        f"trace in `{JOINED_FORMAT}`, validated live with "
        "`joined_trace_problems()`. `trace_status` is "
        "`\"included\"` when the trace rides the "
        "turn-out, `\"not_requested\"` when "
        "`want_trace=False`, or `\"unavailable: "
        "<reason>\"` when there is no trace source, the "
        "core returned none, or the trace failed its "
        "checker — the reason is named, a trace that "
        "fails its checker is never filed as good, and "
        "no trace is ever faked. With an injected turn "
        "callable and no trace callable, the status files "
        "unavailable honestly. The exported trace is "
        "detached JSON-safe data (a JSON round-trip "
        "copy): mutating a turn-out's trace cannot touch "
        "the core's retained state, and the trace's "
        "`turn_id` must agree with the turn-out's.")
    lines.append("")
    lines.append("## 5. The rules")
    lines.append("")
    for rule in COGNITION_RULES:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## 6. Docket")
    lines.append("")
    lines.append(
        "TF-154 has **no share** of TF-016 and found no "
        "new defect in existing code, so this slice does "
        "not edit the docket and adds no entry. TF-016 "
        "totals stand unchanged (137 entries — 32 OPEN / "
        "1 OPEN-ENVIRONMENT / 100 FIXED / 4 "
        "CLOSED-BY-VERDICT), filed in the open here.")
    lines.append("")
    lines.append("## 7. The probes")
    lines.append("")
    for probe in COGNITION_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 8. Filed limits")
    lines.append("")
    lines.append(
        "- Proof grade FIXTURE: the checker and probes "
        "run a knowledge-only core (the TF-084 fixture "
        "shape) on this VM. A live model turn through "
        "the cognition API is NOT PROVEN, never claimed.")
    lines.append(
        "- TF-155 owns the memory API; TF-162 owns "
        "versioning enforcement in depth. TF-154 files "
        "the turn-in / turn-out, the TF-034 version "
        "consumed, and the TF-084 trace carried.")
    lines.append(
        "- TF-153's service shape can carry this API, "
        "but wiring the service turn op to it is NOT "
        "done here: `service_shape.py` is not modified "
        "by this slice, so the service and this API "
        "stand as separate filed shapes until a later "
        "slice wires them.")
    lines.append(
        "- No canon data was touched (TF-017); no "
        "README.md anywhere was touched; no Docker was "
        "used.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_cognition_api_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / COGNITION_API_DOC
    doc.write_text(render_cognition_api_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.cognition_api`."""
    doc = write_cognition_api_doc()
    print(f"cognition api doc regenerated: -> {COGNITION_API_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def cognition_api_problems() -> list[str]:
    """Re-derive TF-154's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    if COGNITION_API_FORMAT != "thoughtforge-cognition-api/1":
        problems.append(
            f"the format is {COGNITION_API_FORMAT!r}, not "
            "thoughtforge-cognition-api/1")
    in_ids = [f["id"] for f in TURN_IN_FIELDS]
    if in_ids != ["user_text", "retrieval_path", "num_drafts",
                  "history", "want_trace"] \
            or len(in_ids) != len(set(in_ids)):
        problems.append(
            f"the TURN_IN_FIELDS filing is not as filed: {in_ids}")
    for field in TURN_IN_FIELDS:
        problems.extend(turn_field_problems(field))
    user_text_field = next(
        (f for f in TURN_IN_FIELDS if f["id"] == "user_text"), {})
    if user_text_field.get("required") is not True:
        problems.append("turn-in: user_text is not filed as required")
    want_trace_field = next(
        (f for f in TURN_IN_FIELDS if f["id"] == "want_trace"), {})
    if want_trace_field.get("default") is not True:
        problems.append("turn-in: want_trace is not filed with default True")
    out_ids = [f["id"] for f in TURN_OUT_FIELDS]
    if out_ids != ["text", "response_id", "turn_id", "mode",
                   "enforcement_passed", "citations", "token_count",
                   "contract_version", "trace", "trace_format",
                   "trace_status"] \
            or len(out_ids) != len(set(out_ids)):
        problems.append(
            f"the TURN_OUT_FIELDS filing is not as filed: {out_ids}")
    for field in TURN_OUT_FIELDS:
        problems.extend(turn_field_problems(field))
    rule_ids = [r["id"] for r in COGNITION_RULES]
    if len(rule_ids) != len(set(rule_ids)) or len(COGNITION_RULES) != 6:
        problems.append("the COGNITION_RULES filing is not six unique rules")
    for rule in COGNITION_RULES:
        problems.extend(cognition_rule_problems(rule))
    probe_ids = [p["id"] for p in COGNITION_PROBES]
    if len(probe_ids) != len(set(probe_ids)) or len(COGNITION_PROBES) != 7:
        problems.append("the COGNITION_PROBES filing is not seven unique probes")

    # The version is TF-034's, and its surface is clean.
    contract = cognition_contract()
    if contract["contract_version"] != API_VERSION \
            or API_VERSION != api_manifest().get("version"):
        problems.append(
            "the filed contract_version is not TF-034's API_VERSION")
    if contract["trace_format"] != JOINED_FORMAT:
        problems.append("the filed trace_format is not TF-084's JOINED_FORMAT")
    surface = verify_api_surface()
    if surface:
        problems.append(
            f"TF-034: verify_api_surface() is not clean: {surface[:3]}")

    # A knowledge-only core turn through the API
    # returns a clean turn-out, trace included.
    try:
        api, _core = _knowledge_only_api()
        out = api.turn("What time does the ferry leave?")
        problems.extend(
            f"knowledge turn: {p}" for p in turn_out_problems(out))
        if out.get("mode") != "knowledge_only":
            problems.append(
                f"knowledge turn: mode is {out.get('mode')!r}, "
                "not knowledge_only")
        if out.get("trace_status") != "included" \
                or out.get("trace") is None:
            problems.append(
                "knowledge turn: the trace is not included: "
                f"trace_status is {out.get('trace_status')!r}")
    except Exception as exc:
        problems.append(f"knowledge turn failed: {exc}")

    # The probes all behave.
    probes = run_cognition_probes()
    if len(probes) != len(COGNITION_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the cognition probes do not all pass: "
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
            "FIXED; TF-154 must not move them")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / COGNITION_API_DOC
    if not doc.exists():
        problems.append(f"the cognition doc {COGNITION_API_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != render_cognition_api_markdown():
        problems.append(
            f"the filed doc {COGNITION_API_DOC} is not the render "
            "verbatim (regenerate with python3 -m "
            "thoughtforge.cognition_api)")
    return problems


def cognition_api_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_cognition_probes()
    return {
        "format": COGNITION_API_FORMAT,
        "doc": COGNITION_API_DOC,
        "contract": cognition_contract(),
        "manifest": api_manifest(),
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "COGNITION_API_DOC",
    "COGNITION_API_FORMAT",
    "COGNITION_PROBES",
    "COGNITION_RULES",
    "TRACE_STATUSES",
    "TURN_IN_FIELDS",
    "TURN_IN_IDS",
    "TURN_OUT_FIELDS",
    "TURN_OUT_IDS",
    "CognitionAPI",
    "cognition_api_as_data",
    "cognition_api_problems",
    "cognition_contract",
    "cognition_rule_problems",
    "main",
    "render_cognition_api_markdown",
    "run_cognition_probes",
    "turn_field_problems",
    "turn_in_problems",
    "turn_out_problems",
    "write_cognition_api_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
