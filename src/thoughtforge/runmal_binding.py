"""
TF-156 — the Rúnmál binding, filed:
Rúnmál's cognition/memory calls can route
through ThoughtForge, behind its backend
table, off by default.

TF-156's roadmap text: *Rúnmál binding:
Rúnmál's cognition/memory calls can route
through ThoughtForge (behind its backend
table; off by default; the parity campaign's
P-slices consume the doc).* It is Part IX
slice 4, following TF-155's memory API,
TF-154's cognition API and TF-153's service
shape. TF-147's filed limits name this slice
as owning the binding "behind its backend
table, off by default".

The measured start, probed live before
anything was built:

  * **No binding existed.** The word
    ThoughtForge appears nowhere in Rúnmál's
    ``src`` tree, and nothing on the
    ThoughtForge side filed a Rúnmál route:
    TF-154 filed the cognition API and TF-155
    the memory API as standalone service
    shapes, TF-147 filed the card export as a
    shape, "not a binding".
  * **Rúnmál's backend table, read (never
    edited, never imported).** Rúnmál's
    ``src/runmal/backends/registry.py`` files
    ``BackendRegistry`` —
    ``register(backend, name=None)`` returns
    ``name or backend.backend_key``,
    ``get(key)``, ``keys()`` sorted — over
    the ``ChatBackend`` ABC in
    ``src/runmal/backends/base.py``:
    ``backend_key`` / ``default_model`` class
    attributes, the abstract methods
    ``chat_stream(messages, params, cancel)``
    (an async event iterator),
    ``list_models()`` and ``health_check()``,
    and the concrete ``count_tokens`` /
    ``filter_params``. The shapes it works
    with are filed in ``backends/messages.py``
    (``ChatMessage``: role / content / name),
    ``backends/models.py`` (``ModelInfo``:
    model_id / display_name / backend_key /
    capabilities; ``Capabilities``: chat /
    embeddings / vision / tools / streaming /
    structured_output / context_length /
    max_output_tokens / token_count_quality /
    supported_samplers; ``HealthStatus``: ok /
    detail / latency_ms), ``backends/params.py``
    (``SamplingParams``: temperature / top_p /
    top_k / min_p / repetition_penalty /
    presence_penalty / frequency_penalty /
    max_output_tokens / stop / seed /
    context_length) and ``backends/events.py``
    (``TextDelta`` / ``UsageUpdate`` / ``Done``
    / ``ErrorEvent``, each with ``to_dict()``).
    Rúnmál's memory calls live in
    ``src/runmal/memory/service.py``
    (``create_memory`` / ``get_memory`` /
    ``list_memories`` / ``delete_memory``).
    Those field names are filed below as
    literal data; NO Rúnmál code is imported
    or copied anywhere in this module.
  * **The neighbours stood clean live**:
    ``service_shape_problems()``,
    ``cognition_api_problems()``,
    ``memory_api_problems()`` and
    ``runmal_cards_problems()`` all returned
    [] on this tree before the build.

**What this slice files**

  * ``BACKEND_TABLE_ROWS`` — how a
    ThoughtForge entry sits in Rúnmál's
    backend table, as literal data: one row
    with ``backend_key`` ``"thoughtforge"``,
    the registry's register / get / keys
    semantics, the ``ChatBackend`` interface
    it must answer (abstract and concrete
    methods, class attributes),
    ``default_model`` None, and
    ``enabled_by_default`` False.
  * ``CALL_ROWS`` — every Rúnmál-side call
    mapped to its ThoughtForge route, as
    literal data. Cognition: the
    ``chat_stream``-shaped call routes to a
    TF-154 ``CognitionAPI`` turn;
    ``list_models`` / ``health_check`` get
    filed honest answers from the cognition
    contract (``FILED_MODEL`` /
    ``FILED_HEALTH``); ``count_tokens`` /
    ``filter_params`` are filed NOT routed —
    they are concrete defaults on Rúnmál's
    own ABC and the binding invents no
    counter and claims no sampler. Memory:
    Rúnmál's memory calls route to TF-155
    ``MemoryAPI``'s four ops (add / inspect /
    prune / export), called, never
    reimplemented — with ``delete_memory``
    filed PARTIAL, because TF-155's prune is
    a lifecycle pass, not a targeted
    single-record delete.
  * ``FIELD_ROWS`` — the message -> turn-in
    and turn-out -> reply mapping, as literal
    data: every field with a disposition,
    none silently dropped or invented. The
    filed assembly rule: the LAST message
    with role ``"user"`` supplies turn-in
    ``user_text`` from its content; every
    other message, in original order, rides
    as turn-in ``history`` (role / content,
    plus name when present). The sampling
    params and the cancel event have no
    turn-in home and are filed as such —
    accepted in the call shape, never passed
    on, never claimed honored.
  * ``RunmalBinding`` — wraps a TF-154
    ``CognitionAPI`` and a TF-155
    ``MemoryAPI`` (each injectable directly,
    by factory, or via a ``turn_callable`` /
    ``memory_store`` passthrough; both built
    lazily — constructing the binding builds
    neither API). **DEFAULT DISABLED**:
    ``DEFAULT_ENABLED = False`` is filed as
    data; while disabled every routed call
    returns a truthful refusal as data
    (``ok=False``, ``status="refused"``, an
    ``error`` naming the binding as disabled)
    and touches neither API. Enabling is an
    explicit act: the constructor flag
    ``enabled=True`` or the ``enable()``
    method (``disable()`` reverses it). When
    enabled, cognition calls route through
    the injected CognitionAPI and memory
    calls through the injected MemoryAPI;
    every response is detached JSON-safe
    data whose ``contract_version`` is
    TF-034's ``API_VERSION``, consumed.
  * ``RUNMAL_BINDING_RULES`` — the binding's
    law as literal data, and
    ``runmal_binding_problems()`` — a checker
    that re-derives every claim live: the
    filings, the TF-034 surface (through the
    neighbour checkers), a disabled binding
    refusing with nothing built or called, a
    knowledge-only core chat round-trip and
    a temp-store memory round-trip through
    the binding, the filed probes, the
    docket header and the filed doc.

**Docket**: TF-156 has no share of TF-016 and
finds no new defect in existing code; it does
NOT edit ``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md``.
TF-016 totals stand unchanged (137 entries —
31 OPEN / 1 OPEN-ENVIRONMENT / 101 FIXED / 4
CLOSED-BY-VERDICT), verified live against
the docket header and filed in the open here
and in the render.

The human-readable doc is
``docs/architecture/RUNMAL_BINDING.md``,
filed verbatim as this module's render
(regenerate with
``python3 -m thoughtforge.runmal_binding``).
It is the artefact the parity campaign's
P-slices consume. Proof grade: FIXTURE — a
fake Rúnmál consumer shaped on the filed
ChatBackend / registry data; NO live Rúnmál
process was routed and NO Rúnmál code was
imported or copied. Live end-to-end routing
is NOT PROVEN — TF-170's Part IX gate owns
the at-least-one-live proof, never claimed
here. TF-157 owns the Samtal binding;
TF-161 owns binding fixtures in depth;
TF-162 owns versioning enforcement in depth.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from thoughtforge.cognition.api_surface import (
    API_VERSION,
    verify_api_surface,
)
from thoughtforge.cognition_api import (
    TURN_IN_FIELDS,
    TURN_OUT_FIELDS,
    CognitionAPI,
    cognition_api_problems,
)
from thoughtforge.memory_api import (
    OP_IDS as MEMORY_OP_IDS,
)
from thoughtforge.memory_api import (
    MemoryAPI,
    memory_api_problems,
)

RUNMAL_BINDING_FORMAT = "thoughtforge-runmal-binding/1"
RUNMAL_BINDING_DOC = "docs/architecture/RUNMAL_BINDING.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

#: The backend_key a ThoughtForge entry sits
#: under in Rúnmál's BackendRegistry.
BACKEND_KEY = "thoughtforge"

#: The binding is OFF unless someone enables
#: it explicitly — filed as data, proven by
#: the probes.
DEFAULT_ENABLED = False

#: The refusal every routed call returns
#: while the binding is disabled — as data,
#: naming the binding as disabled.
DISABLED_ERROR = (
    "runmal binding is disabled: the TF-156 "
    "binding is off by default — enable it "
    "explicitly (RunmalBinding(enabled=True) "
    "or binding.enable()) before any call "
    "routes")

#: Rúnmál's Done.reason is a free string; the
#: binding files its own constant rather than
#: inventing a provider finish reason.
DONE_REASON = "complete"


# ── The backend-table row, as literal data ───────────────────────────────────

BACKEND_TABLE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "thoughtforge_backend",
     "backend_key": BACKEND_KEY,
     "registry": "BackendRegistry "
                 "(runmal/backends/registry.py)",
     "register": "register(backend, name=None) -> str: "
                 "the key is name or "
                 "backend.backend_key, so a ThoughtForge "
                 "backend registers under \"thoughtforge\" "
                 "with no name argument",
     "get": "get(key) -> ChatBackend",
     "keys": "keys() -> list[str], sorted",
     "registry_methods": [
         "register", "get", "keys", "list_all_models",
         "list_models_for", "health_all", "health_for",
         "health_dicts", "token_count",
         "token_count_from_usage"],
     "interface": "ChatBackend (runmal/backends/base.py) "
                  "— the abstract chat backend",
     "abstract_methods": ["chat_stream", "list_models",
                          "health_check"],
     "concrete_methods": ["count_tokens", "filter_params"],
     "class_attributes": ["backend_key", "default_model"],
     "default_model": None,
     "enabled_by_default": False,
     "summary": "one ThoughtForge entry in Rúnmál's "
                "backend table: it answers the ChatBackend "
                "interface under the key \"thoughtforge\"; "
                "the registry's aggregations "
                "(list_all_models / health_all / "
                "token_count) reach it through the same "
                "three abstract methods every backend "
                "answers; it carries no default model, so "
                "Rúnmál's own filter_params default "
                "changes nothing"},
)

BACKEND_ROW_IDS: tuple[str, ...] = tuple(
    r["id"] for r in BACKEND_TABLE_ROWS)


# ── The filed honest answers (list_models / health_check) ────────────────────

#: The one model the binding files for
#: list_models — ModelInfo-shaped exactly as
#: runmal/backends/models.py's to_dict()
#: files it (Capabilities included). Honest:
#: chat only, no streaming, no samplers
#: claimed, token counts estimated.
FILED_MODEL: dict[str, Any] = {
    "model_id": "thoughtforge",
    "display_name": "ThoughtForge cognition (TF-154 API)",
    "backend_key": BACKEND_KEY,
    "capabilities": {
        "chat": True,
        "embeddings": False,
        "vision": False,
        "tools": False,
        "streaming": False,
        "structured_output": False,
        "context_length": None,
        "max_output_tokens": None,
        "token_count_quality": "estimated",
        "supported_samplers": [],
    },
}

#: The health answer an enabled binding files
#: — HealthStatus-shaped exactly as
#: runmal/backends/models.py's to_dict()
#: files it. It reports the binding, never a
#: live model.
FILED_HEALTH: dict[str, Any] = {
    "ok": True,
    "detail": "runmal binding enabled: cognition routes "
              "through TF-154's CognitionAPI and memory "
              "through TF-155's MemoryAPI (contract "
              f"{API_VERSION}); this answer reports the "
              "binding's own state, not a live model — "
              "no live model health is claimed",
    "latency_ms": None,
}


# ── The calls, as literal data ───────────────────────────────────────────────

CALL_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "chat_stream",
     "surface": "cognition",
     "runmal_side": "ChatBackend.chat_stream(messages: "
                    "list[ChatMessage], params: "
                    "SamplingParams, cancel: asyncio.Event) "
                    "-> AsyncIterator[Event]",
     "thoughtforge_route": "TF-154 CognitionAPI.turn — "
                           "the turn-in assembled by the "
                           "filed rule in FIELD_ROWS (last "
                           "user message as user_text, "
                           "every other message as "
                           "history); the turn-out returns "
                           "verbatim in the response's "
                           "turn and as the filed reply "
                           "events",
     "binding_method": "chat(messages, params=None)",
     "routed": True,
     "summary": "one cognition turn through ThoughtForge "
                "for one Rúnmál chat call"},
    {"id": "list_models",
     "surface": "cognition",
     "runmal_side": "ChatBackend.list_models() -> "
                    "list[ModelInfo] (the registry "
                    "aggregates it as list_all_models / "
                    "list_models_for)",
     "thoughtforge_route": "the filed FILED_MODEL row — "
                           "an honest answer from "
                           "TF-154's CognitionAPI contract: "
                           "the one model is the "
                           "ThoughtForge cognition API "
                           "itself; no model process is "
                           "queried and none is invented",
     "binding_method": "list_models()",
     "routed": True,
     "summary": "the backend table's model listing for "
                "the ThoughtForge entry"},
    {"id": "health_check",
     "surface": "cognition",
     "runmal_side": "ChatBackend.health_check() -> "
                    "HealthStatus (the registry "
                    "aggregates it as health_all / "
                    "health_for / health_dicts)",
     "thoughtforge_route": "the filed FILED_HEALTH row — "
                           "reports the binding's own "
                           "enabled state against the "
                           "TF-154 CognitionAPI / TF-155 "
                           "MemoryAPI contract; "
                           "claims no live model health",
     "binding_method": "health_check()",
     "routed": True,
     "summary": "the backend table's health answer for "
                "the ThoughtForge entry"},
    {"id": "count_tokens",
     "surface": "cognition",
     "runmal_side": "ChatBackend.count_tokens(messages, "
                    "model_id) -> TokenCount — a concrete "
                    "default on the ABC (the shared "
                    "runmal.tokens counter over "
                    "concatenated content)",
     "thoughtforge_route": "not routed: the method is "
                           "Rúnmál's own concrete "
                           "default; the binding files "
                           "no counter and invents none — "
                           "FILED_MODEL's "
                           "token_count_quality "
                           "\"estimated\" agrees with "
                           "Rúnmál's default quality",
     "binding_method": None,
     "routed": False,
     "summary": "token counting stays Rúnmál's own "
                "default; nothing routes to "
                "ThoughtForge"},
    {"id": "filter_params",
     "surface": "cognition",
     "runmal_side": "ChatBackend.filter_params(params) "
                    "-> (SamplingParams, warnings) — a "
                    "concrete default on the ABC, driven "
                    "by the backend's default_model",
     "thoughtforge_route": "not routed: the filed "
                           "backend row's default_model "
                           "is None, so Rúnmál's own "
                           "default returns the params "
                           "unchanged with no warnings — "
                           "and FILED_MODEL's "
                           "supported_samplers is empty, "
                           "so no sampler is ever claimed "
                           "supported",
     "binding_method": None,
     "routed": False,
     "summary": "parameter filtering stays Rúnmál's own "
                "default over a None default model"},
    {"id": "memory_add",
     "surface": "memory",
     "runmal_side": "runmal.memory.service.create_memory"
                    "(...) — Rúnmál's memory write call",
     "thoughtforge_route": "TF-155 MemoryAPI.call(\"add\", "
                           "request) — TF-057's "
                           "add_record through TF-155, "
                           "called, never reimplemented: "
                           "the TF-042 gate, the "
                           "manual_entry stamp and the "
                           "duplicate refusal are "
                           "TF-057's by construction",
     "binding_method": "memory_call(\"add\", request) "
                       "/ memory_add(kind, ...)",
     "routed": True,
     "summary": "a Rúnmál-side memory write lands as a "
                "TF-155 add"},
    {"id": "memory_inspect",
     "surface": "memory",
     "runmal_side": "runmal.memory.service.get_memory"
                    "(...) / list_memories(...) — "
                    "Rúnmál's memory read calls",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"inspect\", request) — "
                           "TF-057's inspect_store / "
                           "inspect_records / find_record "
                           "through TF-155, called, never "
                           "reimplemented; read-only",
     "binding_method": "memory_call(\"inspect\", request) "
                       "/ memory_inspect(...)",
     "routed": True,
     "summary": "Rúnmál-side memory reads land as "
                "TF-155 inspects"},
    {"id": "memory_prune",
     "surface": "memory",
     "runmal_side": "runmal.memory.service.delete_memory"
                    "(...) — Rúnmál's memory removal "
                    "call",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"prune\", request) — "
                           "PARTIAL by filing: TF-057's "
                           "prune is a MemoryLifecycle "
                           "pass over the whole store, "
                           "not a targeted single-record "
                           "delete; a targeted delete has "
                           "no ThoughtForge home and is "
                           "NOT routed — the prune is "
                           "what this row honestly offers",
     "binding_method": "memory_call(\"prune\", request) "
                       "/ memory_prune(...)",
     "routed": True,
     "summary": "memory removal routes only as far as "
                "TF-155's lifecycle prune honestly "
                "reaches — no targeted delete is "
                "pretended"},
    {"id": "memory_export",
     "surface": "memory",
     "runmal_side": "no counterpart call in "
                    "runmal.memory.service — Rúnmál's "
                    "rebuild / backup paths are its own",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"export\", request) — "
                           "TF-057's export_store_bundle "
                           "through TF-155, the verified "
                           "TF-052 bundle returned as "
                           "detached data; filed so all "
                           "four TF-155 ops have a row "
                           "and the parity campaign can "
                           "consume the whole surface",
     "binding_method": "memory_call(\"export\", request) "
                       "/ memory_export(...)",
     "routed": True,
     "summary": "the store export is reachable through "
                "the binding even though Rúnmál's "
                "memory service files no export call"},
)

CALL_IDS: tuple[str, ...] = tuple(r["id"] for r in CALL_ROWS)

ROUTED_CALL_IDS: tuple[str, ...] = tuple(
    r["id"] for r in CALL_ROWS if r["routed"])

#: SamplingParams' fields, filed from
#: runmal/backends/params.py — each has a
#: no-home row in FIELD_ROWS.
SAMPLING_PARAM_FIELDS: tuple[str, ...] = (
    "temperature", "top_p", "top_k", "min_p",
    "repetition_penalty", "presence_penalty",
    "frequency_penalty", "max_output_tokens", "stop",
    "seed", "context_length",
)


# ── The field mapping, as literal data ───────────────────────────────────────
#
# Every field on either side has a row with a
# disposition: mapped / mapped_as_history /
# partial / no_home_not_passed /
# no_home_not_wired / no_source_not_set /
# no_source_default_applies / binding_filed.
# None is silently dropped; none is invented.

FIELD_ROWS: tuple[dict[str, Any], ...] = (
    # ── In: Rúnmál messages / params -> TF-154 turn-in ──
    {"direction": "in",
     "id": "messages.last_user.content",
     "source": "ChatMessage.content of the last message "
               "with role \"user\"",
     "target": "turn-in user_text",
     "disposition": "mapped",
     "summary": "the filed assembly rule's first half: "
                "the last user message is the turn's "
                "question — TF-154 requires user_text as "
                "a non-empty string, so an empty last "
                "user content is an invalid request, "
                "named by chat_request_problems()"},
    {"direction": "in",
     "id": "messages.others",
     "source": "every message that is not the last "
               "user message, in original order",
     "target": "turn-in history",
     "disposition": "mapped_as_history",
     "summary": "the filed assembly rule's second "
                "half: earlier user turns, assistant "
                "replies, system and tool messages all "
                "ride as history entries — each a plain "
                "{\"role\", \"content\"} dict, plus "
                "\"name\" when the message carries one; "
                "an empty history is not passed at all "
                "(TF-154 passes only set kwargs)"},
    {"direction": "in",
     "id": "message.role",
     "source": "ChatMessage.role",
     "target": "the history entry's role; for the "
               "user_text source message, the role "
               "selects it and is not itself passed",
     "disposition": "mapped",
     "summary": "roles are preserved verbatim in "
                "history — the binding rewrites no "
                "role and invents none"},
    {"direction": "in",
     "id": "message.name",
     "source": "ChatMessage.name (optional)",
     "target": "the history entry's \"name\" when the "
               "message rides in history; no target "
               "when the message is the user_text "
               "source",
     "disposition": "partial",
     "summary": "TF-154's turn-in files no name "
                "field: a name rides only inside its "
                "history entry; the user_text source "
                "message's name has no home and is "
                "filed here as dropped-by-filing, "
                "never silently"},
    {"direction": "in",
     "id": "params.temperature",
     "source": "SamplingParams.temperature",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.top_p",
     "source": "SamplingParams.top_p",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.top_k",
     "source": "SamplingParams.top_k",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.min_p",
     "source": "SamplingParams.min_p",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.repetition_penalty",
     "source": "SamplingParams.repetition_penalty",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.presence_penalty",
     "source": "SamplingParams.presence_penalty",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.frequency_penalty",
     "source": "SamplingParams.frequency_penalty",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.max_output_tokens",
     "source": "SamplingParams.max_output_tokens",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.stop",
     "source": "SamplingParams.stop",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.seed",
     "source": "SamplingParams.seed",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "params.context_length",
     "source": "SamplingParams.context_length",
     "target": "no target — TURN_IN_FIELDS files no "
               "sampling field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape, never "
                "passed to TF-154, never claimed "
                "honored"},
    {"direction": "in",
     "id": "cancel",
     "source": "chat_stream's cancel (asyncio.Event)",
     "target": "no target",
     "disposition": "no_home_not_wired",
     "summary": "the binding is synchronous library "
                "data shaped on Rúnmál's async "
                "interface; cancellation is not wired "
                "through this filing — a Rúnmál caller "
                "cancelling mid-stream has no routed "
                "call here to cancel (any live async "
                "shape is TF-170's gate business, "
                "filed as a limit)"},
    {"direction": "in",
     "id": "turn_in.retrieval_path",
     "source": "nothing on the Rúnmál side files a "
               "retrieval path",
     "target": "turn-in retrieval_path",
     "disposition": "no_source_not_set",
     "summary": "the binding never sets it; the "
                "core's own default retrieval applies "
                "— no Rúnmál value is invented for it"},
    {"direction": "in",
     "id": "turn_in.num_drafts",
     "source": "nothing on the Rúnmál side files a "
               "draft count",
     "target": "turn-in num_drafts",
     "disposition": "no_source_not_set",
     "summary": "the binding never sets it; the "
                "core's own default draft count "
                "applies"},
    {"direction": "in",
     "id": "turn_in.want_trace",
     "source": "nothing on the Rúnmál side files a "
               "trace request",
     "target": "turn-in want_trace",
     "disposition": "no_source_default_applies",
     "summary": "the binding never sets it; TF-154's "
                "filed default (True) applies, so a "
                "routed turn carries its trace status "
                "honestly in the returned turn"},
    # ── Out: TF-154 turn-out -> the Rúnmál-shaped reply ──
    {"direction": "out",
     "id": "turn.text",
     "source": "turn-out text",
     "target": "the response's turn (verbatim) and "
               "events[0] — a TextDelta-shaped dict "
               "{\"type\": \"text_delta\", \"text\": "
               "...}",
     "disposition": "mapped",
     "summary": "ONE delta carrying the whole text: "
                "the binding is not a streamer and "
                "FILED_MODEL's streaming False agrees — "
                "no token-by-token stream is pretended"},
    {"direction": "out",
     "id": "turn.token_count",
     "source": "turn-out token_count",
     "target": "the response's turn (verbatim) and "
               "events[1] — a UsageUpdate-shaped dict "
               "with total_tokens set to it",
     "disposition": "partial",
     "summary": "TF-154 files ONE count, not a "
                "prompt / completion split: the usage "
                "event files prompt_tokens and "
                "completion_tokens as None rather than "
                "inventing a split"},
    {"direction": "out",
     "id": "turn.response_id",
     "source": "turn-out response_id",
     "target": "the response's turn (verbatim); no "
               "event home",
     "disposition": "mapped",
     "summary": "Rúnmál's events file no response id; "
                "the id rides the binding response's "
                "turn, never dropped"},
    {"direction": "out",
     "id": "turn.turn_id",
     "source": "turn-out turn_id",
     "target": "the response's turn (verbatim); no "
               "event home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim, beside its "
                "trace"},
    {"direction": "out",
     "id": "turn.mode",
     "source": "turn-out mode",
     "target": "the response's turn (verbatim); no "
               "event home",
     "disposition": "mapped",
     "summary": "the turn's mode (knowledge_only when "
                "no model served) rides the turn — a "
                "Rúnmál consumer can see honestly how "
                "the answer was produced"},
    {"direction": "out",
     "id": "turn.enforcement_passed",
     "source": "turn-out enforcement_passed",
     "target": "the response's turn (verbatim); no "
               "event home",
     "disposition": "mapped",
     "summary": "the enforcement verdict rides the "
                "turn verbatim"},
    {"direction": "out",
     "id": "turn.citations",
     "source": "turn-out citations",
     "target": "the response's turn (verbatim); no "
               "event home",
     "disposition": "mapped",
     "summary": "the turn's citations ride the turn "
                "verbatim"},
    {"direction": "out",
     "id": "turn.contract_version",
     "source": "turn-out contract_version",
     "target": "the response's turn (verbatim) and "
               "the binding envelope's "
               "contract_version — the same TF-034 "
               "value",
     "disposition": "mapped",
     "summary": "one version, TF-034's, in both "
                "places; no competing version is "
                "filed"},
    {"direction": "out",
     "id": "turn.trace",
     "source": "turn-out trace",
     "target": "the response's turn (verbatim); no "
               "event home",
     "disposition": "mapped",
     "summary": "TF-084's joined trace rides the turn "
                "when TF-154 includes it; events carry "
                "no trace"},
    {"direction": "out",
     "id": "turn.trace_format",
     "source": "turn-out trace_format",
     "target": "the response's turn (verbatim); no "
               "event home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim beside the "
                "trace"},
    {"direction": "out",
     "id": "turn.trace_status",
     "source": "turn-out trace_status",
     "target": "the response's turn (verbatim); no "
               "event home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim — included / "
                "not_requested / unavailable with its "
                "reason named, exactly as TF-154 "
                "filed it"},
    {"direction": "out",
     "id": "events.done",
     "source": "this binding — the filed constant "
               "DONE_REASON",
     "target": "events[-1] — a Done-shaped dict "
               "{\"type\": \"done\", \"reason\": "
               "DONE_REASON, \"usage\": the usage "
               "event}",
     "disposition": "binding_filed",
     "summary": "Rúnmál's Done.reason is a free "
                "string and TF-154 files no finish "
                "reason: the binding files its own "
                "constant \"complete\" rather than "
                "inventing a provider reason"},
)

FIELD_IDS: tuple[str, ...] = tuple(
    f["id"] for f in FIELD_ROWS)


def field_rows_for(direction: str) -> tuple[dict[str, Any], ...]:
    """One direction's filed field rows, in filing
    order."""
    return tuple(f for f in FIELD_ROWS
                 if f["direction"] == direction)


# ── The rules, as literal data ───────────────────────────────────────────────

RUNMAL_BINDING_RULES: tuple[dict[str, Any], ...] = (
    {"id": "off_by_default",
     "rule": "the binding is off by default: "
             "DEFAULT_ENABLED is False, filed as data; "
             "enabling is an explicit act — the "
             "constructor flag enabled=True or the "
             "enable() method — and disable() returns "
             "the binding to refusal"},
    {"id": "disabled_refusal_is_data_touches_nothing",
     "rule": "while disabled, every routed call "
             "returns a truthful refusal as data "
             "(ok=False, status=\"refused\", an error "
             "naming the binding as disabled) and "
             "touches neither API — no API is built "
             "and no call is routed, proven by the "
             "probes with counting factories"},
    {"id": "called_never_reimplemented",
     "rule": "cognition routes only through TF-154's "
             "CognitionAPI and memory only through "
             "TF-155's MemoryAPI (which itself calls "
             "TF-057's toolset) — this module "
             "implements no cognition, store or "
             "enforcement logic of its own"},
    {"id": "no_runmal_code_imported_or_copied",
     "rule": "no Rúnmál code is imported or copied: "
             "Rúnmál's shapes are filed as literal "
             "data read from its public backend "
             "contract (backends/base.py, registry.py, "
             "messages.py, models.py, params.py, "
             "events.py, memory/service.py), and "
             "nothing under the Rúnmál tree is edited "
             "by this slice"},
    {"id": "every_field_has_a_disposition",
     "rule": "every field on either side has a "
             "FIELD_ROWS row with a disposition — "
             "none is silently dropped and none is "
             "invented; the params and the cancel "
             "event are filed no-home in the open"},
    {"id": "params_never_claimed_honored",
     "rule": "sampling params are accepted in the "
             "call shape, never passed to TF-154 and "
             "never claimed honored; FILED_MODEL's "
             "empty supported_samplers and streaming "
             "False agree with that filing"},
    {"id": "responses_detached_json_safe_versioned",
     "rule": "every response is detached JSON-safe "
             "data (a JSON round-trip copy) whose "
             "contract_version is TF-034's "
             "API_VERSION, consumed from "
             "thoughtforge.cognition.api_surface — "
             "no competing version is invented here"},
)

RULE_IDS: tuple[str, ...] = tuple(
    r["id"] for r in RUNMAL_BINDING_RULES)


# ── The contract as plain data ───────────────────────────────────────────────

def runmal_binding_contract() -> dict[str, Any]:
    """The Rúnmál binding contract as plain
    JSON-safe data, built from the filed literals
    above and TF-034's version. A fresh copy each
    call."""
    return {
        "format": RUNMAL_BINDING_FORMAT,
        "contract_version": API_VERSION,
        "backend_key": BACKEND_KEY,
        "default_enabled": DEFAULT_ENABLED,
        "disabled_error": DISABLED_ERROR,
        "done_reason": DONE_REASON,
        "backend_table": [dict(r) for r in BACKEND_TABLE_ROWS],
        "model": json.loads(json.dumps(FILED_MODEL)),
        "health": dict(FILED_HEALTH),
        "calls": [dict(r) for r in CALL_ROWS],
        "fields": [dict(f) for f in FIELD_ROWS],
        "rules": [dict(r) for r in RUNMAL_BINDING_RULES],
    }


# ── Pure validators ──────────────────────────────────────────────────────────

def backend_row_problems(row: Any) -> list[str]:
    """Judge one backend-table filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["backend row: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("backend row: the row files no id")
    if row.get("backend_key") != BACKEND_KEY:
        problems.append(
            f"backend row {row.get('id')}: backend_key is "
            f"{row.get('backend_key')!r}, not {BACKEND_KEY!r}")
    for key in ("registry", "register", "get", "keys",
                "interface", "summary"):
        if not row.get(key):
            problems.append(
                f"backend row {row.get('id')}: no {key} filed")
    for key in ("registry_methods", "abstract_methods",
                "concrete_methods", "class_attributes"):
        if not isinstance(row.get(key), list) \
                or not row.get(key):
            problems.append(
                f"backend row {row.get('id')}: {key} is "
                "not a non-empty list")
    if "default_model" not in row \
            or row.get("default_model") is not None:
        problems.append(
            f"backend row {row.get('id')}: default_model "
            "must be filed as None")
    if row.get("enabled_by_default") is not False:
        problems.append(
            f"backend row {row.get('id')}: "
            "enabled_by_default is not False")
    return problems


def call_row_problems(row: Any) -> list[str]:
    """Judge one call filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["call: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("call: the row files no id")
    if row.get("surface") not in ("cognition", "memory"):
        problems.append(
            f"call {row.get('id')}: surface is "
            f"{row.get('surface')!r}, not cognition / memory")
    for key in ("runmal_side", "thoughtforge_route", "summary"):
        if not row.get(key):
            problems.append(
                f"call {row.get('id')}: no {key} filed")
    if not isinstance(row.get("routed"), bool):
        problems.append(
            f"call {row.get('id')}: routed is not a boolean")
    elif row["routed"] and not row.get("binding_method"):
        problems.append(
            f"call {row.get('id')}: routed but no "
            "binding_method filed")
    elif not row["routed"] and row.get("binding_method"):
        problems.append(
            f"call {row.get('id')}: not routed but a "
            "binding_method is filed")
    return problems


def field_row_problems(row: Any) -> list[str]:
    """Judge one field filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["field: the row is empty"]
    problems: list[str] = []
    if row.get("direction") not in ("in", "out"):
        problems.append(
            f"field {row.get('id')}: direction is "
            f"{row.get('direction')!r}, not in / out")
    for key in ("id", "source", "target", "disposition",
                "summary"):
        if not row.get(key):
            problems.append(
                f"field {row.get('id')}: no {key} filed")
    return problems


def binding_rule_problems(row: Any) -> list[str]:
    """Judge one rule filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["rule: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("rule: the row files no id")
    if not row.get("rule"):
        problems.append(f"rule {row.get('id')}: no rule filed")
    return problems


def chat_request_problems(
    messages: Any, params: Any = None,
) -> list[str]:
    """Judge one chat request, purely. Empty-safe.

    Shape only: the message list and the filed
    assembly rule's needs (a last user message
    with non-empty content). What the core does
    with the turn is TF-154's business."""
    problems: list[str] = []
    if not isinstance(messages, list) or not messages:
        return ["chat request: messages must be a "
                "non-empty list of message mappings"]
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(
                f"chat request: message {index} is not a "
                "mapping shaped like ChatMessage.to_dict()")
            continue
        for key in message:
            if key not in ("role", "content", "name"):
                problems.append(
                    f"chat request: message {index} files "
                    f"{key!r}, not a ChatMessage field "
                    "(role / content / name)")
        if not isinstance(message.get("role"), str) \
                or not message.get("role"):
            problems.append(
                f"chat request: message {index} role must "
                "be a non-empty string")
        if not isinstance(message.get("content"), str):
            problems.append(
                f"chat request: message {index} content "
                "must be a string")
        if "name" in message \
                and message["name"] is not None \
                and not isinstance(message["name"], str):
            problems.append(
                f"chat request: message {index} name must "
                "be a string when present")
    if not problems:
        users = [m for m in messages if m["role"] == "user"]
        if not users:
            problems.append(
                "chat request: no message with role "
                "\"user\" — the filed assembly rule takes "
                "user_text from the last user message")
        elif not users[-1]["content"]:
            problems.append(
                "chat request: the last user message's "
                "content is empty — TF-154's user_text "
                "must be a non-empty string")
    if params is not None:
        if not isinstance(params, dict):
            problems.append(
                "chat request: params must be a mapping "
                "of SamplingParams fields when present")
        else:
            for key in params:
                if key not in SAMPLING_PARAM_FIELDS:
                    problems.append(
                        f"chat request: params files "
                        f"{key!r}, not a SamplingParams "
                        f"field — the filed fields are "
                        f"{list(SAMPLING_PARAM_FIELDS)}")
    return problems


def binding_response_problems(row: Any) -> list[str]:
    """Judge one binding response dict against the
    envelope, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["response: the row is empty"]
    problems: list[str] = []
    for field in ("call", "ok", "status", "backend_key",
                  "contract_version", "error"):
        if field not in row:
            problems.append(f"response: no {field} filed")
    if problems:
        return problems
    if row.get("call") not in ROUTED_CALL_IDS:
        problems.append(
            f"response: call is {row.get('call')!r}, not "
            f"one of the routed calls {list(ROUTED_CALL_IDS)}")
    if not isinstance(row.get("ok"), bool):
        problems.append("response: ok is not a boolean")
    if not isinstance(row.get("status"), str) \
            or not row.get("status"):
        problems.append("response: status is not a non-empty string")
    if row.get("backend_key") != BACKEND_KEY:
        problems.append(
            f"response: backend_key is "
            f"{row.get('backend_key')!r}, not {BACKEND_KEY!r}")
    if row.get("contract_version") != API_VERSION:
        problems.append(
            f"response: contract_version is "
            f"{row.get('contract_version')!r}, not "
            f"TF-034's {API_VERSION!r}")
    if row.get("ok") is True and row.get("error") is not None:
        problems.append(
            "response: ok is True but an error is filed — "
            "a success files error None")
    if row.get("ok") is False:
        if row.get("status") != "refused":
            problems.append(
                "response: ok is False but status is "
                f"{row.get('status')!r}, not \"refused\"")
        if not isinstance(row.get("error"), str) \
                or not row.get("error"):
            problems.append(
                "response: ok is False but no error names "
                "the refusal")
    try:
        json.dumps(row)
    except (TypeError, ValueError) as exc:
        problems.append(f"response: not JSON-safe: {exc}")
    return problems


# ── The filed assembly rule + the reply events ───────────────────────────────

def assemble_turn_in(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """The filed assembly rule, applied purely:
    the last user message's content is user_text;
    every other message, in original order, is a
    history entry (role / content, plus name when
    present). Assumes chat_request_problems()
    judged the list clean."""
    last_user = max(
        i for i, m in enumerate(messages)
        if m["role"] == "user")
    history: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index == last_user:
            continue
        entry: dict[str, Any] = {
            "role": message["role"],
            "content": message["content"],
        }
        if message.get("name") is not None:
            entry["name"] = message["name"]
        history.append(entry)
    return {"user_text": messages[last_user]["content"],
            "history": history}


def reply_events(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """The filed reply events for one turn-out, in
    Rúnmál's event dict shapes (events.py's
    to_dict() forms): one TextDelta with the
    whole text, one UsageUpdate with the single
    filed count as total_tokens, one Done with
    the binding's filed DONE_REASON."""
    usage = {"type": "usage",
             "prompt_tokens": None,
             "completion_tokens": None,
             "total_tokens": turn.get("token_count")}
    return [
        {"type": "text_delta", "text": turn.get("text", "")},
        usage,
        {"type": "done", "reason": DONE_REASON, "usage": usage},
    ]


# ── The binding ──────────────────────────────────────────────────────────────

def _detached(data: Any) -> Any:
    """A detached JSON-safe copy (a JSON round-trip)."""
    return json.loads(json.dumps(data))


class RunmalBinding:
    """Rúnmál's calls routed through ThoughtForge,
    behind its backend table, off by default.

    Wraps a TF-154 ``CognitionAPI`` and a TF-155
    ``MemoryAPI``: pass ``cognition_api`` /
    ``memory_api`` to inject them,
    ``cognition_factory`` / ``memory_factory`` to
    build them lazily, or ``turn_callable`` /
    ``memory_store`` as passthroughs to the
    lazily built default APIs (TF-154's / TF-155's
    own injection shapes). Both APIs are built
    lazily: constructing the binding builds
    neither.

    The binding is DISABLED unless ``enabled``
    is True at construction or ``enable()`` is
    called: while disabled every routed call
    returns the filed refusal as data and
    touches neither API.
    """

    def __init__(
        self,
        cognition_api: Any = None,
        memory_api: Any = None,
        *,
        enabled: bool = DEFAULT_ENABLED,
        cognition_factory: Callable[[], Any] | None = None,
        memory_factory: Callable[[], Any] | None = None,
        turn_callable: Callable[..., Any] | None = None,
        memory_store: Any = None,
    ) -> None:
        self._cognition_api = cognition_api
        self._memory_api = memory_api
        self._cognition_factory = cognition_factory
        self._memory_factory = memory_factory
        self._turn_callable = turn_callable
        self._memory_store = memory_store
        self._enabled = bool(enabled)

    # ── The explicit enable act ──

    @property
    def enabled(self) -> bool:
        """Whether the binding currently routes."""
        return self._enabled

    def enable(self) -> None:
        """Enable routing — the explicit act the
        off-by-default law requires."""
        self._enabled = True

    def disable(self) -> None:
        """Return the binding to refusal."""
        self._enabled = False

    # ── Lazy API construction ──

    def _get_cognition(self) -> Any:
        if self._cognition_api is None:
            if self._cognition_factory is not None:
                self._cognition_api = self._cognition_factory()
            elif self._turn_callable is not None:
                self._cognition_api = CognitionAPI(
                    turn_callable=self._turn_callable)
            else:
                self._cognition_api = CognitionAPI()
        return self._cognition_api

    def _get_memory(self) -> Any:
        if self._memory_api is None:
            if self._memory_factory is not None:
                self._memory_api = self._memory_factory()
            elif self._memory_store is not None:
                self._memory_api = MemoryAPI(
                    store=self._memory_store)
            else:
                self._memory_api = MemoryAPI()
        return self._memory_api

    # ── Envelopes ──

    def _envelope(
        self, call: str, ok: bool, status: str, **fields: Any,
    ) -> dict[str, Any]:
        return _detached({
            "call": call, "ok": ok, "status": status,
            "backend_key": BACKEND_KEY,
            "contract_version": API_VERSION,
            "error": None if ok else fields.pop(
                "error", ""),
            **fields,
        })

    def _refused(self, call: str) -> dict[str, Any]:
        return self._envelope(
            call, False, "refused", error=DISABLED_ERROR)

    # ── Cognition calls ──

    def chat(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The chat_stream-shaped call: a list of
        ChatMessage-shaped dicts (role / content /
        name) in, one binding response out — the
        TF-154 turn verbatim in ``turn`` and the
        filed reply events in ``events``.

        While disabled, the filed refusal returns
        as data and nothing is judged, built or
        routed. When enabled, an invalid request
        raises ValueError naming the problem(s),
        judged by ``chat_request_problems()``
        before any API is built or called. The
        params are accepted in the call shape and
        filed no-home: never passed on, never
        claimed honored."""
        if not self._enabled:
            return self._refused("chat_stream")
        problems = chat_request_problems(messages, params)
        if problems:
            raise ValueError("; ".join(problems))
        turn_in = assemble_turn_in(messages)
        kwargs: dict[str, Any] = {}
        if turn_in["history"]:
            kwargs["history"] = turn_in["history"]
        turn = self._get_cognition().turn(
            turn_in["user_text"], **kwargs)
        return self._envelope(
            "chat_stream", True, "replied",
            turn=turn, events=reply_events(turn))

    def list_models(self) -> dict[str, Any]:
        """The list_models call: the filed
        FILED_MODEL row as the response's
        ``models`` — an honest answer from the
        cognition contract; no model process is
        queried. While disabled, the filed
        refusal returns as data."""
        if not self._enabled:
            return self._refused("list_models")
        return self._envelope(
            "list_models", True, "listed",
            models=[_detached(FILED_MODEL)])

    def health_check(self) -> dict[str, Any]:
        """The health_check call: the filed
        FILED_HEALTH row as the response's
        ``health`` — it reports the binding,
        never a live model. While disabled, the
        filed refusal returns as data."""
        if not self._enabled:
            return self._refused("health_check")
        return self._envelope(
            "health_check", True, "reported",
            health=dict(FILED_HEALTH))

    # ── Memory calls ──

    def memory_call(
        self,
        op: str,
        request: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """One memory call routed through TF-155's
        MemoryAPI: the TF-155 response rides
        verbatim in ``memory``. Invalid requests
        and unknown ops raise TF-155's named
        ValueError; TF-155's enforcement refusals
        surface through the envelope as data (ok
        False, the TF-155 refusal as ``error``).
        While disabled, the filed refusal returns
        as data and nothing is judged, built or
        routed."""
        call = f"memory_{op}"
        if not self._enabled:
            return self._refused(call)
        memory = self._get_memory().call(op, request, **kwargs)
        if memory.get("ok"):
            return self._envelope(
                call, True, str(memory.get("status")),
                memory=memory)
        return self._envelope(
            call, False, "refused",
            error=str(memory.get("refusal") or ""),
            memory=memory)

    # ── Convenience methods (kwargs in, dict out) ──

    def memory_add(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Add one record through TF-155's add."""
        return self.memory_call("add", {"kind": kind, **fields})

    def memory_inspect(self, **fields: Any) -> dict[str, Any]:
        """Inspect through TF-155's inspect."""
        return self.memory_call("inspect", fields or None)

    def memory_prune(self, **fields: Any) -> dict[str, Any]:
        """Prune through TF-155's prune."""
        return self.memory_call("prune", fields or None)

    def memory_export(self, **fields: Any) -> dict[str, Any]:
        """Export through TF-155's export."""
        return self.memory_call("export", fields or None)


# ── Fixtures (checker + probes) ──────────────────────────────────────────────

class _FakeRegistry:
    """A dict-shaped stand-in for Rúnmál's
    BackendRegistry, built from the filed row's
    semantics only (register returns name or the
    backend's key; get; sorted keys). Fixture
    only — no Rúnmál code."""

    def __init__(self) -> None:
        self._backends: dict[str, Any] = {}

    def register(self, backend: Any, name: str | None = None) -> str:
        key = name or backend["backend_key"]
        self._backends[key] = backend
        return key

    def get(self, key: str) -> Any:
        return self._backends[key]

    def keys(self) -> list[str]:
        return sorted(self._backends)


def _fake_backend() -> dict[str, Any]:
    """The filed ThoughtForge entry as a fake
    backend would carry it into the registry."""
    row = BACKEND_TABLE_ROWS[0]
    return {"backend_key": row["backend_key"],
            "interface": row["interface"],
            "abstract_methods": list(row["abstract_methods"]),
            "default_model": row["default_model"]}


def _stub_turn_out(user_text: str) -> dict[str, Any]:
    """A TF-154-shaped turn-out from a stub turn
    callable (TF-154's fixture pattern)."""
    return {
        "text": f"stub answer to: {user_text}",
        "response_id": "resp_stub", "turn_id": "turn_stub",
        "mode": "stub_mode", "enforcement_passed": True,
        "citations": [], "token_count": 3,
        "contract_version": API_VERSION,
        "trace": None, "trace_format": None,
        "trace_status": "unavailable: stub turn callable",
    }


class _RecordingCognitionAPI:
    """A recording stand-in for TF-154's
    CognitionAPI: files every turn's arguments
    and answers with the stub turn-out."""

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []

    def turn(self, user_text: str, **kwargs: Any) -> dict[str, Any]:
        self.turns.append({"user_text": user_text, **kwargs})
        return _stub_turn_out(user_text)


def _knowledge_binding() -> tuple[RunmalBinding, Any]:
    """An enabled binding over a knowledge-only
    core (TF-154's TF-084 fixture shape) and a
    temp store. Returns the binding and the
    temp home."""
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.knowledge.store import MemoryStore

    tmp = tempfile.mkdtemp(prefix="tf156-")
    core = ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    store = MemoryStore(memory_dir=Path(tmp) / "store")
    binding = RunmalBinding(
        cognition_api=CognitionAPI(core=core),
        memory_api=MemoryAPI(store=store),
        enabled=True,
    )
    return binding, tmp


def _temp_memory_binding() -> tuple[RunmalBinding, Any, str]:
    """An enabled binding whose memory side is a
    TF-155 MemoryAPI over a fresh temp store
    (the cognition side stays unbuilt unless a
    cognition call lands)."""
    from thoughtforge.knowledge.store import MemoryStore

    tmp = tempfile.mkdtemp(prefix="tf156-")
    store = MemoryStore(memory_dir=Path(tmp) / "store")
    binding = RunmalBinding(
        memory_api=MemoryAPI(store=store), enabled=True)
    return binding, store, tmp


# ── The filed probes ─────────────────────────────────────────────────────────

RUNMAL_BINDING_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "disabled_refusal_touches_nothing",
     "kind": "disabled_refusal",
     "detail": "a default binding refuses every "
               "routed call as data (ok=False, the "
               "error naming the binding disabled), "
               "and counting factories prove neither "
               "API is ever built"},
    {"id": "enabled_cognition_routes_by_the_filed_rule",
     "kind": "cognition_round_trip",
     "detail": "an enabled binding routes a chat "
               "call through the injected CognitionAPI "
               "by the filed assembly rule (last user "
               "message as user_text, the rest as "
               "history in order, names riding) and "
               "returns the turn with the filed "
               "text_delta / usage / done events"},
    {"id": "enabled_memory_round_trip_temp_store",
     "kind": "memory_round_trip",
     "detail": "an enabled binding adds through "
               "TF-155 (stamped manual_entry), finds "
               "the record through TF-155's inspect, "
               "and exports the verified bundle — all "
               "through the binding, on a temp store"},
    {"id": "backend_table_row_consistency",
     "kind": "backend_table",
     "detail": "a fake registry shaped on the filed "
               "row registers the fake ThoughtForge "
               "backend under \"thoughtforge\" with no "
               "name argument, answers get / keys as "
               "filed, and the row's abstract methods "
               "and FILED_MODEL's backend_key agree"},
    {"id": "models_and_health_are_honest",
     "kind": "honest_answers",
     "detail": "list_models returns exactly the "
               "filed model (chat only, streaming "
               "False, no samplers claimed, token "
               "counts estimated) and health_check "
               "returns the filed health — reporting "
               "the binding, claiming no live model"},
    {"id": "params_not_passed_and_invalid_raises_named",
     "kind": "params_and_invalid",
     "detail": "filed sampling params reach no "
               "turn (never passed, never claimed); "
               "an invalid enabled request (no user "
               "message, an unfiled params key, an "
               "empty list) raises ValueError naming "
               "the problem; the same garbage to a "
               "disabled binding returns the refusal "
               "as data instead"},
    {"id": "doc_matches_render_and_version_is_tf034",
     "kind": "doc_and_version",
     "detail": "the filed doc is the render "
               "verbatim, the contract version is "
               "TF-034's API_VERSION with its surface "
               "clean, and DEFAULT_ENABLED is False"},
)


def run_runmal_binding_probes() -> list[dict[str, Any]]:
    """Run the filed probes; each returns its
    verdict."""
    results: list[dict[str, Any]] = []
    for probe in RUNMAL_BINDING_PROBES:
        kind = probe["kind"]
        try:
            if kind == "disabled_refusal":
                built: list[str] = []

                def cog_factory() -> Any:
                    built.append("cognition")
                    return _RecordingCognitionAPI()

                def mem_factory() -> Any:
                    built.append("memory")
                    return MemoryAPI()

                binding = RunmalBinding(
                    cognition_factory=cog_factory,
                    memory_factory=mem_factory)
                outs = [
                    binding.chat(
                        [{"role": "user", "content": "hi"}]),
                    binding.list_models(),
                    binding.health_check(),
                    binding.memory_call(
                        "add", {"kind": "user_fact",
                                "summary": "x"}),
                ]
                passed = binding.enabled is False \
                    and built == [] \
                    and all(
                        o.get("ok") is False
                        and o.get("status") == "refused"
                        and o.get("error") == DISABLED_ERROR
                        and binding_response_problems(o) == []
                        for o in outs)
            elif kind == "cognition_round_trip":
                fake = _RecordingCognitionAPI()
                binding = RunmalBinding(
                    cognition_api=fake, enabled=True)
                out = binding.chat([
                    {"role": "system", "content": "be terse",
                     "name": "ops"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "second"},
                ])
                passed = out.get("ok") is True \
                    and fake.turns == [{
                        "user_text": "second",
                        "history": [
                            {"role": "system",
                             "content": "be terse",
                             "name": "ops"},
                            {"role": "user",
                             "content": "first"},
                            {"role": "assistant",
                             "content": "reply"},
                        ]}] \
                    and out["turn"]["text"] \
                    == "stub answer to: second" \
                    and [e["type"] for e in out["events"]] \
                    == ["text_delta", "usage", "done"] \
                    and out["events"][0]["text"] \
                    == "stub answer to: second" \
                    and out["events"][1]["total_tokens"] == 3 \
                    and out["events"][2]["reason"] \
                    == DONE_REASON \
                    and binding_response_problems(out) == []
            elif kind == "memory_round_trip":
                binding, _store, tmp = _temp_memory_binding()
                added = binding.memory_add(
                    "user_fact", summary="ferry at 07:30",
                    record_id="fct_tf156")
                found = binding.memory_inspect(
                    record_id="fct_tf156")
                exported = binding.memory_export(
                    destination=str(Path(tmp) / "bundle.json"))
                passed = added.get("ok") is True \
                    and added["memory"]["status"] == "added" \
                    and added["memory"]["record"]["source"] \
                    == "manual_entry" \
                    and found["memory"]["status"] == "found" \
                    and exported["memory"]["verified"] is True \
                    and binding_response_problems(added) == [] \
                    and binding_response_problems(found) == [] \
                    and binding_response_problems(exported) == []
            elif kind == "backend_table":
                registry = _FakeRegistry()
                row = BACKEND_TABLE_ROWS[0]
                key = registry.register(_fake_backend())
                alias = registry.register(
                    _fake_backend(), name="alias")
                passed = key == BACKEND_KEY \
                    and alias == "alias" \
                    and registry.keys() == ["alias", "thoughtforge"] \
                    and registry.get("thoughtforge")[
                        "abstract_methods"] \
                    == ["chat_stream", "list_models",
                        "health_check"] \
                    and registry.get("thoughtforge")[
                        "default_model"] is None \
                    and FILED_MODEL["backend_key"] == BACKEND_KEY \
                    and backend_row_problems(row) == []
            elif kind == "honest_answers":
                binding = RunmalBinding(
                    cognition_api=_RecordingCognitionAPI(),
                    enabled=True)
                models = binding.list_models()
                health = binding.health_check()
                caps = (models.get("models") or [{}])[0].get(
                    "capabilities") or {}
                passed = models.get("models") == [FILED_MODEL] \
                    and caps.get("chat") is True \
                    and caps.get("streaming") is False \
                    and caps.get("supported_samplers") == [] \
                    and caps.get("token_count_quality") \
                    == "estimated" \
                    and health.get("health") == FILED_HEALTH \
                    and health["health"]["ok"] is True \
                    and health["health"]["latency_ms"] is None \
                    and "no live model health is claimed" in \
                    health["health"]["detail"] \
                    and binding_response_problems(models) == [] \
                    and binding_response_problems(health) == []
            elif kind == "params_and_invalid":
                fake = _RecordingCognitionAPI()
                binding = RunmalBinding(
                    cognition_api=fake, enabled=True)
                out = binding.chat(
                    [{"role": "user", "content": "hi"}],
                    params={"temperature": 0.7, "seed": 3})
                named: list[bool] = []
                for call, needle in (
                        (lambda: binding.chat(
                            [{"role": "assistant",
                              "content": "hi"}]), "user"),
                        (lambda: binding.chat(
                            [{"role": "user", "content": "hi"}],
                            params={"surprise": 1}), "surprise"),
                        (lambda: binding.chat([]), "non-empty")):
                    try:
                        call()
                        named.append(False)
                    except ValueError as exc:
                        named.append(needle in str(exc))
                refused = RunmalBinding().chat(
                    [{"role": "assistant", "content": "hi"}])
                passed = out.get("ok") is True \
                    and fake.turns == [{"user_text": "hi"}] \
                    and all(named) \
                    and refused.get("ok") is False \
                    and refused.get("error") == DISABLED_ERROR
            elif kind == "doc_and_version":
                doc = _REPO_ROOT / RUNMAL_BINDING_DOC
                passed = doc.exists() \
                    and doc.read_text(encoding="utf-8") \
                    == render_runmal_binding_markdown() \
                    and runmal_binding_contract()[
                        "contract_version"] == API_VERSION \
                    and verify_api_surface() == [] \
                    and DEFAULT_ENABLED is False
            else:  # pragma: no cover — the checker names unknown kinds
                passed = False
        except Exception:
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the runmal-binding doc, verbatim) ───────────────────

def render_runmal_binding_markdown() -> str:
    """Render the runmal-binding doc
    deterministically — the exact text filed as
    docs/architecture/RUNMAL_BINDING.md. This is
    the artefact the parity campaign's P-slices
    consume."""
    lines: list[str] = []
    lines.append("# Rúnmál Binding — Rúnmál's Calls "
                 "Through ThoughtForge (TF-156)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/runmal_binding.py` — the "
        "filed doc IS the render: regenerate with "
        "`python3 -m thoughtforge.runmal_binding`. "
        "The checker (`runmal_binding_problems()`) "
        "re-derives every claim live from the tree "
        "and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{RUNMAL_BINDING_FORMAT}`")
    lines.append(
        f"Contract version: `{API_VERSION}` (TF-034's "
        "`API_VERSION` — consumed, not reinvented)")
    lines.append(f"Backend key: `{BACKEND_KEY}`")
    lines.append(
        f"Default enabled: {DEFAULT_ENABLED} — the "
        "binding is OFF by default; enabling is an "
        "explicit act (`RunmalBinding(enabled=True)` "
        "or `binding.enable()`)")
    lines.append(f"Calls: {len(CALL_ROWS)} "
                 f"({len(ROUTED_CALL_IDS)} routed); field "
                 f"rows: {len(FIELD_ROWS)}; rules: "
                 f"{len(RUNMAL_BINDING_RULES)}")
    lines.append(f"Probes: {len(RUNMAL_BINDING_PROBES)}/"
                 f"{len(RUNMAL_BINDING_PROBES)}")
    lines.append("")
    lines.append("## 1. One binding")
    lines.append("")
    lines.append(
        "`RunmalBinding` wraps TF-154's `CognitionAPI` "
        "and TF-155's `MemoryAPI` — each injectable "
        "directly, by factory, or via a `turn_callable` "
        "/ `memory_store` passthrough, and both built "
        "lazily: constructing the binding builds "
        "neither API. While disabled, every routed "
        "call returns a truthful refusal as data "
        "(`ok=False`, `status=\"refused\"`, the "
        "`error` naming the binding as disabled) and "
        "touches neither API. When enabled, cognition "
        "calls route through the CognitionAPI and "
        "memory calls through the MemoryAPI — called, "
        "never reimplemented. An invalid request "
        "raises `ValueError` naming the problem(s), "
        "judged by the pure validators before any API "
        "is built or called. Every response is a "
        "plain JSON-safe detached dict: `call`, `ok`, "
        "`status`, `backend_key`, `contract_version` "
        "(TF-034's `API_VERSION`), `error`, plus the "
        "call's payload (`turn` + `events` for chat, "
        "`models` for list_models, `health` for "
        "health_check, the TF-155 response verbatim "
        "in `memory` for memory calls). No Rúnmál "
        "code is imported or copied anywhere: "
        "Rúnmál's shapes are filed below as literal "
        "data read from its public backend contract.")
    lines.append("")
    lines.append("## 2. The backend-table row")
    lines.append("")
    lines.append("| Row | Backend key | Registry | "
                 "Interface | Abstract methods | "
                 "Default model | Enabled by default | "
                 "What it is |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in BACKEND_TABLE_ROWS:
        methods = ", ".join(
            f"`{m}`" for m in row["abstract_methods"])
        enabled = "yes" if row["enabled_by_default"] else "no"
        lines.append(
            f"| {row['id']} | {row['backend_key']} | "
            f"{row['registry']} | {row['interface']} | "
            f"{methods} | {row['default_model']} | "
            f"{enabled} | {row['summary']} |")
    lines.append("")
    lines.append(
        "Registry semantics, filed from "
        "`runmal/backends/registry.py`: "
        "`register(backend, name=None)` returns `name "
        "or backend.backend_key` — a ThoughtForge "
        "backend registers under `thoughtforge` with "
        "no name argument; `get(key)` returns the "
        "backend; `keys()` is sorted. The registry's "
        "aggregations (`list_all_models`, "
        "`list_models_for`, `health_all`, "
        "`health_for`, `health_dicts`, `token_count`, "
        "`token_count_from_usage`) reach the entry "
        "through the same interface.")
    lines.append("")
    lines.append("## 3. The calls")
    lines.append("")
    lines.append("| Call | Surface | Rúnmál side | "
                 "ThoughtForge route | Binding method | "
                 "Routed |")
    lines.append("|---|---|---|---|---|---|")
    for row in CALL_ROWS:
        routed = "yes" if row["routed"] else "no"
        method = row["binding_method"] or "—"
        lines.append(
            f"| {row['id']} | {row['surface']} | "
            f"{row['runmal_side']} | "
            f"{row['thoughtforge_route']} | {method} | "
            f"{routed} |")
    lines.append("")
    lines.append("## 4. Field mapping — in (Rúnmál -> turn-in)")
    lines.append("")
    lines.append("| Field | Source | Target | "
                 "Disposition | What it is |")
    lines.append("|---|---|---|---|---|")
    for field in field_rows_for("in"):
        lines.append(
            f"| {field['id']} | {field['source']} | "
            f"{field['target']} | "
            f"{field['disposition']} | "
            f"{field['summary']} |")
    lines.append("")
    lines.append("## 5. Field mapping — out (turn-out -> reply)")
    lines.append("")
    lines.append("| Field | Source | Target | "
                 "Disposition | What it is |")
    lines.append("|---|---|---|---|---|")
    for field in field_rows_for("out"):
        lines.append(
            f"| {field['id']} | {field['source']} | "
            f"{field['target']} | "
            f"{field['disposition']} | "
            f"{field['summary']} |")
    lines.append("")
    lines.append("## 6. The rules")
    lines.append("")
    for rule in RUNMAL_BINDING_RULES:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## 7. Docket")
    lines.append("")
    lines.append(
        "TF-156 has **no share** of TF-016 and found "
        "no new defect in existing code, so this "
        "slice does not edit the docket and adds no "
        "entry. TF-016 totals stand unchanged (137 "
        "entries — 31 OPEN / 1 OPEN-ENVIRONMENT / "
        "101 FIXED / 4 CLOSED-BY-VERDICT), verified "
        "live against the docket header and filed "
        "in the open here.")
    lines.append("")
    lines.append("## 8. The probes")
    lines.append("")
    for probe in RUNMAL_BINDING_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 9. Filed limits")
    lines.append("")
    lines.append(
        "- Proof grade FIXTURE: the checker and "
        "probes run a fake Rúnmál consumer shaped on "
        "the filed ChatBackend / registry data. NO "
        "live Rúnmál process was routed and NO "
        "Rúnmál code was imported or copied; live "
        "end-to-end routing is NOT PROVEN — "
        "TF-170's Part IX gate owns the "
        "at-least-one-live proof, never claimed "
        "here.")
    lines.append(
        "- The binding is synchronous library data "
        "shaped on Rúnmál's async interface "
        "(`chat_stream` is an async iterator in "
        "Rúnmál): a routed chat returns its filed "
        "events as one list with a single text "
        "delta, and cancellation is filed no-home. "
        "No streaming is pretended — the filed "
        "model's `streaming` is False.")
    lines.append(
        "- The memory delete is partial by filing: "
        "Rúnmál's `delete_memory` is a targeted "
        "single-record delete and TF-155's prune "
        "is a lifecycle pass; the binding routes "
        "only what prune honestly is and pretends "
        "no targeted delete.")
    lines.append(
        "- Sampling params have no turn-in home: "
        "they are accepted in the call shape, never "
        "passed to TF-154 and never claimed "
        "honored; the filed model's "
        "`supported_samplers` is empty.")
    lines.append(
        "- TF-157 owns the Samtal binding; TF-161 "
        "owns binding fixtures in depth; TF-162 "
        "owns versioning enforcement in depth; "
        "TF-163 owns the local-only proof for "
        "bindings. TF-156 files the Rúnmál binding "
        "shape, its off-by-default routing and "
        "this doc for the parity campaign's "
        "P-slices.")
    lines.append(
        "- No canon data was touched (TF-017); no "
        "README.md anywhere was touched; no Docker "
        "was used; nothing under the Rúnmál tree "
        "was edited.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_runmal_binding_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / RUNMAL_BINDING_DOC
    doc.write_text(render_runmal_binding_markdown(),
                   encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.runmal_binding`."""
    doc = write_runmal_binding_doc()
    print(f"runmal binding doc regenerated: -> "
          f"{RUNMAL_BINDING_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def runmal_binding_problems() -> list[str]:
    """Re-derive TF-156's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    if RUNMAL_BINDING_FORMAT != "thoughtforge-runmal-binding/1":
        problems.append(
            f"the format is {RUNMAL_BINDING_FORMAT!r}, not "
            "thoughtforge-runmal-binding/1")
    if DEFAULT_ENABLED is not False:
        problems.append("DEFAULT_ENABLED is not False")
    if BACKEND_KEY != "thoughtforge":
        problems.append(
            f"BACKEND_KEY is {BACKEND_KEY!r}, not "
            "'thoughtforge'")
    if not DISABLED_ERROR or "disabled" not in DISABLED_ERROR:
        problems.append(
            "DISABLED_ERROR does not name the binding as "
            "disabled")
    for row in BACKEND_TABLE_ROWS:
        problems.extend(backend_row_problems(row))
    if len(BACKEND_TABLE_ROWS) != 1:
        problems.append(
            "the BACKEND_TABLE_ROWS filing is not exactly "
            "one row")
    else:
        row = BACKEND_TABLE_ROWS[0]
        if row["abstract_methods"] != [
                "chat_stream", "list_models", "health_check"]:
            problems.append(
                "the backend row's abstract methods are "
                f"not Rúnmál's three: {row['abstract_methods']}")
        if row["concrete_methods"] != [
                "count_tokens", "filter_params"]:
            problems.append(
                "the backend row's concrete methods are "
                f"not Rúnmál's two: {row['concrete_methods']}")
    if FILED_MODEL.get("backend_key") != BACKEND_KEY \
            or FILED_MODEL.get("model_id") != "thoughtforge":
        problems.append("FILED_MODEL is not the thoughtforge model")
    caps = FILED_MODEL.get("capabilities") or {}
    if caps.get("chat") is not True \
            or caps.get("streaming") is not False \
            or caps.get("supported_samplers") != [] \
            or caps.get("token_count_quality") != "estimated":
        problems.append(
            f"FILED_MODEL's capabilities are not the "
            f"honest filed set: {caps}")
    if FILED_HEALTH.get("ok") is not True \
            or FILED_HEALTH.get("latency_ms") is not None \
            or API_VERSION not in str(FILED_HEALTH.get("detail")):
        problems.append("FILED_HEALTH is not as filed")

    call_ids = [r["id"] for r in CALL_ROWS]
    expected_calls = [
        "chat_stream", "list_models", "health_check",
        "count_tokens", "filter_params", "memory_add",
        "memory_inspect", "memory_prune", "memory_export"]
    if call_ids != expected_calls \
            or len(call_ids) != len(set(call_ids)):
        problems.append(
            f"the CALL_ROWS filing is not as filed: {call_ids}")
    for row in CALL_ROWS:
        problems.extend(call_row_problems(row))
        if row["routed"] and row["surface"] == "cognition" \
                and "CognitionAPI" not in row["thoughtforge_route"]:
            problems.append(
                f"call {row['id']}: the cognition route "
                "does not name TF-154's CognitionAPI")
        if row["routed"] and row["surface"] == "memory" \
                and "MemoryAPI" not in row["thoughtforge_route"]:
            problems.append(
                f"call {row['id']}: the memory route does "
                "not name TF-155's MemoryAPI")
    for op in MEMORY_OP_IDS:
        if f"memory_{op}" not in ROUTED_CALL_IDS:
            problems.append(
                f"TF-155's op {op} has no routed binding row")

    field_ids = [f["id"] for f in FIELD_ROWS]
    if len(field_ids) != len(set(field_ids)):
        problems.append("the FIELD_ROWS filing has duplicate ids")
    for field in FIELD_ROWS:
        problems.extend(field_row_problems(field))
    turn_in_ids = [f["id"] for f in TURN_IN_FIELDS]
    for field_id in turn_in_ids:
        if not any(field_id in f["target"]
                   for f in field_rows_for("in")):
            problems.append(
                f"turn-in field {field_id} has no in-row "
                "target — a field without a disposition")
    turn_out_ids = [f["id"] for f in TURN_OUT_FIELDS]
    out_ids = {f["id"] for f in field_rows_for("out")}
    for field_id in turn_out_ids:
        if f"turn.{field_id}" not in out_ids:
            problems.append(
                f"turn-out field {field_id} has no out-row "
                "— a field without a disposition")
    for param in SAMPLING_PARAM_FIELDS:
        row = next(
            (f for f in FIELD_ROWS
             if f["id"] == f"params.{param}"), None)
        if row is None \
                or row.get("disposition") != "no_home_not_passed":
            problems.append(
                f"SamplingParams field {param} is not "
                "filed no-home in FIELD_ROWS")

    rule_ids = [r["id"] for r in RUNMAL_BINDING_RULES]
    if len(rule_ids) != len(set(rule_ids)) \
            or len(RUNMAL_BINDING_RULES) != 7:
        problems.append(
            "the RUNMAL_BINDING_RULES filing is not seven "
            "unique rules")
    for rule in RUNMAL_BINDING_RULES:
        problems.extend(binding_rule_problems(rule))
    probe_ids = [p["id"] for p in RUNMAL_BINDING_PROBES]
    if len(probe_ids) != len(set(probe_ids)) \
            or len(RUNMAL_BINDING_PROBES) != 7:
        problems.append(
            "the RUNMAL_BINDING_PROBES filing is not "
            "seven unique probes")

    # The contract is TF-034-versioned and its
    # surface is clean.
    contract = runmal_binding_contract()
    if contract["contract_version"] != API_VERSION \
            or contract["default_enabled"] is not False:
        problems.append(
            "the filed contract is not TF-034-versioned "
            "and off by default")
    surface = verify_api_surface()
    if surface:
        problems.append(
            f"TF-034: verify_api_surface() is not clean: "
            f"{surface[:3]}")
    try:
        json.dumps(contract)
    except (TypeError, ValueError) as exc:
        problems.append(f"the contract is not JSON-safe: {exc}")

    # The neighbours stay clean live underneath.
    from thoughtforge.cognition.runmal_cards import (
        runmal_cards_problems,
    )
    from thoughtforge.service_shape import service_shape_problems

    for name, checker in (
            ("TF-153 service_shape", service_shape_problems),
            ("TF-154 cognition_api", cognition_api_problems),
            ("TF-155 memory_api", memory_api_problems),
            ("TF-147 runmal_cards", runmal_cards_problems)):
        found = checker()
        if found:
            problems.append(
                f"{name}: its checker is not clean live: "
                f"{found[:3]}")

    # A disabled binding refuses, building nothing.
    try:
        built: list[str] = []

        def cog_factory() -> Any:
            built.append("cognition")
            return _RecordingCognitionAPI()

        def mem_factory() -> Any:
            built.append("memory")
            return MemoryAPI()

        binding = RunmalBinding(
            cognition_factory=cog_factory,
            memory_factory=mem_factory)
        refused = binding.chat(
            [{"role": "user", "content": "ferry?"}])
        problems.extend(
            f"disabled refusal: {p}"
            for p in binding_response_problems(refused))
        if refused.get("error") != DISABLED_ERROR or built:
            problems.append(
                f"the disabled binding did not refuse "
                f"untouched: built={built}, out={refused}")
    except Exception as exc:
        problems.append(f"disabled check failed: {exc}")

    # An enabled round-trip: a knowledge-only core
    # chat and a temp-store memory add / find,
    # through the binding only.
    try:
        binding, _tmp = _knowledge_binding()
        out = binding.chat([
            {"role": "user",
             "content": "What time does the ferry leave?"}])
        problems.extend(
            f"round-trip chat: {p}"
            for p in binding_response_problems(out))
        if out.get("ok") is not True \
                or out.get("turn", {}).get("mode") \
                != "knowledge_only" \
                or [e["type"] for e in out.get("events", [])] \
                != ["text_delta", "usage", "done"]:
            problems.append(
                f"round-trip chat is not a knowledge_only "
                f"turn with the filed events: {out}")
        added = binding.memory_add(
            "user_fact", summary="ferry at 07:30",
            record_id="fct_tf156_check")
        if added.get("ok") is not True \
                or added.get("memory", {}).get(
                    "record", {}).get("source") \
                != "manual_entry":
            problems.append(
                f"round-trip memory add is not a stamped "
                f"add: {added}")
        found = binding.memory_inspect(
            record_id="fct_tf156_check")
        if found.get("memory", {}).get("status") != "found":
            problems.append(
                f"round-trip memory find: {found}")
    except Exception as exc:
        problems.append(f"round-trip failed: {exc}")

    # The probes all behave.
    probes = run_runmal_binding_probes()
    if len(probes) != len(RUNMAL_BINDING_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the runmal binding probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The docket stands unchanged.
    import re

    docket_text = _DOCKET.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", docket_text)
    if "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, 101 FIXED" \
            not in normalized:
        problems.append(
            "docket: the header totals no longer file 137 "
            "entries — 31 OPEN / 1 OPEN-ENVIRONMENT / 101 "
            "FIXED; TF-156 must not move them")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / RUNMAL_BINDING_DOC
    if not doc.exists():
        problems.append(
            f"the runmal binding doc {RUNMAL_BINDING_DOC} "
            "is missing")
    elif doc.read_text(encoding="utf-8") \
            != render_runmal_binding_markdown():
        problems.append(
            f"the filed doc {RUNMAL_BINDING_DOC} is not "
            "the render verbatim (regenerate with "
            "python3 -m thoughtforge.runmal_binding)")
    return problems


def runmal_binding_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for
    consumers."""
    probes = run_runmal_binding_probes()
    return {
        "format": RUNMAL_BINDING_FORMAT,
        "doc": RUNMAL_BINDING_DOC,
        "contract": runmal_binding_contract(),
        "measured": {
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "BACKEND_KEY",
    "BACKEND_ROW_IDS",
    "BACKEND_TABLE_ROWS",
    "CALL_IDS",
    "CALL_ROWS",
    "DEFAULT_ENABLED",
    "DISABLED_ERROR",
    "DONE_REASON",
    "FIELD_IDS",
    "FIELD_ROWS",
    "FILED_HEALTH",
    "FILED_MODEL",
    "ROUTED_CALL_IDS",
    "RUNMAL_BINDING_DOC",
    "RUNMAL_BINDING_FORMAT",
    "RUNMAL_BINDING_PROBES",
    "RUNMAL_BINDING_RULES",
    "RULE_IDS",
    "SAMPLING_PARAM_FIELDS",
    "RunmalBinding",
    "assemble_turn_in",
    "backend_row_problems",
    "binding_response_problems",
    "binding_rule_problems",
    "call_row_problems",
    "chat_request_problems",
    "field_row_problems",
    "field_rows_for",
    "main",
    "render_runmal_binding_markdown",
    "reply_events",
    "run_runmal_binding_probes",
    "runmal_binding_as_data",
    "runmal_binding_contract",
    "runmal_binding_problems",
    "write_runmal_binding_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
