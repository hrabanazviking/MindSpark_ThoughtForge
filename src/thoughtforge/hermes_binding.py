"""
TF-158 — the Hermes binding, filed:
ThoughtForge as a cognition provider for
the Hermes hack, shaped on the volmarr-core
contract the 2026-10-30 refresh actually
offers.

TF-158's roadmap text: *Hermes binding: the
volmarr-core shaped contract — ThoughtForge
as a cognition provider for the Hermes hack
(consumes the 2026-10-30 refresh's plugin
surfaces).* It is Part IX slice 6, following
TF-157's Samtal binding, TF-156's Rúnmál
binding, TF-155's memory API, TF-154's
cognition API and TF-153's service shape,
and it follows TF-157's estate pattern in
kind.

The measured start, probed live before
anything was built:

  * **No binding existed.** Nothing on the
    ThoughtForge side filed a Hermes route:
    TF-154 filed the cognition API and TF-155
    the memory API as standalone service
    shapes, TF-156 / TF-157 filed the Rúnmál
    and Samtal bindings — no
    ``thoughtforge.hermes_binding`` module
    existed on this tree.
  * **``volmarr-core`` IS a literal name in
    the Hermes tree, read READ-ONLY from
    ``~/workspace/hermes-hack-build`` (never
    edited, never imported).** It is the
    opt-in composition plugin at
    ``plugins/volmarr-core/``:
    ``plugin.yaml`` files ``manifest_version:
    2``, ``name: volmarr-core``, ``version:
    "0.26.0"``, hooks ``on_session_start`` /
    ``on_session_end`` / ``on_session_finalize``
    / ``pre_tool_call`` / ``post_tool_call`` /
    ``pre_llm_call`` / ``post_llm_call`` and a
    large ``config_schema``; ``__init__.py``'s
    ``register(ctx)`` wires bridges through
    ``ctx.register_hook``, tool registrars and
    ``ctx.register_cli_command(name="volmarr",
    ...)``. The architecture law it lives
    under (``ARCHITECTURE_VOLMARR.md``):
    Hermes core stays a narrow waist — the
    shell owns turns, sessions, provider
    resolution, tools, gateways, cron and
    plugin contracts; a composition plugin
    owns registration, lifecycle wiring,
    configuration validation and adapter
    construction, never domain logic or
    durable state; the cognition domain owns
    local/cloud route selection, budgets,
    escalation and telemetry and must not
    reimplement Hermes providers; LLM
    providers are replaceable cognitive
    organs. The plugin surfaces the
    2026-10-30 refresh (``CUSTOMIZATIONS.md``:
    upstream ``6bb377a5`` merged 2026-10-30)
    offers an external cognition provider
    are filed below as literal data:

      - **volmarr-core's cognition attachment**
        (``plugins/volmarr-core/cognition.py``):
        a profile-scoped, loopback-only
        OpenAI-compatible endpoint (default
        ``http://127.0.0.1:18434/v1``) —
        ``GET /models`` answers a catalog
        (``object`` "list", ``data[].id``) and
        ``POST /chat/completions`` takes a
        bounded, non-streaming, tool-free
        request ``{model, messages, max_tokens}``
        whose messages carry ONLY ``role``
        (system / user / assistant) and
        ``content`` (1..128 rows, total
        content <= 64 KiB, the last message a
        user message); the reply is parsed
        strictly (exactly one choice,
        ``message.content``, a ``finish_reason``
        string, integer ``usage.prompt_tokens``
        / ``completion_tokens``). Its CLI face
        is ``hermes volmarr cognition
        {health|route|execute}``.
      - **cognition routing**
        (``plugins/volmarr-core/routing.py``):
        schema ``runeforge.cognition.route``
        version 2 (1 also supported) — a
        metadata-only, content-free request
        (``operation`` / ``input_bytes`` /
        ``complexity`` / capability flags /
        ``mode`` auto|local|deep) decided to a
        route (deterministic / local / cloud /
        blocked) with a reason; the router
        chooses a capability tier without
        choosing or calling a cloud provider.
      - **model-provider plugins**
        (``plugins/model-providers/<name>/``):
        ``plugin.yaml`` with ``kind:
        model-provider`` plus a
        ``ProviderProfile`` subclass registered
        through ``providers.register_provider``
        — the ``custom`` provider is exactly
        "any OpenAI-compatible local endpoint".
      - **memory-provider plugins**
        (``agent/memory_provider.py``'s
        ``MemoryProvider`` ABC, shipped in
        ``plugins/memory/<name>/``, ONE
        external provider at a time selected
        by ``memory.provider``): ``name`` /
        ``is_available`` / ``initialize`` /
        ``system_prompt_block`` / ``prefetch``
        / ``sync_turn`` / tool dispatch /
        ``shutdown``, driven by Hermes's
        MemoryManager.
      - **context engines**
        (``plugins/context_engine/``):
        a ``ContextEngine`` captured through
        ``register_context_engine``, selected
        by ``context.engine``.
      - **the context packet**
        (``plugins/volmarr-core/context_packet.py``):
        schema ``runeforge.memory-packet``
        version 2, assembled from
        ``packet_sources`` producers on the
        ``pre_llm_call`` user side.

    Credentials never ride a cognition
    request: volmarr-core reads its endpoint
    key from a profile file
    (``cognition_api_key_file``, default
    ``secrets/aesir.key``) and the fork's
    ``kista-secret-source`` plugin resolves
    ``kista://`` bindings — credential
    custody is Hermes/Kista's, and TF-160
    owns it on the ThoughtForge side. NO
    Hermes code is imported or copied
    anywhere in this module, and nothing in
    the Hermes tree was edited.
  * **The neighbours stood clean live**:
    ``service_shape_problems()``,
    ``cognition_api_problems()``,
    ``memory_api_problems()``,
    ``runmal_binding_problems()`` and
    ``samtal_binding_problems()`` all returned
    [] on this tree before the build.

**What this slice files**

  * ``PROVIDER_ROWS`` — the plugin surfaces
    ThoughtForge plugs into, as literal data:
    the volmarr-core composition plugin, the
    model-provider profile surface, the
    cognition-routing contract, the
    memory-provider surface and the context
    engine surface — each with its real
    registration shape; the context engine is
    filed as a surface the binding does NOT
    answer (context compression is Hermes's
    engine's own act, not cognition).
  * ``RULE_ROWS`` — the binding's law, as
    literal data: volmarr-core is real and
    named (no Hermes API invented); the
    composition boundary holds (the binding
    is an external provider behind Hermes's
    surfaces — it patches no Hermes core or
    facade and decides no route); off by
    default; called, never reimplemented; no
    Hermes code imported or copied; every
    field disposed; responses detached,
    JSON-safe, TF-034-versioned.
  * ``CALL_ROWS`` — every Hermes-side call
    mapped, as literal data. A routed local
    cognition turn routes to a TF-154
    ``CognitionAPI`` turn and returns as the
    filed completion shape; the catalog and
    health calls answer from the filed
    ``FILED_CATALOG`` / ``FILED_HEALTH`` rows
    (the binding's own state, never a live
    model); memory calls route to TF-155
    ``MemoryAPI``'s four ops, called, never
    reimplemented. Tool execution, gateway /
    messaging, session & state custody,
    credential custody, provider resolution &
    routing decisions and the memory-provider
    lifecycle are filed NOT routed, each with
    its reason — they are Hermes's own acts
    under its own architecture law.
  * ``FIELD_ROWS`` — the Hermes completion
    request -> turn-in mapping and the
    turn-out -> completion mapping, as
    literal data: every field with a
    disposition, none silently dropped or
    invented. The filed rule: the LAST
    message with role ``"user"`` supplies
    turn-in ``user_text``; every other
    message rides as turn-in ``history``
    (role / content verbatim — Hermes
    completion messages carry only those
    two fields); ``max_tokens`` and any
    credential are filed no-home.
  * ``HermesBinding`` — wraps a TF-154
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
    method (``disable()`` reverses it). Every
    response is detached JSON-safe data whose
    ``contract_version`` is TF-034's
    ``API_VERSION``, consumed.
  * ``hermes_binding_problems()`` — a checker
    that re-derives every claim live: the
    filings, the TF-034 surface (through the
    neighbour checkers), a disabled binding
    refusing with nothing built or called, a
    knowledge-only cognition round-trip and
    a temp-store memory round-trip through
    the binding, fake-consumer consistency,
    the filed probes, the docket header and
    the filed doc.

**Docket**: TF-158 has no share of TF-016 and
finds no new defect in existing code; it does
NOT edit ``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md``.
TF-016 totals stand unchanged (137 entries —
32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4
CLOSED-BY-VERDICT), verified live against
the docket header and filed in the open here
and in the render.

The human-readable doc is
``docs/architecture/HERMES_BINDING.md``,
filed verbatim as this module's render
(regenerate with
``python3 -m thoughtforge.hermes_binding``).
Proof grade: FIXTURE — a fake Hermes
consumer shaped on the filed plugin-surface
data; NO live Hermes process was routed and
NO Hermes code was imported or copied. Live
end-to-end routing is NOT PROVEN — TF-170's
Part IX gate owns the at-least-one-live
proof, never claimed here. TF-159 owns the
Ember binding; TF-160 owns credential
custody; TF-161 owns binding fixtures in
depth; TF-162 owns versioning enforcement
in depth.
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
from thoughtforge.runmal_binding import runmal_binding_problems
from thoughtforge.samtal_binding import samtal_binding_problems

HERMES_BINDING_FORMAT = "thoughtforge-hermes-binding/1"
HERMES_BINDING_DOC = "docs/architecture/HERMES_BINDING.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

#: The provider key ThoughtForge answers
#: under on every Hermes surface that names
#: a provider / model.
PROVIDER_KEY = "thoughtforge"

#: The composition plugin the contract is
#: shaped on — a literal name in the Hermes
#: tree (plugins/volmarr-core/plugin.yaml).
PLUGIN_NAME = "volmarr-core"

#: The binding is OFF unless someone enables
#: it explicitly — filed as data, proven by
#: the probes.
DEFAULT_ENABLED = False

#: The refusal every routed call returns
#: while the binding is disabled — as data,
#: naming the binding as disabled.
DISABLED_ERROR = (
    "hermes binding is disabled: the TF-158 "
    "binding is off by default — enable it "
    "explicitly (HermesBinding(enabled=True) "
    "or binding.enable()) before any call "
    "routes")

#: The roles a Hermes completion message may
#: carry (cognition.py's request contract).
MESSAGE_ROLES: tuple[str, ...] = (
    "system", "user", "assistant")

#: The fields a Hermes cognition request may
#: carry (cognition.py's AesirCompletionRequest
#: admits exactly these three).
REQUEST_FIELDS: tuple[str, ...] = (
    "model", "messages", "max_tokens")

#: The fields one Hermes completion message
#: carries — only role and content
#: (cognition.py refuses any other key).
MESSAGE_FIELDS: tuple[str, ...] = ("role", "content")

#: The binding's filed finish reason: Hermes's
#: strict reply parser requires a finish_reason
#: string and TF-154 files none, so the binding
#: files its own constant rather than inventing
#: a provider reason.
FINISH_REASON = "complete"


# ── The provider / plugin-surface rows, as literal data ──────────────────────
#
# One row per plugin surface the refresh
# offers. Each row files the surface's real
# registration shape, read READ-ONLY from the
# Hermes tree; the binding invents no Hermes
# API and registers nothing itself — it files
# the contract a provider answers behind
# those surfaces.

PROVIDER_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "provider_volmarr_core",
     "surface": "volmarr_core_plugin",
     "hermes_side": "plugins/volmarr-core — Volmarr's "
                    "opt-in composition plugin: "
                    "plugin.yaml (manifest_version 2, "
                    "name volmarr-core, version "
                    "0.26.0) and register(ctx) in "
                    "__init__.py; its cognition module "
                    "(cognition.py) attaches a "
                    "profile-scoped, loopback-only "
                    "OpenAI-compatible endpoint and "
                    "its CLI face is hermes volmarr "
                    "cognition {health|route|execute}",
     "registration": "ctx.register_hook(hook_name, "
                     "callback) for the filed hooks; "
                     "tool registrars over "
                     "ctx.register_tool; "
                     "ctx.register_cli_command(name="
                     "\"volmarr\", setup_fn, handler_fn) "
                     "— the composition plugin is the "
                     "only place that knows both the "
                     "Hermes registration APIs and the "
                     "concrete adapters "
                     "(ARCHITECTURE_VOLMARR.md)",
     "plugin_fields": [
         "manifest_version 2", "name volmarr-core",
         "hooks on_session_start / on_session_end / "
         "on_session_finalize / pre_tool_call / "
         "post_tool_call / pre_llm_call / "
         "post_llm_call",
         "config_schema cognition_base_url / "
         "cognition_api_key_file / "
         "cognition_local_max_tokens / ..."],
     "binding_method": "provider_turn(request)",
     "answers_binding": True,
     "enabled_by_default": False,
     "summary": "the named home surface: ThoughtForge "
                "answers as the local cognition "
                "provider behind volmarr-core's "
                "cognition attachment — a routed "
                "local turn lands on provider_turn, "
                "the catalog and health calls land on "
                "catalog() / health_check()"},
    {"id": "provider_model_provider",
     "surface": "model_provider_plugin",
     "hermes_side": "plugins/model-providers/<name>/ "
                    "— plugin.yaml with kind "
                    "model-provider plus a "
                    "ProviderProfile subclass; the "
                    "custom provider is any "
                    "OpenAI-compatible local endpoint "
                    "(Ollama / vLLM / llama.cpp shape)",
     "registration": "providers.register_provider"
                     "(profile) from the plugin's "
                     "__init__.py — a ProviderProfile "
                     "instance registered under its "
                     "provider name",
     "plugin_fields": [
         "plugin.yaml kind model-provider",
         "ProviderProfile (providers.base)",
         "fetch_models / create_client surfaces"],
     "binding_method": "provider_turn(request)",
     "answers_binding": True,
     "enabled_by_default": False,
     "summary": "the generic provider surface: the "
                "same filed turn contract serves as "
                "the OpenAI-compatible completion "
                "shape a ProviderProfile endpoint "
                "answers — one contract, both homes"},
    {"id": "provider_cognition_routing",
     "surface": "cognition_routing",
     "hermes_side": "plugins/volmarr-core/routing.py "
                    "— schema runeforge.cognition.route "
                    "version 2 (1 also supported): a "
                    "metadata-only, content-free "
                    "request decided to a route "
                    "(deterministic / local / cloud / "
                    "blocked) with a reason",
     "registration": "no registration of its own — "
                     "the router is volmarr-core's "
                     "internal application contract, "
                     "reached through the plugin's CLI "
                     "(hermes volmarr cognition route) "
                     "and its execute path",
     "plugin_fields": [
         "schema runeforge.cognition.route",
         "schema_version 1 / 2",
         "mode auto / local / deep",
         "routes deterministic / local / cloud / "
         "blocked"],
     "binding_method": "provider_turn(request)",
     "answers_binding": True,
     "enabled_by_default": False,
     "summary": "the gate, not the engine: the "
                "binding answers a turn only when "
                "Hermes's own routing decides the "
                "local tier — the binding never "
                "decides a route, never selects a "
                "cloud provider and never inspects a "
                "routing request's content (it has "
                "none)"},
    {"id": "provider_memory_provider",
     "surface": "memory_provider_plugin",
     "hermes_side": "agent/memory_provider.py's "
                    "MemoryProvider ABC, shipped in "
                    "plugins/memory/<name>/ and "
                    "selected by memory.provider — "
                    "ONE external provider at a time, "
                    "driven by Hermes's MemoryManager",
     "registration": "register(ctx) / the plugin "
                     "loader constructs one provider "
                     "instance per agent, bound to "
                     "its profile from initialize()'s "
                     "hermes_home",
     "plugin_fields": [
         "name / is_available / initialize",
         "system_prompt_block / prefetch / "
         "queue_prefetch / sync_turn",
         "handle_tool_call / shutdown"],
     "binding_method": "memory_call(op, request)",
     "answers_binding": True,
     "enabled_by_default": False,
     "summary": "the memory surface: ThoughtForge's "
                "store operations (TF-155's four "
                "ops) are what a memory provider "
                "backed by ThoughtForge would call — "
                "the provider lifecycle itself "
                "(initialize / prefetch / sync_turn / "
                "shutdown) is Hermes-driven and is "
                "filed NOT routed in CALL_ROWS"},
    {"id": "provider_context_engine",
     "surface": "context_engine_plugin",
     "hermes_side": "plugins/context_engine/ — a "
                    "ContextEngine captured through "
                    "register_context_engine and "
                    "selected by the context.engine "
                    "config key (the built-in "
                    "compressor is the default)",
     "registration": "register_context_engine"
                     "(engine) inside the engine "
                     "plugin's register(ctx), or a "
                     "ContextEngine subclass found by "
                     "the loader",
     "plugin_fields": [
         "ContextEngine (agent.context_engine)",
         "register_context_engine",
         "config key context.engine"],
     "binding_method": None,
     "answers_binding": False,
     "enabled_by_default": False,
     "summary": "filed as a surface the binding does "
                "NOT answer: context compression is "
                "the context engine's own act — "
                "ThoughtForge is a cognition "
                "provider, not a compressor, and "
                "claims no context-engine seat"},
)

PROVIDER_ROW_IDS: tuple[str, ...] = tuple(
    r["id"] for r in PROVIDER_ROWS)


# ── The filed honest answers (catalog / health) ──────────────────────────────

#: The one model the binding files for the
#: catalog — the OpenAI model-object shape
#: cognition.py's catalog parser reads
#: (object "list", data[].id). Honest: the
#: model IS the ThoughtForge cognition API.
FILED_MODEL: dict[str, Any] = {
    "id": PROVIDER_KEY,
    "object": "model",
    "owned_by": PROVIDER_KEY,
}

#: The catalog an enabled binding files —
#: exactly the shape volmarr-core's probe
#: validates (an OpenAI model list).
FILED_CATALOG: dict[str, Any] = {
    "object": "list",
    "data": [FILED_MODEL],
}

#: The health answer an enabled binding files
#: — AesirHealth-shaped as cognition.py's
#: as_dict() files it (healthy / status /
#: models / latency_ms / error_type), plus a
#: detail naming what it does NOT claim. It
#: reports the binding, never a live Hermes
#: process or a live model, and no credential
#: was read or validated for it.
FILED_HEALTH: dict[str, Any] = {
    "healthy": True,
    "status": "healthy",
    "models": [PROVIDER_KEY],
    "latency_ms": None,
    "error_type": None,
    "detail": "hermes binding enabled: cognition "
              "routes through TF-154's CognitionAPI "
              "and memory through TF-155's MemoryAPI "
              f"(contract {API_VERSION}); this answer "
              "reports the binding's own state, not a "
              "live Hermes process or a live model — "
              "no credential was read or validated "
              "and no live model health is claimed",
}


# ── The rules, as literal data ───────────────────────────────────────────────

RULE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "volmarr_core_real_and_named",
     "rule": "volmarr-core is a literal name in the "
             "Hermes tree (plugins/volmarr-core, "
             "plugin.yaml manifest_version 2, name "
             "volmarr-core, version 0.26.0): the "
             "binding is shaped on the plugin "
             "surfaces that tree actually files — "
             "its cognition attachment, its routing "
             "schema, the model-provider and "
             "memory-provider plugin surfaces — and "
             "invents no Hermes API; where this "
             "slice's brief said 'if volmarr-core "
             "does not exist as a literal name, say "
             "so', the measured truth is that it "
             "does exist, filed here with its "
             "evidence"},
    {"id": "composition_boundary_holds",
     "rule": "the composition boundary holds "
             "(ARCHITECTURE_VOLMARR.md): Hermes core "
             "stays a narrow waist and the shell owns "
             "turns, sessions, provider resolution, "
             "tools and gateways; the binding is an "
             "external cognition provider BEHIND "
             "those surfaces — it patches no Hermes "
             "core or facade, registers nothing "
             "itself, decides no route, resolves no "
             "provider and owns no Hermes domain's "
             "state"},
    {"id": "off_by_default",
     "rule": "the binding is off by default: "
             "DEFAULT_ENABLED is False, filed as "
             "data; enabling is an explicit act — "
             "the constructor flag enabled=True or "
             "the enable() method — and disable() "
             "returns the binding to refusal; while "
             "disabled, every routed call returns a "
             "truthful refusal as data (ok=False, "
             "status=\"refused\", an error naming "
             "the binding as disabled) and touches "
             "neither API — no API is built and no "
             "call is routed, proven by the probes "
             "with counting factories"},
    {"id": "called_never_reimplemented",
     "rule": "cognition routes only through "
             "TF-154's CognitionAPI and memory only "
             "through TF-155's MemoryAPI (which "
             "itself calls TF-057's toolset) — this "
             "module implements no cognition, store "
             "or enforcement logic of its own, and "
             "no Hermes provider, router or manager "
             "logic either"},
    {"id": "no_hermes_code_imported_or_copied",
     "rule": "no Hermes code is imported or "
             "copied: Hermes's shapes are filed as "
             "literal data read READ-ONLY from the "
             "Hermes hack tree (plugins/volmarr-core's "
             "plugin.yaml / cognition.py / routing.py "
             "/ context_packet.py, plugins/model-"
             "providers, agent/memory_provider.py, "
             "plugins/context_engine), and nothing "
             "under the Hermes tree is edited by "
             "this slice; no credential — above all "
             "no endpoint key — is ever filed, "
             "passed or stored by the binding "
             "(TF-160 owns credential custody)"},
    {"id": "every_field_has_a_disposition",
     "rule": "every field on either side has a "
             "FIELD_ROWS row with a disposition — "
             "none is silently dropped and none is "
             "invented; max_tokens and any "
             "credential are filed no-home in the "
             "open"},
    {"id": "responses_detached_json_safe_versioned",
     "rule": "every response is detached "
             "JSON-safe data (a JSON round-trip "
             "copy) whose contract_version is "
             "TF-034's API_VERSION, consumed from "
             "thoughtforge.cognition.api_surface — "
             "no competing version is invented "
             "here"},
)

#: TF-157-style name for the binding's rules.
HERMES_BINDING_RULES = RULE_ROWS

RULE_IDS: tuple[str, ...] = tuple(
    r["id"] for r in RULE_ROWS)


# ── The calls, as literal data ───────────────────────────────────────────────

CALL_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "provider_turn",
     "surface": "cognition",
     "hermes_side": "a local cognition turn Hermes "
                    "routes to the provider: "
                    "volmarr-core's cognition execute "
                    "(hermes volmarr cognition execute; "
                    "POST /chat/completions on the "
                    "loopback endpoint) with the "
                    "bounded request {model, messages, "
                    "max_tokens}, tool-free, "
                    "non-streaming",
     "thoughtforge_route": "TF-154 CognitionAPI.turn — "
                           "the turn-in assembled by "
                           "the filed rule in FIELD_ROWS "
                           "(the last user message as "
                           "user_text, every other "
                           "message as history); the "
                           "turn-out returns verbatim "
                           "in the response's turn and "
                           "as the filed completion shape",
     "binding_method": "provider_turn(request)",
     "routed": True,
     "summary": "one cognition turn through "
                "ThoughtForge for one Hermes-routed "
                "local turn"},
    {"id": "provider_catalog",
     "surface": "cognition",
     "hermes_side": "the model catalog probe: "
                    "volmarr-core's cognition health "
                    "validates the endpoint's GET "
                    "/models catalog (an OpenAI model "
                    "list)",
     "thoughtforge_route": "the filed FILED_CATALOG "
                           "row — an honest answer "
                           "from TF-154's CognitionAPI "
                           "contract: the one model is "
                           "the ThoughtForge cognition "
                           "API itself; no model process "
                           "is queried and none is "
                           "invented",
     "binding_method": "catalog()",
     "routed": True,
     "summary": "the provider catalog for the "
                "ThoughtForge entry"},
    {"id": "provider_health",
     "surface": "cognition",
     "hermes_side": "the health probe: hermes volmarr "
                    "cognition health [--json] "
                    "validates the profile-scoped "
                    "loopback catalog and credential "
                    "file on Hermes's side",
     "thoughtforge_route": "the filed FILED_HEALTH "
                           "row — reports the binding's "
                           "own enabled state against "
                           "the TF-154 CognitionAPI / "
                           "TF-155 MemoryAPI contract; "
                           "claims no live Hermes, no "
                           "live model and no credential "
                           "validation (the credential "
                           "file is Hermes's side, never "
                           "read here)",
     "binding_method": "health_check()",
     "routed": True,
     "summary": "the provider health answer for the "
                "ThoughtForge entry"},
    {"id": "memory_add",
     "surface": "memory",
     "hermes_side": "a Hermes-side memory write "
                    "behind the memory-provider "
                    "surface — a store write for the "
                    "active profile's entity",
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
     "summary": "a Hermes-side memory write lands as "
                "a TF-155 add"},
    {"id": "memory_inspect",
     "surface": "memory",
     "hermes_side": "a Hermes-side memory read behind "
                    "the memory-provider surface",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"inspect\", request) — "
                           "TF-057's inspect_store / "
                           "inspect_records / find_record "
                           "through TF-155, called, never "
                           "reimplemented; read-only",
     "binding_method": "memory_call(\"inspect\", request) "
                       "/ memory_inspect(...)",
     "routed": True,
     "summary": "Hermes-side memory reads land as "
                "TF-155 inspects"},
    {"id": "memory_prune",
     "surface": "memory",
     "hermes_side": "a Hermes-side memory lifecycle "
                    "pass behind the memory-provider "
                    "surface",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"prune\", request) — "
                           "TF-057's MemoryLifecycle pass "
                           "through TF-155, called, never "
                           "reimplemented",
     "binding_method": "memory_call(\"prune\", request) "
                       "/ memory_prune(...)",
     "routed": True,
     "summary": "Hermes-side memory removal routes as "
                "TF-155's lifecycle prune"},
    {"id": "memory_export",
     "surface": "memory",
     "hermes_side": "a Hermes-side store export "
                    "behind the memory-provider "
                    "surface (Hermes-portability: a "
                    "profile's memory must be "
                    "carriable)",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"export\", request) — "
                           "TF-057's export_store_bundle "
                           "through TF-155, the verified "
                           "TF-052 bundle returned as "
                           "detached data; filed so all "
                           "four TF-155 ops have a row",
     "binding_method": "memory_call(\"export\", request) "
                       "/ memory_export(...)",
     "routed": True,
     "summary": "the store export is reachable through "
                "the binding"},
    {"id": "tool_execution",
     "surface": "hermes_only",
     "hermes_side": "tool calls (pre_tool_call / "
                    "post_tool_call hooks, the shell's "
                    "tool registry) — the Hermes shell "
                    "owns tools in the ownership map",
     "thoughtforge_route": "not routed: volmarr-core's "
                           "own cognition execute is "
                           "filed tool-free, and a "
                           "cognition provider executes "
                           "no tool — the binding files "
                           "no tool call, runs no tool "
                           "and answers no tool result",
     "binding_method": None,
     "routed": False,
     "summary": "tool execution stays Hermes's own "
                "act; nothing routes to ThoughtForge"},
    {"id": "gateway_messaging",
     "surface": "hermes_only",
     "hermes_side": "gateways and platform messaging "
                    "— the Hermes shell owns gateways "
                    "in the ownership map",
     "thoughtforge_route": "not routed: the binding "
                           "sends no message on any "
                           "gateway or platform and "
                           "holds no channel — it "
                           "answers cognition calls with "
                           "data, and delivery stays the "
                           "caller's own act",
     "binding_method": None,
     "routed": False,
     "summary": "gateway / messaging stays Hermes's "
                "own act; nothing routes to "
                "ThoughtForge"},
    {"id": "session_state_custody",
     "surface": "hermes_only",
     "hermes_side": "sessions and state custody: "
                    "Hermes's SessionDB is the sole "
                    "canonical transcript owner, and "
                    "volmarr-core's present-state / "
                    "entity stores are its own domains "
                    "under the ownership map",
     "thoughtforge_route": "not routed: the binding "
                           "holds no session, writes no "
                           "entity or present-state "
                           "file, and keeps no transcript "
                           "— history reaches a turn only "
                           "as the caller supplies it in "
                           "the request's messages",
     "binding_method": None,
     "routed": False,
     "summary": "session / state custody stays "
                "Hermes's own act; nothing routes to "
                "ThoughtForge"},
    {"id": "credential_custody",
     "surface": "hermes_only",
     "hermes_side": "credential custody: the profile "
                    "credential file (cognition_api_key_"
                    "file, default secrets/aesir.key) "
                    "and the kista-secret-source "
                    "plugin's kista:// bindings",
     "thoughtforge_route": "not routed: credentials "
                           "live in Hermes/Kista "
                           "custody and TF-160 owns "
                           "custody on the ThoughtForge "
                           "side — the binding reads no "
                           "key file, accepts no key in "
                           "a request (the request "
                           "validator refuses one) and "
                           "stores none",
     "binding_method": None,
     "routed": False,
     "summary": "credential custody stays Hermes / "
                "Kista's own act (TF-160's on this "
                "side); nothing routes to ThoughtForge"},
    {"id": "provider_resolution",
     "surface": "hermes_only",
     "hermes_side": "provider resolution and routing "
                    "decisions: the Hermes shell owns "
                    "provider resolution, and "
                    "volmarr-core's router "
                    "(runeforge.cognition.route) "
                    "chooses a capability tier without "
                    "choosing or calling a cloud "
                    "provider",
     "thoughtforge_route": "not routed: the binding "
                           "never decides a route, "
                           "never resolves or selects a "
                           "provider (cloud or local) and "
                           "never escalates — it answers "
                           "only the turns Hermes routes "
                           "to the local provider seat it "
                           "files",
     "binding_method": None,
     "routed": False,
     "summary": "resolution and routing decisions "
                "stay Hermes's own acts; nothing "
                "routes to ThoughtForge"},
    {"id": "memory_lifecycle",
     "surface": "hermes_only",
     "hermes_side": "the MemoryProvider lifecycle "
                    "driven by Hermes's MemoryManager: "
                    "initialize / system_prompt_block / "
                    "prefetch / queue_prefetch / "
                    "sync_turn / shutdown, and the "
                    "context packet's packet_sources "
                    "assembly on pre_llm_call",
     "thoughtforge_route": "not routed: the lifecycle "
                           "is Hermes-driven under its "
                           "multiplexing contract (one "
                           "instance per profile, bound "
                           "at initialize) — the binding "
                           "implements no lifecycle hook "
                           "and assembles no packet; only "
                           "the store operations behind "
                           "the surface route (the memory "
                           "rows above)",
     "binding_method": None,
     "routed": False,
     "summary": "the memory-provider lifecycle stays "
                "Hermes's own act; only the store "
                "operations route"},
)

CALL_IDS: tuple[str, ...] = tuple(r["id"] for r in CALL_ROWS)

ROUTED_CALL_IDS: tuple[str, ...] = tuple(
    r["id"] for r in CALL_ROWS if r["routed"])

NOT_ROUTED_CALL_IDS: tuple[str, ...] = tuple(
    r["id"] for r in CALL_ROWS if not r["routed"])


# ── The field mapping, as literal data ───────────────────────────────────────
#
# Every field on either side has a row with a
# disposition: mapped / mapped_as_history /
# identity_only_not_passed / no_home_not_passed /
# no_source_not_set / no_source_default_applies /
# binding_filed / partial. None is silently
# dropped; none is invented.

FIELD_ROWS: tuple[dict[str, Any], ...] = (
    # ── In: Hermes request -> TF-154 turn-in ──
    {"direction": "in",
     "id": "request.model",
     "source": "the request's model (cognition.py's "
               "AesirCompletionRequest model: a string "
               "of 1..256 UTF-8 bytes naming the "
               "endpoint's model)",
     "target": "the binding envelope's model and "
               "the completion's model; no turn-in "
               "field",
     "disposition": "identity_only_not_passed",
     "summary": "the model names which provider "
                "model Hermes routed to (the filed "
                "catalog's one id is \"thoughtforge\"); "
                "TF-154's turn-in files no model field, "
                "so the name is never passed as turn "
                "data — it rides the envelope and the "
                "completion so a consumer can see "
                "which seat answered"},
    {"direction": "in",
     "id": "messages.last_user.content",
     "source": "the content of the last message with "
               "role \"user\" (cognition.py requires "
               "the request to end with a user "
               "message)",
     "target": "turn-in user_text",
     "disposition": "mapped",
     "summary": "the filed rule's first half: the "
                "last user message is the turn's "
                "question — TF-154 requires user_text "
                "as a non-empty string, so an empty "
                "last user content (or no user "
                "message) is an invalid request, "
                "named by provider_request_problems()"},
    {"direction": "in",
     "id": "messages.others",
     "source": "every message that is not the last "
               "user message, in original order",
     "target": "turn-in history",
     "disposition": "mapped_as_history",
     "summary": "the filed rule's second half: "
                "system, earlier user and assistant "
                "messages all ride as history entries "
                "— each a plain {\"role\", \"content\"} "
                "dict, verbatim (Hermes completion "
                "messages carry only those two "
                "fields); an empty history is not "
                "passed at all (TF-154 passes only "
                "set kwargs)"},
    {"direction": "in",
     "id": "message.role",
     "source": "a message's role (system / user / "
               "assistant — cognition.py's admitted "
               "set)",
     "target": "the history entry's role; for the "
               "user_text source message, the role "
               "selects it and is not itself passed",
     "disposition": "mapped",
     "summary": "roles are preserved verbatim in "
                "history — the binding rewrites no "
                "role and invents none; a role "
                "outside Hermes's admitted set is an "
                "invalid request, named by the "
                "validator"},
    {"direction": "in",
     "id": "request.max_tokens",
     "source": "the request's max_tokens "
               "(cognition.py bounds it by the "
               "profile's cognition_local_max_tokens, "
               "default 256)",
     "target": "no target — TURN_IN_FIELDS files no "
               "token-cap field",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the call shape (an "
                "integer >= 1 when present, judged by "
                "the validator), never passed to "
                "TF-154 and never claimed honored — "
                "Hermes's own bound stays Hermes's "
                "side of the contract"},
    {"direction": "in",
     "id": "request.api_key",
     "source": "nothing in a Hermes cognition "
               "request carries a credential: "
               "cognition.py's request fields are "
               "exactly model / messages / max_tokens "
               "and the endpoint key lives in the "
               "profile's credential file / Kista",
     "target": "no target — never filed, never "
               "passed, never stored",
     "disposition": "no_home_not_passed",
     "summary": "the binding holds no credential "
                "and authenticates as no one: a "
                "request carrying an api_key (or any "
                "unfiled field) is refused by the "
                "validator, named — credential "
                "custody is TF-160's slice, Hermes / "
                "Kista's act on that side"},
    {"direction": "in",
     "id": "turn_in.retrieval_path",
     "source": "nothing on the Hermes side files a "
               "retrieval path",
     "target": "turn-in retrieval_path",
     "disposition": "no_source_not_set",
     "summary": "the binding never sets it; the "
                "core's own default retrieval applies "
                "— no Hermes value is invented for it"},
    {"direction": "in",
     "id": "turn_in.num_drafts",
     "source": "nothing on the Hermes side files a "
               "draft count",
     "target": "turn-in num_drafts",
     "disposition": "no_source_not_set",
     "summary": "the binding never sets it; the "
                "core's own default draft count "
                "applies"},
    {"direction": "in",
     "id": "turn_in.want_trace",
     "source": "nothing on the Hermes side files a "
               "trace request",
     "target": "turn-in want_trace",
     "disposition": "no_source_default_applies",
     "summary": "the binding never sets it; TF-154's "
                "filed default (True) applies, so a "
                "routed turn carries its trace status "
                "honestly in the returned turn"},
    # ── Out: TF-154 turn-out -> the Hermes-shaped completion ──
    {"direction": "out",
     "id": "turn.text",
     "source": "turn-out text",
     "target": "the response's turn (verbatim) and "
               "the completion's choices[0].message."
               "content — the content Hermes's "
               "strict reply parser reads",
     "disposition": "mapped",
     "summary": "the turn's words are the "
                "completion's words, verbatim: ONE "
                "choice carrying the whole text — "
                "the binding is not a streamer and "
                "Hermes's local execute is filed "
                "non-streaming, so no stream is "
                "pretended"},
    {"direction": "out",
     "id": "turn.response_id",
     "source": "turn-out response_id",
     "target": "the response's turn (verbatim); no "
               "completion field home",
     "disposition": "mapped",
     "summary": "Hermes's completion shape files no "
                "response id; the id rides the "
                "binding response's turn, never "
                "dropped"},
    {"direction": "out",
     "id": "turn.turn_id",
     "source": "turn-out turn_id",
     "target": "the response's turn (verbatim); no "
               "completion field home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim, beside "
                "its trace"},
    {"direction": "out",
     "id": "turn.mode",
     "source": "turn-out mode",
     "target": "the response's turn (verbatim); no "
               "completion field home",
     "disposition": "mapped",
     "summary": "the turn's mode (knowledge_only "
                "when no model served) rides the "
                "turn — a Hermes consumer can see "
                "honestly how its answer was "
                "produced"},
    {"direction": "out",
     "id": "turn.enforcement_passed",
     "source": "turn-out enforcement_passed",
     "target": "the response's turn (verbatim); no "
               "completion field home",
     "disposition": "mapped",
     "summary": "the enforcement verdict rides the "
                "turn verbatim"},
    {"direction": "out",
     "id": "turn.citations",
     "source": "turn-out citations",
     "target": "the response's turn (verbatim); no "
               "completion field home",
     "disposition": "mapped",
     "summary": "the turn's citations ride the "
                "turn verbatim"},
    {"direction": "out",
     "id": "turn.token_count",
     "source": "turn-out token_count",
     "target": "the response's turn (verbatim) and "
               "the completion's usage.total_tokens",
     "disposition": "partial",
     "summary": "TF-154 files ONE count, not a "
                "prompt / completion split: the "
                "completion's usage files "
                "prompt_tokens and completion_tokens "
                "as None rather than inventing a "
                "split — filed in the open because "
                "cognition.py's strict reply parser "
                "requires integer prompt / completion "
                "tokens, so the binding does NOT "
                "claim that parser accepts the filed "
                "usage unchanged; reconciling the "
                "split is the live adapter's "
                "business at TF-170's gate, never "
                "pretended here"},
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
               "completion field home",
     "disposition": "mapped",
     "summary": "TF-084's joined trace rides the "
                "turn when TF-154 includes it; the "
                "completion carries no trace"},
    {"direction": "out",
     "id": "turn.trace_format",
     "source": "turn-out trace_format",
     "target": "the response's turn (verbatim); no "
               "completion field home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim beside "
                "the trace"},
    {"direction": "out",
     "id": "turn.trace_status",
     "source": "turn-out trace_status",
     "target": "the response's turn (verbatim); no "
               "completion field home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim — included "
                "/ not_requested / unavailable with "
                "its reason named, exactly as "
                "TF-154 filed it"},
    {"direction": "out",
     "id": "completion.finish_reason",
     "source": "this binding — the filed constant "
               "FINISH_REASON",
     "target": "the completion's choices[0]."
               "finish_reason",
     "disposition": "binding_filed",
     "summary": "Hermes's strict reply parser "
                "requires a finish_reason string and "
                "TF-154 files no finish reason: the "
                "binding files its own constant "
                "\"complete\" rather than inventing "
                "a provider reason"},
    {"direction": "out",
     "id": "completion.model",
     "source": "the request's model when Hermes "
               "named one, else the filed PROVIDER_KEY",
     "target": "the completion's model — the model "
               "Hermes's reply parser reads back",
     "disposition": "binding_filed",
     "summary": "a completion answers in the name "
                "it was called under: the request's "
                "model rides back, and an unnamed "
                "call answers as the filed catalog "
                "model \"thoughtforge\""},
)

FIELD_IDS: tuple[str, ...] = tuple(
    f["id"] for f in FIELD_ROWS)


def field_rows_for(direction: str) -> tuple[dict[str, Any], ...]:
    """One direction's filed field rows, in filing
    order."""
    return tuple(f for f in FIELD_ROWS
                 if f["direction"] == direction)


# ── The contract as plain data ───────────────────────────────────────────────

def hermes_binding_contract() -> dict[str, Any]:
    """The Hermes binding contract as plain
    JSON-safe data, built from the filed literals
    above and TF-034's version. A fresh copy each
    call."""
    return {
        "format": HERMES_BINDING_FORMAT,
        "contract_version": API_VERSION,
        "provider_key": PROVIDER_KEY,
        "plugin_name": PLUGIN_NAME,
        "default_enabled": DEFAULT_ENABLED,
        "disabled_error": DISABLED_ERROR,
        "finish_reason": FINISH_REASON,
        "providers": [dict(r) for r in PROVIDER_ROWS],
        "model": dict(FILED_MODEL),
        "catalog": json.loads(json.dumps(FILED_CATALOG)),
        "health": dict(FILED_HEALTH),
        "calls": [dict(r) for r in CALL_ROWS],
        "fields": [dict(f) for f in FIELD_ROWS],
        "rules": [dict(r) for r in RULE_ROWS],
    }


# ── Pure validators ──────────────────────────────────────────────────────────

def provider_row_problems(row: Any) -> list[str]:
    """Judge one provider-surface filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["provider row: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("provider row: the row files no id")
    if not row.get("surface"):
        problems.append(
            f"provider row {row.get('id')}: no surface filed")
    for key in ("hermes_side", "registration", "summary"):
        if not row.get(key):
            problems.append(
                f"provider row {row.get('id')}: no {key} filed")
    if not isinstance(row.get("plugin_fields"), list) \
            or not row.get("plugin_fields"):
        problems.append(
            f"provider row {row.get('id')}: plugin_fields "
            "is not a non-empty list")
    if not isinstance(row.get("answers_binding"), bool):
        problems.append(
            f"provider row {row.get('id')}: answers_binding "
            "is not a boolean")
    elif row["answers_binding"] \
            and not row.get("binding_method"):
        problems.append(
            f"provider row {row.get('id')}: answers the "
            "binding but no binding_method is filed")
    elif not row["answers_binding"] \
            and row.get("binding_method"):
        problems.append(
            f"provider row {row.get('id')}: does not "
            "answer the binding but a binding_method "
            "is filed")
    if row.get("enabled_by_default") is not False:
        problems.append(
            f"provider row {row.get('id')}: "
            "enabled_by_default is not False")
    return problems


def call_row_problems(row: Any) -> list[str]:
    """Judge one call filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["call: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("call: the row files no id")
    if row.get("surface") not in (
            "cognition", "memory", "hermes_only"):
        problems.append(
            f"call {row.get('id')}: surface is "
            f"{row.get('surface')!r}, not cognition / "
            "memory / hermes_only")
    for key in ("hermes_side", "thoughtforge_route", "summary"):
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
    if row.get("surface") == "hermes_only" \
            and row.get("routed"):
        problems.append(
            f"call {row.get('id')}: a hermes_only call "
            "is routed — Hermes's own acts never route")
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


def rule_row_problems(row: Any) -> list[str]:
    """Judge one rule filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["rule: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("rule: the row files no id")
    if not row.get("rule"):
        problems.append(f"rule {row.get('id')}: no rule filed")
    return problems


#: TF-156-style alias.
binding_rule_problems = rule_row_problems


def provider_request_problems(request: Any) -> list[str]:
    """Judge one provider-turn request, purely.
    Empty-safe.

    Shape only: the Hermes completion request
    (exactly model / messages / max_tokens; a
    message is exactly role / content with
    Hermes's admitted roles; the last message
    a user message with non-empty content).
    What the core does with the turn is
    TF-154's business."""
    problems: list[str] = []
    if not isinstance(request, dict) or not request:
        return ["provider request: request must be a "
                "mapping shaped like {model, messages, "
                "max_tokens} — the Hermes completion "
                "request cognition.py files"]
    for key in request:
        if key not in REQUEST_FIELDS:
            problems.append(
                f"provider request: request files "
                f"{key!r}, not a filed Hermes request "
                f"field {list(REQUEST_FIELDS)} — above "
                "all, no credential is ever a request "
                "field here")
    if "model" in request:
        if not isinstance(request["model"], str) \
                or not request["model"]:
            problems.append(
                "provider request: model must be a "
                "non-empty string when present")
    if "max_tokens" in request:
        value = request["max_tokens"]
        if isinstance(value, bool) \
                or not isinstance(value, int) \
                or value < 1:
            problems.append(
                "provider request: max_tokens must be "
                "an integer >= 1 when present")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        problems.append(
            "provider request: messages must be a "
            "non-empty list of message mappings")
        return problems
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(
                f"provider request: message {index} is "
                "not a mapping shaped like {role, "
                "content}")
            continue
        for key in message:
            if key not in MESSAGE_FIELDS:
                problems.append(
                    f"provider request: message {index} "
                    f"files {key!r}, not a Hermes "
                    f"message field {list(MESSAGE_FIELDS)} "
                    "— cognition.py's messages carry "
                    "only role and content")
        if message.get("role") not in MESSAGE_ROLES:
            problems.append(
                f"provider request: message {index} "
                f"role is {message.get('role')!r}, not "
                f"one of Hermes's admitted roles "
                f"{list(MESSAGE_ROLES)}")
        if not isinstance(message.get("content"), str):
            problems.append(
                f"provider request: message {index} "
                "content must be a string")
    if not problems:
        users = [m for m in messages
                 if m["role"] == "user"]
        if not users:
            problems.append(
                "provider request: no message with "
                "role \"user\" — the filed assembly "
                "rule takes user_text from the last "
                "user message")
        elif messages[-1]["role"] != "user":
            problems.append(
                "provider request: the last message "
                "is not a user message — cognition.py "
                "requires the request to end with one")
        elif not messages[-1]["content"]:
            problems.append(
                "provider request: the last user "
                "message's content is empty — TF-154's "
                "user_text must be a non-empty string")
    return problems


def binding_response_problems(row: Any) -> list[str]:
    """Judge one binding response dict against the
    envelope, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["response: the row is empty"]
    problems: list[str] = []
    for field in ("call", "ok", "status", "provider_key",
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
    if row.get("provider_key") != PROVIDER_KEY:
        problems.append(
            f"response: provider_key is "
            f"{row.get('provider_key')!r}, not {PROVIDER_KEY!r}")
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


# ── The filed turn-in assembly + the completion shape ────────────────────────

def assemble_turn_in(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """The filed assembly rule, applied purely:
    the last user message's content is user_text;
    every other message, in original order, is a
    history entry (role / content verbatim).
    Assumes provider_request_problems() judged
    the request clean."""
    last_user = max(
        i for i, m in enumerate(messages)
        if m["role"] == "user")
    history: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index == last_user:
            continue
        history.append({
            "role": message["role"],
            "content": message["content"],
        })
    return {"user_text": messages[last_user]["content"],
            "history": history}


def completion_for(
    turn: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    """The filed completion shape for one
    turn-out, in the OpenAI-compatible form
    Hermes's reply parser reads: one choice
    whose message content is the turn's text,
    the binding's filed finish reason, and a
    usage whose single filed count rides as
    total_tokens (the split filed None — see
    FIELD_ROWS' turn.token_count row)."""
    return {
        "model": model or PROVIDER_KEY,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant",
                        "content": turn.get("text", "")},
            "finish_reason": FINISH_REASON,
        }],
        "usage": {"prompt_tokens": None,
                  "completion_tokens": None,
                  "total_tokens": turn.get("token_count")},
    }


# ── The binding ──────────────────────────────────────────────────────────────

def _detached(data: Any) -> Any:
    """A detached JSON-safe copy (a JSON round-trip)."""
    return json.loads(json.dumps(data))


class HermesBinding:
    """Hermes's routed local cognition through
    ThoughtForge, shaped on the volmarr-core
    contract, off by default.

    Wraps a TF-154 ``CognitionAPI`` and a TF-155
    ``MemoryAPI``: pass ``cognition_api`` /
    ``memory_api`` to inject them,
    ``cognition_factory`` / ``memory_factory``
    to build them lazily, or ``turn_callable``
    / ``memory_store`` as passthroughs to the
    lazily built default APIs (TF-154's /
    TF-155's own injection shapes). Both APIs
    are built lazily: constructing the binding
    builds neither.

    The binding is DISABLED unless ``enabled``
    is True at construction or ``enable()`` is
    called: while disabled every routed call
    returns the filed refusal as data and
    touches neither API. The binding performs
    no Hermes act: it executes no tool, sends
    no gateway message, holds no session or
    credential and decides no route — a
    cognition provider only, behind Hermes's
    own surfaces.
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
            "provider_key": PROVIDER_KEY,
            "contract_version": API_VERSION,
            "error": None if ok else fields.pop(
                "error", ""),
            **fields,
        })

    def _refused(self, call: str) -> dict[str, Any]:
        return self._envelope(
            call, False, "refused", error=DISABLED_ERROR)

    # ── The provider's cognition calls ──

    def provider_turn(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """The routed local turn: a Hermes
        completion-shaped request dict
        (``{model, messages, max_tokens}``) in,
        one binding response out — the TF-154
        turn verbatim in ``turn`` and the filed
        completion shape in ``completion``.

        While disabled, the filed refusal returns
        as data and nothing is judged, built or
        routed. When enabled, an invalid request
        raises ValueError naming the problem(s),
        judged by ``provider_request_problems()``
        before any API is built or called."""
        if not self._enabled:
            return self._refused("provider_turn")
        problems = provider_request_problems(request)
        if problems:
            raise ValueError("; ".join(problems))
        turn_in = assemble_turn_in(request["messages"])
        kwargs: dict[str, Any] = {}
        if turn_in["history"]:
            kwargs["history"] = turn_in["history"]
        turn = self._get_cognition().turn(
            turn_in["user_text"], **kwargs)
        model = request.get("model") or PROVIDER_KEY
        return self._envelope(
            "provider_turn", True, "replied",
            model=model, turn=turn,
            completion=completion_for(turn, model))

    def catalog(self) -> dict[str, Any]:
        """The catalog call: the filed
        FILED_CATALOG row as the response's
        ``catalog`` — an honest answer from the
        cognition contract; no model process is
        queried. While disabled, the filed
        refusal returns as data."""
        if not self._enabled:
            return self._refused("provider_catalog")
        return self._envelope(
            "provider_catalog", True, "listed",
            catalog=_detached(FILED_CATALOG))

    def health_check(self) -> dict[str, Any]:
        """The health call: the filed FILED_HEALTH
        row as the response's ``health`` — it
        reports the binding, never a live Hermes
        process or live model, and reads no
        credential. While disabled, the filed
        refusal returns as data."""
        if not self._enabled:
            return self._refused("provider_health")
        return self._envelope(
            "provider_health", True, "reported",
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

class _FakeHermesConsumer:
    """A dict-shaped stand-in for a Hermes-side
    cognition consumer, built from the filed
    plugin-surface data only: it files a
    routing decision in routing.py's shape
    (metadata-only — a route and a reason, no
    content), builds completion-shaped requests
    in cognition.py's shape, and reads the
    filed catalog. Fixture only — no Hermes
    code, no live process."""

    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []

    def route_decision(
        self, operation: str = "cognition.turn",
    ) -> dict[str, Any]:
        decision = {
            "schema": "runeforge.cognition.route",
            "schema_version": 2,
            "route": "local",
            "reason": "local.within_bounds",
            "operation": operation,
            "mode": "auto",
        }
        self.decisions.append(decision)
        return decision

    def request(
        self, user_text: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        messages = [dict(m) for m in history or []]
        messages.append({"role": "user",
                         "content": user_text})
        return {"model": PROVIDER_KEY,
                "messages": messages,
                "max_tokens": 256}

    def catalog_models(
        self, catalog: dict[str, Any],
    ) -> list[str]:
        return [row["id"] for row in catalog["data"]]


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


def _knowledge_binding() -> tuple[HermesBinding, Any]:
    """An enabled binding over a knowledge-only
    core (TF-154's TF-084 fixture shape) and a
    temp store. Returns the binding and the
    temp home."""
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.knowledge.store import MemoryStore

    tmp = tempfile.mkdtemp(prefix="tf158-")
    core = ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    store = MemoryStore(memory_dir=Path(tmp) / "store")
    binding = HermesBinding(
        cognition_api=CognitionAPI(core=core),
        memory_api=MemoryAPI(store=store),
        enabled=True,
    )
    return binding, tmp


def _temp_memory_binding() -> tuple[HermesBinding, Any, str]:
    """An enabled binding whose memory side is a
    TF-155 MemoryAPI over a fresh temp store
    (the cognition side stays unbuilt unless a
    cognition call lands)."""
    from thoughtforge.knowledge.store import MemoryStore

    tmp = tempfile.mkdtemp(prefix="tf158-")
    store = MemoryStore(memory_dir=Path(tmp) / "store")
    binding = HermesBinding(
        memory_api=MemoryAPI(store=store), enabled=True)
    return binding, store, tmp


_REQUEST = {"model": "thoughtforge",
            "messages": [
                {"role": "user",
                 "content": "What time does the ferry leave?"}],
            "max_tokens": 256}


# ── The filed probes ─────────────────────────────────────────────────────────

HERMES_BINDING_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "disabled_refusal_touches_nothing",
     "kind": "disabled_refusal",
     "detail": "a default binding refuses every "
               "routed call as data (ok=False, the "
               "error naming the binding disabled), "
               "and counting factories prove neither "
               "API is ever built"},
    {"id": "enabled_cognition_round_trip_through_binding",
     "kind": "cognition_round_trip",
     "detail": "an enabled binding routes a Hermes "
               "completion-shaped request through the "
               "injected CognitionAPI by the filed "
               "rule (the last user message as "
               "user_text, every other message as "
               "history) and returns the turn with "
               "the filed completion shape (one "
               "choice, the turn's text as its "
               "content, the filed finish reason, "
               "the single count as total_tokens)"},
    {"id": "enabled_memory_round_trip_temp_store",
     "kind": "memory_round_trip",
     "detail": "an enabled binding adds through "
               "TF-155 (stamped manual_entry), finds "
               "the record through TF-155's inspect, "
               "and exports the verified bundle — all "
               "through the binding, on a temp store"},
    {"id": "catalog_and_health_are_honest",
     "kind": "honest_answers",
     "detail": "catalog returns exactly the filed "
               "catalog (an OpenAI model list whose "
               "one model is the ThoughtForge "
               "cognition API itself) and "
               "health_check returns the filed health "
               "— reporting the binding, claiming no "
               "live Hermes, no live model and no "
               "credential validation"},
    {"id": "hermes_only_calls_stay_hermes",
     "kind": "hermes_only",
     "detail": "no tool-execution, gateway, "
               "session-custody, credential-custody, "
               "provider-resolution or "
               "memory-lifecycle call exists in "
               "CALL_ROWS as routed — all six are "
               "filed NOT routed, and the binding "
               "class files no such method"},
    {"id": "fake_consumer_consistency",
     "kind": "fake_consumer",
     "detail": "a fake Hermes consumer shaped on "
               "the filed surface data files a local "
               "routing decision in routing.py's "
               "shape, builds a completion-shaped "
               "request, and routes its turn through "
               "the binding only: the completion's "
               "content is the turn's text, the "
               "completion answers under the "
               "request's model, the consumer's "
               "catalog read agrees with the "
               "binding's catalog, and the filed "
               "provider rows are consistent "
               "(volmarr-core under its literal "
               "plugin name, the context-engine row "
               "answering nothing, the catalog / "
               "health rows agreeing with "
               "PROVIDER_KEY)"},
    {"id": "doc_matches_render_and_version_is_tf034",
     "kind": "doc_and_version",
     "detail": "the filed doc is the render "
               "verbatim, the contract version is "
               "TF-034's API_VERSION with its surface "
               "clean, and DEFAULT_ENABLED is False"},
)


def run_hermes_binding_probes() -> list[dict[str, Any]]:
    """Run the filed probes; each returns its
    verdict."""
    results: list[dict[str, Any]] = []
    for probe in HERMES_BINDING_PROBES:
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

                binding = HermesBinding(
                    cognition_factory=cog_factory,
                    memory_factory=mem_factory)
                outs = [
                    binding.provider_turn(_REQUEST),
                    binding.catalog(),
                    binding.health_check(),
                    binding.memory_call(
                        "add", {"kind": "user_fact",
                                "summary": "x"}),
                    binding.memory_export(),
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
                binding = HermesBinding(
                    cognition_api=fake, enabled=True)
                request = {
                    "model": "thoughtforge",
                    "messages": [
                        {"role": "system",
                         "content": "be terse"},
                        {"role": "user", "content": "first"},
                        {"role": "assistant",
                         "content": "reply"},
                        {"role": "user",
                         "content": "What time does the "
                                    "ferry leave?"}],
                    "max_tokens": 256,
                }
                out = binding.provider_turn(request)
                completion = out.get("completion") or {}
                choices = completion.get("choices") or [{}]
                passed = out.get("ok") is True \
                    and fake.turns == [{
                        "user_text":
                            "What time does the ferry leave?",
                        "history": [
                            {"role": "system",
                             "content": "be terse"},
                            {"role": "user",
                             "content": "first"},
                            {"role": "assistant",
                             "content": "reply"},
                        ]}] \
                    and out["turn"]["text"] \
                    == "stub answer to: What time does " \
                       "the ferry leave?" \
                    and choices[0].get("message", {}).get(
                        "content") == out["turn"]["text"] \
                    and choices[0].get("finish_reason") \
                    == FINISH_REASON \
                    and completion.get("usage", {}).get(
                        "total_tokens") == 3 \
                    and completion.get("model") \
                    == "thoughtforge" \
                    and binding_response_problems(out) == []
            elif kind == "memory_round_trip":
                binding, _store, tmp = _temp_memory_binding()
                added = binding.memory_add(
                    "user_fact", summary="ferry at 07:30",
                    record_id="fct_tf158")
                found = binding.memory_inspect(
                    record_id="fct_tf158")
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
            elif kind == "honest_answers":
                binding = HermesBinding(
                    cognition_api=_RecordingCognitionAPI(),
                    enabled=True)
                catalog = binding.catalog()
                health = binding.health_check()
                passed = catalog.get("catalog") \
                    == FILED_CATALOG \
                    and catalog["catalog"]["object"] \
                    == "list" \
                    and health.get("health") == FILED_HEALTH \
                    and health["health"]["healthy"] is True \
                    and health["health"]["latency_ms"] is None \
                    and "no credential was read or " \
                        "validated" in \
                        health["health"]["detail"] \
                    and binding_response_problems(catalog) == [] \
                    and binding_response_problems(health) == []
            elif kind == "hermes_only":
                by_id = {r["id"]: r for r in CALL_ROWS}
                passed = all(
                    by_id[cid]["routed"] is False
                    for cid in ("tool_execution",
                                "gateway_messaging",
                                "session_state_custody",
                                "credential_custody",
                                "provider_resolution",
                                "memory_lifecycle")) \
                    and not any(
                        hasattr(HermesBinding, name)
                        for name in ("execute_tool", "send",
                                     "send_message",
                                     "open_session",
                                     "get_credential",
                                     "decide_route",
                                     "resolve_provider",
                                     "prefetch", "sync_turn"))
            elif kind == "fake_consumer":
                consumer = _FakeHermesConsumer()
                decision = consumer.route_decision()
                request = consumer.request(
                    "What time does the ferry leave?")
                fake = _RecordingCognitionAPI()
                binding = HermesBinding(
                    cognition_api=fake, enabled=True)
                out = binding.provider_turn(request)
                catalog = binding.catalog()
                passed = decision["route"] == "local" \
                    and decision["schema"] \
                    == "runeforge.cognition.route" \
                    and PLUGIN_NAME == "volmarr-core" \
                    and all(
                        provider_row_problems(r) == []
                        for r in PROVIDER_ROWS) \
                    and {r["id"]: r for r in PROVIDER_ROWS}[
                        "provider_context_engine"][
                            "answers_binding"] is False \
                    and FILED_CATALOG["data"][0]["id"] \
                    == PROVIDER_KEY \
                    and FILED_HEALTH["models"] \
                    == [PROVIDER_KEY] \
                    and provider_request_problems(
                        request) == [] \
                    and out.get("ok") is True \
                    and out["completion"]["choices"][0][
                        "message"]["content"] \
                    == out["turn"]["text"] \
                    and out["completion"]["model"] \
                    == PROVIDER_KEY \
                    and consumer.catalog_models(
                        catalog["catalog"]) == [PROVIDER_KEY] \
                    and binding_response_problems(out) == []
            elif kind == "doc_and_version":
                doc = _REPO_ROOT / HERMES_BINDING_DOC
                passed = doc.exists() \
                    and doc.read_text(encoding="utf-8") \
                    == render_hermes_binding_markdown() \
                    and hermes_binding_contract()[
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


# ── The readable render (the hermes-binding doc, verbatim) ───────────────────

def render_hermes_binding_markdown() -> str:
    """Render the hermes-binding doc
    deterministically — the exact text filed as
    docs/architecture/HERMES_BINDING.md."""
    lines: list[str] = []
    lines.append("# Hermes Binding — ThoughtForge as a "
                 "Cognition Provider for the Hermes "
                 "Hack (TF-158)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/hermes_binding.py` — the "
        "filed doc IS the render: regenerate with "
        "`python3 -m thoughtforge.hermes_binding`. "
        "The checker (`hermes_binding_problems()`) "
        "re-derives every claim live from the tree "
        "and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{HERMES_BINDING_FORMAT}`")
    lines.append(
        f"Contract version: `{API_VERSION}` (TF-034's "
        "`API_VERSION` — consumed, not reinvented)")
    lines.append(f"Provider key: `{PROVIDER_KEY}`")
    lines.append(
        f"Plugin shaped on: `{PLUGIN_NAME}` "
        "(`plugins/volmarr-core` in the Hermes hack "
        "tree — a literal name there, filed with its "
        "evidence in section 2)")
    lines.append(
        f"Default enabled: {DEFAULT_ENABLED} — the "
        "binding is OFF by default; enabling is an "
        "explicit act (`HermesBinding(enabled=True)` "
        "or `binding.enable()`)")
    lines.append(f"Provider surfaces: "
                 f"{len(PROVIDER_ROWS)}; calls: "
                 f"{len(CALL_ROWS)} "
                 f"({len(ROUTED_CALL_IDS)} routed, "
                 f"{len(NOT_ROUTED_CALL_IDS)} filed NOT "
                 f"routed); field rows: {len(FIELD_ROWS)}; "
                 f"rules: {len(RULE_ROWS)}")
    lines.append(f"Probes: {len(HERMES_BINDING_PROBES)}/"
                 f"{len(HERMES_BINDING_PROBES)}")
    lines.append("")
    lines.append("## 1. One binding")
    lines.append("")
    lines.append(
        "`HermesBinding` wraps TF-154's `CognitionAPI` "
        "and TF-155's `MemoryAPI` — each injectable "
        "directly, by factory, or via a `turn_callable` "
        "/ `memory_store` passthrough, and both built "
        "lazily: constructing the binding builds "
        "neither API. While disabled, every routed "
        "call returns a truthful refusal as data "
        "(`ok=False`, `status=\"refused\"`, the "
        "`error` naming the binding as disabled) and "
        "touches neither API. When enabled, a routed "
        "local turn routes through the CognitionAPI "
        "and memory calls through the MemoryAPI — "
        "called, never reimplemented. An invalid "
        "request raises `ValueError` naming the "
        "problem(s), judged by the pure validators "
        "before any API is built or called. Every "
        "response is a plain JSON-safe detached dict: "
        "`call`, `ok`, `status`, `provider_key`, "
        "`contract_version` (TF-034's `API_VERSION`), "
        "`error`, plus the call's payload (`turn` + "
        "`completion` for a provider turn, `catalog` "
        "for the catalog, `health` for health, the "
        "TF-155 response verbatim in `memory` for "
        "memory calls). No Hermes code is imported or "
        "copied anywhere: Hermes's shapes are filed "
        "below as literal data read READ-ONLY from "
        "the Hermes hack tree, and the binding "
        "performs no Hermes act — it executes no "
        "tool, sends no gateway message, holds no "
        "session or credential and decides no route. "
        "A cognition provider only, behind Hermes's "
        "own surfaces.")
    lines.append("")
    lines.append("## 2. The provider surfaces — what Hermes actually offers")
    lines.append("")
    lines.append("| Row | Surface | Hermes side | "
                 "Registration | Binding method | "
                 "Answers the binding | Enabled by "
                 "default | What it is |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in PROVIDER_ROWS:
        answers = "yes" if row["answers_binding"] else "no"
        enabled = "yes" if row["enabled_by_default"] else "no"
        method = row["binding_method"] or "—"
        lines.append(
            f"| {row['id']} | {row['surface']} | "
            f"{row['hermes_side']} | "
            f"{row['registration']} | {method} | "
            f"{answers} | {enabled} | {row['summary']} |")
    lines.append("")
    lines.append(
        "On the brief's open question — does "
        "`volmarr-core` name a specific shaped "
        "contract in the Hermes tree? It DOES exist "
        "as a literal name: `plugins/volmarr-core/` "
        "is Volmarr's opt-in composition plugin "
        "(`plugin.yaml`: manifest_version 2, name "
        "volmarr-core, version 0.26.0; hooks "
        "on_session_start / on_session_end / "
        "on_session_finalize / pre_tool_call / "
        "post_tool_call / pre_llm_call / "
        "post_llm_call). The shaped contract this "
        "binding consumes is the one its cognition "
        "surfaces file: `cognition.py`'s loopback "
        "OpenAI-compatible endpoint (default "
        "`http://127.0.0.1:18434/v1`; `GET /models` "
        "catalog; `POST /chat/completions` with "
        "`{model, messages, max_tokens}`, messages "
        "carrying only role / content, the last a "
        "user message) and `routing.py`'s "
        "`runeforge.cognition.route` version 2 "
        "decisions — alongside the refresh's "
        "generic model-provider (`ProviderProfile` "
        "+ `register_provider`) and memory-provider "
        "(`MemoryProvider` ABC) plugin surfaces. No "
        "ThoughtForge provider exists inside Hermes "
        "today: this slice files the contract such "
        "a provider answers, at fixture grade. The "
        "architecture law over all of it "
        "(`ARCHITECTURE_VOLMARR.md`): Hermes core is "
        "a narrow waist; the shell owns turns, "
        "sessions, provider resolution, tools and "
        "gateways; a composition plugin owns "
        "registration and adapter construction, "
        "never domain logic or durable state; LLM "
        "providers are replaceable cognitive "
        "organs.")
    lines.append("")
    lines.append("## 3. The calls")
    lines.append("")
    lines.append("| Call | Surface | Hermes side | "
                 "ThoughtForge route | Binding method | "
                 "Routed |")
    lines.append("|---|---|---|---|---|---|")
    for row in CALL_ROWS:
        routed = "yes" if row["routed"] else "no"
        method = row["binding_method"] or "—"
        lines.append(
            f"| {row['id']} | {row['surface']} | "
            f"{row['hermes_side']} | "
            f"{row['thoughtforge_route']} | {method} | "
            f"{routed} |")
    lines.append("")
    lines.append("## 4. Field mapping — in (Hermes -> turn-in)")
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
    lines.append("## 5. Field mapping — out (turn-out -> completion)")
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
    for rule in RULE_ROWS:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## 7. Docket")
    lines.append("")
    lines.append(
        "TF-158 has **no share** of TF-016 and found "
        "no new defect in existing code, so this "
        "slice does not edit the docket and adds no "
        "entry. TF-016 totals stand unchanged (137 "
        "entries — 32 OPEN / 1 OPEN-ENVIRONMENT / "
        "100 FIXED / 4 CLOSED-BY-VERDICT), verified "
        "live against the docket header and filed "
        "in the open here.")
    lines.append("")
    lines.append("## 8. The probes")
    lines.append("")
    for probe in HERMES_BINDING_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 9. Filed limits")
    lines.append("")
    lines.append(
        "- Proof grade FIXTURE: the checker and "
        "probes run a fake Hermes consumer shaped "
        "on the filed plugin-surface data. NO live "
        "Hermes process was routed and NO Hermes "
        "code was imported or copied; live "
        "end-to-end routing is NOT PROVEN — "
        "TF-170's Part IX gate owns the "
        "at-least-one-live proof, never claimed "
        "here.")
    lines.append(
        "- The usage split is partial by filing: "
        "TF-154 files one token count and Hermes's "
        "strict reply parser (cognition.py) "
        "requires integer prompt / completion "
        "tokens, so the filed completion carries "
        "the count as total_tokens with the split "
        "None — the binding does NOT claim that "
        "parser accepts the filed usage unchanged; "
        "reconciling the split is the live "
        "adapter's business at TF-170's gate.")
    lines.append(
        "- No credential ever enters the binding: "
        "Hermes carries endpoint keys in profile "
        "files / Kista custody on its side, the "
        "request validator refuses any credential "
        "field by name, and TF-160 owns credential "
        "custody on the ThoughtForge side.")
    lines.append(
        "- Routing is Hermes's act, never the "
        "binding's: the binding answers only turns "
        "Hermes routes to the local provider seat "
        "it files, and it holds no session, writes "
        "no entity / present-state file and keeps "
        "no transcript — history is "
        "caller-supplied in the request's "
        "messages.")
    lines.append(
        "- TF-159 owns the Ember binding; TF-160 "
        "owns credential custody (Kista); TF-161 "
        "owns binding fixtures in depth; TF-162 "
        "owns versioning enforcement in depth; "
        "TF-163 owns the local-only proof for "
        "bindings. TF-158 files the Hermes binding "
        "shape and its off-by-default routing "
        "under the volmarr-core contract.")
    lines.append(
        "- No canon data was touched (TF-017); no "
        "README.md anywhere was touched; no Docker "
        "was used; nothing under the Hermes tree "
        "was edited.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_hermes_binding_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / HERMES_BINDING_DOC
    doc.write_text(render_hermes_binding_markdown(),
                   encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.hermes_binding`."""
    doc = write_hermes_binding_doc()
    print(f"hermes binding doc regenerated: -> "
          f"{HERMES_BINDING_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def hermes_binding_problems() -> list[str]:
    """Re-derive TF-158's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    if HERMES_BINDING_FORMAT != "thoughtforge-hermes-binding/1":
        problems.append(
            f"the format is {HERMES_BINDING_FORMAT!r}, not "
            "thoughtforge-hermes-binding/1")
    if DEFAULT_ENABLED is not False:
        problems.append("DEFAULT_ENABLED is not False")
    if PROVIDER_KEY != "thoughtforge" \
            or PLUGIN_NAME != "volmarr-core":
        problems.append(
            f"the provider / plugin filing is "
            f"{PROVIDER_KEY!r} / {PLUGIN_NAME!r}, not "
            "thoughtforge / volmarr-core")
    if not DISABLED_ERROR or "disabled" not in DISABLED_ERROR:
        problems.append(
            "DISABLED_ERROR does not name the binding as "
            "disabled")
    if FINISH_REASON != "complete":
        problems.append(
            f"FINISH_REASON is {FINISH_REASON!r}, not "
            "the filed constant")

    provider_ids = [r["id"] for r in PROVIDER_ROWS]
    expected_providers = [
        "provider_volmarr_core", "provider_model_provider",
        "provider_cognition_routing",
        "provider_memory_provider",
        "provider_context_engine"]
    if provider_ids != expected_providers \
            or len(provider_ids) != len(set(provider_ids)):
        problems.append(
            f"the PROVIDER_ROWS filing is not as filed: "
            f"{provider_ids}")
    for row in PROVIDER_ROWS:
        problems.extend(provider_row_problems(row))
    context_row = next(
        r for r in PROVIDER_ROWS
        if r["id"] == "provider_context_engine")
    if context_row["answers_binding"] is not False:
        problems.append(
            "the context-engine row answers the binding "
            "— context compression must stay Hermes's "
            "own act")
    if FILED_MODEL.get("id") != PROVIDER_KEY \
            or FILED_CATALOG.get("object") != "list" \
            or FILED_CATALOG.get("data") != [FILED_MODEL]:
        problems.append(
            "FILED_CATALOG is not the filed one-model "
            "OpenAI model list")
    if FILED_HEALTH.get("healthy") is not True \
            or FILED_HEALTH.get("status") != "healthy" \
            or FILED_HEALTH.get("models") != [PROVIDER_KEY] \
            or FILED_HEALTH.get("latency_ms") is not None \
            or "no live model health is claimed" not in \
            str(FILED_HEALTH.get("detail")):
        problems.append("FILED_HEALTH is not as filed")

    call_ids = [r["id"] for r in CALL_ROWS]
    expected_calls = [
        "provider_turn", "provider_catalog",
        "provider_health", "memory_add", "memory_inspect",
        "memory_prune", "memory_export", "tool_execution",
        "gateway_messaging", "session_state_custody",
        "credential_custody", "provider_resolution",
        "memory_lifecycle"]
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
    for call_id in ("tool_execution", "gateway_messaging",
                    "session_state_custody",
                    "credential_custody",
                    "provider_resolution",
                    "memory_lifecycle"):
        row = next(r for r in CALL_ROWS if r["id"] == call_id)
        if row["routed"]:
            problems.append(
                f"Hermes-only proof: call {call_id} is "
                "routed — Hermes's own acts must stay "
                "NOT routed")

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
    key_row = next(
        (f for f in FIELD_ROWS if f["id"] == "request.api_key"),
        None)
    if key_row is None \
            or key_row.get("disposition") != "no_home_not_passed":
        problems.append(
            "the credential field is not filed no-home "
            "in FIELD_ROWS — the binding must never "
            "carry one")

    rule_ids = [r["id"] for r in RULE_ROWS]
    if len(rule_ids) != len(set(rule_ids)) \
            or len(RULE_ROWS) != 7:
        problems.append(
            "the RULE_ROWS filing is not seven unique rules")
    for rule in RULE_ROWS:
        problems.extend(rule_row_problems(rule))
    probe_ids = [p["id"] for p in HERMES_BINDING_PROBES]
    if len(probe_ids) != len(set(probe_ids)) \
            or len(HERMES_BINDING_PROBES) != 7:
        problems.append(
            "the HERMES_BINDING_PROBES filing is not "
            "seven unique probes")

    # The contract is TF-034-versioned and its
    # surface is clean.
    contract = hermes_binding_contract()
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
    from thoughtforge.service_shape import service_shape_problems

    for name, checker in (
            ("TF-153 service_shape", service_shape_problems),
            ("TF-154 cognition_api", cognition_api_problems),
            ("TF-155 memory_api", memory_api_problems),
            ("TF-156 runmal_binding", runmal_binding_problems),
            ("TF-157 samtal_binding", samtal_binding_problems)):
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

        binding = HermesBinding(
            cognition_factory=cog_factory,
            memory_factory=mem_factory)
        refused = binding.provider_turn(_REQUEST)
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
    # turn and a temp-store memory add / find,
    # through the binding only.
    try:
        binding, _tmp = _knowledge_binding()
        out = binding.provider_turn(_REQUEST)
        problems.extend(
            f"round-trip turn: {p}"
            for p in binding_response_problems(out))
        completion = out.get("completion") or {}
        choices = completion.get("choices") or [{}]
        if out.get("ok") is not True \
                or out.get("turn", {}).get("mode") \
                != "knowledge_only" \
                or choices[0].get("message", {}).get(
                    "content") \
                != out.get("turn", {}).get("text") \
                or completion.get("model") != PROVIDER_KEY:
            problems.append(
                f"round-trip turn is not a "
                f"knowledge_only turn with the filed "
                f"completion shape: {out}")
        added = binding.memory_add(
            "user_fact", summary="ferry at 07:30",
            record_id="fct_tf158_check")
        if added.get("ok") is not True \
                or added.get("memory", {}).get(
                    "record", {}).get("source") \
                != "manual_entry":
            problems.append(
                f"round-trip memory add is not a stamped "
                f"add: {added}")
        found = binding.memory_inspect(
            record_id="fct_tf158_check")
        if found.get("memory", {}).get("status") != "found":
            problems.append(
                f"round-trip memory find: {found}")
    except Exception as exc:
        problems.append(f"round-trip failed: {exc}")

    # The probes all behave.
    probes = run_hermes_binding_probes()
    if len(probes) != len(HERMES_BINDING_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the hermes binding probes do not all pass: "
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
            "FIXED; TF-158 must not move them")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / HERMES_BINDING_DOC
    if not doc.exists():
        problems.append(
            f"the hermes binding doc {HERMES_BINDING_DOC} "
            "is missing")
    elif doc.read_text(encoding="utf-8") \
            != render_hermes_binding_markdown():
        problems.append(
            f"the filed doc {HERMES_BINDING_DOC} is not "
            "the render verbatim (regenerate with "
            "python3 -m thoughtforge.hermes_binding)")
    return problems


def hermes_binding_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for
    consumers."""
    probes = run_hermes_binding_probes()
    return {
        "format": HERMES_BINDING_FORMAT,
        "doc": HERMES_BINDING_DOC,
        "contract": hermes_binding_contract(),
        "measured": {
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "CALL_IDS",
    "CALL_ROWS",
    "DEFAULT_ENABLED",
    "DISABLED_ERROR",
    "FIELD_IDS",
    "FIELD_ROWS",
    "FILED_CATALOG",
    "FILED_HEALTH",
    "FILED_MODEL",
    "FINISH_REASON",
    "HERMES_BINDING_DOC",
    "HERMES_BINDING_FORMAT",
    "HERMES_BINDING_PROBES",
    "HERMES_BINDING_RULES",
    "MESSAGE_FIELDS",
    "MESSAGE_ROLES",
    "NOT_ROUTED_CALL_IDS",
    "PLUGIN_NAME",
    "PROVIDER_KEY",
    "PROVIDER_ROW_IDS",
    "PROVIDER_ROWS",
    "REQUEST_FIELDS",
    "ROUTED_CALL_IDS",
    "RULE_IDS",
    "RULE_ROWS",
    "HermesBinding",
    "assemble_turn_in",
    "binding_response_problems",
    "binding_rule_problems",
    "call_row_problems",
    "completion_for",
    "field_row_problems",
    "field_rows_for",
    "hermes_binding_as_data",
    "hermes_binding_contract",
    "hermes_binding_problems",
    "main",
    "provider_request_problems",
    "provider_row_problems",
    "render_hermes_binding_markdown",
    "rule_row_problems",
    "run_hermes_binding_probes",
    "write_hermes_binding_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
