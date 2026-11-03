"""
TF-159 — the Ember binding, filed: the tiny
profile as Ember's cognition layer candidate
— budgets proven on phone_low first.

TF-159's roadmap text: *Ember binding: the
tiny profile as Ember's cognition layer
candidate — budgets proven on phone_low
first.* It is Part IX slice 7, following
TF-158's Hermes binding, TF-157's Samtal
binding, TF-156's Rúnmál binding, TF-155's
memory API, TF-154's cognition API and
TF-153's service shape, and it follows
TF-158's estate pattern in kind.

The measured start, probed live before
anything was built:

  * **No binding existed.** Nothing on the
    ThoughtForge side filed an Ember route:
    TF-154 filed the cognition API and TF-155
    the memory API as standalone service
    shapes, TF-156 / TF-157 / TF-158 filed
    the Rúnmál, Samtal and Hermes bindings —
    no ``thoughtforge.ember_binding`` module
    existed on this tree.
  * **Ember's side of the contract already
    exists and is a candidate, read READ-ONLY
    from ``~/workspace/ember-build`` (never
    edited, never imported).** Project Ember
    is ``ember-agent`` 0.2.0 (MIT): "a small,
    tethered, runs-anywhere AI companion" —
    an AI friend for normal folks on tiny
    hardware, whose tiny-hardware budgets are
    the project's soul. Its E-024 slice bound
    MindSpark ThoughtForge as Ember's
    candidate cognition layer (the layer
    *above* Funi, Ember's model backend)
    against the contract ThoughtForge froze
    at TF-034, and its filing names TF-159
    explicitly: "ThoughtForge's TF-159
    consumes the same contract from TF's
    side" and "ThoughtForge on phone-class
    hardware is TF-159's 'budgets proven on
    phone_low first' burden". The E-024
    shapes, filed below as literal data:

      - **cognition types**
        (``ember.schemas.cognition``):
        ``CognitionReply`` (text / provider /
        api_version / turn_id / mode /
        enforcement_passed / citations /
        token_count — ThoughtForge's
        FinalResponseRecord mapped straight,
        no prompt/completion split invented),
        ``CognitionHealth`` (provider /
        api_version / last_ok) and
        ``CognitionUnavailable`` with its
        reason enum (DISABLED /
        PROFILE_CANNOT_AFFORD /
        RUNTIME_NOT_INSTALLED /
        VERSION_MISMATCH / CONFIG_INVALID /
        UNKNOWN) — returned, never raised.
      - **the profile gate**
        (``CognitionConfig`` in
        ``ember.schemas.config``): enabled
        False by default; provider
        "thoughtforge", the only provider in
        that build; ``expected_api_major`` 1;
        Ember's profiles are
        ``ConversationProfile`` toaster /
        phone / pi / laptop and
        ``_PROFILE_COGNITION_DEFAULTS`` files
        toaster None (cannot afford the
        layer), phone 1, pi 1, laptop 2 as
        the candidate-draft cap handed to
        ThoughtForge's ``think()``; an
        explicit ``num_drafts`` wins, and a
        resolved cap below 1 is refused
        CONFIG_INVALID before any core is
        built. Custody defaults inside
        ``~/.ember/thoughtforge/`` (the E-020
        tether); ``tf_config`` passes through
        to the core unchanged; ``model_path``
        only when set (unset = knowledge-only).
      - **the handle + registry**
        (``ember.spark.cognition``):
        ``CognitionHandle`` (``think(user_text,
        episodes)`` / ``health()`` /
        ``close()``), ``open_cognition`` with
        its refusal order disabled -> profile
        cannot afford -> unknown provider ->
        the provider binding's own open, and
        ``cognition_registry()`` publishing
        the one implemented provider with
        api_major 1 and its afforded profiles.
      - **the adapter**
        (``ember.spark.cognition.thoughtforge``):
        ThoughtForge imported lazily inside
        ``open()`` only, zero new core
        dependencies (absent package = typed
        RUNTIME_NOT_INSTALLED); API-major /
        ``verify_api_surface()`` drift ->
        truthful VERSION_MISMATCH refusal;
        Ember Episodes become a ThoughtForge
        ``ChatHistory`` (``add_user`` /
        ``add_assistant`` per episode, oldest
        first) on the frozen ``history``
        parameter; core exceptions fold into
        ERROR-shaped replies; ``close()``
        unloads via ``unload_model()``.
      - **candidate honesty on Ember's side**:
        chat is NOT rewired — ``ember chat``
        still thinks with Funi; no live
        ThoughtForge turn was claimed there
        (a fake core shaped from the TF-034
        freeze), and no budgets were claimed:
        TF-159 owns the phone_low budget
        proof. NO Ember code is imported or
        copied anywhere in this module, and
        nothing in the Ember tree was edited.
  * **The tiny profile is ThoughtForge's
    ``phone_low``.** ThoughtForge's weakest
    shipped hardware profile
    (``hardware_profiles/phone_low.json``:
    ~2 GB RAM, a <=1.1 B model at q2_k) is
    the floor TF-037 proved the whole core
    on. Ember's ``phone`` profile is the
    Ember-side seat that pairs with it: the
    binding files the phone_low budgets —
    resolved LIVE from the shipped profile
    through the tree's own loaders, never
    invented — as its filed budgets, and
    its checker re-derives every filed
    number live (floor budget TF-037,
    latency budgets TF-033, cold-start
    budgets TF-079, memory budgets TF-056,
    all over the same profile JSON). No
    other ThoughtForge profile's budgets
    are filed or claimed in this slice:
    phone_low first means phone_low only,
    until a later slice proves another.
  * **The neighbours stood clean live**:
    ``service_shape_problems()``,
    ``cognition_api_problems()``,
    ``memory_api_problems()``,
    ``runmal_binding_problems()``,
    ``samtal_binding_problems()`` and
    ``hermes_binding_problems()`` all
    returned [] on this tree before the
    build.

**What this slice files**

  * ``PROFILE_ROWS`` — Ember's four
    conversation profiles as literal data:
    which can afford the cognition layer,
    the draft cap E-024 files for each, and
    the ThoughtForge profile pairing (only
    phone -> phone_low is filed, the pairing
    this slice proves; pi / laptop route
    under their E-024 caps but carry no
    filed ThoughtForge budget pairing here,
    and toaster cannot afford the layer at
    all).
  * ``BUDGET_ROWS`` — the phone_low budgets
    as literal data, every number copied
    from the profile's own definitions in
    this tree and cited to its source
    module: the profile JSON's hardware /
    model / memory / retrieval / generation
    sections (via TF-037's ``floor_budget()``,
    TF-056's memory budget and the profile
    loader), TF-033's latency budget
    (500 / 3,000 / 1,500 ms, total 5,000)
    and TF-079's cold-start budget
    (150 / 1,500 / 3,000 / 3,000 ms, total
    7,650). The checker re-derives each row
    live; a drifted source fails it by name.
  * ``RULE_ROWS`` — the binding's law, as
    literal data: Ember's candidate is real
    and named (no Ember API invented);
    candidate, never a chat rewire; budgets
    proven on phone_low first; toaster
    cannot afford; off by default; called,
    never reimplemented; no Ember code
    imported or copied; every field
    disposed; responses detached, JSON-safe,
    TF-034-versioned.
  * ``CALL_ROWS`` — every Ember-side call
    mapped, as literal data. A routed
    cognition turn routes to a TF-154
    ``CognitionAPI`` turn and returns both
    verbatim and as the filed Ember
    ``CognitionReply`` shape; the profile
    gate, registry and health calls answer
    from the filed rows (the binding's own
    state, never a live Ember process);
    memory calls route to TF-155
    ``MemoryAPI``'s four ops, called, never
    reimplemented. Chat rewiring, the Funi
    model turn, tool execution, persona
    custody, Episode / Well custody and
    credential custody are filed NOT
    routed, each with its reason — they are
    Ember's own acts under E-024 / E-025.
  * ``FIELD_ROWS`` — the Ember request ->
    turn-in mapping and the turn-out ->
    Ember reply mapping, as literal data:
    every field with a disposition, none
    silently dropped or invented. The filed
    rule: the request's ``user_text`` is
    turn-in ``user_text``; each Episode
    rides as two history entries (its
    ``operator_input`` as role "user", its
    ``ember_reply`` as role "assistant",
    oldest first — E-024's ChatHistory
    rule); the turn's draft cap is the
    explicit ``num_drafts`` when given, else
    the profile's E-024 cap; any credential
    is filed no-home (E-024 involves none).
  * ``EmberBinding`` — wraps a TF-154
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
    method (``disable()`` reverses it). An
    enabled turn on the toaster profile
    returns the filed cannot-afford refusal
    as data and routes nothing. Every
    response is detached JSON-safe data
    whose ``contract_version`` is TF-034's
    ``API_VERSION``, consumed.
  * ``ember_binding_problems()`` — a checker
    that re-derives every claim live: the
    filings, the filed budgets against their
    live sources, the TF-034 surface
    (through the neighbour checkers), a
    disabled binding refusing with nothing
    built or called, a toaster refusal
    routing nothing, a knowledge-only
    cognition round-trip inside the
    phone_low response cap and a temp-store
    memory round-trip through the binding,
    fake-consumer consistency, the filed
    probes, the docket header and the filed
    doc.

**Found and fixed in the open** (by this
slice's own first checker run, before any
test was written): the first draft of the
history mapping passed the assembled entries
to TF-154 as a bare list of dicts — the shape
stub APIs accept — and the enabled
round-trip against a real knowledge-only
core failed with ``'list' object has no
attribute 'to_prompt_context'``:
``ThoughtForgeCore.think()`` reads its
``history`` through that method. The binding
now carries Episodes in a ``ChatHistory``
built with ThoughtForge's own classes
(``add_user`` / ``add_assistant`` per
Episode) — E-024's adapter rule, exactly —
while ``assemble_history()`` keeps the pure
data form the filings describe. The defect
never existed in banked code (it was caught
in this slice's unbanked draft), so it takes
no TF-016 entry; it is filed here in the
open.

**Docket**: TF-159 has no share of TF-016 and
finds no new defect in existing code; it does
NOT edit ``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md``.
TF-016 totals stand unchanged (137 entries —
31 OPEN / 1 OPEN-ENVIRONMENT / 101 FIXED / 4
CLOSED-BY-VERDICT), verified live against
the docket header and filed in the open here
and in the render.

The human-readable doc is
``docs/architecture/EMBER_BINDING.md``,
filed verbatim as this module's render
(regenerate with
``python3 -m thoughtforge.ember_binding``).
Proof grade: FIXTURE — a fake Ember consumer
shaped on the filed E-024 data; NO live
Ember process was routed and NO Ember code
was imported or copied. Live end-to-end
routing is NOT PROVEN — TF-170's Part IX
gate owns the at-least-one-live proof,
never claimed here. The budgets are proven
as filed data re-derived live from the
phone_low profile's own definitions plus
TF-037's floor proof of the core at that
profile — no phone hardware ran and no
real small model served a turn here.
TF-160 owns credential custody; TF-161 owns
binding fixtures in depth; TF-162 owns
versioning enforcement in depth.
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
from thoughtforge.cognition.small_model_floor import (
    FLOOR_PROFILE_ID,
    floor_budget,
)
from thoughtforge.cognition_api import (
    TURN_IN_FIELDS,
    TURN_OUT_FIELDS,
    CognitionAPI,
    cognition_api_problems,
)
from thoughtforge.hermes_binding import hermes_binding_problems
from thoughtforge.memory_api import (
    OP_IDS as MEMORY_OP_IDS,
)
from thoughtforge.memory_api import (
    MemoryAPI,
    memory_api_problems,
)
from thoughtforge.runmal_binding import runmal_binding_problems
from thoughtforge.samtal_binding import samtal_binding_problems

EMBER_BINDING_FORMAT = "thoughtforge-ember-binding/1"
EMBER_BINDING_DOC = "docs/architecture/EMBER_BINDING.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

#: The provider key Ember's E-024 files for
#: this binding — the only provider in that
#: build, named "thoughtforge" there.
PROVIDER_KEY = "thoughtforge"

#: Ember's package, as its pyproject files it
#: (ember-agent 0.2.0, read READ-ONLY).
EMBER_PACKAGE = "ember-agent"
EMBER_VERSION = "0.2.0"

#: The ThoughtForge profile the tiny pairing
#: is proven on — the floor profile, TF-037.
TINY_PROFILE_ID = FLOOR_PROFILE_ID

#: The Ember profile that pairs with the
#: tiny ThoughtForge profile in this slice.
EMBER_PHONE_PROFILE = "phone"

#: The Ember profile a request defaults to
#: when it names none — the candidate seat
#: this slice files (the phone pairing).
DEFAULT_EMBER_PROFILE = "phone"

#: The binding is OFF unless someone enables
#: it explicitly — filed as data, proven by
#: the probes (Ember's own CognitionConfig
#: is likewise enabled=False by default).
DEFAULT_ENABLED = False

#: The refusal every routed call returns
#: while the binding is disabled — as data,
#: naming the binding as disabled.
DISABLED_ERROR = (
    "ember binding is disabled: the TF-159 "
    "binding is off by default — enable it "
    "explicitly (EmberBinding(enabled=True) "
    "or binding.enable()) before any call "
    "routes")

#: The refusal an enabled toaster-profile
#: turn returns — as data, naming E-024's
#: PROFILE_CANNOT_AFFORD truth.
TOASTER_ERROR = (
    "ember binding refused: the toaster "
    "profile cannot afford the cognition "
    "layer (E-024 PROFILE_CANNOT_AFFORD) — "
    "it runs a second full turn pipeline "
    "beside Funi, and the smallest profile "
    "has no room for it")

#: Ember's conversation profiles, in
#: ConversationProfile's filed order.
EMBER_PROFILES: tuple[str, ...] = (
    "toaster", "phone", "pi", "laptop")

#: The fields an Ember-side turn request may
#: carry (the CognitionHandle think shape —
#: user_text plus Episodes — with the
#: CognitionConfig profile / draft fields
#: the gate needs).
REQUEST_FIELDS: tuple[str, ...] = (
    "user_text", "episodes", "profile", "num_drafts")

#: The fields one Ember Episode carries into
#: history (episode.operator_input /
#: episode.ember_reply, E-024's adapter).
EPISODE_FIELDS: tuple[str, ...] = (
    "operator_input", "ember_reply")

#: The fields Ember's CognitionReply carries
#: (ember.schemas.cognition, E-024).
REPLY_FIELDS: tuple[str, ...] = (
    "text", "provider", "api_version", "turn_id",
    "mode", "enforcement_passed", "citations",
    "token_count")


# ── The Ember profile rows, as literal data ──────────────────────────────────
#
# One row per ConversationProfile Ember
# files. Each row files E-024's real gate
# truth for that profile, read READ-ONLY
# from the Ember tree; the binding invents
# no Ember profile and affords none itself —
# it files the gate a candidate answers
# behind.

PROFILE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "profile_toaster",
     "ember_profile": "toaster",
     "ember_side": "ConversationProfile.TOASTER — the "
                   "smallest profile, the project's "
                   "namesake floor; CognitionConfig's "
                   "_PROFILE_COGNITION_DEFAULTS files "
                   "it None",
     "thoughtforge_profile": None,
     "affords": False,
     "draft_cap": None,
     "binding_method": None,
     "enabled_by_default": False,
     "summary": "cannot afford the layer: E-024 "
                "refuses it outright "
                "(PROFILE_CANNOT_AFFORD) — a second "
                "full turn pipeline beside Funi has "
                "no room on the smallest machine, "
                "and this binding returns that "
                "refusal as data rather than routing"},
    {"id": "profile_phone",
     "ember_profile": "phone",
     "ember_side": "ConversationProfile.PHONE — "
                   "phone-class hardware in Ember's "
                   "Part III budget book (E-041: phone "
                   "profile, E-048's phone-class "
                   "proof simulated); "
                   "_PROFILE_COGNITION_DEFAULTS files "
                   "its draft cap 1",
     "thoughtforge_profile": "phone_low",
     "affords": True,
     "draft_cap": 1,
     "binding_method": "ember_turn(request)",
     "enabled_by_default": False,
     "summary": "the tiny pairing this slice proves: "
                "Ember's phone seat over "
                "ThoughtForge's phone_low floor — "
                "the filed BUDGET_ROWS are "
                "phone_low's budgets, proven first, "
                "and a phone turn routes with the "
                "E-024 draft cap 1"},
    {"id": "profile_pi",
     "ember_profile": "pi",
     "ember_side": "ConversationProfile.PI — the "
                   "profile CognitionConfig defaults "
                   "to; _PROFILE_COGNITION_DEFAULTS "
                   "files its draft cap 1",
     "thoughtforge_profile": None,
     "affords": True,
     "draft_cap": 1,
     "binding_method": "ember_turn(request)",
     "enabled_by_default": False,
     "summary": "afforded by E-024 with draft cap 1 "
                "and routed by the binding under "
                "that cap — but NO ThoughtForge "
                "profile pairing or budget proof is "
                "filed for it in this slice: "
                "phone_low first means the pi "
                "pairing is a later slice's proof, "
                "never claimed here"},
    {"id": "profile_laptop",
     "ember_profile": "laptop",
     "ember_side": "ConversationProfile.LAPTOP — "
                   "the largest Ember profile; "
                   "_PROFILE_COGNITION_DEFAULTS files "
                   "its draft cap 2 (TF-034's "
                   "by-construction local maximum)",
     "thoughtforge_profile": None,
     "affords": True,
     "draft_cap": 2,
     "binding_method": "ember_turn(request)",
     "enabled_by_default": False,
     "summary": "afforded by E-024 with draft cap 2 "
                "and routed under that cap — no "
                "ThoughtForge budget pairing filed "
                "in this slice (phone_low first)"},
)

PROFILE_ROW_IDS: tuple[str, ...] = tuple(
    r["id"] for r in PROFILE_ROWS)


def profile_row_for(
    ember_profile: str,
) -> dict[str, Any] | None:
    """The filed profile row for one Ember
    profile name, or None when unfiled."""
    for row in PROFILE_ROWS:
        if row["ember_profile"] == ember_profile:
            return row
    return None


def resolved_num_drafts(
    ember_profile: str,
    explicit: int | None = None,
) -> int | None:
    """E-024's draft-cap rule, applied purely:
    an explicit num_drafts wins; otherwise the
    profile's filed cap applies; None when the
    profile cannot afford the layer (or is
    unfiled)."""
    if explicit is not None:
        return explicit
    row = profile_row_for(ember_profile)
    if row is None:
        return None
    return row["draft_cap"]


# ── The filed honest answers (registry / health) ─────────────────────────────

#: The registry an enabled binding files —
#: exactly E-024's cognition_registry() shape
#: (provider / implemented / api_major /
#: afforded_profiles), as data.
FILED_REGISTRY: dict[str, Any] = {
    "provider": PROVIDER_KEY,
    "implemented": True,
    "api_major": 1,
    "afforded_profiles": [
        r["ember_profile"] for r in PROFILE_ROWS
        if r["affords"]],
}

#: The health answer an enabled binding files
#: — E-024's CognitionHealth shape (provider /
#: api_version / last_ok) plus binding-honesty
#: fields. It reports the binding, never a
#: live Ember process or a live model, and no
#: credential was read or validated for it
#: (E-024's binding involves no credential).
FILED_HEALTH: dict[str, Any] = {
    "provider": PROVIDER_KEY,
    "api_version": API_VERSION,
    "last_ok": None,
    "healthy": True,
    "status": "healthy",
    "detail": "ember binding enabled: cognition "
              "routes through TF-154's CognitionAPI "
              "and memory through TF-155's MemoryAPI "
              f"(contract {API_VERSION}) under the "
              "filed phone_low budgets; this answer "
              "reports the binding's own state, not a "
              "live Ember process or a live model — "
              "no credential was read or validated "
              "and no live model health is claimed",
}


# ── The phone_low budgets, as literal data ───────────────────────────────────
#
# Every value below is copied exactly from the
# phone_low profile's own definitions in this
# tree and cited to its source. The checker
# re-derives each row live through those same
# sources (floor_budget() / the latency,
# cold-start and memory budget resolvers /
# the profile JSON loader) — a drifted source
# or a drifted filing fails by name. These are
# the ONLY budgets this binding files:
# phone_low first, phone_low only.

BUDGET_SOURCE_FLOOR = (
    "hardware_profiles/phone_low.json via "
    "thoughtforge.cognition.small_model_floor."
    "floor_budget() (TF-037)")
BUDGET_SOURCE_PROFILE = (
    "hardware_profiles/phone_low.json via "
    "thoughtforge.utils.config."
    "load_hardware_profile('phone_low')")
BUDGET_SOURCE_LATENCY = (
    "thoughtforge.cognition.latency_budgets."
    "budget_for_profile('phone_low') (TF-033)")
BUDGET_SOURCE_COLD = (
    "thoughtforge.inference.cold_start_budgets."
    "budget_for_profile('phone_low') (TF-079)")
BUDGET_SOURCE_MEMORY = (
    "hardware_profiles/phone_low.json via "
    "thoughtforge.knowledge.memory_budgets."
    "budget_for_profile('phone_low') (TF-056)")

BUDGET_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "profile.ram_gb", "category": "profile",
     "value": 2, "source": BUDGET_SOURCE_PROFILE,
     "summary": "the floor machine's RAM, in GB — "
                "the ~2 GB phone-class host the "
                "tiny profile names"},
    {"id": "profile.max_params_b", "category": "profile",
     "value": 1.1, "source": BUDGET_SOURCE_PROFILE,
     "summary": "the largest model the profile "
                "carries, in billions of parameters"},
    {"id": "profile.quantization", "category": "profile",
     "value": "q2_k", "source": BUDGET_SOURCE_PROFILE,
     "summary": "the profile's shipped quantization"},
    {"id": "profile.recommended_models",
     "category": "profile",
     "value": ["TinyLlama-1.1B-Chat",
               "Granite-1B-Instruct"],
     "source": BUDGET_SOURCE_PROFILE,
     "summary": "the profile JSON's own recommended "
                "models — filed, never extended"},
    {"id": "generation.max_response_tokens",
     "category": "generation",
     "value": 180, "source": BUDGET_SOURCE_FLOOR,
     "summary": "the per-turn response cap, in "
                "word-tokens (the profile's "
                "generation section)"},
    {"id": "generation.token_budget_per_turn",
     "category": "generation",
     "value": 180, "source": BUDGET_SOURCE_FLOOR,
     "summary": "the per-turn token budget (the "
                "profile's model section); the floor's "
                "binding response cap is the tighter "
                "of this and max_response_tokens — "
                "both 180, so the cap is 180"},
    {"id": "generation.max_context_tokens",
     "category": "generation",
     "value": 512, "source": BUDGET_SOURCE_FLOOR,
     "summary": "the context ceiling a tiny model "
                "must hold, in tokens"},
    {"id": "generation.draft_count",
     "category": "generation",
     "value": 2, "source": BUDGET_SOURCE_FLOOR,
     "summary": "the profile's own candidate-draft "
                "count — a ThoughtForge-side cap; "
                "Ember's phone gate is stricter (1) "
                "and the stricter cap governs an "
                "Ember phone turn"},
    {"id": "generation.max_refinement_passes",
     "category": "generation",
     "value": 1, "source": BUDGET_SOURCE_FLOOR,
     "summary": "the salvage refinement cap on the "
                "floor (TF-037's fixed defect: the "
                "profile's cap, not a flat 2)"},
    {"id": "generation.temperature",
     "category": "generation",
     "value": 0.7, "source": BUDGET_SOURCE_PROFILE,
     "summary": "the profile's generation temperature"},
    {"id": "generation.repeat_penalty",
     "category": "generation",
     "value": 1.1, "source": BUDGET_SOURCE_PROFILE,
     "summary": "the profile's repeat penalty"},
    {"id": "generation.top_p", "category": "generation",
     "value": 0.9, "source": BUDGET_SOURCE_PROFILE,
     "summary": "the profile's top-p"},
    {"id": "memory.max_memory_bundle_facts",
     "category": "memory",
     "value": 5, "source": BUDGET_SOURCE_FLOOR,
     "summary": "the memory-bundle cap per turn, in "
                "facts (also TF-056's bundle cap)"},
    {"id": "memory.max_vector_results",
     "category": "memory",
     "value": 3, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the vector-results cap per retrieval"},
    {"id": "memory.episodic_store_cap_soft",
     "category": "memory",
     "value": 64, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the episodic store's soft cap, in "
                "records (a report line, not an "
                "eviction trigger — TF-056)"},
    {"id": "memory.episodic_store_cap_hard",
     "category": "memory",
     "value": 128, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the episodic store's hard cap, in "
                "records — eviction fires here"},
    {"id": "memory.user_facts_cap_soft",
     "category": "memory",
     "value": 32, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the user-facts soft cap, in records"},
    {"id": "memory.user_facts_cap_hard",
     "category": "memory",
     "value": 64, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the user-facts hard cap, in records"},
    {"id": "memory.response_patterns_cap_soft",
     "category": "memory",
     "value": 32, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the response-patterns soft cap, in "
                "records"},
    {"id": "memory.response_patterns_cap_hard",
     "category": "memory",
     "value": 64, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the response-patterns hard cap, in "
                "records"},
    {"id": "memory.user_preferences_cap_soft",
     "category": "memory",
     "value": 32, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the user-preferences soft cap, in "
                "records"},
    {"id": "memory.user_preferences_cap_hard",
     "category": "memory",
     "value": 64, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the user-preferences hard cap, in "
                "records"},
    {"id": "retrieval.sql_max_results",
     "category": "retrieval",
     "value": 5, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the SQL retrieval leg's result cap"},
    {"id": "retrieval.vector_top_k",
     "category": "retrieval",
     "value": 3, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the vector retrieval leg's top-k"},
    {"id": "retrieval.retrieval_timeout_ms",
     "category": "retrieval",
     "value": 500, "source": BUDGET_SOURCE_MEMORY,
     "summary": "the retrieval timeout, in ms — the "
                "profile's own stated number, the "
                "same value TF-033's latency budget "
                "reads live for its retrieval stage"},
    {"id": "latency.retrieval_ms", "category": "latency",
     "value": 500, "source": BUDGET_SOURCE_LATENCY,
     "summary": "TF-033's retrieval-stage ceiling "
                "for phone_low, in ms (the profile's "
                "retrieval_timeout_ms, unfactored)"},
    {"id": "latency.generation_ms", "category": "latency",
     "value": 3000, "source": BUDGET_SOURCE_LATENCY,
     "summary": "TF-033's generation-stage ceiling "
                "for phone_low, in ms (the 1,000 ms "
                "desktop base x the filed phone_low "
                "factor 3.0)"},
    {"id": "latency.salvage_ms", "category": "latency",
     "value": 1500, "source": BUDGET_SOURCE_LATENCY,
     "summary": "TF-033's salvage-stage ceiling for "
                "phone_low, in ms (the 500 ms base x "
                "3.0)"},
    {"id": "latency.total_ms", "category": "latency",
     "value": 5000, "source": BUDGET_SOURCE_LATENCY,
     "summary": "the latency total — the three stage "
                "budgets' sum by construction, never "
                "an independent number"},
    {"id": "cold_start.profile_load_ms",
     "category": "cold_start",
     "value": 150, "source": BUDGET_SOURCE_COLD,
     "summary": "TF-079's profile-load ceiling for "
                "phone_low, in ms (the 50 ms base x "
                "the TF-033 factor 3.0)"},
    {"id": "cold_start.core_construction_ms",
     "category": "cold_start",
     "value": 1500, "source": BUDGET_SOURCE_COLD,
     "summary": "TF-079's core-construction ceiling "
                "for phone_low, in ms (500 x 3.0)"},
    {"id": "cold_start.engine_load_ms",
     "category": "cold_start",
     "value": 3000, "source": BUDGET_SOURCE_COLD,
     "summary": "TF-079's no-model engine-load "
                "ceiling for phone_low, in ms "
                "(1,000 x 3.0)"},
    {"id": "cold_start.first_token_ms",
     "category": "cold_start",
     "value": 3000, "source": BUDGET_SOURCE_COLD,
     "summary": "TF-079's first-token ceiling for "
                "phone_low, in ms (the campaign-set "
                "1,000 ms base x 3.0 — a ceiling for "
                "the shape, not a model's measured "
                "TTFT; a live model's TTFT is NOT "
                "PROVEN, TF-079's own filing)"},
    {"id": "cold_start.total_ms",
     "category": "cold_start",
     "value": 7650, "source": BUDGET_SOURCE_COLD,
     "summary": "the cold-start TTFT ceiling — the "
                "four stage budgets' sum by "
                "construction"},
)

BUDGET_IDS: tuple[str, ...] = tuple(
    r["id"] for r in BUDGET_ROWS)

BUDGET_CATEGORIES: tuple[str, ...] = (
    "profile", "generation", "memory", "retrieval",
    "latency", "cold_start")


def budget_value(budget_id: str) -> Any:
    """One filed budget's value, by id."""
    for row in BUDGET_ROWS:
        if row["id"] == budget_id:
            return row["value"]
    raise KeyError(budget_id)


def expected_budget_values() -> dict[str, Any]:
    """The phone_low budgets re-derived LIVE from
    their source definitions in this tree — the
    values BUDGET_ROWS must file, by id. This is
    the 'proven on phone_low first' derivation:
    the profile JSON through the tree's own
    loaders and the TF-037 / TF-056 / TF-033 /
    TF-079 resolvers, never a second table."""
    from thoughtforge.inference.cold_start_budgets import (
        budget_for_profile as cold_budget_for,
    )
    from thoughtforge.knowledge.memory_budgets import (
        budget_for_profile as memory_budget_for,
    )
    from thoughtforge.cognition.latency_budgets import (
        budget_for_profile as latency_budget_for,
    )
    from thoughtforge.utils.config import (
        load_hardware_profile,
    )

    profile = load_hardware_profile(TINY_PROFILE_ID)
    floor = floor_budget(TINY_PROFILE_ID)
    memory = memory_budget_for(TINY_PROFILE_ID)
    latency = latency_budget_for(TINY_PROFILE_ID)
    cold = cold_budget_for(TINY_PROFILE_ID)
    generation = profile.get("generation") or {}
    return {
        "profile.ram_gb": profile["hardware"]["ram_gb"],
        "profile.max_params_b":
            profile["model"]["max_params_b"],
        "profile.quantization":
            profile["model"]["quantization"],
        "profile.recommended_models":
            list(profile["model"]["recommended_models"]),
        "generation.max_response_tokens":
            floor.max_response_tokens,
        "generation.token_budget_per_turn":
            floor.token_budget_per_turn,
        "generation.max_context_tokens":
            floor.max_context_tokens,
        "generation.draft_count": floor.draft_count,
        "generation.max_refinement_passes":
            floor.max_refinement_passes,
        "generation.temperature": generation["temperature"],
        "generation.repeat_penalty":
            generation["repeat_penalty"],
        "generation.top_p": generation["top_p"],
        "memory.max_memory_bundle_facts":
            floor.max_memory_bundle_facts,
        "memory.max_vector_results":
            memory.max_vector_results,
        "memory.episodic_store_cap_soft":
            memory.caps["episode"]["soft"],
        "memory.episodic_store_cap_hard":
            memory.caps["episode"]["hard"],
        "memory.user_facts_cap_soft":
            memory.caps["user_fact"]["soft"],
        "memory.user_facts_cap_hard":
            memory.caps["user_fact"]["hard"],
        "memory.response_patterns_cap_soft":
            memory.caps["response_pattern"]["soft"],
        "memory.response_patterns_cap_hard":
            memory.caps["response_pattern"]["hard"],
        "memory.user_preferences_cap_soft":
            memory.caps["user_preference"]["soft"],
        "memory.user_preferences_cap_hard":
            memory.caps["user_preference"]["hard"],
        "retrieval.sql_max_results": memory.sql_max_results,
        "retrieval.vector_top_k": memory.vector_top_k,
        "retrieval.retrieval_timeout_ms":
            memory.retrieval_timeout_ms,
        "latency.retrieval_ms": latency.retrieval_ms,
        "latency.generation_ms": latency.generation_ms,
        "latency.salvage_ms": latency.salvage_ms,
        "latency.total_ms": latency.total_ms,
        "cold_start.profile_load_ms": cold.profile_load_ms,
        "cold_start.core_construction_ms":
            cold.core_construction_ms,
        "cold_start.engine_load_ms": cold.engine_load_ms,
        "cold_start.first_token_ms": cold.first_token_ms,
        "cold_start.total_ms": cold.total_ms,
    }


# ── The rules, as literal data ───────────────────────────────────────────────

RULE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "ember_candidate_real_and_named",
     "rule": "Ember's candidate is real and named: "
             "Project Ember (ember-agent 0.2.0) bound "
             "ThoughtForge at its E-024 under the "
             "TF-034 freeze — ember.schemas.cognition's "
             "types, CognitionConfig's gate, "
             "ember.spark.cognition's handle / registry "
             "and its thoughtforge adapter — and E-024's "
             "filing names TF-159 as the ThoughtForge-side "
             "consumer of the same contract; the binding "
             "is shaped on those filed surfaces and "
             "invents no Ember API"},
    {"id": "candidate_never_a_chat_rewire",
     "rule": "candidate means candidate: Ember's chat "
             "turn still thinks with Funi and this "
             "binding does not rewire it — the binding "
             "answers cognition calls with data, and "
             "wiring the candidate into Ember's chat is "
             "a later deliberate act on Ember's side, "
             "never a silent swap of the companion's "
             "mind"},
    {"id": "budgets_proven_on_phone_low_first",
     "rule": "the binding's budgets are phone_low's "
             "budgets, proven first: every BUDGET_ROWS "
             "value is copied from the phone_low "
             "profile's own definitions in this tree "
             "(its shipped JSON, TF-037's floor budget, "
             "TF-056's memory budget, TF-033's latency "
             "budget, TF-079's cold-start budget) and "
             "the checker re-derives each live — no "
             "number is invented, and no other "
             "profile's budgets are filed or claimed "
             "until a later slice proves them"},
    {"id": "toaster_cannot_afford",
     "rule": "the toaster profile cannot afford the "
             "layer (E-024): an enabled toaster turn "
             "returns the filed PROFILE_CANNOT_AFFORD "
             "refusal as data and routes nothing — the "
             "refusal is the truth about the smallest "
             "machine, not a malfunction"},
    {"id": "off_by_default",
     "rule": "the binding is off by default: "
             "DEFAULT_ENABLED is False, filed as data "
             "(matching Ember's own CognitionConfig "
             "enabled=False); enabling is an explicit "
             "act — the constructor flag enabled=True "
             "or the enable() method — and disable() "
             "returns the binding to refusal; while "
             "disabled, every routed call returns a "
             "truthful refusal as data (ok=False, "
             "status=\"refused\", an error naming the "
             "binding as disabled) and touches neither "
             "API — no API is built and no call is "
             "routed, proven by the probes with "
             "counting factories"},
    {"id": "called_never_reimplemented",
     "rule": "cognition routes only through TF-154's "
             "CognitionAPI and memory only through "
             "TF-155's MemoryAPI (which itself calls "
             "TF-057's toolset) — this module "
             "implements no cognition, store or "
             "enforcement logic of its own, and no "
             "Ember handle, gate or adapter logic "
             "either"},
    {"id": "no_ember_code_imported_or_copied",
     "rule": "no Ember code is imported or copied: "
             "Ember's shapes are filed as literal data "
             "read READ-ONLY from the Ember tree "
             "(ember.schemas.cognition, "
             "ember.schemas.config's CognitionConfig / "
             "ConversationProfile, ember.spark."
             "cognition's handle and registry, its "
             "thoughtforge adapter, and E-024's doc), "
             "and nothing under the Ember tree is "
             "edited by this slice; no credential is "
             "ever filed, passed or stored by the "
             "binding (E-024's binding involves none "
             "at all — TF-160 owns custody)"},
    {"id": "every_field_has_a_disposition",
     "rule": "every field on either side has a "
             "FIELD_ROWS row with a disposition — "
             "none is silently dropped and none is "
             "invented; any credential is filed "
             "no-home in the open"},
    {"id": "responses_detached_json_safe_versioned",
     "rule": "every response is detached JSON-safe "
             "data (a JSON round-trip copy) whose "
             "contract_version is TF-034's "
             "API_VERSION, consumed from "
             "thoughtforge.cognition.api_surface — "
             "no competing version is invented here"},
)

#: TF-158-style name for the binding's rules.
EMBER_BINDING_RULES = RULE_ROWS

RULE_IDS: tuple[str, ...] = tuple(
    r["id"] for r in RULE_ROWS)


# ── The calls, as literal data ───────────────────────────────────────────────

CALL_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "ember_turn",
     "surface": "cognition",
     "ember_side": "one cognition turn on Ember's "
                   "handle: CognitionHandle.think"
                   "(user_text, episodes) on a handle "
                   "opened through open_cognition for "
                   "an afforded profile (E-024)",
     "thoughtforge_route": "TF-154 CognitionAPI.turn — "
                           "the turn-in assembled by "
                           "the filed rule in FIELD_ROWS "
                           "(the request's user_text, "
                           "each Episode as two history "
                           "entries, the resolved draft "
                           "cap as num_drafts); the "
                           "turn-out returns verbatim "
                           "in the response's turn and "
                           "mapped straight into the "
                           "filed CognitionReply shape",
     "binding_method": "ember_turn(request)",
     "routed": True,
     "summary": "one cognition turn through "
                "ThoughtForge for one Ember handle "
                "turn"},
    {"id": "profile_gate",
     "surface": "ember_gate",
     "ember_side": "the profile gate read: "
                   "CognitionConfig.affords() / "
                   "resolved_num_drafts over "
                   "ConversationProfile (E-024)",
     "thoughtforge_route": "the filed PROFILE_ROWS "
                           "row — an honest answer "
                           "from E-024's own gate "
                           "table: affords, the draft "
                           "cap, and the ThoughtForge "
                           "profile pairing filed for "
                           "that profile (phone_low "
                           "for phone, none filed for "
                           "the others in this slice)",
     "binding_method": "profile_gate(profile)",
     "routed": True,
     "summary": "the gate truth for one Ember "
                "profile, as data"},
    {"id": "registry",
     "surface": "ember_gate",
     "ember_side": "the registry read: "
                   "ember.spark.cognition."
                   "cognition_registry() publishing "
                   "the one implemented provider, its "
                   "API major and afforded profiles",
     "thoughtforge_route": "the filed FILED_REGISTRY "
                           "row — E-024's registry "
                           "shape as data; no Ember "
                           "process is queried",
     "binding_method": "registry()",
     "routed": True,
     "summary": "the provider registry for the "
                "ThoughtForge entry"},
    {"id": "health_check",
     "surface": "ember_gate",
     "ember_side": "the health read: CognitionHandle."
                   "health() answering a "
                   "CognitionHealth (provider / "
                   "api_version / last_ok) for "
                   "ember-doctor-shaped consumers",
     "thoughtforge_route": "the filed FILED_HEALTH "
                           "row — reports the binding's "
                           "own enabled state against "
                           "the TF-154 CognitionAPI / "
                           "TF-155 MemoryAPI contract; "
                           "claims no live Ember, no "
                           "live model and no credential "
                           "validation (E-024 involves "
                           "no credential)",
     "binding_method": "health_check()",
     "routed": True,
     "summary": "the health answer for the "
                "ThoughtForge entry"},
    {"id": "memory_add",
     "surface": "memory",
     "ember_side": "a memory write for the "
                   "ThoughtForge core Ember hosts "
                   "(its store under Ember custody, "
                   "~/.ember/thoughtforge/ by "
                   "E-024's defaults) — not Ember's "
                   "own Well (see episode_well_custody)",
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
     "summary": "an Ember-hosted memory write lands "
                "as a TF-155 add"},
    {"id": "memory_inspect",
     "surface": "memory",
     "ember_side": "a memory read for the hosted "
                   "ThoughtForge core",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"inspect\", request) — "
                           "TF-057's inspect_store / "
                           "inspect_records / find_record "
                           "through TF-155, called, never "
                           "reimplemented; read-only",
     "binding_method": "memory_call(\"inspect\", request) "
                       "/ memory_inspect(...)",
     "routed": True,
     "summary": "Ember-hosted memory reads land as "
                "TF-155 inspects"},
    {"id": "memory_prune",
     "surface": "memory",
     "ember_side": "a memory lifecycle pass for the "
                   "hosted ThoughtForge core",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"prune\", request) — "
                           "TF-057's MemoryLifecycle pass "
                           "through TF-155, called, never "
                           "reimplemented",
     "binding_method": "memory_call(\"prune\", request) "
                       "/ memory_prune(...)",
     "routed": True,
     "summary": "Ember-hosted memory removal routes "
                "as TF-155's lifecycle prune"},
    {"id": "memory_export",
     "surface": "memory",
     "ember_side": "a store export for the hosted "
                   "ThoughtForge core (the companion's "
                   "memory must be carriable with its "
                   "Ember folder — the E-020 tether)",
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
     "summary": "the store export is reachable "
                "through the binding"},
    {"id": "chat_rewire",
     "surface": "ember_only",
     "ember_side": "Ember's chat turn: ember chat "
                   "still thinks with Funi — E-024 "
                   "binds the candidate and does not "
                   "wire it into the turn",
     "thoughtforge_route": "not routed: this binding "
                           "never rewires Ember's chat, "
                           "never replaces the Funi turn "
                           "and files no chat wiring — "
                           "wiring the candidate in is a "
                           "later deliberate act on "
                           "Ember's side",
     "binding_method": None,
     "routed": False,
     "summary": "the chat rewire stays Ember's own "
                "later act; nothing routes to "
                "ThoughtForge"},
    {"id": "funi_model_turn",
     "surface": "ember_only",
     "ember_side": "the Funi model turn: Ember's "
                   "model backend answering one prompt "
                   "with one completion, one layer "
                   "below the cognition layer",
     "thoughtforge_route": "not routed: the binding is "
                           "the cognition layer above "
                           "Funi, never a model backend "
                           "— it runs no prompt, loads "
                           "no model and answers no "
                           "completion request",
     "binding_method": None,
     "routed": False,
     "summary": "the Funi turn stays Ember's own "
                "act; nothing routes to ThoughtForge"},
    {"id": "tool_execution",
     "surface": "ember_only",
     "ember_side": "tool calls under Ember's own "
                   "tools discipline (E-026) — the "
                   "companion's tools are Ember's "
                   "surface",
     "thoughtforge_route": "not routed: a cognition "
                           "layer candidate executes no "
                           "tool — the binding files no "
                           "tool call, runs no tool and "
                           "answers no tool result",
     "binding_method": None,
     "routed": False,
     "summary": "tool execution stays Ember's own "
                "act; nothing routes to ThoughtForge"},
    {"id": "persona_custody",
     "surface": "ember_only",
     "ember_side": "persona custody: Ember's persona "
                   "core (E-025) owns who the "
                   "companion is — persona files, "
                   "resolution and the prompt wall",
     "thoughtforge_route": "not routed: the binding "
                           "holds no persona, resolves "
                           "none and assembles no prompt "
                           "— persona reaches a turn "
                           "only inside ThoughtForge's "
                           "own core, under its own "
                           "config (tf_config, Ember's "
                           "side at open)",
     "binding_method": None,
     "routed": False,
     "summary": "persona custody stays Ember's own "
                "act; nothing routes to ThoughtForge"},
    {"id": "episode_well_custody",
     "surface": "ember_only",
     "ember_side": "Episode / Well custody: Ember's "
                   "Well is the companion's own "
                   "memory store; Episodes are its "
                   "records",
     "thoughtforge_route": "not routed: the binding "
                           "keeps no Episode and reads "
                           "no Well — episodes reach a "
                           "turn only as the caller "
                           "supplies them in the "
                           "request, and the TF-155 "
                           "memory rows above are the "
                           "hosted ThoughtForge core's "
                           "store, never Ember's Well",
     "binding_method": None,
     "routed": False,
     "summary": "Episode / Well custody stays "
                "Ember's own act; nothing routes to "
                "ThoughtForge"},
    {"id": "credential_custody",
     "surface": "ember_only",
     "ember_side": "credential custody: E-024's "
                   "filing is explicit — no credential "
                   "is involved in this binding at all",
     "thoughtforge_route": "not routed: credentials "
                           "have no home here and "
                           "TF-160 owns custody on the "
                           "ThoughtForge side — the "
                           "binding reads no key file, "
                           "accepts no key in a request "
                           "(the request validator "
                           "refuses one) and stores none",
     "binding_method": None,
     "routed": False,
     "summary": "credential custody stays unfiled "
                "here by E-024's own truth (TF-160's "
                "on this side); nothing routes to "
                "ThoughtForge"},
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
# gate_only_not_passed / no_home_not_passed /
# no_source_not_set / no_source_default_applies /
# construction_only / binding_filed. None is
# silently dropped; none is invented.

FIELD_ROWS: tuple[dict[str, Any], ...] = (
    # ── In: Ember request -> TF-154 turn-in ──
    {"direction": "in",
     "id": "request.user_text",
     "source": "the request's user_text — the "
               "CognitionHandle.think(user_text, ...) "
               "argument (E-024)",
     "target": "turn-in user_text",
     "disposition": "mapped",
     "summary": "the turn's question, verbatim — "
                "TF-154 requires user_text as a "
                "non-empty string, so an empty "
                "user_text is an invalid request, "
                "named by ember_request_problems()"},
    {"direction": "in",
     "id": "episodes.operator_input",
     "source": "each Episode's operator_input (the "
               "person's side of one saved exchange)",
     "target": "turn-in history — one user-role "
               "entry per Episode, in order",
     "disposition": "mapped_as_history",
     "summary": "E-024's ChatHistory rule, first "
                "half: the adapter calls add_user per "
                "episode, oldest first — the binding "
                "builds the same ChatHistory with "
                "ThoughtForge's own classes (the "
                "object TF-154's core reads via "
                "to_prompt_context) and files this "
                "mapping as its data form"},
    {"direction": "in",
     "id": "episodes.ember_reply",
     "source": "each Episode's ember_reply (the "
               "companion's side of the same saved "
               "exchange)",
     "target": "turn-in history — one assistant-role "
               "entry per Episode, immediately after "
               "its operator entry",
     "disposition": "mapped_as_history",
     "summary": "E-024's ChatHistory rule, second "
                "half: add_assistant per episode — "
                "the companion's earlier words ride "
                "as assistant history, never rewritten "
                "as user words; an empty episodes list "
                "passes no history at all (TF-154 "
                "passes only set kwargs)"},
    {"direction": "in",
     "id": "request.profile",
     "source": "the request's profile — an Ember "
               "ConversationProfile name (toaster / "
               "phone / pi / laptop); absent, the "
               "filed DEFAULT_EMBER_PROFILE (phone, "
               "the candidate seat this slice proves) "
               "applies",
     "target": "the binding's gate and the response "
               "envelope; no turn-in field",
     "disposition": "gate_only_not_passed",
     "summary": "the profile decides whether the "
                "turn may route at all (toaster: the "
                "cannot-afford refusal) and which "
                "draft cap applies; TF-154's turn-in "
                "files no profile field, so the name "
                "is never passed as turn data"},
    {"direction": "in",
     "id": "profile.draft_cap",
     "source": "the profile's E-024 draft cap "
               "(the profile defaults table: phone 1, "
               "pi 1, laptop 2, toaster None)",
     "target": "turn-in num_drafts, when no "
               "explicit num_drafts is given",
     "disposition": "mapped",
     "summary": "the profile's cap is the turn's "
                "draft cap by default — on phone it "
                "is 1, stricter than phone_low's own "
                "draft_count 2, and the stricter cap "
                "governs"},
    {"direction": "in",
     "id": "request.num_drafts",
     "source": "the request's explicit num_drafts "
               "(CognitionConfig.num_drafts, E-024)",
     "target": "turn-in num_drafts",
     "disposition": "mapped",
     "summary": "an explicit num_drafts wins over "
                "the profile cap (E-024's rule); it "
                "must be an integer >= 1 when present "
                "(TF-034's ruling — a value below 1 "
                "is a caller error, named by the "
                "validator), and on the toaster "
                "profile no explicit value buys "
                "affordance: the gate refuses first"},
    {"direction": "in",
     "id": "request.api_key",
     "source": "nothing in an Ember cognition "
               "request carries a credential: E-024 "
               "files that no credential is involved "
               "in this binding at all",
     "target": "no target — never filed, never "
               "passed, never stored",
     "disposition": "no_home_not_passed",
     "summary": "the binding holds no credential "
                "and authenticates as no one: a "
                "request carrying an api_key (or any "
                "unfiled field) is refused by the "
                "validator, named — credential "
                "custody is TF-160's slice"},
    {"direction": "in",
     "id": "turn_in.retrieval_path",
     "source": "nothing on the Ember side files a "
               "retrieval path",
     "target": "turn-in retrieval_path",
     "disposition": "no_source_not_set",
     "summary": "the binding never sets it; the "
                "core's own default retrieval applies "
                "— no Ember value is invented for it"},
    {"direction": "in",
     "id": "turn_in.want_trace",
     "source": "nothing on the Ember side files a "
               "trace request",
     "target": "turn-in want_trace",
     "disposition": "no_source_default_applies",
     "summary": "the binding never sets it; TF-154's "
                "filed default (True) applies, so a "
                "routed turn carries its trace status "
                "honestly in the returned turn"},
    {"direction": "in",
     "id": "config.tf_config",
     "source": "CognitionConfig.tf_config (E-024) — "
               "the plain dict Ember passes through "
               "to the ThoughtForge core constructor "
               "at open time on Ember's side",
     "target": "no turn-in field — core "
               "construction data on Ember's side of "
               "the contract",
     "disposition": "construction_only",
     "summary": "tf_config is open-time data, never "
                "turn data: this binding's APIs are "
                "injected already-constructed (or "
                "built by TF-154's / TF-155's own "
                "defaults), so no tf_config is "
                "accepted in a request, shadowed or "
                "invented here"},
    {"direction": "in",
     "id": "config.custody_paths",
     "source": "CognitionConfig.memory_dir / db_path "
               "(E-024) — defaulting inside the "
               "~/.ember/thoughtforge/ folder under "
               "the E-020 tether",
     "target": "no turn-in field — custody data on "
               "Ember's side of the contract",
     "disposition": "construction_only",
     "summary": "where the hosted core keeps its "
                "state is Ember's custody decision at "
                "open; the binding is handed its APIs "
                "and files no path of its own"},
    # ── Out: TF-154 turn-out -> the Ember-shaped reply ──
    {"direction": "out",
     "id": "turn.text",
     "source": "turn-out text",
     "target": "the response's turn (verbatim) and "
               "the reply's text — the CognitionReply "
               "field Ember's handle returns",
     "disposition": "mapped",
     "summary": "the turn's words are the reply's "
                "words, verbatim — an honestly empty "
                "text rides as-is, never dressed up "
                "(E-024's adapter rule)"},
    {"direction": "out",
     "id": "turn.response_id",
     "source": "turn-out response_id",
     "target": "the response's turn (verbatim); no "
               "CognitionReply field home",
     "disposition": "mapped",
     "summary": "Ember's CognitionReply files no "
                "response id; the id rides the "
                "binding response's turn, never "
                "dropped"},
    {"direction": "out",
     "id": "turn.turn_id",
     "source": "turn-out turn_id",
     "target": "the response's turn (verbatim) and "
               "the reply's turn_id",
     "disposition": "mapped",
     "summary": "CognitionReply carries the turn id "
                "— the same value in both places"},
    {"direction": "out",
     "id": "turn.mode",
     "source": "turn-out mode",
     "target": "the response's turn (verbatim) and "
               "the reply's mode",
     "disposition": "mapped",
     "summary": "the turn's mode (knowledge_only "
                "when no model served) rides both "
                "places — an Ember consumer can see "
                "honestly how its answer was produced"},
    {"direction": "out",
     "id": "turn.enforcement_passed",
     "source": "turn-out enforcement_passed",
     "target": "the response's turn (verbatim) and "
               "the reply's enforcement_passed",
     "disposition": "mapped",
     "summary": "the enforcement verdict rides both "
                "places verbatim"},
    {"direction": "out",
     "id": "turn.citations",
     "source": "turn-out citations",
     "target": "the response's turn (verbatim) and "
               "the reply's citations",
     "disposition": "mapped",
     "summary": "the turn's citations ride both "
                "places (the reply carries them as a "
                "detached list — CognitionReply files "
                "a tuple on Ember's side, a JSON list "
                "here, same strings in same order)"},
    {"direction": "out",
     "id": "turn.token_count",
     "source": "turn-out token_count",
     "target": "the response's turn (verbatim) and "
               "the reply's token_count",
     "disposition": "mapped",
     "summary": "the provider's single count for "
                "the answer text — E-024 invents no "
                "prompt/completion split because "
                "ThoughtForge's record carries none, "
                "and neither does this binding"},
    {"direction": "out",
     "id": "turn.contract_version",
     "source": "turn-out contract_version",
     "target": "the response's turn (verbatim), the "
               "binding envelope's contract_version "
               "and the reply's api_version — the "
               "same TF-034 value",
     "disposition": "mapped",
     "summary": "one version, TF-034's, in every "
                "place it appears; no competing "
                "version is filed"},
    {"direction": "out",
     "id": "turn.trace",
     "source": "turn-out trace",
     "target": "the response's turn (verbatim); no "
               "CognitionReply field home",
     "disposition": "mapped",
     "summary": "TF-084's joined trace rides the "
                "turn when TF-154 includes it; "
                "Ember's reply shape carries no trace"},
    {"direction": "out",
     "id": "turn.trace_format",
     "source": "turn-out trace_format",
     "target": "the response's turn (verbatim); no "
               "CognitionReply field home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim beside the "
                "trace"},
    {"direction": "out",
     "id": "turn.trace_status",
     "source": "turn-out trace_status",
     "target": "the response's turn (verbatim); no "
               "CognitionReply field home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim — included "
                "/ not_requested / unavailable with "
                "its reason named, exactly as TF-154 "
                "filed it"},
    {"direction": "out",
     "id": "reply.provider",
     "source": "this binding — the filed "
               "PROVIDER_KEY",
     "target": "the reply's provider",
     "disposition": "binding_filed",
     "summary": "the reply answers in the provider "
                "name Ember's registry files: "
                "thoughtforge, the only provider "
                "in E-024's build"},
    {"direction": "out",
     "id": "reply.api_version",
     "source": "TF-034's API_VERSION, consumed",
     "target": "the reply's api_version — the "
               "version CognitionReply reports and "
               "CognitionHealth carries",
     "disposition": "binding_filed",
     "summary": "the reply reports the frozen "
                "contract version the turn was "
                "answered under — Ember's adapter "
                "refuses a different major at open "
                "(E-024), and this binding files the "
                "same 1.0.0 it consumes"},
)

FIELD_IDS: tuple[str, ...] = tuple(
    f["id"] for f in FIELD_ROWS)


def field_rows_for(direction: str) -> tuple[dict[str, Any], ...]:
    """One direction's filed field rows, in filing
    order."""
    return tuple(f for f in FIELD_ROWS
                 if f["direction"] == direction)


# ── The contract as plain data ───────────────────────────────────────────────

def ember_binding_contract() -> dict[str, Any]:
    """The Ember binding contract as plain
    JSON-safe data, built from the filed literals
    above and TF-034's version. A fresh copy each
    call."""
    return {
        "format": EMBER_BINDING_FORMAT,
        "contract_version": API_VERSION,
        "provider_key": PROVIDER_KEY,
        "ember_package": EMBER_PACKAGE,
        "ember_version": EMBER_VERSION,
        "tiny_profile": TINY_PROFILE_ID,
        "default_ember_profile": DEFAULT_EMBER_PROFILE,
        "default_enabled": DEFAULT_ENABLED,
        "disabled_error": DISABLED_ERROR,
        "toaster_error": TOASTER_ERROR,
        "profiles": [dict(r) for r in PROFILE_ROWS],
        "budgets": [dict(r) for r in BUDGET_ROWS],
        "registry": json.loads(json.dumps(FILED_REGISTRY)),
        "health": dict(FILED_HEALTH),
        "calls": [dict(r) for r in CALL_ROWS],
        "fields": [dict(f) for f in FIELD_ROWS],
        "rules": [dict(r) for r in RULE_ROWS],
    }


# ── Pure validators ──────────────────────────────────────────────────────────

def profile_row_problems(row: Any) -> list[str]:
    """Judge one Ember-profile filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["profile row: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("profile row: the row files no id")
    if row.get("ember_profile") not in EMBER_PROFILES:
        problems.append(
            f"profile row {row.get('id')}: ember_profile "
            f"is {row.get('ember_profile')!r}, not one of "
            f"Ember's profiles {list(EMBER_PROFILES)}")
    for key in ("ember_side", "summary"):
        if not row.get(key):
            problems.append(
                f"profile row {row.get('id')}: no {key} filed")
    if not isinstance(row.get("affords"), bool):
        problems.append(
            f"profile row {row.get('id')}: affords is "
            "not a boolean")
    elif row["affords"] and not row.get("binding_method"):
        problems.append(
            f"profile row {row.get('id')}: affords the "
            "layer but no binding_method is filed")
    elif not row["affords"] and row.get("binding_method"):
        problems.append(
            f"profile row {row.get('id')}: cannot afford "
            "the layer but a binding_method is filed")
    if row.get("affords"):
        cap = row.get("draft_cap")
        if isinstance(cap, bool) or not isinstance(cap, int) \
                or cap < 1:
            problems.append(
                f"profile row {row.get('id')}: affords "
                "but draft_cap is not an integer >= 1")
    elif row.get("draft_cap") is not None:
        problems.append(
            f"profile row {row.get('id')}: cannot afford "
            "but a draft_cap is filed")
    if row.get("thoughtforge_profile") is not None \
            and row.get("thoughtforge_profile") != TINY_PROFILE_ID:
        problems.append(
            f"profile row {row.get('id')}: files a "
            f"ThoughtForge pairing "
            f"{row.get('thoughtforge_profile')!r} that is "
            f"not the tiny profile {TINY_PROFILE_ID!r} — "
            "no other pairing is proven in this slice")
    if row.get("enabled_by_default") is not False:
        problems.append(
            f"profile row {row.get('id')}: "
            "enabled_by_default is not False")
    return problems


def budget_row_problems(row: Any) -> list[str]:
    """Judge one budget filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["budget row: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("budget row: the row files no id")
    if row.get("category") not in BUDGET_CATEGORIES:
        problems.append(
            f"budget row {row.get('id')}: category is "
            f"{row.get('category')!r}, not one of "
            f"{list(BUDGET_CATEGORIES)}")
    if "value" not in row or row.get("value") is None:
        problems.append(
            f"budget row {row.get('id')}: no value filed")
    for key in ("source", "summary"):
        if not row.get(key):
            problems.append(
                f"budget row {row.get('id')}: no {key} filed")
    return problems


def call_row_problems(row: Any) -> list[str]:
    """Judge one call filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["call: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("call: the row files no id")
    if row.get("surface") not in (
            "cognition", "memory", "ember_gate",
            "ember_only"):
        problems.append(
            f"call {row.get('id')}: surface is "
            f"{row.get('surface')!r}, not cognition / "
            "memory / ember_gate / ember_only")
    for key in ("ember_side", "thoughtforge_route", "summary"):
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
    if row.get("surface") == "ember_only" \
            and row.get("routed"):
        problems.append(
            f"call {row.get('id')}: an ember_only call "
            "is routed — Ember's own acts never route")
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


def ember_request_problems(request: Any) -> list[str]:
    """Judge one ember-turn request, purely.
    Empty-safe.

    Shape only: the Ember handle request (exactly
    user_text / episodes / profile / num_drafts; an
    episode is exactly operator_input / ember_reply,
    both strings, the operator side non-empty; the
    profile one of Ember's four; an explicit
    num_drafts an integer >= 1). What the core does
    with the turn is TF-154's business."""
    problems: list[str] = []
    if not isinstance(request, dict) or not request:
        return ["ember request: request must be a "
                "mapping shaped like {user_text, "
                "episodes, profile, num_drafts} — the "
                "Ember handle request E-024 files"]
    for key in request:
        if key not in REQUEST_FIELDS:
            problems.append(
                f"ember request: request files {key!r}, "
                f"not a filed Ember request field "
                f"{list(REQUEST_FIELDS)} — above all, no "
                "credential is ever a request field here")
    if not isinstance(request.get("user_text"), str) \
            or not request.get("user_text"):
        problems.append(
            "ember request: user_text must be a "
            "non-empty string — TF-154's turn-in "
            "requires it")
    if "profile" in request \
            and request["profile"] not in EMBER_PROFILES:
        problems.append(
            f"ember request: profile is "
            f"{request['profile']!r}, not one of "
            f"Ember's profiles {list(EMBER_PROFILES)}")
    if "num_drafts" in request:
        value = request["num_drafts"]
        if isinstance(value, bool) \
                or not isinstance(value, int) \
                or value < 1:
            problems.append(
                "ember request: num_drafts must be an "
                "integer >= 1 when present (TF-034's "
                "ruling — a value below 1 is a caller "
                "error)")
    episodes = request.get("episodes")
    if episodes is not None:
        if not isinstance(episodes, list):
            problems.append(
                "ember request: episodes must be a list "
                "of episode mappings")
            return problems
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                problems.append(
                    f"ember request: episode {index} is "
                    "not a mapping shaped like "
                    "{operator_input, ember_reply}")
                continue
            for key in episode:
                if key not in EPISODE_FIELDS:
                    problems.append(
                        f"ember request: episode {index} "
                        f"files {key!r}, not an Ember "
                        f"episode field "
                        f"{list(EPISODE_FIELDS)}")
            if not isinstance(
                    episode.get("operator_input"), str) \
                    or not episode.get("operator_input"):
                problems.append(
                    f"ember request: episode {index} "
                    "operator_input must be a non-empty "
                    "string")
            if not isinstance(
                    episode.get("ember_reply"), str):
                problems.append(
                    f"ember request: episode {index} "
                    "ember_reply must be a string")
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


# ── The filed history assembly + the reply shape ─────────────────────────────

def assemble_history(
    episodes: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """The filed assembly rule, applied purely:
    each Episode, in order, becomes two history
    entries — its operator_input as role "user",
    its ember_reply as role "assistant" (E-024's
    ChatHistory rule as data). Assumes
    ember_request_problems() judged the request
    clean."""
    history: list[dict[str, Any]] = []
    for episode in episodes or []:
        history.append({
            "role": "user",
            "content": episode["operator_input"],
        })
        history.append({
            "role": "assistant",
            "content": episode["ember_reply"],
        })
    return history


def chat_history_for(
    episodes: list[dict[str, Any]] | None,
) -> Any:
    """The history object a routed turn actually
    carries: a ``ChatHistory`` built with
    ThoughtForge's own classes — ``add_user`` /
    ``add_assistant`` per Episode, oldest first
    (E-024's adapter rule, exactly). None when
    there are no Episodes (TF-154 passes only
    set kwargs, and the core's frozen default
    is no history).

    The object matters, not just the data:
    TF-154 passes ``history`` through to
    ``ThoughtForgeCore.think()``, which reads
    it via ``to_prompt_context()`` — a bare list
    of dicts is accepted by stub APIs but a
    real core refuses it ('list' object has no
    attribute 'to_prompt_context', measured
    live by this slice's first checker run and
    fixed in the open here)."""
    entries = assemble_history(episodes)
    if not entries:
        return None
    from thoughtforge.cognition.chat_history import (
        ChatHistory,
    )

    history = ChatHistory()
    for episode in episodes or []:
        history.add_user(episode["operator_input"])
        history.add_assistant(episode["ember_reply"])
    return history


def reply_for(
    turn: dict[str, Any],
) -> dict[str, Any]:
    """The filed CognitionReply shape for one
    turn-out, in E-024's ember.schemas.cognition
    form: the turn's fields mapped straight, the
    binding's provider key, TF-034's version as
    api_version — no token split invented."""
    return {
        "text": turn.get("text", ""),
        "provider": PROVIDER_KEY,
        "api_version": API_VERSION,
        "turn_id": turn.get("turn_id", ""),
        "mode": turn.get("mode", ""),
        "enforcement_passed": bool(
            turn.get("enforcement_passed", False)),
        "citations": list(turn.get("citations") or []),
        "token_count": turn.get("token_count"),
    }


# ── The binding ──────────────────────────────────────────────────────────────

def _detached(data: Any) -> Any:
    """A detached JSON-safe copy (a JSON round-trip)."""
    return json.loads(json.dumps(data))


class EmberBinding:
    """Ember's candidate cognition through
    ThoughtForge, shaped on E-024's contract,
    off by default, budgets phone_low first.

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
    touches neither API. An enabled turn on
    the toaster profile returns the filed
    cannot-afford refusal as data and routes
    nothing. The binding performs no Ember
    act: it rewires no chat, runs no Funi
    turn, executes no tool, holds no persona,
    Episode, Well or credential — a cognition
    layer candidate only, behind Ember's own
    E-024 surfaces.
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

    # ── The candidate's cognition calls ──

    def ember_turn(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """The routed candidate turn: an
        Ember-handle-shaped request dict
        (``{user_text, episodes, profile,
        num_drafts}``) in, one binding response
        out — the TF-154 turn verbatim in
        ``turn`` and the filed CognitionReply
        shape in ``reply``.

        While disabled, the filed refusal returns
        as data and nothing is judged, built or
        routed. When enabled, an invalid request
        raises ValueError naming the problem(s),
        judged by ``ember_request_problems()``
        before any API is built or called; a
        toaster-profile request returns the
        filed cannot-afford refusal as data and
        routes nothing."""
        if not self._enabled:
            return self._refused("ember_turn")
        problems = ember_request_problems(request)
        if problems:
            raise ValueError("; ".join(problems))
        profile = request.get(
            "profile", DEFAULT_EMBER_PROFILE)
        row = profile_row_for(profile)
        if row is None or not row["affords"]:
            return self._envelope(
                "ember_turn", False, "refused",
                error=TOASTER_ERROR, profile=profile)
        drafts = resolved_num_drafts(
            profile, request.get("num_drafts"))
        kwargs: dict[str, Any] = {}
        history = chat_history_for(request.get("episodes"))
        if history is not None:
            kwargs["history"] = history
        if drafts is not None:
            kwargs["num_drafts"] = drafts
        turn = self._get_cognition().turn(
            request["user_text"], **kwargs)
        return self._envelope(
            "ember_turn", True, "replied",
            profile=profile, turn=turn,
            reply=reply_for(turn))

    def profile_gate(
        self,
        profile: str,
    ) -> dict[str, Any]:
        """The gate call: the filed PROFILE_ROWS
        row for one Ember profile, as the
        response's ``gate`` — affords, the draft
        cap and the filed ThoughtForge pairing.
        An unfiled profile name raises ValueError.
        While disabled, the filed refusal returns
        as data."""
        if not self._enabled:
            return self._refused("profile_gate")
        row = profile_row_for(profile)
        if row is None:
            raise ValueError(
                f"ember profile {profile!r} is not one "
                f"of Ember's profiles "
                f"{list(EMBER_PROFILES)}")
        gate = _detached(row)
        return self._envelope(
            "profile_gate", True,
            "afforded" if row["affords"]
            else "cannot_afford",
            gate=gate)

    def registry(self) -> dict[str, Any]:
        """The registry call: the filed
        FILED_REGISTRY row as the response's
        ``registry`` — E-024's registry shape as
        data; no Ember process is queried. While
        disabled, the filed refusal returns as
        data."""
        if not self._enabled:
            return self._refused("registry")
        return self._envelope(
            "registry", True, "listed",
            registry=_detached(FILED_REGISTRY))

    def health_check(self) -> dict[str, Any]:
        """The health call: the filed FILED_HEALTH
        row as the response's ``health`` — it
        reports the binding, never a live Ember
        process or live model, and reads no
        credential. While disabled, the filed
        refusal returns as data."""
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

class _FakeEmberConsumer:
    """A dict-shaped stand-in for an Ember-side
    cognition consumer, built from the filed
    E-024 data only: it files a CognitionConfig-
    shaped dict (enabled False by default,
    provider thoughtforge, expected_api_major
    1), builds handle-shaped requests from
    Episodes, and reads the filed registry.
    Fixture only — no Ember code, no live
    process."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {
            "enabled": False,
            "provider": PROVIDER_KEY,
            "profile": DEFAULT_EMBER_PROFILE,
            "expected_api_major": 1,
            "num_drafts": None,
        }

    def request(
        self, user_text: str,
        episodes: list[dict[str, Any]] | None = None,
        profile: str = DEFAULT_EMBER_PROFILE,
    ) -> dict[str, Any]:
        return {"user_text": user_text,
                "episodes": [dict(e) for e in episodes or []],
                "profile": profile}

    def registry_providers(
        self, registry: dict[str, Any],
    ) -> list[str]:
        return [registry["provider"]]


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


def _knowledge_binding() -> tuple[EmberBinding, Any]:
    """An enabled binding over a knowledge-only
    core (TF-154's TF-084 fixture shape) and a
    temp store. Returns the binding and the
    temp home."""
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.knowledge.store import MemoryStore

    tmp = tempfile.mkdtemp(prefix="tf159-")
    core = ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    store = MemoryStore(memory_dir=Path(tmp) / "store")
    binding = EmberBinding(
        cognition_api=CognitionAPI(core=core),
        memory_api=MemoryAPI(store=store),
        enabled=True,
    )
    return binding, tmp


def _temp_memory_binding() -> tuple[EmberBinding, Any, str]:
    """An enabled binding whose memory side is a
    TF-155 MemoryAPI over a fresh temp store
    (the cognition side stays unbuilt unless a
    cognition call lands)."""
    from thoughtforge.knowledge.store import MemoryStore

    tmp = tempfile.mkdtemp(prefix="tf159-")
    store = MemoryStore(memory_dir=Path(tmp) / "store")
    binding = EmberBinding(
        memory_api=MemoryAPI(store=store), enabled=True)
    return binding, store, tmp


_REQUEST = {"user_text": "What time does the ferry leave?",
            "profile": "phone",
            "episodes": [
                {"operator_input": "hello",
                 "ember_reply": "hi there"}]}


# ── The filed probes ─────────────────────────────────────────────────────────

EMBER_BINDING_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "disabled_refusal_touches_nothing",
     "kind": "disabled_refusal",
     "detail": "a default binding refuses every "
               "routed call as data (ok=False, the "
               "error naming the binding disabled), "
               "and counting factories prove neither "
               "API is ever built"},
    {"id": "enabled_cognition_round_trip_through_binding",
     "kind": "cognition_round_trip",
     "detail": "an enabled binding routes an Ember "
               "handle-shaped request through the "
               "injected CognitionAPI by the filed "
               "rule (the request's user_text, each "
               "Episode as two history entries carried "
               "in a ChatHistory built with "
               "ThoughtForge's own classes, the "
               "phone draft cap 1 as num_drafts) and "
               "returns the turn with the filed "
               "CognitionReply shape (the turn's "
               "fields mapped straight, the provider "
               "key, TF-034's version)"},
    {"id": "enabled_memory_round_trip_temp_store",
     "kind": "memory_round_trip",
     "detail": "an enabled binding adds through "
               "TF-155 (stamped manual_entry), finds "
               "the record through TF-155's inspect, "
               "and exports the verified bundle — all "
               "through the binding, on a temp store"},
    {"id": "budgets_are_phone_low_live",
     "kind": "budgets_live",
     "detail": "every BUDGET_ROWS value equals the "
               "value re-derived live from the "
               "phone_low profile's own definitions "
               "(floor_budget / memory / latency / "
               "cold-start resolvers over the shipped "
               "JSON), and a knowledge-only turn "
               "through the binding lands inside the "
               "filed phone_low response cap (180 "
               "tokens)"},
    {"id": "toaster_refused_and_gate_is_honest",
     "kind": "toaster_gate",
     "detail": "an enabled toaster-profile turn "
               "returns the filed cannot-afford "
               "refusal as data and routes nothing "
               "(the recording API files no turn); "
               "the profile gate answers phone as "
               "afforded with draft cap 1 and the "
               "phone_low pairing, toaster as "
               "cannot_afford, and the registry "
               "files exactly E-024's afforded "
               "profiles"},
    {"id": "fake_consumer_consistency",
     "kind": "fake_consumer",
     "detail": "a fake Ember consumer shaped on "
               "the filed E-024 data files its "
               "CognitionConfig-shaped dict (off by "
               "default, provider thoughtforge, "
               "expected major 1), builds a "
               "handle-shaped request, and routes "
               "its turn through the binding only: "
               "the reply's text is the turn's text, "
               "the reply's provider is the "
               "consumer's provider, the consumer's "
               "registry read agrees with the "
               "binding's registry, and the filed "
               "profile rows are consistent (phone "
               "paired with phone_low, toaster "
               "affording nothing)"},
    {"id": "doc_matches_render_and_version_is_tf034",
     "kind": "doc_and_version",
     "detail": "the filed doc is the render "
               "verbatim, the contract version is "
               "TF-034's API_VERSION with its surface "
               "clean, and DEFAULT_ENABLED is False"},
)


def run_ember_binding_probes() -> list[dict[str, Any]]:
    """Run the filed probes; each returns its
    verdict."""
    results: list[dict[str, Any]] = []
    for probe in EMBER_BINDING_PROBES:
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

                binding = EmberBinding(
                    cognition_factory=cog_factory,
                    memory_factory=mem_factory)
                outs = [
                    binding.ember_turn(_REQUEST),
                    binding.profile_gate("phone"),
                    binding.registry(),
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
                binding = EmberBinding(
                    cognition_api=fake, enabled=True)
                out = binding.ember_turn(_REQUEST)
                reply = out.get("reply") or {}
                turn_call = fake.turns[0] if fake.turns else {}
                history = turn_call.get("history")
                history_entries = (
                    history.to_messages()
                    if history is not None else [])
                passed = out.get("ok") is True \
                    and len(fake.turns) == 1 \
                    and turn_call.get("user_text") \
                    == "What time does the ferry leave?" \
                    and turn_call.get("num_drafts") == 1 \
                    and history_entries == [
                        {"role": "user",
                         "content": "hello"},
                        {"role": "assistant",
                         "content": "hi there"}] \
                    and out["turn"]["text"] \
                    == "stub answer to: What time does " \
                       "the ferry leave?" \
                    and reply.get("text") \
                    == out["turn"]["text"] \
                    and reply.get("provider") == PROVIDER_KEY \
                    and reply.get("api_version") == API_VERSION \
                    and reply.get("turn_id") == "turn_stub" \
                    and set(reply) == set(REPLY_FIELDS) \
                    and binding_response_problems(out) == []
            elif kind == "memory_round_trip":
                binding, _store, tmp = _temp_memory_binding()
                added = binding.memory_add(
                    "user_fact", summary="ferry at 07:30",
                    record_id="fct_tf159")
                found = binding.memory_inspect(
                    record_id="fct_tf159")
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
            elif kind == "budgets_live":
                expected = expected_budget_values()
                filed = {r["id"]: r["value"]
                         for r in BUDGET_ROWS}
                binding, _tmp = _knowledge_binding()
                out = binding.ember_turn(_REQUEST)
                cap = budget_value(
                    "generation.max_response_tokens")
                passed = filed == expected \
                    and cap == 180 \
                    and out.get("ok") is True \
                    and out.get("turn", {}).get("mode") \
                    == "knowledge_only" \
                    and out["turn"]["token_count"] <= cap \
                    and binding_response_problems(out) == []
            elif kind == "toaster_gate":
                fake = _RecordingCognitionAPI()
                binding = EmberBinding(
                    cognition_api=fake, enabled=True)
                refused = binding.ember_turn(
                    {**_REQUEST, "profile": "toaster"})
                gate_phone = binding.profile_gate("phone")
                gate_toaster = binding.profile_gate(
                    "toaster")
                registry = binding.registry()
                passed = refused.get("ok") is False \
                    and refused.get("status") == "refused" \
                    and refused.get("error") == TOASTER_ERROR \
                    and fake.turns == [] \
                    and gate_phone["gate"]["affords"] is True \
                    and gate_phone["gate"]["draft_cap"] == 1 \
                    and gate_phone["gate"][
                        "thoughtforge_profile"] \
                    == TINY_PROFILE_ID \
                    and gate_toaster["gate"]["affords"] \
                    is False \
                    and gate_toaster["status"] \
                    == "cannot_afford" \
                    and registry["registry"] \
                    == FILED_REGISTRY \
                    and registry["registry"][
                        "afforded_profiles"] \
                    == ["phone", "pi", "laptop"] \
                    and binding_response_problems(
                        refused) == [] \
                    and binding_response_problems(
                        gate_phone) == []
            elif kind == "fake_consumer":
                consumer = _FakeEmberConsumer()
                request = consumer.request(
                    "What time does the ferry leave?",
                    episodes=[{"operator_input": "hello",
                               "ember_reply": "hi there"}])
                fake = _RecordingCognitionAPI()
                binding = EmberBinding(
                    cognition_api=fake, enabled=True)
                out = binding.ember_turn(request)
                registry = binding.registry()
                passed = consumer.config["enabled"] is False \
                    and consumer.config["provider"] \
                    == PROVIDER_KEY \
                    and consumer.config[
                        "expected_api_major"] == 1 \
                    and all(
                        profile_row_problems(r) == []
                        for r in PROFILE_ROWS) \
                    and profile_row_for(
                        "phone")["thoughtforge_profile"] \
                    == TINY_PROFILE_ID \
                    and profile_row_for(
                        "toaster")["affords"] is False \
                    and ember_request_problems(
                        request) == [] \
                    and out.get("ok") is True \
                    and out["reply"]["text"] \
                    == out["turn"]["text"] \
                    and out["reply"]["provider"] \
                    == consumer.config["provider"] \
                    and consumer.registry_providers(
                        registry["registry"]) \
                    == [PROVIDER_KEY] \
                    and binding_response_problems(out) == []
            elif kind == "doc_and_version":
                doc = _REPO_ROOT / EMBER_BINDING_DOC
                passed = doc.exists() \
                    and doc.read_text(encoding="utf-8") \
                    == render_ember_binding_markdown() \
                    and ember_binding_contract()[
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


# ── The readable render (the ember-binding doc, verbatim) ────────────────────

def render_ember_binding_markdown() -> str:
    """Render the ember-binding doc
    deterministically — the exact text filed as
    docs/architecture/EMBER_BINDING.md."""
    lines: list[str] = []
    lines.append("# Ember Binding — the Tiny Profile as "
                 "Ember's Cognition Layer Candidate "
                 "(TF-159)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/ember_binding.py` — the "
        "filed doc IS the render: regenerate with "
        "`python3 -m thoughtforge.ember_binding`. "
        "The checker (`ember_binding_problems()`) "
        "re-derives every claim live from the tree "
        "and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{EMBER_BINDING_FORMAT}`")
    lines.append(
        f"Contract version: `{API_VERSION}` (TF-034's "
        "`API_VERSION` — consumed, not reinvented)")
    lines.append(f"Provider key: `{PROVIDER_KEY}`")
    lines.append(
        f"Ember side: `{EMBER_PACKAGE}` "
        f"{EMBER_VERSION} — its E-024 candidate "
        "binding (read READ-ONLY from the Ember "
        "tree; no Ember code imported or copied)")
    lines.append(
        f"Tiny profile: `{TINY_PROFILE_ID}` "
        "(ThoughtForge's weakest shipped hardware "
        "profile — the budgets in section 3 are its "
        "budgets, proven first)")
    lines.append(
        f"Default enabled: {DEFAULT_ENABLED} — the "
        "binding is OFF by default; enabling is an "
        "explicit act (`EmberBinding(enabled=True)` "
        "or `binding.enable()`)")
    lines.append(f"Ember profiles: "
                 f"{len(PROFILE_ROWS)}; budget rows: "
                 f"{len(BUDGET_ROWS)}; calls: "
                 f"{len(CALL_ROWS)} "
                 f"({len(ROUTED_CALL_IDS)} routed, "
                 f"{len(NOT_ROUTED_CALL_IDS)} filed NOT "
                 f"routed); field rows: {len(FIELD_ROWS)}; "
                 f"rules: {len(RULE_ROWS)}")
    lines.append(f"Probes: {len(EMBER_BINDING_PROBES)}/"
                 f"{len(EMBER_BINDING_PROBES)}")
    lines.append("")
    lines.append("## 1. One binding")
    lines.append("")
    lines.append(
        "`EmberBinding` wraps TF-154's `CognitionAPI` "
        "and TF-155's `MemoryAPI` — each injectable "
        "directly, by factory, or via a `turn_callable` "
        "/ `memory_store` passthrough, and both built "
        "lazily: constructing the binding builds "
        "neither API. While disabled, every routed "
        "call returns a truthful refusal as data "
        "(`ok=False`, `status=\"refused\"`, the "
        "`error` naming the binding as disabled) and "
        "touches neither API. When enabled, a routed "
        "candidate turn routes through the "
        "CognitionAPI and memory calls through the "
        "MemoryAPI — called, never reimplemented. An "
        "enabled turn on the toaster profile returns "
        "the filed cannot-afford refusal as data and "
        "routes nothing. An invalid request raises "
        "`ValueError` naming the problem(s), judged "
        "by the pure validators before any API is "
        "built or called. Every response is a plain "
        "JSON-safe detached dict: `call`, `ok`, "
        "`status`, `provider_key`, "
        "`contract_version` (TF-034's `API_VERSION`), "
        "`error`, plus the call's payload (`turn` + "
        "`reply` for an ember turn, `gate` for the "
        "profile gate, `registry` for the registry, "
        "`health` for health, the TF-155 response "
        "verbatim in `memory` for memory calls). No "
        "Ember code is imported or copied anywhere: "
        "Ember's shapes are filed below as literal "
        "data read READ-ONLY from the Ember tree, and "
        "the binding performs no Ember act — it "
        "rewires no chat, runs no Funi turn, executes "
        "no tool, and holds no persona, Episode, Well "
        "or credential. A cognition layer candidate "
        "only, behind Ember's own E-024 surfaces.")
    lines.append("")
    lines.append("## 2. The Ember profiles — E-024's gate")
    lines.append("")
    lines.append("| Row | Ember profile | Ember side | "
                 "ThoughtForge profile | Affords | "
                 "Draft cap | Binding method | Enabled "
                 "by default | What it is |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in PROFILE_ROWS:
        affords = "yes" if row["affords"] else "no"
        enabled = "yes" if row["enabled_by_default"] else "no"
        method = row["binding_method"] or "—"
        pairing = row["thoughtforge_profile"] or "—"
        cap = row["draft_cap"]
        cap_text = str(cap) if cap is not None else "—"
        lines.append(
            f"| {row['id']} | {row['ember_profile']} | "
            f"{row['ember_side']} | {pairing} | "
            f"{affords} | {cap_text} | {method} | "
            f"{enabled} | {row['summary']} |")
    lines.append("")
    lines.append(
        "On the pairing: only Ember's `phone` profile "
        "is filed with a ThoughtForge profile pairing "
        "in this slice — `phone_low`, the floor "
        "TF-037 proved the whole core on. Ember's "
        "`pi` and `laptop` profiles are afforded by "
        "E-024 and routed under their filed draft "
        "caps (1 and 2), but no ThoughtForge budget "
        "pairing is filed or claimed for them here: "
        "phone_low first means the other pairings are "
        "later slices' proofs. The `toaster` — the "
        "smallest profile, Ember's namesake floor — "
        "cannot afford the layer at all (E-024's "
        "`PROFILE_CANNOT_AFFORD`), and this binding "
        "returns that refusal as data rather than "
        "routing a turn the machine cannot carry.")
    lines.append("")
    lines.append("## 3. The budgets — proven on phone_low first")
    lines.append("")
    lines.append(
        "Every value below is copied exactly from the "
        "`phone_low` profile's own definitions in "
        "this tree and cited to its source module; "
        "the checker re-derives each row live through "
        "those same sources (TF-037's `floor_budget()`, "
        "TF-056's memory budget, TF-033's latency "
        "budget and TF-079's cold-start budget over "
        "`hardware_profiles/phone_low.json`), so a "
        "drifted source or a drifted filing fails by "
        "name. These are the only budgets this binding "
        "files.")
    lines.append("")
    lines.append("| Budget | Category | Value | "
                 "Source | What it is |")
    lines.append("|---|---|---|---|---|")
    for row in BUDGET_ROWS:
        value = row["value"]
        value_text = (
            ", ".join(str(v) for v in value)
            if isinstance(value, list) else str(value))
        lines.append(
            f"| {row['id']} | {row['category']} | "
            f"{value_text} | {row['source']} | "
            f"{row['summary']} |")
    lines.append("")
    lines.append("## 4. The calls")
    lines.append("")
    lines.append("| Call | Surface | Ember side | "
                 "ThoughtForge route | Binding method | "
                 "Routed |")
    lines.append("|---|---|---|---|---|---|")
    for row in CALL_ROWS:
        routed = "yes" if row["routed"] else "no"
        method = row["binding_method"] or "—"
        lines.append(
            f"| {row['id']} | {row['surface']} | "
            f"{row['ember_side']} | "
            f"{row['thoughtforge_route']} | {method} | "
            f"{routed} |")
    lines.append("")
    lines.append("## 5. Field mapping — in (Ember -> turn-in)")
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
    lines.append("## 6. Field mapping — out (turn-out -> reply)")
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
    lines.append("## 7. The rules")
    lines.append("")
    for rule in RULE_ROWS:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## 8. Docket")
    lines.append("")
    lines.append(
        "TF-159 has **no share** of TF-016 and found "
        "no new defect in existing code, so this "
        "slice does not edit the docket and adds no "
        "entry. TF-016 totals stand unchanged (137 "
        "entries — 31 OPEN / 1 OPEN-ENVIRONMENT / "
        "101 FIXED / 4 CLOSED-BY-VERDICT), verified "
        "live against the docket header and filed "
        "in the open here.")
    lines.append("")
    lines.append("## 9. The probes")
    lines.append("")
    for probe in EMBER_BINDING_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 10. Filed limits")
    lines.append("")
    lines.append(
        "- Proof grade FIXTURE: the checker and "
        "probes run a fake Ember consumer shaped "
        "on the filed E-024 data. NO live Ember "
        "process was routed and NO Ember code was "
        "imported or copied; live end-to-end "
        "routing is NOT PROVEN — TF-170's Part IX "
        "gate owns the at-least-one-live proof, "
        "never claimed here.")
    lines.append(
        "- The budgets are proven as filed data "
        "re-derived live from the phone_low "
        "profile's own definitions, plus TF-037's "
        "floor proof that the core holds at that "
        "profile (a fixture model on a simulated "
        "profile, on the VM — TF-037's own filed "
        "grade): no phone hardware ran, no real "
        "small model served a turn here, and "
        "TF-079's filed cold-start numbers are "
        "ceilings for the shape, not a live "
        "model's measured TTFT.")
    lines.append(
        "- Candidate means candidate on both sides: "
        "Ember's chat still thinks with Funi "
        "(E-024 did not rewire it) and this "
        "binding files no rewire — wiring the "
        "candidate into Ember's chat turn is a "
        "later deliberate act on Ember's side.")
    lines.append(
        "- Only the phone -> phone_low pairing is "
        "budget-proven here. Pi and laptop turns "
        "route under their E-024 draft caps, but "
        "their ThoughtForge budget pairings are "
        "NOT PROVEN and not filed; the toaster "
        "cannot afford the layer at all.")
    lines.append(
        "- No credential ever enters the binding: "
        "E-024's binding involves no credential at "
        "all, the request validator refuses any "
        "credential field by name, and TF-160 owns "
        "credential custody on the ThoughtForge "
        "side.")
    lines.append(
        "- TF-160 owns credential custody (Kista); "
        "TF-161 owns binding fixtures in depth; "
        "TF-162 owns versioning enforcement in "
        "depth; TF-163 owns the local-only proof "
        "for bindings. TF-159 files the Ember "
        "binding shape, its off-by-default routing "
        "under E-024's contract, and the phone_low "
        "budget proof that E-024 named as this "
        "slice's burden.")
    lines.append(
        "- No canon data was touched (TF-017); no "
        "README.md anywhere was touched; no Docker "
        "was used; nothing under the Ember tree "
        "was edited.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_ember_binding_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / EMBER_BINDING_DOC
    doc.write_text(render_ember_binding_markdown(),
                   encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.ember_binding`."""
    doc = write_ember_binding_doc()
    print(f"ember binding doc regenerated: -> "
          f"{EMBER_BINDING_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def ember_binding_problems() -> list[str]:
    """Re-derive TF-159's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    if EMBER_BINDING_FORMAT != "thoughtforge-ember-binding/1":
        problems.append(
            f"the format is {EMBER_BINDING_FORMAT!r}, not "
            "thoughtforge-ember-binding/1")
    if DEFAULT_ENABLED is not False:
        problems.append("DEFAULT_ENABLED is not False")
    if PROVIDER_KEY != "thoughtforge" \
            or EMBER_PACKAGE != "ember-agent" \
            or EMBER_VERSION != "0.2.0":
        problems.append(
            f"the provider / package filing is "
            f"{PROVIDER_KEY!r} / {EMBER_PACKAGE!r} "
            f"{EMBER_VERSION!r}, not thoughtforge / "
            "ember-agent 0.2.0")
    if TINY_PROFILE_ID != "phone_low" \
            or FLOOR_PROFILE_ID != "phone_low":
        problems.append(
            f"the tiny profile filing is "
            f"{TINY_PROFILE_ID!r} / {FLOOR_PROFILE_ID!r}, "
            "not phone_low")
    if not DISABLED_ERROR or "disabled" not in DISABLED_ERROR:
        problems.append(
            "DISABLED_ERROR does not name the binding as "
            "disabled")
    if not TOASTER_ERROR \
            or "cannot afford" not in TOASTER_ERROR:
        problems.append(
            "TOASTER_ERROR does not name the toaster "
            "profile as unable to afford the layer")

    profile_ids = [r["id"] for r in PROFILE_ROWS]
    expected_profiles = [
        "profile_toaster", "profile_phone",
        "profile_pi", "profile_laptop"]
    if profile_ids != expected_profiles \
            or len(profile_ids) != len(set(profile_ids)):
        problems.append(
            f"the PROFILE_ROWS filing is not as filed: "
            f"{profile_ids}")
    for row in PROFILE_ROWS:
        problems.extend(profile_row_problems(row))
    expected_gate = {
        "toaster": (False, None, None),
        "phone": (True, 1, "phone_low"),
        "pi": (True, 1, None),
        "laptop": (True, 2, None),
    }
    for name, (affords, cap, pairing) in expected_gate.items():
        row = profile_row_for(name)
        if row is None or row["affords"] is not affords \
                or row["draft_cap"] != cap \
                or row["thoughtforge_profile"] != pairing:
            problems.append(
                f"the E-024 gate filing for {name} drifted: "
                f"{row}")
    if FILED_REGISTRY.get("provider") != PROVIDER_KEY \
            or FILED_REGISTRY.get("implemented") is not True \
            or FILED_REGISTRY.get("api_major") != 1 \
            or FILED_REGISTRY.get("afforded_profiles") \
            != ["phone", "pi", "laptop"]:
        problems.append(
            "FILED_REGISTRY is not E-024's filed "
            "registry (thoughtforge / implemented / "
            "major 1 / phone+pi+laptop)")
    if FILED_HEALTH.get("provider") != PROVIDER_KEY \
            or FILED_HEALTH.get("api_version") != API_VERSION \
            or FILED_HEALTH.get("healthy") is not True \
            or FILED_HEALTH.get("last_ok") is not None \
            or "no live model health is claimed" not in \
            str(FILED_HEALTH.get("detail")):
        problems.append("FILED_HEALTH is not as filed")

    # The budgets re-derive live from phone_low.
    budget_ids = [r["id"] for r in BUDGET_ROWS]
    if len(budget_ids) != len(set(budget_ids)):
        problems.append("the BUDGET_ROWS filing has duplicate ids")
    for row in BUDGET_ROWS:
        problems.extend(budget_row_problems(row))
    try:
        expected_budgets = expected_budget_values()
        if set(budget_ids) != set(expected_budgets):
            problems.append(
                f"the BUDGET_ROWS ids are not the "
                f"phone_low derivation's ids: filed "
                f"{sorted(set(budget_ids) ^ set(expected_budgets))}")
        for row in BUDGET_ROWS:
            want = expected_budgets.get(row["id"])
            if row["value"] != want:
                problems.append(
                    f"budget {row['id']}: filed "
                    f"{row['value']!r}, the live phone_low "
                    f"source files {want!r}")
        floor = floor_budget(TINY_PROFILE_ID)
        if floor.profile_id != TINY_PROFILE_ID \
                or floor.response_token_cap != 180:
            problems.append(
                f"the TF-037 floor no longer resolves "
                f"phone_low with a 180 response cap: "
                f"{floor.to_dict()}")
    except Exception as exc:
        problems.append(f"budget derivation failed: {exc}")

    call_ids = [r["id"] for r in CALL_ROWS]
    expected_calls = [
        "ember_turn", "profile_gate", "registry",
        "health_check", "memory_add", "memory_inspect",
        "memory_prune", "memory_export", "chat_rewire",
        "funi_model_turn", "tool_execution",
        "persona_custody", "episode_well_custody",
        "credential_custody"]
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
    for call_id in ("chat_rewire", "funi_model_turn",
                    "tool_execution", "persona_custody",
                    "episode_well_custody",
                    "credential_custody"):
        row = next(r for r in CALL_ROWS if r["id"] == call_id)
        if row["routed"]:
            problems.append(
                f"Ember-only proof: call {call_id} is "
                "routed — Ember's own acts must stay "
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
    if set(REPLY_FIELDS) != {
            "text", "provider", "api_version", "turn_id",
            "mode", "enforcement_passed", "citations",
            "token_count"}:
        problems.append(
            "the REPLY_FIELDS filing is not E-024's "
            "CognitionReply field set")

    rule_ids = [r["id"] for r in RULE_ROWS]
    if len(rule_ids) != len(set(rule_ids)) \
            or len(RULE_ROWS) != 9:
        problems.append(
            "the RULE_ROWS filing is not nine unique rules")
    for rule in RULE_ROWS:
        problems.extend(rule_row_problems(rule))
    probe_ids = [p["id"] for p in EMBER_BINDING_PROBES]
    if len(probe_ids) != len(set(probe_ids)) \
            or len(EMBER_BINDING_PROBES) != 7:
        problems.append(
            "the EMBER_BINDING_PROBES filing is not "
            "seven unique probes")

    # The contract is TF-034-versioned and its
    # surface is clean.
    contract = ember_binding_contract()
    if contract["contract_version"] != API_VERSION \
            or contract["default_enabled"] is not False \
            or contract["tiny_profile"] != TINY_PROFILE_ID:
        problems.append(
            "the filed contract is not TF-034-versioned, "
            "off by default and phone_low-filed")
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
            ("TF-157 samtal_binding", samtal_binding_problems),
            ("TF-158 hermes_binding", hermes_binding_problems)):
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

        binding = EmberBinding(
            cognition_factory=cog_factory,
            memory_factory=mem_factory)
        refused = binding.ember_turn(_REQUEST)
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
    # turn inside the phone_low response cap and a
    # temp-store memory add / find, through the
    # binding only; plus the toaster refusal.
    try:
        binding, _tmp = _knowledge_binding()
        out = binding.ember_turn(_REQUEST)
        problems.extend(
            f"round-trip turn: {p}"
            for p in binding_response_problems(out))
        cap = budget_value("generation.max_response_tokens")
        if out.get("ok") is not True \
                or out.get("turn", {}).get("mode") \
                != "knowledge_only" \
                or out.get("reply", {}).get("text") \
                != out.get("turn", {}).get("text") \
                or out.get("reply", {}).get("provider") \
                != PROVIDER_KEY \
                or out.get("turn", {}).get(
                    "token_count", 10 ** 9) > cap:
            problems.append(
                f"round-trip turn is not a "
                f"knowledge_only turn inside the "
                f"phone_low cap with the filed reply "
                f"shape: {out}")
        toaster = binding.ember_turn(
            {**_REQUEST, "profile": "toaster"})
        if toaster.get("ok") is not False \
                or toaster.get("error") != TOASTER_ERROR:
            problems.append(
                f"the toaster turn did not refuse as "
                f"filed: {toaster}")
        added = binding.memory_add(
            "user_fact", summary="ferry at 07:30",
            record_id="fct_tf159_check")
        if added.get("ok") is not True \
                or added.get("memory", {}).get(
                    "record", {}).get("source") \
                != "manual_entry":
            problems.append(
                f"round-trip memory add is not a stamped "
                f"add: {added}")
        found = binding.memory_inspect(
            record_id="fct_tf159_check")
        if found.get("memory", {}).get("status") != "found":
            problems.append(
                f"round-trip memory find: {found}")
    except Exception as exc:
        problems.append(f"round-trip failed: {exc}")

    # The probes all behave.
    probes = run_ember_binding_probes()
    if len(probes) != len(EMBER_BINDING_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the ember binding probes do not all pass: "
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
            "FIXED; TF-159 must not move them")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / EMBER_BINDING_DOC
    if not doc.exists():
        problems.append(
            f"the ember binding doc {EMBER_BINDING_DOC} "
            "is missing")
    elif doc.read_text(encoding="utf-8") \
            != render_ember_binding_markdown():
        problems.append(
            f"the filed doc {EMBER_BINDING_DOC} is not "
            "the render verbatim (regenerate with "
            "python3 -m thoughtforge.ember_binding)")
    return problems


def ember_binding_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for
    consumers."""
    probes = run_ember_binding_probes()
    return {
        "format": EMBER_BINDING_FORMAT,
        "doc": EMBER_BINDING_DOC,
        "contract": ember_binding_contract(),
        "measured": {
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "BUDGET_CATEGORIES",
    "BUDGET_IDS",
    "BUDGET_ROWS",
    "CALL_IDS",
    "CALL_ROWS",
    "DEFAULT_EMBER_PROFILE",
    "DEFAULT_ENABLED",
    "DISABLED_ERROR",
    "EMBER_BINDING_DOC",
    "EMBER_BINDING_FORMAT",
    "EMBER_BINDING_PROBES",
    "EMBER_BINDING_RULES",
    "EMBER_PACKAGE",
    "EMBER_PROFILES",
    "EMBER_VERSION",
    "EPISODE_FIELDS",
    "FIELD_IDS",
    "FIELD_ROWS",
    "FILED_HEALTH",
    "FILED_REGISTRY",
    "NOT_ROUTED_CALL_IDS",
    "PROFILE_ROW_IDS",
    "PROFILE_ROWS",
    "PROVIDER_KEY",
    "REPLY_FIELDS",
    "REQUEST_FIELDS",
    "ROUTED_CALL_IDS",
    "RULE_IDS",
    "RULE_ROWS",
    "TINY_PROFILE_ID",
    "TOASTER_ERROR",
    "EmberBinding",
    "assemble_history",
    "chat_history_for",
    "binding_response_problems",
    "binding_rule_problems",
    "budget_row_problems",
    "budget_value",
    "call_row_problems",
    "ember_binding_as_data",
    "ember_binding_contract",
    "ember_binding_problems",
    "ember_request_problems",
    "expected_budget_values",
    "field_row_problems",
    "field_rows_for",
    "main",
    "profile_row_for",
    "profile_row_problems",
    "render_ember_binding_markdown",
    "reply_for",
    "resolved_num_drafts",
    "rule_row_problems",
    "run_ember_binding_probes",
    "write_ember_binding_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
