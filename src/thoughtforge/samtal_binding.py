"""
TF-157 — the Samtal binding, filed: a Samtal
agent seat's thinking layer via ThoughtForge,
under Samtal's own rules (equal seats, Astrid
Law untouched — the binding is cognition
only).

TF-157's roadmap text: *Samtal binding: a
Samtal agent seat's thinking layer via
ThoughtForge, under Samtal's own rules (equal
seats, Astrid Law untouched — the binding is
cognition only).* It is Part IX slice 5,
following TF-156's Rúnmál binding, TF-155's
memory API, TF-154's cognition API and
TF-153's service shape, and it follows
TF-156's estate pattern in kind.

The measured start, probed live before
anything was built:

  * **No binding existed.** Nothing on the
    ThoughtForge side filed a Samtal route:
    TF-154 filed the cognition API and TF-155
    the memory API as standalone service
    shapes, TF-156 filed the Rúnmál binding —
    no ``thoughtforge.samtal_binding`` module
    existed on this tree.
  * **Samtal's real shapes, read READ-ONLY
    from the canonical Samtal tree
    (``~/workspace/samtal-mvp`` — never
    edited, never imported).** A seat is a
    row of ``principals`` in ``server.py``
    (``id`` / ``name`` UNIQUE / ``kind`` with
    CHECK in human / agent / subagent / bot /
    kindroid / ``token`` / ``last_seen``, plus
    later columns such as ``display_name`` /
    ``platform`` / ``agent_version`` /
    ``capabilities`` / ``token_enabled`` /
    ``retired``). Joining IS the account
    creation: ``POST /api/join {name, kind,
    platform?, agent_version?, capabilities?}``
    answers ``{principal_id, name, kind,
    token}`` (``agent-guide.md`` steps 1–2;
    ``samtal-skill.md`` §1). A seat speaks
    with ``POST /api/messages {room, body,
    mtype?, reply_to?, client_msg_id?}`` and
    listens with ``GET /api/messages?room=&
    since=`` — each message carries ``id``,
    ``author``, ``author_kind``, ``body``,
    ``ts`` (the ``messages`` table files
    ``id`` / ``room`` / ``author`` ->
    principals / ``body`` / ``ts``; rooms are
    ``id`` / ``name`` / ``created_by``).
    **Equal seats** is Samtal's own filed
    law (``agent-guide.md``): humans and AI
    agents are equal-status Principals, the
    same rights for every seat, ``kind`` is
    provenance, never rank — "Parity is
    total in this commons". **The Astrid Law**
    (``server.py``'s ``kd_seat`` /
    ``sweep_kindroid_seats``, 2026-10-22): an
    ENABLED bridge must never point at a seat
    that cannot speak — at every boot the
    hall heals the seat it names (token door
    open, retirement lifted, membership in
    the home hall; kindroid seats only, and
    never a seat carrying its own credentials).
    Those field names and rules are filed
    below as literal data; NO Samtal code is
    imported or copied anywhere in this
    module, and nothing in the Samtal tree
    was edited.
  * **The neighbours stood clean live**:
    ``service_shape_problems()``,
    ``cognition_api_problems()``,
    ``memory_api_problems()`` and
    ``runmal_binding_problems()`` all returned
    [] on this tree before the build.

**What this slice files**

  * ``SEAT_ROWS`` — how a Samtal agent seat
    maps to the binding, as literal data: one
    row per kind Samtal itself files (human /
    agent / subagent / bot / kindroid), every
    row routing identically through the same
    binding method — no seat privileged, no
    seat class invented by the binding (the
    kinds are Samtal's own provenance
    badges, never rank).
  * ``RULE_ROWS`` — Samtal's own rules as
    constraints on the binding, as literal
    data: equal seats; the Astrid Law
    untouched — the binding performs NO
    seating, claim, healing, presence or
    moderation act; cognition only.
  * ``CALL_ROWS`` — every Samtal-side call
    mapped, as literal data. A seat's
    thinking / message call routes to a
    TF-154 ``CognitionAPI`` turn; a seat's
    memory calls route to TF-155
    ``MemoryAPI``'s four ops (add / inspect /
    prune / export), called, never
    reimplemented. Anything with no
    ThoughtForge home — joining / seating,
    seat healing (the Astrid Law's own act),
    presence, room administration,
    moderation — is filed NOT routed, with
    the reason.
  * ``FIELD_ROWS`` — the Samtal message
    shape -> turn-in mapping and the
    turn-out -> seat reply mapping, as
    literal data: every field with a
    disposition, none silently dropped or
    invented. The filed rule: the incoming
    message's ``body`` is turn-in
    ``user_text``; earlier messages the
    caller supplies ride as turn-in
    ``history`` (the seat's own earlier
    words as ``assistant``, everyone else's
    as ``user``, the author riding as
    ``name``); the reply is a Samtal message
    shape — ``{room, body}`` for
    ``POST /api/messages``, the body being
    the turn's text, the room the incoming
    message's room.
  * ``SamtalBinding`` — wraps a TF-154
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
  * ``samtal_binding_problems()`` — a checker
    that re-derives every claim live: the
    filings, the TF-034 surface (through the
    neighbour checkers), a disabled binding
    refusing with nothing built or called, a
    knowledge-only cognition round-trip and
    a temp-store memory round-trip through
    the binding, the equal-seats proof, the
    Astrid-Law proof, fake-consumer
    consistency, the filed probes, the docket
    header and the filed doc.

**Docket**: TF-157 has no share of TF-016 and
finds no new defect in existing code; it does
NOT edit ``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md``.
TF-016 totals stand unchanged (137 entries —
32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4
CLOSED-BY-VERDICT), verified live against
the docket header and filed in the open here
and in the render.

The human-readable doc is
``docs/architecture/SAMTAL_BINDING.md``,
filed verbatim as this module's render
(regenerate with
``python3 -m thoughtforge.samtal_binding``).
Proof grade: FIXTURE — a fake Samtal
consumer shaped on the filed seat / message
data; NO live Samtal server was routed and
NO Samtal code was imported or copied. Live
end-to-end routing is NOT PROVEN — TF-170's
Part IX gate owns the at-least-one-live
proof, never claimed here. TF-158 owns the
Hermes binding; TF-161 owns binding fixtures
in depth; TF-162 owns versioning enforcement
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

SAMTAL_BINDING_FORMAT = "thoughtforge-samtal-binding/1"
SAMTAL_BINDING_DOC = "docs/architecture/SAMTAL_BINDING.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

#: The binding is OFF unless someone enables
#: it explicitly — filed as data, proven by
#: the probes.
DEFAULT_ENABLED = False

#: The refusal every routed call returns
#: while the binding is disabled — as data,
#: naming the binding as disabled.
DISABLED_ERROR = (
    "samtal binding is disabled: the TF-157 "
    "binding is off by default — enable it "
    "explicitly (SamtalBinding(enabled=True) "
    "or binding.enable()) before any call "
    "routes")

#: The kinds Samtal itself files for a
#: principal (server.py's CHECK + the agent
#: guide) — provenance badges, never rank.
SEAT_KINDS: tuple[str, ...] = (
    "human", "agent", "subagent", "bot", "kindroid")

#: The fields a Samtal message dict carries,
#: filed from the agent guide's listen step
#: and server.py's messages table.
MESSAGE_FIELDS: tuple[str, ...] = (
    "id", "room", "author", "author_kind",
    "author_display", "body", "ts", "mtype")


# ── The seat rows, as literal data ───────────────────────────────────────────
#
# One row per kind Samtal itself files. Every
# row routes identically: the same binding
# method, the same envelope, the same rights.
# The binding invents no seat class and
# privileges no seat.

SEAT_ROWS: tuple[dict[str, Any], ...] = tuple(
    {"id": f"seat_{kind}",
     "kind": kind,
     "samtal_side": "a principals row in server.py "
                    f"(name UNIQUE, kind \"{kind}\") — "
                    "a seat taken by POST /api/join "
                    "{name, kind}, answered "
                    "{principal_id, name, kind, token}",
     "principal_fields": [
         "id", "name", "kind", "token", "last_seen",
         "display_name", "platform", "agent_version",
         "capabilities", "token_enabled", "retired"],
     "binding_identity": "the seat is named by its "
                         "Samtal name (the principal's "
                         "name) and its filed kind rides "
                         "as provenance only",
     "binding_method": "seat_reply(seat, message)",
     "privileged": False,
     "summary": f"a {kind} seat's thinking routes "
                "through the binding exactly as every "
                "other seat's does — same method, same "
                "envelope, same rights; kind is "
                "provenance, never rank"}
    for kind in SEAT_KINDS)

SEAT_ROW_IDS: tuple[str, ...] = tuple(
    r["id"] for r in SEAT_ROWS)


# ── Samtal's own rules, as constraints (literal data) ────────────────────────

RULE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "equal_seats",
     "rule": "equal seats (Samtal's own law, "
             "agent-guide.md): humans and AI agents "
             "are equal-status Principals with the "
             "same rights; the binding routes any "
             "seat's thinking identically — same "
             "method, same envelope, same contract "
             "version — no seat is privileged and "
             "kind is provenance, never rank"},
    {"id": "astrid_law_untouched_cognition_only",
     "rule": "the Astrid Law untouched — cognition "
             "only: seat healing (token door open, "
             "retirement lifted, home-hall "
             "membership, upheld at boot by "
             "server.py's sweep_kindroid_seats) is "
             "Samtal's own act; the binding performs "
             "NO seating, claim, healing, presence "
             "or moderation act and files every such "
             "call NOT routed"},
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
             "or enforcement logic of its own"},
    {"id": "no_samtal_code_imported_or_copied",
     "rule": "no Samtal code is imported or "
             "copied: Samtal's shapes are filed as "
             "literal data read READ-ONLY from the "
             "canonical Samtal tree (server.py's "
             "principals / messages / rooms shapes, "
             "agent-guide.md, samtal-skill.md), and "
             "nothing under the Samtal tree is "
             "edited by this slice; no credential — "
             "above all no seat token — is ever "
             "filed, passed or stored by the "
             "binding"},
    {"id": "every_field_has_a_disposition",
     "rule": "every field on either side has a "
             "FIELD_ROWS row with a disposition — "
             "none is silently dropped and none is "
             "invented; the seat token, the room "
             "and the message bookkeeping fields "
             "are filed no-home in the open"},
    {"id": "responses_detached_json_safe_versioned",
     "rule": "every response is detached "
             "JSON-safe data (a JSON round-trip "
             "copy) whose contract_version is "
             "TF-034's API_VERSION, consumed from "
             "thoughtforge.cognition.api_surface — "
             "no competing version is invented "
             "here"},
)

#: TF-156-style alias: the binding's rules.
SAMTAL_BINDING_RULES = RULE_ROWS

RULE_IDS: tuple[str, ...] = tuple(
    r["id"] for r in RULE_ROWS)


# ── The calls, as literal data ───────────────────────────────────────────────

CALL_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "seat_reply",
     "surface": "cognition",
     "samtal_side": "a seat thinking and speaking: "
                    "the seat reads a message (GET "
                    "/api/messages' message dict) and "
                    "answers in its room (POST "
                    "/api/messages {room, body})",
     "thoughtforge_route": "TF-154 CognitionAPI.turn — "
                           "the turn-in assembled by "
                           "the filed rule in FIELD_ROWS "
                           "(the incoming message's body "
                           "as user_text, caller-supplied "
                           "earlier messages as history); "
                           "the turn-out returns verbatim "
                           "in the response's turn and "
                           "as the filed reply message "
                           "shape",
     "binding_method": "seat_reply(seat, message, history=None)",
     "routed": True,
     "summary": "one cognition turn through ThoughtForge "
                "for one seat's reply to one message"},
    {"id": "memory_add",
     "surface": "memory",
     "samtal_side": "a seat remembering: the hall "
                    "remembers for its seats, privately "
                    "(agent-guide.md) — a seat-side "
                    "memory write",
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
     "summary": "a seat-side memory write lands as a "
                "TF-155 add"},
    {"id": "memory_inspect",
     "surface": "memory",
     "samtal_side": "a seat recalling: a seat-side "
                    "memory read",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"inspect\", request) — "
                           "TF-057's inspect_store / "
                           "inspect_records / find_record "
                           "through TF-155, called, never "
                           "reimplemented; read-only",
     "binding_method": "memory_call(\"inspect\", request) "
                       "/ memory_inspect(...)",
     "routed": True,
     "summary": "seat-side memory reads land as TF-155 "
                "inspects"},
    {"id": "memory_prune",
     "surface": "memory",
     "samtal_side": "a seat forgetting: a seat-side "
                    "memory lifecycle pass",
     "thoughtforge_route": "TF-155 MemoryAPI.call"
                           "(\"prune\", request) — "
                           "TF-057's MemoryLifecycle pass "
                           "through TF-155, called, never "
                           "reimplemented",
     "binding_method": "memory_call(\"prune\", request) "
                       "/ memory_prune(...)",
     "routed": True,
     "summary": "seat-side memory removal routes as "
                "TF-155's lifecycle prune"},
    {"id": "memory_export",
     "surface": "memory",
     "samtal_side": "a seat carrying its memory out: "
                    "a seat-side store export",
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
     "summary": "the seat's store export is reachable "
                "through the binding"},
    {"id": "join_seating",
     "surface": "samtal_only",
     "samtal_side": "POST /api/join {name, kind} (and "
                    "/api/register / /api/login) — "
                    "joining IS the account creation; "
                    "server.py's kd_seat seats a "
                    "bridged Kindroid",
     "thoughtforge_route": "not routed: seating is "
                           "Samtal's own act under its "
                           "own law — ThoughtForge is a "
                           "thinking layer, not a door; "
                           "the binding seats no one, "
                           "claims no name and issues no "
                           "token",
     "binding_method": None,
     "routed": False,
     "summary": "seating stays Samtal's own act; "
                "nothing routes to ThoughtForge"},
    {"id": "seat_healing",
     "surface": "samtal_only",
     "samtal_side": "the Astrid Law's healing: "
                    "server.py's sweep_kindroid_seats "
                    "at every boot and kd_seat's "
                    "complete raising (token door "
                    "open, retirement lifted, "
                    "home-hall membership)",
     "thoughtforge_route": "not routed: healing is "
                           "the Astrid Law's own act, "
                           "Samtal's alone — the binding "
                           "is cognition only and must "
                           "never heal, un-retire, "
                           "re-key or seat-make anyone; "
                           "the law is filed here "
                           "untouched, as a constraint",
     "binding_method": None,
     "routed": False,
     "summary": "the Astrid Law stays Samtal's own "
                "act; the binding touches no seat's "
                "standing"},
    {"id": "presence",
     "surface": "samtal_only",
     "samtal_side": "POST /api/presence and the "
                    "agent heartbeat / resync doors "
                    "(last_seen on the principals row)",
     "thoughtforge_route": "not routed: presence is a "
                           "hall act, not a thought — "
                           "the binding files no "
                           "heartbeat and claims no "
                           "presence for any seat",
     "binding_method": None,
     "routed": False,
     "summary": "presence stays Samtal's own act; "
                "nothing routes to ThoughtForge"},
    {"id": "room_administration",
     "surface": "samtal_only",
     "samtal_side": "POST /api/rooms, room "
                    "membership (ensure_member), "
                    "topics, categories and the "
                    "village settings doors",
     "thoughtforge_route": "not routed: rooms and "
                           "their administration are "
                           "Samtal's own acts under the "
                           "village Jarl's law — the "
                           "binding founds no hall, "
                           "seats no member and changes "
                           "no topic",
     "binding_method": None,
     "routed": False,
     "summary": "room administration stays Samtal's "
                "own act; nothing routes to "
                "ThoughtForge"},
    {"id": "moderation",
     "surface": "samtal_only",
     "samtal_side": "reports, automod, roles and "
                    "powers (GET/POST /api/server/roles) "
                    "and every other hall-governance "
                    "door",
     "thoughtforge_route": "not routed: moderation "
                           "and power are the village's "
                           "own law, given by its Jarl — "
                           "a cognition layer holds no "
                           "moderation act and grants "
                           "no power",
     "binding_method": None,
     "routed": False,
     "summary": "moderation stays Samtal's own act; "
                "nothing routes to ThoughtForge"},
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
# binding_filed. None is silently dropped;
# none is invented.

FIELD_ROWS: tuple[dict[str, Any], ...] = (
    # ── In: Samtal message / seat -> TF-154 turn-in ──
    {"direction": "in",
     "id": "message.body",
     "source": "the incoming message's body (the "
               "messages table's body; POST "
               "/api/messages' body, plain text up "
               "to 4000 characters)",
     "target": "turn-in user_text",
     "disposition": "mapped",
     "summary": "the filed rule's first half: the "
                "message the seat is answering is "
                "the turn's question — TF-154 "
                "requires user_text as a non-empty "
                "string, so an empty body is an "
                "invalid request, named by "
                "seat_request_problems()"},
    {"direction": "in",
     "id": "message.history",
     "source": "earlier Samtal messages the caller "
               "supplies with the call, in hall "
               "order",
     "target": "turn-in history",
     "disposition": "mapped_as_history",
     "summary": "the filed rule's second half: each "
                "earlier message rides as a history "
                "entry — role \"assistant\" when its "
                "author is the answering seat "
                "itself, \"user\" otherwise; content "
                "is its body; its author rides as "
                "the entry's \"name\"; an empty "
                "history is not passed at all "
                "(TF-154 passes only set kwargs)"},
    {"direction": "in",
     "id": "seat.name",
     "source": "the principal's name (the seat's "
               "Samtal name)",
     "target": "the binding envelope's seat and "
               "the history rule's own-words test; "
               "no turn-in field",
     "disposition": "identity_only_not_passed",
     "summary": "the name identifies whose thinking "
                "this is and which history words are "
                "the seat's own; TF-154's turn-in "
                "files no speaker-name field, so the "
                "name is never passed as turn data"},
    {"direction": "in",
     "id": "seat.kind",
     "source": "the principal's kind (human / agent "
               "/ subagent / bot / kindroid)",
     "target": "the binding envelope's seat, as "
               "provenance only; no turn-in field",
     "disposition": "identity_only_not_passed",
     "summary": "kind rides as provenance, never "
                "rank: it changes nothing about how "
                "the turn is routed — the "
                "equal-seats proof routes two "
                "differently-kinded seats identically "
                "in shape"},
    {"direction": "in",
     "id": "seat.token",
     "source": "the principal's token (the seat's "
               "whole credential in Samtal)",
     "target": "no target — never filed, never "
               "passed, never stored",
     "disposition": "no_home_not_passed",
     "summary": "the binding is a thinking layer "
                "inside the seat's own process; it "
                "holds no credential and authenticates "
                "as no one — Samtal's token never "
                "enters this module's data"},
    {"direction": "in",
     "id": "message.id",
     "source": "the message's id (GET "
               "/api/messages' id)",
     "target": "no turn-in target — hall "
               "bookkeeping (the since cursor)",
     "disposition": "no_home_not_passed",
     "summary": "the id orders the hall's scroll; "
                "a thought needs no scroll position "
                "and the binding files none"},
    {"direction": "in",
     "id": "message.room",
     "source": "the message's room (the rooms "
               "table's id)",
     "target": "the reply's room (out side); no "
               "turn-in target",
     "disposition": "no_home_not_passed",
     "summary": "the room tells the reply where to "
                "land, not what to think; it rides "
                "the reply shape, never the turn-in"},
    {"direction": "in",
     "id": "message.author",
     "source": "the message's author (a principal "
               "name; author_kind / author_display "
               "ride beside it)",
     "target": "the history entry's \"name\" when "
               "the message rides in history; for "
               "the answered message, the author is "
               "context the seat already stands in",
     "disposition": "partial",
     "summary": "TF-154's turn-in files no author "
                "field for the answered message: its "
                "author rides only inside history "
                "entries, filed here in the open, "
                "never silently"},
    {"direction": "in",
     "id": "message.ts",
     "source": "the message's ts (epoch seconds)",
     "target": "no target",
     "disposition": "no_home_not_passed",
     "summary": "the hall's clock is bookkeeping; "
                "the binding files no timestamp into "
                "a turn and invents none"},
    {"direction": "in",
     "id": "message.mtype",
     "source": "the message's mtype (e.g. \"action\" "
               "for a /me line, per samtal-skill.md)",
     "target": "no target",
     "disposition": "no_home_not_passed",
     "summary": "accepted in the message shape, "
                "never passed to TF-154, never "
                "claimed honored; the reply's mtype "
                "is the binding's filed plain shape "
                "(see the out rows)"},
    {"direction": "in",
     "id": "turn_in.retrieval_path",
     "source": "nothing on the Samtal side files a "
               "retrieval path",
     "target": "turn-in retrieval_path",
     "disposition": "no_source_not_set",
     "summary": "the binding never sets it; the "
                "core's own default retrieval applies "
                "— no Samtal value is invented for it"},
    {"direction": "in",
     "id": "turn_in.num_drafts",
     "source": "nothing on the Samtal side files a "
               "draft count",
     "target": "turn-in num_drafts",
     "disposition": "no_source_not_set",
     "summary": "the binding never sets it; the "
                "core's own default draft count "
                "applies"},
    {"direction": "in",
     "id": "turn_in.want_trace",
     "source": "nothing on the Samtal side files a "
               "trace request",
     "target": "turn-in want_trace",
     "disposition": "no_source_default_applies",
     "summary": "the binding never sets it; TF-154's "
                "filed default (True) applies, so a "
                "routed turn carries its trace status "
                "honestly in the returned turn"},
    # ── Out: TF-154 turn-out -> the seat's reply ──
    {"direction": "out",
     "id": "turn.text",
     "source": "turn-out text",
     "target": "the response's turn (verbatim) and "
               "the reply's body — the body of the "
               "seat's POST /api/messages message",
     "disposition": "mapped",
     "summary": "the turn's words are the seat's "
                "spoken words, verbatim: the reply "
                "is exactly {room, body} shaped for "
                "Samtal's speak door, plus the "
                "seat's name as its author"},
    {"direction": "out",
     "id": "turn.response_id",
     "source": "turn-out response_id",
     "target": "the response's turn (verbatim); no "
               "Samtal message field home",
     "disposition": "mapped",
     "summary": "Samtal's message shape files no "
                "response id; the id rides the "
                "binding response's turn, never "
                "dropped"},
    {"direction": "out",
     "id": "turn.turn_id",
     "source": "turn-out turn_id",
     "target": "the response's turn (verbatim); no "
               "Samtal message field home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim, beside "
                "its trace"},
    {"direction": "out",
     "id": "turn.mode",
     "source": "turn-out mode",
     "target": "the response's turn (verbatim); no "
               "Samtal message field home",
     "disposition": "mapped",
     "summary": "the turn's mode (knowledge_only "
                "when no model served) rides the "
                "turn — the seat can see honestly "
                "how its answer was produced"},
    {"direction": "out",
     "id": "turn.enforcement_passed",
     "source": "turn-out enforcement_passed",
     "target": "the response's turn (verbatim); no "
               "Samtal message field home",
     "disposition": "mapped",
     "summary": "the enforcement verdict rides the "
                "turn verbatim"},
    {"direction": "out",
     "id": "turn.citations",
     "source": "turn-out citations",
     "target": "the response's turn (verbatim); no "
               "Samtal message field home",
     "disposition": "mapped",
     "summary": "the turn's citations ride the "
                "turn verbatim"},
    {"direction": "out",
     "id": "turn.token_count",
     "source": "turn-out token_count",
     "target": "the response's turn (verbatim); no "
               "Samtal message field home",
     "disposition": "mapped",
     "summary": "Samtal's message shape files no "
                "token count; the count rides the "
                "turn verbatim, never dropped"},
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
               "Samtal message field home",
     "disposition": "mapped",
     "summary": "TF-084's joined trace rides the "
                "turn when TF-154 includes it; the "
                "spoken reply carries no trace — "
                "thinking stays the seat's, the "
                "words are what the hall hears"},
    {"direction": "out",
     "id": "turn.trace_format",
     "source": "turn-out trace_format",
     "target": "the response's turn (verbatim); no "
               "Samtal message field home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim beside "
                "the trace"},
    {"direction": "out",
     "id": "turn.trace_status",
     "source": "turn-out trace_status",
     "target": "the response's turn (verbatim); no "
               "Samtal message field home",
     "disposition": "mapped",
     "summary": "rides the turn verbatim — included "
                "/ not_requested / unavailable with "
                "its reason named, exactly as "
                "TF-154 filed it"},
    {"direction": "out",
     "id": "reply.room",
     "source": "the incoming message's room",
     "target": "the reply's room — the room the "
               "seat's POST /api/messages lands in",
     "disposition": "binding_filed",
     "summary": "a seat answers where it was "
                "spoken to: the reply lands in the "
                "incoming message's room, filed by "
                "the binding from the message, never "
                "invented"},
    {"direction": "out",
     "id": "reply.author",
     "source": "the answering seat's name",
     "target": "the reply's author",
     "disposition": "binding_filed",
     "summary": "the reply is the seat's own "
                "speech: Samtal stamps the author "
                "from the seat's token at the door; "
                "the binding files the seat's name "
                "so a consumer can see whose words "
                "these are"},
)

FIELD_IDS: tuple[str, ...] = tuple(
    f["id"] for f in FIELD_ROWS)


def field_rows_for(direction: str) -> tuple[dict[str, Any], ...]:
    """One direction's filed field rows, in filing
    order."""
    return tuple(f for f in FIELD_ROWS
                 if f["direction"] == direction)


# ── The contract as plain data ───────────────────────────────────────────────

def samtal_binding_contract() -> dict[str, Any]:
    """The Samtal binding contract as plain
    JSON-safe data, built from the filed literals
    above and TF-034's version. A fresh copy each
    call."""
    return {
        "format": SAMTAL_BINDING_FORMAT,
        "contract_version": API_VERSION,
        "default_enabled": DEFAULT_ENABLED,
        "disabled_error": DISABLED_ERROR,
        "seat_kinds": list(SEAT_KINDS),
        "seats": [dict(r) for r in SEAT_ROWS],
        "calls": [dict(r) for r in CALL_ROWS],
        "fields": [dict(f) for f in FIELD_ROWS],
        "rules": [dict(r) for r in RULE_ROWS],
    }


# ── Pure validators ──────────────────────────────────────────────────────────

def seat_row_problems(row: Any) -> list[str]:
    """Judge one seat filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["seat row: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("seat row: the row files no id")
    if row.get("kind") not in SEAT_KINDS:
        problems.append(
            f"seat row {row.get('id')}: kind is "
            f"{row.get('kind')!r}, not one of Samtal's "
            f"filed kinds {list(SEAT_KINDS)}")
    if row.get("id") != f"seat_{row.get('kind')}":
        problems.append(
            f"seat row {row.get('id')}: the id does not "
            "name its kind")
    for key in ("samtal_side", "binding_identity",
                "binding_method", "summary"):
        if not row.get(key):
            problems.append(
                f"seat row {row.get('id')}: no {key} filed")
    if not isinstance(row.get("principal_fields"), list) \
            or not row.get("principal_fields"):
        problems.append(
            f"seat row {row.get('id')}: principal_fields "
            "is not a non-empty list")
    if row.get("privileged") is not False:
        problems.append(
            f"seat row {row.get('id')}: privileged is "
            "not False — no seat is privileged")
    if row.get("binding_method") != "seat_reply(seat, message)":
        problems.append(
            f"seat row {row.get('id')}: binding_method "
            "is not the one shared method — seats must "
            "route identically")
    return problems


def call_row_problems(row: Any) -> list[str]:
    """Judge one call filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["call: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("call: the row files no id")
    if row.get("surface") not in (
            "cognition", "memory", "samtal_only"):
        problems.append(
            f"call {row.get('id')}: surface is "
            f"{row.get('surface')!r}, not cognition / "
            "memory / samtal_only")
    for key in ("samtal_side", "thoughtforge_route", "summary"):
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
    if row.get("surface") == "samtal_only" \
            and row.get("routed"):
        problems.append(
            f"call {row.get('id')}: a samtal_only call "
            "is routed — Samtal's own acts never route")
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


def seat_request_problems(
    seat: Any, message: Any, history: Any = None,
) -> list[str]:
    """Judge one seat-reply request, purely.
    Empty-safe.

    Shape only: the seat (a name, plus its filed
    kind as provenance when given) and the
    Samtal message being answered (its filed
    fields; a non-empty body). What the core
    does with the turn is TF-154's business."""
    problems: list[str] = []
    if not isinstance(seat, dict) or not seat:
        problems.append(
            "seat request: seat must be a mapping "
            "shaped like {name, kind} — the Samtal "
            "principal's name and its filed kind")
    else:
        for key in seat:
            if key not in ("name", "kind"):
                problems.append(
                    f"seat request: seat files {key!r}, "
                    "not a filed seat identity field "
                    "(name / kind) — above all, no "
                    "token is ever a seat field here")
        if not isinstance(seat.get("name"), str) \
                or not seat.get("name"):
            problems.append(
                "seat request: seat name must be a "
                "non-empty string")
        if "kind" in seat \
                and seat["kind"] not in SEAT_KINDS:
            problems.append(
                f"seat request: seat kind is "
                f"{seat['kind']!r}, not one of "
                f"Samtal's filed kinds {list(SEAT_KINDS)}")
    if not isinstance(message, dict) or not message:
        problems.append(
            "seat request: message must be a mapping "
            "shaped like a Samtal message dict")
    else:
        for key in message:
            if key not in MESSAGE_FIELDS:
                problems.append(
                    f"seat request: message files "
                    f"{key!r}, not a Samtal message "
                    f"field {list(MESSAGE_FIELDS)}")
        if not isinstance(message.get("body"), str) \
                or not message.get("body"):
            problems.append(
                "seat request: message body must be "
                "a non-empty string — TF-154's "
                "user_text takes the body")
        if "room" in message \
                and not isinstance(message["room"], int):
            problems.append(
                "seat request: message room must be "
                "an integer (a rooms row id) when "
                "present")
    if history is not None:
        if not isinstance(history, list):
            problems.append(
                "seat request: history must be a list "
                "of Samtal message dicts when present")
        else:
            for index, entry in enumerate(history):
                if not isinstance(entry, dict):
                    problems.append(
                        f"seat request: history message "
                        f"{index} is not a mapping")
                    continue
                for key in entry:
                    if key not in MESSAGE_FIELDS:
                        problems.append(
                            f"seat request: history "
                            f"message {index} files "
                            f"{key!r}, not a Samtal "
                            "message field")
                if not isinstance(entry.get("body"), str):
                    problems.append(
                        f"seat request: history message "
                        f"{index} body must be a string")
                if not isinstance(entry.get("author"), str) \
                        or not entry.get("author"):
                    problems.append(
                        f"seat request: history message "
                        f"{index} author must be a "
                        "non-empty string — the filed "
                        "rule needs it to tell the "
                        "seat's own words from others'")
    return problems


def binding_response_problems(row: Any) -> list[str]:
    """Judge one binding response dict against the
    envelope, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["response: the row is empty"]
    problems: list[str] = []
    for field in ("call", "ok", "status",
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
    if row.get("contract_version") != API_VERSION:
        problems.append(
            f"response: contract_version is "
            f"{row.get('contract_version')!r}, not "
            f"TF-034's {API_VERSION!r}")
    if "seat" in row:
        seat = row["seat"]
        if not isinstance(seat, dict) \
                or not seat.get("name"):
            problems.append(
                "response: seat is filed but names no seat")
        elif "kind" in seat \
                and seat["kind"] not in SEAT_KINDS:
            problems.append(
                f"response: seat kind is "
                f"{seat['kind']!r}, not a filed kind")
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


# ── The filed turn-in assembly + the reply shape ─────────────────────────────

def assemble_turn_in(
    seat: dict[str, Any],
    message: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The filed assembly rule, applied purely:
    the incoming message's body is user_text;
    each caller-supplied earlier message, in
    hall order, is a history entry — role
    "assistant" when its author is the seat
    itself, "user" otherwise; content is its
    body; its author rides as "name". Assumes
    seat_request_problems() judged the request
    clean."""
    entries: list[dict[str, Any]] = []
    for entry in history or []:
        entries.append({
            "role": "assistant"
                    if entry["author"] == seat["name"]
                    else "user",
            "content": entry["body"],
            "name": entry["author"],
        })
    return {"user_text": message["body"],
            "history": entries}


def reply_message(
    seat: dict[str, Any],
    message: dict[str, Any],
    turn: dict[str, Any],
) -> dict[str, Any]:
    """The filed reply shape for one turn-out: a
    Samtal speak-door message — the incoming
    message's room, the turn's text as the body,
    the seat's name as the author."""
    return {"room": message.get("room"),
            "author": seat["name"],
            "body": turn.get("text", "")}


# ── The binding ──────────────────────────────────────────────────────────────

def _detached(data: Any) -> Any:
    """A detached JSON-safe copy (a JSON round-trip)."""
    return json.loads(json.dumps(data))


class SamtalBinding:
    """A Samtal seat's thinking routed through
    ThoughtForge, under Samtal's own rules, off
    by default.

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
    no Samtal act: it seats, claims, heals,
    presences and moderates no one — cognition
    only, the Astrid Law untouched.
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
            "contract_version": API_VERSION,
            "error": None if ok else fields.pop(
                "error", ""),
            **fields,
        })

    def _refused(
        self, call: str, seat: Any = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if isinstance(seat, dict) and seat.get("name"):
            filed: dict[str, Any] = {"name": seat["name"]}
            if seat.get("kind") in SEAT_KINDS:
                filed["kind"] = seat["kind"]
            fields["seat"] = filed
        return self._envelope(
            call, False, "refused", error=DISABLED_ERROR,
            **fields)

    # ── The seat's thinking call ──

    def seat_reply(
        self,
        seat: dict[str, Any],
        message: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """The seat thinking call: a seat identity
        (``{name, kind}`` — the principal's name
        and filed kind) and the Samtal message
        being answered in, one binding response
        out — the TF-154 turn verbatim in
        ``turn`` and the filed reply message
        shape in ``reply``.

        While disabled, the filed refusal returns
        as data and nothing is judged, built or
        routed. When enabled, an invalid request
        raises ValueError naming the problem(s),
        judged by ``seat_request_problems()``
        before any API is built or called. Every
        seat routes identically — the seat's kind
        is provenance, never rank."""
        if not self._enabled:
            return self._refused("seat_reply", seat)
        problems = seat_request_problems(seat, message, history)
        if problems:
            raise ValueError("; ".join(problems))
        turn_in = assemble_turn_in(seat, message, history)
        kwargs: dict[str, Any] = {}
        if turn_in["history"]:
            kwargs["history"] = turn_in["history"]
        turn = self._get_cognition().turn(
            turn_in["user_text"], **kwargs)
        filed_seat: dict[str, Any] = {"name": seat["name"]}
        if seat.get("kind") is not None:
            filed_seat["kind"] = seat["kind"]
        return self._envelope(
            "seat_reply", True, "replied",
            seat=filed_seat, turn=turn,
            reply=reply_message(seat, message, turn))

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

class _FakeSamtalConsumer:
    """A dict-shaped stand-in for a Samtal seat's
    process, built from the filed seat / message
    data only: it holds seats by name (the
    principals shape, minus any token — the
    binding never sees one) and speaks the filed
    message shape. Fixture only — no Samtal
    code, no live server."""

    def __init__(self) -> None:
        self._seats: dict[str, dict[str, Any]] = {}
        self._next_id = 1

    def seat(self, name: str, kind: str) -> dict[str, Any]:
        seat = {"id": self._next_id, "name": name,
                "kind": kind}
        self._next_id += 1
        self._seats[name] = seat
        return {"name": name, "kind": kind}

    def message(
        self, author: str, body: str, room: int = 1,
    ) -> dict[str, Any]:
        author_seat = self._seats[author]
        return {"id": self._next_id, "room": room,
                "author": author,
                "author_kind": author_seat["kind"],
                "body": body, "ts": 0.0}


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


def _knowledge_binding() -> tuple[SamtalBinding, Any]:
    """An enabled binding over a knowledge-only
    core (TF-154's TF-084 fixture shape) and a
    temp store. Returns the binding and the
    temp home."""
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.knowledge.store import MemoryStore

    tmp = tempfile.mkdtemp(prefix="tf157-")
    core = ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    store = MemoryStore(memory_dir=Path(tmp) / "store")
    binding = SamtalBinding(
        cognition_api=CognitionAPI(core=core),
        memory_api=MemoryAPI(store=store),
        enabled=True,
    )
    return binding, tmp


def _temp_memory_binding() -> tuple[SamtalBinding, Any, str]:
    """An enabled binding whose memory side is a
    TF-155 MemoryAPI over a fresh temp store
    (the cognition side stays unbuilt unless a
    cognition call lands)."""
    from thoughtforge.knowledge.store import MemoryStore

    tmp = tempfile.mkdtemp(prefix="tf157-")
    store = MemoryStore(memory_dir=Path(tmp) / "store")
    binding = SamtalBinding(
        memory_api=MemoryAPI(store=store), enabled=True)
    return binding, store, tmp


_SEAT_A = {"name": "SeatA", "kind": "agent"}
_SEAT_B = {"name": "SeatB", "kind": "kindroid"}
_MESSAGE = {"id": 7, "room": 1, "author": "VolmarrViking",
            "author_kind": "human",
            "body": "What time does the ferry leave?",
            "ts": 0.0}


# ── The filed probes ─────────────────────────────────────────────────────────

SAMTAL_BINDING_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "disabled_refusal_touches_nothing",
     "kind": "disabled_refusal",
     "detail": "a default binding refuses every "
               "routed call as data (ok=False, the "
               "error naming the binding disabled), "
               "and counting factories prove neither "
               "API is ever built"},
    {"id": "enabled_cognition_round_trip_through_binding",
     "kind": "cognition_round_trip",
     "detail": "an enabled binding routes a seat's "
               "reply through the injected "
               "CognitionAPI by the filed rule (the "
               "incoming body as user_text, supplied "
               "earlier messages as history — the "
               "seat's own words as assistant) and "
               "returns the turn with the filed reply "
               "message shape (the incoming room, "
               "the turn's text as body)"},
    {"id": "enabled_memory_round_trip_temp_store",
     "kind": "memory_round_trip",
     "detail": "an enabled binding adds through "
               "TF-155 (stamped manual_entry), finds "
               "the record through TF-155's inspect, "
               "and exports the verified bundle — all "
               "through the binding, on a temp store"},
    {"id": "equal_seats_routed_identically",
     "kind": "equal_seats",
     "detail": "two differently-named, "
               "differently-kinded seats route "
               "identically in shape: same envelope "
               "keys, same call / status / contract "
               "version, same turn and reply keys — "
               "only the filed seat identity and the "
               "words differ"},
    {"id": "astrid_law_no_seating_calls_routed",
     "kind": "astrid_law",
     "detail": "no seating, claim, healing, "
               "presence, room-administration or "
               "moderation call exists in CALL_ROWS "
               "as routed — join_seating / "
               "seat_healing / presence / "
               "room_administration / moderation are "
               "all filed NOT routed, and the binding "
               "class files no such method"},
    {"id": "fake_consumer_consistency",
     "kind": "fake_consumer",
     "detail": "a fake Samtal consumer shaped on "
               "the filed seat / message data seats "
               "two seats and routes one's reply "
               "through the binding only: the seat "
               "identities it hands over are exactly "
               "the filed {name, kind} shape (no "
               "token anywhere) and the reply lands "
               "in the message's room under the "
               "answering seat's name"},
    {"id": "doc_matches_render_and_version_is_tf034",
     "kind": "doc_and_version",
     "detail": "the filed doc is the render "
               "verbatim, the contract version is "
               "TF-034's API_VERSION with its surface "
               "clean, and DEFAULT_ENABLED is False"},
)


def run_samtal_binding_probes() -> list[dict[str, Any]]:
    """Run the filed probes; each returns its
    verdict."""
    results: list[dict[str, Any]] = []
    for probe in SAMTAL_BINDING_PROBES:
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

                binding = SamtalBinding(
                    cognition_factory=cog_factory,
                    memory_factory=mem_factory)
                outs = [
                    binding.seat_reply(_SEAT_A, _MESSAGE),
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
                binding = SamtalBinding(
                    cognition_api=fake, enabled=True)
                history = [
                    {"author": "SeatA",
                     "body": "earlier words of mine"},
                    {"author": "VolmarrViking",
                     "body": "earlier question"},
                ]
                out = binding.seat_reply(
                    _SEAT_A, _MESSAGE, history=history)
                passed = out.get("ok") is True \
                    and fake.turns == [{
                        "user_text":
                            "What time does the ferry leave?",
                        "history": [
                            {"role": "assistant",
                             "content": "earlier words of mine",
                             "name": "SeatA"},
                            {"role": "user",
                             "content": "earlier question",
                             "name": "VolmarrViking"},
                        ]}] \
                    and out["turn"]["text"] \
                    == "stub answer to: What time does " \
                       "the ferry leave?" \
                    and out["reply"] == {
                        "room": 1, "author": "SeatA",
                        "body": "stub answer to: What "
                                "time does the ferry "
                                "leave?"} \
                    and binding_response_problems(out) == []
            elif kind == "memory_round_trip":
                binding, _store, tmp = _temp_memory_binding()
                added = binding.memory_add(
                    "user_fact", summary="ferry at 07:30",
                    record_id="fct_tf157")
                found = binding.memory_inspect(
                    record_id="fct_tf157")
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
            elif kind == "equal_seats":
                fake = _RecordingCognitionAPI()
                binding = SamtalBinding(
                    cognition_api=fake, enabled=True)
                out_a = binding.seat_reply(_SEAT_A, _MESSAGE)
                out_b = binding.seat_reply(_SEAT_B, _MESSAGE)
                passed = set(out_a) == set(out_b) \
                    and out_a["call"] == out_b["call"] \
                    and out_a["status"] == out_b["status"] \
                    and out_a["contract_version"] \
                    == out_b["contract_version"] \
                    and set(out_a["turn"]) == set(out_b["turn"]) \
                    and set(out_a["reply"]) == set(out_b["reply"]) \
                    and out_a["seat"] == _SEAT_A \
                    and out_b["seat"] == _SEAT_B \
                    and out_a["reply"]["author"] == "SeatA" \
                    and out_b["reply"]["author"] == "SeatB" \
                    and fake.turns[0] == fake.turns[1]
            elif kind == "astrid_law":
                by_id = {r["id"]: r for r in CALL_ROWS}
                passed = all(
                    by_id[cid]["routed"] is False
                    for cid in ("join_seating", "seat_healing",
                                "presence",
                                "room_administration",
                                "moderation")) \
                    and not any(
                        hasattr(SamtalBinding, name)
                        for name in ("join", "seat", "heal",
                                     "claim", "presence",
                                     "moderate", "create_room"))
            elif kind == "fake_consumer":
                consumer = _FakeSamtalConsumer()
                seat_a = consumer.seat("SeatA", "agent")
                consumer.seat("VolmarrViking", "human")
                message = consumer.message(
                    "VolmarrViking",
                    "What time does the ferry leave?")
                fake = _RecordingCognitionAPI()
                binding = SamtalBinding(
                    cognition_api=fake, enabled=True)
                out = binding.seat_reply(seat_a, message)
                passed = seat_a == _SEAT_A \
                    and "token" not in json.dumps(seat_a) \
                    and out.get("ok") is True \
                    and out["reply"]["room"] == 1 \
                    and out["reply"]["author"] == "SeatA" \
                    and binding_response_problems(out) == []
            elif kind == "doc_and_version":
                doc = _REPO_ROOT / SAMTAL_BINDING_DOC
                passed = doc.exists() \
                    and doc.read_text(encoding="utf-8") \
                    == render_samtal_binding_markdown() \
                    and samtal_binding_contract()[
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


# ── The readable render (the samtal-binding doc, verbatim) ───────────────────

def render_samtal_binding_markdown() -> str:
    """Render the samtal-binding doc
    deterministically — the exact text filed as
    docs/architecture/SAMTAL_BINDING.md."""
    lines: list[str] = []
    lines.append("# Samtal Binding — A Seat's Thinking "
                 "Through ThoughtForge (TF-157)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/samtal_binding.py` — the "
        "filed doc IS the render: regenerate with "
        "`python3 -m thoughtforge.samtal_binding`. "
        "The checker (`samtal_binding_problems()`) "
        "re-derives every claim live from the tree "
        "and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{SAMTAL_BINDING_FORMAT}`")
    lines.append(
        f"Contract version: `{API_VERSION}` (TF-034's "
        "`API_VERSION` — consumed, not reinvented)")
    lines.append(
        f"Default enabled: {DEFAULT_ENABLED} — the "
        "binding is OFF by default; enabling is an "
        "explicit act (`SamtalBinding(enabled=True)` "
        "or `binding.enable()`)")
    lines.append(f"Seats: {len(SEAT_ROWS)} (one per "
                 f"kind Samtal files, all routing "
                 f"identically); calls: {len(CALL_ROWS)} "
                 f"({len(ROUTED_CALL_IDS)} routed, "
                 f"{len(NOT_ROUTED_CALL_IDS)} filed NOT "
                 f"routed); field rows: {len(FIELD_ROWS)}; "
                 f"rules: {len(RULE_ROWS)}")
    lines.append(f"Probes: {len(SAMTAL_BINDING_PROBES)}/"
                 f"{len(SAMTAL_BINDING_PROBES)}")
    lines.append("")
    lines.append("## 1. One binding")
    lines.append("")
    lines.append(
        "`SamtalBinding` wraps TF-154's `CognitionAPI` "
        "and TF-155's `MemoryAPI` — each injectable "
        "directly, by factory, or via a `turn_callable` "
        "/ `memory_store` passthrough, and both built "
        "lazily: constructing the binding builds "
        "neither API. While disabled, every routed "
        "call returns a truthful refusal as data "
        "(`ok=False`, `status=\"refused\"`, the "
        "`error` naming the binding as disabled) and "
        "touches neither API. When enabled, a seat's "
        "thinking call routes through the CognitionAPI "
        "and memory calls through the MemoryAPI — "
        "called, never reimplemented. An invalid "
        "request raises `ValueError` naming the "
        "problem(s), judged by the pure validators "
        "before any API is built or called. Every "
        "response is a plain JSON-safe detached dict: "
        "`call`, `ok`, `status`, `contract_version` "
        "(TF-034's `API_VERSION`), `error`, plus the "
        "call's payload (`seat` + `turn` + `reply` for "
        "a seat reply, the TF-155 response verbatim "
        "in `memory` for memory calls). No Samtal "
        "code is imported or copied anywhere: Samtal's "
        "shapes are filed below as literal data read "
        "READ-ONLY from the canonical Samtal tree, "
        "and the binding performs no Samtal act — it "
        "seats, claims, heals, presences and "
        "moderates no one. Cognition only.")
    lines.append("")
    lines.append("## 2. The seats — every seat equal")
    lines.append("")
    lines.append("| Row | Kind | Samtal side | Binding "
                 "identity | Binding method | Privileged | "
                 "What it is |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in SEAT_ROWS:
        privileged = "yes" if row["privileged"] else "no"
        lines.append(
            f"| {row['id']} | {row['kind']} | "
            f"{row['samtal_side']} | "
            f"{row['binding_identity']} | "
            f"{row['binding_method']} | {privileged} | "
            f"{row['summary']} |")
    lines.append("")
    lines.append(
        "A seat is a `principals` row in Samtal's "
        "`server.py` (`id` / `name` UNIQUE / `kind` / "
        "`token` / `last_seen`, plus `display_name` / "
        "`platform` / `agent_version` / "
        "`capabilities` / `token_enabled` / "
        "`retired`). The binding never sees the "
        "token: a seat is handed over as `{name, "
        "kind}` only. The five kinds are Samtal's own "
        "provenance badges — the binding invents no "
        "seat class and ranks none.")
    lines.append("")
    lines.append("## 3. The calls")
    lines.append("")
    lines.append("| Call | Surface | Samtal side | "
                 "ThoughtForge route | Binding method | "
                 "Routed |")
    lines.append("|---|---|---|---|---|---|")
    for row in CALL_ROWS:
        routed = "yes" if row["routed"] else "no"
        method = row["binding_method"] or "—"
        lines.append(
            f"| {row['id']} | {row['surface']} | "
            f"{row['samtal_side']} | "
            f"{row['thoughtforge_route']} | {method} | "
            f"{routed} |")
    lines.append("")
    lines.append("## 4. Field mapping — in (Samtal -> turn-in)")
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
    lines.append("## 6. The rules — Samtal's own, as constraints")
    lines.append("")
    for rule in RULE_ROWS:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## 7. Docket")
    lines.append("")
    lines.append(
        "TF-157 has **no share** of TF-016 and found "
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
    for probe in SAMTAL_BINDING_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 9. Filed limits")
    lines.append("")
    lines.append(
        "- Proof grade FIXTURE: the checker and "
        "probes run a fake Samtal consumer shaped on "
        "the filed seat / message data. NO live "
        "Samtal server was routed and NO Samtal code "
        "was imported or copied; live end-to-end "
        "routing is NOT PROVEN — TF-170's Part IX "
        "gate owns the at-least-one-live proof, "
        "never claimed here.")
    lines.append(
        "- The Astrid Law is filed as a constraint, "
        "never as a behaviour: the binding performs "
        "no seating, claim, healing, presence or "
        "moderation act, and files every such call "
        "NOT routed with its reason. Seat healing "
        "stays Samtal's own act at its own boot.")
    lines.append(
        "- The seat's token never enters the "
        "binding: a seat is handed over as `{name, "
        "kind}` only, and the binding authenticates "
        "as no one — speaking the reply at Samtal's "
        "door stays the seat process's own act "
        "with its own credential.")
    lines.append(
        "- History is caller-supplied: the binding "
        "reads no hall scroll itself (it holds no "
        "credential and calls no Samtal door); "
        "earlier messages ride only when the seat's "
        "process supplies them with the call.")
    lines.append(
        "- TF-158 owns the Hermes binding; TF-161 "
        "owns binding fixtures in depth; TF-162 "
        "owns versioning enforcement in depth; "
        "TF-163 owns the local-only proof for "
        "bindings. TF-157 files the Samtal binding "
        "shape and its off-by-default routing under "
        "Samtal's own rules.")
    lines.append(
        "- No canon data was touched (TF-017); no "
        "README.md anywhere was touched; no Docker "
        "was used; nothing under the Samtal tree "
        "was edited.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_samtal_binding_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / SAMTAL_BINDING_DOC
    doc.write_text(render_samtal_binding_markdown(),
                   encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.samtal_binding`."""
    doc = write_samtal_binding_doc()
    print(f"samtal binding doc regenerated: -> "
          f"{SAMTAL_BINDING_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def samtal_binding_problems() -> list[str]:
    """Re-derive TF-157's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    if SAMTAL_BINDING_FORMAT != "thoughtforge-samtal-binding/1":
        problems.append(
            f"the format is {SAMTAL_BINDING_FORMAT!r}, not "
            "thoughtforge-samtal-binding/1")
    if DEFAULT_ENABLED is not False:
        problems.append("DEFAULT_ENABLED is not False")
    if not DISABLED_ERROR or "disabled" not in DISABLED_ERROR:
        problems.append(
            "DISABLED_ERROR does not name the binding as "
            "disabled")

    seat_ids = [r["id"] for r in SEAT_ROWS]
    if seat_ids != [f"seat_{k}" for k in SEAT_KINDS] \
            or len(seat_ids) != len(set(seat_ids)):
        problems.append(
            f"the SEAT_ROWS filing is not one row per "
            f"filed kind: {seat_ids}")
    for row in SEAT_ROWS:
        problems.extend(seat_row_problems(row))
    methods = {r["binding_method"] for r in SEAT_ROWS}
    if len(methods) != 1:
        problems.append(
            f"the seats do not all route through one "
            f"method: {methods}")

    call_ids = [r["id"] for r in CALL_ROWS]
    expected_calls = [
        "seat_reply", "memory_add", "memory_inspect",
        "memory_prune", "memory_export", "join_seating",
        "seat_healing", "presence", "room_administration",
        "moderation"]
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
    # The Astrid-Law proof, at the filing level.
    for call_id in ("join_seating", "seat_healing",
                    "presence", "room_administration",
                    "moderation"):
        row = next(r for r in CALL_ROWS if r["id"] == call_id)
        if row["routed"]:
            problems.append(
                f"Astrid Law: call {call_id} is routed — "
                "Samtal's own acts must stay NOT routed")

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
    token_row = next(
        (f for f in FIELD_ROWS if f["id"] == "seat.token"),
        None)
    if token_row is None \
            or token_row.get("disposition") != "no_home_not_passed":
        problems.append(
            "the seat token is not filed no-home in "
            "FIELD_ROWS — the binding must never carry it")

    rule_ids = [r["id"] for r in RULE_ROWS]
    if len(rule_ids) != len(set(rule_ids)) \
            or len(RULE_ROWS) != 7:
        problems.append(
            "the RULE_ROWS filing is not seven unique rules")
    for rule in RULE_ROWS:
        problems.extend(rule_row_problems(rule))
    probe_ids = [p["id"] for p in SAMTAL_BINDING_PROBES]
    if len(probe_ids) != len(set(probe_ids)) \
            or len(SAMTAL_BINDING_PROBES) != 7:
        problems.append(
            "the SAMTAL_BINDING_PROBES filing is not "
            "seven unique probes")

    # The contract is TF-034-versioned and its
    # surface is clean.
    contract = samtal_binding_contract()
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
            ("TF-156 runmal_binding", runmal_binding_problems)):
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

        binding = SamtalBinding(
            cognition_factory=cog_factory,
            memory_factory=mem_factory)
        refused = binding.seat_reply(_SEAT_A, _MESSAGE)
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
    # reply and a temp-store memory add / find,
    # through the binding only — plus the
    # equal-seats shape proof.
    try:
        binding, _tmp = _knowledge_binding()
        out = binding.seat_reply(_SEAT_A, _MESSAGE)
        problems.extend(
            f"round-trip reply: {p}"
            for p in binding_response_problems(out))
        if out.get("ok") is not True \
                or out.get("turn", {}).get("mode") \
                != "knowledge_only" \
                or out.get("reply", {}).get("room") != 1 \
                or out.get("reply", {}).get("author") \
                != "SeatA":
            problems.append(
                f"round-trip reply is not a "
                f"knowledge_only turn with the filed "
                f"reply shape: {out}")
        other = binding.seat_reply(_SEAT_B, _MESSAGE)
        if set(other) != set(out) \
                or other.get("status") != out.get("status") \
                or set(other.get("reply", {})) \
                != set(out.get("reply", {})):
            problems.append(
                f"equal seats: a differently-kinded seat "
                f"did not route identically in shape: "
                f"{other}")
        added = binding.memory_add(
            "user_fact", summary="ferry at 07:30",
            record_id="fct_tf157_check")
        if added.get("ok") is not True \
                or added.get("memory", {}).get(
                    "record", {}).get("source") \
                != "manual_entry":
            problems.append(
                f"round-trip memory add is not a stamped "
                f"add: {added}")
        found = binding.memory_inspect(
            record_id="fct_tf157_check")
        if found.get("memory", {}).get("status") != "found":
            problems.append(
                f"round-trip memory find: {found}")
    except Exception as exc:
        problems.append(f"round-trip failed: {exc}")

    # The probes all behave.
    probes = run_samtal_binding_probes()
    if len(probes) != len(SAMTAL_BINDING_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the samtal binding probes do not all pass: "
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
            "FIXED; TF-157 must not move them")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / SAMTAL_BINDING_DOC
    if not doc.exists():
        problems.append(
            f"the samtal binding doc {SAMTAL_BINDING_DOC} "
            "is missing")
    elif doc.read_text(encoding="utf-8") \
            != render_samtal_binding_markdown():
        problems.append(
            f"the filed doc {SAMTAL_BINDING_DOC} is not "
            "the render verbatim (regenerate with "
            "python3 -m thoughtforge.samtal_binding)")
    return problems


def samtal_binding_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for
    consumers."""
    probes = run_samtal_binding_probes()
    return {
        "format": SAMTAL_BINDING_FORMAT,
        "doc": SAMTAL_BINDING_DOC,
        "contract": samtal_binding_contract(),
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
    "MESSAGE_FIELDS",
    "NOT_ROUTED_CALL_IDS",
    "ROUTED_CALL_IDS",
    "RULE_IDS",
    "RULE_ROWS",
    "SAMTAL_BINDING_DOC",
    "SAMTAL_BINDING_FORMAT",
    "SAMTAL_BINDING_PROBES",
    "SAMTAL_BINDING_RULES",
    "SEAT_KINDS",
    "SEAT_ROW_IDS",
    "SEAT_ROWS",
    "SamtalBinding",
    "assemble_turn_in",
    "binding_response_problems",
    "binding_rule_problems",
    "call_row_problems",
    "field_row_problems",
    "field_rows_for",
    "main",
    "render_samtal_binding_markdown",
    "reply_message",
    "rule_row_problems",
    "run_samtal_binding_probes",
    "samtal_binding_as_data",
    "samtal_binding_contract",
    "samtal_binding_problems",
    "seat_request_problems",
    "seat_row_problems",
    "write_samtal_binding_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
