"""TF-076 — The OpenAI-compatible backend: any server
speaking the common shape binds under the TF-065 contract.

Roadmap: "any server speaking the common shape binds under
the same contract."

Measured start (probed this tick, VM):

- **The common shape was already reachable in fact, but
  only as a side effect**: ``LMStudioBackend``'s own
  docstring claims it "works with LM Studio, vLLM,
  text-generation-webui, Oobabooga, LocalAI, and any server
  that implements the /v1/chat/completions endpoint", and
  the config factory maps the ``openai_compatible`` backend
  type onto that class. There was no generic binding of
  its own: no family identity (``backend_name`` says
  ``lmstudio:...`` whatever the server is), no filed
  shape, and no rules for where the common shape's many
  dialects diverge.
- **The LM Studio leg's response parsing breaches the
  contract on real-shaped server answers** (mocked probes
  this tick, D-OAI-1): a ``finish_reason`` of
  ``content_filter`` or ``tool_calls`` passes straight
  through into ``GenerationResponse.finish_reason`` —
  outside TF-065's ``FINISH_REASONS`` vocabulary
  (stop / length / error) — and a ``content: null``
  message (the shape's normal tool-call/filter answer)
  lands as ``text=None`` in a field every consumer treats
  as ``str``. An empty ``choices`` list surfaces only as
  the cryptic ``'list index out of range'``.
- **A conventional OpenAI base URL breaks the LM Studio
  leg** (D-OAI-2): the shape's own convention is a base
  ending in ``/v1`` (``https://api.openai.com/v1``,
  ``http://host:8000/v1``), and the leg appends
  ``/v1/chat/completions`` unconditionally — probe:
  base ``https://api.example.test/v1`` produced
  ``https://api.example.test/v1/v1/chat/completions``.
- **A response without ``usage`` reports zero tokens for
  real text** (D-OAI-3): several shape-speaking servers
  omit the usage block; the leg read
  ``usage.get("completion_tokens", 0)`` and reported 0 —
  a false count, where TF-074's Aesir binding already
  established the honest pattern (the server's count when
  reported, the declared contract estimate otherwise).
- **Server error bodies were discarded**: a 401 whose
  body carries the shape's own
  ``{"error": {"message": "..."}}`` surfaced as the bare
  string ``HTTP 401`` — the server's stated reason lost.

THE SHAPE (``OPENAI_SHAPE`` as data) — the common subset
this binding speaks, and nothing beyond it:

- ``POST /v1/chat/completions`` (non-streamed here; the
  shape's SSE streaming exists but this backend claims
  only ``simulated`` streaming through the TF-065 base
  default — a native stream leg is a later slice's to
  prove per server family, never assumed for all).
- ``GET /v1/models`` — the model list (ids). Server-level
  reachability only: a listing is not proof the model
  generates (the TF-074 catalog lesson, restated for
  this family in ``model_listed``).
- Request fields sent: ``messages``, ``temperature``,
  ``max_tokens``, ``stream: false``, ``model`` when
  configured, ``stop`` when the request carries it
  (the shape serves stop — the opposite of Aesir's
  ruling), ``seed`` when the request carries one
  (TF-065's seed duty; servers that ignore an unknown
  seed field are the shape's own dialect fact, and the
  capability record says ``honoured`` in the contract's
  sense: forwarded, never silently dropped).
- Auth: an ``Authorization: Bearer <key>`` header only
  when a key is configured. Local shape-servers commonly
  accept any or no key; sending a placeholder key (the
  LM Studio preset's ``lm-studio`` default) to a server
  that validates keys is a false credential — the
  generic binding sends none when none was given.
- Response: ``choices[0].message.content`` (a string),
  ``choices[0].finish_reason``, ``usage.completion_tokens``
  when the server reports it.

THE RULES (``OPENAI_RULES`` as data), implemented by
:class:`OpenAICompatibleBackend` and — for the three
defects — by the shared parser
:func:`chat_completion_outcome`, which ``LMStudioBackend``
now also uses (D-OAI-1/2/3 FIXED in both legs; the LM
Studio class stays as the family's LM Studio preset —
its defaults, name, and factory seat unchanged):

- ``base_url_v1_normalised`` — a base URL ending in
  ``/v1`` (with or without a trailing slash) names the
  same server as the bare root; routes are built from
  the root, so the doubled-``/v1`` URL can never be
  produced.
- ``finish_vocabulary_enforced`` — ``stop`` and
  ``length`` pass; any other finish reason
  (``tool_calls``, ``content_filter``, ``function_call``,
  …) becomes error data naming the server's reason.
  This backend sends no tools, so such a finish is a
  server-side event the contract has no honest success
  word for — never passed through out-of-vocabulary,
  never dressed as a completion.
- ``content_is_text_or_error`` — message content must
  be a string; null or non-string content becomes error
  data naming the shape breach. ``text`` on a
  ``GenerationResponse`` is never ``None``.
- ``no_choices_is_named_error`` — a response with no
  choices is error data saying so, not an index error.
- ``tokens_usage_or_declared_estimate`` —
  ``tokens_generated`` is the server's
  ``usage.completion_tokens`` when reported, the
  contract estimate (declared in the capability record)
  otherwise — never a silent 0 over real text.
- ``errors_are_data_with_the_servers_reason`` — HTTP
  failures return TF-065 error responses naming the
  status and the server's own error message, parsed
  from both error shapes (``{"error": {"message": …}}``
  and ``{"error": "…"}``); ``generate()`` never raises.
- ``auth_only_when_configured`` — no key configured,
  no Authorization header sent.
- ``health_is_server_level_model_listed_separately`` —
  ``health_check()`` is ``GET /v1/models`` answering
  exactly 200 (the D-HEALTH-1 standard); whether the
  configured model is listed is ``model_listed()``'s
  separate, weaker-or-stronger answer — health never
  silently claims it (D-HEALTH-2's per-model road stays
  TF-077's; this surface is the data it will consume).
- ``one_shape_many_servers_no_live_claim`` — the named
  server families in ``KNOWN_SERVERS`` are shape claims
  from the tree's own docstring and the shape's public
  convention, each filed ``shape_bound_not_live_proven``:
  no real third-party server has faced this backend on
  this VM (D-OAI-4, owned by TF-086's gate).

Placement in the estate: like TF-073's FallbackBackend
and TF-074's AesirBackend, this backend is a
``UnifiedBackend`` conformed and tested against the
TF-065 contract directly, but it does NOT join
``SHIPPED_BACKENDS`` or the config factory — the
factory's ``openai_compatible`` seat keeps returning
the LM Studio preset until TF-153 owns admitting new
backends, and TF-065's shipped set stays the four it
pinned. One deliberate contract amendment IS made in
scope: this module imports ``requests`` at top level
like its sibling backends, so it joins
``REQUESTS_CONSUMERS`` (D-DEP-1's set, amended on
purpose, five → six).

Proof grade FIXTURE + LIVE-HTTP-LOOPBACK (mocked
transports in the shape's own payload forms, the
TF-065 checker over the real class, one real core turn,
and a real HTTP round trip against a stdlib fixture
server on loopback — the wire, headers, and parsing
exercised end to end) — a real third-party server
(LM Studio, vLLM, LocalAI, the hosted OpenAI API, …)
on the other end is NOT PROVEN (D-OAI-4).
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any
from urllib.parse import urlparse

import requests

from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)
# TF-025 (D-DUP-1): _build_messages' one implementation lives in
# utils/common.py; this name is its alias.
from thoughtforge.utils.common import build_messages as _build_messages

logger = logging.getLogger(__name__)

OPENAI_FORMAT = "thoughtforge-openai-compatible-backend/1"

#: Finish reasons the shape may return that the TF-065
#: contract vocabulary can carry as success.
SERVED_FINISH_REASONS = ("stop", "length")


def normalize_base_url(base_url: str) -> str:
    """The server root for a shape base URL (D-OAI-2).

    The shape's convention lets a base URL end in ``/v1``
    (``https://api.openai.com/v1``); routes are built as
    ``<root>/v1/...``, so a ``/v1`` suffix — with or
    without a trailing slash — is stripped here, once,
    for every leg of this family.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


def server_error_message(resp: Any) -> str:
    """The server's own error message, from the shape's
    error forms: ``{"error": {"message": "..."}}`` or
    ``{"error": "..."}``. '' when neither parses."""
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001 — judged, not hidden
        return ""
    if not isinstance(data, dict):
        return ""
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or "")
    if isinstance(err, str):
        return err
    return str(data.get("message") or "")


def chat_completion_outcome(data: Any) -> dict[str, Any]:
    """Judge one chat-completion body against the shape's
    rules (D-OAI-1 / D-OAI-3) — the ONE parser for this
    family, shared by OpenAICompatibleBackend and
    LMStudioBackend.

    Returns ``{"ok": True, "text", "finish_reason",
    "completion_tokens"}`` (``completion_tokens`` None
    when the server reported no usage) or
    ``{"ok": False, "error"}`` naming the breach.
    """
    if not isinstance(data, dict):
        return {"ok": False,
                "error": "openai-compatible: response body is "
                         f"{type(data).__name__}, not a JSON "
                         "object"}
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"ok": False,
                "error": "openai-compatible: response carries "
                         "no choices"}
    choice = choices[0]
    if not isinstance(choice, dict):
        return {"ok": False,
                "error": "openai-compatible: first choice is "
                         f"{type(choice).__name__}, not an object"}
    finish = choice.get("finish_reason") or "stop"
    if finish not in SERVED_FINISH_REASONS:
        return {"ok": False,
                "error": f"openai-compatible: server "
                         f"finish_reason {finish!r} is outside "
                         f"the TF-065 contract vocabulary "
                         f"(stop/length/error) — this backend "
                         f"sends no tools, so the finish is "
                         f"never dressed as a completion"}
    message = choice.get("message")
    if not isinstance(message, dict):
        return {"ok": False,
                "error": "openai-compatible: choice carries "
                         "no message object"}
    content = message.get("content")
    if not isinstance(content, str):
        return {"ok": False,
                "error": f"openai-compatible: message content "
                         f"is {type(content).__name__}, not "
                         f"text (finish_reason {finish!r}) — "
                         f"never passed on as a None text"}
    usage = data.get("usage")
    tokens: int | None = None
    if isinstance(usage, dict):
        raw = usage.get("completion_tokens")
        if isinstance(raw, int):
            tokens = raw
    return {"ok": True, "text": content,
            "finish_reason": finish,
            "completion_tokens": tokens}


# ── The shape, as data ─────────────────────────────────────────

OPENAI_SHAPE: tuple[dict[str, Any], ...] = (
    {"id": "chat_route",
     "claim": "POST /v1/chat/completions, non-streamed "
              "(stream: false always in this binding)",
     "status": "bound"},
    {"id": "models_route",
     "claim": "GET /v1/models returns data[] entries with "
              "an id — the server-level health + catalog "
              "surface",
     "status": "bound"},
    {"id": "sse_streaming",
     "claim": "the shape defines SSE streaming for the "
              "chat route; servers vary in dialect details",
     "status": "not_bound_simulated_claimed"},
    {"id": "tokenise_route",
     "claim": "the common shape guarantees no tokenise / "
              "token-count endpoint",
     "status": "not_served_estimate_declared"},
    {"id": "embeddings_route",
     "claim": "POST /v1/embeddings exists in the shape "
              "but ThoughtForge binds embeddings in its "
              "knowledge layer (TF-053), never here",
     "status": "not_bound"},
)

#: Server families named as speaking the common shape.
#: Status is a shape claim, never a live proof (D-OAI-4).
KNOWN_SERVERS: tuple[dict[str, Any], ...] = (
    {"id": "lm_studio", "status": "shape_bound_not_live_proven",
     "note": "the family's in-tree preset (LMStudioBackend) "
             "targets this server; TF-076's defaults match "
             "its conventional port 1234 shape"},
    {"id": "vllm", "status": "shape_bound_not_live_proven",
     "note": "named in LMStudioBackend's docstring"},
    {"id": "text_generation_webui_oobabooga",
     "status": "shape_bound_not_live_proven",
     "note": "named in LMStudioBackend's docstring"},
    {"id": "localai", "status": "shape_bound_not_live_proven",
     "note": "named in LMStudioBackend's docstring"},
    {"id": "llama_cpp_server",
     "status": "shape_bound_not_live_proven",
     "note": "llama.cpp's server speaks the chat shape"},
    {"id": "hosted_openai_api",
     "status": "shape_bound_not_live_proven",
     "note": "the shape's origin; a remote placement — "
             "binding it is a config act naming the host, "
             "and its conventional base ends in /v1 "
             "(D-OAI-2's case)"},
)

OPENAI_RULES: tuple[dict[str, Any], ...] = (
    {"id": "base_url_v1_normalised",
     "rule": "a base URL ending in /v1 names the same "
             "server as the bare root; routes are built "
             "from the root — the doubled-/v1 URL can "
             "never be produced (D-OAI-2)",
     "measured_by": "payload-capture pins over bare, "
                    "/v1, and /v1/ bases: one identical "
                    "route URL"},
    {"id": "finish_vocabulary_enforced",
     "rule": "stop and length pass; any other finish "
             "reason becomes error data naming the "
             "server's reason — never passed through "
             "out-of-vocabulary (D-OAI-1)",
     "measured_by": "content_filter and tool_calls "
                    "response pins: error responses "
                    "naming the reason, empty text"},
    {"id": "content_is_text_or_error",
     "rule": "message content must be a string; null or "
             "non-string content becomes error data — "
             "GenerationResponse.text is never None "
             "(D-OAI-1)",
     "measured_by": "content: null pin — an error "
                    "response, not a None text"},
    {"id": "no_choices_is_named_error",
     "rule": "a response with no choices is error data "
             "saying so, never an index error",
     "measured_by": "empty-choices and missing-choices "
                    "pins naming 'no choices'"},
    {"id": "tokens_usage_or_declared_estimate",
     "rule": "tokens_generated is the server's "
             "usage.completion_tokens when reported, the "
             "declared contract estimate otherwise — never "
             "a silent 0 over real text (D-OAI-3)",
     "measured_by": "usage-present and usage-absent pins; "
                    "the absent case equals count_tokens"},
    {"id": "errors_are_data_with_the_servers_reason",
     "rule": "HTTP failures return TF-065 error responses "
             "naming the status and the server's own error "
             "message (both error shapes parsed); "
             "generate() never raises",
     "measured_by": "401 pins in both error shapes + a "
                    "raising-transport pin"},
    {"id": "auth_only_when_configured",
     "rule": "an Authorization Bearer header is sent "
             "only when an api_key is configured — no "
             "placeholder credential is ever sent to a "
             "validating server",
     "measured_by": "header capture pins with and "
                    "without a configured key"},
    {"id": "health_is_server_level_model_listed_separately",
     "rule": "health_check() is GET /v1/models at exactly "
             "200; the configured model's listing is "
             "model_listed()'s separate answer (True / "
             "False / None when unasked or unanswerable)",
     "measured_by": "health pins over 200/500/raising; "
                    "model_listed pins: listed, absent, "
                    "no model configured, server down"},
    {"id": "stop_is_served_by_the_shape",
     "rule": "the shape serves stop sequences, so a "
             "carried stop is forwarded in the payload — "
             "the opposite ruling from Aesir's, each from "
             "its own server's truth",
     "measured_by": "payload pin: stop present when "
                    "carried, absent when not"},
    {"id": "live_proof_not_claimed",
     "rule": "no real third-party shape-speaking server "
             "has been on the other end of this backend; "
             "proofs are fixture + loopback-HTTP grade "
             "and the live road stays owed (D-OAI-4)",
     "measured_by": "KNOWN_SERVERS statuses; TF-086's "
                    "gate carries the NOT PROVEN line"},
)


class OpenAICompatibleBackend(UnifiedBackend):
    """Any OpenAI-compatible server, bound as a
    ThoughtForge backend under the TF-065 contract
    (TF-076)."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "",
        api_key: str = "",
        timeout: int = 120,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    # ── Identity / plumbing ─────────────────────────────

    def backend_name(self) -> str:
        if self.model:
            return f"openai-compatible:{self.model}"
        host = urlparse(self.base_url).netloc or self.base_url
        return f"openai-compatible:{host}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: GenerationRequest) -> dict:
        payload: dict[str, Any] = {
            "messages": _build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model
        if request.stop:
            # The shape serves stop (OPENAI_RULES) — forward
            # it, exactly as carried.
            payload["stop"] = request.stop
        if request.seed is not None:
            # TF-065 seed duty: the shape's seed field.
            payload["seed"] = request.seed
        return payload

    def _error_response(
        self, start: float, error: str,
    ) -> GenerationResponse:
        return GenerationResponse(
            text="",
            error=error,
            finish_reason="error",
            backend_used=self.backend_name(),
            latency_ms=(time.monotonic() - start) * 1000,
        )

    # ── generate ────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.monotonic()
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=self._payload(request),
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                message = server_error_message(resp)
                label = (f"openai-compatible HTTP "
                         f"{resp.status_code}")
                return self._error_response(
                    start,
                    f"{label}: {message}" if message else label)
            try:
                data = resp.json()
            except Exception as exc:  # noqa: BLE001 — data
                return self._error_response(
                    start,
                    f"openai-compatible: response body is not "
                    f"JSON ({exc})")
            outcome = chat_completion_outcome(data)
            if not outcome["ok"]:
                return self._error_response(start, outcome["error"])
            tokens = outcome["completion_tokens"]
            if tokens is None:
                # Declared fallback (capabilities note): the
                # contract estimate, never a silent 0.
                tokens = self.count_tokens(outcome["text"])
            return GenerationResponse(
                text=outcome["text"],
                tokens_generated=int(tokens),
                finish_reason=outcome["finish_reason"],
                backend_used=self.backend_name(),
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 — failure is data
            logger.error(
                "OpenAICompatibleBackend.generate error: %s", exc)
            return self._error_response(start, str(exc))

    # ── health / catalog ────────────────────────────────

    def health_check(self) -> bool:
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(), timeout=5)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001 — health never raises
            return False

    def list_models(self) -> list[str]:
        """Catalog ids from /v1/models (NOT a generability
        claim — the TF-074 catalog lesson)."""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])
                    if isinstance(m, dict) and "id" in m]
        except Exception as exc:  # noqa: BLE001 — stated, empty
            logger.warning(
                "OpenAICompatibleBackend.list_models failed: %s",
                exc)
            return []

    def model_listed(self) -> bool | None:
        """Is the configured model in the server's list?

        None when no model is configured or the server
        could not be asked — unknown, never dressed as
        absent. A True here is a listing, not proof the
        model generates.
        """
        if not self.model:
            return None
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(), timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            ids = [m.get("id") for m in data.get("data", [])
                   if isinstance(m, dict)]
            return self.model in ids
        except Exception:  # noqa: BLE001 — unknown, stated
            return None

    # ── capabilities ────────────────────────────────────

    def capabilities(self):
        """TF-065 contract record: simulated streaming (no
        generate_stream override — the shape's SSE dialects
        vary per server and none is assumed), estimate
        token counting (the shape guarantees no tokenise
        route), seed honoured via the shape's seed field,
        placement local_or_remote (the same binding serves
        a loopback server or a named remote host — which
        one is the config's act, stated by the URL)."""
        from thoughtforge.inference.backend_contract import (
            BackendCapabilities,
        )

        return BackendCapabilities(
            name=self.backend_name(),
            streaming="simulated",
            token_counting="estimate",
            seed="honoured",
            placement="local_or_remote",
            model=self.model,
            notes=(
                "health is server-level (/v1/models); the "
                "configured model's listing is "
                "model_listed()'s separate answer",
                "tokens_generated is the server's usage "
                "count when the response reports it, the "
                "contract estimate otherwise",
                "finish reasons outside stop/length are "
                "error data naming the server's reason — "
                "this backend sends no tools",
                "no Authorization header is sent when no "
                "api_key is configured",
            ),
        )


def openai_as_data() -> dict[str, Any]:
    return {
        "format": OPENAI_FORMAT,
        "shape": [dict(e) for e in OPENAI_SHAPE],
        "rules": [dict(r) for r in OPENAI_RULES],
        "known_servers": [dict(s) for s in KNOWN_SERVERS],
        "served_finish_reasons": list(SERVED_FINISH_REASONS),
    }


# ── Fake transport for the live probes ─────────────────────────


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _chat_payload(text="Four", finish="stop", usage=True):
    payload: dict[str, Any] = {
        "id": "chatcmpl-fake", "object": "chat.completion",
        "created": 1, "model": "fake-model",
        "choices": [{"index": 0,
                     "message": {"role": "assistant",
                                 "content": text},
                     "finish_reason": finish}],
    }
    if usage:
        payload["usage"] = {"prompt_tokens": 9,
                            "completion_tokens": 1,
                            "total_tokens": 10}
    return payload


class FakeOpenAIServer:
    """A callable fake transport speaking the shape:
    records every call, serves one chat payload and one
    model list. ``post`` / ``get`` match the requests
    call signatures this module uses."""

    def __init__(self, text="Four", finish="stop", usage=True,
                 models=("fake-model",), chat_status=200,
                 models_status=200, chat_payload=None):
        self.calls: list[dict[str, Any]] = []
        self._chat = (chat_payload if chat_payload is not None
                      else _chat_payload(text, finish, usage))
        self._chat_status = chat_status
        self._models_status = models_status
        self._models = {"object": "list",
                        "data": [{"id": m, "object": "model"}
                                 for m in models]}

    def post(self, url, **kwargs):
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return _FakeResponse(self._chat_status, self._chat)

    def get(self, url, **kwargs):
        self.calls.append({"method": "GET", "url": url, **kwargs})
        if url.endswith("/v1/models"):
            return _FakeResponse(self._models_status, self._models)
        return _FakeResponse(404, {"error": {"message": "no route"}})


# ── Checkers ───────────────────────────────────────────────────


def openai_problems() -> list[str]:
    """The filed estate judged against the code."""
    problems: list[str] = []

    for name, rows in (("OPENAI_SHAPE", OPENAI_SHAPE),
                       ("OPENAI_RULES", OPENAI_RULES),
                       ("KNOWN_SERVERS", KNOWN_SERVERS)):
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            problems.append(f"{name} ids are not unique")
    statuses = {s["id"]: s["status"] for s in KNOWN_SERVERS}
    for server, status in statuses.items():
        if status != "shape_bound_not_live_proven":
            problems.append(f"known server {server} no longer "
                            f"filed shape-only ({status})")
    shape = {e["id"]: e["status"] for e in OPENAI_SHAPE}
    if shape.get("sse_streaming") != "not_bound_simulated_claimed":
        problems.append("SSE streaming no longer filed as "
                        "not bound / simulated-claimed")
    if shape.get("tokenise_route") != \
            "not_served_estimate_declared":
        problems.append("tokenise route no longer filed as "
                        "not served")

    if not issubclass(OpenAICompatibleBackend, UnifiedBackend):
        problems.append(
            "OpenAICompatibleBackend is not a UnifiedBackend")
    if (OpenAICompatibleBackend.generate_stream
            is not UnifiedBackend.generate_stream):
        problems.append("OpenAICompatibleBackend overrides "
                        "generate_stream — its simulated claim "
                        "would be false")

    from thoughtforge.inference.backend_contract import (
        REQUESTS_CONSUMERS,
        capabilities_problems,
    )

    caps = OpenAICompatibleBackend().capabilities().to_dict()
    problems += [f"openai record: {p}"
                 for p in capabilities_problems(caps)]
    if "src/thoughtforge/inference/openai_compatible.py" not \
            in REQUESTS_CONSUMERS:
        problems.append("openai_compatible is not an announced "
                        "REQUESTS_CONSUMERS member (D-DEP-1 set)")

    # The family's shared parser has exactly one home: the
    # LM Studio preset must consume it, not keep a private
    # copy (D-DUP-1's rule, D-OAI-1's fix).
    from thoughtforge.inference import lmstudio_backend

    if lmstudio_backend.chat_completion_outcome is not \
            chat_completion_outcome:
        problems.append("LMStudioBackend does not use the "
                        "family's shared chat parser")
    if lmstudio_backend.normalize_base_url is not \
            normalize_base_url:
        problems.append("LMStudioBackend does not use the "
                        "family's shared base-URL normaliser")
    return problems


def openai_live_problems() -> list[str]:
    """The binding re-derived live over a fake shape-speaking
    transport (this VM)."""
    from unittest.mock import patch

    problems: list[str] = []
    post = "thoughtforge.inference.openai_compatible.requests.post"
    get = "thoughtforge.inference.openai_compatible.requests.get"

    server = FakeOpenAIServer()
    backend = OpenAICompatibleBackend(
        base_url="http://127.0.0.1:8000/v1", model="fake-model",
        api_key="test-key")
    req = GenerationRequest(prompt="What is two plus two?",
                            seed=7, stop=["\n\n"])
    with patch(post, side_effect=server.post):
        resp = backend.generate(req)
    if resp.text != "Four" or resp.tokens_generated != 1:
        problems.append(f"generate over the fake server gave "
                        f"{resp!r}")
    posts = [c for c in server.calls if c["method"] == "POST"]
    if len(posts) != 1 or posts[0]["url"] != \
            "http://127.0.0.1:8000/v1/chat/completions":
        problems.append(f"generate called {posts!r} (the /v1 "
                        f"base was not normalised)")
    else:
        payload = posts[0]["json"]
        if set(payload) != {"model", "messages", "temperature",
                            "max_tokens", "stream", "stop", "seed"}:
            problems.append(f"wire payload keys "
                            f"{sorted(payload)} are not the "
                            f"shape's sent set")
        if posts[0]["headers"].get("Authorization") != \
                "Bearer " + "test-key":
            problems.append("configured key not sent in the "
                            "Authorization header")

    # Finish vocabulary + null content, through the parser.
    for finish in ("content_filter", "tool_calls"):
        outcome = chat_completion_outcome(_chat_payload(
            text="x", finish=finish))
        if outcome["ok"] or finish not in outcome["error"]:
            problems.append(f"finish {finish} not refused as "
                            f"named error data: {outcome!r}")
    outcome = chat_completion_outcome({
        "choices": [{"message": {"content": None},
                     "finish_reason": "stop"}]})
    if outcome["ok"]:
        problems.append("null content accepted by the parser")
    outcome = chat_completion_outcome({"choices": []})
    if outcome["ok"] or "no choices" not in outcome["error"]:
        problems.append(f"empty choices outcome {outcome!r}")

    # Usage absent -> the declared estimate, not 0.
    server2 = FakeOpenAIServer(text="the ferry", usage=False)
    with patch(post, side_effect=server2.post):
        resp2 = backend.generate(
            GenerationRequest(prompt="the ferry?"))
    if resp2.tokens_generated != backend.count_tokens("the ferry"):
        problems.append(f"no-usage tokens {resp2.tokens_generated} "
                        f"!= the declared estimate")

    # Health + model listing, separately.
    with patch(get, side_effect=server.get):
        if backend.health_check() is not True:
            problems.append("health False on a serving fake")
        if backend.model_listed() is not True:
            problems.append("model_listed not True for the "
                            "listed model")
    absent = OpenAICompatibleBackend(
        base_url="http://127.0.0.1:8000", model="not-there")
    with patch(get, side_effect=server.get):
        if absent.model_listed() is not False:
            problems.append("model_listed not False for an "
                            "absent model")
    return problems


__all__ = [
    "KNOWN_SERVERS",
    "OPENAI_FORMAT",
    "OPENAI_RULES",
    "OPENAI_SHAPE",
    "SERVED_FINISH_REASONS",
    "FakeOpenAIServer",
    "OpenAICompatibleBackend",
    "chat_completion_outcome",
    "normalize_base_url",
    "openai_as_data",
    "openai_live_problems",
    "openai_problems",
    "server_error_message",
]
