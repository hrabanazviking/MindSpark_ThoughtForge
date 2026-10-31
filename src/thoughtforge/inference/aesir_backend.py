"""TF-074 — The Aesir backend: Project Aesir bound under
the TF-065 contract, its capability ledger respected.

Roadmap: "Project Aesir bound as a backend under TF-065's
contract (its capability ledger respected — what Aesir
proves, ThoughtForge may claim; no more)."

Measured start (probed this tick, VM):

- **No Aesir backend existed anywhere in the tree** — no
  module, no factory entry, no config key. The four shipped
  backends were ollama / lmstudio / huggingface /
  turboquant, and Aesir was named only as a road
  (TF-065's doc, TF-071's tolerance note).
- **No live Aesir server answers on this VM**
  (``127.0.0.1:18434`` refuses the connection): every proof
  in this slice is fixture/mock grade, and a live Aesir
  serve is NOT PROVEN — the same standing Rúnmál recorded
  (P-009/F-02: the owed live smoke is a purplelongship
  leg). Filed as D-AESIR-1, owned by TF-086's gate.
- The cognition core's requests carry exactly prompt /
  temperature / max_tokens / seed (probed in core.py) —
  ``stop`` is never set by the core, which matters because
  Aesir serves no stop anywhere (below).

WHAT AESIR SERVES — the ledger this binding respects
(``AESIR_LEDGER`` as data). Sources are Aesir's own
versioned statements, read 2026-10-31 in Aesir's repo
checkout on this VM (HEAD ``1eed9706`` — the same pin
Rúnmál's P-009 bound to): ``docs/API_SUPPORT_MATRIX_V1.md``
(the v1 supported subset, S18/S19/S23/S24),
``CAPABILITY_LEDGER.md`` (AES-SRV-006/008/009), and
``docs/NATIVE_SERVICE.md``; cross-checked against
Rúnmál's P-009 readiness ledger and its F-02 repair,
which closed the same divergences on Rúnmál's adapter.

- Chat: ``POST /v1/chat/completions`` in BOTH modes
  (native + ``--ollama``), a strict field subset —
  ``model``, ``messages``, ``stream``, ``max_tokens``,
  ``temperature``, ``top_p``, ``seed``, plus a bounded
  Boolean-only ``response_format`` and a bounded Gemma
  tool subset this backend never sends. ``stop``,
  ``logprobs`` and ``stream_options`` are REJECTED fields
  (unknown fields fail 400 before generation).
- The full sampler set (top_k / min_p / repeat_penalty
  included) lives only on the native ``POST /v1/generate``
  route — a text paradigm, not chat. This backend binds
  the chat route (ThoughtForge's requests are chat-shaped)
  and therefore claims only the chat subset; the native
  route is a recorded alternative, not bound here.
- ``GET /v1/models`` (both modes) lists the CATALOG —
  ``data[]`` carries only ``id`` (+ object/created/
  owned_by). Catalog visibility does not imply runnable:
  exactly ONE resident model can generate, a
  non-resident model returns 404, and requests are
  serialized through that one session.
- Readiness: ``GET /health/ready`` (native mode only;
  404 in ``--ollama`` mode) reports the RESIDENT model's
  readiness — the truthful health signal (Rúnmál D5).
- Ports/auth: native serve binds loopback ``127.0.0.1``
  port 18434 with a mandatory bearer key; ``--ollama``
  mode binds 11434 with no bearer. There is no
  external-listen option — placement is local, by
  Aesir's own design.
- Bounds: at most 128 messages, each decoded text field
  at most 64 KiB, POST body at most 131072 bytes, roles
  system/user/assistant with string content, the final
  message from the user. Context/generation admission
  failures return 422; native errors are shaped
  ``{"error": {"code": N, "message": "..."}}`` and never
  echo request data.
- Usage: the non-streamed chat JSON carries ``usage``;
  the SSE stream carries NO usage.
- Streaming: SSE ``data:`` events, role delta first,
  terminal choice with ``stop``/``length``, then exactly
  one ``data: [DONE]``. A post-header failure can end
  the stream with NO terminal marker — clients must
  treat EOF without the terminal as incomplete.
- NOT SERVED, and never claimed by this backend:
  embeddings (no route in either mode — "Embeddings
  remain Ollama", AES-SRV-006/008; ThoughtForge binds
  embeddings elsewhere, never to Aesir), vision /
  multimodal input (rejected on every generation
  route), general tool calls and general structured
  output (only the bounded subsets above exist, and
  this backend sends neither).

THE BINDING RULES (``AESIR_RULES`` as data), implemented
by :class:`AesirBackend`:

- ``bind_served_subset_only`` — the wire payload is
  exactly the served chat subset: model (when set),
  messages, ``stream``, ``max_tokens``, ``temperature``,
  and ``seed`` when the request carries one (the chat
  route honours it — TF-065's seed duty). No
  ``stream_options`` is ever sent (Rúnmál's D1: it broke
  every request against the real server).
- ``stop_refused_never_dropped`` — Aesir serves ``stop``
  on no route. A request carrying stop sequences is
  refused with an error response naming the refusal,
  before any HTTP call — never silently dropped
  (the TF-065 seed lesson, applied to stop) and never
  faked client-side.
- ``health_is_resident_readiness`` — native-mode
  ``health_check()`` is ``/health/ready``: True exactly
  on HTTP 200 with ``ready`` true in the body. In
  ``--ollama`` mode the health routes do not exist, so
  health is ``/v1/models`` reachability and is labelled
  as the weaker claim it is.
- ``catalog_is_not_readiness`` — ``list_models()``
  returns catalog ids; nothing in this backend treats
  a listing as proof a model can generate.
- ``admission_bounds_checked_before_sending`` — the
  128-message / 64 KiB-field / 131072-byte-body /
  role / final-user bounds are checked client-side and
  refused as error data before a doomed round trip.
- ``incomplete_stream_is_failure`` — the stream must
  end with a finish reason or ``[DONE]``; EOF without
  either raises (Rúnmál's D6 repair, TF-072's rule),
  and an SSE error event raises likewise.
- ``errors_are_data_with_status`` — every HTTP failure
  becomes a TF-065 error response (never a raise out of
  ``generate()``): the status and the server's own
  message (native shape parsed) are named; a 422 is
  named as context/generation admission.
- ``token_counting_stays_estimate`` — Aesir's v1
  surface has no tokenise route, so ``count_tokens``
  is the contract estimate, declared ``estimate``;
  ``tokens_generated`` on a response is the server's
  own ``usage.completion_tokens`` when Aesir reports
  it, the declared estimate otherwise.
- ``one_resident_model`` — the configured model is
  sent on every request (Aesir generates only with
  the resident one; a mismatch is Aesir's 404, surfaced
  as error data, never retried against another model).

Placement in the estate: like TF-073's FallbackBackend,
this backend is a ``UnifiedBackend`` conformed and
tested against the TF-065 contract directly, but it
does NOT join ``SHIPPED_BACKENDS`` or the config
factory — TF-153 owns the service/factory shape that
admits new backends, and TF-065's shipped set stays the
four it pinned. One deliberate contract amendment IS
made in scope: this module imports ``requests`` at top
level like its sibling backends, so it joins
``REQUESTS_CONSUMERS`` (D-DEP-1's set, amended on
purpose — the checker fails a consumer that arrives
unannounced, and this one is announced).

Proof grade FIXTURE (mocked HTTP legs in Aesir's own
v1 shapes — payloads, SSE streams, native error bodies,
readiness payloads — plus live contract checkers, VM)
— a live Aesir serve on the other end is NOT PROVEN
(D-AESIR-1; no server on this VM, and Aesir runs on
the laptop line), Aesir + TurboQuant interplay is
TF-075's, never claimed.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from typing import Any, Iterator

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

AESIR_FORMAT = "thoughtforge-aesir-backend/1"

#: Aesir's own admission bounds (v1 matrix + native service doc).
AESIR_BOUNDS: dict[str, Any] = {
    "max_messages": 128,
    "max_field_bytes": 64 * 1024,
    "max_body_bytes": 131072,
    "roles": ("system", "user", "assistant"),
}

#: The chat route's served sampler/field truth for the fields
#: a ThoughtForge GenerationRequest can carry.
AESIR_CHAT_FIELDS: tuple[dict[str, Any], ...] = (
    {"field": "temperature", "served": True,
     "wire": "temperature"},
    {"field": "max_tokens", "served": True, "wire": "max_tokens"},
    {"field": "seed", "served": True, "wire": "seed"},
    {"field": "stop", "served": False,
     "wire": None,
     "note": "rejected on the chat route and on the native "
             "/v1/generate route; refused client-side, never "
             "sent, never silently dropped"},
)

# ── The ledger, as data ────────────────────────────────────────

AESIR_LEDGER: tuple[dict[str, Any], ...] = (
    {"id": "chat_route_both_modes",
     "claim": "POST /v1/chat/completions is served in native "
              "and --ollama modes, strict field subset",
     "source": "Aesir API_SUPPORT_MATRIX_V1 (routes table)",
     "status": "served_bound"},
    {"id": "models_route_is_catalog",
     "claim": "GET /v1/models lists the catalog; data[] "
              "carries only id/object/created/owned_by; "
              "visibility does not imply runnable",
     "source": "Aesir API_SUPPORT_MATRIX_V1; CAPABILITY_LEDGER "
               "AES-SRV-006",
     "status": "served_bound"},
    {"id": "health_ready_native_only",
     "claim": "GET /health/ready (native mode only) reports "
              "the resident model's readiness, 200 ready / "
              "503 not_ready",
     "source": "Aesir API_SUPPORT_MATRIX_V1; NATIVE_SERVICE",
     "status": "served_bound"},
    {"id": "native_generate_full_samplers",
     "claim": "POST /v1/generate (native only) serves the "
              "full sampler set incl. top_k/min_p/"
              "repeat_penalty — a text paradigm this chat "
              "binding does not take",
     "source": "Aesir API_SUPPORT_MATRIX_V1 (accepted fields)",
     "status": "served_not_bound"},
    {"id": "one_resident_model",
     "claim": "exactly one resident model generates; "
              "requests are serialized; a non-resident model "
              "returns 404",
     "source": "Aesir NATIVE_SERVICE (bounds/threat model); "
               "API matrix",
     "status": "served_bound"},
    {"id": "ports_auth_loopback",
     "claim": "native serve: 127.0.0.1:18434, bearer key "
              "mandatory; --ollama mode: 127.0.0.1:11434, no "
              "bearer; no external-listen option exists",
     "source": "Aesir NATIVE_SERVICE (startup/auth)",
     "status": "served_bound"},
    {"id": "admission_bounds",
     "claim": "at most 128 messages, 64 KiB per text field, "
              "131072-byte POST body, final message from "
              "the user; admission failures return 422",
     "source": "Aesir API_SUPPORT_MATRIX_V1; NATIVE_SERVICE",
     "status": "served_bound"},
    {"id": "usage_in_json_not_stream",
     "claim": "the non-streamed chat JSON carries usage; "
              "the SSE stream carries no usage",
     "source": "Aesir API_SUPPORT_MATRIX_V1 (streaming state)",
     "status": "served_bound"},
    {"id": "stream_terminal_marker",
     "claim": "SSE ends with a terminal choice (stop/length) "
              "and exactly one [DONE]; a post-header failure "
              "can end the stream with neither",
     "source": "Aesir API_SUPPORT_MATRIX_V1 (S18/S19)",
     "status": "served_bound"},
    {"id": "embeddings_not_served",
     "claim": "no embeddings route exists in either mode; "
              "Aesir's own posture: embeddings remain Ollama",
     "source": "Aesir CAPABILITY_LEDGER AES-SRV-006/008; "
               "CURRENT_STATUS via Rúnmál P-009 D3",
     "status": "not_served"},
    {"id": "vision_rejected",
     "claim": "vision / multimodal input is rejected on "
              "every generation route",
     "source": "Aesir API_SUPPORT_MATRIX_V1 (accepted fields)",
     "status": "not_served"},
    {"id": "general_tools_not_served",
     "claim": "general tool calls / structured output are "
              "not served — only a bounded Gemma "
              "single-function subset and a Boolean-only "
              "response_format exist; this backend sends "
              "neither",
     "source": "Aesir API_SUPPORT_MATRIX_V1 (S23/S24)",
     "status": "not_served"},
)

AESIR_RULES: tuple[dict[str, Any], ...] = (
    {"id": "bind_served_subset_only",
     "rule": "the wire payload is exactly the served chat "
             "subset (model when set, messages, stream, "
             "max_tokens, temperature, seed when carried); "
             "no rejected field is ever sent",
     "measured_by": "captured payload keys pinned exactly; "
                    "no stream-options key in any payload"},
    {"id": "stop_refused_never_dropped",
     "rule": "a request carrying stop sequences is refused "
             "as error data naming the refusal, before any "
             "HTTP call — Aesir serves stop on no route",
     "measured_by": "a stop-carrying request returns an "
                    "error response and the transport records "
                    "zero calls"},
    {"id": "health_is_resident_readiness",
     "rule": "native health_check() is /health/ready (200 + "
             "ready true, exactly); --ollama-mode health is "
             "/v1/models reachability, labelled weaker",
     "measured_by": "mocked readiness pins over 200-ready / "
                    "200-not-ready / 503 / unreachable, and "
                    "the ollama-mode /v1/models leg"},
    {"id": "catalog_is_not_readiness",
     "rule": "list_models() returns catalog ids and is never "
             "used as a readiness or generability claim",
     "measured_by": "list returns ids while a readiness "
                    "check on the same fake server reports "
                    "not ready"},
    {"id": "admission_bounds_checked_before_sending",
     "rule": "128 messages / 64 KiB field / 131072-byte body "
             "/ role / final-user bounds are refused "
             "client-side as error data, before sending",
     "measured_by": "one test per bound: error response, "
                    "zero transport calls"},
    {"id": "incomplete_stream_is_failure",
     "rule": "a stream ending with neither a finish reason "
             "nor [DONE] raises; an SSE error event raises; "
             "a completed stream yields its chunks in order",
     "measured_by": "SSE fixture pins: complete stream, "
                    "EOF-without-terminal raises after the "
                    "yielded prefix, error event raises"},
    {"id": "errors_are_data_with_status",
     "rule": "HTTP failures return TF-065 error responses "
             "naming the status and the server's message "
             "(native error shape parsed); 422 is named "
             "context/generation admission; generate() "
             "never raises",
     "measured_by": "mocked 400/401/404/422/503 pins + a "
                    "raising transport leg"},
    {"id": "token_counting_stays_estimate",
     "rule": "count_tokens is the contract estimate (no "
             "tokenise route is served); tokens_generated "
             "is the server's usage count when reported, "
             "the declared estimate otherwise",
     "measured_by": "count equals the contract estimate on "
                    "the TF-071 corpus; usage and no-usage "
                    "response pins"},
    {"id": "one_resident_model",
     "rule": "the configured model is sent on every request; "
             "a 404 (non-resident) is surfaced as error data "
             "and never retried against another model",
     "measured_by": "the 404 pin records exactly one "
                    "transport call, to the configured model"},
    {"id": "live_proof_not_claimed",
     "rule": "no live Aesir serve has been on the other end "
             "of this backend; every proof here is fixture "
             "grade and the live smoke stays owed (D-AESIR-1)",
     "measured_by": "the doc and this rule state it; TF-086's "
                    "gate carries the NOT PROVEN line"},
)


def _server_error_message(resp: Any) -> str:
    """Aesir's error message, from its own shapes: native
    ``{"error": {"code": N, "message": "..."}}`` or the
    ollama-mode ``{"error": "..."}``. '' when neither parses
    (error bodies never echo request data, Aesir's rule)."""
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001 — judged, not hidden
        return ""
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or "")
    if isinstance(err, str):
        return err
    return ""


class AesirBackend(UnifiedBackend):
    """Project Aesir's native serve, bound as a ThoughtForge
    backend under the TF-065 contract (TF-074)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18434",
        model: str = "",
        api_key: str = "",
        timeout: int = 120,
        mode: str = "native",
    ) -> None:
        if mode not in ("native", "ollama"):
            raise ValueError(
                f"AesirBackend mode must be 'native' or 'ollama', "
                f"got {mode!r}")
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            # A /v1-rooted base (Rúnmál's convention) names the
            # same server; health routes live at the root.
            root = root[: -len("/v1")]
        self.base_url = root
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.mode = mode

    # ── Identity / plumbing ───────────────────────────────────

    def backend_name(self) -> str:
        return f"aesir:{self.model or 'resident'}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _admission_problem(
        self, messages: list[dict], payload: dict,
    ) -> str | None:
        """Aesir's admission bounds, checked before sending."""
        bounds = AESIR_BOUNDS
        if not messages:
            return "aesir admission: no messages to send"
        if len(messages) > bounds["max_messages"]:
            return (f"aesir admission: {len(messages)} messages "
                    f"exceeds Aesir's {bounds['max_messages']}-"
                    f"message bound")
        for msg in messages:
            if msg.get("role") not in bounds["roles"]:
                return (f"aesir admission: role "
                        f"{msg.get('role')!r} is not one of "
                        f"Aesir's {list(bounds['roles'])}")
            content = msg.get("content")
            if not isinstance(content, str):
                return ("aesir admission: message content must "
                        "be a string (multimodal content is "
                        "rejected by Aesir)")
            if len(content.encode("utf-8")) > bounds["max_field_bytes"]:
                return ("aesir admission: a message field "
                        "exceeds Aesir's 64 KiB bound")
        if messages[-1].get("role") != "user":
            return ("aesir admission: the final message must "
                    "be from the user")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(body) > bounds["max_body_bytes"]:
            return (f"aesir admission: request body of "
                    f"{len(body)} bytes exceeds Aesir's "
                    f"{bounds['max_body_bytes']}-byte bound")
        return None

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict:
        payload: dict[str, Any] = {
            "messages": _build_messages(request),
            "stream": stream,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if self.model:
            payload["model"] = self.model
        if request.seed is not None:
            # TF-065 seed duty: the chat route accepts seed —
            # forward the carried seed, and only then.
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

    def _http_error(self, resp: Any) -> str:
        message = _server_error_message(resp)
        label = (f"aesir HTTP {resp.status_code}")
        if resp.status_code == 422:
            label += " (context/generation admission)"
        elif resp.status_code == 404:
            label += " (model not resident)"
        return f"{label}: {message}" if message else label

    # ── generate ──────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.monotonic()
        if request.stop:
            # Aesir serves stop on no route; refusing honestly
            # beats silently dropping it (AESIR_RULES).
            return self._error_response(
                start,
                "aesir: stop sequences are not served by "
                "Aesir's chat route (rejected field) — request "
                "refused, not silently dropped")
        payload = self._payload(request, stream=False)
        problem = self._admission_problem(payload["messages"], payload)
        if problem:
            return self._error_response(start, problem)
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return self._error_response(start, self._http_error(resp))
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            text: str = (choice.get("message") or {}).get("content") or ""
            finish: str = choice.get("finish_reason") or "stop"
            if finish not in ("stop", "length"):
                return self._error_response(
                    start,
                    f"aesir: unexpected finish_reason {finish!r} "
                    f"(this backend sends no tools; the served "
                    f"terminal reasons are stop/length)")
            usage = data.get("usage") or {}
            tokens = usage.get("completion_tokens")
            if tokens is None:
                # Declared fallback (capabilities note): the
                # contract estimate, never dressed as a
                # server count.
                tokens = self.count_tokens(text)
            return GenerationResponse(
                text=text,
                tokens_generated=int(tokens),
                finish_reason=finish,
                backend_used=self.backend_name(),
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 — failure is data
            logger.error("AesirBackend.generate error: %s", exc)
            return self._error_response(start, str(exc))

    # ── streaming (native: the SSE leg) ───────────────────────

    def generate_stream(self, request: GenerationRequest) -> Iterator[str]:
        if request.stop:
            raise RuntimeError(
                "aesir: stop sequences are not served by "
                "Aesir's chat route (rejected field) — request "
                "refused, not silently dropped")
        payload = self._payload(request, stream=True)
        problem = self._admission_problem(payload["messages"], payload)
        if problem:
            raise RuntimeError(problem)
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
            stream=True,
        )
        if resp.status_code != 200:
            raise RuntimeError(self._http_error(resp))
        saw_terminal = False
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                saw_terminal = True
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("error"):
                err = obj["error"]
                detail = (err.get("message") if isinstance(err, dict)
                          else err)
                raise RuntimeError(f"aesir stream error: {detail}")
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if content:
                yield content
            if choice.get("finish_reason"):
                saw_terminal = True
        if not saw_terminal:
            # Aesir's own instruction (S18/S19): EOF without
            # the terminal is incomplete, never a success.
            raise RuntimeError(
                "aesir stream ended without a terminal marker "
                "(no finish reason, no [DONE]) — incomplete, "
                "not a successful generation")

    # ── health / readiness / catalog ──────────────────────────

    def health_check(self) -> bool:
        try:
            if self.mode == "ollama":
                # The native health routes 404 in this mode;
                # reachability of the models route is the
                # weaker claim, and it is labelled as such.
                resp = requests.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(), timeout=5)
                return resp.status_code == 200
            resp = requests.get(
                f"{self.base_url}/health/ready",
                headers=self._headers(), timeout=5)
            if resp.status_code != 200:
                return False
            data = resp.json()
            return isinstance(data, dict) and data.get("ready") is True
        except Exception:  # noqa: BLE001 — health never raises
            return False

    def readiness_check(self) -> dict[str, Any]:
        """The resident-model readiness report (native mode).

        Returns Aesir's own ``/health/ready`` payload with
        the HTTP status; ``ready`` is None when the server
        could not be asked or the mode does not serve the
        route — unknown, never dressed as not-ready.
        """
        if self.mode != "native":
            return {
                "format": AESIR_FORMAT, "served": False,
                "ready": None,
                "reason": "/health/ready is native-mode only "
                          "(404 in --ollama mode)",
            }
        try:
            resp = requests.get(
                f"{self.base_url}/health/ready",
                headers=self._headers(), timeout=5)
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001 — judged below
                payload = {}
            ready = (payload.get("ready")
                     if isinstance(payload, dict) else None)
            return {
                "format": AESIR_FORMAT, "served": True,
                "http_status": resp.status_code,
                "ready": ready if isinstance(ready, bool) else None,
                "payload": payload if isinstance(payload, dict) else {},
            }
        except Exception as exc:  # noqa: BLE001 — unknown, stated
            return {
                "format": AESIR_FORMAT, "served": True,
                "ready": None, "error": str(exc),
            }

    def model_listed(self) -> bool | None:
        """TF-077 (D-HEALTH-2): is the configured model in
        the server's /v1/models catalog? Tri-state,
        mirroring TF-076's surface: None when no model is
        configured or the server could not be asked. A
        True here is CATALOG membership only — the catalog
        is not readiness (AESIR_LEDGER); the native
        readiness_check() remains the stronger check."""
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

    def list_models(self) -> list[str]:
        """Catalog ids (NOT readiness — AESIR_LEDGER)."""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])
                    if isinstance(m, dict) and "id" in m]
        except Exception as exc:  # noqa: BLE001 — stated, empty
            logger.warning("AesirBackend.list_models failed: %s", exc)
            return []

    # ── capabilities ──────────────────────────────────────────

    def capabilities(self):
        """TF-065 contract record: native SSE streaming,
        estimate token counting (Aesir v1 serves no tokenise
        route), seed honoured via the chat route's seed
        field, placement local (loopback only, by Aesir's
        own design)."""
        from thoughtforge.inference.backend_contract import (
            BackendCapabilities,
        )

        return BackendCapabilities(
            name=self.backend_name(),
            streaming="native",
            token_counting="estimate",
            seed="honoured",
            placement="local",
            model=self.model,
            notes=(
                "one resident model serves, serialized; the "
                "/v1/models catalog is not readiness",
                "tokens_generated is Aesir's usage count when "
                "the response reports it, the contract "
                "estimate otherwise",
                "stop is not served on any Aesir route — a "
                "request carrying it is refused, never "
                "silently dropped",
                "embeddings are not served by Aesir and are "
                "never bound to this backend",
            ),
        )


def aesir_as_data() -> dict[str, Any]:
    return {
        "format": AESIR_FORMAT,
        "ledger": [dict(e) for e in AESIR_LEDGER],
        "rules": [dict(r) for r in AESIR_RULES],
        "chat_fields": [dict(f) for f in AESIR_CHAT_FIELDS],
        "bounds": dict(AESIR_BOUNDS),
    }


# ── Fake transport for the live probes ─────────────────────────


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, lines=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._lines = lines or []

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


def _chat_payload(text="Four", finish="stop", usage=True):
    payload: dict[str, Any] = {
        "id": "chatcmpl-aesir", "object": "chat.completion",
        "created": 1, "model": "gemma4-e2b:latest",
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


# ── Checkers ───────────────────────────────────────────────────


def aesir_problems() -> list[str]:
    """The filed estate judged against the code (no live runs)."""
    problems: list[str] = []

    for name, rows in (("AESIR_LEDGER", AESIR_LEDGER),
                       ("AESIR_RULES", AESIR_RULES)):
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            problems.append(f"{name} ids are not unique")
    statuses = {e["id"]: e["status"] for e in AESIR_LEDGER}
    for not_served in ("embeddings_not_served", "vision_rejected",
                       "general_tools_not_served"):
        if statuses.get(not_served) != "not_served":
            problems.append(f"ledger no longer files {not_served} "
                            f"as not_served")
    stop = next(f for f in AESIR_CHAT_FIELDS if f["field"] == "stop")
    if stop["served"] is not False:
        problems.append("chat field table no longer files stop "
                        "as not served")

    if not issubclass(AesirBackend, UnifiedBackend):
        problems.append("AesirBackend is not a UnifiedBackend")
    if (AesirBackend.generate_stream
            is UnifiedBackend.generate_stream):
        problems.append("AesirBackend does not override "
                        "generate_stream (its native claim is false)")

    from thoughtforge.inference.backend_contract import (
        REQUESTS_CONSUMERS,
        capabilities_problems,
    )

    caps = AesirBackend().capabilities().to_dict()
    problems += [f"aesir record: {p}"
                 for p in capabilities_problems(caps)]
    if "src/thoughtforge/inference/aesir_backend.py" not \
            in REQUESTS_CONSUMERS:
        problems.append("aesir_backend is not an announced "
                        "REQUESTS_CONSUMERS member (D-DEP-1 set)")

    src = inspect.getsource(AesirBackend)
    if "/v1/chat/completions" not in src:
        problems.append("backend no longer posts the chat route")
    if "/health/ready" not in src:
        problems.append("backend no longer reads /health/ready")
    if "stream_options" in src:
        problems.append("the backend sends stream_options — "
                        "a rejected field (Rúnmál D1 class)")
    # The payload builder's behaviour, judged directly: even a
    # stop-carrying request produces a payload with no stop
    # key (generate() refuses such requests before sending;
    # the builder itself must never add the rejected field).
    payload = AesirBackend()._payload(
        GenerationRequest(prompt="x", stop=["y"]), stream=False)
    if "stop" in payload or "stream_options" in payload:
        problems.append("the payload builder adds a rejected "
                        "field (stop/stream_options)")
    return problems


def aesir_live_problems() -> list[str]:
    """The binding re-derived live over a fake Aesir transport
    speaking Aesir's own v1 shapes (this VM)."""
    from unittest.mock import patch

    problems: list[str] = []
    req = GenerationRequest(prompt="What is two plus two?",
                            seed=7)

    calls: list[dict[str, Any]] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _FakeResponse(200, _chat_payload())

    backend = AesirBackend(model="gemma4-e2b:latest",
                           api_key="test-key")
    with patch("thoughtforge.inference.aesir_backend.requests.post",
               side_effect=fake_post):
        resp = backend.generate(req)
    if resp.text != "Four" or resp.tokens_generated != 1:
        problems.append(f"generate over fake Aesir gave {resp!r}")
    if len(calls) != 1 or calls[0]["url"] != \
            "http://127.0.0.1:18434/v1/chat/completions":
        problems.append(f"generate called {calls!r}")
    else:
        payload = calls[0]["json"]
        if set(payload) != {"model", "messages", "stream",
                            "max_tokens", "temperature", "seed"}:
            problems.append(f"wire payload keys {sorted(payload)} "
                            f"are not the served subset")
        if calls[0]["headers"].get("Authorization") != \
                "Bearer test-key":
            problems.append("bearer key not sent in native mode")

    # Stop refusal: no transport call at all.
    calls.clear()
    with patch("thoughtforge.inference.aesir_backend.requests.post",
               side_effect=fake_post):
        refused = backend.generate(
            GenerationRequest(prompt="hi", stop=["\n\n"]))
    if not refused.error or "stop" not in refused.error:
        problems.append("stop-carrying request was not refused")
    if calls:
        problems.append("stop refusal still made a transport call")

    # Readiness: ready exactly on 200 + ready true.
    def fake_get_ready(url, **kwargs):
        return _FakeResponse(200, {"ready": True,
                                   "status": "ready",
                                   "model": "gemma4-e2b:latest"})

    with patch("thoughtforge.inference.aesir_backend.requests.get",
               side_effect=fake_get_ready):
        if backend.health_check() is not True:
            problems.append("health_check False on a ready server")
        report = backend.readiness_check()
    if report.get("ready") is not True:
        problems.append(f"readiness_check gave {report!r}")

    # Streaming: a complete SSE stream yields its text; an
    # unterminated one raises.
    sse = [
        'data: {"choices":[{"delta":{"role":"assistant"},'
        '"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"Fo"},'
        '"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"ur"},'
        '"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    with patch("thoughtforge.inference.aesir_backend.requests.post",
               return_value=_FakeResponse(200, lines=sse)):
        text = "".join(backend.generate_stream(req))
    if text != "Four":
        problems.append(f"streamed text {text!r} != 'Four'")
    with patch("thoughtforge.inference.aesir_backend.requests.post",
               return_value=_FakeResponse(200, lines=sse[:2])):
        try:
            "".join(backend.generate_stream(req))
            problems.append("unterminated stream did not raise")
        except RuntimeError as exc:
            if "terminal" not in str(exc):
                problems.append(f"unterminated stream raised {exc!r}")
    return problems


__all__ = [
    "AESIR_BOUNDS",
    "AESIR_CHAT_FIELDS",
    "AESIR_FORMAT",
    "AESIR_LEDGER",
    "AESIR_RULES",
    "AesirBackend",
    "aesir_as_data",
    "aesir_live_problems",
    "aesir_problems",
]
