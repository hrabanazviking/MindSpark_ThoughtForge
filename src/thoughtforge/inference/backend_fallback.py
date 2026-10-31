"""TF-073 — Backend health & fallback: the stated policy.

Roadmap: "Backend health & fallback: a dead backend
degrades to the next per the stated policy, announced
honestly."

Measured start (probed this tick, VM):

- **No fallback existed anywhere.** The core holds exactly
  one ``UnifiedBackend`` (``ThoughtForgeCore._unified_backend``);
  when it answers with an error, TF-031 records the failure
  and the turn degrades to knowledge-only. Nothing in the
  tree could try a second backend — there was no chain, no
  order, no policy, stated or otherwise.
- **Health existed but meant four different things**, one
  per backend, none of them stated: HF asked its per-model
  info endpoint but accepted *any* status below 500 (probe:
  401 and 404 both returned True — a model that does not
  exist, or credentials the host rejects, counted as
  "healthy"); ollama and lmstudio check a server-level
  endpoint and never look at whether the configured model
  is present or loaded; TurboQuant checks that the model
  file exists and llama_cpp is importable, without loading.
- **D-ORPHAN-2**: ``utils/retry.py`` (CircuitBreaker /
  retry) protected no production call — grep found it
  referenced only by its own module and the phase-8 tests.

THE POLICY (``FALLBACK_RULES`` as data), implemented by
:class:`FallbackBackend` — a ``UnifiedBackend`` wrapping an
ordered chain, so any consumer that can hold one backend
(the core included, unchanged) can hold the chain:

- ``order_is_the_policy`` — the chain is tried strictly in
  the caller's order; nothing reorders it. The first
  backend that serves, serves.
- ``health_gate_before_generate`` — before a backend is
  asked to generate, its ``health_check()`` must return
  True; a False (or, defensively, a raising) health check
  skips it without a generate call, reason recorded.
- ``error_response_is_failure`` — a response carrying an
  error (finish_reason ``error`` or a non-empty error
  string, TF-065's failure-as-data shape) is a failure:
  the next backend is tried and the failed response's
  text is never presented as an answer.
- ``empty_response_is_failure`` — a response with no
  usable text is a failure too (TF-031's ruling in the
  core: an empty completion answered nothing), reason
  recorded, next backend tried.
- ``exception_is_failure_data`` — a backend whose
  ``generate()`` raises (a TF-065 contract breach) is
  caught, recorded, and the next backend tried; the
  wrapper's own ``generate()`` never raises.
- ``circuit_breaker_skips_known_dead`` — each backend
  carries its own ``utils/retry.CircuitBreaker``
  (D-ORPHAN-2's production wiring: the breaker, not the
  exception-shaped ``retry`` decorator, fits a contract
  whose failures are data). Generate outcomes are
  recorded on it; at the threshold the breaker opens and
  the backend is skipped *without* a health or generate
  call — reason ``circuit_open`` — until the recovery
  timeout lets the breaker's own half-open probe through.
- ``announced_never_silent`` — every skip and failure,
  and which backend finally served, is recorded on the
  wrapper's ``last_report`` (structured, see
  :class:`FallbackReport`) and logged. A success served
  by a fallback is a *degraded* success and the report
  says so; the served response itself is the serving
  backend's own, unmodified.
- ``all_failed_is_stated_whole`` — when every backend
  fails, the wrapper returns an error response whose
  error string names every backend tried and its reason
  ("all backends failed: a: …; b: …"), ``backend_used``
  is the wrapper's name, ``finish_reason`` is ``error``,
  and the text is empty — never one backend's error
  dressed as the whole story, never fabricated text.
- ``stream_fallback_before_first_chunk_only`` — streaming
  follows the same gates, but a fallback may only take
  over *before the first chunk is yielded*: once a chunk
  has been delivered the stream belongs to that backend
  (a later failure propagates to the caller, TF-072's
  rule — the caller decides what a partial means). A
  stream that yields nothing, or raises before its first
  chunk, is a failure and the next backend is tried.
- ``chain_health_is_any_member`` — the wrapper's own
  ``health_check()`` is True exactly when at least one
  non-circuit-open member reports healthy, checked in
  chain order, short-circuiting at the first True; it
  never raises.

HEALTH SEMANTICS (``HEALTH_SEMANTICS`` as data, judged
live against each shipped class's source): what each
backend's ``health_check()`` actually proves — and what
it does not. The per-model gap for the two server-shaped
backends is filed as D-HEALTH-2, owned by TF-077 (the
routing slice that will consume per-model health).

FIXED in scope:

- **D-HEALTH-1** — ``HuggingFaceBackend.health_check``
  accepted any status < 500: 401 (rejected credentials)
  and 404 (no such model) both read as healthy. It now
  returns True exactly on 200, the per-model info
  endpoint's success.
- **D-ORPHAN-2** — the ``CircuitBreaker`` now protects
  production backend calls through this wrapper (one
  breaker per chain member, outcomes recorded on every
  generate). The ``retry`` decorator itself stays
  unused in production, stated plainly: it retries
  *exceptions*, and backend failure under the TF-065
  contract is data, not an exception — wiring it here
  would have meant converting data back into raises.

Proof grade FIXTURE (stub backends over the real wrapper,
the real CircuitBreaker, and mocked HTTP for the shipped
health checks, VM) — a live server's death-and-degradation
in production is NOT PROVEN (no live backend servers on
this VM), and which chain a deployment should run is
TF-077's routing table, never claimed.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Iterator

from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.utils.retry import CircuitBreaker

logger = logging.getLogger(__name__)

FALLBACK_FORMAT = "thoughtforge-backend-fallback/1"

#: Outcomes an attempt can end with, in the report.
ATTEMPT_OUTCOMES = (
    "served",
    "error",
    "empty",
    "exception",
    "health_false",
    "health_raised",
    "circuit_open",
    "stream_error",
    "stream_empty",
)

FALLBACK_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "order_is_the_policy",
        "rule": "the chain is tried strictly in the caller's order; "
                "no health history, capability, or cost reorders it — "
                "the first backend that serves, serves",
        "measured_by": "call-order pins: a healthy primary's "
                       "fallbacks are never called, and a degraded "
                       "chain serves from the first working member",
    },
    {
        "id": "health_gate_before_generate",
        "rule": "a backend is asked to generate only after its "
                "health_check() returns True; False — or a health "
                "check that raises, defensively — skips it without "
                "a generate call, reason recorded",
        "measured_by": "a health-False stub records generate() "
                       "called 0 times and outcome health_false",
    },
    {
        "id": "error_response_is_failure",
        "rule": "a response with finish_reason 'error' or a "
                "non-empty error string is a failure (TF-065's "
                "failure-as-data); the next backend is tried and "
                "the failed response's text is never the answer",
        "measured_by": "an error stub carrying decoy text is "
                       "passed over; the fallback's text serves",
    },
    {
        "id": "empty_response_is_failure",
        "rule": "a response with no usable (non-whitespace) text "
                "answered nothing (TF-031) and is a failure, "
                "reason recorded, next backend tried",
        "measured_by": "an empty-text stub is passed over with "
                       "outcome empty",
    },
    {
        "id": "exception_is_failure_data",
        "rule": "a generate() that raises (a TF-065 breach) is "
                "caught and recorded as the attempt's failure; "
                "the wrapper's generate() itself never raises",
        "measured_by": "a raising stub is passed over with "
                       "outcome exception and the chain serves",
    },
    {
        "id": "circuit_breaker_skips_known_dead",
        "rule": "each member carries a utils/retry CircuitBreaker "
                "fed by its generate outcomes; once open, the "
                "member is skipped without a health or generate "
                "call (outcome circuit_open) until the breaker's "
                "recovery timeout allows its half-open probe",
        "measured_by": "after threshold consecutive failures the "
                       "stub's health+generate call counts stop "
                       "growing while the chain keeps serving",
    },
    {
        "id": "announced_never_silent",
        "rule": "every skip and failure, and the backend that "
                "served, is recorded in last_report and logged; a "
                "fallback-served success is marked degraded in "
                "the report — never a silent substitution",
        "measured_by": "last_report after a degraded generate "
                       "names each attempt's outcome and "
                       "degraded == True",
    },
    {
        "id": "all_failed_is_stated_whole",
        "rule": "when every member fails, the returned error "
                "response names every backend and its reason, "
                "backend_used is the wrapper's name, text is "
                "empty — no single failure dressed as the whole "
                "story, no fabricated text",
        "measured_by": "an all-dead chain's error string contains "
                       "every member's name and reason",
    },
    {
        "id": "stream_fallback_before_first_chunk_only",
        "rule": "streaming uses the same gates, but takeover is "
                "allowed only before the first chunk is yielded; "
                "once a chunk is delivered the stream is that "
                "backend's (a later failure propagates, TF-072); "
                "a stream yielding nothing, or raising before "
                "its first chunk, is a failure",
        "measured_by": "an empty-stream primary falls back and "
                       "the fallback's chunks serve whole; a "
                       "mid-stream failure after a chunk raises "
                       "with the fallback never called",
    },
    {
        "id": "chain_health_is_any_member",
        "rule": "the wrapper's health_check() is True exactly "
                "when at least one non-circuit-open member "
                "reports healthy, in order, short-circuiting; "
                "it never raises",
        "measured_by": "health pins over stub chains: first-True "
                       "short-circuits, all-False is False, a "
                       "raising member counts as False",
    },
)

#: What each shipped backend's health_check() proves, filed
#: as data and judged live against the classes' sources.
HEALTH_SEMANTICS: tuple[dict[str, Any], ...] = (
    {
        "backend": "huggingface",
        "check": "per-model info endpoint status",
        "proves": "the named model exists and the host answers "
                  "200 for it (D-HEALTH-1: exactly 200, never "
                  "any status below 500)",
        "does_not_prove": "that an inference call will succeed "
                          "(the model may still be loading — "
                          "the backend's own 503 retry leg owns "
                          "that)",
    },
    {
        "backend": "ollama",
        "check": "server /api/tags status",
        "proves": "the server answers 200",
        "does_not_prove": "that the configured model is pulled "
                          "(D-HEALTH-2, TF-077)",
    },
    {
        "backend": "lmstudio",
        "check": "server /v1/models status",
        "proves": "the server answers 200",
        "does_not_prove": "that the configured model is loaded "
                          "(D-HEALTH-2, TF-077)",
    },
    {
        "backend": "turboquant",
        "check": "model file exists + llama_cpp importable",
        "proves": "the file is present and the engine library "
                  "can be imported",
        "does_not_prove": "that the model loads (loading is "
                          "lazy, at first generate)",
    },
)


@dataclass(frozen=True)
class FallbackAttempt:
    backend: str
    outcome: str          # one of ATTEMPT_OUTCOMES
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "outcome": self.outcome,
                "detail": self.detail}


@dataclass(frozen=True)
class FallbackReport:
    attempts: tuple[FallbackAttempt, ...] = ()
    served_by: str = ""
    degraded: bool = False
    all_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FALLBACK_FORMAT,
            "attempts": [a.to_dict() for a in self.attempts],
            "served_by": self.served_by,
            "degraded": self.degraded,
            "all_failed": self.all_failed,
        }


class FallbackBackend(UnifiedBackend):
    """An ordered chain of backends under the TF-073 policy.

    The wrapper is itself a ``UnifiedBackend``: consumers
    (the cognition core included) hold it exactly as they
    hold a single backend, unchanged.
    """

    def __init__(
        self,
        backends: list[UnifiedBackend],
        *,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ) -> None:
        if not backends:
            raise ValueError("FallbackBackend needs at least one backend")
        self._backends = list(backends)
        self._breakers = [
            CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
            for _ in self._backends
        ]
        self.last_report = FallbackReport()

    # ── Identity / introspection ──────────────────────────────

    def backend_name(self) -> str:
        return "fallback:" + "+".join(
            b.backend_name() for b in self._backends)

    def chain_names(self) -> list[str]:
        return [b.backend_name() for b in self._backends]

    def breaker_states(self) -> dict[str, str]:
        return {
            b.backend_name(): br.state.value
            for b, br in zip(self._backends, self._breakers)
        }

    # ── The gates ─────────────────────────────────────────────

    def _gate(self, idx: int, attempts: list[FallbackAttempt]) -> bool:
        """Circuit + health gates. True = may generate."""
        backend = self._backends[idx]
        name = backend.backend_name()
        if not self._breakers[idx].allow_request():
            attempts.append(FallbackAttempt(name, "circuit_open"))
            logger.warning("FallbackBackend: %s skipped — circuit open", name)
            return False
        try:
            healthy = backend.health_check()
        except Exception as exc:  # noqa: BLE001 — defensive, contract says never
            attempts.append(FallbackAttempt(name, "health_raised", str(exc)))
            logger.warning(
                "FallbackBackend: %s health_check raised %s — skipped",
                name, exc)
            return False
        if not healthy:
            attempts.append(FallbackAttempt(name, "health_false"))
            logger.warning(
                "FallbackBackend: %s health_check False — skipped", name)
            return False
        return True

    def _finish(
        self, attempts: list[FallbackAttempt], served_by: str = "",
    ) -> FallbackReport:
        first = self._backends[0].backend_name()
        report = FallbackReport(
            attempts=tuple(attempts),
            served_by=served_by,
            degraded=bool(served_by) and served_by != first,
            all_failed=not served_by,
        )
        self.last_report = report
        return report

    # ── generate ──────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        attempts: list[FallbackAttempt] = []
        for idx, backend in enumerate(self._backends):
            name = backend.backend_name()
            if not self._gate(idx, attempts):
                continue
            try:
                resp = backend.generate(request)
            except Exception as exc:  # noqa: BLE001 — failure is data here
                self._breakers[idx].record_failure()
                attempts.append(FallbackAttempt(name, "exception", str(exc)))
                logger.warning(
                    "FallbackBackend: %s generate raised %s — degrading",
                    name, exc)
                continue
            if resp.error or resp.finish_reason == "error":
                self._breakers[idx].record_failure()
                attempts.append(
                    FallbackAttempt(name, "error", resp.error or "error"))
                logger.warning(
                    "FallbackBackend: %s error %r — degrading to next",
                    name, resp.error)
                continue
            if not resp.text or not resp.text.strip():
                self._breakers[idx].record_failure()
                attempts.append(FallbackAttempt(name, "empty"))
                logger.warning(
                    "FallbackBackend: %s returned no text — degrading",
                    name)
                continue
            self._breakers[idx].record_success()
            attempts.append(FallbackAttempt(name, "served"))
            report = self._finish(attempts, served_by=name)
            if report.degraded:
                logger.warning(
                    "FallbackBackend: degraded — served by %s after %s",
                    name,
                    [(a.backend, a.outcome) for a in attempts[:-1]],
                )
            return resp
        report = self._finish(attempts)
        error = "all backends failed: " + "; ".join(
            f"{a.backend}: {a.outcome}"
            + (f" ({a.detail})" if a.detail else "")
            for a in report.attempts
        )
        logger.error("FallbackBackend: %s", error)
        return GenerationResponse(
            text="",
            error=error,
            finish_reason="error",
            backend_used=self.backend_name(),
        )

    # ── streaming ─────────────────────────────────────────────

    def generate_stream(self, request: GenerationRequest) -> Iterator[str]:
        attempts: list[FallbackAttempt] = []
        for idx, backend in enumerate(self._backends):
            name = backend.backend_name()
            if not self._gate(idx, attempts):
                continue
            delivered = False
            try:
                for chunk in backend.generate_stream(request):
                    if not delivered:
                        delivered = True
                        attempts.append(FallbackAttempt(name, "served"))
                        self._breakers[idx].record_success()
                        self._finish(attempts, served_by=name)
                    yield chunk
            except Exception as exc:  # noqa: BLE001 — judged by delivery
                if delivered:
                    # TF-072: the stream is this backend's now;
                    # the caller decides what a partial means.
                    raise
                self._breakers[idx].record_failure()
                attempts.append(
                    FallbackAttempt(name, "stream_error", str(exc)))
                logger.warning(
                    "FallbackBackend: %s stream failed before its "
                    "first chunk (%s) — degrading", name, exc)
                continue
            if delivered:
                return
            self._breakers[idx].record_failure()
            attempts.append(FallbackAttempt(name, "stream_empty"))
            logger.warning(
                "FallbackBackend: %s stream yielded nothing — degrading",
                name)
        self._finish(attempts)

    # ── health / tokens / capabilities ────────────────────────

    def health_check(self) -> bool:
        for idx, backend in enumerate(self._backends):
            if not self._breakers[idx].allow_request():
                continue
            try:
                if backend.health_check():
                    return True
            except Exception:  # noqa: BLE001 — a member's health never raises here
                continue
        return False

    def count_tokens(self, text: str) -> int:
        """Counted by the primary: the chain's stated basis is
        the first member's (its record may change with load —
        the primary's own capability record says which)."""
        return self._backends[0].count_tokens(text)

    def capabilities(self):
        """The primary's record under the wrapper's name, with
        the chain announced in the notes — the serving member
        may differ per call, and last_report says which one
        actually served; the record never claims a member's
        powers the primary does not have."""
        from thoughtforge.inference.backend_contract import BackendCapabilities

        primary = self._backends[0].capabilities()
        return BackendCapabilities(
            name=self.backend_name(),
            generate=primary.generate,
            streaming=primary.streaming,
            token_counting=primary.token_counting,
            seed=primary.seed,
            health=primary.health,
            placement=primary.placement,
            model=primary.model,
            notes=tuple(primary.notes) + (
                "TF-073 fallback chain, in order: "
                + " -> ".join(self.chain_names())
                + "; a degraded serve is announced in last_report",
            ),
        )


def fallback_as_data() -> dict[str, Any]:
    return {
        "format": FALLBACK_FORMAT,
        "rules": [dict(r) for r in FALLBACK_RULES],
        "attempt_outcomes": list(ATTEMPT_OUTCOMES),
        "health_semantics": [dict(s) for s in HEALTH_SEMANTICS],
    }


# ── Stubs for the live probes ──────────────────────────────────


def _stub_chain() -> FallbackBackend:
    """dead (health) -> erroring -> serving, the canonical probe."""
    from thoughtforge.inference.unified_backend import (
        GenerationResponse as _Resp,
    )

    class Stub(UnifiedBackend):
        def __init__(self, name, *, health=True, mode="serve", text=""):
            self._name = name
            self._health = health
            self._mode = mode
            self._text = text
            self.generate_calls = 0

        def generate(self, request):
            self.generate_calls += 1
            if self._mode == "raise":
                raise RuntimeError(f"{self._name} exploded")
            if self._mode == "error":
                return _Resp(text="decoy", error=f"{self._name} is dead",
                             finish_reason="error",
                             backend_used=self._name)
            if self._mode == "empty":
                return _Resp(text="", backend_used=self._name)
            return _Resp(text=self._text or f"text from {self._name}",
                         backend_used=self._name)

        def health_check(self):
            return self._health

        def backend_name(self):
            return self._name

    return FallbackBackend([
        Stub("dead:1", health=False),
        Stub("erroring:2", mode="error"),
        Stub("serving:3"),
    ])


# ── Checkers ───────────────────────────────────────────────────


def fallback_problems() -> list[str]:
    """The filed estate judged against the code (no live runs)."""
    problems: list[str] = []

    ids = [r["id"] for r in FALLBACK_RULES]
    if len(ids) != len(set(ids)):
        problems.append("FALLBACK_RULES ids are not unique")
    if [s["backend"] for s in HEALTH_SEMANTICS] != [
            "huggingface", "ollama", "lmstudio", "turboquant"]:
        problems.append("HEALTH_SEMANTICS does not cover the four "
                        "shipped backends in order")

    if not issubclass(FallbackBackend, UnifiedBackend):
        problems.append("FallbackBackend is not a UnifiedBackend")

    # D-ORPHAN-2's fix, pinned: the wrapper wires the production
    # CircuitBreaker from utils/retry.
    src = inspect.getsource(FallbackBackend)
    if "CircuitBreaker" not in src:
        problems.append("FallbackBackend no longer wires CircuitBreaker")

    # D-HEALTH-1's fix, pinned in HF's own source.
    from thoughtforge.inference.hf_backend import HuggingFaceBackend

    hsrc = inspect.getsource(HuggingFaceBackend.health_check)
    if "status_code == 200" not in hsrc or "status_code < 500" in hsrc:
        problems.append(
            "HuggingFaceBackend.health_check no longer requires "
            "exactly 200 (D-HEALTH-1 regression)")

    # The server-shaped semantics, pinned where they live.
    from thoughtforge.inference.lmstudio_backend import LMStudioBackend
    from thoughtforge.inference.ollama_backend import OllamaBackend
    from thoughtforge.inference.turboquant_backend import TurboQuantBackend

    if "/api/tags" not in inspect.getsource(OllamaBackend.health_check):
        problems.append("ollama health_check no longer reads /api/tags")
    if "/v1/models" not in inspect.getsource(LMStudioBackend.health_check):
        problems.append("lmstudio health_check no longer reads /v1/models")
    tq = inspect.getsource(TurboQuantBackend.health_check)
    if "exists()" not in tq or "llama_cpp" not in tq:
        problems.append("turboquant health_check semantics drifted")
    return problems


def fallback_live_problems() -> list[str]:
    """The policy re-derived live over stub chains (this VM)."""
    problems: list[str] = []
    req = GenerationRequest(prompt="prompt")

    chain = _stub_chain()
    resp = chain.generate(req)
    if resp.text != "text from serving:3":
        problems.append(f"degraded chain served {resp.text!r}")
    report = chain.last_report
    if [a.outcome for a in report.attempts] != [
            "health_false", "error", "served"]:
        problems.append(
            f"attempt outcomes {[a.outcome for a in report.attempts]}")
    if not report.degraded or report.served_by != "serving:3":
        problems.append("degraded serve not announced in the report")
    if chain._backends[0].generate_calls != 0:
        problems.append("health gate let a generate through")

    # All-failed: the error names the whole chain.
    from thoughtforge.inference.unified_backend import (
        GenerationResponse as _Resp,
    )

    class Dead(UnifiedBackend):
        def __init__(self, name):
            self._name = name

        def generate(self, request):
            return _Resp(text="", error=f"{self._name} down",
                         finish_reason="error", backend_used=self._name)

        def health_check(self):
            return True

        def backend_name(self):
            return self._name

    dead = FallbackBackend([Dead("a:1"), Dead("b:2")])
    resp2 = dead.generate(req)
    if resp2.finish_reason != "error" or resp2.text != "":
        problems.append("all-failed response is not a clean error")
    for name in ("a:1", "b:2"):
        if name not in resp2.error:
            problems.append(f"all-failed error does not name {name}")
    if not dead.last_report.all_failed:
        problems.append("all-failed not recorded in the report")

    # Circuit breaker: threshold failures open it, then the
    # member is skipped with no further calls while a fallback
    # keeps serving.
    class Flaky(UnifiedBackend):
        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            return _Resp(text="", error="flaky down",
                         finish_reason="error", backend_used="flaky:1")

        def health_check(self):
            return True

        def backend_name(self):
            return "flaky:1"

    class Good(UnifiedBackend):
        def generate(self, request):
            return _Resp(text="good text", backend_used="good:2")

        def health_check(self):
            return True

        def backend_name(self):
            return "good:2"

    flaky = Flaky()
    chain2 = FallbackBackend([flaky, Good()], failure_threshold=2,
                             recovery_timeout=60.0)
    chain2.generate(req)
    chain2.generate(req)
    calls_at_open = flaky.calls
    resp3 = chain2.generate(req)
    if resp3.text != "good text":
        problems.append("chain with an open breaker stopped serving")
    if flaky.calls != calls_at_open:
        problems.append("open breaker still let calls through")
    if chain2.last_report.attempts[0].outcome != "circuit_open":
        problems.append("circuit_open skip not recorded first")
    return problems


__all__ = [
    "ATTEMPT_OUTCOMES",
    "FALLBACK_FORMAT",
    "FALLBACK_RULES",
    "HEALTH_SEMANTICS",
    "FallbackAttempt",
    "FallbackBackend",
    "FallbackReport",
    "fallback_as_data",
    "fallback_live_problems",
    "fallback_problems",
]
