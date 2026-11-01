"""TF-084 — Inference traces join cognition traces: one
exportable story per turn.

Roadmap: "Inference traces join cognition traces
(TF-038): one exportable story per turn."

Measured start (probed live this tick):

- **TF-038's trace carried no inference.** The cognition
  trace (``cognition/traces.py``, format
  ``thoughtforge-turn-trace/1``) exports route / retrieval /
  scaffold / candidates / fragments / failures / final /
  timings — but nothing about the inference that produced
  the candidates: which backend or engine served, with what
  request parameters, how long each call took, what the
  backend itself reported, or why a call failed beyond the
  failure record's one line. Its own §7 named this slice as
  the consumer that joins the two sides.
- **The inference side recorded nothing per turn.** A
  ``GenerationResponse`` carries backend_used / latency_ms /
  finish_reason / tokens_generated / error, and a
  ``GenerationResult`` carries tokens_prompt / duration_ms /
  stopped_reason / profile / model — but both objects are
  consumed by the core and discarded; no per-turn record
  survived a call. Fallback attempts lived only on
  ``FallbackBackend.last_report`` until the next call
  overwrote them.
- **D-HON-1 was this slice's assigned defect.** The
  streamed unified-backend path lost the backend's error
  string: ``UnifiedBackend.generate_stream``'s default
  yields nothing at all when its response carries an
  error, so a streamed turn could only record "backend
  returned no text (error or empty)" while the plain path
  recorded ``resp.error`` exactly. **FIXED in scope** (see
  ``TRACE_RULES`` ``error_detail_is_exact``): the default
  stream now stashes the response it obtained on
  ``backend.last_stream_response`` (the yield-nothing
  contract itself is unchanged — TF-026/TF-065's pins
  stand), ``TurboQuantBackend``'s load-failure stream leg
  stashes the same way, and the core's streamed path reads
  the stash (or a fallback chain's ``last_report``) so the
  streamed failure record and the joined trace carry the
  exact error string, exactly as the plain path does.

THE JOIN (this module):

- :class:`InferenceRecorder` — an ordered, per-turn record
  of inference calls. The cognition core owns one per turn
  (created in ``_run_turn``, continued across a resume of
  the same turn) and records **every leaf generation call**
  at the call site: unified-backend generate / stream,
  local-engine generate / stream, the repair pass, and a
  model-load failure. :class:`TracingBackend` is the same
  recorder behind a ``UnifiedBackend`` wrapper, for
  consumers that hold a backend outside the core.
- :func:`join_traces` — the one exportable story:
  ``{format, turn_id, cognition, inference,
  reconciliation}``. The cognition trace rides **whole and
  unmodified** (TF-038's format, judged by TF-038's own
  checker); the inference section is this module's format;
  the reconciliation block is *derived* from the two
  sections — call counts, failed-call count vs the
  cognition trace's ``generation_failures`` count, served
  candidate-call tokens vs the candidates' own
  ``token_estimate`` sum, repair tokens, total inference
  latency — and :func:`joined_trace_problems` recomputes
  it, so a tampered reconciliation is caught by name.
- **No prompt text, no response text.** TF-038 §7 ruled
  prompts out of the cognition trace (a turn builds many
  prompts; the trace records the turn, not a transcript);
  this slice extends the ruling inference-side: a call
  records the prompt's *size* (chars + token count with
  its basis named) and the response's *size* (chars +
  tokens with their basis named) — the texts themselves
  live in the cognition trace's candidates, once.
- **Latency is named per call, never conflated.**
  ``latency_ms`` is the wall around the leaf call as the
  recorder measured it; ``reported_latency_ms`` is what
  the backend/engine itself reported
  (``GenerationResponse.latency_ms`` /
  ``GenerationResult.duration_ms``), kept beside it, never
  substituted for it.
- **Tracing never breaks a turn.** The core records
  through a guard that swallows recorder errors at debug
  level (a trace is a record, not a gate — the same
  posture as TF-031's failure records, which the trace
  carries but never raises through).

The core surface is ``ThoughtForgeCore.last_turn_joined_trace()``
(None before the first turn), recorded in
``cognition/core_docs.py``'s POST_FREEZE_ADDITIONS as a
TF-034 MINOR addition beside ``last_turn_trace``.

Proof grade FIXTURE (real core turns on this VM —
knowledge-only, stub-backend plain + streamed, stated
backend failure plain vs streamed carrying the *same*
exact error after the D-HON-1 fix, stub-engine turns,
a FallbackBackend chain whose attempts ride the call
record, and hand-built records for the engine legs a
no-model VM cannot run) — a live model's joined trace
NOT PROVEN (no ``llama_cpp``, no GGUF in the tree; the
engine legs are proven against stub engines shaped
exactly as the real one's results), never claimed.

The human-readable write-up is
``docs/architecture/JOINED_TRACES.md``.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

from thoughtforge.cognition.traces import trace_problems
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)

logger = logging.getLogger(__name__)

#: The inference section's own format.
INFERENCE_FORMAT = "thoughtforge-inference-trace/1"

#: The joined document's format.
JOINED_FORMAT = "thoughtforge-joined-trace/1"

#: The joined document's sections, in story order.
JOINED_SECTIONS: tuple[str, ...] = (
    "cognition",
    "inference",
    "reconciliation",
)

#: The leaf call kinds a recorder can carry.
CALL_KINDS: tuple[str, ...] = (
    "backend_generate",
    "backend_stream",
    "engine_generate",
    "engine_stream",
    "engine_repair",
    "model_load",
)

#: The outcomes a call can end with. ``interrupted`` is
#: TF-027's cut — stated, never counted as a failure.
CALL_OUTCOMES: tuple[str, ...] = (
    "served",
    "error",
    "empty",
    "exception",
    "interrupted",
)

#: Outcomes that are failures (TF-031 records each of
#: these on the turn as a generation failure).
FAILED_OUTCOMES: tuple[str, ...] = ("error", "empty", "exception")

TRACE_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "one_story_per_turn",
        "rule": "a joined trace is exactly one turn's story: "
                "the TF-038 cognition trace whole and "
                "unmodified, the inference calls that turn "
                "made in call order, and a reconciliation "
                "derived from the two — one turn_id across "
                "all three sections",
    },
    {
        "id": "every_leaf_call_recorded",
        "rule": "the core records every leaf generation call "
                "it makes directly, at its call site — "
                "backend generate / stream, engine generate "
                "/ stream, the repair pass, and a model-load "
                "failure — served, failed, empty, and cut "
                "alike; a knowledge-only turn records zero "
                "calls and its generation_source is stated "
                "as none, never dressed as an inference turn",
    },
    {
        "id": "salvage_refine_passes_are_a_named_gap",
        "rule": "D-TRACE-1 (FIXED by TF-097): the refine "
                "passes FragmentSalvage makes through the "
                "engine inside its own forge() are recorded "
                "individually by salvage itself in its "
                "rescue record (refinement/"
                "rescue_provenance.py, format "
                "thoughtforge-rescue-record/1 — outcome, "
                "score, size, rejection reasons, failure, "
                "wall + reported latencies per pass), "
                "carried on the final response record into "
                "the cognition trace's final section, and "
                "so into this joined trace inside its "
                "cognition section. They are NOT inference-"
                "section calls: this recorder records the "
                "core's own leaf calls, and an aggregate "
                "invented from the salvage result would "
                "have been fabrication — the record is "
                "built by salvage as each pass runs, "
                "which is why the gap this rule's id names "
                "is closed from salvage's side, not here",
    },
    {
        "id": "prompt_and_response_text_never_recorded",
        "rule": "an inference call records sizes, never "
                "texts: prompt chars + a token count with "
                "its basis named, response chars + tokens "
                "with their basis named. The texts live "
                "once, in the cognition trace's candidates "
                "(TF-038 §7's prompt ruling, extended "
                "inference-side)",
    },
    {
        "id": "error_detail_is_exact",
        "rule": "D-HON-1 (found by TF-031, FIXED by "
                "TF-084): a failed call's error is the "
                "backend/engine's own string, verbatim, on "
                "the streamed path exactly as on the plain "
                "path. The default generate_stream still "
                "yields nothing on an error response (the "
                "TF-026/TF-065 contract stands), but it now "
                "stashes that response on "
                "last_stream_response; TurboQuantBackend's "
                "load-failure stream leg stashes likewise; "
                "the core's streamed path and this recorder "
                "read the stash — or a FallbackBackend's "
                "last_report — before falling back to the "
                "generic no-text detail, which now stands "
                "only for a genuinely detail-less empty",
    },
    {
        "id": "latency_named_never_conflated",
        "rule": "latency_ms is the recorder's wall around "
                "the leaf call; reported_latency_ms is "
                "what the backend or engine itself "
                "reported, filed beside it — neither is "
                "ever substituted for the other, and a "
                "source that reported nothing files None",
    },
    {
        "id": "stream_tokens_basis_is_stated",
        "rule": "a streamed call has no per-call token "
                "report from its source, so its recorded "
                "tokens_generated is the completed text's "
                "count on the basis the core itself used "
                "for the candidate (word count after "
                "stream, basis word_count_after_stream) — "
                "the same number the candidate carries, "
                "never a model count dressed as one",
    },
    {
        "id": "reconciliation_is_recomputed",
        "rule": "the reconciliation block is derived, "
                "never trusted: joined_trace_problems "
                "recomputes every field from the cognition "
                "+ inference sections and names any "
                "disagreement — including the two cross-"
                "section identities: failed inference calls "
                "== the cognition trace's generation_"
                "failures count, and served candidate-call "
                "tokens == the candidates' token_estimate "
                "sum (repair tokens are reconciled "
                "separately: a repair produces a final, "
                "not a candidate; and a repair call "
                "returning empty files no TF-031 failure "
                "— the core composes from it as-is — so "
                "the failed count excludes exactly that "
                "one shape, stated in _reconciliation)",
    },
    {
        "id": "fallback_attempts_ride_the_call",
        "rule": "when the serving backend is a TF-073 "
                "FallbackBackend, its last_report attempts "
                "(each member's outcome, in order) ride "
                "the call record as fallback_attempts — "
                "the chain's story is part of the turn's "
                "story, captured before the next call can "
                "overwrite the report",
    },
    {
        "id": "tracing_never_breaks_a_turn",
        "rule": "recording is a guard-wrapped side effect "
                "in the core: a recorder failure is logged "
                "at debug level and the turn proceeds "
                "exactly as it would untraced. A trace is "
                "a record, not a gate",
    },
    {
        "id": "live_model_join_not_proven",
        "rule": "a live model's joined trace is NOT "
                "PROVEN: no llama_cpp and no GGUF exist "
                "on this VM, so the engine legs are "
                "proven against stub engines returning "
                "the real GenerationResult shape, and "
                "TF-127/TF-175's live panels own the "
                "road. TF-086's truth table must carry "
                "the line",
    },
)


# ── The recorder ──────────────────────────────────────────────────────────────


def build_call(
    *,
    kind: str,
    source: str,
    backend: str = "",
    mode: str = "",
    profile_id: str | None = None,
    model: str | None = None,
    prompt_chars: int = 0,
    prompt_tokens: int | None = None,
    prompt_tokens_basis: str = "none",
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    streamed: bool = False,
    native: bool | None = None,
    text_chars: int = 0,
    tokens_generated: int | None = None,
    tokens_basis: str = "none",
    finish_reason: str = "",
    latency_ms: float = 0.0,
    reported_latency_ms: float | None = None,
    outcome: str = "served",
    error: str = "",
    chunks: int | None = None,
    fallback_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One inference call as detached JSON-safe data.

    The one builder every recording site uses — core call
    sites, :class:`TracingBackend`, and the engine helpers
    below — so a call's shape has exactly one home. A model
    is filed by filename (the tree's standing identity for
    a model — TF-066/TF-078), never by filesystem path.
    """
    if model and "/" in str(model):
        model = Path(str(model)).name
    return {
        "sequence": 0,  # the recorder assigns this
        "kind": kind,
        "source": source,
        "backend": backend,
        "mode": mode,
        "profile_id": profile_id,
        "model": model,
        "request": {
            "prompt_chars": prompt_chars,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_basis": prompt_tokens_basis,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "streamed": streamed,
            "native": native,
        },
        "response": {
            "text_chars": text_chars,
            "tokens_generated": tokens_generated,
            "tokens_basis": tokens_basis,
            "finish_reason": finish_reason,
            "latency_ms": round(float(latency_ms), 3),
            "reported_latency_ms": (
                round(float(reported_latency_ms), 3)
                if reported_latency_ms is not None else None
            ),
        },
        "outcome": outcome,
        "error": error,
        "chunks": chunks,
        "fallback_attempts": copy.deepcopy(fallback_attempts),
    }


class InferenceRecorder:
    """An ordered per-turn record of inference calls.

    Plain data in, plain data out: :meth:`record` appends a
    built call with its sequence assigned; :meth:`as_data`
    exports a detached copy in ``INFERENCE_FORMAT``.
    """

    def __init__(self, turn_id: str = "") -> None:
        self.turn_id = turn_id
        self.calls: list[dict[str, Any]] = []

    def record(self, call: dict[str, Any]) -> dict[str, Any]:
        call = copy.deepcopy(call)
        call["sequence"] = len(self.calls) + 1
        self.calls.append(call)
        return call

    def record_call(self, **kwargs: Any) -> dict[str, Any]:
        return self.record(build_call(**kwargs))

    def as_data(self) -> dict[str, Any]:
        return {
            "format": INFERENCE_FORMAT,
            "turn_id": self.turn_id,
            "generation_source": generation_source(self.calls),
            "calls": copy.deepcopy(self.calls),
        }


def generation_source(calls: list[dict[str, Any]]) -> str:
    """What produced a turn's candidates, from its calls."""
    kinds = {c.get("kind") for c in calls}
    has_backend = any(str(k).startswith("backend_") for k in kinds)
    has_engine = any(str(k).startswith("engine_") for k in kinds)
    if has_backend and has_engine:
        return "mixed"
    if has_backend:
        return "unified_backend"
    if has_engine:
        return "turboquant_engine"
    return "none"


def inference_as_data(
    recorder: InferenceRecorder | list[dict[str, Any]] | dict[str, Any] | None,
    turn_id: str = "",
) -> dict[str, Any]:
    """Normalise any recorder shape to the inference section."""
    if recorder is None:
        return {
            "format": INFERENCE_FORMAT,
            "turn_id": turn_id,
            "generation_source": "none",
            "calls": [],
        }
    if isinstance(recorder, InferenceRecorder):
        return recorder.as_data()
    if isinstance(recorder, dict):
        return copy.deepcopy(recorder)
    calls = copy.deepcopy(list(recorder))
    return {
        "format": INFERENCE_FORMAT,
        "turn_id": turn_id,
        "generation_source": generation_source(calls),
        "calls": calls,
    }


# ── Recording helpers for the engine shape ────────────────────────────────────


def record_engine_result(
    recorder: InferenceRecorder,
    *,
    result: Any,
    prompt: str,
    mode: str = "",
    kind: str = "engine_generate",
    backend: str | None = None,
    wall_ms: float = 0.0,
) -> dict[str, Any]:
    """Record one ``TurboQuantEngine.generate`` result.

    ``result`` is the real ``GenerationResult`` shape
    (text / tokens_generated / tokens_prompt / duration_ms
    / stopped_reason / model_path / profile_id) — read by
    attribute, so the real engine and a shaped stub record
    identically.
    """
    text = getattr(result, "text", "") or ""
    return recorder.record_call(
        kind=kind,
        source="turboquant_engine",
        backend=backend or "turboquant_engine",
        mode=mode,
        profile_id=getattr(result, "profile_id", None) or None,
        model=(Path(getattr(result, "model_path", "")).name
               if getattr(result, "model_path", "") else None),
        prompt_chars=len(prompt),
        prompt_tokens=getattr(result, "tokens_prompt", None),
        prompt_tokens_basis="engine_usage",
        text_chars=len(text),
        tokens_generated=getattr(result, "tokens_generated", None),
        tokens_basis="engine_reported",
        finish_reason=getattr(result, "stopped_reason", "") or "",
        latency_ms=wall_ms,
        reported_latency_ms=getattr(result, "duration_ms", None),
        outcome="served" if text.strip() else "empty",
        error="" if text.strip() else "engine returned no text "
                                       "(empty generation)",
    )


# ── The tracing backend wrapper ───────────────────────────────────────────────


class TracingBackend(UnifiedBackend):
    """A ``UnifiedBackend`` wrapper that records every call.

    For consumers that hold a backend outside the cognition
    core: the wrapper delegates generate / generate_stream /
    health_check / backend_name / count_tokens /
    capabilities to the wrapped backend exactly, and
    records each generation call on its recorder — the
    served response is the wrapped backend's own,
    unmodified. A streamed error's detail is read from
    the wrapped backend's ``last_stream_response`` stash
    (D-HON-1) or its fallback ``last_report``.
    """

    def __init__(
        self,
        backend: UnifiedBackend,
        recorder: InferenceRecorder | None = None,
    ) -> None:
        self._backend = backend
        self.recorder = recorder if recorder is not None \
            else InferenceRecorder()

    @property
    def wrapped(self) -> UnifiedBackend:
        return self._backend

    def backend_name(self) -> str:
        return self._backend.backend_name()

    def health_check(self) -> bool:
        return self._backend.health_check()

    def count_tokens(self, text: str) -> int:
        return self._backend.count_tokens(text)

    def capabilities(self):  # noqa: ANN202 — the wrapped record's type
        return self._backend.capabilities()

    def _fallback_attempts(self) -> list[dict[str, Any]] | None:
        report = getattr(self._backend, "last_report", None)
        attempts = getattr(report, "attempts", None)
        if not attempts:
            return None
        out = []
        for attempt in attempts:
            if hasattr(attempt, "to_dict"):
                out.append(attempt.to_dict())
            else:
                out.append({
                    "backend": getattr(attempt, "backend", ""),
                    "outcome": getattr(attempt, "outcome", ""),
                    "detail": getattr(attempt, "detail", ""),
                })
        return out

    def _stream_error_detail(self) -> str:
        stash = getattr(self._backend, "last_stream_response", None)
        error = getattr(stash, "error", "") if stash is not None else ""
        if error:
            return str(error)
        attempts = self._fallback_attempts()
        if attempts:
            failed = [a for a in attempts if a.get("outcome") != "served"]
            if failed:
                return "; ".join(
                    f"{a.get('backend')}: {a.get('outcome')}"
                    + (f" ({a.get('detail')})" if a.get("detail") else "")
                    for a in failed
                )
        return ""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        started = time.perf_counter()
        try:
            resp = self._backend.generate(request)
        except Exception as exc:  # a contract breach — recorded, re-raised
            self.recorder.record_call(
                kind="backend_generate", source="unified_backend",
                backend=self._backend.backend_name(),
                prompt_chars=len(request.prompt),
                prompt_tokens=self._safe_count(request.prompt),
                prompt_tokens_basis="backend_count",
                temperature=request.temperature,
                max_tokens=request.max_tokens, seed=request.seed,
                latency_ms=(time.perf_counter() - started) * 1000,
                outcome="exception",
                error=f"{type(exc).__name__}: {exc}",
                fallback_attempts=self._fallback_attempts(),
            )
            raise
        wall_ms = (time.perf_counter() - started) * 1000
        outcome = "served"
        error_detail = resp.error or ""
        if resp.error or resp.finish_reason == "error":
            outcome = "error"
            if not error_detail:
                error_detail = ("backend reported finish_reason "
                                "'error' with no error detail")
        elif not resp.text or not resp.text.strip():
            outcome = "empty"
        self.recorder.record_call(
            kind="backend_generate", source="unified_backend",
            backend=resp.backend_used or self._backend.backend_name(),
            prompt_chars=len(request.prompt),
            prompt_tokens=self._safe_count(request.prompt),
            prompt_tokens_basis="backend_count",
            temperature=request.temperature,
            max_tokens=request.max_tokens, seed=request.seed,
            text_chars=len(resp.text or ""),
            tokens_generated=resp.tokens_generated,
            tokens_basis="backend_reported",
            finish_reason=resp.finish_reason,
            latency_ms=wall_ms,
            reported_latency_ms=resp.latency_ms or None,
            outcome=outcome, error=error_detail,
            fallback_attempts=self._fallback_attempts(),
        )
        return resp

    def generate_stream(self, request: GenerationRequest) -> Iterator[str]:
        started = time.perf_counter()
        native = type(self._backend).generate_stream is not \
            UnifiedBackend.generate_stream
        parts: list[str] = []
        chunks = 0
        try:
            for chunk in self._backend.generate_stream(request):
                parts.append(chunk)
                chunks += 1
                yield chunk
        except Exception as exc:
            self.recorder.record_call(
                kind="backend_stream", source="unified_backend",
                backend=self._backend.backend_name(),
                prompt_chars=len(request.prompt),
                prompt_tokens=self._safe_count(request.prompt),
                prompt_tokens_basis="backend_count",
                temperature=request.temperature,
                max_tokens=request.max_tokens, seed=request.seed,
                streamed=True, native=native,
                text_chars=len("".join(parts)),
                tokens_generated=len("".join(parts).split()),
                tokens_basis="word_count_after_stream",
                latency_ms=(time.perf_counter() - started) * 1000,
                outcome="exception",
                error=f"{type(exc).__name__}: {exc}",
                chunks=chunks,
                fallback_attempts=self._fallback_attempts(),
            )
            raise
        text = "".join(parts)
        if text.strip():
            outcome, error = "served", ""
        else:
            error = self._stream_error_detail()
            outcome = "error" if error else "empty"
            if not error:
                error = "backend returned no text (error or empty)"
        self.recorder.record_call(
            kind="backend_stream", source="unified_backend",
            backend=self._backend.backend_name(),
            prompt_chars=len(request.prompt),
            prompt_tokens=self._safe_count(request.prompt),
            prompt_tokens_basis="backend_count",
            temperature=request.temperature,
            max_tokens=request.max_tokens, seed=request.seed,
            streamed=True, native=native,
            text_chars=len(text),
            tokens_generated=len(text.split()),
            tokens_basis="word_count_after_stream",
            latency_ms=(time.perf_counter() - started) * 1000,
            outcome=outcome, error=error, chunks=chunks,
            fallback_attempts=self._fallback_attempts(),
        )

    def _safe_count(self, text: str) -> int | None:
        try:
            return self._backend.count_tokens(text)
        except Exception:  # noqa: BLE001 — a count is colour, not a gate
            return None


# ── The join ──────────────────────────────────────────────────────────────────


def _reconciliation(
    cognition: dict[str, Any] | None,
    inference: dict[str, Any],
) -> dict[str, Any]:
    calls = inference.get("calls") or []
    served_candidate = [
        c for c in calls
        if c.get("outcome") == "served"
        and c.get("kind") != "engine_repair"
    ]
    repair = [c for c in calls if c.get("kind") == "engine_repair"]
    # Failed = the calls TF-031 records as generation
    # failures. One stated exception: a repair call that
    # returns empty text is recorded with outcome empty,
    # but the core files no failure record for it (it
    # composes from the empty repair as-is) — so the
    # identity below counts it as an empty call, not a
    # failed one. Its own outcome still says empty.
    failed = [
        c for c in calls
        if c.get("outcome") in FAILED_OUTCOMES
        and not (c.get("kind") == "engine_repair"
                 and c.get("outcome") == "empty")
    ]
    candidate_tokens_recorded = sum(
        (c.get("response") or {}).get("tokens_generated") or 0
        for c in served_candidate
    )
    repair_tokens = sum(
        (c.get("response") or {}).get("tokens_generated") or 0
        for c in repair if c.get("outcome") == "served"
    )
    latency_total = round(sum(
        (c.get("response") or {}).get("latency_ms") or 0.0
        for c in calls
    ), 3)
    if cognition is None:
        return {
            "candidate_count": None,
            "inference_call_count": len(calls),
            "served_call_count": len(
                [c for c in calls if c.get("outcome") == "served"]),
            "failed_call_count": len(failed),
            "generation_failure_count": None,
            "candidate_tokens_recorded": candidate_tokens_recorded,
            "candidate_tokens_in_cognition": None,
            "repair_tokens_recorded": repair_tokens,
            "inference_latency_ms_total": latency_total,
            "cognition_generation_ms": None,
        }
    candidates = cognition.get("candidates") or []
    return {
        "candidate_count": len(candidates),
        "inference_call_count": len(calls),
        "served_call_count": len(
            [c for c in calls if c.get("outcome") == "served"]),
        "failed_call_count": len(failed),
        "generation_failure_count": len(
            cognition.get("generation_failures") or []),
        "candidate_tokens_recorded": candidate_tokens_recorded,
        "candidate_tokens_in_cognition": sum(
            c.get("token_estimate") or 0 for c in candidates),
        "repair_tokens_recorded": repair_tokens,
        "inference_latency_ms_total": latency_total,
        "cognition_generation_ms": (
            cognition.get("timings") or {}).get("generation_ms"),
    }


def join_traces(
    cognition_trace: dict[str, Any] | None,
    inference: InferenceRecorder | list[dict[str, Any]] | dict[str, Any] | None,
) -> dict[str, Any]:
    """One exportable story per turn: cognition + inference.

    ``cognition_trace`` is a TF-038 trace (or None for an
    inference-only export — the reconciliation's
    cognition-derived fields then file None, stated, never
    zero-filled). The inference section's turn_id falls
    back to the cognition trace's when its own is empty;
    a genuine disagreement is left in place for the
    checker to name.
    """
    cognition = copy.deepcopy(cognition_trace)
    turn_id = (cognition or {}).get("turn_id") or ""
    inference_data = inference_as_data(inference, turn_id=turn_id)
    if not inference_data.get("turn_id"):
        inference_data["turn_id"] = turn_id
    if not turn_id:
        turn_id = inference_data.get("turn_id") or ""
    return {
        "format": JOINED_FORMAT,
        "turn_id": turn_id,
        "cognition": cognition,
        "inference": inference_data,
        "reconciliation": _reconciliation(cognition, inference_data),
    }


def joined_to_json(joined: dict[str, Any]) -> str:
    """The joined trace as one JSON document."""
    return json.dumps(joined, indent=2, ensure_ascii=False) + "\n"


def write_joined_trace(joined: dict[str, Any], path: str | Path) -> Path:
    """Write ``joined`` to ``path`` as JSON; returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(joined_to_json(joined), encoding="utf-8")
    return target


# ── The checkers ──────────────────────────────────────────────────────────────


def inference_problems(inference: Any) -> list[str]:
    """Problems with an inference section, as data."""
    if not isinstance(inference, dict):
        return ["inference section is not a mapping"]
    problems: list[str] = []
    if inference.get("format") != INFERENCE_FORMAT:
        problems.append(
            f"inference format is {inference.get('format')!r}, "
            f"not {INFERENCE_FORMAT!r}")
    calls = inference.get("calls")
    if not isinstance(calls, list):
        problems.append("inference calls are not a list")
        return problems
    if inference.get("generation_source") != generation_source(calls):
        problems.append("generation_source disagrees with the calls")
    for idx, call in enumerate(calls):
        label = f"call {idx + 1}"
        if call.get("sequence") != idx + 1:
            problems.append(f"{label}: sequence is "
                            f"{call.get('sequence')!r}, not {idx + 1}")
        if call.get("kind") not in CALL_KINDS:
            problems.append(f"{label}: kind {call.get('kind')!r} "
                            "is not a known call kind")
        outcome = call.get("outcome")
        if outcome not in CALL_OUTCOMES:
            problems.append(f"{label}: outcome {outcome!r} "
                            "is not a known outcome")
        if outcome == "served" and call.get("error"):
            problems.append(f"{label}: a served call carries an error")
        if outcome == "error" and not call.get("error"):
            problems.append(f"{label}: an error call carries "
                            "no error detail (D-HON-1)")
        if outcome == "exception" and not call.get("error"):
            problems.append(f"{label}: an exception call carries "
                            "no error detail")
        response = call.get("response") or {}
        for key in ("text_chars", "latency_ms"):
            value = response.get(key)
            if isinstance(value, (int, float)) and value < 0:
                problems.append(f"{label}: response {key} is negative")
        tokens = response.get("tokens_generated")
        if isinstance(tokens, (int, float)) and tokens < 0:
            problems.append(f"{label}: tokens_generated is negative")
    return problems


def joined_trace_problems(joined: Any) -> list[str]:
    """Problems with a joined trace, as data ([] = sound).

    Judges the document as a self-contained record: the
    format, every section present, the cognition section
    judged by TF-038's own checker, the inference section
    by :func:`inference_problems`, one turn_id across the
    document, and the reconciliation recomputed field by
    field — including the two cross-section identities
    (failed calls == generation failures; served
    candidate-call tokens == candidate token estimates).
    """
    if not isinstance(joined, dict):
        return ["joined trace is not a mapping"]
    problems: list[str] = []
    if joined.get("format") != JOINED_FORMAT:
        problems.append(
            f"joined format is {joined.get('format')!r}, "
            f"not {JOINED_FORMAT!r}")
    for section in JOINED_SECTIONS:
        if section not in joined:
            problems.append(f"joined trace is missing section {section!r}")
    turn_id = joined.get("turn_id")
    if not turn_id:
        problems.append("joined trace has no turn_id")

    cognition = joined.get("cognition")
    if cognition is not None:
        if not isinstance(cognition, dict):
            problems.append("cognition section is not a mapping")
            cognition = None
        else:
            problems.extend(
                f"cognition: {p}" for p in trace_problems(cognition))
            if turn_id and cognition.get("turn_id") not in (None, turn_id):
                problems.append(
                    "cognition turn_id "
                    f"{cognition.get('turn_id')!r} disagrees with "
                    f"joined turn_id {turn_id!r}")

    inference = joined.get("inference")
    if inference is not None:
        problems.extend(inference_problems(inference))
        if isinstance(inference, dict):
            inf_turn = inference.get("turn_id")
            if turn_id and inf_turn not in (None, "", turn_id):
                problems.append(
                    f"inference turn_id {inf_turn!r} disagrees "
                    f"with joined turn_id {turn_id!r}")
    else:
        inference = {"calls": []}

    reconciliation = joined.get("reconciliation")
    if isinstance(inference, dict):
        expected = _reconciliation(
            cognition if isinstance(cognition, dict) else None,
            inference)
        if reconciliation != expected:
            problems.append(
                "reconciliation disagrees with the sections "
                f"(filed {reconciliation!r}, recomputed {expected!r})")
        elif isinstance(cognition, dict):
            if expected["failed_call_count"] != \
                    expected["generation_failure_count"]:
                problems.append(
                    "failed inference calls "
                    f"({expected['failed_call_count']}) disagree "
                    "with the cognition trace's generation "
                    f"failures ({expected['generation_failure_count']})")
            if expected["candidate_tokens_recorded"] != \
                    expected["candidate_tokens_in_cognition"]:
                problems.append(
                    "served candidate-call tokens "
                    f"({expected['candidate_tokens_recorded']}) "
                    "disagree with the candidates' token "
                    f"estimates ({expected['candidate_tokens_in_cognition']})")
    return problems


def joined_as_data(
    cognition_trace: dict[str, Any] | None = None,
    recorder: InferenceRecorder | None = None,
) -> dict[str, Any]:
    """The estate as data (format + rules + shapes)."""
    return {
        "format": JOINED_FORMAT,
        "inference_format": INFERENCE_FORMAT,
        "sections": list(JOINED_SECTIONS),
        "call_kinds": list(CALL_KINDS),
        "call_outcomes": list(CALL_OUTCOMES),
        "rules": [dict(r) for r in TRACE_RULES],
        "proof": {
            "joined_core_turns": "proven_fixture",
            "d_hon_1_stream_error_detail": "fixed",
            "live_model_join": "not_proven",
        },
    }


__all__ = [
    "CALL_KINDS",
    "CALL_OUTCOMES",
    "FAILED_OUTCOMES",
    "INFERENCE_FORMAT",
    "JOINED_FORMAT",
    "JOINED_SECTIONS",
    "TRACE_RULES",
    "InferenceRecorder",
    "TracingBackend",
    "build_call",
    "generation_source",
    "inference_as_data",
    "inference_problems",
    "join_traces",
    "joined_as_data",
    "joined_to_json",
    "joined_trace_problems",
    "record_engine_result",
    "write_joined_trace",
]
