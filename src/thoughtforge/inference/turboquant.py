"""
TurboQuantEngine — Universal inference engine for ThoughtForge.

Wraps llama-cpp-python with:
  - Automatic hardware profile selection
  - Backend-aware model loading (CUDA / ROCm / Vulkan / Metal / CPU)
  - Strict token budget enforcement per hardware profile
  - Multi-draft generation for fragment salvage pipeline
  - Graceful degradation when llama-cpp-python is not installed

Usage:
    engine = TurboQuantEngine(model_path="/models/phi-3-mini-q4.gguf")
    engine.load()
    drafts = engine.generate_drafts(prompt, num_drafts=3)
    engine.unload()

    # Or as a context manager:
    with TurboQuantEngine(model_path="...") as engine:
        result = engine.generate(prompt)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from thoughtforge.inference.backends import BackendInfo, build_llama_kwargs, select_backend
from thoughtforge.inference.profiles import (
    ProfileMatch,
    ProfileSelector,
    ResolvedGenerationParams,
    resolve_generation_params,
)
from thoughtforge.inference.quant_policy import QuantVerdict, check_quant_for_profile

logger = logging.getLogger(__name__)

# Sentinel for unloaded state
_NOT_LOADED = object()


# ── Parameter and result types ─────────────────────────────────────────────────

@dataclass
class GenerationParams:
    """
    Per-call generation parameters. Values of None inherit from hardware profile defaults.
    """
    max_tokens: int | None = None
    temperature: float | None = None
    repeat_penalty: float | None = None
    top_p: float | None = None
    top_k: int = 40
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    # TF-036: per-call sampling seed (determinism mode). None =
    # unseeded; when set it is forwarded to the llama call, and
    # omitted entirely when None so unseeded calls are unchanged.
    seed: int | None = None


@dataclass
class GenerationResult:
    """Output from a single generation call."""
    text: str
    tokens_generated: int
    tokens_prompt: int
    total_tokens: int
    duration_ms: int
    tokens_per_second: float
    stopped_reason: str = ""        # "length" | "stop_token" | "eos"
    model_path: str = ""
    profile_id: str = ""


@dataclass
class DraftSet:
    """A set of draft responses from multi-draft generation."""
    drafts: list[GenerationResult]
    prompt: str
    profile_id: str
    total_duration_ms: int

    @property
    def texts(self) -> list[str]:
        return [d.text for d in self.drafts]

    @property
    def best(self) -> GenerationResult:
        return max(self.drafts, key=lambda d: d.tokens_generated)


# ── Engine ─────────────────────────────────────────────────────────────────────

class TurboQuantEngine:
    """
    Universal quantized inference engine.

    Supports all hardware tiers from phone/Pi Zero to 70B server GPU.
    Same interface across all tiers — only the model and token budget differ.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        profile_id: str = "auto",
        backend: str = "auto",
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            model_path:     Path to a GGUF model file.
                            If None, the engine loads but generate() will raise.
            profile_id:     Hardware profile to use ("auto" = auto-detect).
            backend:        Inference backend ("auto" = best available).
            n_ctx:          Context window size (None = from profile).
            n_gpu_layers:   GPU layer offload count (None = from profile).
            n_threads:      CPU thread count (None = auto).
            verbose:        Enable llama.cpp verbose logging.
        """
        self.model_path = Path(model_path) if model_path else None
        self.profile_id_override = profile_id
        self.backend_override = backend
        self._n_ctx_override = n_ctx
        self._n_gpu_layers_override = n_gpu_layers
        self._n_threads = n_threads
        self._verbose = verbose

        self._model: Any = _NOT_LOADED
        self.quant_verdict: QuantVerdict | None = None
        self._profile_match: ProfileMatch | None = None
        self._params: ResolvedGenerationParams | None = None
        self._backend: BackendInfo | None = None
        self._loaded: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the model into memory. Must be called before generate()."""
        if self._loaded:
            logger.debug("Model already loaded — skipping")
            return

        # Resolve profile
        selector = ProfileSelector()
        if self.profile_id_override == "auto":
            self._profile_match = selector.auto_select()
        else:
            self._profile_match = selector.load(self.profile_id_override)

        self._params = resolve_generation_params(self._profile_match.profile)
        logger.info(
            "Profile: %s (%s)",
            self._profile_match.profile_id,
            self._profile_match.reason,
        )

        # TF-068 (D-QUANT-1): the profile's quant is policy, not a
        # suggestion nobody reads. Check the loaded file's
        # filename-provable quant against the profile's policy
        # default and record the verdict; a mismatch is announced,
        # never silently passed (and never a hard refusal — the
        # operator may load another quant deliberately, but not
        # unknowingly). A filename proving no quant is recorded
        # as unprovable, never rounded up.
        self.quant_verdict = check_quant_for_profile(
            self._profile_match.profile_id,
            self.model_path.name if self.model_path else None,
        )
        if self.quant_verdict.status == "mismatch":
            logger.warning(
                "Quant policy mismatch: profile %s defaults to %s but "
                "the loaded file %s proves %s — loading as instructed, "
                "mismatch recorded (TF-068 quant policy)",
                self.quant_verdict.profile_id,
                self.quant_verdict.expected,
                self.model_path.name if self.model_path else "",
                self.quant_verdict.found,
            )
        elif self.quant_verdict.status == "unprovable":
            logger.info(
                "Quant policy: %s proves no quant for profile %s "
                "(%s) — recorded as unprovable, not a mismatch",
                self.model_path.name if self.model_path else "",
                self.quant_verdict.profile_id,
                self.quant_verdict.gap,
            )

        # Resolve backend
        self._backend = select_backend(
            preferred=self.backend_override
            if self.backend_override != "auto"
            else self._params.inference_backend,
            fallbacks=self._params.inference_backend_fallbacks,
        )

        if self.model_path is None:
            logger.info(
                "TurboQuantEngine: no model_path — engine ready but generate() will raise "
                "until a model is loaded via load_model_path()"
            )
            self._loaded = True
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Download a GGUF model and pass its path to TurboQuantEngine(model_path=...)"
            )

        self._load_llama()
        self._loaded = True
        logger.info("TurboQuantEngine ready: %s [%s]", self.model_path.name, self._backend.name)

    def load_model_path(self, model_path: str | Path) -> None:
        """Load (or reload) a specific model file, replacing any current model."""
        self.unload()
        self.model_path = Path(model_path)
        self._loaded = False
        self.load()

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not _NOT_LOADED and self._model is not None:
            try:
                del self._model
                self._model = _NOT_LOADED
                logger.info("Model unloaded")
            except Exception as e:
                logger.warning("Error unloading model: %s", e)
        self._loaded = False

    def __enter__(self) -> "TurboQuantEngine":
        self.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.unload()

    # ── Primary generation API ─────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> GenerationResult:
        """
        Generate a single response to a prompt.

        Args:
            prompt: The full prompt string (including system context).
            params: Override generation parameters. None = use profile defaults.

        Returns:
            GenerationResult with text and metadata.
        """
        self._assert_ready()
        assert self._params is not None  # guaranteed by _assert_ready

        p = self._merge_params(params)
        max_tokens = self._enforce_token_budget(p["max_tokens"])
        stop_sequences = p.get("stop", [])

        t_start = time.perf_counter()

        try:
            llama = self._model
            output = llama(
                prompt,
                max_tokens=max_tokens,
                temperature=p["temperature"],
                repeat_penalty=p["repeat_penalty"],
                top_p=p["top_p"],
                top_k=p.get("top_k", 40),
                stop=stop_sequences,
                echo=False,
                **({"seed": p["seed"]} if p.get("seed") is not None else {}),
            )
        except Exception as e:
            logger.error("Generation failed: %s", e)
            raise RuntimeError(f"TurboQuantEngine generation error: {e}") from e

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

        text = output["choices"][0]["text"].strip() if output.get("choices") else ""
        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        tps = completion_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0

        finish_reason = output["choices"][0].get("finish_reason", "") if output.get("choices") else ""

        result = GenerationResult(
            text=text,
            tokens_generated=completion_tokens,
            tokens_prompt=prompt_tokens,
            total_tokens=total_tokens,
            duration_ms=elapsed_ms,
            tokens_per_second=round(tps, 1),
            stopped_reason=finish_reason or "eos",
            model_path=str(self.model_path or ""),
            profile_id=self._profile_match.profile_id if self._profile_match else "unknown",
        )

        logger.debug(
            "Generated %d tokens in %dms (%.1f t/s)",
            completion_tokens, elapsed_ms, tps,
        )
        return result

    def generate_stream(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> "Iterator[str]":
        """
        Stream a single generation, yielding text deltas as the
        model produces them (TF-026).

        The llama call is made with ``stream=True``; each yielded
        item is one choice-delta's text, in order, unstripped —
        ``"".join(generate_stream(...))`` is the raw completion,
        where ``generate()`` returns the same text stripped.
        Parameters and the token budget are merged and enforced
        exactly as in ``generate()``. A failure mid-stream raises
        RuntimeError, like ``generate()``; deltas already yielded
        stay yielded — the caller decides what a partial stream
        means (the cognition core discards the partial candidate).

        TF-072 (D-STREAM-1): the underlying llama stream is
        closed on EVERY exit — exhaustion, a mid-stream error,
        and the consumer closing this generator early
        (GeneratorExit) — via the ``finally`` below. Before this
        fix the stream iterator was simply abandoned on the
        error and early-close paths, leaving the model's
        streaming state to garbage collection.
        """
        self._assert_ready()
        assert self._params is not None  # guaranteed by _assert_ready

        p = self._merge_params(params)
        max_tokens = self._enforce_token_budget(p["max_tokens"])
        stop_sequences = p.get("stop", [])

        stream = None
        try:
            llama = self._model
            stream = llama(
                prompt,
                max_tokens=max_tokens,
                temperature=p["temperature"],
                repeat_penalty=p["repeat_penalty"],
                top_p=p["top_p"],
                top_k=p.get("top_k", 40),
                stop=stop_sequences,
                echo=False,
                stream=True,
                **({"seed": p["seed"]} if p.get("seed") is not None else {}),
            )
            for chunk in stream:
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not choices:
                    continue
                delta = choices[0].get("text", "")
                if delta:
                    yield delta
        except Exception as e:
            logger.error("Streaming generation failed: %s", e)
            raise RuntimeError(f"TurboQuantEngine streaming error: {e}") from e
        finally:
            # TF-072: close the model's stream iterator whatever
            # way this generator ends. GeneratorExit is not an
            # Exception and is not converted above; it still runs
            # this finally, so a consumer's close() reaches the
            # model. A stream with no close() (a plain iterator)
            # needs nothing; a failing close never masks the
            # stream's own outcome.
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("llama stream close failed", exc_info=True)

    def generate_drafts(
        self,
        prompt: str,
        num_drafts: int | None = None,
        params: GenerationParams | None = None,
    ) -> DraftSet:
        """
        Generate multiple draft responses for the fragment salvage pipeline.

        The number of drafts defaults to the hardware profile's `draft_count`,
        scaled down automatically on constrained hardware.

        Args:
            prompt:     The full prompt string.
            num_drafts: Override draft count. None = from hardware profile.
            params:     Override generation parameters.

        Returns:
            DraftSet with all generated drafts.
        """
        self._assert_ready()
        assert self._params is not None

        count = num_drafts if num_drafts is not None else self._params.draft_count
        count = max(1, count)

        logger.debug("Generating %d drafts", count)
        t_start = time.perf_counter()

        drafts: list[GenerationResult] = []
        for i in range(count):
            # Vary temperature slightly across drafts for diversity
            draft_params = GenerationParams(
                max_tokens=params.max_tokens if params else None,
                temperature=(
                    (params.temperature if params and params.temperature else None)
                    or self._params.temperature
                ) + (i * 0.05),
                repeat_penalty=params.repeat_penalty if params else None,
                top_p=params.top_p if params else None,
                stop=params.stop if params else [],
                # TF-036: a seeded draft set stays reproducible
                # AND varied — draft i carries seed + i.
                seed=(
                    (params.seed + i)
                    if params is not None and params.seed is not None
                    else None
                ),
            )
            try:
                result = self.generate(prompt, draft_params)
                drafts.append(result)
                logger.debug("Draft %d/%d: %d tokens", i + 1, count, result.tokens_generated)
            except Exception as e:
                logger.warning("Draft %d failed: %s — skipping", i + 1, e)

        total_ms = int((time.perf_counter() - t_start) * 1000)

        if not drafts:
            raise RuntimeError("All draft generation attempts failed")

        return DraftSet(
            drafts=drafts,
            prompt=prompt,
            profile_id=self._profile_match.profile_id if self._profile_match else "unknown",
            total_duration_ms=total_ms,
        )

    # ── Token estimation ───────────────────────────────────────────────────────

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.
        Uses llama.cpp tokenizer if loaded, otherwise falls back to rough heuristic.

        Basis (TF-071): the model path returns the llama.cpp
        tokeniser's count, which by its default includes the
        BOS special token — so a loaded engine counts "" as 1.
        The fallback is the TF-065 contract estimate, delegated
        to ``backend_contract.estimate_tokens`` (its one home;
        this method's pre-TF-071 private copy of the heuristic
        counted "" as 1 even with no model loaded, breaching
        the contract's empty rule — D-TOK-1).
        """
        if self._model is not _NOT_LOADED and self._model is not None:
            try:
                tokens = self._model.tokenize(text.encode("utf-8"))
                return len(tokens)
            except Exception:
                pass
        # The contract estimate (TF-065/TF-071): ~4 chars per
        # token, script-aware, 0 for empty.
        from thoughtforge.inference.backend_contract import estimate_tokens

        return estimate_tokens(text)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def profile_id(self) -> str | None:
        return self._profile_match.profile_id if self._profile_match else None

    @property
    def params(self) -> ResolvedGenerationParams | None:
        return self._params

    @property
    def backend_name(self) -> str | None:
        return self._backend.name if self._backend else None

    @property
    def max_response_tokens(self) -> int:
        return self._params.max_response_tokens if self._params else 250

    @property
    def draft_count(self) -> int:
        return self._params.draft_count if self._params else 3

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_llama(self) -> None:
        """Load the model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is not installed.\n"
                "Install it with: pip install llama-cpp-python\n"
                "For GPU support: pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu124"
            ) from e

        assert self._params is not None
        assert self._backend is not None
        assert self.model_path is not None

        n_ctx = self._n_ctx_override or self._params.max_context_tokens
        n_gpu = self._n_gpu_layers_override
        if n_gpu is None:
            n_gpu = self._params.n_gpu_layers

        kwargs = build_llama_kwargs(
            backend=self._backend,
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu,
            n_threads=self._n_threads,
            verbose=self._verbose,
        )

        logger.info(
            "Loading model: %s (ctx=%d, gpu_layers=%d, backend=%s)",
            self.model_path.name, n_ctx, n_gpu, self._backend.name,
        )

        t_start = time.perf_counter()
        self._model = Llama(**kwargs)
        elapsed = time.perf_counter() - t_start

        logger.info("Model loaded in %.1fs", elapsed)

    def _assert_ready(self) -> None:
        """Raise if the engine is not in a usable state for generation."""
        if not self._loaded:
            raise RuntimeError(
                "TurboQuantEngine is not loaded. Call load() first."
            )
        if self._model is _NOT_LOADED or self._model is None:
            raise RuntimeError(
                "No model is loaded. Provide model_path= to TurboQuantEngine() "
                "or call load_model_path() before generating."
            )

    def _merge_params(self, override: GenerationParams | None) -> dict[str, Any]:
        """Merge override GenerationParams with hardware profile defaults."""
        assert self._params is not None
        defaults = self._params
        p = override or GenerationParams()
        return {
            "max_tokens": p.max_tokens if p.max_tokens is not None else defaults.max_response_tokens,
            "temperature": p.temperature if p.temperature is not None else defaults.temperature,
            "repeat_penalty": p.repeat_penalty if p.repeat_penalty is not None else defaults.repeat_penalty,
            "top_p": p.top_p if p.top_p is not None else defaults.top_p,
            "top_k": p.top_k,
            "stop": p.stop,
            "seed": p.seed,
        }

    def _enforce_token_budget(self, requested_tokens: int) -> int:
        """
        Enforce the hardware profile's token budget.
        Never exceeds token_budget_per_turn; also clips to max_response_tokens.
        """
        assert self._params is not None
        budget = self._params.token_budget_per_turn
        max_resp = self._params.max_response_tokens
        return min(requested_tokens, budget, max_resp)


# ── Convenience factory ────────────────────────────────────────────────────────

def create_engine(
    model_path: str | Path | None = None,
    profile_id: str = "auto",
    backend: str = "auto",
) -> TurboQuantEngine:
    """
    Create and load a TurboQuantEngine with sensible defaults.

    Args:
        model_path:  Path to GGUF model file (optional — can be loaded later).
        profile_id:  Hardware profile ("auto" = auto-detect).
        backend:     Inference backend ("auto" = best available).

    Returns:
        A loaded TurboQuantEngine ready for generation.
    """
    engine = TurboQuantEngine(
        model_path=model_path,
        profile_id=profile_id,
        backend=backend,
    )
    engine.load()
    return engine
