"""Typed exception hierarchy for MindSpark: ThoughtForge.

Every exception carries:
  - message      — human-readable description
  - context      — dict of relevant state at the time of the error
  - recoverable  — True if self-healing is plausible
  - suggested_fix — one-line user-facing advice

Usage:
    raise BackendUnavailableError(
        "Ollama not reachable",
        context={"url": base_url},
        suggested_fix="Run: ollama serve",
    )
"""

from __future__ import annotations

from typing import Any


class ThoughtForgeError(Exception):
    """Base class for all ThoughtForge errors."""

    def __init__(
        self,
        message: str = "",
        *,
        context: dict[str, Any] | None = None,
        recoverable: bool = False,
        suggested_fix: str = "",
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = context or {}
        self.recoverable: bool = recoverable
        self.suggested_fix: str = suggested_fix

    def __str__(self) -> str:
        parts = [self.message]
        if self.suggested_fix:
            parts.append(f"Fix: {self.suggested_fix}")
        return " — ".join(parts)


# ── Config errors ──────────────────────────────────────────────────────────────

class ConfigError(ThoughtForgeError):
    """Problem with configuration files."""


class ConfigMissingError(ConfigError):
    """Required configuration file not found."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("suggested_fix", "Run: python setup_thoughtforge.py")
        super().__init__(message, **kwargs)


class ConfigCorruptError(ConfigError):
    """Configuration file exists but cannot be parsed."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("suggested_fix", "Run: python forge_doctor.py --fix")
        super().__init__(message, **kwargs)


# ── Backend errors ─────────────────────────────────────────────────────────────

class BackendError(ThoughtForgeError):
    """Problem communicating with a generation backend."""


class BackendUnavailableError(BackendError):
    """Backend is not reachable (not running, wrong port, etc.)."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("suggested_fix", "Check that your backend is running (e.g. ollama serve)")
        super().__init__(message, **kwargs)


class BackendTimeoutError(BackendError):
    """Backend took too long to respond."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("suggested_fix", "Try a smaller model or increase timeout in config")
        super().__init__(message, **kwargs)


class BackendAuthError(BackendError):
    """Authentication with backend failed (bad API key, etc.)."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", False)
        kwargs.setdefault("suggested_fix", "Check your API key in configs/user_config.yaml")
        super().__init__(message, **kwargs)


class ModelNotFoundError(BackendError):
    """The requested model does not exist on the backend."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("suggested_fix", "Pull the model first: ollama pull <model>")
        super().__init__(message, **kwargs)


# ── Knowledge / DB errors ──────────────────────────────────────────────────────

class KnowledgeError(ThoughtForgeError):
    """Problem with the knowledge database or retrieval."""


class DatabaseCorruptError(KnowledgeError):
    """SQLite database failed integrity check."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("suggested_fix", "Run: python forge_doctor.py --fix")
        super().__init__(message, **kwargs)


class DatabaseLockedError(KnowledgeError):
    """SQLite database is locked by another process."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("suggested_fix", "Close other ThoughtForge instances and retry")
        super().__init__(message, **kwargs)


class RetrievalError(KnowledgeError):
    """Knowledge retrieval failed unexpectedly."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("suggested_fix", "Try rebuilding the knowledge base: python forge_memory.py reference")
        super().__init__(message, **kwargs)


# ── Memory errors ──────────────────────────────────────────────────────────────

class MemoryStoreError(ThoughtForgeError):
    """Problem with the memory store (episodic, preferences, etc.)."""


class MemoryFileCorruptError(MemoryStoreError):
    """A memory file contains invalid data."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        kwargs.setdefault("suggested_fix", "Run: python forge_doctor.py --fix to repair memory files")
        super().__init__(message, **kwargs)


class MemoryWriteError(MemoryStoreError):
    """Failed to write a memory entry."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", False)
        kwargs.setdefault("suggested_fix", "Check disk space and file permissions in the memory directory")
        super().__init__(message, **kwargs)


# ── Pipeline errors ────────────────────────────────────────────────────────────

class PipelineError(ThoughtForgeError):
    """Problem in the cognition pipeline."""


class ScaffoldError(PipelineError):
    """Failed to build the cognition scaffold."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", True)
        super().__init__(message, **kwargs)


class EnforcementError(PipelineError):
    """Response failed the enforcement gate and could not be salvaged."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", False)
        super().__init__(message, **kwargs)


# ── Validation errors ──────────────────────────────────────────────────────────

class ValidationError(ThoughtForgeError):
    """Input or data failed validation."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("recoverable", False)
        super().__init__(message, **kwargs)
