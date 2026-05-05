"""Input validation and sanitisation for ThoughtForge.

Applied at all system boundaries: user queries, config files, model paths, API keys.
Functions never raise on bad input — they sanitise or return validation error lists.
The one exception is validate_model_path(), which raises ModelNotFoundError with a
helpful message when the path doesn't exist.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from thoughtforge.utils.errors import ModelNotFoundError, ValidationError

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_QUERY_CHARS = 4096
MAX_SYSTEM_PROMPT_CHARS = 8192

_REQUIRED_CONFIG_KEYS: set[str] = {"backend"}
_VALID_BACKENDS: set[str] = {
    "ollama", "lmstudio", "openai_compatible", "huggingface", "turboquant", "none", "",
}

# Rough format patterns (format-check only — no network calls)
_HF_TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{10,}$")
_OPENAI_KEY_RE = re.compile(r"^sk-[A-Za-z0-9]{20,}$")


# ── Query sanitisation ────────────────────────────────────────────────────────

def sanitise_query(text: str) -> str:
    """Strip null bytes, control characters, excessive whitespace.
    Normalise unicode. Truncate to MAX_QUERY_CHARS. Never raises — always returns str.

    Args:
        text: Raw user input string.

    Returns:
        Cleaned string, safe to pass into the cognition pipeline.
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove control characters (C0 range) except \t \n \r
    text = "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r") or (not unicodedata.category(ch).startswith("C"))
    )

    # Normalise unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Collapse runs of whitespace (but preserve newlines for multiline queries)
    lines = text.splitlines()
    lines = [" ".join(line.split()) for line in lines]
    text = "\n".join(lines)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Truncate
    if len(text) > MAX_QUERY_CHARS:
        text = text[:MAX_QUERY_CHARS]

    return text


def sanitise_system_prompt(text: str) -> str:
    """Same as sanitise_query but with a larger limit for system prompts."""
    cleaned = sanitise_query(text)
    return cleaned[:MAX_SYSTEM_PROMPT_CHARS]


# ── Config validation ─────────────────────────────────────────────────────────

def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a parsed user_config dict.

    Args:
        config: Dict from yaml.safe_load of user_config.yaml.

    Returns:
        List of human-readable error strings. Empty list = valid.
    """
    errors: list[str] = []

    if not isinstance(config, dict):
        return ["Config must be a YAML mapping, not a scalar or list."]

    # Required keys
    for key in _REQUIRED_CONFIG_KEYS:
        if key not in config:
            errors.append(f"Missing required key: '{key}'")

    # Backend validity
    backend = str(config.get("backend", "")).strip().lower()
    if backend not in _VALID_BACKENDS:
        errors.append(
            f"Unknown backend '{backend}'. "
            f"Valid options: {', '.join(sorted(_VALID_BACKENDS) - {''})}."
        )

    # Backend-specific required fields
    if backend == "ollama":
        if not config.get("ollama_url"):
            errors.append("Ollama backend requires 'ollama_url'.")
    elif backend in ("lmstudio", "openai_compatible"):
        url = config.get("lmstudio_url") or config.get("openai_base_url")
        if not url:
            errors.append(f"{backend} backend requires 'lmstudio_url' or 'openai_base_url'.")
    elif backend == "huggingface":
        if not config.get("hf_model"):
            errors.append("HuggingFace backend requires 'hf_model'.")
    elif backend == "turboquant":
        gguf_path = config.get("gguf_model_path", "")
        if not gguf_path:
            errors.append("TurboQuant backend requires 'gguf_model_path'.")
        elif not Path(str(gguf_path)).exists():
            errors.append(
                f"gguf_model_path does not exist: '{gguf_path}'. "
                "Download the model first."
            )

    return errors


# ── File path validation ──────────────────────────────────────────────────────

def validate_model_path(path: str | Path) -> Path:
    """Validate a GGUF model file path.

    Args:
        path: Path to a .gguf file.

    Returns:
        Resolved Path if valid.

    Raises:
        ModelNotFoundError: If the file does not exist, with a helpful message.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ModelNotFoundError(
            f"Model file not found: {p}",
            context={"path": str(p)},
            suggested_fix=(
                "Download a model via 'python setup_thoughtforge.py' or "
                "set 'gguf_model_path' to an existing .gguf file."
            ),
        )
    if not p.is_file():
        raise ModelNotFoundError(
            f"Model path is not a file: {p}",
            context={"path": str(p)},
        )
    return p


# ── API key format checks ─────────────────────────────────────────────────────

def validate_api_key(key: str, provider: str) -> bool:
    """Format-check an API key (no network calls).

    Args:
        key: The key string to check.
        provider: One of "huggingface", "openai".

    Returns:
        True if the format looks plausible, False otherwise.
        An empty string always returns False.
    """
    if not key or not isinstance(key, str):
        return False

    provider = provider.lower()
    if provider == "huggingface":
        return bool(_HF_TOKEN_RE.match(key))
    if provider in ("openai", "openai_compatible"):
        return bool(_OPENAI_KEY_RE.match(key))

    # Unknown provider — just check it's non-empty and looks like a token
    return len(key) >= 8 and " " not in key


# ── JSONL line validation ─────────────────────────────────────────────────────

def validate_jsonl_line(line: str) -> tuple[bool, str]:
    """Check whether a single JSONL line is valid JSON.

    Returns:
        (True, "") if valid, (False, error_message) if not.
    """
    import json
    stripped = line.strip()
    if not stripped:
        return True, ""   # blank lines are fine
    try:
        json.loads(stripped)
        return True, ""
    except json.JSONDecodeError as exc:
        return False, str(exc)
