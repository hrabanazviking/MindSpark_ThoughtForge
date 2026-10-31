"""Ollama HTTP backend for ThoughtForge."""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

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


class OllamaBackend(UnifiedBackend):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def backend_name(self) -> str:
        return f"ollama:{self.model}"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.monotonic()
        try:
            messages = _build_messages(request)
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            }
            if request.stop:
                payload["options"]["stop"] = request.stop  # type: ignore[index]
            if request.seed is not None:
                # TF-065 seed duty: Ollama honours a sampler seed
                # through options.seed — forward the carried seed.
                payload["options"]["seed"] = request.seed  # type: ignore[index]

            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            text: str = data.get("message", {}).get("content", "")
            finish_reason = "stop" if data.get("done") else "length"
            tokens_gen: int = data.get("eval_count", 0)

            return GenerationResponse(
                text=text,
                tokens_generated=tokens_gen,
                finish_reason=finish_reason,
                backend_used=self.backend_name(),
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            logger.error("OllamaBackend.generate error: %s", exc)
            return GenerationResponse(
                text="",
                error=str(exc),
                finish_reason="error",
                backend_used=self.backend_name(),
                latency_ms=(time.monotonic() - start) * 1000,
            )

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def capabilities(self):
        """TF-065 contract record: simulated streaming (no
        generate_stream override), estimate token counting
        (no tokeniser endpoint is assumed), seed honoured
        via options.seed, served by a local-or-remote Ollama."""
        from thoughtforge.inference.backend_contract import BackendCapabilities

        return BackendCapabilities(
            name=self.backend_name(),
            streaming="simulated",
            token_counting="estimate",
            seed="honoured",
            placement="local_or_remote",
            model=self.model,
        )

    def model_listed(self) -> bool | None:
        """TF-077 (D-HEALTH-2): is the configured model in
        the server's /api/tags list? Tri-state, mirroring
        TF-076's surface: None when the server could not
        be asked — unknown, never dressed as absent (the
        older list_local_models() returns [] for both).
        Ollama names carry tags; a configured name without
        one matches the server's implied ':latest'."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            names = [m.get("name") for m in data.get("models", [])
                     if isinstance(m, dict)]
            if self.model in names:
                return True
            if ":" not in self.model \
                    and f"{self.model}:latest" in names:
                return True
            return False
        except Exception:  # noqa: BLE001 — unknown, stated
            return None

    def list_local_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.warning("OllamaBackend.list_local_models failed: %s", exc)
            return []

    def pull_model(
        self,
        model_name: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> bool:
        """
        Pull a model from the Ollama library.
        Streams progress lines; calls progress_callback(status) for each one if supplied.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": True},
                stream=True,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    line_data = json.loads(raw_line)
                    status: str = line_data.get("status", "")
                    if progress_callback and status:
                        progress_callback(status)
                    if line_data.get("error"):
                        logger.error("Ollama pull error: %s", line_data["error"])
                        return False
                except json.JSONDecodeError:
                    pass
            return True
        except Exception as exc:
            logger.error("OllamaBackend.pull_model failed: %s", exc)
            return False

