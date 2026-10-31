"""LM Studio / generic OpenAI-compatible HTTP backend for ThoughtForge.

Works with LM Studio, vLLM, text-generation-webui, Oobabooga, LocalAI, and any
server that implements the /v1/chat/completions endpoint.
"""

from __future__ import annotations

import logging
import time

import requests

from thoughtforge.inference.openai_compatible import (
    chat_completion_outcome,
    normalize_base_url,
)
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)
# TF-025 (D-DUP-1): _build_messages' one implementation lives in
# utils/common.py; this name is its alias.
from thoughtforge.utils.common import build_messages as _build_messages

logger = logging.getLogger(__name__)


class LMStudioBackend(UnifiedBackend):
    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "",
        api_key: str = "lm-studio",
        timeout: int = 120,
    ) -> None:
        # TF-076 (D-OAI-2): the family's shared normaliser —
        # a base ending in /v1 names the same server, and
        # the routes below are built from the root.
        self.base_url = normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def backend_name(self) -> str:
        label = self.model or "lmstudio"
        return f"lmstudio:{label}"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.monotonic()
        try:
            messages = _build_messages(request)
            payload: dict = {
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
            }
            if self.model:
                payload["model"] = self.model
            if request.stop:
                payload["stop"] = request.stop
            if request.seed is not None:
                # TF-065 seed duty: the OpenAI-shaped seed field.
                payload["seed"] = request.seed

            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # TF-076 (D-OAI-1 / D-OAI-3): the family's ONE
            # shared parser judges the body — finish reasons
            # stay inside the TF-065 vocabulary, content is
            # text or a named error (never None), no choices
            # is a named error, and a missing usage block
            # falls back to the declared contract estimate
            # instead of a silent 0.
            outcome = chat_completion_outcome(data)
            if not outcome["ok"]:
                return GenerationResponse(
                    text="",
                    error=outcome["error"],
                    finish_reason="error",
                    backend_used=self.backend_name(),
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            tokens_gen = outcome["completion_tokens"]
            if tokens_gen is None:
                tokens_gen = self.count_tokens(outcome["text"])

            return GenerationResponse(
                text=outcome["text"],
                tokens_generated=int(tokens_gen),
                finish_reason=outcome["finish_reason"],
                backend_used=self.backend_name(),
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            logger.error("LMStudioBackend.generate error: %s", exc)
            return GenerationResponse(
                text="",
                error=str(exc),
                finish_reason="error",
                backend_used=self.backend_name(),
                latency_ms=(time.monotonic() - start) * 1000,
            )

    def health_check(self) -> bool:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=headers,
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def capabilities(self):
        """TF-065 contract record: simulated streaming (no
        generate_stream override), estimate token counting
        (the OpenAI-compatible shape guarantees no tokeniser
        endpoint), seed honoured via the seed field, served
        by a local-or-remote server."""
        from thoughtforge.inference.backend_contract import BackendCapabilities

        return BackendCapabilities(
            name=self.backend_name(),
            streaming="simulated",
            token_counting="estimate",
            seed="honoured",
            placement="local_or_remote",
            model=self.model,
        )

    def list_models(self) -> list[str]:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as exc:
            logger.warning("LMStudioBackend.list_models failed: %s", exc)
            return []

