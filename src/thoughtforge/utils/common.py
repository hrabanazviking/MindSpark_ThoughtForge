"""Shared helpers — the one home for the tree's copied private helpers.

TF-025 closes TF-011's D-DUP-1: three helpers existed as drift-prone
copies across the tree —

  * ``_now_iso``          ×3 — knowledge/models.py, cognition/chat_history.py,
                            cognition/core.py (one byte-identical pair)
  * ``_split_sentences``  ×2 — cognition/core.py, refinement/salvage.py
                            (identical body, docstring differed by 3 words)
  * ``_build_messages``   ×2 — inference/ollama_backend.py,
                            inference/lmstudio_backend.py (identical logic)

The implementations now live here, once. The old private names remain
available in their original modules as aliases of these functions
(``_now_iso = now_iso`` and kin), so no importer — including the
existing tests that import ``_split_sentences`` from core and from
salvage — had to change, but there is exactly one body per helper
left to drift.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (the tree's timestamp format)."""
    return datetime.now(timezone.utc).isoformat()


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation + whitespace."""
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_messages(request: Any) -> list[dict]:
    """Chat messages for a GenerationRequest-shaped object.

    If the request already carries ``messages`` they are returned
    unchanged; otherwise the list is built from ``system_prompt``
    (when non-empty) followed by the ``prompt`` as the user message.
    Duck-typed on purpose: both HTTP backends share the
    ``unified_backend.GenerationRequest`` shape.
    """
    if request.messages:
        return request.messages

    msgs: list[dict] = []
    if request.system_prompt:
        msgs.append({"role": "system", "content": request.system_prompt})
    msgs.append({"role": "user", "content": request.prompt})
    return msgs
