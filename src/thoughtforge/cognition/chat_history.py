"""Multi-turn conversation history with OpenAI-format export and persistence.

History discipline (TF-024 — see
``thoughtforge.cognition.history_discipline`` and
``docs/architecture/CHAT_HISTORY_DISCIPLINE.md``): the window and
the token budget are ENFORCED on every append, not merely applied
at render time; the plain-text render is char-budgeted and never
returns empty just because the newest message is oversized; and
whatever trimming drops is offered to the registered summarisation
hook (or counted on ``dropped_count`` when no hook is registered).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from thoughtforge.cognition.history_discipline import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    SUMMARY_CONTEXT_PREFIX,
    SUMMARY_MESSAGE_PREFIX,
    Summariser,
    select_window,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChatMessage:
    role: str           # "user" | "assistant" | "system"
    content: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = field(default_factory=_now_iso)


class ChatHistory:
    def __init__(
        self,
        max_turns: int = DEFAULT_MAX_TURNS,
        system_prompt: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        summariser: Summariser | None = None,
    ) -> None:
        self.messages: list[ChatMessage] = []
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt.strip()
        self.summariser = summariser
        self.summary: str = ""
        self.dropped_count: int = 0

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_user(self, text: str) -> str:
        """Append a user message and return the assigned turn_id."""
        msg = ChatMessage(role="user", content=text)
        self.messages.append(msg)
        self.trim_to_budget()
        return msg.turn_id

    def add_assistant(self, text: str, turn_id: str = "") -> None:
        msg = ChatMessage(
            role="assistant",
            content=text,
            turn_id=turn_id or uuid.uuid4().hex[:8],
        )
        self.messages.append(msg)
        self.trim_to_budget()

    def clear(self) -> None:
        self.messages = []
        self.summary = ""
        self.dropped_count = 0

    # ── Context rendering ─────────────────────────────────────────────────────

    def to_messages(self) -> list[dict]:
        """
        Return an OpenAI-format messages list.
        System prompt is prepended if set, then the summary of
        trimmed-away turns (if a summariser produced one), then the
        windowed messages (pair-aligned, token-budgeted).
        """
        window, _ = select_window(
            self.messages, max_turns=self.max_turns, max_tokens=self.max_tokens
        )
        result: list[dict] = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        if self.summary:
            result.append(
                {"role": "system", "content": f"{SUMMARY_MESSAGE_PREFIX}{self.summary}"}
            )
        result.extend({"role": m.role, "content": m.content} for m in window)
        return result

    def to_prompt_context(self, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """
        Render conversation as plain text for non-chat backends.
        Keeps the newest lines that fit under max_chars, oldest of
        the kept set first; the summary (if any) rides first and is
        charged against the same budget. A single newest line longer
        than the whole budget is truncated to fit — never dropped
        into an empty render.
        """
        if max_chars <= 0:
            return ""
        lines: list[str] = []
        if self.summary:
            lines.append(f"{SUMMARY_CONTEXT_PREFIX}{self.summary}")
        lines.extend(
            f"{m.role.upper()}: {m.content}"
            for m in select_window(
                self.messages, max_turns=self.max_turns, max_tokens=self.max_tokens
            )[0]
        )
        kept: list[str] = []
        total = 0
        for line in reversed(lines):
            if total + len(line) + 1 > max_chars:
                if not kept:
                    # Newest line alone exceeds the budget: truncate
                    # it to fit rather than rendering nothing.
                    kept.append(line[:max_chars])
                break
            kept.insert(0, line)
            total += len(line) + 1
        return "\n".join(kept)

    def trim_to_budget(
        self,
        max_turns: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Enforce the turn window + token budget on stored messages.

        Dropped messages go to the summarisation hook when one is
        registered (the previous summary rides in as a leading
        system message, so the hook can fold it forward); otherwise
        they are counted on ``dropped_count``.
        """
        kept, dropped = select_window(
            self.messages,
            max_turns=self.max_turns if max_turns is None else max_turns,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        if not dropped:
            return
        self.messages = kept
        self.dropped_count += len(dropped)
        if self.summariser is not None:
            try:
                payload = list(dropped)
                if self.summary:
                    payload.insert(
                        0,
                        ChatMessage(role="system", content=self.summary),
                    )
                new_summary = self.summariser(payload)
                if new_summary:
                    self.summary = str(new_summary).strip()
            except Exception as exc:  # a bad hook must not lose the turn
                logger.warning("Summarisation hook failed: %s", exc)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "summary": self.summary,
            "dropped_count": self.dropped_count,
            "messages": [asdict(m) for m in self.messages],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.debug("ChatHistory saved to %s (%d messages)", path, len(self.messages))

    @classmethod
    def load(cls, path: Path) -> ChatHistory:
        with path.open("r", encoding="utf-8") as f:
            payload: dict = json.load(f)
        history = cls(
            max_turns=payload.get("max_turns", DEFAULT_MAX_TURNS),
            max_tokens=payload.get("max_tokens", DEFAULT_MAX_TOKENS),
            system_prompt=payload.get("system_prompt", ""),
        )
        history.summary = payload.get("summary", "")
        history.dropped_count = int(payload.get("dropped_count", 0))
        for raw in payload.get("messages", []):
            history.messages.append(
                ChatMessage(
                    role=raw["role"],
                    content=raw.get("content", raw.get("text", "")),
                    turn_id=raw.get("turn_id", uuid.uuid4().hex[:8]),
                    timestamp=raw.get("timestamp", _now_iso()),
                )
            )
        logger.debug("ChatHistory loaded from %s (%d messages)", path, len(history.messages))
        return history

    # ── Convenience ───────────────────────────────────────────────────────────

    def format_for_display(self) -> str:
        if not self.messages:
            return "(no history)"
        lines = []
        for m in self.messages:
            prefix = "You   >" if m.role == "user" else "Forge >"
            lines.append(f"{prefix} {m.content}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.messages)
