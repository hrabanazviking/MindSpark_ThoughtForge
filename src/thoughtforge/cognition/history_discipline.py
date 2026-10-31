"""Chat history discipline (TF-024) — budgets, windowing, summarisation hooks.

Before TF-024, ``ChatHistory`` *declared* a window (``max_turns=20``)
but never enforced it: ``add_user`` / ``add_assistant`` appended
without bound, only the OpenAI export applied the window at render
time, and the plain-text render had a char budget but no token
budget and no way to keep what trimming threw away. A long session
therefore grew storage (and any caller that read ``.messages``
directly) without limit, and dropped turns vanished silently.

This module is the discipline, specified as data + pure functions:

- **Budgets** — ``DEFAULT_MAX_TURNS`` (20 turns = 40 messages),
  ``DEFAULT_MAX_TOKENS`` (1,000 word-tokens of stored content) and
  ``DEFAULT_MAX_CHARS`` (2,000 chars for the plain-text render).
  ``CORE_HISTORY_MAX_CHARS`` (1,500) is the core loop's named render
  budget for history injection — previously a magic literal in
  ``core.think``.
- **Token convention** — the tree's own: one token == one
  whitespace-separated word (``core`` counts a final response's
  tokens as ``len(text.split())``; TF-021 pinned the same rule for
  prompt costs). No external tokeniser is imported.
- **Windowing** — :func:`select_window` keeps the most recent
  ``max_turns`` pairs under the token budget. The kept window never
  *opens* with an orphan assistant message when older messages were
  dropped: a reply whose question was trimmed is dropped with it,
  so the model never sees an answer floating without its turn.
  A lone oversized message is never dropped by the token budget —
  the render layer truncates it instead (see ``ChatHistory``).
- **Summarisation hooks** — trimming is destructive, so what it
  drops is offered to a hook: ``Summariser`` is any callable taking
  the dropped messages (oldest first, with the previous summary —
  if any — prepended as a ``system`` message) and returning the
  new summary text. No hook is registered by default; drops are
  then counted on ``ChatHistory.dropped_count`` so the loss is at
  least visible, never silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

# A summariser receives the dropped messages (oldest first; the
# previous summary, when one exists, is prepended by the caller as
# a system-role message) and returns the replacement summary text.
Summariser = Callable[[list], str]

DEFAULT_MAX_TURNS = 20        # turns (user+assistant pairs) kept
DEFAULT_MAX_TOKENS = 1000     # word-tokens of stored message content
DEFAULT_MAX_CHARS = 2000      # plain-text render budget (chars)
CORE_HISTORY_MAX_CHARS = 1500  # core.think's history-injection budget

SUMMARY_CONTEXT_PREFIX = "SUMMARY: "
SUMMARY_MESSAGE_PREFIX = "Summary of earlier conversation: "


def estimate_tokens(text: str) -> int:
    """Word-token estimate — the tree's own counting convention."""
    return len(text.split())


@dataclass(frozen=True)
class HistoryBudget:
    """The three budgets that bound a chat history."""

    max_turns: int = DEFAULT_MAX_TURNS
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_chars: int = DEFAULT_MAX_CHARS

    @property
    def max_messages(self) -> int:
        return self.max_turns * 2


def select_window(
    messages: Sequence,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[list, list]:
    """Split ``messages`` into (kept, dropped) under the budgets.

    Kept = the newest messages, at most ``max_turns * 2`` of them,
    trimmed from the front until their total word-token estimate is
    within ``max_tokens`` — except that a single remaining message
    is always kept, however large (the render layer truncates it).
    If anything was dropped and the kept window would open with an
    assistant message, that orphan is dropped too. Dropped are
    returned oldest-first, in original order, orphan included.
    """
    msgs = list(messages)
    if max_turns <= 0:
        return [], msgs
    kept = msgs[-(max_turns * 2):] if len(msgs) > max_turns * 2 else list(msgs)

    # Token budget: drop oldest until within budget, never the last one.
    total = sum(estimate_tokens(m.content) for m in kept)
    start = 0
    while total > max_tokens and len(kept) - start > 1:
        total -= estimate_tokens(kept[start].content)
        start += 1
    if start:
        kept = kept[start:]

    # Pair alignment: no orphan assistant opening a truncated window.
    if len(kept) < len(msgs) and kept and kept[0].role == "assistant":
        kept = kept[1:]

    # Kept is always a suffix of msgs, so dropped is the exact prefix.
    dropped = msgs[: len(msgs) - len(kept)]
    return kept, dropped
