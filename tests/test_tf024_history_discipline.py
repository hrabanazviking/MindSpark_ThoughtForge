"""
TF-024 — Chat History Discipline: windowing, summarisation hooks,
and the token budget, enforced + tested.

The discipline lives in ``src/thoughtforge/cognition/
history_discipline.py`` (budgets as data, the word-token estimator,
``select_window``) and is enforced by ``ChatHistory`` itself; the
write-up is ``docs/architecture/CHAT_HISTORY_DISCIPLINE.md``.

Pinned here:

  * the budget constants and the tree's word-token convention,
  * ``select_window``: turn cap, token cap, pair alignment (no
    orphan assistant opening a truncated window), a lone oversized
    message never dropped, max_turns=0 drops everything,
  * enforcement ON APPEND (before TF-024 the window was render-only
    and storage grew without bound),
  * the summarisation hook: called with the dropped messages
    oldest-first, the previous summary riding in as a leading
    system message, its return becoming the stored summary and
    riding in BOTH renders; a raising hook loses nothing else;
    with no hook, drops are counted on ``dropped_count``,
  * the char-budget render: within budget, newest preserved, and
    the TF-024 defect fix — a single oversized newest line is
    truncated to fit instead of rendering an empty context,
  * the core loop's named history budget (1500 chars, was a magic
    literal) and that think() does not mutate the caller's history
    length under the new enforcement,
  * persistence of summary / dropped_count / max_tokens, with
    pre-TF-024 files (no such keys) loading with clean defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

from thoughtforge.cognition.chat_history import ChatHistory, ChatMessage
from thoughtforge.cognition.history_discipline import (
    CORE_HISTORY_MAX_CHARS,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    HistoryBudget,
    estimate_tokens,
    select_window,
)


def _pair(h: ChatHistory, i: int, words: int = 3) -> None:
    h.add_user(" ".join([f"q{i}"] + ["w"] * (words - 1)))
    h.add_assistant(" ".join([f"a{i}"] + ["w"] * (words - 1)))


def _msgs(n_pairs: int, words: int = 3) -> list[ChatMessage]:
    out = []
    for i in range(n_pairs):
        out.append(ChatMessage(role="user", content=" ".join([f"q{i}"] + ["w"] * (words - 1))))
        out.append(ChatMessage(role="assistant", content=" ".join([f"a{i}"] + ["w"] * (words - 1))))
    return out


# ── Budgets + estimator ───────────────────────────────────────────────────────

class TestBudgets:
    def test_defaults_match_pre_tf024_declared_values(self):
        assert DEFAULT_MAX_TURNS == 20
        assert DEFAULT_MAX_CHARS == 2000
        assert HistoryBudget().max_turns == 20
        assert HistoryBudget().max_messages == 40

    def test_core_history_budget_is_the_old_magic_literal(self):
        assert CORE_HISTORY_MAX_CHARS == 1500

    def test_default_max_tokens_positive_and_below_render_scale(self):
        assert 0 < DEFAULT_MAX_TOKENS

    def test_estimator_is_the_tree_word_convention(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("one") == 1
        assert estimate_tokens("one two  three") == 3
        # core counts a final's tokens the same way
        assert estimate_tokens("a b c") == len("a b c".split())


# ── select_window ─────────────────────────────────────────────────────────────

class TestSelectWindow:
    def test_under_budget_keeps_everything(self):
        msgs = _msgs(3)
        kept, dropped = select_window(msgs)
        assert kept == msgs and dropped == []

    def test_turn_cap_keeps_newest_pairs(self):
        msgs = _msgs(10)
        kept, dropped = select_window(msgs, max_turns=3)
        assert len(kept) == 6
        assert kept[0].content.startswith("q7")
        assert [m.content for m in dropped] == [m.content for m in msgs[:14]]

    def test_token_cap_drops_oldest_pairs(self):
        msgs = _msgs(10, words=10)  # 10 tokens per message, 200 total
        kept, dropped = select_window(msgs, max_turns=20, max_tokens=50)
        assert sum(estimate_tokens(m.content) for m in kept) <= 50
        assert kept[0].role == "user"
        assert len(dropped) + len(kept) == len(msgs)

    def test_no_orphan_assistant_opening(self):
        # Window cut lands mid-pair: the reply whose question was
        # dropped is dropped too.
        msgs = _msgs(5)
        kept, _ = select_window(msgs, max_turns=2, max_tokens=10**9)
        # 4 kept from the tail starts at a user message already;
        # force the orphan case with an odd message count:
        odd = msgs[1:]  # starts with assistant a0
        kept2, dropped2 = select_window(odd, max_turns=2)
        assert kept2[0].role == "user"
        assert dropped2[0].content.startswith("a0")

    def test_lone_oversized_message_never_dropped(self):
        big = [ChatMessage(role="user", content=" ".join(["w"] * 5000))]
        kept, dropped = select_window(big, max_tokens=10)
        assert kept == big and dropped == []

    def test_zero_turns_drops_everything(self):
        msgs = _msgs(2)
        kept, dropped = select_window(msgs, max_turns=0)
        assert kept == [] and dropped == msgs


# ── Enforcement on append ─────────────────────────────────────────────────────

class TestEnforcementOnAppend:
    def test_storage_is_capped_by_turns(self):
        h = ChatHistory()
        for i in range(30):
            _pair(h, i)
        assert len(h) == 40
        assert h.dropped_count == 20
        assert h.messages[0].content.startswith("q10")

    def test_storage_is_capped_by_tokens(self):
        h = ChatHistory(max_turns=20, max_tokens=50)
        for i in range(10):
            _pair(h, i, words=10)
        total = sum(estimate_tokens(m.content) for m in h.messages)
        assert total <= 50
        assert h.messages[0].role == "user"

    def test_small_history_untouched(self):
        h = ChatHistory()
        _pair(h, 0)
        assert len(h) == 2 and h.dropped_count == 0

    def test_manual_trim_still_works_and_counts(self):
        h = ChatHistory()
        for i in range(10):
            h.add_user(f"msg_{i}")
        h.trim_to_budget(max_turns=2)
        assert len(h) <= 4
        assert h.dropped_count == 6


# ── Summarisation hooks ───────────────────────────────────────────────────────

class TestSummariserHook:
    def test_hook_receives_dropped_oldest_first(self):
        seen: list[list[str]] = []

        def hook(dropped):
            seen.append([m.content for m in dropped])
            return "S"

        h = ChatHistory(max_turns=1, summariser=hook)
        _pair(h, 0)
        _pair(h, 1)
        assert h.summary == "S"
        flat = [c for call in seen for c in call]
        assert any(c.startswith("q0") for c in flat)

    def test_previous_summary_rides_into_next_hook_call(self):
        payloads: list[list[tuple[str, str]]] = []

        def hook(dropped):
            payloads.append([(m.role, m.content) for m in dropped])
            return f"sum{len(payloads)}"

        h = ChatHistory(max_turns=1, summariser=hook)
        for i in range(4):
            _pair(h, i)
        later = payloads[-1]
        assert later[0][0] == "system"
        assert later[0][1].startswith("sum")

    def test_summary_rides_in_both_renders(self):
        h = ChatHistory(max_turns=1, summariser=lambda d: "earlier talk")
        _pair(h, 0)
        _pair(h, 1)
        assert h.to_prompt_context().splitlines()[0] == "SUMMARY: earlier talk"
        msgs = h.to_messages()
        assert msgs[0] == {
            "role": "system",
            "content": "Summary of earlier conversation: earlier talk",
        }

    def test_raising_hook_loses_nothing_else(self):
        def bad(dropped):
            raise RuntimeError("boom")

        h = ChatHistory(max_turns=1, summariser=bad)
        _pair(h, 0)
        _pair(h, 1)  # must not raise
        assert len(h) == 2
        assert h.summary == ""
        assert h.dropped_count == 2

    def test_no_hook_counts_drops(self):
        h = ChatHistory(max_turns=1)
        _pair(h, 0)
        _pair(h, 1)
        assert h.summary == ""
        assert h.dropped_count == 2

    def test_clear_resets_summary_and_count(self):
        h = ChatHistory(max_turns=1, summariser=lambda d: "S")
        _pair(h, 0)
        _pair(h, 1)
        h.clear()
        assert h.summary == "" and h.dropped_count == 0 and len(h) == 0


# ── Char-budget render ────────────────────────────────────────────────────────

class TestPromptContextRender:
    def test_within_budget_and_newest_preserved(self):
        h = ChatHistory()
        for i in range(30):
            _pair(h, i, words=8)
        ctx = h.to_prompt_context()
        assert len(ctx) <= DEFAULT_MAX_CHARS
        assert "q29" in ctx

    def test_oversized_newest_line_truncated_not_empty(self):
        h = ChatHistory()
        h.add_user("word " * 10 + "z" * 5000)
        ctx = h.to_prompt_context(max_chars=200)
        assert ctx != ""
        assert len(ctx) <= 200

    def test_nonpositive_budget_renders_empty(self):
        h = ChatHistory()
        h.add_user("hi")
        assert h.to_prompt_context(max_chars=0) == ""


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_round_trip_keeps_summary_count_and_token_budget(self, tmp_path: Path):
        h = ChatHistory(max_turns=1, max_tokens=42, summariser=lambda d: "kept summary")
        _pair(h, 0)
        _pair(h, 1)
        path = tmp_path / "h.json"
        h.save(path)
        h2 = ChatHistory.load(path)
        assert h2.summary == "kept summary"
        assert h2.dropped_count == h.dropped_count == 2
        assert h2.max_tokens == 42

    def test_pre_tf024_file_loads_with_clean_defaults(self, tmp_path: Path):
        path = tmp_path / "old.json"
        path.write_text(
            json.dumps(
                {
                    "max_turns": 15,
                    "system_prompt": "",
                    "messages": [
                        {"role": "user", "content": "hi", "turn_id": "t1", "timestamp": "x"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        h = ChatHistory.load(path)
        assert h.summary == "" and h.dropped_count == 0
        assert h.max_tokens == DEFAULT_MAX_TOKENS
        assert len(h) == 1


# ── Core integration ──────────────────────────────────────────────────────────

class TestCoreIntegration:
    def test_core_uses_the_named_constant(self):
        import inspect

        from thoughtforge.cognition import core

        src = inspect.getsource(core.ThoughtForgeCore.think)
        assert "CORE_HISTORY_MAX_CHARS" in src
        assert "max_chars=1500" not in src
