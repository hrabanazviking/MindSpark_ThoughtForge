# ThoughtForge Architecture — Chat History Discipline (TF-024)

Chat history discipline: windowing, summarisation hooks, and the
token budget, enforced + tested. Built 2026-10-31 on branch
`yrsa/thoughtforge-road-build`.

The discipline lives in `src/thoughtforge/cognition/
history_discipline.py` (budgets as data, the token estimator,
`select_window`) and is enforced by `ChatHistory`
(`src/thoughtforge/cognition/chat_history.py`); the tests are
`tests/test_tf024_history_discipline.py` (26 tests). Before this
slice the history *declared* discipline and practised almost none
of it (§1).

## 1. What was true before

- `max_turns=20` was a **render-time suggestion only**: `add_user` /
  `add_assistant` appended without bound, so stored history (and
  anything reading `.messages` directly, e.g. `format_for_display`)
  grew forever. Only `to_messages()` applied the window, and only
  when someone called it.
- There was **no token budget anywhere**: the plain-text render
  had a char cap (2,000 default; the core passed a magic `1500`),
  nothing counted tokens.
- Trimming was **silent loss**: `trim_to_budget` was manual, and
  what it dropped vanished with no record.
- The render had an **empty-context defect**: `to_prompt_context`
  walked newest-first and `break`ed on the first line that did not
  fit — so a single newest message longer than the budget rendered
  an **empty** history, exactly when the newest turn mattered most.
- The window could **open mid-pair**: `messages[-40:]` can start
  with an assistant reply whose user turn sits just outside the
  window.

## 2. The budgets (as data)

| Budget | Value | Where it binds |
|---|---|---|
| `DEFAULT_MAX_TURNS` | 20 turns (40 messages) | storage + both renders |
| `DEFAULT_MAX_TOKENS` | 1,000 word-tokens | storage + both renders |
| `DEFAULT_MAX_CHARS` | 2,000 chars | `to_prompt_context` default |
| `CORE_HISTORY_MAX_CHARS` | 1,500 chars | `core.think` history injection (was a magic literal; same value, now named) |

Token counting is the tree's own convention — one token per
whitespace-separated word (`estimate_tokens`); the core counts a
final response's tokens the same way (TF-021 pinned this rule for
prompt costs). No external tokeniser is imported. Consequence,
stated plainly: the char budget binds the core's prompt injection
first (1,500 chars ≈ 200–250 word-tokens); the token budget binds
storage and the OpenAI-format export, where no char cap exists.

## 3. Windowing

`select_window(messages, max_turns, max_tokens)` returns
`(kept, dropped)`:

1. keep the newest `max_turns * 2` messages;
2. drop from the front until the kept set's token estimate fits
   `max_tokens` — but **never drop the last remaining message**,
   however large (the render truncates it instead, §5);
3. if anything was dropped and the kept window would open with an
   assistant message, drop that **orphan** too — a reply must not
   reach the model floating without its turn.

`ChatHistory` now calls this on **every append** (and from
`trim_to_budget`, which gained a `max_tokens` parameter), so the
budgets are storage invariants, not export cosmetics. `load()`
deliberately does **not** re-trim: persistence stays faithful to
the file; the next append enforces.

## 4. Summarisation hooks

Trimming is destructive, so what it drops is offered to a hook
instead of vanishing:

- A **summariser** is any callable `list[ChatMessage] -> str`,
  registered as `ChatHistory(summariser=...)`. On each trim it
  receives the dropped messages, oldest first, with the **previous
  summary prepended as a `system` message** when one exists — so
  the hook folds history forward instead of summarising fragments
  in isolation.
- Its return becomes `history.summary`, which rides in **both**
  renders: first line `SUMMARY: …` of the prompt context (charged
  against the char budget like any other line) and a `system`
  message `Summary of earlier conversation: …` in `to_messages()`
  (after the system prompt). Summary and `dropped_count` persist
  through `save` / `load`; pre-TF-024 files load with clean
  defaults.
- **No hook is registered by default and no summariser is shipped
  in this slice** — inventing summary text without a model call
  would be fabrication, and wiring a model-backed summariser is a
  generation concern (the write-back / episode machinery of TF-047
  is the natural owner). Until then, drops are **counted** on
  `dropped_count` — visible, never silent.
- A hook that raises loses nothing else: the exception is logged,
  the trim stands, the summary stays as it was.

## 5. The render fix

`to_prompt_context` now renders from the windowed messages (§3)
and, when the newest line alone exceeds the char budget, **truncates
that line to fit** instead of returning an empty context. A
non-positive budget renders empty, as specified.

## 6. Measured (this VM, 2026-10-31)

Synthetic conversations, counted from the code's own outputs:

- 30 pairs × 8-word messages: storage caps at **40 messages**,
  `dropped_count=20`; `to_messages()` returns 40; default render
  **1,971 chars** (≤2,000); core-budget render **1,453 chars**
  (≤1,500).
- Token cap binding: `max_tokens=50`, 10 pairs × 10-token messages
  → storage caps at **4 messages / 40 tokens**, `dropped_count=16`,
  window still opens on a user message.
- Oversized newest message (~5,000 chars) at a 200-char budget:
  render is **exactly 200 chars**, not empty (pre-TF-024: `""`).

## 7. What this slice does not claim

- No model-backed summariser is wired (§4) — the hook is the
  contract; TF-047 owns a natural caller.
- History still enters the prompt only through the core's
  fact-block injection at 1,500 chars; per-profile history budgets
  belong to TF-032/TF-033.
- Turn-state typing of the history's caller side is TF-025.
