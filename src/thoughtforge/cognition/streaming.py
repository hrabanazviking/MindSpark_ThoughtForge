"""Streaming turns — the event contract for streamed generation.

TF-026. Before this slice the cognition core was all-or-nothing:
``think()`` ran the whole turn (route → retrieve → scaffold →
generate → salvage → enforce) and returned one finished
``FinalResponseRecord``. No caller could watch a turn arrive.
The backends matched: Ollama and LM Studio were called with
``"stream": False`` and ``TurboQuantEngine.generate`` returned a
completed result only.

The discipline pinned here — *scaffold pre-stream, salvage
post-stream, nothing in between pretending*:

  * **Pre-stream is the whole cognition front half.** Routing,
    retrieval and scaffold building happen *before* the first
    chunk, exactly as in ``think()`` — the prompt a streamed
    candidate is generated from is the same scaffolded candidate
    prompt ``think()`` builds (TF-023's contract). Streaming
    changes delivery, never the thinking that precedes it.
  * **Only candidate generation streams.** A candidate's text may
    arrive incrementally (``candidate_start`` → ``chunk``* →
    ``candidate_end``). The events say so honestly: a chunk event
    carries ``native=True`` only when the generation source itself
    yielded incrementally; when a source only returns completed
    text, the core re-chunks that text for delivery and marks
    every chunk ``native=False`` — simulated streaming is labelled,
    never passed off as the model's own pacing.
  * **Post-stream is not streamed at all.** Fragment scoring,
    salvage (which may drop, reorder or *regenerate* text),
    enforcement (which may append a forge note) and the quality
    gate all operate on completed candidate text. Their output —
    the final answer — therefore arrives as ONE ``final`` event
    carrying the finished ``FinalResponseRecord``. Emitting the
    final text token-by-token would be a lie: no token of it was
    final until salvage and enforcement had run.
  * **A failed candidate is an event, not a silence.** If one
    mode's generation raises, a ``candidate_error`` event carries
    the failure and the turn continues with the remaining modes —
    exactly ``think()``'s per-mode failure handling, made visible.
    If every candidate fails, the turn degrades to the same
    knowledge-only final ``think()`` would have produced.
  * **The turn state is the same object.** A streamed turn builds
    the identical ``RuntimeTurnState`` (TF-025), retained on
    ``ThoughtForgeCore.last_turn_state``; the ``final`` event's
    record IS ``turn.final_response``. Streaming is a second
    *view* of one pipeline, not a second pipeline.

Event order for one turn is fixed: ``turn_started`` first, then
per-candidate groups in scaffold mode order, then exactly one
terminal event — ``final``, or TF-027's ``interrupted`` when the
turn is cut (see ``cognition/interruption.py``). ``chunk_words``
is the one fallback chunker: deterministic, word-preserving,
shared by every non-native path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from thoughtforge.knowledge.models import FinalResponseRecord

EVENT_KINDS: tuple[str, ...] = (
    "turn_started",
    "candidate_start",
    "chunk",
    "candidate_end",
    "candidate_error",
    "final",
    # TF-027: a cut turn's terminal event, INSTEAD of "final" —
    # the event order law is otherwise unchanged (turn_started
    # first, per-candidate groups, exactly one terminal event).
    "interrupted",
)

#: Fallback chunk size, in words, for completed-text sources.
WORDS_PER_CHUNK = 8


@dataclass
class StreamEvent:
    """One event of a streamed turn (see module docstring).

    ``text`` carries: the chunk text on ``chunk`` events, the
    candidate's full assembled text on ``candidate_end``, the
    error message on ``candidate_error``, and the final answer
    text on ``final`` (whose ``final`` field carries the record
    itself). On a TF-027 ``interrupted`` event, ``text`` carries
    the interrupt reason and ``final`` stays None — a cut turn
    has no final record to carry. ``native`` is meaningful on
    ``chunk`` events only.
    """

    kind: str
    turn_id: str
    mode: str = ""
    candidate_id: str = ""
    text: str = ""
    native: bool = False
    final: FinalResponseRecord | None = None


def chunk_words(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> Iterator[str]:
    """Yield ``text`` in deterministic word-preserving chunks.

    Words are split on whitespace and re-joined with single spaces,
    so ``" ".join(chunk_words(t))`` equals the whitespace-normalised
    text — the chunker never invents or drops a word. This is a
    *delivery* chunker for sources that return completed text; the
    candidate record always keeps the source's original text, not
    a reconstruction from chunks.
    """
    if words_per_chunk < 1:
        raise ValueError("words_per_chunk must be >= 1")
    words = text.split()
    for i in range(0, len(words), words_per_chunk):
        yield " ".join(words[i:i + words_per_chunk])
