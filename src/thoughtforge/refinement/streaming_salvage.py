"""TF-099 — Streaming salvage: what can be rescued
mid-stream vs post-hoc, specified honestly per
strategy.

Roadmap: "Streaming salvage: what can be rescued
mid-stream vs post-hoc, specified honestly per
strategy."

Measured start (probed live this tick):

- **No refinement surface consumed a stream at
  all.** Every module in ``refinement/`` takes
  complete text (``forge`` a completed candidate
  list, the TF-090/091/092 gates a complete
  string); none takes an iterator of chunks.
  TF-026's streamed turn matches: candidate chunks
  are delivered raw as they arrive, and salvage +
  enforcement run only on the completed candidate
  text, their answer arriving as the single
  ``final`` event. Whatever could honestly be
  rescued *while* a candidate is still arriving
  was unspecified and unbuilt.
- **A loop is only provable at its threshold.**
  Probed through TF-091's real gate: one or two
  repeats of "The bell rings." judge ``clean``;
  the third repeat judges ``rescued``. A streaming
  rescuer therefore cannot both deliver a second
  repeat immediately and ever cut the loop — the
  second repeat must be *held* until the next
  sentence either proves the loop (a third) or
  clears it (anything different). Emphasis —
  TF-091's rule that twice may be emphasis — is
  exactly the held sentence's release condition.
- **Some judgements are unsafe on partial text,
  probed.** TF-090's ``enforce_format`` on the
  first 12 chars of a JSON answer (``{"ferry":
  "d``) already returns ``repaired`` — its
  truncation closure would close a value the
  model was still writing. TF-090's
  ``repair_truncation`` on a mid-stream prefix
  cuts a sentence the model had not finished.
  Truncation and format are properties of an
  *ending*: they can only be judged when the
  stream ends, never mid-stream.
- **Per-sentence judgements are the same
  mid-stream as post-hoc.** TF-092's drift gate
  already judges sentence by sentence with
  TF-060's ``detect_language``; a *completed*
  sentence's hint does not change when later
  sentences arrive. The same holds for a
  period-1 sentence run under TF-091's
  normalisation. Those two — and only those
  two — are mid-stream-safe, and only at a
  sentence boundary, by deletion only.

What this slice builds:

- **THE PLACEMENT TABLE** (``STRATEGY_PLACEMENTS``
  as data): every refinement strategy named with
  its placement — ``mid_stream`` (safe at a
  sentence boundary, deletion only),
  ``stream_end`` (judged when the stream closes:
  format, truncation), or ``post_hoc`` (needs
  the completed whole: selection among drafts,
  fragment ranking, refine passes and both of
  their guards, citation/policy verdicts, the
  length/genericness gate, refusal telling, and
  all accounting) — each with the reason, filed
  so a future slice that moves a strategy must
  amend the table against its reason.
- **THE STREAMING RESCUER** (``StreamingSalvage``
  / ``run_streaming_salvage``): a caller-invoked,
  pull-based sentence buffer. Chunks are fed as
  they arrive; a sentence is judged only once
  complete (its terminator seen with following
  whitespace, or at ``finish()``). Mid-stream it
  applies exactly the two mid-stream placements:
  period-1 sentence-loop cutting under TF-091's
  threshold + normalisation (first occurrence
  delivered, second held, third proves the loop
  and both held and further repeats are dropped,
  every drop recorded) and, when the caller
  supplies an expected language, per-sentence
  drift dropping under TF-092's rules (off-
  language completed sentences dropped before
  delivery, unknown-hint sentences neutral and
  kept — never translated). Nothing else is
  applied mid-stream, and nothing is ever
  reworded, completed, or added.
- **THE END OF THE STREAM IS STATED, NOT
  RESCUED PAST ITS GRADE.** ``finish()`` releases
  a held repeat (no third arrived — emphasis),
  treats a buffer ending in a terminator as a
  final complete sentence, and records any
  unterminated tail verbatim as an incomplete
  tail that is *not* delivered and never
  completed (TF-090's truncation gate owns the
  ending's verdict; if no complete sentence ever
  arrived the result says so by name instead of
  dressing the tail as an answer).
- **MID-STREAM IS NOT FINAL.** A streamed
  candidate is a candidate (TF-026): the rescued
  text this module returns still owes the whole
  post-hoc stack — TF-093's policy surface over
  the completed text — before it is an answer.
  The result carries ``post_hoc_still_required``
  so no caller can mistake a mid-stream rescue
  for a final verdict. Majority/whole-answer
  drift flagging (TF-092) likewise stays
  post-hoc: mid-stream drops are per sentence,
  recorded, and if every sentence is dropped the
  result is empty and says why — never a
  minority fragment dressed as the answer, and
  never a silent empty.
- **THE SURFACES ARE UNCHANGED.** TF-091's and
  TF-092's gates, ``FragmentSalvage``, the core
  and TF-026's event contract are untouched;
  this is a separate caller-invoked surface in
  the TF-090 shape. Wiring it into the core's
  ``think_stream`` would change TF-026's filed
  chunk contract (chunks today are the source's
  own, ``native``-marked) — that is a core
  slice's call, not taken here (TF-164 owns
  streaming across bindings).

Proof grade FIXTURE — chunked synthetic streams
(split mid-word and mid-sentence) run through
the real rescuer on this VM, with the period-1
loop legs proven equal to TF-091's post-hoc gate
and the drift legs to TF-092's on the same
texts. A live model's streamed output is NOT
PROVEN (no model in tree), never claimed.

The human-readable write-up is
``docs/architecture/STREAMING_SALVAGE.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

#: This module's estate format.
STREAM_SALVAGE_FORMAT = "thoughtforge-streaming-salvage/1"

STREAM_SALVAGE_DOC = "docs/architecture/STREAMING_SALVAGE.md"

#: The placements a strategy may carry.
PLACEMENTS: tuple[str, ...] = (
    "mid_stream", "stream_end", "post_hoc")

# ── The rules ───────────────────────────────────────────────────────────────

STREAM_RULES: tuple[dict[str, Any], ...] = (
    {"id": "only_sentence_local_deletion_is_mid_stream",
     "rule": "exactly two strategies run mid-stream — "
             "period-1 sentence-loop cutting (TF-091's "
             "share) and per-sentence drift dropping "
             "(TF-092's share) — each judged on a "
             "completed sentence and each deleting only; "
             "every other strategy is stream_end or "
             "post_hoc in STRATEGY_PLACEMENTS, with its "
             "reason filed"},
    {"id": "a_sentence_is_judged_only_when_complete",
     "rule": "a sentence becomes judgeable when its "
             "terminator is seen with following "
             "whitespace, or at finish() when the "
             "buffer ends in a terminator; a partial "
             "sentence is never judged, cut, or "
             "delivered as rescued text"},
    {"id": "the_second_repeat_is_held_not_delivered",
     "rule": "the first repeat of the previous "
             "sentence is held undelivered: a third "
             "proves the loop (held and third and "
             "further repeats dropped, recorded), any "
             "different sentence or finish() releases "
             "it as emphasis — TF-091's threshold, "
             "kept live mid-stream"},
    {"id": "drift_is_per_sentence_never_a_guess",
     "rule": "drift dropping needs a caller-supplied "
             "expected language (TF-092's rule: no "
             "signal, no guess; an unsupported one "
             "raises ValueError by name); a completed "
             "sentence whose TF-060 hint is a known "
             "different language is dropped before "
             "delivery, an unknown-hint sentence is "
             "neutral and kept, and nothing is ever "
             "translated"},
    {"id": "the_tail_is_recorded_never_completed",
     "rule": "an unterminated tail at finish() is "
             "recorded verbatim as an incomplete tail "
             "and not delivered; if no complete "
             "sentence ever arrived the result names "
             "that (no_complete_sentence) instead of "
             "dressing the tail as an answer — TF-090's "
             "truncation gate owns the ending's verdict"},
    {"id": "an_empty_rescue_says_why",
     "rule": "if every sentence is dropped the result "
             "text is empty and the cuts say why, cut "
             "by cut; a streaming rescue never returns "
             "a silent empty and never promotes a "
             "minority fragment to hide one"},
    {"id": "mid_stream_is_not_final",
     "rule": "the rescued text is a candidate, not an "
             "answer: post_hoc_still_required is always "
             "True, TF-093's policy surface still owns "
             "the verdict over the completed text, and "
             "TF-092's majority/whole-answer flagging "
             "stays post-hoc even where its per-sentence "
             "share ran mid-stream"},
    {"id": "pull_based_no_run_ahead",
     "rule": "the rescuer is fed chunk by chunk and "
             "releases a sentence only from chunks "
             "already fed (TF-072's discipline): "
             "run_streaming_salvage pulls one chunk at "
             "a time and never prefetches beyond the "
             "chunk that completed a released sentence"},
    {"id": "the_surfaces_are_unchanged",
     "rule": "TF-091's and TF-092's gates, "
             "FragmentSalvage, the core and TF-026's "
             "event contract are unchanged; this is a "
             "caller-invoked surface, and wiring it "
             "into think_stream is a core slice's call "
             "(TF-164 owns streaming across bindings)"},
)

# ── The placement table ─────────────────────────────────────────────────────
#: Every refinement strategy, its placement, and the
#: reason — judged live against the tree by the
#: data checker (the mid-stream set must be exactly
#: the two sentence-local deletion strategies).

STRATEGY_PLACEMENTS: tuple[dict[str, Any], ...] = (
    {"strategy": "sentence_loop_cut_period_1",
     "owner": "TF-091", "placement": "mid_stream",
     "reason": "a period-1 run is provable from "
               "completed sentences alone under "
               "TF-091's normalisation + threshold "
               "(>= 3), with the second repeat held "
               "until the third proves it; deletion "
               "only"},
    {"strategy": "drift_drop_per_sentence",
     "owner": "TF-092", "placement": "mid_stream",
     "reason": "TF-092 already judges per completed "
               "sentence with TF-060's hints; a "
               "completed sentence's hint does not "
               "change when later sentences arrive; "
               "deletion only, never translation"},
    {"strategy": "format_enforcement_json",
     "owner": "TF-090", "placement": "stream_end",
     "reason": "partial JSON is expected mid-stream "
               "and TF-090's closure already "
               "'repairs' a 12-char prefix — applied "
               "mid-stream it would close values "
               "still being written; validity and "
               "truncation closure are properties of "
               "the ending"},
    {"strategy": "truncation_repair",
     "owner": "TF-090", "placement": "stream_end",
     "reason": "whether a text is truncated is "
               "unknowable until the stream ends; "
               "applied to a mid-stream prefix it "
               "cuts a sentence still being written "
               "(probed)"},
    {"strategy": "draft_scoring_and_selection",
     "owner": "TF-087", "placement": "post_hoc",
     "reason": "selection ranks completed candidates "
               "against each other; a partial "
               "candidate's length/citation score "
               "would rank a draft against text that "
               "does not exist yet"},
    {"strategy": "fragment_extraction",
     "owner": "TF-087", "placement": "post_hoc",
     "reason": "fragments are ranked across the top "
               "completed drafts; mid-stream there is "
               "no ranking to extract against"},
    {"strategy": "engine_refine_passes",
     "owner": "TF-087", "placement": "post_hoc",
     "reason": "a refine pass is a second engine call "
               "over completed fragments (TF-098 "
               "budgets it as one); there is no "
               "completed text to refine mid-stream"},
    {"strategy": "fabrication_guard",
     "owner": "TF-088", "placement": "post_hoc",
     "reason": "the guard judges a complete refine "
               "output against the source pool and "
               "its verdict is an adoption decision "
               "over that whole output; a partial "
               "output's numbers and negations can "
               "still be completed by later chunks"},
    {"strategy": "persona_voice_guard",
     "owner": "TF-096", "placement": "post_hoc",
     "reason": "voice is judged relative to a source "
               "and, at selection, among completed "
               "drafts; a partial draft has no settled "
               "voice to judge"},
    {"strategy": "loop_cut_cycles_and_word_loops",
     "owner": "TF-091", "placement": "post_hoc",
     "reason": "period 2-3 cycles need a multi-"
               "sentence window held back whole, and "
               "word loops run inside and across "
               "sentences whose ends are not yet "
               "known; only the period-1 share is "
               "mid-stream"},
    {"strategy": "drift_majority_flagging",
     "owner": "TF-092", "placement": "post_hoc",
     "reason": "whether drift is the majority, or "
               "the whole answer, is knowable only "
               "when every sentence has arrived"},
    {"strategy": "citation_and_policy_enforcement",
     "owner": "TF-049/TF-093", "placement": "post_hoc",
     "reason": "TF-093's verdict precedence composes "
               "repairs and judgements over the exact "
               "final text; a claim is complete only "
               "in its finished sentence in its "
               "finished answer"},
    {"strategy": "enforcement_gate_length_genericness",
     "owner": "TF-093", "placement": "post_hoc",
     "reason": "length and genericness are "
               "whole-answer properties"},
    {"strategy": "policy_configuration_disposition",
     "owner": "TF-094", "placement": "post_hoc",
     "reason": "strictness disposes of a verdict; "
               "there is no verdict to dispose of "
               "until the post-hoc surface has run"},
    {"strategy": "refusal_honesty",
     "owner": "TF-095", "placement": "post_hoc",
     "reason": "a refusal message tells of a final "
               "block; mid-stream there is no block "
               "to tell of"},
    {"strategy": "rescue_provenance_and_cost",
     "owner": "TF-097/TF-098", "placement": "post_hoc",
     "reason": "the rescue record and its cost are "
               "accounting over completed passes; a "
               "mid-stream total would be revised by "
               "every later pass"},
    {"strategy": "salvage_metrics",
     "owner": "TF-089", "placement": "post_hoc",
     "reason": "rates are measured over a completed "
               "corpus run, never over a stream in "
               "flight"},
)

_PLACEMENT_BY_STRATEGY = {
    p["strategy"]: p for p in STRATEGY_PLACEMENTS}

#: The strategies filed as mid-stream — exactly the
#: two sentence-local deletion strategies, pinned.
MID_STREAM_STRATEGIES: tuple[str, ...] = tuple(
    p["strategy"] for p in STRATEGY_PLACEMENTS
    if p["placement"] == "mid_stream")

# ── Sentence machinery ──────────────────────────────────────────────────────

#: A sentence is complete in the buffer when its
#: terminator is followed by whitespace (more text
#: has arrived past it). At finish(), a terminator
#: at the very end also completes a sentence.
_SENTENCE_END = re.compile(r"[.!?](?=\s)")

_TERMINATORS = (".", "!", "?")


def _sentence_norm(text: str) -> str:
    """TF-091's sentence normalisation, restated:
    casefold, whitespace collapsed, terminal
    punctuation stripped. The live checker proves
    this module's loop legs land exactly where
    TF-091's own gate lands on the same texts."""
    norm = " ".join(text.casefold().split())
    return norm.rstrip(".!?").strip()


@dataclass(frozen=True)
class StreamCut:
    """One mid-stream drop, recorded as it happens."""

    kind: str            # sentence_loop | drift_sentence |
                         # incomplete_tail
    text: str            # the dropped text, verbatim
    detected_language: str | None = None   # drift only

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text,
                "detected_language": self.detected_language}


@dataclass
class StreamSalvageResult:
    """The outcome of one streamed rescue."""

    text: str                       # delivered sentences,
                                    # joined with one space
    verdict: str                    # clean | rescued | empty
    cuts: tuple[StreamCut, ...] = ()
    sentences_delivered: int = 0
    loop_repeats_dropped: int = 0
    drift_sentences_dropped: int = 0
    incomplete_tail: str | None = None
    no_complete_sentence: bool = False
    expected_language: str | None = None
    post_hoc_still_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "verdict": self.verdict,
            "cuts": [c.to_dict() for c in self.cuts],
            "sentences_delivered": self.sentences_delivered,
            "loop_repeats_dropped": self.loop_repeats_dropped,
            "drift_sentences_dropped":
                self.drift_sentences_dropped,
            "incomplete_tail": self.incomplete_tail,
            "no_complete_sentence": self.no_complete_sentence,
            "expected_language": self.expected_language,
            "post_hoc_still_required":
                self.post_hoc_still_required,
        }


class StreamingSalvage:
    """A pull-based, sentence-buffered streaming
    rescuer. Feed chunks as they arrive; each
    ``feed`` returns the sentences that became
    deliverable from that chunk (possibly none,
    possibly with a held repeat still pending).
    ``finish`` closes the stream and returns the
    full result. Deletion only: no sentence is
    ever reworded, completed, or added."""

    def __init__(self, expected_language: str | None = None):
        if expected_language is not None:
            from thoughtforge.knowledge.multilingual import (
                KNOWN_LANGUAGES,
            )

            if expected_language not in KNOWN_LANGUAGES:
                raise ValueError(
                    f"unsupported expected language "
                    f"{expected_language!r}; known languages: "
                    f"{', '.join(KNOWN_LANGUAGES)} — the "
                    "gate never guesses a language")
        self.expected_language = expected_language
        self._buffer = ""
        self._delivered: list[str] = []
        self._cuts: list[StreamCut] = []
        self._last_norm: str | None = None
        self._held_repeat: str | None = None
        self._loop_active = False
        self._loop_dropped = 0
        self._drift_dropped = 0
        self._finished = False

    # ── feeding ─────────────────────────────────────

    def feed(self, chunk: str) -> list[str]:
        """Feed one chunk; return the sentences this
        chunk made deliverable, in order."""
        if self._finished:
            raise ValueError("feed() after finish()")
        self._buffer += chunk
        released: list[str] = []
        while True:
            match = _SENTENCE_END.search(self._buffer)
            if match is None:
                break
            end = match.start() + 1
            sentence = self._buffer[:end].strip()
            self._buffer = self._buffer[end:].lstrip()
            if sentence:
                released.extend(self._judge(sentence))
        return released

    def finish(self) -> StreamSalvageResult:
        """Close the stream: a held repeat is
        released (emphasis), a buffer ending in a
        terminator is a final complete sentence,
        and any other tail is recorded verbatim as
        an incomplete tail, never delivered."""
        if self._finished:
            raise ValueError("finish() called twice")
        self._finished = True
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            if tail.endswith(_TERMINATORS):
                self._judge(tail)
            else:
                self._cuts.append(StreamCut(
                    kind="incomplete_tail", text=tail))
        incomplete = next(
            (c.text for c in self._cuts
             if c.kind == "incomplete_tail"), None)
        # A held repeat with no third is emphasis.
        if self._held_repeat is not None:
            self._deliver(self._held_repeat)
            self._held_repeat = None
        text = " ".join(self._delivered)
        no_complete = not self._delivered and incomplete is not None
        if not self._delivered:
            verdict = "empty"
        elif self._cuts:
            verdict = "rescued"
        else:
            verdict = "clean"
        return StreamSalvageResult(
            text=text, verdict=verdict,
            cuts=tuple(self._cuts),
            sentences_delivered=len(self._delivered),
            loop_repeats_dropped=self._loop_dropped,
            drift_sentences_dropped=self._drift_dropped,
            incomplete_tail=incomplete,
            no_complete_sentence=no_complete,
            expected_language=self.expected_language,
            post_hoc_still_required=True)

    # ── the two mid-stream judgements ───────────────

    def _judge(self, sentence: str) -> list[str]:
        """Judge one completed sentence: drift first
        (a dropped sentence never enters the loop
        state), then the period-1 loop holdback."""
        if self.expected_language is not None:
            from thoughtforge.knowledge.multilingual import (
                detect_language,
            )

            hint = detect_language(sentence)
            if hint.language != "unknown" and \
                    hint.language != self.expected_language:
                self._cuts.append(StreamCut(
                    kind="drift_sentence", text=sentence,
                    detected_language=hint.language))
                self._drift_dropped += 1
                return []
        norm = _sentence_norm(sentence)
        if self._last_norm is None or norm != self._last_norm:
            released = []
            if self._held_repeat is not None:
                released.append(self._held_repeat)
                self._deliver(self._held_repeat)
                self._held_repeat = None
            self._loop_active = False
            self._last_norm = norm
            self._deliver(sentence)
            released.append(sentence)
            return released
        # norm == last delivered norm: a repeat.
        if self._loop_active:
            self._drop_loop_repeat(sentence)
            return []
        if self._held_repeat is None:
            self._held_repeat = sentence
            return []
        # The third occurrence proves the loop: the
        # held second and this third are both dropped.
        held, self._held_repeat = self._held_repeat, None
        self._loop_active = True
        self._drop_loop_repeat(held)
        self._drop_loop_repeat(sentence)
        return []

    def _drop_loop_repeat(self, sentence: str) -> None:
        self._cuts.append(StreamCut(
            kind="sentence_loop", text=sentence))
        self._loop_dropped += 1

    def _deliver(self, sentence: str) -> None:
        self._delivered.append(sentence)


def salvage_stream(
    chunks: Iterable[str],
    expected_language: str | None = None,
) -> Iterator[str]:
    """Yield the rescued stream's deliverable
    sentences as they are released, pulling one
    chunk at a time (TF-072: no run-ahead). The
    final incomplete tail is never yielded; callers
    who need the cuts and the tail use
    ``run_streaming_salvage``."""
    rescuer = StreamingSalvage(
        expected_language=expected_language)
    yielded = 0
    for chunk in chunks:
        for sentence in rescuer.feed(chunk):
            yielded += 1
            yield sentence
    rescuer.finish()
    # finish() may deliver sentences feed() never
    # released — a held repeat cleared as emphasis,
    # a final sentence completed by its terminator
    # at the very end. feed()'s releases are exactly
    # the deliveries made while feeding, in order,
    # so anything delivered past the yielded count
    # is finish()'s, in order.
    yield from rescuer._delivered[yielded:]


def run_streaming_salvage(
    chunks: Iterable[str],
    expected_language: str | None = None,
) -> tuple[StreamSalvageResult, list[str]]:
    """Run a whole chunk stream through the rescuer,
    pulling one chunk at a time. Returns the result
    and the sentences in the order they were
    released (feed releases, then finish's)."""
    rescuer = StreamingSalvage(
        expected_language=expected_language)
    released: list[str] = []
    for chunk in chunks:
        released.extend(rescuer.feed(chunk))
    before = len(rescuer._delivered)
    result = rescuer.finish()
    released.extend(rescuer._delivered[before:])
    return result, released


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic throughout (TF-017's Lantern Hollow,
# never canon). Chunks are split awkwardly — mid-
# word and mid-sentence — on purpose: the rescuer
# must be indifferent to chunk boundaries.

STREAM_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "clean_chunked_mid_word",
     "expected_language": None,
     "chunks": ["The ferry lea", "ves at dawn. The lan",
                "tern keeper checks the river level ",
                "each morning."],
     "expected_text": "The ferry leaves at dawn. The "
                      "lantern keeper checks the river "
                      "level each morning.",
     "expected_cuts": [],
     "expected_verdict": "clean"},
    {"id": "sentence_loop_five",
     "expected_language": None,
     "chunks": ["The bell rings. The bell ri",
                "ngs. The bell rings. The bell rings.",
                " The bell rings."],
     "expected_text": "The bell rings.",
     "expected_cuts": ["sentence_loop"] * 4,
     "expected_verdict": "rescued"},
    {"id": "double_repeat_is_emphasis",
     "expected_language": None,
     "chunks": ["The bell rings. The bell rings.",
                " Wrenna smiles at the sound."],
     "expected_text": "The bell rings. The bell rings. "
                      "Wrenna smiles at the sound.",
     "expected_cuts": [],
     "expected_verdict": "clean"},
    {"id": "double_repeat_at_stream_end",
     "expected_language": None,
     "chunks": ["The bell rings. The bell rings."],
     "expected_text": "The bell rings. The bell rings.",
     "expected_cuts": [],
     "expected_verdict": "clean"},
    {"id": "loop_then_recovery",
     "expected_language": None,
     "chunks": ["The bell rings. The bell rings. ",
                "The bell rings. The bell rings. ",
                "Then the harbour goes quiet."],
     "expected_text": "The bell rings. Then the "
                      "harbour goes quiet.",
     "expected_cuts": ["sentence_loop"] * 3,
     "expected_verdict": "rescued"},
    {"id": "drift_sentence_dropped",
     "expected_language": "en",
     "chunks": ["The ferry leaves at dawn. El ferry ",
                "sale al amanecer desde el muelle ",
                "norte. The lantern keeper checks ",
                "the river level each morning."],
     "expected_text": "The ferry leaves at dawn. The "
                      "lantern keeper checks the river "
                      "level each morning.",
     "expected_cuts": ["drift_sentence"],
     "expected_verdict": "rescued"},
    {"id": "drift_all_dropped_says_why",
     "expected_language": "en",
     "chunks": ["El ferry sale al amanecer. ",
                "El muelle norte esta tranquilo."],
     "expected_text": "",
     "expected_cuts": ["drift_sentence"] * 2,
     "expected_verdict": "empty"},
    {"id": "unknown_hint_sentence_kept",
     "expected_language": "en",
     "chunks": ["The ferry leaves at dawn. ",
                "Lantern Hollow. Wrenna records the ",
                "river level each morning in chalk."],
     "expected_text": "The ferry leaves at dawn. "
                      "Lantern Hollow. Wrenna records "
                      "the river level each morning in "
                      "chalk.",
     "expected_cuts": [],
     "expected_verdict": "clean"},
    {"id": "no_language_no_drift_judgement",
     "expected_language": None,
     "chunks": ["El ferry sale al amanecer desde el ",
                "muelle norte."],
     "expected_text": "El ferry sale al amanecer "
                      "desde el muelle norte.",
     "expected_cuts": [],
     "expected_verdict": "clean"},
    {"id": "incomplete_tail_recorded",
     "expected_language": None,
     "chunks": ["The ferry leaves at dawn. The lan",
                "tern keeper checks the"],
     "expected_text": "The ferry leaves at dawn.",
     "expected_cuts": ["incomplete_tail"],
     "expected_verdict": "rescued"},
    {"id": "tail_only_no_complete_sentence",
     "expected_language": None,
     "chunks": ["The lantern keeper checks the"],
     "expected_text": "",
     "expected_cuts": ["incomplete_tail"],
     "expected_verdict": "empty"},
    {"id": "empty_stream",
     "expected_language": None,
     "chunks": [],
     "expected_text": "",
     "expected_cuts": [],
     "expected_verdict": "empty"},
)

_FIXTURE_BY_ID = {f["id"]: f for f in STREAM_FIXTURES}


def run_stream_fixture(fixture_id: str):
    """Run one filed fixture; returns (result,
    released sentences in release order)."""
    fixture = _FIXTURE_BY_ID[fixture_id]
    return run_streaming_salvage(
        fixture["chunks"],
        expected_language=fixture["expected_language"])


# ── Checkers ────────────────────────────────────────────────────────────────


def streaming_salvage_problems() -> list[str]:
    """Data-integrity checker: rule, placement and
    fixture ids unique; placements valid and covering
    the required strategy set; the mid-stream set
    exactly the two sentence-local deletion
    strategies; every owner named; the TF-091 /
    TF-092 gates and FragmentSalvage carrying no
    streaming surface of their own (this module is
    the one streaming salvage surface)."""
    import inspect

    problems: list[str] = []
    rule_ids = [r["id"] for r in STREAM_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate STREAM_RULES ids")
    strategies = [p["strategy"] for p in STRATEGY_PLACEMENTS]
    if len(strategies) != len(set(strategies)):
        problems.append("duplicate strategy placements")
    fixture_ids = [f["id"] for f in STREAM_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate STREAM_FIXTURES ids")
    for placement in STRATEGY_PLACEMENTS:
        if placement["placement"] not in PLACEMENTS:
            problems.append(
                f"strategy {placement['strategy']}: invalid "
                f"placement {placement['placement']!r}")
        if not placement.get("owner"):
            problems.append(
                f"strategy {placement['strategy']}: no owner")
        if not placement.get("reason"):
            problems.append(
                f"strategy {placement['strategy']}: no reason")
    if set(MID_STREAM_STRATEGIES) != {
            "sentence_loop_cut_period_1",
            "drift_drop_per_sentence"}:
        problems.append(
            "the mid-stream set is not exactly the two "
            "sentence-local deletion strategies")
    required = {
        "draft_scoring_and_selection", "fragment_extraction",
        "engine_refine_passes", "fabrication_guard",
        "persona_voice_guard", "format_enforcement_json",
        "truncation_repair", "loop_cut_cycles_and_word_loops",
        "drift_majority_flagging",
        "citation_and_policy_enforcement",
        "enforcement_gate_length_genericness",
        "policy_configuration_disposition", "refusal_honesty",
        "rescue_provenance_and_cost", "salvage_metrics",
        "sentence_loop_cut_period_1", "drift_drop_per_sentence",
    }
    if set(strategies) != required:
        problems.append(
            f"placement coverage differs: missing "
            f"{sorted(required - set(strategies))}, extra "
            f"{sorted(set(strategies) - required)}")
    from thoughtforge.refinement import (
        drift_rescue,
        loop_rescue,
        salvage,
    )

    for module in (loop_rescue, drift_rescue, salvage):
        source = inspect.getsource(module)
        if "StreamingSalvage" in source:
            problems.append(
                f"{module.__name__} grew a streaming "
                "surface — TF-099's surface is the one "
                "streaming salvage surface")
    return problems


def streaming_salvage_live_problems() -> list[str]:
    """Live checker: every filed fixture lands
    exactly as filed (text, cut kinds in order,
    verdict, released order == delivered text);
    the period-1 loop legs equal TF-091's post-hoc
    gate on the same whole texts; the drift legs
    equal TF-092's rescued text where TF-092
    rescues; pull-based feeding releases nothing
    before a sentence completes; the error legs
    raise by name."""
    problems: list[str] = []
    for fixture in STREAM_FIXTURES:
        result, released = run_stream_fixture(fixture["id"])
        if result.text != fixture["expected_text"]:
            problems.append(
                f"fixture {fixture['id']}: text "
                f"{result.text!r} != "
                f"{fixture['expected_text']!r}")
        kinds = [c.kind for c in result.cuts]
        if kinds != fixture["expected_cuts"]:
            problems.append(
                f"fixture {fixture['id']}: cuts {kinds} "
                f"!= {fixture['expected_cuts']}")
        if result.verdict != fixture["expected_verdict"]:
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{result.verdict} != "
                f"{fixture['expected_verdict']}")
        if " ".join(released) != result.text:
            problems.append(
                f"fixture {fixture['id']}: released order "
                "does not join to the result text")
        if result.post_hoc_still_required is not True:
            problems.append(
                f"fixture {fixture['id']}: "
                "post_hoc_still_required is not True")
    # Equivalence with the post-hoc gates, same texts.
    from thoughtforge.refinement.drift_rescue import (
        rescue_drift,
    )
    from thoughtforge.refinement.loop_rescue import (
        rescue_loops,
    )

    for fixture_id in ("sentence_loop_five",
                       "loop_then_recovery"):
        fixture = _FIXTURE_BY_ID[fixture_id]
        whole = "".join(fixture["chunks"])
        streamed, _ = run_stream_fixture(fixture_id)
        if rescue_loops(whole).text != streamed.text:
            problems.append(
                f"fixture {fixture_id}: streaming loop "
                "result differs from TF-091's post-hoc "
                "gate on the same text")
    fixture = _FIXTURE_BY_ID["drift_sentence_dropped"]
    whole = "".join(fixture["chunks"])
    streamed, _ = run_stream_fixture("drift_sentence_dropped")
    post_hoc = rescue_drift(whole, "en")
    if post_hoc.verdict != "rescued" or \
            post_hoc.text != streamed.text:
        problems.append(
            "the streaming drift leg differs from "
            "TF-092's post-hoc rescue on the same text")
    # Pull-based: nothing is released before a
    # sentence completes, and a held repeat is not
    # released by the chunk that carries it.
    rescuer = StreamingSalvage()
    if rescuer.feed("The bell ri") != []:
        problems.append("a partial sentence was released")
    if rescuer.feed("ngs. The bell rings.") != \
            ["The bell rings."]:
        problems.append("the held repeat was released "
                        "early (or the first was not)")
    try:
        StreamingSalvage(expected_language="fr")
        problems.append("an unsupported expected "
                        "language did not raise")
    except ValueError as exc:
        if "fr" not in str(exc):
            problems.append("the unsupported-language "
                            "error does not name its value")
    return problems


def streaming_salvage_as_data() -> dict[str, Any]:
    return {
        "format": STREAM_SALVAGE_FORMAT,
        "stream_salvage_doc": STREAM_SALVAGE_DOC,
        "rules": [dict(r) for r in STREAM_RULES],
        "placements": [
            dict(p) for p in STRATEGY_PLACEMENTS],
        "mid_stream_strategies": list(MID_STREAM_STRATEGIES),
        "fixtures": [
            {k: v for k, v in f.items()}
            for f in STREAM_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_stream": "not_proven",
            "post_hoc_stack": "still_required",
            "tf091_period_1_equivalence": "proven_on_fixtures",
            "tf092_minority_drift_equivalence":
                "proven_on_fixtures",
            "core_think_stream": "unchanged",
        },
    }
