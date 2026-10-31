"""
The long-conversation proof — a 100-turn synthetic saga
(TF-063).

The roadmap asks: *100-turn synthetic saga; memory stays
coherent, citations stay true (Part VII harness).*

The measured starting state, before this slice
(probed this tick, keeper 10:22, on this VM):

  * **The longest conversation proof in the tree was
    20 turns** (TF-028's ``COHERENCE_SCRIPT``), and it
    proved the *history* channel: facts established in
    chat, carried in the threaded ``ChatHistory``, and
    scored for drift in the answers. Nothing ran 100
    turns.
  * **No proof exercised the personal store over a
    long conversation.** TF-028's fixture engine never
    touches ``MemoryStore``; the store-side slices
    (TF-041–TF-062) each prove one operation in
    isolation — a handful of records, a handful of
    retrievals. Whether retrieval still returns the
    *right* record, with a *true* citation, at turn
    100 as reliably as at turn 1 was unproven.
  * **Citation truth was never scored mechanically
    over a conversation.** TF-048 proved every
    retrieved item *carries* a payload; TF-049 judges
    a single answer's claims. No harness checked,
    turn after turn, that the citation an answer
    carries *resolves* to a real record and that the
    record it resolves to *actually says what the
    answer claims* — the two ways a citation can lie
    (fabricated marker; real marker, wrong record).

What this slice builds — ``knowledge/long_conversation.py``:

  * **The position, stated as data**
    (``SAGA_RULES``): the saga is exactly 100 turns;
    memory is the **store, not the history** — the
    runner carries no chat history between turns, so
    a fact survives only if the store holds it and
    retrieval finds it (the history channel is
    TF-028's proof, and TF-024's budgets drop old
    history by rule — a 100-turn conversation cannot
    rest on it); a citation is true only when it
    resolves **and** the resolved record's own text
    is what the answer states; a contradiction or a
    false citation zeroes its turn; the saga is
    synthetic and deterministic (TF-017's Lantern
    Hollow, never canon; no model, no sampling).
  * **The facts** (``SAGA_FACTS``): ten synthetic
    facts, each landed in a ``MemoryStore`` through
    the store's own gated append path (TF-042's
    contract + TF-062's screen both apply — the saga
    facts pass both, pinned). Each fact carries its
    probe question, the required phrase groups an
    answer must state, and the forbidden phrases
    that contradict it (TF-028's anchor shape).
  * **The script** (``SAGA_SCRIPT``): 100 turns
    generated as data — ten rounds × the ten facts,
    each round re-asking every fact's question in a
    new frame (``ROUND_FRAMES``), so every fact is
    probed ten times spread across the whole saga,
    first at turns 1–10 and last at turns 91–100.
    Every user turn is unique (pinned).
  * **The grounded answerer** (the runner's model
    stand-in): each turn, the runner retrieves from
    the store with TF-060's personal-store search
    and answers **from the retrieved record's own
    summary, verbatim**, appending that record's
    TF-048 citation marker in brackets. This is the
    honest fixture for a memory-grounded model: if
    retrieval returns the wrong record, the answer
    is the wrong record's text with the wrong
    record's citation, and the scorer catches both.
    An empty retrieval answers the stated fallback
    with no citation — never a fabricated one.
  * **The scorer** (:func:`score_saga`, pure over a
    transcript + the store's marker map): per turn —
    retention over the probed fact's required groups;
    **zero** on any forbidden phrase; **zero** on a
    citation failure, classified as *missing* (no
    bracketed marker), *fabricated* (the marker
    resolves to no record in the store), or *untrue*
    (the marker resolves to a different record, or
    the answer's stated text is not that record's
    summary). The report carries the first window
    (turns 1–10), the last window (turns 91–100),
    **drift = first − last**, citation-failure and
    fabricated-citation counts, and TF-028's verdict
    shape: ``PASS`` only when the last window holds,
    drift is small, and there are no contradiction
    turns and no citation failures; an empty
    transcript is ``REVIEW`` at 0.0, never a vacuous
    pass.
  * **The runner** (:func:`run_saga`): seeds nothing
    itself — it runs over a store the caller built
    with :func:`build_saga_store` — retrieves per
    turn, records retrieval accuracy (the top
    record is the probed fact) beside the score, and
    proves the saga **read-only**: the store's bytes
    are snapshotted before and after and must be
    identical (TF-060's search is read-only; the
    saga adds no write of its own). A
    ``drift_from_turn`` option makes the answerer
    answer with the *next* fact's text and citation
    from that turn on — a coherent-looking,
    fully-cited, wrong answer — giving the scorer
    ground truth to catch in the last window.

Stated boundaries:

  * **Fixture grade.** The answerer is a
    deterministic grounded stand-in, not a live
    model: this slice proves the *store + retrieval
    + citation* leg over 100 turns, and that the
    harness catches drift, contradiction, and false
    citations when they are injected. A live small
    model's behaviour over 100 turns is Part VII's
    road (TF-125 re-runs saga-shaped scripts against
    real models) and is NOT PROVEN here.
  * **One retrieval leg.** The saga retrieves
    through TF-060's personal-store search (the
    store's own search, TF-062-screened). The
    knowledge-DB forge leg over 100 turns is not
    claimed: no real lived-in store or DB exists on
    this machine (TF-041).
  * **Citation truth is textual, by rule.** "True"
    means the cited record's stored summary is
    exactly the text the answer states (folded
    comparison, TF-060's ``fold_text``) — a stated,
    mechanical rule in TF-049's spirit, not a
    semantic judgement about paraphrase. A paraphrasing
    live model would be judged by TF-049's claim
    check, not by this scorer.
  * **The store is unchanged by the saga** — no
    writeback (TF-047), lifecycle pass (TF-045), or
    quarantine act (TF-062) runs inside it; memory
    *change* over a long conversation is those
    slices' proofs, composed by consumers.

The human-readable write-up is
``docs/architecture/LONG_CONVERSATION_PROOF.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from thoughtforge.knowledge.multilingual import fold_text, search_store

# ── The position, as data ────────────────────────────────────────────────────

SAGA_RULES: tuple[dict[str, Any], ...] = (
    {"name": "exactly_one_hundred_turns",
     "rule": "the saga is 100 turns: ten synthetic facts, each "
             "probed once per round over ten rounds, so every fact "
             "is asked first in turns 1-10 and last in turns 91-100"},
    {"name": "memory_is_the_store_not_the_history",
     "rule": "the runner carries no chat history between turns; a "
             "fact survives only if the store holds it and "
             "retrieval finds it (the history channel is TF-028's "
             "proof, and TF-024's budgets drop old history by rule)"},
    {"name": "answers_come_from_retrieval",
     "rule": "the grounded answerer answers from the retrieved "
             "record's own summary verbatim, with that record's "
             "citation marker; an empty retrieval answers the "
             "stated fallback with no citation, never a fabricated "
             "one"},
    {"name": "citation_must_resolve_and_be_true",
     "rule": "a citation is true only when its marker resolves to "
             "a record in the store, that record is the record the "
             "turn probed, and the answer's stated text is that "
             "record's summary (folded equality)"},
    {"name": "contradiction_or_false_citation_zeroes_the_turn",
     "rule": "a turn stating a forbidden phrase for its probed "
             "fact, or carrying a missing / fabricated / untrue "
             "citation, scores 0 regardless of its retention"},
    {"name": "saga_is_read_only",
     "rule": "running the saga never writes to the store: its "
             "bytes are snapshotted before and after and must be "
             "identical; memory change belongs to TF-045/TF-047"},
    {"name": "synthetic_and_deterministic",
     "rule": "every fact, question and answer is TF-017's "
             "synthetic Lantern Hollow material (never canon); no "
             "model, no sampling, no clock — two runs are "
             "byte-identical"},
)

SAGA_TURNS: int = 100
FIRST_WINDOW: int = 10     # turns 1–10
LAST_WINDOW: int = 10      # turns 91–100

PASS_LAST_MIN: float = 0.80
PASS_DRIFT_MAX: float = 0.20

#: The grounded answerer's answer to a retrieval that
#: returned nothing — stated, citation-free, never
#: fabricated.
FALLBACK_ANSWER: str = "I cannot recall that from memory."

# ── The facts ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SagaFact:
    """One synthetic fact the saga plants and probes.

    ``required`` is a tuple of phrase groups (a group
    is satisfied when any alternative appears in the
    answer, folded); ``forbidden`` phrases each
    contradict the fact outright (TF-028's anchors).
    """

    fact_id: str
    record_id: str
    summary: str
    tags: tuple[str, ...]
    question: str
    required: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...] = ()


SAGA_FACTS: tuple[SagaFact, ...] = (
    SagaFact(
        "ferry_minutes", "fct_tf063_01",
        "the ferry crossing at lantern hollow takes eleven "
        "minutes in calm water",
        ("ferry", "crossing"),
        "how long does the ferry crossing take in calm water?",
        required=(("eleven minutes", "11 minutes"),),
        forbidden=("twelve minutes", "twenty minutes",
                   "five minutes"),
    ),
    SagaFact(
        "tide_bell", "fct_tf063_02",
        "the tide bell rings twice at dusk over lantern hollow",
        ("tide", "bell"),
        "how many times does the tide bell ring at dusk?",
        required=(("twice", "rings two", "two rings"), ("dusk",)),
        forbidden=("rings once", "rings three times", "at dawn"),
    ),
    SagaFact(
        "red_lantern", "fct_tf063_03",
        "when the river sarn runs high, wrenna hangs a red "
        "lantern at the ferry post",
        ("river", "warning"),
        "what sign appears when the river sarn runs high?",
        required=(("red lantern",), ("runs high", "river is high")),
        forbidden=("blue lantern", "green lantern", "white lantern"),
    ),
    SagaFact(
        "river_record", "fct_tf063_04",
        "the river level is written in chalk on the mill door",
        ("river", "record"),
        "where is the river level written, and with what?",
        required=(("chalk",), ("mill door",)),
        forbidden=("in a ledger", "in a diary", "on a tablet"),
    ),
    SagaFact(
        "keeper_flask", "fct_tf063_05",
        "wrenna the lantern keeper carries spare oil in a "
        "brass flask",
        ("keeper", "oil"),
        "who carries the spare lantern oil, and in what?",
        required=(("wrenna",), ("brass flask",)),
        forbidden=("clay jug", "iron pot"),
    ),
    SagaFact(
        "last_crossing", "fct_tf063_06",
        "after the second tide bell, no ferry crossing departs "
        "lantern hollow",
        ("ferry", "schedule"),
        "can anyone cross after the second tide bell?",
        required=(("second tide bell",), ("no ferry", "no crossing")),
        forbidden=("runs all night", "ferry runs all night"),
    ),
    SagaFact(
        "ferry_hand", "fct_tf063_07",
        "tomas the ferry hand poles the boat between the "
        "willow narrows and the mill landing",
        ("ferry", "narrows"),
        "who poles the boat between the willow narrows?",
        required=(("tomas",), ("willow narrows",)),
        forbidden=("wrenna poles", "the miller poles"),
    ),
    SagaFact(
        "winter_timetable", "fct_tf063_08",
        "the winter timetable is pinned inside the boathouse "
        "door at lantern hollow",
        ("timetable", "winter"),
        "where is the winter timetable pinned?",
        required=(("winter timetable",), ("boathouse",)),
        forbidden=("ferry office", "mill door"),
    ),
    SagaFact(
        "miller_barley", "fct_tf063_09",
        "the miller grinds barley flour on market mornings "
        "at lantern hollow",
        ("miller", "market"),
        "what does the miller grind on market mornings?",
        required=(("barley flour",), ("market mornings",)),
        forbidden=("wheat flour", "rye flour"),
    ),
    SagaFact(
        "swallows", "fct_tf063_10",
        "swallows nest under the boathouse eaves each spring "
        "at lantern hollow",
        ("birds", "spring"),
        "which birds nest under the boathouse eaves in spring?",
        required=(("swallows",), ("boathouse eaves",)),
        forbidden=("sparrows nest", "martins nest"),
    ),
)

_FACT_BY_ID: dict[str, SagaFact] = {f.fact_id: f for f in SAGA_FACTS}

#: The ten round frames: round r asks every fact's
#: question inside frame r, so the same fact is probed
#: in ten different wordings of *asking* across the
#: saga (the question itself — what retrieval keys on —
#: is the fact's own).
ROUND_FRAMES: tuple[str, ...] = (
    "{q}",
    "Tell me again — {q}",
    "Remind me: {q}",
    "For the traveller's notes, {q}",
    "Checking the long memory: {q}",
    "As we discussed much earlier, {q}",
    "One more check — {q}",
    "For the record: {q}",
    "Late in our talk, {q}",
    "Finally, once more: {q}",
)

# ── The script ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SagaTurn:
    """One scripted turn: its 1-based number, the round
    (1–10), the fact it probes, and the user's words."""

    turn_number: int
    round_number: int
    fact_id: str
    user: str
    question: str


def _build_script() -> tuple[SagaTurn, ...]:
    turns: list[SagaTurn] = []
    for round_index, frame in enumerate(ROUND_FRAMES):
        for fact in SAGA_FACTS:
            turns.append(SagaTurn(
                turn_number=len(turns) + 1,
                round_number=round_index + 1,
                fact_id=fact.fact_id,
                user=frame.format(q=fact.question),
                question=fact.question,
            ))
    return tuple(turns)


SAGA_SCRIPT: tuple[SagaTurn, ...] = _build_script()

# ── The store legs ───────────────────────────────────────────────────────────


def build_saga_store(memory_dir: Any) -> Any:
    """A ``MemoryStore`` at ``memory_dir`` seeded with
    the ten saga facts through the store's own gated
    append path (TF-042's contract + TF-062's screen
    both apply). Stamps ``source='user_statement'``,
    so each fact's citation marker is
    ``user_statement:<record_id>`` by TF-048's rule."""
    from pathlib import Path

    from thoughtforge.knowledge.models import UserFactRecord
    from thoughtforge.knowledge.store import MemoryStore

    store = MemoryStore(memory_dir=Path(memory_dir))
    for fact in SAGA_FACTS:
        store.append_profile_record(UserFactRecord(
            record_id=fact.record_id,
            tags=list(fact.tags),
            summary=fact.summary,
            fact_value={"fact": fact.fact_id},
            source="user_statement",
        ))
    return store


def marker_map(store: Any) -> dict[str, dict[str, str]]:
    """The store's citation map: marker -> {record_id,
    summary} for every profile record, markers derived
    by TF-048's own rule. This is the ground truth a
    citation is judged against: a marker absent from
    this map is fabricated."""
    from thoughtforge.knowledge.models import derive_citation

    markers: dict[str, dict[str, str]] = {}
    for record in store.load_profile_records():
        payload = derive_citation(
            record.record_id, record.record_type,
            {"source": record.source})
        markers[payload.marker] = {
            "record_id": record.record_id,
            "summary": record.summary,
        }
    return markers


def _store_bytes(store: Any) -> dict[str, bytes]:
    from pathlib import Path

    root = Path(store.dir)
    return {
        path.name: path.read_bytes()
        for path in sorted(root.iterdir()) if path.is_file()
    }

# ── Scoring ──────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class SagaTurnScore:
    """One turn's saga score."""

    turn_number: int
    fact_id: str
    user: str
    text: str
    retention: float
    contradictions: tuple[str, ...]      # fact ids contradicted
    citation_status: str                 # true | missing | fabricated | untrue
    citation_marker: str                 # "" when missing
    score: float                         # 0.0 on contradiction / citation failure


@dataclass
class SagaReport:
    """The whole saga's verdict."""

    total_turns: int
    mean_score: float
    first_mean: float                    # turns 1–10
    last_mean: float                     # turns 91–100
    drift: float                         # first_mean − last_mean
    contradiction_turns: int
    citation_failures: int               # missing + fabricated + untrue
    fabricated_citations: int
    untrue_citations: int
    missing_citations: int
    fact_retention: dict[str, float]     # per fact, over its 10 probes
    per_turn: list[SagaTurnScore] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if (
            self.total_turns > 0
            and self.last_mean >= PASS_LAST_MIN
            and self.drift <= PASS_DRIFT_MAX
            and self.contradiction_turns == 0
            and self.citation_failures == 0
        ):
            return "PASS"
        return "REVIEW"


def _score_turn(
    turn: SagaTurn,
    text: str,
    markers: dict[str, dict[str, str]],
) -> SagaTurnScore:
    fact = _FACT_BY_ID[turn.fact_id]
    folded = fold_text(text)
    satisfied = 0
    required_groups = 0
    for group in fact.required:
        required_groups += 1
        if any(fold_text(alt) in folded for alt in group):
            satisfied += 1
    retention = satisfied / required_groups if required_groups else 1.0
    contradictions = (
        (fact.fact_id,) if any(
            fold_text(phrase) in folded for phrase in fact.forbidden)
        else ()
    )

    found = _CITATION_RE.findall(text)
    marker = found[-1].strip() if found else ""
    if not marker:
        status = "missing"
    elif marker not in markers:
        status = "fabricated"
    else:
        resolved = markers[marker]
        stated = _CITATION_RE.sub("", text).strip()
        if resolved["record_id"] != fact.record_id:
            status = "untrue"
        elif fold_text(stated.rstrip(".")) != fold_text(
                resolved["summary"].rstrip(".")):
            status = "untrue"
        else:
            status = "true"

    score = 0.0 if (contradictions or status != "true") else retention
    return SagaTurnScore(
        turn_number=turn.turn_number,
        fact_id=turn.fact_id,
        user=turn.user,
        text=text,
        retention=retention,
        contradictions=contradictions,
        citation_status=status,
        citation_marker=marker,
        score=score,
    )


def score_saga(
    transcript: list[tuple[str, str]],
    markers: dict[str, dict[str, str]],
    script: tuple[SagaTurn, ...] = SAGA_SCRIPT,
) -> SagaReport:
    """Score a saga transcript (``(user, assistant)``
    pairs, in script order) for coherence + citation
    truth against the store's marker map.

    The transcript is paired with the script by
    position; its user turns must be the script's (a
    mismatch is a harness error, not drift, and
    raises). An empty transcript scores 0.0 / REVIEW —
    never a vacuous pass.
    """
    if len(transcript) > len(script):
        raise ValueError("transcript is longer than the saga script")
    per_turn: list[SagaTurnScore] = []
    for (user, text), turn in zip(transcript, script):
        if user != turn.user:
            raise ValueError(
                f"transcript turn {turn.turn_number} does not match "
                f"the script: {user!r} != {turn.user!r}")
        per_turn.append(_score_turn(turn, text, markers))

    if not per_turn:
        return SagaReport(
            total_turns=0, mean_score=0.0, first_mean=0.0,
            last_mean=0.0, drift=0.0, contradiction_turns=0,
            citation_failures=0, fabricated_citations=0,
            untrue_citations=0, missing_citations=0,
            fact_retention={}, per_turn=[])

    scores = [t.score for t in per_turn]
    first = scores[:FIRST_WINDOW]
    last = scores[-LAST_WINDOW:] if len(scores) >= LAST_WINDOW \
        else list(scores)
    first_mean = sum(first) / len(first)
    last_mean = sum(last) / len(last)

    fact_scores: dict[str, list[float]] = {}
    for turn_score in per_turn:
        fact_scores.setdefault(turn_score.fact_id, []).append(
            turn_score.score)

    return SagaReport(
        total_turns=len(per_turn),
        mean_score=sum(scores) / len(scores),
        first_mean=first_mean,
        last_mean=last_mean,
        drift=first_mean - last_mean,
        contradiction_turns=sum(
            1 for t in per_turn if t.contradictions),
        citation_failures=sum(
            1 for t in per_turn if t.citation_status != "true"),
        fabricated_citations=sum(
            1 for t in per_turn if t.citation_status == "fabricated"),
        untrue_citations=sum(
            1 for t in per_turn if t.citation_status == "untrue"),
        missing_citations=sum(
            1 for t in per_turn if t.citation_status == "missing"),
        fact_retention={
            k: sum(v) / len(v) for k, v in sorted(fact_scores.items())
        },
        per_turn=per_turn,
    )

# ── The runner ───────────────────────────────────────────────────────────────


@dataclass
class SagaRun:
    """A finished saga: the transcript, its report,
    retrieval accuracy, and the read-only proof."""

    transcript: list[tuple[str, str]]
    report: SagaReport
    retrieval_hits: int                  # turns whose top record was the probed fact
    screened_out_total: int              # TF-062 exclusions across all retrievals
    store_unchanged: bool


def _answer_from_record(record: Any) -> str:
    from thoughtforge.knowledge.citations import citation_for_record

    summary = str(record.raw.get("summary", "")).strip()
    marker = citation_for_record(record).marker
    text = summary[:1].upper() + summary[1:]
    if not text.endswith("."):
        text += "."
    return f"{text} [{marker}]"


def run_saga(
    store: Any,
    drift_from_turn: int | None = None,
) -> SagaRun:
    """Run the 100-turn saga over ``store``.

    Each turn retrieves through TF-060's personal-
    store search on the fact's question and the
    grounded answerer answers from the top record
    (see the module docstring). With
    ``drift_from_turn`` (1-based), the answerer
    instead answers from the *next* saga fact's
    stored record from that turn on — a cited,
    coherent-looking, wrong answer the scorer must
    catch. The store is only read (byte-pinned).
    """
    before = _store_bytes(store)
    markers = marker_map(store)
    transcript: list[tuple[str, str]] = []
    retrieval_hits = 0
    screened_out_total = 0

    for turn in SAGA_SCRIPT:
        fact = _FACT_BY_ID[turn.fact_id]
        bundle, report = search_store(store, fact.question)
        screened_out_total += len(report.get("screened_out", []))
        top = bundle.activated_records[0] if bundle.activated_records \
            else None
        if top is not None and top.record_id == fact.record_id:
            retrieval_hits += 1

        drifting = (
            drift_from_turn is not None
            and turn.turn_number >= drift_from_turn
        )
        if drifting:
            index = SAGA_FACTS.index(fact)
            wrong = SAGA_FACTS[(index + 1) % len(SAGA_FACTS)]
            answer = _drift_answer(wrong)
        elif top is None:
            answer = FALLBACK_ANSWER
        else:
            answer = _answer_from_record(top)
        transcript.append((turn.user, answer))

    return SagaRun(
        transcript=transcript,
        report=score_saga(transcript, markers),
        retrieval_hits=retrieval_hits,
        screened_out_total=screened_out_total,
        store_unchanged=_store_bytes(store) == before,
    )


def _drift_answer(fact: SagaFact) -> str:
    """The drifting answerer's text for a fact: its
    summary, capitalised, with its own true marker —
    a citation that resolves, attached to the wrong
    turn's answer (untrue for that turn by rule)."""
    text = fact.summary[:1].upper() + fact.summary[1:]
    if not text.endswith("."):
        text += "."
    return f"{text} [user_statement:{fact.record_id}]"

# ── The live checker ─────────────────────────────────────────────────────────


def long_conversation_problems() -> list[str]:
    """Re-derive TF-063's claims from the live tree;
    [] = holds.

    Pins: the rules / counts as data; the script's
    shape (100 turns, 10 rounds × 10 facts, unique
    user turns, every fact probed exactly 10 times,
    first probes in turns 1–10 and last in 91–100);
    the facts landing through the gate (10 records,
    adversarial pass, markers as expected); the full
    saga PASSing perfectly (mean / first / last 1.0,
    drift 0, no contradiction or citation failure,
    retrieval 100/100, screened_out 0, store bytes
    unchanged); determinism (a second run
    byte-identical); the negative controls (a
    fabricated marker is named fabricated, a wrong
    record's marker is untrue, a contradiction zeroes
    its turn, an empty transcript is REVIEW at 0.0);
    and the drift control (drifting from turn 91:
    last window 0.0, drift 1.0, REVIEW)."""
    problems: list[str] = []

    import tempfile
    from pathlib import Path

    if [row["name"] for row in SAGA_RULES] != [
            "exactly_one_hundred_turns",
            "memory_is_the_store_not_the_history",
            "answers_come_from_retrieval",
            "citation_must_resolve_and_be_true",
            "contradiction_or_false_citation_zeroes_the_turn",
            "saga_is_read_only",
            "synthetic_and_deterministic"]:
        problems.append("SAGA_RULES drifted")
    if len(SAGA_FACTS) != 10 or len(ROUND_FRAMES) != 10:
        problems.append("SAGA_FACTS / ROUND_FRAMES count drifted")
    if len(SAGA_SCRIPT) != SAGA_TURNS:
        problems.append(f"saga script holds {len(SAGA_SCRIPT)} turns")
    if len({t.user for t in SAGA_SCRIPT}) != SAGA_TURNS:
        problems.append("saga user turns are not unique")
    probes: dict[str, list[int]] = {}
    for turn in SAGA_SCRIPT:
        probes.setdefault(turn.fact_id, []).append(turn.turn_number)
    for fact in SAGA_FACTS:
        numbers = probes.get(fact.fact_id, [])
        if len(numbers) != 10 or min(numbers) > 10 or max(numbers) < 91:
            problems.append(
                f"fact {fact.fact_id} probed {len(numbers)} times "
                f"({numbers[:1]}..{numbers[-1:]})")

    with tempfile.TemporaryDirectory(prefix="tf063-") as tmp:
        store = build_saga_store(Path(tmp))
        records = store.load_profile_records()
        if len(records) != len(SAGA_FACTS):
            problems.append(f"saga store holds {len(records)} records")
        markers = marker_map(store)
        if set(markers) != {
                f"user_statement:{f.record_id}" for f in SAGA_FACTS}:
            problems.append(f"saga markers drifted: {sorted(markers)}")

        run = run_saga(store)
        report = run.report
        if report.verdict != "PASS" or report.mean_score != 1.0:
            problems.append(
                f"saga did not PASS perfectly: {report.verdict} "
                f"mean {report.mean_score}")
        if report.first_mean != 1.0 or report.last_mean != 1.0 \
                or report.drift != 0.0:
            problems.append(
                f"saga windows drifted: first {report.first_mean} "
                f"last {report.last_mean} drift {report.drift}")
        if report.citation_failures or report.contradiction_turns:
            problems.append(
                f"saga failures: citations {report.citation_failures} "
                f"contradictions {report.contradiction_turns}")
        if run.retrieval_hits != SAGA_TURNS:
            problems.append(
                f"retrieval hit {run.retrieval_hits}/{SAGA_TURNS}")
        if run.screened_out_total != 0:
            problems.append("saga facts were screened out at retrieval")
        if not run.store_unchanged:
            problems.append("the saga wrote to the store")

        second = run_saga(store)
        if second.transcript != run.transcript:
            problems.append("two saga runs differ (not deterministic)")

        # Negative controls on the scorer.
        fake = list(run.transcript)
        user0, text0 = fake[0]
        fake[0] = (user0, text0.replace(
            f"[user_statement:{SAGA_FACTS[0].record_id}]",
            "[user_statement:fct_tf063_99]"))
        fake_report = score_saga(fake, markers)
        if fake_report.per_turn[0].citation_status != "fabricated" \
                or fake_report.per_turn[0].score != 0.0 \
                or fake_report.fabricated_citations != 1:
            problems.append("a fabricated citation is not caught")

        wrong = list(run.transcript)
        user1, _ = wrong[1]
        wrong_fact = SAGA_FACTS[1]
        wrong[1] = (user1, _drift_answer(
            SAGA_FACTS[(1 + 1) % len(SAGA_FACTS)]))
        wrong_report = score_saga(wrong, markers)
        if wrong_report.per_turn[1].citation_status != "untrue" \
                or wrong_report.per_turn[1].score != 0.0:
            problems.append("a wrong-record citation is not untrue")
        _ = wrong_fact

        contra = list(run.transcript)
        user2, text2 = contra[0]
        contra[0] = (user2, text2 + " It takes twelve minutes.")
        contra_report = score_saga(contra, markers)
        if contra_report.per_turn[0].score != 0.0 \
                or not contra_report.per_turn[0].contradictions:
            problems.append("a contradiction does not zero its turn")

        empty = score_saga([], markers)
        if empty.verdict != "REVIEW" or empty.mean_score != 0.0:
            problems.append("an empty transcript is not REVIEW at 0.0")

        drifted = run_saga(store, drift_from_turn=91)
        if drifted.report.verdict != "REVIEW" \
                or drifted.report.last_mean != 0.0 \
                or drifted.report.drift != 1.0:
            problems.append(
                f"the drift control is not caught: "
                f"{drifted.report.verdict} last "
                f"{drifted.report.last_mean} drift "
                f"{drifted.report.drift}")
    return problems


__all__ = [
    "FALLBACK_ANSWER",
    "FIRST_WINDOW",
    "LAST_WINDOW",
    "ROUND_FRAMES",
    "SAGA_FACTS",
    "SAGA_RULES",
    "SAGA_SCRIPT",
    "SAGA_TURNS",
    "SagaFact",
    "SagaReport",
    "SagaRun",
    "SagaTurn",
    "SagaTurnScore",
    "build_saga_store",
    "long_conversation_problems",
    "marker_map",
    "run_saga",
    "score_saga",
]
