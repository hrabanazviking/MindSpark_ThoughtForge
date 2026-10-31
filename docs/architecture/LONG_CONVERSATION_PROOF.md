# TF-063 — The Long-Conversation Proof: 100 Turns, Memory Coherent, Citations True

*Part III, slice TF-063. Code:
`src/thoughtforge/knowledge/long_conversation.py`
(the rules, the facts, the script, the grounded
answerer, the scorer, the runner, the checker).
Tests: `tests/test_tf063_long_conversation.py`.
The harness shape is Part VII's (TF-028's drift
scorer, extended), consumed here for Part III's
gate question.*

## 1. The question and the measured starting state

The roadmap asks for the long-conversation proof:
a **100-turn synthetic saga** in which **memory
stays coherent** and **citations stay true**.
Measured this tick by direct probe (keeper 10:22,
this VM):

- **The longest conversation proof in the tree was
  20 turns.** TF-028's coherence script runs 20
  turns and proves the *history* channel — facts
  established in chat and carried in the threaded
  `ChatHistory`. Nothing ran 100 turns, and TF-024's
  history budgets drop old history by rule, so a
  100-turn conversation cannot rest on history
  anyway.
- **No proof exercised the personal store over a
  long conversation.** TF-028's fixture engine never
  touches `MemoryStore`; the store slices
  (TF-041–TF-062) each prove one operation in
  isolation. Whether retrieval still returns the
  right record at turn 100 as reliably as at turn 1
  was unproven.
- **Citation truth was never scored mechanically
  over a conversation.** TF-048 proved every
  retrieved item *carries* a payload and TF-049
  judges a single answer. No harness checked, turn
  after turn, that an answer's citation *resolves*
  to a real record and that the record it resolves
  to *actually says what the answer claims* — the
  two ways a citation can lie (a fabricated marker;
  a real marker on the wrong record).

## 2. The position (SAGA_RULES, as data)

| Rule | Meaning |
|---|---|
| `exactly_one_hundred_turns` | 100 turns: ten facts × ten rounds; every fact probed first in turns 1–10 and last in turns 91–100 |
| `memory_is_the_store_not_the_history` | The runner carries **no chat history** between turns; a fact survives only if the store holds it and retrieval finds it |
| `answers_come_from_retrieval` | The grounded answerer answers from the retrieved record's own summary verbatim, with that record's citation marker; an empty retrieval answers the stated fallback with **no** citation — never a fabricated one |
| `citation_must_resolve_and_be_true` | True = the marker resolves in the store, to the record the turn probed, and the answer's stated text **is** that record's summary (folded equality) |
| `contradiction_or_false_citation_zeroes_the_turn` | A forbidden phrase, or a missing / fabricated / untrue citation, scores the turn 0 regardless of retention |
| `saga_is_read_only` | The store's bytes are snapshotted before and after and must be identical; memory *change* belongs to TF-045/TF-047 |
| `synthetic_and_deterministic` | TF-017's Lantern Hollow only, never canon; no model, no sampling, no clock — two runs are byte-identical |

## 3. The saga

**Facts.** Ten synthetic facts (ferry minutes, tide
bell, red lantern, chalk river record, keeper's
flask, last crossing, ferry hand, winter timetable,
miller's barley, boathouse swallows), each with a
probe question, required phrase groups, and
forbidden contradiction phrases in TF-028's anchor
shape. They land through the store's own gated
append path (`source='user_statement'`, so each
citation marker is `user_statement:<record_id>` by
TF-048's rule) and pass TF-062's adversarial screen.

**Script.** `SAGA_SCRIPT` is generated as data:
ten rounds × the ten facts, each round re-asking
every fact's question inside a new frame ("Tell me
again — …", "As we discussed much earlier, …",
"Finally, once more: …"). All 100 user turns are
unique, and every fact is probed exactly ten times,
spread across the whole saga (pinned).

**Retrieval.** Each turn retrieves through TF-060's
personal-store search on the fact's question
(TF-062-screened at retrieval); the runner records
retrieval accuracy — the top record is the probed
fact — beside the answer score.

**Scoring.** `score_saga` is pure over a transcript
plus the store's marker map (marker → record id +
summary, derived by TF-048's own rule). Per turn:
retention over the probed fact's required groups;
zero on a contradiction; zero on a citation failure,
classified **missing** (no bracketed marker),
**fabricated** (resolves to no record), or
**untrue** (resolves to a different record, or the
stated text is not the cited record's summary). The
report carries the first window (turns 1–10), the
last window (turns 91–100), drift = first − last,
per-fact retention, and TF-028's verdict shape:
`PASS` only when the last window holds, drift is
small, and there are no contradiction turns and no
citation failures. An empty transcript is `REVIEW`
at 0.0 — never a vacuous pass.

## 4. The measured result

On this VM, at banking: the full saga **PASSes
perfectly** — mean / first / last all **1.0**,
drift **0.0**, retrieval **100/100**, zero
contradiction turns, zero citation failures, zero
screened-out records, the store's bytes identical
before and after, and a second run byte-identical
to the first. The negative controls are pinned in
the checker: a fabricated marker is named
fabricated and zeroes its turn; a wrong record's
marker is untrue; a contradiction zeroes its turn;
and the **drift control** (the answerer answering
from the *next* fact's record from turn 91) drops
the last window to 0.0 with drift 1.0 and a `REVIEW`
verdict — the harness catches a cited,
coherent-looking, wrong answer. An empty store
answers the fallback on all 100 turns with no
citation and scores `REVIEW` at 0.0.

## 5. Boundaries

- **Fixture grade.** The answerer is a deterministic
  grounded stand-in, not a live model. This slice
  proves the store + retrieval + citation leg over
  100 turns, and that the harness catches drift,
  contradiction, and false citations when injected.
  A live small model's 100-turn behaviour is Part
  VII's road (TF-125) and is **NOT PROVEN** here.
- **One retrieval leg.** The saga uses TF-060's
  personal-store search. The knowledge-DB forge leg
  over 100 turns is not claimed — no real lived-in
  store or DB exists on this machine (TF-041).
- **Citation truth is textual, by rule.** "True"
  means folded equality with the cited record's
  stored summary — a stated mechanical rule in
  TF-049's spirit, not a semantic judgement about
  paraphrase (a paraphrasing live model is judged
  by TF-049's claim check, not this scorer).
- **No memory change inside the saga** — no
  writeback, lifecycle pass, or quarantine act runs;
  those are TF-047/TF-045/TF-062's proofs, composed
  by consumers.
- **Proof grade: FIXTURE** — synthetic Lantern
  Hollow facts and questions (TF-017's world, never
  canon) on this VM.
