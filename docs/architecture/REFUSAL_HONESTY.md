# TF-095 — Refusal Honesty: the User Learns That, and Why

*When enforcement blocks an answer, the user now
learns **that** it was blocked and **why**, with
the salvage path offered. The estate lives in
`src/thoughtforge/refinement/refusal_honesty.py`
(`REFUSAL_RULES` / `REASONS` / `REFUSAL_FIXTURES`
as data, `refusal_notice` / `refusal_message`,
the checkers `refusal_honesty_problems` /
`refusal_honesty_live_problems`, format
`thoughtforge-refusal-honesty/1`) — a telling
layer over TF-093's and TF-094's real results.
No surface, gate, or salvage code is changed.*

## The plain truth first

Until this slice, a refused answer was silent.
TF-093's surface and TF-094's configured layer
return a block as data — `text=None` and a
`problems` tuple of machine codes
(`empty_output`, `no_json_span`,
`missing_required_key: ferry`,
`citation:uncited_claim`, …) — and nothing in
the tree turned that record into something a
user could read. The codes were the only "why"
that existed, and no message anywhere said an
answer had been blocked at all. TF-090 through
TF-094 each filed the telling to this slice;
TF-031's stated *generation* failures are the
precedent in shape, but a block is a different
event: generation succeeded, and enforcement
refused to pass the result.

## The telling

Every notice is built by `refusal_notice(result)`
from the result record alone, in three parts:

1. **That.** The message opens with the marker
   `Blocked by the forge (TF-095):` and the plain
   statement that the answer was blocked by
   enforcement and has not been shown. A TF-094
   strict block is told as what it is: the answer
   was *flagged*, and under the strict level a
   flagged answer does not leave.
2. **Why.** Every problem the surface recorded
   is named **verbatim** — detail included, as in
   `missing_required_key: ferry` — beside the
   plain-language reason catalogued for it in
   `REASONS`. The catalogue covers every refusal
   code the shipped estate's filed fixtures
   (TF-090–TF-094) can produce; the data checker
   recomputes that coverage from those fixture
   sets on every run, so a future gate's new code
   fails the checker until its reason is filed.
3. **What next.** Each reason carries its honest
   next step, and the message offers those paths,
   deduplicated in problem order: regenerate with
   the retrieved knowledge in view so a claim can
   be cited, ask again in the expected language,
   ask again for the expected structured shape,
   generate again asking for a fuller or more
   specific answer — each phrased as an offer,
   because **the path is offered, never run by
   this slice**: the telling never regenerates,
   never calls salvage, and never changes a
   strictness level.

## The honesty edges

- **An uncatalogued problem is named, never
  dressed.** A code with no filed reason (a
  future gate's invention) appears verbatim,
  marked in the notice as uncatalogued, with
  the generic offered path. It is never dropped
  from the telling and never given an invented
  reason.
- **No problem named is stated, not filled.**
  A refusal whose record names no problem is
  told exactly that way — the gap is the
  message's stated reason.
- **The blocked text cannot leak.** The telling
  takes the result, never the draft: there is
  no parameter through which blocked text could
  enter the message. TF-094 already returns
  `text=None` on a block; this layer keeps the
  guarantee structural instead of trusting it,
  and the live checker asserts no filed draft
  appears in its own message.
- **Only a block gets a refusal message.**
  Building a notice from a clean, rescued, or
  flagged result raises `ValueError` (TF-031's
  precedent). A flagged answer under standard
  *leaves* — it is not a block, and it is not
  dressed as one.

## Filed limits, not hidden

- **The core does not yet route through the
  policy surfaces** (TF-093/TF-094 filed the
  same limit: `_compose_final` still calls
  `EnforcementGate` directly), so no shipped
  turn yet emits this message end-to-end. This
  slice builds the telling every blocked result
  can be rendered with; wiring it into a
  consumer is that consumer's slice — TF-102
  owns the user-visible quality marks at the
  product surface.
- **Reasons are catalogued per code, not per
  situation.** Two blocks with the same code
  get the same reason; the record's check
  outcomes (TF-093) carry the situation's
  detail, and TF-097 owns carrying the record
  into the trace.
- **The catalogue's coverage is fixture-proven,
  not source-proven:** it is recomputed from
  the filed fixture sets of TF-090–TF-094. A
  code a gate can produce but no fixture files
  would ride the uncatalogued leg — named
  verbatim, honestly marked — until filed.

TF-016 totals are unchanged: 88 entries —
45 OPEN / 1 OPEN-ENVIRONMENT / 39 FIXED /
3 CLOSED-BY-VERDICT (no docket entry is owned
by TF-095; the slice closes a gap the roadmap
itself assigned, and adds no entry).

## Proof grade

FIXTURE — every filed case (eight live blocks
through the real TF-093 / TF-094 surfaces: an
empty draft, an unrepairable structured ask, a
missing required field, a truncation with no
complete sentence, an unsupported expected
language, and three strict blocks including a
two-flag block; plus two synthetic records
pinning the uncatalogued and unnamed legs) is
recomputed live on this VM, and its notice is
judged against the record — marker first, every
problem verbatim with its reason and offered
path, no draft text in the message. A live
model's blocked answers are **NOT PROVEN**
(no model in tree); TF-101 runs the full
policy stack over TF-100's chaos corpus.
