# TF-093 — Enforcement Mapped + Completed: One Policy Surface

*Every enforcement check in the refinement estate
now runs through one surface —
`enforce_policy` in
`src/thoughtforge/refinement/policy_surface.py`
(`POLICY_RULES` / `POLICY_CHECKS` /
`POLICY_FIXTURES` as data, the checkers
`policy_surface_problems` /
`policy_surface_live_problems`, format
`thoughtforge-policy-surface/1`) — which names
every check in execution order, the caller signal
each one needs, and how their verdicts combine.
TF-049's cite-or-explain is unified into it, and
the two citation rules that stood unreconciled
since TF-049 are reconciled here, on the record.*

## The plain truth first

Until this slice, enforcement was five separate
surfaces. `EnforcementGate` judged length,
genericness and a legacy any-QID citation floor,
with TF-049's per-claim cite-or-explain ANDed
into its citation leg when a caller passed
records. TF-090's format gate, TF-091's loop
gate and TF-092's drift gate were each
caller-invoked beside it. No single place named
every check, their order, the signal each one
needs, or how their verdicts combine — a caller
had to know all five surfaces and compose them
by hand, and a check nobody invoked was a check
that silently never ran.

This slice does not build a sixth gate. It
builds the **map and the composition**: the
existing gates are called, never re-implemented,
and the surface records one outcome per check,
in order, whatever the verdict.

## The map

| # | Check | Owner | Signal the caller supplies |
|---|---|---|---|
| 1 | format | TF-090 | the ask was structured |
| 2 | truncation | TF-090 | truncation is suspected |
| 3 | loop | TF-091 | none — always judged on prose |
| 4 | drift | TF-092 | the expected language |
| 5 | citation | TF-049 + the legacy floor | records and/or retrieved QIDs |
| 6 | length | EnforcementGate | none |
| 7 | genericness | EnforcementGate | none |

**Signals are caller-supplied, never guessed**
(the TF-090 shape, generalised). A check whose
signal was not supplied is recorded `skipped`
with its reason — never reported as passed, and
the surface never invents a signal from the
draft. Salvage has no structured-ask signal;
that is why TF-090's gate was caller-invoked,
and the rule carries to the whole surface.

**Order is repair before judgement.** Format
and truncation act first (they can change the
text's shape), loop and drift rescues second
(deletion only, TF-088's line throughout), and
citation / length / genericness judge **last —
on the exact text that would leave**. A citation
that survives only inside a cut segment does
not count, and a rescued text is judged as
rescued.

## The citation reconciliation

TF-049 filed the conflict twice and left the
call to this slice, in its own words: the
per-claim rule passes an answer that claims no
stored fact; the legacy any-QID floor fails
that same answer when QIDs were retrieved but
none is cited. The floor is claim-blind — the
very defect TF-049 was built to fix.

**The ruling, recorded in `POLICY_RULES` as
`citation_reconciled_per_claim_when_records`:**

- **When the caller supplies `records`, the
  per-claim cite-or-explain rule is the citation
  law of this surface.** An answer claiming no
  stored fact passes its citation check even
  when retrieved QIDs exist uncited. The floor
  is recorded `subsumed_by_per_claim` in the
  citation outcome — the surface invokes
  `EnforcementGate` with its floor leg inert
  and its records leg live — not silently
  dropped.
- **When `records` are not supplied, the legacy
  floor is the only citation rule available and
  applies exactly as before.**

`EnforcementGate` itself is **unchanged**. Its
no-records behaviour is byte-identical, and its
with-records AND remains its own standing
behaviour for its own callers (the core's
compose path among them). The reconciliation is
this surface's stated policy — TF-049 asked for
the call to be "recorded in the policy, not
taken silently", and it is. The slice's live
checker proves the contrast against the real
gate: the gate's with-records floor flags the
no-claim answer; the unified surface, under
per-claim law, passes it; the gate's no-records
behaviour is verified unmoved in the same run.

## Regeneration: ruled not wired, here

CITATION_POLICY names `regenerate` as an option
and TF-049 assigned its wiring to this
unification. The ruling: **the unified surface
never calls a model** — no gate in this estate
does, and a checker that secretly spent a
second model call would break TF-098's cost
discipline before it arrives. The shipped
citation action stays `flag`;
`regeneration_required` stays False. What the
surface does hand the caller is the flagged
verdict **with the full per-claim record**, so a
caller who regenerates does so *with* the
knowledge context that TF-023's repair pass —
the estate's only regeneration path — lacks: a
regenerated answer can cite what it was shown.
That hand-off, not a hidden model call inside a
checker, is the honest wiring. TF-094 owns
strictness configuration; TF-095 owns telling
the user that and why an answer was flagged.

## Two more rulings the map had to make

- **Salvage scoring is not enforcement.**
  TF-049 §5 left this slice the question whether
  salvage's internal QID-only draft scoring moves
  to markers. It does not: that score *ranks*
  drafts and enforces nothing, and moving it
  would rewrite TF-087's catalogue and TF-089's
  filed metrics mid-Part. `FragmentSalvage` is
  untouched by this slice, as by TF-090/091/092.
- **Structured answers are judged as
  structure.** On a structured ask the prose
  checks (loop, drift, citation, length,
  genericness) are recorded
  `skipped_not_applicable_structured`: a JSON
  object's word count is not a prose length and
  its values are not prose claims. The format
  gate alone judges it, schema leg included.

## How verdicts combine

Precedence is filed: **refused > flagged >
rescued > clean**.

- **Refused** — a check refused (an unrepairable
  structured output, a truncation with no
  complete sentence, an expected language the
  TF-060 hint cannot judge, empty output). The
  surface returns no usable text, the problems
  are named, and every later check is recorded
  `skipped_after_refusal` — never passed.
- **Flagged** — drift flagged verbatim, or a
  citation / length / genericness failure. The
  text returns (annotated by the gate where the
  gate annotates) with every flag named:
  `drift:whole_answer_off_language`,
  `citation:uncited_claim`,
  `citation:foreign_marker`,
  `citation:no_retrieved_qids_cited`,
  `length:too_short`,
  `genericness:hard_generic_phrase`.
- **Rescued** — a repair or a cut was applied
  and every judgement passed. A rescued text
  must judge `clean` on a second pass under the
  same signals; the live checker proves it for
  every filed rescue.
- **Clean** — every supplied-signal check
  passed and nothing was changed.

## The fixtures

16 filed cases, all synthetic, every one run
live through the real gates by the slice's
checker: a clean prose answer with no signals
(unsupplied-signal checks recorded skipped, not
passed); a loop rescue and a drift rescue each
judged clean afterwards; whole-answer drift
flagged verbatim; an uncited stored-fact claim
flagged with the gate's note; the reconciliation
pair (a no-claim answer clean under per-claim
law with the floor subsumed, and the same answer
flagged under the legacy floor without records);
a cited claim clean; a structured output
repaired and a structured refusal (later checks
skipped, never passed); a prose truncation
repaired and one refused for having no complete
sentence; a hard generic phrase flagged; a
too-short answer flagged with the gate's note;
empty output refused; and an unsupported
expected language refused by name.

## Filed limits, not hidden

- **The surface composes gates; it adds no new
  detection.** Every limit of the underlying
  gates stands exactly as filed: TF-049's
  term-overlap matching is a heuristic, not
  entailment; TF-091 detects exact loops only;
  TF-092 judges by TF-060's hints over
  en / es / de; TF-090 repairs only by
  removal / re-delimiting.
- **The core does not yet compose through this
  surface.** `_compose_final` still calls
  `EnforcementGate` directly (its TF-049 wiring
  stands, pinned by TF-049's checker, which this
  slice does not move). The surface is the
  caller-facing policy for refinement consumers;
  routing the core's compose path through it is
  a core slice's call, not silently taken here —
  the API freeze (TF-034) is respected.
- **A flagged answer still leaves with the
  gate's annotation**, as it did before this
  slice; turning a flag into a user-facing
  refusal is TF-095's work, and strictness
  levels are TF-094's.

## Docket

- No TF-016 entry is owned by TF-093 and none
  changes status: the unification completes the
  work TF-049, TF-090, TF-091 and TF-092 filed
  forward to it. Totals unchanged: 88 entries —
  45 OPEN / 1 OPEN-ENVIRONMENT / 39 FIXED /
  3 CLOSED-BY-VERDICT. D-SALV-1 stays OPEN for
  its one remaining share (rambling compression
  in general), exactly as TF-092 left it.

## Proof grade

FIXTURE — every filed case recomputed live
through the real gates on this VM, the
reconciliation proven against the real
`EnforcementGate`, rescued texts proven clean
on a second pass. A live model's behaviour
under the unified surface is **NOT PROVEN**
(no model in tree); TF-100's chaos corpus owns
real-shaped failures and TF-101 runs the full
policy stack — this surface — over them.
