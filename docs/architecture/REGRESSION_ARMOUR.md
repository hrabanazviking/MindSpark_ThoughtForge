# TF-103 — Regression Armour: Every Named Bad Output, Fixtured

*Every historical bad output named in the repo's
docs now carries exactly one registry case, and
its fix is trusted only while its fixture is
re-proven. The estate lives in
`src/thoughtforge/refinement/regression_armour.py`
(`ARMOUR_RULES` / `ARMOUR_CASES` /
`ARMOUR_FIXTURES` / `GAP_FAMILIES` as data, the
checkers `regression_armour_problems` /
`regression_armour_live_problems`, format
`thoughtforge-regression-armour/1`). No gate,
surface, or fixing module is changed, and no
existing fixture's expectation is edited. No
docket entry was assigned to TF-103 and none is
added: TF-016's totals stand unchanged at 90.*

## The plain truth first

The campaign's docs name the bad outputs one
slice at a time, in their measured-start
passages: the refine pass that invented a price
of 50 silver coins and was adopted on score
(TF-088), the 20× loop returned byte-identical
(TF-087's map, fixed by TF-091), the router's
dead upper-case patterns misrouting "the api"
(TF-020), the stored fact silently overwritten
by its contradiction (TF-050), the `/v1/v1`
double path and the parser legs that dressed a
`content_filter` finish as a completion
(TF-076), the bare `Error generating response.`
(TF-031), and more. Each fixing slice filed its
own fixtures in its own module — but nothing in
the tree answered, in one place: *which
historical bad outputs exist, where is each one
fixtured, and is its fix re-proven today?* A
fixture set in an unvisited module is armour
nobody checks for rust. This slice is the
registry and the re-proof, and nothing else.

## The five armour kinds

| Kind | Meaning | Cases |
|---|---|---|
| `rerun` | the fixing slice's own fixture set exists; this slice re-runs that module's own live checker today and requires it green | 12 |
| `direct` | this slice files a **new** fixture (`ARMOUR_FIXTURES`) and runs it against the real fix surface | 5 |
| `reference` | the fixture lives in the fixing slice's test file; the data checker verifies the named pin (needle) is present in that file on disk | 3 |
| `gap_open` | a fixture exists (TF-100's corpus) but the fix has **not** landed; the gap is registered with its owner and its exact shape is pinned | 3 |
| `not_proven` | no live output has ever existed to fixture; registered with the reason, never counted as covered | 2 |

**Covered: 20 of 25 cases.** The other five are
the honest remainder below — three gaps and two
absences, named, never dressed as coverage.

## The direct fixtures (new in this slice)

- **Router dead patterns (TF-020).** The four
  measured misroutes, re-run through the real
  `InputRouter`: "tell me about the api" →
  intent `technical_spec_request`; "how do i
  plan the phases" → `planning_request`;
  "i don't know what to do anymore" →
  `emotional_support`; "i've been thinking about
  the old hall" → tone `curious`.
- **Contradiction overwrite (TF-050).** A new
  fact (ferry at midnight) against the stored
  fact (ferry at dawn), same slot: the real
  `assess_contradiction` must surface exactly
  one conflict under the `fact_value_slot` rule
  with a stated resolution — a silent overwrite
  surfaces none and fails.
- **Double-/v1 (TF-076, D-OAI-2).**
  `normalize_base_url("http://127.0.0.1:1234/v1")`
  must strip the trailing `/v1`, so no
  `/v1/v1/chat/completions` can be built.
- **Parser outcome legs (TF-076, D-OAI-1).**
  Through the real `chat_completion_outcome`:
  a `content_filter` finish is an error naming
  the finish, never a completion; `content:
  null` is an error naming the NoneType, never a
  `None` text; empty `choices` is an error, not
  an `IndexError`; and a plain completion still
  passes with its text intact (the armour
  guards both directions).
- **The 20× loop at text level (TF-091).**
  `"The bell rings."` × 20 through the real
  `rescue_loops`: verdict `rescued`, exactly one
  occurrence surviving.

## The gaps — armour, not coverage

TF-101's scored run filed 9 mismatches in 3
named gaps. They are registered `gap_open`,
each with the owner TF-101 filed, and the live
checker recomputes TF-101's real run and pins
the shape exactly: **9 mismatches — 4
hallucination (owner TF-088), 3 persona-voice
(owner TF-096), 2 genericness (owner TF-093) —
each naming its filed `NAMED_GAPS` id.** If an
owner's fix lands, this checker goes red until
the registry is updated in the open — that is
the armour working, not failing. TF-104's Part
V gate inherits these gaps exactly as pinned
here.

## The absences — not proven, not fixtured

- **D-AESIR-1:** no live Aesir server has ever
  faced the TF-074 backend, so no live output —
  good or bad — exists to fixture.
- **D-OAI-4:** no real third-party
  OpenAI-shaped server has faced the TF-076
  backend; its mocked legs are covered by the
  `rerun` case, a live one cannot be invented.

## Honesty edges

- **Rerun never re-implements.** A `rerun` case
  calls the fixing module's own live checker
  unchanged. The armour cannot drift from the
  fix it claims to prove, because it contains
  no copy of any gate, parser, or guard.
- **References are verified from disk.** The
  data checker reads each `reference` case's
  test file and requires its needle present;
  the registry cannot outlive the fixture it
  names.
- **Docs are verified from disk.** Every case's
  `source_doc` must exist; a case with no doc
  would be an invention, and a named bad output
  with no case is a registry gap — the census
  behind this slice is the docs themselves.

## Filed limits, not hidden

- The registry is a census of the docs as they
  stand at this slice. A future slice that
  names a new historical bad output must add
  its case in its own banking commit; the data
  checker cannot see a doc nobody wrote.
- `rerun` trust is exactly as strong as the
  fixing module's live checker; this slice
  judges those checkers green or red, it does
  not re-audit their insides (their own slices
  and TF-192's roadmap audit own that).

TF-016 totals are unchanged: 90 entries —
43 OPEN / 1 OPEN-ENVIRONMENT / 43 FIXED /
3 CLOSED-BY-VERDICT (no docket entry is owned
by TF-103; the registry references D-SALV-2,
D-SALV-3, D-TRACE-1, D-OAI-1, D-OAI-2, D-HON-1,
D-DEP-1, D-AESIR-1 and D-OAI-4 as the docket
ids of cases it covers or registers, and
changes none of their states).

## Proof grade

FIXTURE — the registry judged from disk
(source docs, test needles), the rerun legs
recomputed through the real live checkers, the
direct fixtures run against the real router /
contradiction / parser / loop surfaces, and
the gap shape recomputed by TF-101's real
runner, all on this VM. A live model's
regression behaviour is **NOT PROVEN** (no
model in tree); TF-104's gate publishes the
Part V score this armour guards.
