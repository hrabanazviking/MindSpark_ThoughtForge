# ThoughtForge Architecture — Memory Fidelity Suite (TF-123)

Memory fidelity suite: cite-or-explain scored
mechanically (citations present, citations *true*).
Part VII's third slice, built on TF-121's harness and
TF-122's suite pattern. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`benchmarks/memory_suite.py` (`RECORDS`,
`MEMORY_FACTS`, `PRESENT_CASES`, `TRUE_CASES`,
`EXPLAIN_CASES`, `SUITE_AXES`, `SUITE_RULES`, checked
by `memory_suite_problems()`), registered in
`benchmarks/harness.py` with a pinned baseline. The
tests are `tests/test_tf123_memory_suite.py`. This
slice **adds and fixes D-BENCH-9** (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **No memory fidelity scorer existed anywhere.**
  The harness stood at 10 benchmarks / 3 fixtures /
  3 baselines / 21 metric definitions, all checkers
  clean (TF-048's `citation_problems`, TF-049's
  `citation_enforcement_problems`, TF-093's policy
  surface, the harness itself); its
  `retrieval_precision` entry said "TF-123 (memory
  fidelity suite consumes it)" — filed, unbuilt.
- **Citation truth was scored by nothing — probed.**
  Against TF-043's real timetable chunk (stored
  fact: the lantern hollow ferry *departs dawn*),
  TF-049's `check_citations` passed the false claim
  "…departs at midnight [ref:4301]" identically to
  the true dawn claim. TF-049 judges presence: a
  citation can be present, well-formed and resolved
  while the claim it backs is false, and no checker
  in the estate could tell the two apart. TF-049's
  own doc (§5) filed the boundary honestly —
  "TF-123 will score this properly" — but the gap
  was never numbered in the TF-016 docket.
- **The explain side was scored by nothing —
  probed.** An answer matching no stored fact, with
  no citation and no statement that the reasoning
  is the model's own, passed `check_citations`
  (TF-049's policy: on no matching claims, pass) and
  passes TF-093's unified surface under the same
  rule. Cite-or-explain's second leg had no scorer,
  and no estate module defines an own-reasoning
  marker (the scaffold, TF-049 and TF-093 were all
  grepped: none exists).
- What already worked, and is reused below:
  fabricated citations are caught — `[ref:9999]`
  and `[Q999]` both flag foreign under TF-049, and
  a claim citing the wrong record is flagged
  uncited for the record it actually matches.
  TF-043's pinned baseline re-measured clean live
  (precision 0.8889 / recall 0.9630 at the defaults,
  `baseline_problems` empty).

## 2. The suite

`benchmarks/memory_suite.py` scores three axes over
scripted synthetic cases, all against one filed
stored-fact world — TF-043's corpus, imported and
reused, never copied: `RECORDS` builds nine
`ActivatedRecord`s programmatically from
`retrieval_precision`'s filed rows (four entities,
three reference chunks, the GeoNames place, the
ConceptNet ferry edge), each carrying its TF-048
citation payload by construction. Six of them carry
filed facts (`MEMORY_FACTS`: the dawn departure, the
ferry's passengers-and-cargo, the mill's grain/flour
prices, the observatory's telescope-and-stars, the
stored place, the ferry crossing the river), each
with the phrases a supporting claim must carry and
the phrases that contradict the stored fact.

| Axis | Corpus | Scorer | Filed result |
|---|---|---|---|
| citations present | 12 cases: 10 cited + 1 missing citation + 1 citation in the wrong sentence | TF-049's real `check_citations` | 10/12 → **0.8333** |
| citations true | 12 cases: 9 true + 1 cited-but-false + 1 fabricated + 1 wrong-record | `citation_truth` (below) | 9/12 → **0.75** |
| explain | 12 cases: 10 owned + 1 unowned assertion + 1 disguised memory claim | filed ownership rule + TF-049's no-match leg | 10/12 → **0.8333** |

**Overall 0.8056 over 36 scored cases — PASS**
(every axis at or above the filed 0.75 threshold;
the suite passes only when every axis passes).
The truth axis is the hardest of the three, exactly
at the threshold — the measured shape, not a tuned
one: of its three filed failures, the cited-but-
false case is the D-BENCH-9 probe itself.

**Truth, mechanically** (`citation_truth`): an
answer's citations are true exactly when it carries
at least one citation, no foreign marker appears
(TF-049's leg), every cited marker *resolves* to a
retrieved record, that record *matches the citing
claim* under TF-049's real `match_claim_to_record`,
and the claim carries the cited fact's filed
required phrases with none of its breaking phrases.
Presence alone scores the present axis only — the
midnight claim holds on present and fails on true,
pinned both ways by the checker and the tests.

**Explain, mechanically**: a non-memory case holds
only when TF-049's checker matches no stored
record, the answer carries no citation marker, and
one of the filed `OWNERSHIP_PHRASES` is present
("in my own reasoning", "my own inference",
"i reason that", "i infer that", "speaking for
myself", "my own reasoning"). The vocabulary is
filed as suite data in the open — no estate module
defines one (§1), and the suite claims no more
than the vocabulary it files.

An empty axis scores 0.0; an empty suite scores
0.0 / REVIEW. No vacuous pass anywhere (the TF-122
precedent).

## 3. Harness registration

In `benchmarks/harness.py`, in the open:

- a new `memory_suite` benchmark entry (kind
  `suite`) bound live to
  `benchmarks.memory_suite.run_memory_suite`, with
  a new `memory_suite_corpus` fixture — the estate
  now files **11 benchmarks / 4 fixtures / 4
  baselines / 26 metric definitions**;
- the `retrieval_precision` entry's suite-slice
  note now files TF-123 DONE, with its notes
  stating the mechanical consumption (records built
  from the corpus rows; the checker re-running
  `evaluate_retrieval` against TF-043's pinned
  baseline live);
- `METRIC_DEFINITIONS` extended with the suite's
  axis metrics (`present_score`, `true_score`,
  `explain_score`, `overall_score`, `total_cases`
  for the `memory_suite` benchmark), each stating
  what it actually computes;
- a new pinned baseline, measured live on this VM
  twice, identical, latency pinned nowhere:

**`memory_suite_synthetic_v1`** — present 0.8333,
true 0.75, explain 0.8333, overall 0.8056, 36
cases; target overall ≥ 0.75 **met**. (TF-121's
test pins for the index / fixture counts were
updated in the open: 10 → 11 benchmarks, 3 → 4
fixtures. **Every earlier pinned baseline is
preserved exactly** — persona 0.71666…, persona
suite 0.8454, the empty-store profile baseline —
never re-pinned; `baseline_problems()` is clean
over all four.)

## 4. The TF-043 leg, live

The suite's checker does not merely bind TF-043's
rows as data: `memory_suite_problems()` builds
TF-043's fixture DB fresh in a temp directory,
re-runs `evaluate_retrieval` at the defaults and
requires its pinned baseline unregressed, and runs
a real `KnowledgeForge.retrieve("Lantern Hollow
ferry", path="sql", min_score=0.0)` requiring the
filed records back with their TF-048 markers
(`QF4301` → `QF4301`, `4301` → `ref:4301`). That is
the consumption the harness entry promised since
TF-121, proven in the checker rather than asserted.

## 5. Docket dispositions (TF-016)

- **D-BENCH-9 — added and FIXED by this slice.**
  The defect, as probed in §1: citation *truth* and
  the *explain* side were scored by nothing — a
  cited-but-false claim passed TF-049's presence
  check identically to the true claim, and an
  unowned non-memory answer passed every citation
  check in the estate. TF-049's doc assigned the
  scoring to TF-123 without numbering it; this
  slice numbers it per the docket's rule (nothing
  found stays unnumbered) and fixes it in the
  scoring sense assigned: both legs now have a
  mechanical scorer (`benchmarks/memory_suite.py`)
  with a pinned harness baseline. Enforcement is
  unchanged and the limit is stated plainly: no
  gate yet *blocks* a cited-but-false answer at
  generation time — the suite scores the behaviour;
  wiring the truth judge into enforcement is not
  claimed here and stands as future work for the
  Part V / TF-129 line to take or refuse openly.
- No other entry moves. TF-016 totals are
  therefore 124 entries — **35 OPEN, 1
  OPEN-ENVIRONMENT, 85 FIXED, 3 CLOSED-BY-VERDICT**
  (D-BENCH-9 added + FIXED). Counting note, in
  TF-122's manner: the docket's header totals and
  a row-level recount have disagreed before this
  slice (the header counts D-LIC-2 FIXED while its
  row reads OPEN-narrowed, and the primary-owner
  part counts sum to 121 against the header's 123)
  — both mismatches are pre-existing, are preserved
  unchanged in shape (parts now sum 122 against
  124, D-BENCH-9's owner part VII counted), and are
  noted here rather than silently "repaired".

## 6. Filed limits

- The cases are **scripted synthetic fixtures**
  (TF-017): invented answers judged against
  TF-043's invented stored facts, never canon,
  never a model's live output; `data/` was never
  opened. The suite proves the scoring end to end;
  **a live small model's citation behaviour is NOT
  PROVEN** here (TF-125 / TF-126's road).
- Truth is judged by the filed phrase rules plus
  TF-049's term-overlap matcher — the
  deterministic, checkable subset. A false claim
  phrased to carry the required phrases and dodge
  the breaking ones is not caught; broader truth
  (semantic entailment against the stored fact) is
  judgement, not smuggled into a matcher.
- The explain axis inherits TF-049's matching
  limits in reverse: a memory claim paraphrased
  below the matcher's thresholds reads as
  non-memory and can hold on ownership alone.
- The suite baseline pins the scripted run only;
  TF-043's own baseline is consumed, never
  re-pinned, and its two filed findings (GeoNames
  threshold-dead at the defaults, the ConceptNet
  ferry edge riding shared-term queries) stand as
  TF-043 filed them.

Proof grade: FIXTURE — synthetic scripted cases
through the real TF-048 / TF-049 / TF-093 machinery
and TF-043's real retrieval on this VM. No live
model was run; that is NOT PROVEN, never claimed.
