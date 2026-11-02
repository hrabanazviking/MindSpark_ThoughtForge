# TF-097 — Provenance of Rescue

*A salvaged answer now carries its rescue record.
The estate lives in
`src/thoughtforge/refinement/rescue_provenance.py`
(`RESCUE_RULES` / `RESCUE_FIXTURES` as data,
`rescue_record_problems` /
`rescue_record_result_problems` /
`rescue_provenance_problems` /
`rescue_provenance_live_problems`, formats
`thoughtforge-rescue-provenance/1` for the estate
and `thoughtforge-rescue-record/1` for the record
itself), with the record built inside
`refinement/salvage.py`'s `forge()`, carried on
`SalvageResult.rescue_record` and the final
response record, and exported in the TF-038
trace's final section — so TF-084's joined trace
carries it inside its cognition section.
Docket: D-TRACE-1, filed by TF-084, FIXED here.*

## The plain truth first

Before this slice, a salvage result knew only its
path string — and the string could not tell the
story. Probed live at this slice's start, through
the real `FragmentSalvage`: TF-096's benign
*adopted* refine fixture and its voice-*rejected*
refine fixture both returned `salvage_path ==
"refine_pass_1"` with `passes_used == 1`. Adopted
and rejected were indistinguishable. TF-087 had
already filed why: `refine_pass_N` names passes
*attempted*, not adopted, and the rescue record
was assigned to carry that distinction.

The passes were also invisible to the trace.
TF-084's joined trace records the core's own leaf
calls, but the refine passes salvage makes
through the engine inside its own `forge()` were
recorded nowhere — the core hands the engine over
and cannot see them. TF-084 filed that as
D-TRACE-1 and named the trap in advance: an
aggregate invented afterwards from the salvage
result would be fabrication. The passes had to be
recorded by salvage itself, as each one runs.
That is what this slice does.

## The record

Every `forge()` result carries a rescue record
(format `thoughtforge-rescue-record/1`), built
from what the forge itself did and measured:

- **candidates** — every candidate as scored:
  id, char count, length / citation / composite
  scores, citations found. Sizes and scores,
  never texts (the texts live once, in the
  trace's candidates and final — TF-084's ruling).
- **selection** — the selected candidate id,
  whether TF-096's persona filter applied, and
  how many candidates were persona-consistent
  (`null` when no persona was in force).
- **fragments_kept** — the count salvage's own
  extraction kept.
- **refine_passes** — each pass individually:
  pass number; outcome (`adopted`,
  `rejected_fabrication`, `rejected_voice`,
  `not_improved`, `failed`); whether it counted
  in `passes_used`; its score and output size;
  the guards' own problem lists verbatim
  (TF-088's named codes such as `new_number:50`,
  TF-096's avoid entries); the failure string
  when it failed; and two latencies, named never
  conflated — `latency_ms`, the wall salvage
  measured around the engine call, and
  `reported_latency_ms`, what the engine itself
  reported (`None` when it reported nothing).
- **final** — `salvage_path` (the attempted
  naming, unchanged) beside `final_source`
  (`selected_draft` / `refine_pass` / `empty`)
  and `final_pass`: the adopted pass the final
  text actually came from, or `None`. The
  two-pass fixture pins the debt in one record:
  path `refine_pass_2`, text from pass 1, pass 2
  rejected as fabrication.

## A failed attempt is provenance too

An engine exception on pass 1 makes `forge()`
fall back to the best draft with `passes_used ==
0` — and before this slice that attempt left no
trace anywhere: the result looked exactly like a
turn where refine never ran. The record carries
the attempt (`outcome == "failed"`, the error
string verbatim, `counted_in_passes_used ==
False`), while `passes_used` keeps its standing
meaning — completed generations only.

## Into the trace

The core's `_compose_final` passes the record
onto the `FinalResponseRecord` (a new defaulted
field; knowledge-only and model-failed finals
carry `None`, stated, never an invented record).
TF-038's trace export files it in the final
section, and `trace_problems` judges a carried
record by this module's checker plus a
cross-check that its candidate rows resolve to
the trace's candidates. TF-084's joined trace
rides the cognition trace whole, so the joined
story now carries the passes inside its
cognition section; the inference section is
unchanged — its recorder records the core's own
leaf calls, and salvage's passes are not the
core's calls. TF-084's
`salvage_refine_passes_are_a_named_gap` rule
keeps its id and now records the closure:
D-TRACE-1 FIXED by TF-097, from salvage's side.

The record is derived, never trusted:
`rescue_record_problems` recomputes its internal
identities — candidate composites from their
filed scores, `passes_used` from the counted
passes, `final_pass` as the last adopted pass,
the final confidence from the selected candidate
or the adopted pass — and names any disagreement.

## What did not change

The scoring formula, TF-088's fabrication guard,
TF-096's voice guard, the adoption rule,
`passes_used` semantics, and `salvage_path`
naming are all unchanged; no new core method or
property is added — one defaulted field on
`SalvageResult`, one on `FinalResponseRecord`,
and the record riding the trace's existing final
section.

## Proof grade

FIXTURE — eight filed fixtures (empty, selection,
adopted, fabrication-rejected, voice-rejected,
not-improved, engine-failure fallback, and
adopt-then-fabricate over two passes) run live
through the real `FragmentSalvage` with stub
engines and the real `PromptBuilder`, each record
judged against its result and its filing; plus a
real core turn on a stub engine whose TF-038
trace and TF-084 joined trace both carry the
record and judge clean whole (two `not_improved`
passes recorded individually). A live model's
rescue record is NOT PROVEN — no model exists in
the tree — and never claimed; the record's shape
does not depend on who wrote the candidates.
