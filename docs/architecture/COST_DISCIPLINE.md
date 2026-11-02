# TF-098 — Cost Discipline

*Refinement passes are now capped per profile at
the engine-call site itself, and what refinement
spends is counted. The estate lives in
`src/thoughtforge/refinement/cost_discipline.py`
(`COST_RULES` / `COST_FIXTURES` /
`COST_RECORD_FIXTURES` as data,
`resolve_refine_budget` / `profile_refine_cap` /
`refine_cost`, the checkers
`cost_discipline_problems` /
`cost_discipline_live_problems`, format
`thoughtforge-cost-discipline/1`), with the cap
enforced inside `refinement/salvage.py`'s
`forge()` and the core passing its selected
profile id through. Docket: D-COST-1, found by
this slice's probes, FIXED here; D-PROF-4's
general share closed here (its floor share was
TF-037's).*

## The plain truth first

A refine pass is a second (or third, or fifth)
call to a small model on small hardware. Before
this slice, the only thing standing between a
caller and an unbounded number of those calls
was the core: TF-032/TF-037's config surface
resolves the selected profile's
`generation.max_refinement_passes` and hands it
to salvage as `max_passes`. Probed live at this
slice's start, through the real `FragmentSalvage`
with a counting stub engine whose every output
improves: a direct `forge(max_passes=5)` made
**5 engine calls** and returned `refine_pass_5` —
beyond every shipped profile's cap, the largest
of which is 3. Salvage itself knew no profile
existed, so every direct caller bypassed the
caps entirely. A negative request silently meant
zero. And nothing anywhere counted what
refinement had spent: TF-097's rescue record
carries each pass's outcome and latencies, but no
total existed — no attempted / adopted / wasted
counts, no latency sums — and token counts are
read nowhere in `refinement/`.

That gap is filed as D-COST-1 and fixed here.

## The caps are the profiles' own

There is no cap table in the new module.
`profile_refine_cap` reads a profile's
`generation.max_refinement_passes` live from its
shipped JSON through `load_hardware_profile`
(the TF-033 shape), so the caps in force are
always the profiles' own:

| profile | cap |
|---|---|
| pi_zero | 0 |
| phone_low | 1 |
| pi_5 | 2 |
| desktop_cpu | 2 |
| desktop_gpu | 3 |
| server_gpu | 3 |

An unknown profile raises `ValueError` naming
the valid ids — a bogus profile never silently
inherits a cap. The data checker proves the
filed fixtures cover every shipped profile and
that each fixture's effective count equals the
live resolution.

## A ceiling, not a target

`resolve_refine_budget(requested, profile_id)`
returns a `RefineBudget` (profile, request, cap,
effective, governed-by). Under a profile the
effective count is `min(requested, cap)` — a
caller asking for 1 on desktop_gpu gets 1, never
the cap of 3. No request named means the standing
default (2, salvage's own `_MAX_REFINE_PASSES`;
the checker verifies TF-032's
`DEFAULT_REFINE_PASSES` agrees) is the request.
A negative or non-integer request raises
`ValueError` naming the value through this
resolver — a caller bug is never silently read
as zero. (Through `forge()` with no profile
named, the pre-slice legs stand unchanged,
stated in the module rather than papered over.)

**Zero means zero.** Under pi_zero, `forge()`
makes no engine call at all and returns the best
draft — the profile's promise kept at the call
site, not only through the core.

**No profile, no invented cap.** With no profile
named there is no cap to apply, and this slice
invents none: the caller's request stands,
exactly as before — one filed fixture pins a
no-profile request of 5 landing at 5, so the
boundary is tested, not assumed. Through the
core the leg is already disciplined: TF-032
hands compose the standing 2 when no profile is
selected.

## Salvage enforces it itself

`FragmentSalvage.forge` gains a defaulted
`profile_id` keyword. When one is supplied, the
effective pass count is resolved through this
module before any engine call is made; the
counting-engine fixtures prove the calls land
exactly on the cap for every shipped profile
(requested 5 throughout). The core's
`_compose_final` passes its selected profile id
beside the already-resolved `max_passes` —
defence in depth at the call site; the effective
number through the core is unchanged on every
profile, so core behaviour is byte-identical.
With no profile id, every leg of `forge()` is
byte-identical to before this slice.

## The cost, counted

`refine_cost` totals a TF-097 rescue record —
after judging it by TF-097's own checker (an
unsound record raises and is never totalled) —
in the unit the profiles budget: **engine
calls**. It counts passes attempted, completed,
adopted, completed-but-not-adopted (the wasted
spend: a second call that bought nothing is
named as such), and failed; and it sums the two
latencies apart, exactly as TF-097 files them —
`wall_ms` (salvage's measured wall) and
`reported_ms` (what engines reported, `None`
when none did — never a fabricated zero). The
filed record fixtures pin the arithmetic against
TF-097's own fixtures: adopt-then-fabricate
totals 2 attempted / 2 completed / 1 adopted /
1 not adopted / 7.0 reported ms; the engine-
failure record totals 1 attempted / 0 completed
/ 1 failed with `reported_ms` None.

Filed limit: tokens per pass are not carried in
TF-097's record and are not invented here —
TF-133's cost panel owns tokens + time per
suite, and TF-101 scores the stack over
TF-100's chaos corpus.

## What did not change

The scoring formula, TF-088's fabrication guard,
TF-096's voice guard, the adoption rule,
`passes_used` semantics, and `salvage_path`
naming are unchanged; the profile JSONs are read,
never edited; the config surface's resolution is
reused as the core's request, never duplicated.

## Proof grade

FIXTURE — every shipped profile's cap exercised
live through the real `FragmentSalvage` with a
counting stub engine on this VM, the cost
accounting proven against TF-097's filed
records, the resolver's error legs raising by
name. A live model's refine cost is NOT PROVEN
— no model exists in the tree — and never
claimed.
