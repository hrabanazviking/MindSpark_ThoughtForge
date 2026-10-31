# TF-033 — Latency Budgets per Profile for the Core Loop

*Part II, slice TF-033. Code:
`src/thoughtforge/cognition/latency_budgets.py`
(`LatencyBudget`, `StageVerdict`, `LatencyVerdict`,
`budget_for_profile`, `all_budgets`, `stage_measurements`,
`check_turn_latency`), exposed on `ThoughtForgeCore` as
`core.latency_budget` / `core.last_turn_latency()`. Tests:
`tests/test_tf033_latency_budgets.py` (23 tests).*

## 1. The gap, as found

The core loop measured time but budgeted none of it. TF-025 made
the turn state's three stage timings real (`retrieval_ms`,
`generation_ms`, `salvage_ms`), and the perf tracker records
`think.total` — but no code or test anywhere stated how long a
turn may take, on any profile. A turn ten times slower than its
neighbours was indistinguishable from a fast one.

The profiles, meanwhile, each carried one honest latency number
nobody read: `retrieval.retrieval_timeout_ms` (TF-012 recorded
the whole `retrieval` section as read by no code — D-PROF-1).

## 2. The budget table, and where every number comes from

| Profile | Retrieval (ms) | Generation (ms) | Salvage (ms) | Total (ms) |
|---|---|---|---|---|
| server_gpu | 100 | 500 | 250 | 850 |
| desktop_gpu | 200 | 750 | 375 | 1,325 |
| desktop_cpu (VM) | 500 | 1,000 | 500 | 2,000 |
| pi_5 | 1,000 | 2,000 | 1,000 | 4,000 |
| phone_low | 500 | 3,000 | 1,500 | 5,000 |
| pi_zero | 2,000 | 5,000 | 2,500 | 9,500 |

- **Retrieval** is each profile's OWN stated
  `retrieval_timeout_ms`, read live from its JSON through
  `load_hardware_profile` at resolution time — never copied into
  a second table. (Note phone_low's 500: the profile states the
  same retrieval timeout as desktop_cpu, so the budget does too;
  the factor below is not applied to a number the profile itself
  owns.) This is the first live consumer of any profile
  `retrieval` value; D-PROF-1 keeps the rest of that section.
- **Generation / salvage** are campaign-set ceilings: the
  desktop_cpu base (1,000 / 500 ms) scaled by a stated hardware
  factor per profile (server_gpu ×0.5, desktop_gpu ×0.75,
  desktop_cpu ×1, pi_5 ×2, phone_low ×3, pi_zero ×5), recorded as
  data in `PROFILE_FACTORS`. A shipped profile with no factor
  recorded fails resolution loudly rather than silently
  inheriting desktop numbers.
- **Total** is never an independent number: it is the stage sum
  by construction (`LatencyBudget.total_ms`), so the table
  cannot contradict itself.

## 3. What the VM measurements actually showed

Measured this slice on this VM (7 turns each, empty-store gate
path and an instant stub backend, TF-032/TF-031 test shapes):

- Every stage measured **0 ms** at the turn state's integer-ms
  granularity, on both paths — the core loop's own work at
  fixture scale is sub-millisecond per stage.
- Wall-clock per turn: ~1 ms warm; the very first turn of a
  process measured 51 ms (warm-up).
- `ProfileSelector().auto_select()` on this machine returns
  **desktop_cpu** (re-verified live; TF-012's finding stands) —
  that is the VM profile the enforcement below binds to, and a
  test pins the selector's answer so a machine change fails
  loudly instead of enforcing the wrong profile's budget.

The ceilings therefore carry roughly an order of magnitude of
headroom over fixture reality — deliberately: this VM is shared
with other campaigns, and a wall measured under contention is
not a baseline. These are ceilings (a slow turn is a defect),
not targets and not predictions.

**Proof grade, stated plainly:** FIXTURE generation on the VM
profile only. No real model ran on this VM for this slice, so
nothing here claims a real model's generation latency — that is
TF-079 (cold-start budgets) and TF-175 (the laptop panel)'s
road. The other five profiles' budgets are *derived* by the
stated rule, NOT PROVEN on their hardware; TF-069 (profile
runner) and TF-128 (profile panels) own proving them. A
populated-store retrieval timing likewise belongs to TF-043's
fixtures; the retrieval ceiling here is the profile's own
stated timeout, enforced on the turns the tests run.

## 4. The checker and the core surface

`check_turn_latency(turn, budget, total_ms=None)` is pure: it
reads the turn state's three real stage fields and returns a
`LatencyVerdict` — per-stage `StageVerdict`s (measured, budget,
within), `overruns` naming the offenders in turn order, and
`within_budget`. A `total` stage is judged only when the caller
supplies its own wall-clock measurement around the turn: the
turn state carries no total field, and presenting a stage-sum
as a wall total would be a falsehood. A stage that did not run
measures 0 (TF-025: salvage 0 on the knowledge-only path means
"did not run") and is within budget; the 0 is recorded in the
verdict, visible, never hidden.

On the core: `core.latency_budget` resolves the TF-032 selected
profile's budget, defaulting to the VM profile (desktop_cpu)
when no profile was selected — a verdict always has a stated
budget to answer to. `core.last_turn_latency(total_ms=None)`
returns the last finished turn's verdict, or None before the
first turn.

## 5. Enforcement (in tests, per the roadmap's words)

On the VM profile, through the real core:

- knowledge-only turns (×3) and stub-backend turns (×3) land
  inside every stage budget and the wall-clock total;
- a deliberately slow backend (2.5 s against the 1,000 ms
  generation / 2,000 ms total budgets) is caught with overruns
  named exactly `["generation", "total"]` — the budget is a
  gate, not a decoration;
- the checker itself is unit-pinned (derivation, at-budget
  boundary, overrun order, no-total-without-wall, dict shapes).

## 6. Docket

D-PROF-3 (core profile-blindness) is **narrowed again**, not
closed: a turn's latency is now judged per profile, and the
retrieval ceiling is a live profile value. Retrieval top_k,
salvage pass counts, history budgets and mode temperatures
stay flat — TF-037 (small-model floor) and TF-098 (refinement
cost) keep those remainders. No new defect was discovered by
this slice's measurements; the sub-millisecond stage readings
are a granularity fact about the integer-ms fields, recorded
in §3, not a defect — wall-clock verdicts carry the fine grain.
