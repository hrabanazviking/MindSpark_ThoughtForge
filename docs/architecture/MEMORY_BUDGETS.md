# TF-056 — Memory Budgets: Store Size + Retrieval Cost Capped per Profile

*Part III, slice TF-056. Code:
`src/thoughtforge/knowledge/memory_budgets.py` (the budgets as
data, resolved live from the shipped profile JSONs; the
retrieval plan; store measurement; the budget report;
`enforce_budget`; and `memory_budgets_problems()` re-deriving
every claim below from the live tree), wiring in
`knowledge/lifecycle.py` (an optional profile whose caps the
prune enforces), `knowledge/forge.py` (optional sql/vector
retrieval caps), and `cognition/core.py` (lifecycle built on
the selected profile, retrieval through the plan, the
`memory_budget` property). Tests:
`tests/test_tf056_memory_budgets.py`.*

## 1. The question and the measured starting state

The roadmap asks: is store size and retrieval cost capped per
hardware profile, with eviction by the lifecycle rules?
TF-012 had already named the gap (D-PROF-1): every profile
JSON carries `memory` and `retrieval` sections that no code
read. Measured again this tick, by direct probe:

- **The lifecycle's caps were flat.** `STORE_CAPS` (facts /
  preferences / patterns 128 soft / 256 hard, episodes
  256 / 512) applied on every profile. Seventy distinct facts
  — over phone_low's hard cap of 64 and pi_zero's of 32 — went
  through a default light prune untouched: light's score
  thresholds are unreachable (TF-045), so only caps act under
  light, and the cap in force was 256.
- **The flat table is one profile's budget.** STORE_CAPS is
  exactly desktop_cpu/desktop_gpu's JSON caps for the four
  record kinds. Phones and servers were both silently running
  the desktop's budget. The two caps profiles do not carry
  (recent_queue 64/128, open_loops 6/8) are the flat table's
  alone, and stay flat — stated, not smuggled.
- **Retrieval cost was flat where hardware is smallest.**
  The core retrieved at top_k=10 on every profile, over
  phone_low's bundle cap (5), sql cap (5) and vector cap (3),
  and pi_zero's (3 / 3 / 2); the forge's legs scale from that
  top_k (sql candidates 2 × top_k, vector top_k), so the cost
  multiplied from an uncapped base.

## 2. The budgets, as data

`budget_for_profile(profile_id)` resolves a `MemoryBudget`
LIVE from the shipped JSON through the tree's own loader —
no second copy of the numbers exists to drift:

| Profile | Facts / Prefs / Patterns hard | Episodes hard | Bundle | SQL | Vector |
|---|---|---|---|---|---|
| pi_zero | 32 | 64 | 3 | 3 | 2 |
| phone_low | 64 | 128 | 5 | 5 | 3 |
| pi_5 | 128 | 256 | 10 | 10 | 5 |
| desktop_cpu / desktop_gpu | 256 | 512 | 15 | 15 | 8 / 10 |
| server_gpu | 512 | 1024 | 30 | 30 | 20 |

(Soft caps are exactly half the hard caps in every shipped
JSON; the checker pins soft ≤ hard and the per-kind ladder
pi_zero ≤ phone_low ≤ pi_5 ≤ desktop ≤ server as monotonic.)
An unknown profile raises `ValueError` naming the valid ids —
a budget is never silently another profile's.

## 3. Store size, stated honestly

`measure_store` reports live record counts per kind (loaded
records — archived/deleted rows are already rewritten out of
their files, TF-045's physical retirement) and file bytes
PER FILE. Per-kind bytes are not claimed: facts and
preferences share one file, so a per-kind byte split would be
invented. `budget_report` judges each kind: `within_soft`,
`over_soft` (reported, never evicted — soft is a warning
line), or `over_hard` (eviction is owed); `within_budget`
means no kind is over hard.

## 4. Eviction — by the lifecycle rules, nothing else

`enforce_budget(store, profile_id, mode="light")` runs one
ordinary lifecycle prune whose caps are the profile's.
`MemoryLifecycle(store, profile_id=…)` resolves those caps at
construction; with no profile its caps are the flat table,
byte-identical to before this slice. The eviction itself is
the lifecycle's own cap leg (TF-045): lowest survival score
first, protected records never cap-evicted. Consequences,
pinned by tests:

- 70 facts under phone_low, light pass: exactly 64 remain,
  the 6 lowest-survival records gone; the same store under
  the default lifecycle is untouched.
- A protected surplus cannot be evicted away: with 68
  protected facts + 2 plain over the phone cap, only the 2
  plain records go, 68 remain, and the after-report still
  says `over_hard` — protection outranks the cap and the
  report refuses to call that "within budget".
- Episodes and patterns follow their own profile caps
  (pi_zero: 70 episodes → 64, 70 patterns → 32).

## 5. Retrieval cost, capped at all three legs

`retrieval_plan(profile_id, requested_top_k=10)` caps and
never raises: bundle = min(requested, max_memory_bundle_facts)
— phone_low 5, pi_zero 3, pi_5/desktop/server pass the
standing 10 through (a 25 request under server's 30 cap also
passes). The core, when a profile is selected (TF-032),
retrieves through the plan; `KnowledgeForge.retrieve` accepts
optional `sql_max_results` / `vector_top_k` and mins them
against its own scaling (sql 2 × top_k, vector top_k) — so no
leg exceeds its profile's stated cost (pinned by a spy: the
phone plan reaches the SQL leg as 5, not 20). With no caps
passed, the forge's scaling is byte-identical to before, and
a core with no selected profile makes the standing flat call
exactly as before (pinned argument-for-argument). The core's
`memory_budget` property exposes the selected profile's
budget (None unselected) — recorded as a post-freeze minor
addition (TF-034 rules) in `POST_FREEZE_ADDITIONS`.

## 6. Docket

- **D-PROF-1 narrowed:** the profiles' `memory` + `retrieval`
  sections are now read live and enforced. Its `deployment`
  section, `max_params_b` and `recommended_models` remain
  unread — TF-069 (profile runner) and TF-110 (subset craft)
  keep those shares.
- **D-PROF-3 narrowed again:** retrieval top_k and store
  eviction are no longer flat when a profile is selected.
  History budgets and mode temperatures stay flat, owned by
  their existing slices.
- Totals unchanged: 53 entries — 40 OPEN / 1 OPEN-ENVIRONMENT
  / 10 FIXED / 2 CLOSED-BY-VERDICT (narrowings, not closures).

## 7. Boundaries, stated

- Budgets cap COUNTS per kind and retrieval sizes; they do
  not cap the knowledge DB (the shared world reference estate,
  TF-046 — subset sizing for small profiles is TF-110's) and
  they do not impose a byte ceiling on the memory store,
  because no shipped profile states one — inventing one would
  be a claim, not a budget. File bytes are measured and
  reported so a future byte budget has honest data.
- Threshold tables (archive/delete scores) stay flat across
  profiles by design: budgets govern how MUCH is kept, modes
  govern what is retired; per-profile threshold tuning is
  still NOT PROVEN and not claimed.
- Proof grade FIXTURE (synthetic stores, the TF-043 corpus,
  stub/spy retrieval legs, VM) — budget behaviour over a real
  lived-in store and on real phone/Pi hardware is NOT PROVEN
  (TF-069/TF-082/TF-173 own the hardware legs), never claimed.
