# TF-037 — The Small-Model Floor

*Part II, slice TF-037. Code:
`src/thoughtforge/cognition/small_model_floor.py`
(`FloorBudget`, `floor_budget`, `FloorFixtureEngine`,
`FloorObservation`, `check_floor_turn`, `run_floor_conversation`,
`salvage_passes_used`), plus the refinement-cap wiring in
`src/thoughtforge/cognition/config_surface.py` and
`src/thoughtforge/cognition/core.py`. Tests:
`tests/test_tf037_small_model_floor.py` (23 tests).*

## 1. The question

Part II proved the cognition core piece by piece, on comfortable
budgets. The floor question is whether the *whole* core holds
together at the smallest end of its own stated range: the
`phone_low` profile — the weakest hardware profile the tree
ships (~2 GB RAM, a model of at most 1.1 B parameters) — driven
by the smallest fixture model in the estate. "Simulated profile"
is the honest phrase throughout: the conversation *selects*
phone_low through TF-032's config surface on this VM. No phone
ran, and no real small model ran; §5 states what that leaves
unproven.

## 2. The floor budget — the profile's own numbers, read live

`floor_budget()` resolves every number from the shipped
phone_low JSON through the tree's own loaders
(`load_hardware_profile` + the selector's
`resolve_generation_params`) — a view, never a copied table:

| Dimension | Cap | Source in the profile |
|---|---|---|
| Response tokens | 180 | `generation.max_response_tokens` |
| Per-turn token budget | 180 | `model.token_budget_per_turn` |
| Prompt context | 512 word-tokens | `model.max_context_tokens` |
| Drafts per turn | 2 | `model.draft_count` |
| Refinement passes | 1 | `generation.max_refinement_passes` |
| Memory-bundle facts | 5 | `memory.max_memory_bundle_facts` |
| Latency | 500 / 3,000 / 1,500 ms, total 5,000 | TF-033's phone_low budget |

The binding response cap is the *tighter* of the two token
claims (both are 180 on phone_low, so the rule costs nothing
here and can never loosen the floor on a profile whose claims
disagree). The bundle-fact cap is recorded as data but **not
exercised** by the floor run: the run's store is empty, so its
bundles are empty, and populated-store retrieval is Part III's
road (TF-041 onward). Claiming it would be false.

## 3. Defect found and fixed in scope: the refinement cap was dead

The profile's `max_refinement_passes` was resolved by the
inference resolver and read by **no cognition code** — salvage
ran a flat 2 refine passes on every profile (TF-012's D-PROF-4).
Measured by this slice's own pre-fix floor probe: under
phone_low, **7 of 20 turns ran `refine_pass_2`** — a second
small-model call the floor profile expressly forbids (it
promises 1; pi_zero promises 0 and also got 2).

The fix follows TF-032's generation-cap pattern exactly:
`ConversationConfig` now resolves
`generation.max_refinement_passes` alongside
`max_response_tokens` from the selected profile, and the core's
compose step hands it to `FragmentSalvage.forge` as
`max_passes`. With no profile selected the standing 2 passes
stand byte-identical — TF-028's 54-call pin passes untouched,
and a default core's refining turn still lands on
`refine_pass_2` (pinned in this slice's tests beside pi_zero's
`best_draft`, candidates-only, 2 calls for the same turn).

## 4. The smallest fixture model

`FloorFixtureEngine` wraps TF-028's coherence fixture — same
script, same request-line routing, same deterministic answers
in TF-017's synthetic world — and adds the one discipline the
floor demands: a `max_tokens` cap on the generation params is
honoured by word-truncation, as a real small model's sampler
must honour it, and every call's cap is recorded. (TF-028's
engine deliberately *ignores* params so its drift measurements
stay pure; the floor engine *obeys* the cap, because the floor
question is precisely whether the core behaves when the model
is this small and this constrained.)

## 5. What the floor run measured

TF-028's full 20-turn coherence script, threaded history and
all, through a phone_low-selected core, judged per turn by the
pure `check_floor_turn` on five dimensions:

- **Within the floor on all 20 turns** — no overrun on any
  dimension on any turn.
- Measured maxima: response **30** tokens (cap 180), largest
  generation prompt **338** word-tokens (cap 512), **2**
  candidates per turn (cap 2), **1** refine pass (cap 1) on
  exactly the 7 turns that refine at all.
- Engine calls: **47** (40 candidate + 7 refine) — the pre-fix
  run's 54, minus exactly the 7 forbidden second passes.
- **Coherence held**: TF-028's scorer over the same transcript
  reads PASS, mean / early / late 1.000, drift 0.000, zero
  contradictions. The floor is not merely within its budgets —
  the conversation keeps its facts straight inside them.
- The checker is a gate, not a decoration: a stub engine
  returning a 400-word answer at the 180 cap is caught with
  `response_tokens` named, and each dimension's overrun —
  including a latency stage overrun hiding under a
  within-budget total — is unit-pinned by name.

**Proof grade, stated plainly:** FIXTURE model on a SIMULATED
profile, on this VM. Nothing here claims a real small model's
behaviour at these budgets (TF-126 owns the live re-run) or
phone hardware's (TF-069/TF-128 own the profile panels).
Latency at the floor is judged against TF-033's derived
phone_low budget on VM timings — real phone timings are not
proven.

## 6. Docket

- **D-PROF-4 narrowed by TF-037**: the resolved
  `max_refinement_passes` is now live in the cognition core
  (config surface → compose → salvage). TF-098 keeps the
  general refinement-cost discipline — measured cost of a
  refine pass across all profiles, as a budget not a habit.
- **D-PROF-3 narrowed again**: of TF-012's flat list, the core
  now honours per profile — generation token cap (TF-032),
  latency budgets (TF-033), refinement pass cap (TF-037).
  Still flat: retrieval top_k, history budgets, mode
  temperatures — TF-098 and the remaining owners keep those.
- No new defect was discovered beyond the refinement cap
  above, which this slice fixed rather than logging.
