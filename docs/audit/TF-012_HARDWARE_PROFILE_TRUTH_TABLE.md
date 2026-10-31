# TF-012 — Hardware-Profile Truth Table v1 (Part I Audit)

Measured 2026-10-30 on branch `yrsa/thoughtforge-road-build`.
Sources: the six shipped profiles in `hardware_profiles/*.json`
(the claims), `src/thoughtforge/inference/profiles.py` (selector +
resolver, the only profile-reading code), `utils/config.py`
(loader), and the cognition core's actual constants (the reality
a user turn meets). BUILD_PLAN_v1's profile table (BP §, lines
41–46) agrees with the JSONs on every row except pi_zero's model
size (plan "1B subset" vs JSON `max_params_b` 0.5).

## 1. The headline: profiles govern one engine, not the product

Exactly one production consumer reads profiles at all:
`TurboQuantEngine.load()` (inference/turboquant.py:145–151)
auto-selects a profile and resolves generation params from it.
The cognition core — the layer every user turn actually runs —
**never loads a profile** (TF-010's D-BENCH-1, generalised):
core constructs `FragmentSalvage()`, `ScaffoldBuilder()`,
`EnforcementGate()`, `ChatHistory()` all with flat defaults, and
creates the engine with `model_path` only (core.py:228); the
profile is chosen inside the engine, for the engine.

Even inside the resolver, only part of each JSON is read.
`resolve_generation_params` (profiles.py:378) consumes
`hardware.{inference_backend, inference_backend_fallback,
vram_gb}`, `model.{quantization, draft_count, max_context_tokens,
token_budget_per_turn}`, and `generation.{max_response_tokens,
temperature, repeat_penalty, top_p, max_refinement_passes}` —
and its resolved `max_refinement_passes` is then **read by no
code at all** (0 references outside profiles.py): the salvage
layer's pass count is its own flat `_MAX_REFINE_PASSES = 2`
(refinement/salvage.py:41) on every profile, including pi_zero,
whose JSON promises 0.

**Read by no code anywhere** (grep-verified, 0 hits outside
profiles.py): the entire `memory` section of every profile
(`max_memory_bundle_facts`, `max_vector_results`, all eight
episodic/user-facts/patterns/preferences soft+hard caps), the
entire `retrieval` section (`sql_max_results`, `vector_top_k`,
`retrieval_timeout_ms`), `model.max_params_b`,
`model.recommended_models`, and the entire `deployment` section.
The caps' would-be enforcer, MemoryLifecycle, is itself unwired
(TF-011 D-ORPHAN-1) — the budget system is doubly decorative.

## 2. The truth table

"Applied" = reaches a running turn. Engine-only = affects
TurboQuantEngine generation, never the core's own behaviour.

| Profile | Claimed budget (JSON) | Selector's real floor (profiles.py `_select_id`) | What the code actually applies |
|---|---|---|---|
| `pi_zero` | 0.5 GB RAM, ≤0.5B q2_k, ctx 256, 120 tok/turn, drafts 1, refinement 0, bundle 3 facts, subset ≤0.5 GB | ARM/Pi with RAM **<2 GB** — or a *mobile* device with RAM <1 GB (a phone can land on the Pi Zero profile) | Engine-only: quant/drafts/ctx/token budget. Refinement is 2 passes, not 0. Memory/retrieval caps: nothing. Subset: never built (TF-011 D-ORPHAN-3) |
| `phone_low` | ~2 GB RAM, ≤1.1B q2_k, ctx 512, 180 tok/turn, drafts 2, refinement 1, bundle 5 facts | Mobile with RAM **≥1 GB** (any size above — no higher phone tier exists); also the fallback for **x86 machines with RAM <6 GB** (a low-RAM laptop gets the phone profile) | Engine-only, as above. A 16 GB phone and a 1 GB phone receive the identical 2 GB budget |
| `pi_5` | "4GB+", ≤3B q4_k_m, ctx 2048, 200 tok/turn, drafts 3, refinement 2, bundle 10 facts | Pi/ARM-non-mobile with RAM **≥2 GB** — 2–3.9 GB devices receive a profile budgeted for 4 GB | Engine-only, as above |
| `desktop_cpu` | "8GB+", ≤7B q4_k_m, ctx 4096, 250 tok/turn, drafts 3, refinement 2, bundle 15 facts | x86 with RAM **≥6 GB**. **Proven live this run:** this VM (7.7 GB RAM, x86_64) auto-selects desktop_cpu — under the JSON's own stated 8 GB floor | Engine-only, as above |
| `desktop_gpu` | VRAM 12 GB nominal ("8–16GB VRAM"), ≤13B q8_0, ctx 8192, 500 tok/turn, drafts 3, refinement 3 | VRAM **≥6 GB** + any GPU backend (CUDA/Vulkan/DirectML/ROCm) — a 6 GB card receives the 12 GB-budget profile; resolve ladder gives it n_gpu_layers=20 | Engine-only, as above; refinement resolved to 3 but unused (flat 2 in salvage) |
| `server_gpu` | 64 GB RAM, 24 GB+ VRAM, ≤70B fp16, ctx 32768, 2000 tok/turn, drafts 5, refinement 3, bundle 30 facts | VRAM ≥24 + CUDA/ROCm — **RAM is never checked**: an 8 GB-RAM machine with a 24 GB card selects the 64 GB profile and its 2000-token budgets | Engine-only, as above |

Global flat reality, identical on all six profiles (core's own
constants, none profile-derived): retrieval `top_k=10`
(core.py:251, over forge defaults sql 15 / vector 10);
unified-backend generation `max_tokens=512` (core.py:332);
knowledge-only citations capped at 5 records (core.py:467);
chat window 20 turns / 2000 chars (chat_history.py:28, :66);
temperatures from core's per-mode table 0.50–0.80
(core.py `_mode_temperature`), not any profile's flat 0.7.

Two smaller selector truths: the class docstring's rule order
(desktop_cpu third, phone/pi after) is not the code's order
(Pi/ARM and mobile are decided before the x86 rule, and the
docstring never mentions the x86→phone_low fallback); and when
RAM detection fails entirely, the detector *assumes 4 GB*
(profiles.py `_detect_ram_gb`) — pi_5 on ARM, phone_low on x86,
from a guess. The `n_gpu_layers` ladder (0/20/40/−1 at VRAM
0/6/12/24) exists only in the resolver; no JSON states it.

## 3. What is genuinely sound

- All six JSONs are valid, load, and are exercised by tests:
  test_phase2_inference.py covers the selector, the loader, and
  `resolve_generation_params` per profile (its lines 92–245).
  The JSON↔resolver half of the system is tested; the
  profile↔core half has no test because there is no connection
  to test.
- The plan, the JSONs, and the resolver's defaults agree with
  each other wherever the resolver reads. Every lie in this
  table is a *wiring* lie (claimed budgets nobody enforces) or a
  *threshold* lie (selector floors below the JSONs' own stated
  floors), not a data inconsistency.

## 4. Verdict for the road

- **D-PROF-1** — Profile `memory`/`retrieval`/`deployment`
  sections and `max_params_b`/`recommended_models` are read by
  no code: the per-profile budget claims are unenforced on
  every profile. Owners: TF-056 (memory budgets), TF-069
  (profile runner), TF-110 (subset craft).
- **D-PROF-2** — Selector floors ≠ the JSONs' claimed floors on
  all six rows (table §2), including phone_low as an x86
  fallback and pi_zero as a phone landing. Owner: TF-069.
- **D-PROF-3** — The cognition core is profile-blind: flat
  retrieval, generation, salvage, history, and temperature
  constants on every profile. Owners: TF-032 (cognition config
  surface), TF-033 (latency budgets), TF-037 (small-model
  floor). This is the parent defect of D-BENCH-1 (TF-010).
- **D-PROF-4** — Resolved `max_refinement_passes` is computed
  and never used; salvage runs 2 passes even on pi_zero's
  promised 0. Owner: TF-098 (refinement cost discipline).
- Truth-table v2 is TF-086's to issue, from TF-069's runner
  measurements — v1's standard: a budget is "true" only when a
  named line of code enforces it.

Suite at banking: 618 passed, 1 skipped, 1 known env failure
(D-ENV-1) under the TF-004 method — baseline held, no new failures.
