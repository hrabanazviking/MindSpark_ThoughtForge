# TF-069 — Profile Runner

**The plain truth first: a profile nobody boots is a
brochure.** The six hardware profiles declared hardware,
budgets, quant defaults, recommended models and deployment
shapes — and, until this slice, nothing ever booted one
end-to-end. The pieces existed separately (TF-056 budgets,
TF-068's quant policy, the core, the engine), and the
engine's own boot was shaped by the *host*, not the
profile: on this VM a desktop_gpu load selected vulkan and
a server_gpu load fell back to cpu, whatever the profile
declares. This slice builds the runner that boots each
profile's stack **in simulation** — hardware taken from
the profile's own JSON, never from host detection — with
the profile's RAM ceiling enforced as a **hard limit**.
The estate lives in `src/thoughtforge/inference/profile_runner.py`
(format `thoughtforge-profile-runner/1`).

## The boot — nine legs, in order

1. **profile** — the shipped JSON loads live through the
   tree's own loader; the id matches.
2. **params** — `resolve_generation_params` resolves, and
   the resolved quant equals TF-068's policy default.
3. **selector** — a `HardwareSnapshot` built from the
   profile's *declared* hardware (ram/vram/arch from the
   JSON, cuda when the declared backend is cuda, the
   pi/phone class flags from the profile's own class) is
   fed to the selector's own rule. **All six profiles
   select themselves** on their declared machines.
4. **memory_budget** — TF-056's budget resolves live; the
   booted core's lifecycle carries exactly the profile's
   caps; a fresh store reports within budget; the
   retrieval plan only caps (phone_low → bundle 5,
   pi_zero → 3).
5. **core** — a real `ThoughtForgeCore` boots in temp
   dirs on the profile and answers one knowledge-only
   turn. Per profile: max response tokens 180 / 120 /
   250 / 400 / 600 / 2000, refinement passes as declared
   (pi_zero → 0).
6. **engine** — a real no-model `TurboQuantEngine` load
   resolves the same params and records TF-068's
   `no_model` quant verdict with the policy default as
   the expected quant. The backend it picks is the
   **host's** truth, recorded as `host_backend` — never
   claimed as the profile's declared backend.
7. **ram_ceiling** — the hard leg, below.
8. **recommended_models** — each recommended name judged
   against the profile's own `max_params_b` where the
   name carries a parameter count; a name carrying none
   is `unprovable`, never guessed.
9. **deployment** — the deployment section read live and
   reported, extras included (pi_zero's
   `knowledge_subset` + its 0.5 GB subset cap, server_gpu's
   `multi_gpu`). Reported as data only: several shipped
   runtime lists name `docker` — that is the profile's
   claim on paper, and this runner never runs Docker.

## The RAM ceiling — hard, calibrated from the tree

A model's accounted size is its parameter count × a
GB-per-billion-parameters figure:

- For catalogue quants the figure is **calibrated from
  the curated catalogue itself**, computed live: parse
  each entry's parameter count from its name/description,
  take per quant the **maximum** size/params ratio (a
  ceiling is judged at the worst in-tree figure, not the
  average). Measured: Q4_K_M 0.667 (Llama-3.2-3B is the
  worst), Q5_K_M 0.833, Q8_0 1.100 (Mistral-7B),
  IQ2_XS 0.279. Catalogue entries carrying no count (the
  Phi-3 rows) are counted as unparseable and never used.
- **fp16 is definitional**: 16 bits = 2.0 GB per billion
  parameters; no calibration needed.
- **q2_k has no figure in-tree at all** — no catalogue
  entry declares it. Its accounted size is `unprovable`,
  filed as a gap, never estimated from a neighbour.

Enforcement has two forms:

- **A requested model is gated.** `run_profile_boot(pid,
  requested_model=(params_b, quant))` judges the request
  first: accounted size above the ceiling
  (ram_gb + vram_gb, as declared) → the boot is
  **refused before any stack leg runs**; an unprovable
  quant is likewise refused — a ceiling that cannot be
  computed cannot be passed silently. A fitting request
  boots.
- **With no request, the leg judges the profile's own
  cap**: `max_params_b` at the profile's default quant
  against the same ceiling, filed as data.

## Measured ceiling results (D-PROF-5)

| Profile | Cap accounting | Ceiling | Verdict |
|---|---|---|---|
| phone_low | 1.1B × q2_k — uncalibrated | 2 GB | **unprovable** |
| pi_zero | 0.5B × q2_k — uncalibrated | 0.5 GB | **unprovable** |
| pi_5 | 3B × 0.667 = 2.0 GB | 4 GB | fit |
| desktop_cpu | 7B × 0.667 = 4.67 GB | 8 GB | fit |
| desktop_gpu | 13B × 1.1 = 14.3 GB | 28 GB | fit |
| server_gpu | 70B × 2.0 = 140.0 GB | 88 GB | **exceeds** |

And two recommended names exceed their own profile's
cap, measured by name-parsing alone:

- **pi_zero recommends TinyLlama-1.1B** against its own
  0.5B cap.
- **desktop_cpu recommends LLaMA-3-8B** against its own
  7.0B cap.

The Phi-3 recommendations (pi_5, desktop_cpu) carry no
parameter count in their names and are filed
`unprovable`. Name parsing reads the largest `<number>B`
token — for Mixtral-8x7B that is 7, an understatement of
the mixture's total, stated here so nobody mistakes the
parser for a spec sheet.

**Overall verdicts:** every stack boots
(`boots_with_ceiling_gaps` for five profiles, each gap
named above); **desktop_gpu boots gap-free**. The two
q2_k rows are TF-068's constrained exceptions keeping
their stated debt — TF-127's live panel, or a shipped
q2_k size figure, is the road that proves them; the
server_gpu fp16-at-cap row and the two over-cap
recommendations are assigned to TF-086's truth table v2
to settle (raise the declared memory, lower the cap, or
change the default — a policy call, not a runner call).

## Docket

- **D-PROF-1 narrowed again**: `deployment`,
  `max_params_b` and `recommended_models` are now read
  live by this runner. TF-110 keeps the subset craft.
- **D-PROF-2 narrowed**: self-selection on declared
  hardware is proven for all six profiles. The floor
  mismatches on *other* machines (a 2 GB Pi handed the
  pi_5 profile, and kin) stay with TF-086.
- **D-PROF-5 added + OPEN** (61 entries — 39 OPEN /
  1 OPEN-ENVIRONMENT / 19 FIXED / 2 CLOSED-BY-VERDICT):
  profiles promising models their own caps/ceilings
  exclude, as measured above; TF-086 owns the verdicts.

## Proof grade

FIXTURE/SIMULATION — real core and engine boots in temp
dirs on this VM, simulated hardware from the profiles'
own declarations. A boot on real phone/Pi/server
hardware is **NOT PROVEN** (TF-082 owns the real-Pi
road), and a loaded model's real footprint is **NOT
PROVEN** — the accounting is catalogue-calibrated, not
weighed. Never claimed.
