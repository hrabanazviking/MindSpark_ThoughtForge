# TF-068 — Quant Selection Policy (ADR)

**The plain truth first: a default nobody can explain is a
guess, and a recommendation nobody reads is not a policy.**
The hardware profiles shipped quant defaults with no stated
basis, and TF-066 proved the engine never read them at all —
any GGUF loaded on any profile, silently (D-QUANT-1). This
slice records the decision properly, as an ADR: each
profile's default re-derived from TF-067's measured curve,
ratified or excepted with its reason stated, and the
decision wired into the engine's load path so it can never
again pass silently. The estate lives in
`src/thoughtforge/inference/quant_policy.py` (format
`thoughtforge-quant-selection-policy/1`).

## Status

Accepted (TF-068). Supersedes: nothing — the profile
values were unreasoned recommendations. Re-review triggers
are named per row below; TF-127's live panel can reopen
any row.

## Context

TF-067's published curve (fixture grade — a real
measurement of the fixture corpus through the real scorer,
not of any live model):

| Quant | Mean | Loss vs fp16 | Class (this ADR) |
|---|---|---|---|
| iq2_xs | 0.0000 | 1.0000 | catastrophic |
| q2_k | 0.4375 | 0.5625 | steep_loss |
| q4_k_m | 0.8125 | 0.1875 | working_band |
| q5_k_m | 0.7083 | 0.2917 | working_band |
| q8_0 | 1.0000 | 0.0000 | parity |
| fp16 | 1.0000 | 0.0000 | parity (the reference) |

Classes are computed from the curve's own numbers (mean
0.0 → catastrophic; loss 0.0 → parity; mean ≥ 0.70 →
working band; else steep loss) and the checker recomputes
them. Two honesty limits bind every decision below:

- **The curve is not monotone.** q4_k_m out-scores q5_k_m
  on the eight-prompt corpus (TF-067's filed
  `NON_MONOTONE_STEPS`). Eight prompts show the curve's
  *shape* — catastrophic at the bottom, steep at q2_k, a
  working band from q4_k_m up, parity at the top — they do
  not rank neighbouring levels.
- **No fit measurement exists in-tree.** No
  bits-per-weight or file-size-per-quant figure is stated
  anywhere (TF-066), so no default is raised on a guess
  that a higher quant fits a constrained profile. Fit
  proof belongs to TF-069 (budget-enforced profile
  runner), TF-082 (real Pi), and TF-127 (live panel).

## Decision — the rules

1. **quality_floor** — a default sits in the working band
   or at parity, unless its row is a recorded constrained
   exception.
2. **never_catastrophic** — iq2_xs (mean 0.0,
   contradictions on 7 of 8 prompts) is never a default,
   at any size pressure.
3. **neighbours_not_ranked** — inside the working band
   this corpus ranks nobody; q5_k_m is not promoted over
   a shipped q4_k_m default on this evidence.
4. **parity_needs_no_fp16** — where a level measures at
   parity with fp16, the curve gives no quality reason to
   prefer fp16 over it.
5. **constrained_exception** — a below-floor default is a
   size-class exception, never a quality claim: the row
   states the measured fixture cost and the trigger that
   forces re-review; a trigger proving a floor-level
   quant fits forces the default up.
6. **announced_not_silent** — enforcement is a recorded,
   announced verdict at load time, never silence and
   never a hard refusal (below).

## Decision — per profile

| Profile | Default | Decision | Basis |
|---|---|---|---|
| phone_low | q2_k | keep — **constrained exception** | Below the floor (steep_loss, fixture loss 0.5625) — kept only for the ~2 GB / 1.1B class the profile declares. Trigger: TF-069 budgets + TF-127 panel |
| pi_zero | q2_k | keep — **constrained exception** | Below the floor (steep_loss, fixture loss 0.5625) — kept only for the 0.5 GB / 0.5B knowledge-subset class. Trigger: TF-082 real-Pi proof + TF-127 panel |
| pi_5 | q4_k_m | keep | The working band's floor; q5_k_m not promoted (rule 3) |
| desktop_cpu | q4_k_m | keep | The working band's floor; q5_k_m not promoted (rule 3) |
| desktop_gpu | q8_0 | keep | Parity with fp16 on the curve (rule 4) |
| server_gpu | fp16 | keep | The reference level itself; no size pressure buys anything by compressing |

No default changes this slice — because the curve gives
no *proven* basis to change one. The four at/above the
floor are kept on evidence; the two below it are kept
only as named exceptions whose cost (0.5625 fixture loss
each) is now on the record instead of invisible.

## Enforcement (D-QUANT-1, FIXED)

`TurboQuantEngine.load` runs
`quant_policy.check_quant_for_profile(profile, filename)`
on every load and records the verdict on the engine
(`engine.quant_verdict`):

- **match** — the filename proves the policy default.
- **mismatch** — the filename proves a different quant:
  recorded *and* announced (warning log). The load still
  proceeds — the operator may choose another quant
  deliberately, but never again unknowingly.
- **unprovable** — the filename proves no quant, or only
  TF-066's generic class (`-q4`/`-q5`,
  `filename_under_specifies_variant`): recorded as
  unprovable, never rounded up to the default and never
  misreported as a mismatch.
- **no_model** — the engine loaded without a model path.

The check reads the *filename* (TF-066's
`quant_of_filename`), not the file's header — header
inspection remains a does-not in TF-066's duties table,
whose `enforce_profile_quant` row this slice moves to
*does*.

## Consequences

- D-QUANT-1 closes. TF-016 totals: 60 entries — 38 OPEN /
  1 OPEN-ENVIRONMENT / 19 FIXED / 2 CLOSED-BY-VERDICT.
- The policy's open debt is exactly the two constrained
  exceptions, owned by TF-069 / TF-082 / TF-127 as named.
- Proof grade stays **FIXTURE**, TF-067's own: this ADR
  reasons from a fixture curve's shape. TF-127's live
  panel is the road that can overturn any row — and when
  it banks, this ADR is re-read against it.
