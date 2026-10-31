# TF-075 — Aesir + TurboQuant Interplay

**Quantised models served by Aesir, measured through
ThoughtForge's fixtures: the curve through Aesir equals
the local fixture curve exactly, and the quant policy
check now judges served residents too.**
`src/thoughtforge/inference/aesir_quant.py`
(`INTERPLAY_RULES` / `RESIDENT_MODEL_IDS` /
`FakeAesirQuantServer` / `run_aesir_quant_panel`, format
`thoughtforge-aesir-quant-interplay/1`). Tests:
`tests/test_tf075_aesir_turboquant.py`.

## Measured start (probed this slice)

- TF-074 bound Aesir as a backend, but nothing connected
  it to the quant estate: TF-067's panel ran only against
  its corpus directly, and TF-068's policy check was
  wired only into the *local* engine's load path — the
  served path carried no quant check at all
  (**D-AESIR-2**, fixed here).
- The TF-074 wire payload has **no quant field** (model /
  messages / stream / max_tokens / temperature / seed).
  Quant cannot be sent to Aesir; it can only be the
  resident model Aesir is serving.
- Aesir's own docs name that resident by GGUF filename
  (`gemma-4-E2B-it-Q4_K_M.gguf`, Aesir `docs/GEMMA4_CUDA.md`
  @ `1eed9706`) — so TF-066's filename vocabulary, built
  for local TurboQuant loads, is the same vocabulary
  that proves a served resident's quant.
- Aesir serves **one resident model, serialized**
  (TF-074's ledger): a quant panel through Aesir is a
  serial sweep, one resident at a time.

## The interplay rules (INTERPLAY_RULES)

- **Quant is identity, not a wire field.** A level
  reaches Aesir only as the resident model's id; every
  payload in the sweep is pinned quant-key-free, with
  the level present only in `model`.
- **One resident, so the panel is serial.** The sweep
  runs `QUANT_LEVELS` in order; the fixture server's
  call log proves it: 48 calls, each level's 8
  contiguous, in panel order.
- **Same panel, same scorer, same curve.** The
  measurement is TF-067's runner/prompts/scorer with
  only the generation source changed to a real
  `AesirBackend` over a fake Aesir transport. The
  recomputed curve through Aesir —
  iq2_xs 0.0000, q2_k 0.4375, q4_k_m 0.8125,
  q5_k_m 0.7083, q8_0 1.0000, fp16 1.0000 — equals
  TF-067's published `CURVE` exactly (filed as
  `CURVE_THROUGH_AESIR`, checker-recomputed live):
  the binding transports answers, it does not bend
  scores.
- **Policy check at identity grade (D-AESIR-2, FIXED).**
  TF-068's `check_quant_for_profile` now also judges
  served residents via `resident_verdict`, and
  `profile_residents()` files the expected resident
  for all six profiles — verdict `match`, with
  TF-068's curve class and decision carried verbatim.
  The grade is honestly weaker than the local check:
  locally the engine loads the very file whose name it
  checks; served, ThoughtForge sees only the identity
  string Aesir reports — a misnamed file on the server
  would pass this check. Identity grade, never load
  grade.
- **Unprovable resident, never assumed.** A tag-shaped
  id (`gemma4-e2b:latest`, TF-074's catalog example)
  proves no quant: verdict `unprovable` /
  `filename_proves_no_quant` — never a match, never a
  default assumed.
- **Serving does not change the curve class.**
  phone_low's and pi_zero's q2_k residents remain
  constrained exceptions (steep_loss, fixture loss
  0.5625): Aesir buys distribution, not quality.
- **Error is panel failure, not a zero.** A 404
  (model-not-resident) answer mid-sweep raises naming
  the resident and the server's error; a partial sweep
  publishes no curve and scores no false zero.
- **Usage counts come from the server.** Per level,
  `tokens_generated` is Aesir's reported
  `usage.completion_tokens` (TF-074's rule, exercised
  here; the fixture server reports the contract
  estimate of each answer, so the counts check).

The fixture residents are synthetic, in Aesir's own
GGUF-filename shape over TF-017's Lantern Hollow world
(`lantern-hollow-<QUANT>.gguf`), one per panel level,
each round-tripping through `quant_of_filename` to
exactly its level.

## Docket

- **D-AESIR-2 filed + FIXED by TF-075** — the served
  path bypassed TF-068's quant policy entirely (the
  check was wired only into `TurboQuantEngine.load`;
  `AesirBackend` served any resident unchecked). The
  fix is the interplay module above: `resident_verdict`
  + `profile_residents` + the sweep's own checks, at
  identity grade, stated. TF-016 totals: 71 entries —
  41 OPEN / 1 OPEN-ENVIRONMENT / 27 FIXED /
  2 CLOSED-BY-VERDICT (primary-owner Part IV 20 → 21).
- **D-AESIR-1 stands OPEN, owned by TF-086** — no live
  Aesir serve has carried this panel; the live smoke
  (a real Aesir serve on the laptop line, residents at
  more than one quant) remains owed.

## Proof grade and limits

FIXTURE proof, on this VM: TF-067's recorded corpus
served by a fake Aesir transport in Aesir's v1 shapes,
through the real `AesirBackend` and TF-067's real
runner/scorer; TF-068's real verdict type; live
checkers. **NOT PROVEN**: a live Aesir serve of
quantised models (D-AESIR-1); any real model's
quantisation loss (TF-127's live panel owns it); the
resident identity's truth against the file actually
loaded on the server (identity grade only, above).
