# TF-086 — Part IV Gate: TurboQuant & the Inference Layer, with Truth Table v2 (Part IV Gate)

Closed 2026-10-31 on branch `yrsa/thoughtforge-road-build`.
Roadmap wording: "Part IV gate: quant curve published, Aesir
backend green under contract tests, per-profile truth table
v2 issued."

## The gate verdict

**Part IV is COMPLETE. All 22 slices (TF-065–TF-085) are
banked with evidence in `docs/architecture/`, and the three
gate proofs were re-measured fresh at this gate — not quoted
from their slices:**

1. **Quant curve published — GREEN.** TF-067's curve stands
   as filed in `inference/quant_quality.py` and its checker
   returns `[]` at this gate, recomputing the curve from the
   corpus: iq2_xs **0.0000** (loss 1.0000, catastrophic),
   q2_k **0.4375** (loss 0.5625, steep), q4_k_m **0.8125**
   (loss 0.1875), q5_k_m **0.7083** (loss 0.2917) — the
   filed non-monotone step, q4_k_m above q5_k_m on this
   eight-prompt corpus, stands unsmoothed — q8_0 **1.0000**
   and fp16 **1.0000** (parity on this corpus). The curve
   is FIXTURE grade: a real measurement of the fixture
   corpus through the real scorer, not of any live model.
   TF-068's policy, chosen from this curve, also re-checks
   clean at the gate through TF-075's served-path estate
   (`aesir_quant_problems() == []`): every profile's
   expected Aesir resident verdicts match, at identity
   grade.
2. **Aesir backend green under contract tests — GREEN at
   fixture grade.** `backend_contract_problems()` over the
   real `AesirBackend` class returns `[]` at this gate —
   the binding conforms to TF-065's contract v1 by the
   contract's own checker, exactly as TF-074 filed it. The
   targeted gate test set was re-run as one set at this
   gate: `test_tf065` + `test_tf067` + `test_tf074` +
   `test_tf075` + `test_tf069` = **133 tests, 133 passed**.
   The honest limit stands in the table below and in
   ruling 3: the backend has never faced a live Aesir
   serve (D-AESIR-1) — "green" here means contract-green
   against mocked HTTP legs in Aesir's own v1 shapes,
   never live-proven.
3. **Per-profile truth table v2 — ISSUED below (§2).**
   It supersedes TF-012's v1: where v1 found profiles
   governing one engine and nothing else, v2 records a
   tree where the selected profile now governs response
   caps (TF-032), latency budgets (TF-033), refinement
   passes (TF-037), memory + retrieval caps (TF-056),
   simulated boots with a hard RAM ceiling (TF-069),
   cold-start budgets (TF-079), a memory-pressure
   governor (TF-080), measured concurrency shapes
   (TF-081), a real-Pi execution (TF-082), and GPU-optional
   proof on two hosts (TF-083). What still does not hold
   is named row by row — v2's standard is v1's: a budget
   is "true" only when a named line of code enforces it,
   and a capability is "proven" only at the grade its
   evidence earned.

## 2. Hardware-profile truth table v2

Declared = the shipped JSON's claim. Selector floor = the
operative floor in `profiles.py` (TF-012's measurements,
re-stated — TF-069 proved only self-selection on each
profile's own declared hardware, 6/6). Cap fit = TF-069's
accounting (catalogue-calibrated GB-per-B; fp16 definitional
2.0; q2_k uncalibrated in-tree). Cold TTFT = TF-079's
medians of 3 simulated boots on this VM (fixture first
token — shape only, no model computes). Batch = TF-081's
parallel-cores speedup on the fixture panel (medians of 3).

| Profile | Declared (RAM+VRAM, default quant, ctx, cap) | Selector's operative floor | Boot verdict (TF-069) | Cap fit at default | Cold TTFT (ms) | Batch speedup | Proof grade / hardware truth |
|---|---|---|---|---|---|---|---|
| `pi_zero` | 0.5+0 GB, q2_k, 256, 0.5B | ARM/Pi RAM <2 GB, or mobile RAM <1 GB | boots_with_ceiling_gaps | **unprovable** (q2_k uncalibrated) | 199.084 | 5.353× | Simulation + profile executed on a real Pi 5 (TF-082); **Pi Zero hardware itself NOT PROVEN** (none in the house); live model NOT PROVEN |
| `phone_low` | 2+0 GB, q2_k, 512, 1.1B | Mobile RAM ≥1 GB; also the x86 fallback under 6 GB | boots_with_ceiling_gaps | **unprovable** (q2_k uncalibrated) | 100.211 | 4.355× | Simulation on this VM only; **real phone NOT PROVEN** (TF-173 owns); live model NOT PROVEN |
| `pi_5` | 4+0 GB, q4_k_m, 2048, 3B | Pi/ARM-non-mobile RAM ≥2 GB | boots_with_ceiling_gaps | fit (2.0 GB accounted / 4 GB) | 101.556 | 2.570× | **REAL HARDWARE, fixture legs** — boots, detector, offline embedding encode, cold + batch panels executed on a real Raspberry Pi 5 (TF-082); live-model generation on the Pi NOT PROVEN (llama_cpp absent, no wheel served — TF-174 owns) |
| `desktop_cpu` | 8+0 GB, q4_k_m, 4096, 7B | x86 RAM ≥6 GB (this VM, 7.7 GB, auto-selects it) | boots_with_ceiling_gaps | fit (4.67 / 8 GB) | 181.860 | 3.536× | Fixture/simulation on this GPU-less VM; the one profile whose cold-start + latency budgets are wall-clock **enforced** in tests (TF-033/TF-079); live model NOT PROVEN |
| `desktop_gpu` | 16+12 GB, q8_0, 8192, 13B | VRAM ≥6 GB + any GPU backend | **boots** (the only gap-free boot) | fit (14.3 / 28 GB) | 158.732 | 2.375× | Simulation here (its as-declared load selects **vulkan by library presence** on this GPU-less VM — ruling 4); laptop line (RTX 2060) binds cuda for its declared load (TF-083); live model NOT PROVEN |
| `server_gpu` | 64+24 GB, fp16, 32768, 70B | VRAM ≥24 + CUDA/ROCm; **RAM never checked** | boots_with_ceiling_gaps | **exceeds_ceiling at its default** (70B fp16 accounts 140.0 GB vs 88 GB declared) — ruling 1 | 95.231 | 4.629× | Simulation only; no server hardware in the house has run it; live model NOT PROVEN |

Per-profile rows the table cannot hold, each filed by its
owning slice and carried here as v2 truth:

- **Embedding assignment (TF-053/TF-070):** pi_zero is
  assigned **no embedding runtime** (SQL-only by rule);
  phone_low is assigned the ONNX runtime; the other four
  carry sentence-transformers assignments. On this VM the
  package is absent (D-ENV-1), so VM-side availability is
  honestly unavailable; on the real Pi 5, pi_5 measured
  **available_offline** with a real encode (TF-082).
- **ONNX flag (D-ONNX-2) — ruling 5 below:** pi_zero's
  `deployment.onnx_export: true` stands against its `none`
  assignment: **flag_without_assignment** (TF-070's filed
  verdict). phone_low's identical flag is consistent
  (true + onnx runtime).
- **Concurrency shape (D-BATCH-2):** no profile declares a
  concurrency, batch, or thread parameter; `n_threads` is
  auto (None) everywhere. The per-profile shape v2 carries
  is TF-081's measured one: the backend contract serves
  one request per call, drafts run as a sequential loop at
  the profile's `draft_count` (pi_zero 1, phone_low 2,
  pi_5/desktop_cpu/desktop_gpu 3, server_gpu 5), parallel
  cores overlap fully (max in flight 4 of 4 on all six),
  one shared core never overlaps (TF-035's turn lock,
  re-proven at batch scale). A deployment's declared
  concurrency remains TF-153's service-shape decision.
- **Memory pressure (TF-080):** the governor's ladder
  (context first, six steps, prompt/model/protected
  records never shed, refusal by name) is filed per
  profile with its maximum load at gauge exactly 1.0 by
  construction — fixture grade; a loaded model's real
  footprint under genuine pressure is NOT PROVEN on every
  row.
- **Model cards (TF-078):** every card's
  `profiles_proven` is **empty** and `live_proof` is
  not_proven — no catalogue model has ever loaded here
  (no GGUF in-tree, llama_cpp absent). "Supported" in v2
  means carded in `inference/model_cards.py` with declared
  size + filename-proven-or-generic quant; it does not
  mean run.

## 3. Gate rulings on the entries assigned to this gate

TF-016's docket sent seven entries to this gate. Each is
ruled here on its filed evidence; the docket is updated
in this same slice.

1. **D-PROF-5 (profiles promise models their own caps
   exclude) — ruled, entry stays OPEN at its remaining
   owners.** The three mechanical fixes the entry offered
   (raise declared memory, lower caps, change defaults)
   are each rejected on the evidence: the declared memory
   figures are the machine *classes* the profiles name
   (no measurement says the classes are bigger); TF-068
   ratified server_gpu's fp16 default on the curve (the
   reference level; no size pressure buys anything by
   compressing); and the 70B cap is *true at q4_k_m*
   (46.7 GB accounted, fits 88 GB) — the falsehood was
   never any single number, it was reading cap and
   default together as a fit promise. **v2's rule, which
   governs every claim from this gate forward: a profile's
   `max_params_b` is a parameter-count cap, never a fit
   promise; fit is judged per (params, quant) pair by
   TF-069's `check_model_fit`, which refuses
   exceeds_ceiling and unprovable alike.** Under that
   rule the rows stand exactly as the table files them:
   server_gpu exceeds at its fp16 default (a 44B fp16
   model is its accounted ceiling at 88 GB / 2.0 GB-per-B);
   pi_zero's TinyLlama-1.1B and desktop_cpu's LLaMA-3-8B
   recommendations **exceed their own caps and are not
   supported on those profiles**, whatever the JSONs list
   (TF-069's runner fails those legs by name on every
   run); phone_low/pi_zero caps stay UNPROVABLE until a
   q2_k size figure exists. Remaining owners, unchanged:
   **TF-127** (the live panel may calibrate q2_k and weigh
   real fit) and **TF-185** (the release truth table must
   carry these rows verbatim). Editing the shipped JSONs
   to hide the rows is expressly not the fix — the runner
   recomputes them live, and a JSON edit would falsify
   TF-069/TF-078's pinned evidence instead of the claim.
2. **D-PROF-2 (selector floors ≠ claimed floors) — v2
   share discharged, entry stays OPEN.** The table above
   publishes both floor sets side by side, which is the
   truth-table act the entry assigned here. The operative
   truth for any consumer: **the selector's floors are
   the floors**; the JSON descriptions are class
   descriptions, not thresholds. The sub-items (selector
   docstring order, the 4 GB failed-detection assumption,
   the resolver-only `n_gpu_layers` ladder — now clamped
   honestly by TF-083's D-GPU-1 fix) remain as filed for
   TF-192's final audit; no code changes at this gate.
3. **D-AESIR-1 (Aesir backend never faced a live serve) —
   carried as NOT PROVEN, owner moved.** v2 and every
   Part IV artefact state it: the binding is
   contract-green at fixture grade only. The owed smoke
   (one real Aesir serve + one ThoughtForge turn through
   `AesirBackend`) is a laptop-line leg; it moves to
   **TF-175** (the laptop line, Part X), which runs where
   a real Aesir serve can stand. Until it banks, no
   consumer may cite Part IV as live Aesir proof.
4. **D-GPU-2 (vulkan probers accept a library as a
   device) — carried at library grade, owner moved.**
   v2's rule: a vulkan "available" report anywhere in
   Part IV's estate means **libvulkan present**, never
   device-proven — on this VM the library exists,
   `vulkaninfo` and any GPU device do not, and
   desktop_gpu's as-declared load selecting vulkan is
   filed as exactly that artefact (TF-083). The prober
   code is not changed at this gate: proving a stricter
   probe needs a host with a real device to test both
   directions, which is the laptop line — owner moves to
   **TF-175**, TF-083's two-host evidence as its record.
5. **D-ONNX-2 (pi_zero's onnx_export flag) — ruled: the
   flag is false for pi_zero, v2 is the correction of
   record.** TF-070's verdict stands as filed
   (flag_without_assignment; pi_zero's vector leg is
   SQL-only by TF-053's rule). No consumer may cite
   pi_zero's JSON flag as ONNX support; v2's row governs.
   The JSON correction itself joins the release-shaping
   pass — owner moves to **TF-185**, which owns the
   shipped files' final form — because the elder phase-5
   test estate pins the shipped flag and a one-line JSON
   edit at a gate would break pinned evidence in four
   suites without changing any behaviour. Entry stays
   OPEN until TF-185 banks the correction.
6. **D-CARD-2 ("recommended" vs "supported" name
   different sets) — CLOSED-BY-VERDICT.** The ambiguity,
   not either list, was the defect. Ruled: the two words
   name different sets *by definition* from this gate —
   **recommended** = a profile JSON's suggestion list
   (6 carded / 5 family-only / 4 with no card at all, per
   TF-078's live audit: Granite-1B, Gemma-2B, LLaMA-3-13B,
   Mixtral-8x7B carry **no size truth anywhere in-tree**
   and may never be cited as supported); **supported** =
   carded in `inference/model_cards.py`, with declared
   size and filename-proven-or-generic quant, and with
   `profiles_proven` empty until a live run proves one.
   Neither list changes; TF-136/TF-183 own the root
   MODEL_CARD.md drift (D-CARD-1), which is a separate
   entry and stays OPEN at them.

7. **D-OAI-4 (OpenAI-compatible backend never faced a
   real third-party server) — carried as NOT PROVEN,
   owner moved.** The fixture transports and the stdlib
   loopback round trip exercised the wire in the shape's
   own forms; no vendor dialect (LM Studio, vLLM,
   LocalAI, llama.cpp server, hosted API) has faced the
   backend. A real server can stand on the laptop line —
   owner moves to **TF-175**, beside rulings 3 and 4.

**Docket (TF-016) at the gate: 86 entries — 44 OPEN /
1 OPEN-ENVIRONMENT / 38 FIXED / 3 CLOSED-BY-VERDICT**
(D-CARD-2 closed by this gate's verdict; primary-owner
Part IV count unchanged at 35). Every other Part IV entry
assigned forward keeps its owner: D-BATCH-2 → TF-153
(v2 carries its shape, §2), D-COLD-1/D-PRESS-1/D-BATCH-1's
live shares → TF-082 (discharged at fixture grade for the
Pi) / TF-175, D-DEP-3 → TF-069/TF-110 as filed, D-CFG-3's
remainder → TF-153, D-ORPHAN-3's remainders → TF-110 /
TF-167 / TF-179. TF-192 verifies at the end that no entry
is left open without a reason.

## Plain-language summary (the Volmarr-facing half of this gate)

Here is what Part IV did, in plain words:

- **The engine room now has one contract, and every
  engine obeys it.** At the start of Part IV, the ways
  ThoughtForge talked to models were a pile of separate
  habits: health checks that called a missing model
  "healthy", a token counter that was really counting
  words, error messages that vanished on the streamed
  path. All of that is now one written contract with a
  machine checker, and every backend — the local engine,
  Ollama, LM Studio, the OpenAI-shaped binding, and
  **Aesir** — is measured against it and passes.
- **Aesir is a first-class backend, honestly graded.**
  Project Aesir binds under the same contract, its
  quantised residents get the same policy verdicts as
  local loads, and its answers flow through the same
  traces. The plain limit: it has only ever faced a
  test-double speaking Aesir's shapes, never a live
  Aesir server — that smoke test is now owned by the
  laptop-line work in Part X, and nothing in Part IV
  claims otherwise.
- **Compression is now measured, not marketed.** The
  part built the first quant curve in the tree that was
  actually computed: the deepest compression levels lose
  badly (the smallest setting scored zero — it contradicted
  the facts on seven of eight questions), the middle band
  works, and the top two settings tie. Each hardware
  profile's default compression was then chosen *from
  that curve*, with the two smallest profiles kept on
  the harsh setting only as named exceptions whose cost
  is written down.
- **Hardware profiles finally tell the truth — in the
  new truth table (v2, §2 above).** A profile now really
  does govern memory, speed budgets, repair passes, and
  what happens when memory runs out (shed context first,
  refuse by name rather than crash). The table also
  names what is still not true: the biggest profile
  promises a model size its own memory cannot hold at
  its default setting, two profiles recommend models
  bigger than their own caps, the two smallest profiles'
  fit cannot be proven at all yet, and one tiny profile
  claims an export feature it has no machinery for.
  Those rows are published, not hidden, and each has a
  named owner.
- **Real hardware touched the work exactly once, and it
  mattered.** The whole stack ran on a real Raspberry
  Pi 5 — and immediately found a real bug no desktop
  test could see (every ARM machine was being labelled
  a phone). It is fixed. A live model generating on the
  Pi, on a phone, or anywhere else is still **NOT
  PROVEN**: there is no model file and no model loader
  on any house machine yet, and Part VII's live panels
  own that road.

## The evidence, slice by slice

| Slice | Artefact | Headline |
|---|---|---|
| TF-065 | `BACKEND_CONTRACT.md` | Contract v1: duties, capability vocabularies, dependency rulings; D-DEP-1/D-DEP-5 fixed |
| TF-066 | `TURBOQUANT_MAPPED.md` | TurboQuant's method mapped from code; filename quant vocabulary proven; D-QUANT-2 fixed |
| TF-067 | `QUANT_QUALITY_FIXTURES.md` | The published curve (gate proof 1, re-measured here) |
| TF-068 | `QUANT_SELECTION_POLICY.md` | Per-profile defaults chosen from the curve as an ADR; D-QUANT-1 fixed (load-time verdicts) |
| TF-069 | `PROFILE_RUNNER.md` | Every profile boots in simulation under a hard RAM ceiling; D-PROF-5 measured |
| TF-070 | `ONNX_EXPORT_PATH.md` | ONNX exports embeddings only; D-ONNX-1 fixed; D-ONNX-2 ruled at this gate |
| TF-071 | `TOKENIZER_DISCIPLINE.md` | One counting discipline, script-aware; D-TOK-1/2/3 fixed |
| TF-072 | `STREAMING_BACKPRESSURE.md` | Streams close what they open, on every exit; D-STREAM-1 fixed |
| TF-073 | `BACKEND_HEALTH_AND_FALLBACK.md` | Health means 200; fallback chain with circuit breakers; D-HEALTH-1, D-ORPHAN-2 fixed |
| TF-074 | `AESIR_BACKEND.md` | Aesir bound under the contract (gate proof 2); live serve NOT PROVEN (ruling 3) |
| TF-075 | `AESIR_TURBOQUANT_INTERPLAY.md` | Served residents get TF-068 verdicts at identity grade; D-AESIR-2 fixed; the curve through Aesir equals TF-067's exactly |
| TF-076 | `OPENAI_COMPATIBLE_BACKEND.md` | The common shape bound as its own backend, one shared parser; D-OAI-1/2/3 fixed; real server NOT PROVEN (D-OAI-4 → TF-175's line in Part X evidence) |
| TF-077 | `LOCAL_FIRST_ROUTING.md` | Local-first routing table published; remote opt-in only; D-HEALTH-2 fixed |
| TF-078 | `MODEL_CARDS.md` | Per-model cards from measured truth; profiles_proven empty on every card; D-CARD-2 closed at this gate |
| TF-079 | `COLD_START_BUDGETS.md` | Cold start in four stages, budgets per profile, VM profile enforced; D-COLD-1 fixed at fixture grade |
| TF-080 | `MEMORY_PRESSURE.md` | The pressure governor: shed by rule, refuse by name; D-PRESS-1 fixed at fixture grade |
| TF-081 | `BATCH_CONCURRENCY.md` | Concurrency shapes measured per profile; D-BATCH-1 fixed at fixture grade; D-BATCH-2 → TF-153 |
| TF-082 | `PI_PROFILE.md` | The stack on a real Pi 5; D-DET-1 found on-device + fixed; Pi shares of D-COLD-1/D-BATCH-1 discharged at fixture grade |
| TF-083 | `GPU_OPTIONAL.md` | No profile secretly requires a GPU, proven on two hosts; D-GPU-1 fixed; D-GPU-2 ruled at this gate |
| TF-084 | `JOINED_TRACES.md` | Inference traces join cognition traces — one exportable story per turn; D-HON-1 fixed; D-TRACE-1 → TF-097 |
| TF-085 | `BACKEND_GUIDE.md` | The contract as a 14-step recipe + worked example; D-GUIDE-1 fixed at guide grade |

## Road re-cut assessment

Part IV changed no Part V–X slice's wording or order.
Three assignments move by this gate's rulings, recorded
here per TF-018's rule: D-AESIR-1's live smoke TF-086 →
**TF-175**; D-GPU-2's device-grade probe proof TF-086 →
**TF-175**; D-OAI-4's real-server proof TF-086 →
**TF-175**; D-ONNX-2's JSON correction TF-086 →
**TF-185**. Part V inherits a backend estate whose
contract, traces (TF-084 — its TF-097 rescue-record leg is
already roaded), and enforcement surfaces are frozen as
data.

## Suite at the gate (measured this slice, TF-014 method)

`PYTHONPATH=src python3 -m pytest tests/ -q` on this VM:
**2370 collected / 2368 passed / 1 skipped / 1 failed
(D-ENV-1, unchanged — the phase8 dependency check naming
sentence-transformers absent on this VM, an environment
entry not a code defect)** — exactly TF-085's banking
line. The gate's targeted re-run (TF-065 + TF-067 +
TF-074 + TF-075 + TF-069) is 133 / 133 passed, and the
three gate proofs above were re-measured by direct runs
of `quant_quality_problems()`, `aesir_quant_problems()`,
`backend_contract_problems([AesirBackend()])`, and
`profile_runner_problems()` at this gate, all returning
`[]`. This slice adds one gate doc plus the TF-016
docket rulings and no code or tests, so the count does
not move at banking.

## What opens next

Part V — Refinement: Salvage & Enforcement (TF-087–
TF-104), starting with TF-087: salvage mapped —
salvage.py's strategies catalogued, each with fixtures
of broken input.
