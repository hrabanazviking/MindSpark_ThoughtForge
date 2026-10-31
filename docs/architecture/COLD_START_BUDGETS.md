# TF-079 — Cold-Start Budgets

*Time-to-first-token per profile measured + capped in
tests where the VM allows. The estate lives in
`src/thoughtforge/inference/cold_start_budgets.py`
(`COLD_START_RULES` / `MEASURED_ROWS` as data,
`budget_for_profile()` / `measure_cold_start()` /
`check_cold_start()`, format
`thoughtforge-cold-start-budgets/1`).*

## The plain truth first

No cold-start or time-to-first-token measurement
existed anywhere in this tree. Warm turns were
budgeted (TF-033); profiles were booted (TF-069); no
surface timed a boot, a load, or a first token. And
no real model exists on this VM — no GGUF in-tree,
`llama_cpp` absent — so a live model's TTFT is
**NOT PROVEN** for every profile, stated here first
rather than last. What this slice proves is the
*stack's* cold start, in simulation, through a
fixture backend, with budgets a slow start is
provably caught against.

## Measured start

Probed live this tick:

- Grep for cold / ttft / first-token over src, tests
  and docs finds **no measurement surface** — only
  unrelated uses of "cold" (scaffold temperatures,
  cold thread state in the specs). D-COLD-1 (TF-016).
- **TF-033 covers warm turns only, and says so.** Its
  budgets judge the stages of a turn already running
  on a constructed core; its docstring defers
  real-model latency to "TF-079/TF-175's road", and
  its one cold observation (a 51 ms first turn
  against ~1 ms warm turns) is a note, not a budget.
- **TF-069 boots every profile but records no
  timing** — nine legs per profile, verdicts filed,
  no duration anywhere.
- **The engine load was unmeasured** —
  `TurboQuantEngine.load` resolves, selects and
  constructs with no timing; the backends'
  `latency_ms` times generation calls on a loaded
  backend, never a load.

## What cold start means here

Four stages, in boot order — TTFT is their sum, by
construction, never an independent number:

1. **profile_load** — the shipped profile JSON loads
   through the tree's own loader.
2. **core_construction** — a real `ThoughtForgeCore`
   constructed on that profile in temp dirs (TF-069's
   simulation shape: hardware from the profile JSON,
   not host detection).
3. **engine_load** — a real **no-model**
   `TurboQuantEngine` load, recording TF-068's
   `no_model` quant verdict. No model file loads:
   none exists here to load.
4. **first_token** — generation start to the first
   yielded chunk of a streaming generation through a
   fixture backend (three chunks, so the first chunk
   is distinct from the whole).

Measured beside the stages on the same run, never
summed into TTFT (each starts its own clock, so
summing would double count):

- **generation_complete_ms** — generation start to
  stream exhaustion. Distinct from first_token for a
  multi-chunk (native-shaped) stream; for a
  *simulated* backend (TF-072: ollama / lmstudio /
  huggingface yield one chunk that is a completed
  `generate()`) the first chunk **is** the complete
  generation and the two are equal by construction —
  so no separate streaming-TTFT gap exists to file:
  the shape difference is measured and stated, and
  TF-084 (inference traces) / TF-175 (the laptop
  panel) own the roads that consume it.
- **first_answer_ms** — one non-streaming
  `generate()` through the same fixture: the first
  complete answer, as opposed to its first chunk.

## What the VM measurements actually showed

Medians of 3 simulated boots per profile, measured
live this tick on this VM by the slice's own
`median_cold_start` (filed as `MEASURED_ROWS`,
re-judged for shape and within-budget truth by the
checker; a re-measurement is never demanded to
reproduce a filed millisecond — this VM is shared):

| profile | profile load | core construction | engine load | first token | generation complete | first answer | **TTFT** |
|---|---|---|---|---|---|---|---|
| desktop_cpu (VM) | 3.311 | 91.026 | 87.516 | 0.007 | 0.010 | 0.012 | **181.860** |
| desktop_gpu | 2.846 | 73.278 | 82.603 | 0.005 | 0.008 | 0.012 | **158.732** |
| phone_low | 0.270 | 29.883 | 70.053 | 0.005 | 0.008 | 0.011 | **100.211** |
| pi_5 | 0.294 | 11.470 | 89.787 | 0.005 | 0.008 | 0.012 | **101.556** |
| pi_zero | 0.234 | 80.989 | 117.855 | 0.006 | 0.010 | 0.014 | **199.084** |
| server_gpu | 0.729 | 10.160 | 84.338 | 0.004 | 0.008 | 0.011 | **95.231** |

*(all values ms, medians of 3)*

The reading, plainly: **engine load and core
construction dominate** (~10–118 ms each, varying
with VM contention); profile load is sub-millisecond
to a few ms; the fixture's first token is ~0.005 ms
because no model computes anything — it proves the
measurement shape, not a model's speed. These are
VM timings of *simulated* boots for every profile:
the non-VM rows say how this stack boots under those
profiles' shapes on this host, not how a phone or a
Pi boots.

## The budget table, and where every number comes from

| profile | profile load | core construction | engine load | first token | **total (TTFT)** |
|---|---|---|---|---|---|
| server_gpu | 25 | 250 | 500 | 500 | **1,275** |
| desktop_gpu | 38 | 375 | 750 | 750 | **1,913** |
| desktop_cpu (VM) | 50 | 500 | 1,000 | 1,000 | **2,550** |
| pi_5 | 100 | 1,000 | 2,000 | 2,000 | **5,100** |
| phone_low | 150 | 1,500 | 3,000 | 3,000 | **7,650** |
| pi_zero | 250 | 2,500 | 5,000 | 5,000 | **12,750** |

*(all values ms)*

- **Scaling reuses TF-033's `PROFILE_FACTORS`**
  imported live (server_gpu ×0.5, desktop_gpu ×0.75,
  desktop_cpu ×1, pi_5 ×2, phone_low ×3, pi_zero ×5)
  — no second factor table exists to drift, and the
  checker pins the identity.
- **The desktop_cpu bases are set from the
  measurements above** with roughly an order of
  magnitude of headroom: profile_load 50 ms (medians
  0.234–3.311), core_construction 500 ms (medians
  10.160–91.026), engine_load 1000 ms (medians
  70.053–117.855). Deliberately generous: this VM is
  shared, and walls measured under contention are
  not baselines (TF-033's warning).
- **The first_token base (1000 ms) is campaign-set**
  at TF-033's generation base, stated as such: the
  fixture measures ~0.005 ms, and a fixture-derived
  cap would be a flaky sub-millisecond number
  pretending to be a model TTFT.
- **Total is the stage sum by construction**, for
  budgets and measurements alike.

## Enforcement — where the VM allows

- **The VM profile gets real wall-clock
  enforcement**: `desktop_cpu` (TF-033's
  `VM_PROFILE_ID`, re-verified live as what this
  machine auto-selects — the checker pins it) must
  land inside every stage budget and the TTFT total
  in a live `measure_cold_start`, in tests.
- **The deterministic cap proof needs no wall
  clock**: `scripted_measurement` builds a
  measurement from stated delays — a 60,000 ms first
  token on desktop_cpu is caught with overruns named
  exactly `["first_token", "total"]` and the profile
  carried on the verdict; a fast scripted start
  passes; pi_zero's scaled cap catches a 6,000 ms
  first token against its 5,000 ms stage budget by
  the same name (its 12,750 ms total still holds —
  a stage overrun does not need the total to fail). An
  injectable `FakeClock` closes the loop through the
  real measurement path: a fixture backend scripted
  to delay its first chunk by 1,234 ms measures
  `first_token_ms == 1234.0` exactly (real work reads
  0 against the fake clock — the scripted delay is
  the only time that passes) and fails the verdict.
- **The other five profiles are filed data**, judged
  in simulation/fixture time: their filed medians
  sit inside their scaled budgets, recomputed by the
  checker. Real-hardware enforcement is NOT PROVEN —
  TF-082 owns the real Pi, TF-175 the laptop line,
  and TF-086's truth table must carry both lines.

## Docket

**D-COLD-1 added + FIXED at fixture/simulation
grade**: no cold-start budget and no TTFT measurement
existed anywhere (TF-033's warm budgets explicitly
deferred the road here; TF-069's boots recorded no
timing; the engine load was unmeasured). This slice
is the fix — the estate above, enforced on the VM
profile. The live-model and real-hardware shares are
not closed by it: they stand NOT PROVEN under
TF-082 / TF-175, exactly as the entry states.

## Proof grade

FIXTURE/SIMULATION — real profile loads, core
constructions and no-model engine loads on this VM,
a fixture streaming backend, medians of 3 simulated
boots per profile, a deterministic scripted cap
proof, and real wall-clock enforcement on the VM
profile only. A real model's cold start and TTFT, a
GGUF file load, and every non-VM profile's real
hardware are **NOT PROVEN**, never claimed.
