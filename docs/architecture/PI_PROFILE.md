# TF-082 — The Pi Profile: the Whole Stack on Real Hardware

*The Pi profile, proven on the real Pi or NOT PROVEN
stated plainly. The estate lives in
`src/thoughtforge/inference/pi_profile.py`
(`PI_RULES` as data, the on-device measurements filed
as `HOST_EVIDENCE` / `DETECTOR_ON_PI` / `BOOT_ON_PI` /
`EMBEDDINGS_ON_PI` / `ENCODE_ON_PI` /
`COLD_START_ON_PI` / `BATCH_ON_PI` / `GGUF_ON_PI`,
format `thoughtforge-pi-profile/1`).*

## The plain truth first

Until this slice, every Pi number in the campaign was
a simulation run on an x86_64 VM — TF-069's boots,
TF-079's cold starts, TF-081's batch panel, TF-053's
embedding verdicts — and each of those slices said so
and handed the real-hardware road here. The house has
a real Pi and it was reachable: this slice cloned the
campaign tree onto it (sparse, at the exact head this
slice banks on, `52f7630`) and ran the whole stack
there. **The stack boots and behaves on the real Pi
5, leg for leg as the simulation promised** — and the
hardware immediately repaid the visit by exposing a
real detector defect no VM run could have seen
(D-DET-1, fixed in scope).

What did **not** get proven is stated just as plainly:
no live model generated anywhere in this slice
(`llama_cpp` is absent on the Pi and its index serves
no wheel for it), and no Pi Zero exists in the house —
every pi_zero row is that profile executed on the
Pi 5, never 512 MB hardware.

## The device

| Fact | Measured |
|---|---|
| Host | `purpleviking1` |
| Hardware | Raspberry Pi 5 Model B Rev 1.1 (device-tree and cpuinfo agree) |
| Arch / kernel | aarch64, `6.12.112-v8-16k-hailo+` |
| CPU / RAM | 4 cores, 16 GB (14,394 MB available at probe time) |
| Python | 3.11.2 (system) |
| SoC temperature | 44.65 °C at probe start, 43.55 °C at end |
| Probe peak RSS | 512.1 MB — one pi_zero's entire declared RAM, as colour |

## The defect the hardware found — D-DET-1

`HardwareDetector.detect()` computed `is_mobile`
**before** detecting RAM. `_detect_mobile` reads
`ram_gb` — still the snapshot default `0.0` at that
point — so its `ram < 4.0` clause labelled *every* ARM
machine mobile. Measured on the Pi: a 16 GB machine
came back `is_mobile = True`.

Selection survived by luck of ordering: `_select_id`
checks `is_raspberry_pi` before the mobile branch, so
the Pi still selected pi_5. A non-Pi ARM Linux machine
(an ARM server, a big ARM board) would not have been
so lucky — with `is_mobile` stuck true, the
`(is_arm and not is_mobile)` Pi-class branch never
fires and the machine lands on `phone_low`. On x86 the
bug was invisible (`is_arm` is false, the clause
short-circuits), which is why four parts of VM-side
audits never saw it.

**Fix (in scope):** `detect()` now detects RAM before
the mobile check — an ordering change, no rule
changed. This slice's checker pins the source order,
and its live checker proves the behaviour both ways:
a 16 GB ARM snapshot is not mobile and selects pi_5
(Pi or not); a 2 GB ARM snapshot is still mobile, as
the clause intends.

## The legs, on the device

**Detector.** The tree's own detector identified the
machine — aarch64, `is_raspberry_pi` true (the
cpuinfo `Model` line), 15.83 GB RAM detected — and
the selector chose **pi_5**, the profile this
hardware is for.

**Boots (TF-069's runner, on-device).** Both Pi
profiles booted with verdicts *identical* to the VM
simulation:

- **pi_5** — all nine legs pass (profile, params,
  selector, ram_ceiling, recommended_models,
  deployment, core, memory_budget, engine);
  verdict `boots_with_ceiling_gaps`, the gap being
  the recommendation whose name carries no parameter
  count (TF-069's filed reason), cap row `fit`
  (3B × q4_k_m accounts 2.0 GB against the 4 GB
  declared ceiling), engine host backend honestly
  `cpu`.
- **pi_zero** — verdict `boots_with_ceiling_gaps`
  with the same two failing legs as in simulation:
  `ram_ceiling` (its q2_k cap is unprovable in-tree)
  and `recommended_models` (TinyLlama-1.1B against
  its own 0.5B cap — D-PROF-5's rows, reproduced on
  hardware).

**Embeddings (TF-053's discipline, on-device).** The
VM's verdict for the ST profiles is
`unavailable_package_missing` (D-ENV-1). The Pi's is
the opposite: **pi_5 `available_offline`** — package
importable *and* all-MiniLM-L6-v2 in the local cache.
And the leg genuinely runs there: with the network
logically off (`HF_HUB_OFFLINE=1`), the model loaded
from cache in 645.346 ms and encoded two sentences in
824.75 ms into 384-dim vectors (sentence-transformers
5.4.1). pi_zero stays `not_assigned` on the Pi too —
SQL-only by rule, on any machine. phone_low's ONNX
leg is unavailable for the artefact reason
(onnxruntime *is* importable on the Pi; no exported
artefact exists) — TF-070's road.

**Cold starts (TF-079's medians of 3, on-device).**
pi_5 TTFT 7.043 ms (profile 0.115 / core 3.848 /
engine 3.074 / first token 0.006); pi_zero TTFT
4.667 ms. Fixture grade, as TF-079 defines it — the
first token is a fixture's and the engine carries no
model — and the Pi's walls are far below the
contended VM's filed rows (95–199 ms). Recorded as
measured; the comparison says something about the
VM's contention, not about any model.

**Batch panel (TF-081's panel, one run, pi_5).**
Sequential 204.578 ms, parallel 53.275 ms (max in
flight 4 of 4, speedup 3.84×), same-core 204.037 ms
(max in flight 1, ratio 0.997), drafts 150.329 ms
(max in flight 1). TF-081's shapes — parallel only
across cores, serialised on one core, drafts serial
— reproduce on ARM exactly.

## Not proven on the Pi — the honest remainder

- **Live-model generation: NOT PROVEN.** The Pi
  holds four real GGUFs — TinyLlama-1.1B-Chat at
  Q4_0 (637.7 MB), Q5_K_M (783.0), Q6_K (904.4) and
  Q8_0 (1170.8), inventoried on-device through
  TF-066's own filename parser — but the engine's
  loader is `llama_cpp`, which is absent from the
  Pi's system Python and both house venvs probed,
  and `pip install --only-binary :all:
  llama-cpp-python` on the Pi answers "No matching
  distribution found" (measured this tick): no
  aarch64 wheel is served. A source build, or a
  binary-shaped backend, is TF-174's road; TF-086's
  truth table must carry this line.
- **Pi Zero hardware: NOT PROVEN.** There is no
  512 MB Pi in the house. The pi_zero profile runs
  on the Pi 5 and its simulation gaps reproduce,
  but the hardware class itself was never touched
  (TF-174).
- **A soak, thermals under load, and the Hailo
  neighbour:** one probe run at ~44 °C is a reading,
  not an endurance claim (TF-176 owns the soak).
  The Pi's custom kernel carries the house's Hailo
  work; nothing in this slice touched it.

## Docket

- **D-DET-1 added + FIXED by TF-082**: the detector's
  RAM/mobile ordering defect, above. TF-016 totals
  move to 82 entries — 44 OPEN / 1 OPEN-ENVIRONMENT /
  35 FIXED / 2 CLOSED-BY-VERDICT (Part IV 31 → 32).
- The Pi shares of **D-COLD-1** and **D-BATCH-1** are
  discharged at fixture grade by the rows above
  (their live-model and truth-table shares stand,
  owned by TF-086/TF-174/TF-175 as filed). TF-053's
  per-machine embedding verdict for pi_5 is now
  measured on the machine itself:
  `available_offline`.

## Proof grade

REAL HARDWARE, fixture legs — the real tree's real
boots, detector, embedding encode, cold starts and
batch panel executed on a real Raspberry Pi 5 at the
campaign head. A live model's generation on the Pi
is **NOT PROVEN** (no `llama_cpp`); Pi Zero hardware
is **NOT PROVEN** (none in the house). Never claimed.
