# TF-083 — GPU-Optional Proof: No Profile Secretly Requires a GPU

*No profile secretly requires a GPU; CPU paths
measured on the laptop line. The estate lives in
`src/thoughtforge/inference/gpu_optional.py`
(`GPU_RULES` as data, the profile estate filed as
`PROFILE_GPU_ROWS` / `BACKEND_OFFLOAD`, the hosts
as `HOST_VM` / `HOST_LAPTOP`, the panels as
`CPU_PANEL_VM` / `CPU_PANEL_LAPTOP`, format
`thoughtforge-gpu-optional/1`). The one fix lives
in `src/thoughtforge/inference/backends.py`
(`build_llama_kwargs`), with the engine's load log
in `turboquant.py` reporting the effective count.*

## The plain truth first

Read plainly, the profile estate never required a
GPU: four of the six shipped profiles declare a CPU
path outright, and the two GPU profiles declare CPU
as a fallback in their own JSONs. The machinery
mostly honoured that — this campaign's every panel
has run on a GPU-less VM — but "mostly" hid one
real secret requirement, in the one place a profile
meets a model: the layer-offload count handed to
llama.cpp was resolved from the profile's *declared*
VRAM and merged over whichever backend was actually
selected. On a GPU-less machine, server_gpu
selected the CPU backend and still carried
**n_gpu_layers=-1** — CPU in name, all-layers-to-GPU
in fact. That is fixed in scope (D-GPU-1, below).

The CPU paths are now measured as a panel of their
own on two hosts: this GPU-less VM, and the laptop
line (purplelongship — a GPU machine whose CPU
paths are proven with the GPU present but
unselected). What is *not* proven is stated just
as plainly at the end: no live model generated on
CPU anywhere in this slice.

## The declarations, judged live

| Profile | Declared backend | Fallbacks | GPU / VRAM | Resolved layers |
|---|---|---|---|---|
| desktop_cpu | cpu | — | null / 0 | 0 |
| desktop_gpu | cuda | vulkan, directml, cpu | consumer / 12 | 40 |
| phone_low | cpu | — | null / 0 | 0 |
| pi_5 | cpu | — | null / 0 | 0 |
| pi_zero | cpu | — | null / 0 | 0 |
| server_gpu | cuda | cpu | datacenter / 24 | -1 |

The checker re-derives every cell from the shipped
JSONs and the resolver on each run: a non-CPU
profile that lost its CPU fallback would fail the
suite by name — that is what "no secret
requirement" means as a standing proof rather than
a one-day observation.

The backends' own declarations point the same way:
cuda / rocm / metal / vulkan declare offload (-1),
directml and cpu declare 0, and the cpu prober is
unconditionally available (priority 100) —
`select_backend` falls back to it, constructing a
minimal CPU backend even if detection itself
failed. Selection can never strand a profile for
want of a GPU.

## The defect the GPU-less host found — D-GPU-1

`TurboQuantEngine._load_llama` resolves
`n_gpu_layers` from the profile (the resolver
ladder of TF-012/D-PROF-2: -1 at ≥24 GB declared
VRAM, 40 at ≥12, 20 at ≥6, else 0), and
`build_llama_kwargs` merged that count *over* the
selected backend's own kwargs. The ladder knows
nothing of the machine; the selection knows
nothing of the ladder. Measured pre-fix on this VM
(no GPU, no `nvidia-smi`, no torch):

- **server_gpu** — preferred cuda unavailable →
  declared fallback selects **cpu** → built kwargs
  carried **n_gpu_layers=-1**.
- **desktop_gpu** — with cpu selected (the last of
  its declared fallbacks), the built kwargs would
  have carried **40**.

Whether a CPU-only llama.cpp build would have
errored on those counts or silently ignored them
is NOT PROVEN — `llama_cpp` is absent on both
hosts measured, so no model load ever consumed
the kwargs. The defect is at the forge's own
contract layer: the load said cpu and handed over
a GPU instruction. A deployment that *did* load
would have met it as a crash or a lie, depending
on the build.

**Fix (in scope):** `build_llama_kwargs` — the
tree's one kwargs constructor — now clamps the
count to 0 for any backend whose own declared
kwargs carry `n_gpu_layers == 0` (cpu; directml,
whose offload is a build property, not a kwarg).
The selected backend is the truth; the profile's
ladder applies only to a backend that can offload.
A GPU-capable backend keeps the caller's count
exactly (cuda + 32 → 32, pinned). The engine's
load log reports the effective count from the
built kwargs, not the pre-clamp resolved value.
Post-fix on this VM: server_gpu's effective count
under its CPU selection is **0**; desktop_gpu via
vulkan keeps 40, as a GPU-class backend should.

This narrows nothing in D-PROF-2: the ladder's
existence only in the resolver (stated in no JSON)
stays TF-086's truth-table share, as filed.

## The CPU panels

Per profile, on each host: a no-model
`TurboQuantEngine` load with the backend forced to
cpu, a no-model load as the profile declares it,
and a real `ThoughtForgeCore` construction + one
knowledge-only turn (the TF-069 shape). Medians of
3 runs per leg. Fixture walls on shared machines —
shapes, not model speeds.

**This VM** (htch-runtime, x86_64, 4 cores, 7.75 GB
RAM, vram 0.0, has_cuda False, detector
auto-selects desktop_cpu; available backends:
vulkan by library presence — D-GPU-2 — and cpu):

| Profile | CPU load | Declared load | Core + turn |
|---|---|---|---|
| desktop_cpu | 63.988 ms (cpu) | 32.037 ms (cpu) | 58.898 ms ✓ |
| desktop_gpu | 20.592 ms (cpu) | 22.481 ms (vulkan) | 4.012 ms ✓ |
| phone_low | 6.218 ms (cpu) | 13.373 ms (cpu) | 3.650 ms ✓ |
| pi_5 | 29.392 ms (cpu) | 11.250 ms (cpu) | 10.874 ms ✓ |
| pi_zero | 15.763 ms (cpu) | 15.250 ms (cpu) | 6.499 ms ✓ |
| server_gpu | 18.570 ms (cpu) | 66.504 ms (cpu) | 4.123 ms ✓ |

Every profile loads on CPU and answers here. Note
server_gpu's declared leg: on a GPU-less host its
own fallback order lands it on cpu unaided — the
estate working as declared, once the kwargs tell
the same story (D-GPU-1).

**The laptop line** (purplelongship, Ryzen 9
4900HS, 16 threads, 38.59 GB RAM, RTX 2060 Max-Q
6 GB, has_cuda True, detector auto-selects
desktop_gpu; backends cuda / vulkan / cpu; sparse
clone of this branch at the same head, 2222a52):

| Profile | CPU load | Declared load | Core + turn |
|---|---|---|---|
| desktop_cpu | 975.493 ms (cpu) | 1325.467 ms (cpu) | 1.282 ms ✓ |
| desktop_gpu | 1184.621 ms (cpu) | 1318.055 ms (cuda) | 1.140 ms ✓ |
| phone_low | 1233.098 ms (cpu) | 1125.153 ms (cpu) | 1.337 ms ✓ |
| pi_5 | 1205.072 ms (cpu) | 1271.910 ms (cpu) | 2.043 ms ✓ |
| pi_zero | 1357.764 ms (cpu) | 1371.904 ms (cpu) | 1.356 ms ✓ |
| server_gpu | 1040.627 ms (cpu) | 1158.987 ms (cuda) | 1.074 ms ✓ |

The laptop's engine-load walls are ~1.0–1.4 s
against the VM's ~6–66 ms — measured under a load
average of ~15.6 on 16 threads while other
campaigns worked the machine. Filed as measured
with the load beside them; they are contention
colour, never baselines (the standing D-14 rule).
The verdicts are the content: on the GPU machine,
every profile's CPU path *also* runs (forced-cpu
loads land on cpu for all six), the GPU profiles'
declared loads bind cuda as their JSONs intend,
and every core answers.

## The rest of the tree

A sweep of `src/` for hard CUDA/GPU requirements
finds none beyond the backend estate above: the
ONNX embedder defaults `use_gpu=False` and always
keeps the CPU execution provider in its list (the
CUDA provider is prepended only on request); the
HuggingFace backend is a remote API client and
names no local device; the embeddings module is
CPU-shaped (its per-profile truth is TF-053's
estate); the cognition core names no device at
all.

## Filed OPEN — D-GPU-2, the library-grade vulkan probe

Both vulkan probes (`backends._probe_vulkan`,
`profiles._check_vulkan`) accept the mere presence
of `libvulkan.so.1` as a working GPU backend. On
this VM the library exists, `vulkaninfo` does not,
and no GPU device exists — yet vulkan reports
available, and desktop_gpu's declared load selects
it here (see the VM panel). A library is not a
device. This is not a *requirement* — cpu stands
behind vulkan in the profile's own fallback order,
and D-GPU-1's clamp bounds what a no-offload
backend can be handed — but the probe's truth
grade is a library's, not a device's. Filed OPEN
as D-GPU-2, owned by TF-086's per-profile truth
table v2, which must carry the line.

## Not proven — the honest remainder

- **Live-model generation on CPU: NOT PROVEN** on
  either host. `llama_cpp` is absent on this VM
  and on the laptop (both measured this tick), and
  no GGUF exists in the tree: every load above is
  a no-model load and every turn a knowledge-only
  one. TF-127's live panel and TF-175's laptop
  panel own the road; TF-086's truth table must
  carry this line.
- **A GPU-less laptop: NOT PROVEN as hardware.**
  The laptop line's machine carries a GPU; its CPU
  paths are proven with the GPU present but
  unselected. The GPU-less host proof is this VM's,
  at desktop_cpu grade.
- **What a CPU-only llama.cpp build does with a
  nonzero layer count: NOT PROVEN** (no
  `llama_cpp` anywhere measured) — D-GPU-1 is
  fixed at the contract layer, where the forge's
  own words were wrong, without needing that
  answer.

## Docket

- **D-GPU-1 added + FIXED by TF-083**: the
  resolved-ladder-over-selected-backend defect,
  above.
- **D-GPU-2 added OPEN by TF-083**: the
  library-grade vulkan probe, above, assigned to
  TF-086's truth table v2.
- TF-016 totals move to 84 entries — 45 OPEN / 1
  OPEN-ENVIRONMENT / 36 FIXED / 2
  CLOSED-BY-VERDICT (Part IV 32 → 34).

## Proof grade

FIXTURE — real per-profile engine loads
(forced-CPU and as-declared) and real core turns
on two hosts (this GPU-less VM; the laptop line
under stated contention), the model-load kwargs
judged live per profile, the declarations judged
live against the shipped JSONs. A live model's
generation on CPU is **NOT PROVEN** (`llama_cpp`
absent on both hosts); GPU-less laptop hardware
is **NOT PROVEN** (the laptop carries a GPU).
Never claimed.
