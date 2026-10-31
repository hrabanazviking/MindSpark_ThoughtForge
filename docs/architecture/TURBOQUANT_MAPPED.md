# TF-066 — TurboQuant Mapped

**The plain truth first: TurboQuant does not quantise anything.**
`TurboQuantEngine` (`src/thoughtforge/inference/turboquant.py`)
contains no quantisation code at all — no function that
quantises or dequantises, no weight transformation, no
quantisation parameter. What it does is load a model file that
was *already* quantised by somebody else's tooling, and run it
well. The quantisation happened upstream, when the GGUF file
was made (llama.cpp's quantize tool or the publisher's export
pipeline), before ThoughtForge ever saw the file. This slice
documents the method from the code, files it as data in
`src/thoughtforge/inference/turboquant_map.py` (format
`thoughtforge-turboquant-map/1`), and judges it live.

## What the engine actually does

| Duty | Where |
|---|---|
| Loads a pre-quantised GGUF via `llama_cpp.Llama(**kwargs)` — kwargs exactly `model_path, n_ctx, n_threads, verbose, n_gpu_layers` | `turboquant._load_llama` + `backends.build_llama_kwargs` |
| Resolves the hardware profile + generation params | `turboquant.load`, `profiles.resolve_generation_params` |
| Selects the acceleration backend (CUDA > ROCm > Metal > Vulkan > DirectML > CPU) | `backends.select_backend` |
| Enforces the profile's per-turn token budget on every call | `turboquant._enforce_token_budget` |
| Generates, streams, and multi-drafts over the loaded model | `turboquant.generate*` |
| Estimates tokens (model tokeniser when loaded; ~4 chars/token estimate until then) | `turboquant.estimate_tokens` (TF-065) |

And what it does **not** do, each pinned by test:

- **Quantise weights** — no code path exists (AST-pinned).
- **Inspect the loaded file's quant type** — the engine never
  parses the filename, reads a GGUF header, or otherwise learns
  whether it loaded Q2_K or fp16.
- **Enforce the profile's recommended quant** — the profile's
  `model.quantization` is resolved into
  `ResolvedGenerationParams.quantization` and then read
  *nowhere* in the engine (**D-QUANT-1**, OPEN — TF-068's
  selection policy is where a choice, and any enforcement,
  belongs; measured this slice by source pin).

## Where quant actually enters the system

Three data surfaces, none of them weight-transforming code:

1. **The filename vocabulary** — `backends.QUANT_SUFFIXES`
   maps 9 types (`q2_k, q3_k_m, q4_k_m, q4_0, q5_k_m, q6_k,
   q8_0, fp16, bf16`) to GGUF filename suffixes. Until this
   slice it was *consumed nowhere* (**D-QUANT-2, FIXED here**:
   `turboquant_map` is its live consumer, AST/data-pinned).
   The catalogue's `IQ2_XS` is absent from it — the map adds it
   as a browser-only type rather than pretending the suffix
   table ever held it.
2. **Profile recommendations** — `hardware_profiles/*.json`:
   `phone_low`/`pi_zero` → `q2_k`, `pi_5`/`desktop_cpu` →
   `q4_k_m`, `desktop_gpu` → `q8_0`, `server_gpu` → `fp16`
   (read live by the map; the resolver's default is `q4_k_m`).
   A recommendation, not a gate — see D-QUANT-1.
3. **The download catalogue** — `model_browser.CURATED_MODELS`
   (15 entries; declared quants Q4_K_M ×9, Q5_K_M ×2, Q8_0 ×2,
   IQ2_XS ×2). Filename inference (`quant_of_filename`, which
   handles the dot form `.Q4_K_M.gguf` the raw suffix map
   misses) proves the declared quant for 13 of 15. The two
   Phi-3 filenames (`-q4`, `-q5`) under-specify the variant —
   they cannot prove K_M from the name — and are filed as
   `FILENAME_QUANT_GAPS`, never rounded up.

The only in-tree code that *performs* a quantisation is not
TurboQuant: `inference/onnx_export.py`'s int8 ONNX path
(TF-070 owns its truth) and the embedding estate's
`model_quantized.onnx` (TF-053). Neither is claimed here.

## The loss claims, with what loss claims

**No quantitative loss figure exists anywhere in the shipped
tree** — no perplexity delta, no quality percentage, no
measured curve. Every loss-adjacent claim is filed in
`LOSS_CLAIMS` with its measurement state, and all are
unmeasured:

| Claim | Source | Verdict |
|---|---|---|
| "Quantization: 2-bit, 3-bit, 4-bit, 8-bit, fp16, bf16" | `inference/__init__.py` | PARTIAL inventory — the classes exist as q2_k / q3_k_m (the only 3-bit) / q4 family / q8_0 / fp16 / bf16; iq2_xs missing from the suffix map |
| "Q8 — near full-precision quality" | model browser copy | UNMEASURED — TF-067 owns the measurement |
| "Q5 — higher fidelity" | model browser copy | UNMEASURED — TF-067 |
| "IQ2 — extreme compression" (72B: 19 GB vs 43 GB at Q4_K_M) | model browser sizes | Catalogue-stated sizes, never weighed in-tree; quality cost UNMEASURED — TF-067 |
| "2-bit phone → fp16 server" | `docs/index.md` | Matches the profile table — a range statement, not a loss figure |
| "expect noticeable quality degradation, particularly in reasoning" (1B quantisation) | `docs/specs/TurboQuant_Cognition_Blueprint.md` | A research expectation in a spec, not a shipped-code claim; TF-067's curve is the first in-tree evidence either way |

TF-067 measures the curve; TF-068 chooses per-profile defaults
*from that curve*. Until then, no default in this tree rests on
a measured loss claim, and this map does not pretend otherwise.

## Proof grade

FIXTURE/STATIC (AST + source pins over the real engine, live
calls to `build_llama_kwargs`, the real catalogue and profile
JSONs, VM) — no GGUF was loaded and no quantisation was run;
the quality of any quant level is NOT PROVEN (TF-067), a
loaded engine's behaviour is TF-065/TF-067's road, never
claimed here.
