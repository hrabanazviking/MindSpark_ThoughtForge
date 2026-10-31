# TF-070 — ONNX Export Path

**The plain truth first: this path exports one kind of
thing — an embedding model — and on most machines,
including this one, it exports nothing at all.** The
ONNX path exists so a small embedding model can run on
hardware too small for the PyTorch stack (TF-053 assigns
it to phone_low). But its claims had never been audited:
the dimension it reported was hardcoded, its
"quantized" flag echoed the request whether or not any
quantization happened, and nothing in the tree actually
calls it. This slice files the truth as data
(`src/thoughtforge/inference/onnx_path.py`, format
`thoughtforge-onnx-path/1`), fixes the lies in scope
(D-ONNX-1), and exercises the exportable set end-to-end
through stub-module fixtures — because no real export
can run on a machine with none of the packages.

## What exports, what doesn't

| | |
|---|---|
| **Exports** | A sentence-transformer embedding model to a directory (`model.onnx`, optionally `model_quantized.onnx`), optimum feature-extraction first, `torch.onnx` fallback |
| **Quantizes** | Dynamic int8, optimum path only (ORTQuantizer, avx512_vnni config) — and only ever *reported* when a quantized file really landed |
| **Embeds** | `ONNXEmbedder`: onnxruntime session over an exported directory, mean pooling over the attention mask, L2-normalised; the plain file is preferred, the quantized file the fallback — and which one loaded is now recorded (`onnx_path`, `quantized`) |
| **Never exports** | Language models. No GGUF catalogue entry, no TurboQuant model, no cognition component has any export path; `OnnxExporter` is the only exporter in the tree and `export_embedder` its only entry |
| **Never runs package-free** | Export needs optimum or torch (plus transformers); embedding needs onnxruntime + transformers. Absence is an ImportError with an install hint — measured live on this VM, where all four packages are absent and no ONNX artefact exists in the embeddings home |

**The exportable set is exactly TF-053's registry:**
one model, all-MiniLM-L6-v2 (384-dim, Apache-2.0, hub
spelling `sentence-transformers/all-MiniLM-L6-v2` — the
exporter's own default). Any other sentence-transformer
name is *accepted* by the code but unregistered: no
dimension, size or licence truth exists for it in-tree.

**Nothing calls the path.** No module in `src/` invokes
the exporter or the embedder; TF-053 checks whether an
artefact *exists* in the embeddings home, but landing
one there is an operator's manual act. The path is a
standalone tool, not a wired pipeline stage.

## D-ONNX-1 — the lies, fixed

- `embedding_dim` was hardcoded **384 for every model**
  (the comment claimed it was "probed at runtime"; no
  probe existed). Now `_probe_embedding_dim` reads the
  model's config `hidden_size`, falls back to the TF-053
  registry dimension for a registered model, and only
  then to 384.
- `ExportResult.quantized` **echoed the request**: with
  the quantizer unavailable, or producing no quantized
  file, the result still claimed quantized over the
  plain path. Now `_quantize_optimum` returns
  (path, actually_done) and the result reports what
  really happened.
- The torch fallback **silently dropped** a `quantize`
  request (the parameter never reached it). Now the
  request is passed through, announced in the log as
  unhonourable on that path, and reported `False`.
- The result doc promised a **"mock" exporter** that
  does not exist; removed.

## D-ONNX-2 — a flag without a consumer (filed, TF-086)

Per-profile rows, recomputed live: phone_low is
**consistent** (deployment flag `true`, TF-053 runtime
`onnx`). **pi_zero declares `onnx_export: true` but
TF-053 assigns it no embedding runtime at all** (512 MB
carries no embedding model honestly) — the flag
promises an export consumer that profile does not have.
The other four profiles declare `false` and use the
sentence-transformers runtime. TF-086's per-profile
truth table v2 owns the verdict.

## The fixtures

`tests/test_tf070_onnx_path.py` injects stub optimum /
torch / transformers / onnxruntime modules into
`sys.modules` (restored afterwards) and runs the **real**
exporter and embedder against them: optimum priority,
the config-probed dimension (768, not 384), the registry
fallback, quantized-when-it-lands and
never-when-it-doesn't, the torch fallback's honest
`quantized=False`, the embedder's file choice recorded,
mean pooling that excludes the padded position's
[100, 100], L2 normalisation, and empty input.

## Proof grade

FIXTURE/STATIC — AST and source pins over the real
module, stub-module runs of the real logic, live
profile JSONs and the TF-053 registry, on this VM. A
**real** model's export, and a real ONNX session's
embeddings, are **NOT PROVEN**: no optimum, torch,
onnxruntime or transformers exists on this machine and
no artefact exists in-tree. A machine carrying the
declared `onnx` extra owns that road. Never claimed.
