"""TF-070 — ONNX export path: the truth about what
exports, what doesn't, and the fixtures for the
exportable set.

Roadmap: "onnx_export's truth — what exports, what
doesn't, fixtures for the exportable set".

Measured start (probed this tick, VM):

- The path exports exactly ONE kind of thing: a
  sentence-transformer *embedding* model to a directory
  of ONNX files (``model.onnx``, optionally
  ``model_quantized.onnx``), via optimum's
  feature-extraction export first, torch.onnx second.
  Nothing exports a language model: no GGUF catalogue
  entry, no TurboQuant model, no cognition component
  has an export path at all (AST: ``OnnxExporter`` is
  the only exporter in the tree, and its sole entry
  point is ``export_embedder``).
- Nothing in ``src/`` *calls* the exporter or the
  embedder (grep: the only consumers are TF-053's
  name-canonicalisation probe and the tests). The path
  is a standalone tool, not a wired pipeline stage:
  TF-053 assigns phone_low the ONNX *runtime* and
  checks for the artefact in the embeddings home, but
  landing an artefact there is an operator's manual
  act, named as such.
- On this VM the path exports NOTHING and embeds
  NOTHING: optimum, torch, onnxruntime and transformers
  are all absent (find_spec False x4), so
  ``export_embedder`` raises its ImportError and
  ``ONNXEmbedder`` raises its ImportError — both
  measured live this tick. No ONNX artefact exists in
  the embeddings home either (TF-053's
  ``onnx_artefact_status``: present False).
- The exportable set is exactly the TF-053 registry:
  one registered embedding model
  (all-MiniLM-L6-v2, 384-dim, hub spelling
  ``sentence-transformers/all-MiniLM-L6-v2`` — the
  exporter's own default, canonicalised by TF-053).
  Any other sentence-transformer name is *accepted* by
  the code (it is passed straight to the hub loaders)
  but is unregistered, so no dimension, size or licence
  truth exists for it in-tree.

D-ONNX-1 — FOUND + FIXED in this slice. The optimum
path lied twice and the torch fallback once:

- ``embedding_dim`` was hardcoded 384 for every model
  (the comment even claimed "probed at runtime" — no
  probe existed). Fixed: ``_probe_embedding_dim`` reads
  the model's config ``hidden_size``, falls back to the
  TF-053 registry dimension for a registered model, and
  only then to the default model's 384.
- ``ExportResult.quantized`` echoed the *request*: when
  the ORTQuantizer was unavailable, or produced no
  quantized file, the result still claimed
  quantized=True over the plain path. Fixed:
  ``_quantize_optimum`` returns (path, actually_done)
  and the result reports what really happened.
- The torch fallback silently dropped a ``quantize``
  request (the parameter never reached it). Fixed: the
  request is passed through, announced in the log as
  unhonourable on that path, and reported False.

Also fixed, observability: ``ONNXEmbedder`` now records
``onnx_path`` (the file actually loaded — plain
preferred, quantized the fallback) and ``quantized``;
silence about which artefact serves is over. And the
``ExportResult.exporter_used`` doc no longer promises a
"mock" exporter that does not exist.

D-ONNX-2 — FOUND, filed, assigned TF-086: pi_zero's
deployment section declares ``onnx_export: true`` but
TF-053 assigns pi_zero NO embedding runtime at all
(runtime ``none`` — 512 MB carries no embedding model
honestly). The flag promises an export consumer that
profile does not have; phone_low's identical flag is
consistent (flag true, runtime onnx). TF-086's
per-profile truth table v2 owns the verdict (fix the
flag, or scope what it means).

THE FIXTURES (tests/test_tf070_onnx_path.py): the
exportable set is exercised end-to-end with stub
optimum / torch / transformers / onnxruntime modules
injected into ``sys.modules`` (and restored) — the
real export logic (priority, quantization honesty, dim
probe, file layout) and the real embedder logic
(session choice, mean pooling over the attention mask,
normalisation, empty input) run against deterministic
fakes, because no real export can run on this machine.
That is a fixture proof of the *logic*, never of a real
model's export quality.

Proof grade FIXTURE/STATIC (AST + source pins over the
real module, stub-module fixture runs of the real
exporter and embedder, live profile JSONs + TF-053
registry, VM) — a real model's export or a real ONNX
session's embeddings are NOT PROVEN (no optimum /
torch / onnxruntime / transformers on this VM, no
artefact in-tree; a machine carrying the declared
``onnx`` extra owns that road), never claimed.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

from thoughtforge.etl.embedding_discipline import (
    EMBEDDING_MODELS,
    PROFILE_EMBEDDINGS,
    canonical_model_name,
    onnx_artefact_status,
)
from thoughtforge.utils.config import load_hardware_profile

PATH_FORMAT = "thoughtforge-onnx-path/1"

_ONNX_MODULE = Path(__file__).with_name("onnx_export.py")

#: The packages each leg of the path needs importable.
EXPORT_PACKAGES: dict[str, tuple[str, ...]] = {
    "optimum_path": ("optimum", "transformers"),
    "torch_path": ("torch", "transformers"),
    "embedder": ("onnxruntime", "transformers"),
}

#: What the path does and does not do, as data — every
#: row judged live by the checker below.
PATH_DUTIES: dict[str, dict[str, Any]] = {
    "export_embedding_model": {
        "does": True, "anchor": "export_embedder",
        "note": "a sentence-transformer embedding model, "
                "optimum feature-extraction first, "
                "torch.onnx fallback",
    },
    "quantize_int8": {
        "does": True, "anchor": "_quantize_optimum",
        "note": "optimum path only, dynamic int8 "
                "(avx512_vnni config); reported only "
                "when a quantized file really landed",
    },
    "embed_from_export": {
        "does": True, "anchor": "ONNXEmbedder",
        "note": "onnxruntime session, mean pooling over "
                "the attention mask, L2-normalised",
    },
    "export_language_model": {
        "does": False, "anchor": None,
        "note": "no GGUF / catalogue / TurboQuant model "
                "has any export path",
    },
    "quantize_on_torch_path": {
        "does": False, "anchor": None,
        "note": "the torch fallback cannot quantize; a "
                "request there is announced, never "
                "silently dropped or falsely reported",
    },
    "run_without_optional_packages": {
        "does": False, "anchor": None,
        "note": "export needs optimum or torch (plus "
                "transformers); embedding needs "
                "onnxruntime + transformers; absence is "
                "an ImportError with an install hint",
    },
}


def package_status() -> dict[str, bool]:
    """Which of the path's packages are importable on
    THIS machine, measured live (find_spec)."""
    names = {p for pkgs in EXPORT_PACKAGES.values() for p in pkgs}
    return {name: importlib.util.find_spec(name) is not None
            for name in sorted(names)}


def capability() -> dict[str, Any]:
    """This machine's ONNX capability, as data: which
    legs could run here at all (package presence only —
    a present package does not prove a model's export)."""
    status = package_status()
    legs = {
        leg: all(status[p] for p in pkgs)
        for leg, pkgs in EXPORT_PACKAGES.items()
    }
    return {
        "packages": status,
        "legs": legs,
        "export_possible": legs["optimum_path"] or legs["torch_path"],
        "embed_possible": legs["embedder"],
        "artefact": onnx_artefact_status(),
    }


def exportable_models() -> dict[str, dict[str, Any]]:
    """The exportable set: exactly TF-053's registered
    embedding models, with the truth the registry holds
    for each (dimension, hub spelling, licence)."""
    return {
        name: {
            "hub_id": row["hub_id"],
            "dimension": row["dimension"],
            "size_mb": row["size_mb"],
            "licence": row["licence"],
        }
        for name, row in EMBEDDING_MODELS.items()
    }


def profile_onnx_rows() -> dict[str, dict[str, Any]]:
    """Per shipped profile: its deployment ``onnx_export``
    flag (read live from its JSON) beside TF-053's
    embedding assignment, with a verdict per row."""
    rows: dict[str, dict[str, Any]] = {}
    for pid, assignment in PROFILE_EMBEDDINGS.items():
        flag = bool(load_hardware_profile(pid)["deployment"]["onnx_export"])
        runtime = assignment["runtime"]
        if runtime == "onnx":
            verdict = "consistent" if flag else "runtime_without_flag"
        elif runtime == "none":
            verdict = "flag_without_assignment" if flag else "consistent_none"
        else:
            verdict = "flag_without_runtime" if flag else "consistent"
        rows[pid] = {
            "deployment_onnx_export": flag,
            "embedding_runtime": runtime,
            "verdict": verdict,
        }
    return rows


#: The one filed inconsistency (D-ONNX-2, TF-086 owns
#: the verdict): profiles whose row verdict is not a
#: consistent one, recomputed by the checker.
PROFILE_ONNX_GAPS: frozenset[str] = frozenset({"pi_zero"})


def onnx_path_as_data() -> dict[str, Any]:
    """The whole estate as JSON-safe data."""
    return {
        "format": PATH_FORMAT,
        "duties": {k: dict(v) for k, v in PATH_DUTIES.items()},
        "exportable_models": exportable_models(),
        "profile_rows": profile_onnx_rows(),
        "profile_gaps": sorted(PROFILE_ONNX_GAPS),
        "capability": capability(),
    }


# ── Live checker ────────────────────────────────────────────────────────────

def _source() -> str:
    return _ONNX_MODULE.read_text(encoding="utf-8")


def _class_methods(tree: ast.Module, class_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return set()


def onnx_path_problems() -> list[str]:
    """Re-derive the path's claims live; [] means true."""
    problems: list[str] = []
    source = _source()
    tree = ast.parse(source)

    # The exporter is the only exporter, and embeds are
    # its only product: exactly the two classes, and
    # export_embedder the only public export entry.
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    if classes != {"ExportResult", "OnnxExporter", "ONNXEmbedder"}:
        problems.append(f"onnx_export classes changed: {sorted(classes)}")
    exporter_methods = _class_methods(tree, "OnnxExporter")
    if "export_embedder" not in exporter_methods:
        problems.append("OnnxExporter.export_embedder missing")

    # Priority + honesty pins (D-ONNX-1's fixes stay fixed):
    if source.index("_export_with_optimum") > source.index("_export_with_torch(") \
            or "ONNX export requires either 'optimum' or 'torch'" not in source:
        problems.append("exporter priority / ImportError truth changed")
    if "_probe_embedding_dim" not in source \
            or "AutoConfig.from_pretrained" not in source:
        problems.append("embedding_dim is no longer probed from config")
    if "actually" not in source or "quantized=quantized" not in source:
        problems.append("quantized no longer reports the measured outcome")
    if source.count("embedding_dim=384") or "embedding_dim=384," in source:
        problems.append("a hardcoded 384 dimension returned")
    if "self.onnx_path = onnx_path" not in source:
        problems.append("embedder no longer records the loaded file")
    torch_body = source[source.index("def _export_with_torch"):]
    if "cannot quantize" not in torch_body:
        problems.append("torch path no longer announces unhonoured quantize")

    # The exportable set is exactly the registry, and the
    # exporter's default is a registered model:
    from thoughtforge.inference import onnx_export

    if canonical_model_name(onnx_export._DEFAULT_MODEL) \
            not in EMBEDDING_MODELS:
        problems.append("exporter default is not a registered model")
    models = exportable_models()
    if set(models) != set(EMBEDDING_MODELS):
        problems.append("exportable set != TF-053 registry")
    for name, row in models.items():
        if row["dimension"] != EMBEDDING_MODELS[name]["dimension"]:
            problems.append(f"{name}: dimension drifted from registry")

    # Capability is measured, not filed: recompute.
    cap = capability()
    status = package_status()
    if cap["packages"] != status:
        problems.append("capability packages do not match find_spec")
    for leg, pkgs in EXPORT_PACKAGES.items():
        if cap["legs"][leg] != all(status[p] for p in pkgs):
            problems.append(f"capability leg {leg} mismeasured")

    # When NO export leg can run, the exporter must say
    # so (the ImportError is this machine's measured
    # truth); when one can, this probe does not run —
    # a real export is never attempted by a checker.
    if not cap["export_possible"]:
        import tempfile

        from thoughtforge.inference.onnx_export import OnnxExporter

        try:
            with tempfile.TemporaryDirectory(prefix="tf070-") as tmp:
                OnnxExporter().export_embedder(output_dir=Path(tmp))
            problems.append("export succeeded with no exporter package")
        except ImportError:
            pass
    if not cap["embed_possible"]:
        from thoughtforge.inference.onnx_export import ONNXEmbedder

        try:
            ONNXEmbedder(Path("/nonexistent-tf070"))
            problems.append("embedder loaded with no runtime package")
        except ImportError:
            pass

    # Profile rows recompute; the filed gap is exact.
    rows = profile_onnx_rows()
    gaps = {pid for pid, r in rows.items()
            if r["verdict"] not in ("consistent", "consistent_none")}
    if gaps != set(PROFILE_ONNX_GAPS):
        problems.append(f"profile ONNX gaps {sorted(gaps)} != filed "
                        f"{sorted(PROFILE_ONNX_GAPS)}")
    if rows["phone_low"]["verdict"] != "consistent":
        problems.append("phone_low's ONNX row is no longer consistent")

    return problems
