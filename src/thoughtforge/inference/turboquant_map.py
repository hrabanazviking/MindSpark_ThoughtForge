"""TF-066 — TurboQuant mapped: what turboquant.py actually does.

The name promises quantisation. The code, read whole, does not
contain any. This module files the map as data and judges it
live against the sources, in the campaign's standing shape:

- **The engine quantises nothing.** ``TurboQuantEngine``
  (``inference/turboquant.py``, 551 lines) has no quantise /
  dequantise function, no weight transformation, and no
  quantisation parameter. It loads a GGUF file — a file that
  was quantised *elsewhere*, by llama.cpp's own tooling or the
  publisher's export, before ThoughtForge ever saw it — through
  ``llama_cpp.Llama(**kwargs)``, where the kwargs are exactly
  ``model_path, n_ctx, n_threads, verbose, n_gpu_layers``
  (``backends.build_llama_kwargs``). The quant type of the
  loaded model is a property of the *file*, and the engine
  never reads it: no suffix parsing, no header inspection, no
  ``.quantization`` read anywhere in the module.
- **What the engine does instead** is the discipline around a
  pre-quantised file: hardware-profile resolution, acceleration
  backend selection, strict per-turn token budgets, generation
  / streaming / multi-draft, and token estimation (model
  tokeniser when loaded, the ~4-chars/token estimate until
  then — TF-065).
- **Where quant actually enters the system** is three data
  surfaces, none of them code that transforms weights:
  ``backends.QUANT_SUFFIXES`` (a filename-suffix vocabulary —
  defined in TF-008's era and, until this slice, *consumed
  nowhere*: **D-QUANT-2, FIXED here** by making this map its
  live consumer), the hardware profiles' ``model.quantization``
  recommendation (resolved into
  ``ResolvedGenerationParams.quantization`` and then *never
  read by the engine* — **D-QUANT-1, OPEN, owned by TF-068**,
  whose selection policy is where enforcement belongs), and
  ``model_browser.CURATED_MODELS`` (a download catalogue whose
  entries carry a declared quant each).
- **The only in-tree code that performs a quantisation** is
  not TurboQuant at all: ``inference/onnx_export.py``'s int8
  ONNX path (TF-070 owns its truth), and the embedding estate's
  ``model_quantized.onnx`` (TF-053). Neither is claimed here.
- **Loss claims**: the shipped tree carries *no quantitative
  loss figure anywhere* — no perplexity delta, no quality
  percentage, no measured curve. Every loss-adjacent claim is
  qualitative (``LOSS_CLAIMS``): Q8_0's "near full-precision
  quality", Q5's "higher fidelity", IQ2's "extreme
  compression". They are catalogue copy, unmeasured in-tree;
  TF-067's fixtures own turning them into a measured curve,
  and TF-068 chooses defaults from that curve — never from
  the copy.

Filename inference (``quant_of_filename``) is deliberately
more careful than the suffix map it consumes: the map's
hyphen-prefixed suffixes miss the catalogue's dot form
(``.Q4_K_M.gguf``) outright, and two catalogue filenames
(Phi-3's ``-q4`` / ``-q5``) under-specify the variant — they
cannot prove K_M from the name, and the map says so as data
(``FILENAME_QUANT_GAPS``) instead of rounding up.
"""

from __future__ import annotations

import ast
from pathlib import Path

from thoughtforge.inference.backends import QUANT_SUFFIXES, BackendInfo, build_llama_kwargs
from thoughtforge.inference.model_browser import ModelBrowser
from thoughtforge.utils.config import list_hardware_profiles, load_hardware_profile

MAP_FORMAT = "thoughtforge-turboquant-map/1"

# ── What the engine does (and does not do) ─────────────────────────────────

ENGINE_DUTIES: tuple[dict, ...] = (
    {"id": "load_gguf", "does": True,
     "what": "loads a pre-quantised GGUF file via llama_cpp.Llama",
     "anchor": "turboquant._load_llama"},
    {"id": "profile_resolution", "does": True,
     "what": "resolves the hardware profile and generation params",
     "anchor": "turboquant.load / profiles.resolve_generation_params"},
    {"id": "backend_selection", "does": True,
     "what": "selects the acceleration backend (cuda/rocm/metal/vulkan/directml/cpu)",
     "anchor": "backends.select_backend"},
    {"id": "token_budget", "does": True,
     "what": "enforces the profile's per-turn token budget on every call",
     "anchor": "turboquant._enforce_token_budget"},
    {"id": "generate_stream_drafts", "does": True,
     "what": "generate / generate_stream / generate_drafts over the loaded model",
     "anchor": "turboquant.generate*"},
    {"id": "token_estimation", "does": True,
     "what": "model tokeniser when loaded, ~4 chars/token estimate until then",
     "anchor": "turboquant.estimate_tokens"},
    {"id": "quantise_weights", "does": False,
     "what": "quantises weights — NO code path exists; the GGUF arrives quantised",
     "anchor": "absent from turboquant.py (AST-pinned)"},
    {"id": "inspect_loaded_quant", "does": False,
     "what": "reads the loaded file's quant type — never parsed, never checked",
     "anchor": "absent from turboquant.py (source-pinned)"},
    {"id": "enforce_profile_quant", "does": False,
     "what": "enforces the profile's recommended quant — resolved, never read (D-QUANT-1)",
     "anchor": "profiles.resolve_generation_params -> read nowhere in turboquant.py"},
)

# ── The quant vocabulary ────────────────────────────────────────────────────
# Canonical keys are the suffix map's, lower-case. ``family`` is
# read off the name only (K-quant / legacy / i-quant / full
# precision) — the tree states no bits-per-weight figure for any
# type, and this map invents none.

def _family(key: str) -> str:
    if key in ("fp16", "bf16"):
        return "full_precision"
    if key.startswith("iq"):
        return "i_quant"
    if "_k" in key:
        return "k_quant"
    return "legacy_quant"


QUANT_TYPES: tuple[dict, ...] = tuple(
    {"id": key, "family": _family(key), "source": "backends.QUANT_SUFFIXES",
     "suffixes": list(suffixes)}
    for key, suffixes in QUANT_SUFFIXES.items()
) + (
    {"id": "iq2_xs", "family": "i_quant", "source": "model_browser_only",
     "suffixes": ["-IQ2_XS.gguf", "-iq2_xs.gguf"]},
)

KNOWN_QUANTS = frozenset(row["id"] for row in QUANT_TYPES)

# Generic filename tokens that name a bit class but not a variant.
GENERIC_QUANTS = frozenset({"q4", "q5", "q8"})


def _normalise(token: str) -> str:
    return token.lower().replace("-", "_").replace(".", "_").strip("_")


def quant_of_filename(filename: str) -> str | None:
    """The quant a GGUF filename proves, or None.

    Parses the token immediately before ``.gguf`` (dot- and
    hyphen-separated forms alike), so ``.Q4_K_M.gguf`` resolves
    where the raw suffix map cannot. A generic token (``-q4``)
    returns the generic class — it does NOT prove the declared
    K_M variant, and callers must not round it up.
    """
    name = Path(filename).name
    if not name.lower().endswith(".gguf"):
        return None
    stem = name[: -len(".gguf")]
    token = _normalise(stem.replace("-", ".").split(".")[-1])
    # Multi-part tokens arrive joined (q4_k_m); single tokens are
    # already whole. Re-join the tail while it matches a known id.
    parts = stem.replace("-", "_").replace(".", "_").lower().split("_")
    for start in range(len(parts)):
        candidate = "_".join(p for p in parts[start:] if p)
        if candidate in KNOWN_QUANTS:
            return candidate
        if candidate == "f16":  # the suffix map's own -F16 form of fp16
            return "fp16"
    if token in GENERIC_QUANTS:
        return token
    return None


def filename_quant_gap(entry) -> str | None:
    """Why a catalogue entry's filename does not prove its
    declared quant, or None when it does."""
    inferred = quant_of_filename(entry.filename)
    declared = _normalise(entry.quantization)
    if inferred == declared:
        return None
    if inferred in GENERIC_QUANTS and declared.startswith(inferred + "_"):
        return "filename_under_specifies_variant"
    return f"filename_proves_{inferred}_declared_{declared}"


FILENAME_QUANT_GAPS: tuple[dict, ...] = tuple(
    {"filename": m.filename, "declared": m.quantization, "gap": gap}
    for m in ModelBrowser.CURATED_MODELS
    if (gap := filename_quant_gap(m)) is not None
)

# ── Profile recommendations (read live) ─────────────────────────────────────

def profile_quants() -> dict[str, str]:
    """Each shipped hardware profile's recommended quant, live
    from its JSON (the resolver's own default when unstated)."""
    out: dict[str, str] = {}
    for pid in sorted(list_hardware_profiles()):
        profile = load_hardware_profile(pid)
        out[pid] = str(profile.get("model", {}).get("quantization", "q4_k_m")).lower()
    return out


# ── Loss claims, as found ───────────────────────────────────────────────────
# Every loss-adjacent claim in the shipped tree, with its kind
# and its measurement state. ``measured_in_tree`` is False for
# all of them: TF-067 owns the first measured curve.

LOSS_CLAIMS: tuple[dict, ...] = (
    {"claim": "Quantization: 2-bit, 3-bit, 4-bit, 8-bit, fp16, bf16",
     "source": "inference/__init__.py module docstring",
     "kind": "inventory",
     "measured_in_tree": False,
     "verdict": "PARTIAL — the suffix vocabulary holds the classes as "
                "q2_k / q3_k_m / q4 family / q8_0 / fp16 / bf16; the "
                "catalogue's iq2_xs is absent from the suffix map"},
    {"claim": "Q8 — near full-precision quality",
     "source": "model_browser CURATED_MODELS (Llama-3.1-8B Q8_0 description)",
     "kind": "qualitative_loss",
     "measured_in_tree": False,
     "verdict": "UNMEASURED catalogue copy — TF-067 owns the measurement"},
    {"claim": "Q5 — higher fidelity",
     "source": "model_browser CURATED_MODELS (Phi-3 / Llama-3.2 Q5_K_M descriptions)",
     "kind": "qualitative_loss",
     "measured_in_tree": False,
     "verdict": "UNMEASURED catalogue copy — TF-067 owns the measurement"},
    {"claim": "IQ2 — extreme compression (72B at 19 GB vs 43 GB at Q4_K_M, catalogue sizes)",
     "source": "model_browser CURATED_MODELS (IQ2_XS entries)",
     "kind": "size",
     "measured_in_tree": False,
     "verdict": "Catalogue-stated sizes, never weighed in-tree; the "
                "quality cost is UNMEASURED — TF-067 owns it"},
    {"claim": "2-bit phone -> fp16 server",
     "source": "docs/index.md TurboQuant row",
     "kind": "inventory",
     "measured_in_tree": False,
     "verdict": "MATCHES the profile table (phone_low/pi_zero q2_k, "
                "server_gpu fp16) — a range statement, not a loss figure"},
    {"claim": "expect noticeable quality degradation, particularly in reasoning (1B quantisation)",
     "source": "docs/specs/TurboQuant_Cognition_Blueprint.md (research spec)",
     "kind": "qualitative_loss",
     "measured_in_tree": False,
     "verdict": "A research expectation, not a shipped-code claim; "
                "TF-067's curve is the first in-tree evidence either way"},
)


# ── Live checkers ───────────────────────────────────────────────────────────

def _turboquant_source() -> str:
    import thoughtforge.inference.turboquant as tq

    return Path(tq.__file__).read_text(encoding="utf-8")


def turboquant_map_problems() -> list[str]:
    """Re-derive the whole map live; [] means the map is true."""
    problems: list[str] = []
    source = _turboquant_source()
    tree = ast.parse(source)

    fn_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if any("quantiz" in name or "dequant" in name for name in fn_names):
        problems.append("engine grew a quantisation function — map is stale")
    if ".quantization" in source:
        problems.append("engine now reads .quantization — D-QUANT-1/map stale")

    kwargs = build_llama_kwargs(
        BackendInfo(name="cpu", available=True, priority=100,
                    llama_cpp_kwargs={"n_gpu_layers": 0}),
        model_path="x.gguf",
    )
    if set(kwargs) != {"model_path", "n_ctx", "n_threads", "verbose", "n_gpu_layers"}:
        problems.append(f"llama kwargs changed: {sorted(kwargs)}")

    suffix_ids = {row["id"] for row in QUANT_TYPES if row["source"] == "backends.QUANT_SUFFIXES"}
    if suffix_ids != set(QUANT_SUFFIXES):
        problems.append("QUANT_TYPES drifted from QUANT_SUFFIXES")

    for entry in ModelBrowser.CURATED_MODELS:
        if _normalise(entry.quantization) not in KNOWN_QUANTS:
            problems.append(f"catalogue quant unknown: {entry.quantization}")
    gaps = {(g["filename"], g["gap"]) for g in FILENAME_QUANT_GAPS}
    live_gaps = {
        (m.filename, filename_quant_gap(m))
        for m in ModelBrowser.CURATED_MODELS
        if filename_quant_gap(m) is not None
    }
    if gaps != live_gaps:
        problems.append(f"filename gaps drifted: {sorted(live_gaps)}")

    for pid, quant in profile_quants().items():
        if quant not in KNOWN_QUANTS:
            problems.append(f"profile {pid} recommends unknown quant {quant}")

    if any(row["measured_in_tree"] for row in LOSS_CLAIMS):
        problems.append("a loss claim claims in-tree measurement — TF-067 has not banked")

    return problems
