"""Backend contract v1 (TF-065) — what a backend must provide.

Before TF-065 the four shipped backends shared only the thin
``UnifiedBackend`` ABC (``generate`` / ``health_check`` /
``backend_name`` plus TF-026's default ``generate_stream``).
Probed this slice, none of the contract questions the roadmap
names had an answer anywhere in the tree:

- **Capability truth** — no backend declared anything: no
  ``capabilities()``, no ``count_tokens()`` on the base or on
  any backend (probe: all four ``has_capabilities == False``,
  ``has_count_tokens == False``).
- **Seed** — TF-036 put ``seed`` on ``GenerationRequest`` and
  left honouring it as "TF-065's contract duty". Only the
  TurboQuant wrapper forwarded it; the ollama / lmstudio / hf
  ``generate()`` sources contained no seed at all, so a seeded
  (deterministic-mode) turn silently ran unseeded on three of
  four backends.
- **Streaming** — no backend overrode ``generate_stream``, so
  every backend streamed *simulated* (completed text re-chunked
  by the core) — including TurboQuant, whose engine has a
  native streaming generator the wrapper never exposed.
- **Tokenise** — no backend could count tokens. The HF backend
  reported ``tokens_generated`` as ``len(text.split())`` — a
  word count wearing a token count's name, declared nowhere;
  the TurboQuant engine's ``estimate_tokens`` (model tokeniser
  when loaded, ~4-chars heuristic otherwise) was unreachable
  through the wrapper.
- **Dependencies (D-DEP-1 / D-DEP-5)** — ``requests`` was a
  top-level, unguarded import in four inference modules and
  declared nowhere in ``pyproject.toml`` (it worked only
  transitively, TF-008); transformers / huggingface_hub /
  onnxruntime / optimum / psutil were likewise declared
  nowhere, though every one of their import sites is lazy or
  guarded.

This module is the contract, as data + checkers:

- ``CONTRACT_DUTIES`` — the duties, each with its rule.
- ``BackendCapabilities`` — the capability record every
  backend must be able to state about itself, with the
  vocabularies (``STREAMING_MODES`` / ``TOKEN_COUNTING_MODES``
  / ``SEED_MODES`` / ``PLACEMENTS``) its fields must use.
  A capability is a *claim*: the checkers below compare the
  claim against the code (a backend declaring ``native``
  streaming must actually override ``generate_stream``; one
  declaring ``honoured`` seed must forward the seed in its
  generate path).
- ``estimate_tokens`` — the contract estimate: the tree's
  standing ~4-chars-per-token heuristic (the same one
  ``TurboQuantEngine.estimate_tokens`` falls back to — by
  delegation since TF-071, which also made the estimate
  script-aware: CJK/kana/Hangul characters count one token
  each, after a real BPE tokeniser measured the flat
  heuristic 0.88-relative-error wrong on CJK text), named
  as an estimate wherever it is used.
- Dependency rulings (``DEPENDENCY_RULINGS``): ``requests``
  is DECLARED as a core dependency (the backends import it
  unconditionally, so a minimal install must carry it);
  transformers / huggingface_hub / onnxruntime / optimum /
  psutil stay OPTIONAL — every import site lazy or guarded,
  pinned by an AST checker — and are declared as optional
  extras so an installer can ask for them by name.
- ``BACKEND_CONFIG_KEYS`` — the factory's live config keys
  (D-CFG-3's backend share): exactly the keys
  ``load_backend_from_config`` reads, pinned against its
  source; the dead keys (``profile`` / ``chat_*`` /
  ``knowledge_db_built``) are not backend keys and the
  factory must never grow a read of them (TF-153 owns the
  file's final shape).

Conformance wired this slice: ``UnifiedBackend`` gains the
default ``count_tokens`` (the contract estimate) and
``capabilities`` (the conservative default — simulated
streaming unless the class overrides ``generate_stream``,
estimate counting, seed ``not_supported``); each shipped
backend declares its own record and is conformed where the
probe found drift — seed forwarded by ollama (``options.seed``),
lmstudio (the OpenAI-shaped ``seed`` field) and hf (client +
raw-request parameters); TurboQuant's wrapper exposes the
engine's native ``generate_stream`` and counts through the
engine when it is loaded (``model``), the contract estimate
until then (``estimate`` — its capability record is read
live, so it never claims the model tokeniser before load).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONTRACT_FORMAT = "thoughtforge-backend-contract/1"

STREAMING_MODES = ("native", "simulated", "none")
TOKEN_COUNTING_MODES = ("model", "estimate", "none")
SEED_MODES = ("honoured", "not_supported")
PLACEMENTS = ("local", "remote", "local_or_remote", "unknown")
FINISH_REASONS = ("stop", "length", "error")

# The duties, as data. `required` duties are owed by every
# backend; a backend that cannot meet one says so in its
# capability record, in the vocabulary — never silently.
CONTRACT_DUTIES: tuple[dict[str, Any], ...] = (
    {
        "duty": "generate",
        "required": True,
        "rule": "generate(request) returns a GenerationResponse; "
                "failures are data (finish_reason 'error' + the "
                "error string), never a raise out of generate()",
    },
    {
        "duty": "generate_stream",
        "required": True,
        "rule": "generate_stream(request) yields text chunks; the "
                "base default yields the completed generate() text "
                "as one chunk (nothing on error) and is labelled "
                "simulated — a backend overriding it is native and "
                "must yield incrementally in order",
    },
    {
        "duty": "count_tokens",
        "required": True,
        "rule": "count_tokens(text) returns the backend's own count: "
                "the model's tokeniser when the backend has one "
                "loaded ('model'), otherwise the contract estimate, "
                "declared as 'estimate' in the capability record; "
                "an empty string counts 0",
    },
    {
        "duty": "capabilities",
        "required": True,
        "rule": "capabilities() states the backend's truth in the "
                "contract vocabularies, read live (a property that "
                "changes with load state is reported as it stands "
                "at call time); claiming native streaming without "
                "overriding generate_stream, or honoured seed "
                "without forwarding it, is a contract breach the "
                "checkers catch",
    },
    {
        "duty": "seed",
        "required": False,
        "rule": "a backend that can seed its sampler MUST forward "
                "GenerationRequest.seed when it is not None "
                "(TF-036's duty, assigned here); one that cannot "
                "declares seed 'not_supported' — silently dropping "
                "a carried seed is a breach",
    },
    {
        "duty": "health_check",
        "required": True,
        "rule": "health_check() returns a bool and never raises; "
                "False means the backend cannot serve now",
    },
    {
        "duty": "backend_name",
        "required": True,
        "rule": "backend_name() identifies the backend + model "
                "('family:model'); GenerationResponse.backend_used "
                "carries it on every response, success or error",
    },
    {
        "duty": "finish_reason",
        "required": True,
        "rule": "finish_reason is one of FINISH_REASONS "
                "('stop' / 'length' / 'error'); 'error' always "
                "carries the error string",
    },
)

# The shipped backends, as data: factory id -> (module, class).
SHIPPED_BACKENDS: tuple[dict[str, str], ...] = (
    {"id": "ollama", "module": "thoughtforge.inference.ollama_backend",
     "class": "OllamaBackend"},
    {"id": "lmstudio", "module": "thoughtforge.inference.lmstudio_backend",
     "class": "LMStudioBackend"},
    {"id": "huggingface", "module": "thoughtforge.inference.hf_backend",
     "class": "HuggingFaceBackend"},
    {"id": "turboquant", "module": "thoughtforge.inference.turboquant_backend",
     "class": "TurboQuantBackend"},
)

# The factory's live config keys (D-CFG-3, backend share) —
# exactly the cfg.get(...) names in load_backend_from_config.
BACKEND_CONFIG_KEYS: tuple[str, ...] = (
    "backend",
    "gguf_model_path",
    "hf_model",
    "hf_token",
    "lmstudio_model",
    "lmstudio_url",
    "ollama_model",
    "ollama_url",
    "openai_api_key",
    "openai_base_url",
    "openai_model",
)

# Dependency rulings (D-DEP-1 / D-DEP-5). `requests` is a hard
# import in exactly these modules — the set is pinned by AST.
REQUESTS_CONSUMERS: tuple[str, ...] = (
    "src/thoughtforge/inference/aesir_backend.py",
    "src/thoughtforge/inference/hf_backend.py",
    "src/thoughtforge/inference/lmstudio_backend.py",
    "src/thoughtforge/inference/model_browser.py",
    "src/thoughtforge/inference/ollama_backend.py",
    "src/thoughtforge/inference/openai_compatible.py",
)

DEPENDENCY_RULINGS: tuple[dict[str, Any], ...] = (
    {"package": "requests", "ruling": "core",
     "extra": None,
     "rule": "top-level unguarded import in the six "
             "REQUESTS_CONSUMERS modules -> declared in "
             "[project] dependencies (D-DEP-1 FIXED by "
             "declaration; the set was four at TF-065, "
             "grew to five at TF-074 when the Aesir backend "
             "joined as an announced consumer, and to six "
             "at TF-076 when the OpenAI-compatible backend "
             "joined the same way)"},
    {"package": "huggingface_hub", "ruling": "optional",
     "extra": "hub",
     "rule": "lazy in-function imports only (hf_backend, "
             "model_browser); absence degrades to the raw "
             "requests path"},
    {"package": "transformers", "ruling": "optional",
     "extra": "hub",
     "rule": "lazy in-function imports only (onnx_export); "
             "also carried by the 'onnx' extra"},
    {"package": "onnxruntime", "ruling": "optional",
     "extra": "onnx",
     "rule": "one guarded import in onnx_export, which "
             "raises a named ImportError when absent"},
    {"package": "optimum", "ruling": "optional",
     "extra": "onnx",
     "rule": "lazy in-function imports only (onnx_export)"},
    {"package": "psutil", "ruling": "optional",
     "extra": "system",
     "rule": "try-guarded imports only (profiles, wikidata); "
             "both sites fall back without it"},
)

OPTIONAL_PACKAGES = tuple(
    r["package"] for r in DEPENDENCY_RULINGS if r["ruling"] == "optional"
)


#: Character ranges whose scripts tokenise at roughly one
#: token per character in real BPE / SentencePiece vocabularies
#: (CJK ideographs, kana, Hangul syllables, CJK compat
#: ideographs). TF-071 measured the pre-fix estimate — a flat
#: ``len // 4`` over all characters — against a real BPE
#: tokeniser: Latin prose landed within ~25%, but an 11-char
#: CJK sentence counted 2 against the tokeniser's 16 (a 0.88
#: relative error), because the heuristic silently assumed a
#: Latin-script character density. The estimate is therefore
#: script-aware: script-dense characters count 1 token each,
#: the rest keep the ~4-chars-per-token heuristic. Latin-only
#: text is counted exactly as before TF-071.
SCRIPT_TOKEN_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
)


def _is_script_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in SCRIPT_TOKEN_RANGES)


def estimate_tokens(text: str) -> int:
    """The contract estimate: ~4 chars per token (one token
    per script-dense character — TF-071), 0 for empty.

    The same heuristic ``TurboQuantEngine.estimate_tokens``
    falls back to when no model is loaded (the engine now
    delegates here, so the heuristic has exactly one home).
    It is an estimate wherever it is used and is never
    reported as a model count.
    """
    if not text:
        return 0
    script_chars = sum(1 for ch in text if _is_script_char(ch))
    rest = len(text) - script_chars
    return max(1, script_chars + rest // 4)


@dataclass(frozen=True)
class BackendCapabilities:
    """One backend's capability truth, in contract vocabulary."""

    name: str = ""
    generate: bool = True
    streaming: str = "simulated"
    token_counting: str = "estimate"
    seed: str = "not_supported"
    health: bool = True
    placement: str = "unknown"
    model: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CONTRACT_FORMAT,
            "name": self.name,
            "generate": self.generate,
            "streaming": self.streaming,
            "token_counting": self.token_counting,
            "seed": self.seed,
            "health": self.health,
            "placement": self.placement,
            "model": self.model,
            "notes": list(self.notes),
        }


def default_capabilities(backend: Any) -> BackendCapabilities:
    """The conservative default for a backend declaring nothing.

    Streaming is derived from the class itself: native only
    when the class overrides the base ``generate_stream``.
    Everything else states the floor the base class provides.
    """
    from thoughtforge.inference.unified_backend import UnifiedBackend

    native = type(backend).generate_stream is not UnifiedBackend.generate_stream
    return BackendCapabilities(
        name=backend.backend_name(),
        streaming="native" if native else "simulated",
        notes=("default contract capabilities — the backend "
               "declares no record of its own",),
    )


def capabilities_problems(caps: dict[str, Any]) -> list[str]:
    """Judge one capability record (a ``to_dict()``) as data."""
    problems: list[str] = []
    if caps.get("format") != CONTRACT_FORMAT:
        problems.append(f"format is not {CONTRACT_FORMAT}")
    if not caps.get("name"):
        problems.append("capability record names no backend")
    if caps.get("generate") is not True:
        problems.append("generate duty is required by the contract")
    if caps.get("health") is not True:
        problems.append("health_check duty is required by the contract")
    if caps.get("streaming") not in STREAMING_MODES:
        problems.append(f"streaming {caps.get('streaming')!r} not in vocabulary")
    if caps.get("token_counting") not in TOKEN_COUNTING_MODES:
        problems.append(
            f"token_counting {caps.get('token_counting')!r} not in vocabulary")
    if caps.get("seed") not in SEED_MODES:
        problems.append(f"seed {caps.get('seed')!r} not in vocabulary")
    if caps.get("placement") not in PLACEMENTS:
        problems.append(f"placement {caps.get('placement')!r} not in vocabulary")
    return problems


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def dependency_problems(root: Path | None = None) -> list[str]:
    """D-DEP-1 / D-DEP-5, judged live against pyproject + source.

    - ``requests`` is declared in the core dependencies, and
      the set of modules importing it at top level is exactly
      ``REQUESTS_CONSUMERS``.
    - Every optional package is declared in its named extra
      and imported at top level NOWHERE in ``src/`` (lazy or
      guarded sites only).
    """
    root = root or _repo_root()
    problems: list[str] = []
    text = (root / "pyproject.toml").read_text(encoding="utf-8")

    core_block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    if '"requests' not in core_block:
        problems.append("requests is not declared in core dependencies")

    extras_block = text.split("[project.optional-dependencies]", 1)[1]
    for ruling in DEPENDENCY_RULINGS:
        if ruling["ruling"] != "optional":
            continue
        extra = ruling["extra"]
        if f'{extra} = [' not in extras_block:
            problems.append(f"optional extra {extra!r} is not declared")
        else:
            block = extras_block.split(f"{extra} = [", 1)[1].split("]", 1)[0]
            if f'"{ruling["package"]}' not in block:
                problems.append(
                    f"{ruling['package']} is not declared in extra {extra!r}")

    src = root / "src"
    requests_consumers: list[str] = []
    for path in sorted(src.rglob("*.py")):
        top = _top_level_imports(path)
        rel = path.relative_to(root).as_posix()
        if "requests" in top:
            requests_consumers.append(rel)
        for pkg in OPTIONAL_PACKAGES:
            if pkg in top:
                problems.append(f"{pkg} imported at top level in {rel}")
    if tuple(requests_consumers) != REQUESTS_CONSUMERS:
        problems.append(
            f"requests top-level consumers {requests_consumers} "
            f"!= contract set {list(REQUESTS_CONSUMERS)}")
    return problems


def factory_problems() -> list[str]:
    """The factory reads exactly BACKEND_CONFIG_KEYS (D-CFG-3)."""
    import inspect

    from thoughtforge.inference import unified_backend

    problems: list[str] = []
    tree = ast.parse(inspect.getsource(unified_backend))
    # cfg.get reads in the factory function's own body only.
    factory = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "load_backend_from_config")
    read = set()
    for node in ast.walk(factory):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    read.add(arg.value)
    if read != set(BACKEND_CONFIG_KEYS):
        problems.append(
            f"factory config keys {sorted(read)} != "
            f"contract keys {sorted(BACKEND_CONFIG_KEYS)}")
    return problems


def shipped_backends() -> list[Any]:
    """Instantiate the four shipped backends, cheaply.

    Construction is side-effect-free by each class's own
    design (HF builds at most an InferenceClient; TurboQuant
    loads lazily), and the TurboQuant instance points at a
    deliberately absent model file.
    """
    import importlib

    instances: list[Any] = []
    for row in SHIPPED_BACKENDS:
        cls = getattr(importlib.import_module(row["module"]), row["class"])
        if row["id"] == "turboquant":
            instances.append(cls(model_path=Path("/nonexistent/tf065.gguf")))
        else:
            instances.append(cls())
    return instances


def backend_contract_problems(
    backends: list[Any] | None = None,
) -> list[str]:
    """Re-derive the whole contract live over the shipped set."""
    import inspect

    from thoughtforge.inference.unified_backend import UnifiedBackend

    problems: list[str] = []
    problems += dependency_problems()
    problems += factory_problems()

    duty_names = {d["duty"] for d in CONTRACT_DUTIES}
    for name in ("generate", "generate_stream", "count_tokens",
                 "capabilities", "seed", "health_check",
                 "backend_name", "finish_reason"):
        if name not in duty_names:
            problems.append(f"contract duty {name} missing from CONTRACT_DUTIES")

    for backend in (backends if backends is not None else shipped_backends()):
        label = backend.backend_name()
        if not isinstance(backend, UnifiedBackend):
            problems.append(f"{label}: not a UnifiedBackend")
            continue
        caps = backend.capabilities().to_dict()
        problems += [f"{label}: {p}" for p in capabilities_problems(caps)]
        if caps["name"] != label:
            problems.append(
                f"{label}: capability record names {caps['name']!r}")
        if caps["placement"] == "unknown":
            problems.append(f"{label}: shipped backend declares no placement")
        native = (type(backend).generate_stream
                  is not UnifiedBackend.generate_stream)
        if (caps["streaming"] == "native") != native:
            problems.append(
                f"{label}: streaming claim {caps['streaming']!r} "
                f"contradicts the class (override present: {native})")
        if caps["seed"] == "honoured":
            # Forwarding may live in generate() or in a helper
            # it calls (HF forwards on its client + raw legs),
            # so the claim is judged over the class source.
            src = inspect.getsource(type(backend))
            if "seed" not in src:
                problems.append(f"{label}: seed claimed honoured, not forwarded")
        # count_tokens behaviour: empty is 0, text is positive,
        # and a longer text never counts less (estimate + model
        # tokenisers both hold this on ordinary prose).
        if backend.count_tokens("") != 0:
            problems.append(f"{label}: count_tokens('') != 0")
        short = backend.count_tokens("the ferry leaves at dawn")
        long = backend.count_tokens("the ferry leaves at dawn " * 8)
        if not (0 < short <= long):
            problems.append(
                f"{label}: count_tokens not positive/monotone "
                f"({short} vs {long})")
        # health_check never raises and returns a bool.
        try:
            health = backend.health_check()
            if not isinstance(health, bool):
                problems.append(f"{label}: health_check returned {health!r}")
        except Exception as exc:  # noqa: BLE001 — the checker judges
            problems.append(f"{label}: health_check raised {exc!r}")
    return problems


__all__ = [
    "BACKEND_CONFIG_KEYS",
    "CONTRACT_DUTIES",
    "CONTRACT_FORMAT",
    "DEPENDENCY_RULINGS",
    "FINISH_REASONS",
    "OPTIONAL_PACKAGES",
    "PLACEMENTS",
    "REQUESTS_CONSUMERS",
    "SEED_MODES",
    "SHIPPED_BACKENDS",
    "STREAMING_MODES",
    "TOKEN_COUNTING_MODES",
    "BackendCapabilities",
    "SCRIPT_TOKEN_RANGES",
    "backend_contract_problems",
    "capabilities_problems",
    "default_capabilities",
    "dependency_problems",
    "estimate_tokens",
    "factory_problems",
    "shipped_backends",
]
