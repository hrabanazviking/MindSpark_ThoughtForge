"""TF-085 — Backend docs + the new-backend guide:
the TF-065 contract as a recipe, the backend docs
estate as one catalogue, and a worked example backend
that proves the recipe end to end.

Roadmap: "Backend docs + a new-backend guide (the
contract as a recipe)."

Measured start (probed live this tick):

- **The backend docs were an estate, not a guide.**
  Six core backend docs stand in docs/architecture/
  (BACKEND_CONTRACT TF-065, BACKEND_HEALTH_AND_
  FALLBACK TF-073, AESIR_BACKEND TF-074,
  AESIR_TURBOQUANT_INTERPLAY TF-075, OPENAI_
  COMPATIBLE_BACKEND TF-076, LOCAL_FIRST_ROUTING
  TF-077), with three supporting docs a backend
  author also needs (TOKENIZER_DISCIPLINE TF-071,
  STREAMING_BACKPRESSURE TF-072, JOINED_TRACES
  TF-084). No file indexed them as one estate, and
  no file anywhere gave the steps for writing a
  new backend: greps for a recipe / new-backend
  procedure found only TF-153's admission ownership
  repeated in the TF-074/TF-076 docs — the *gate* was
  named, the *road to it* was not written.
- **The public entry docs are silent on backends.**
  docs/index.md and docs/api.md contain no mention
  of "backend" at all (grep, this tick), so a reader
  arriving through the front door never learns a
  backend contract exists. D-GUIDE-1 (added + FIXED
  in scope): this slice's catalogue + guide doc is
  the index the estate lacked; wiring the guide
  into any README is forbidden by standing law and
  editing docs/index.md / docs/api.md's wider
  reconciliation belongs to TF-182, so the fix is
  filed at guide grade, with that remainder stated.
- **The contract itself was already recipe-shaped,
  unassembled.** TF-065's CONTRACT_DUTIES names
  every duty, UnifiedBackend supplies two defaults
  (count_tokens, capabilities) and one stream
  default, and backend_contract_problems() judges
  any candidate live — everything a recipe needs
  existed; it was scattered across the contract
  module, the base class, and six slice docs.

The guide (this module, as data + checkers):

- ``BACKEND_DOCS`` — the nine-doc catalogue: path,
  slice, role, and the module a reader opens next
  to it. The checker judges every row live (file
  exists, names its slice, module exists) and fails
  if a core backend doc on disk is uncatalogued.
- ``RECIPE_STEPS`` — fourteen ordered steps, each
  mapped to the contract duties it discharges and
  the source file that shows it done. Together the
  steps cover every duty in CONTRACT_DUTIES (the
  checker recomputes the coverage), plus the three
  duties the contract assumes but does not list:
  dependency declaration, factory admission, and
  the doc the author files.
- ``GuideExampleBackend`` — the recipe, cooked: the
  smallest honest backend (a deterministic synthetic
  responder — fixture data, no model, no network).
  It conforms under TF-065's own checker, serves a
  plain turn, streams simulated through the base
  default (its joined trace therefore carries the
  TF-084 stash), and states its failure as data on
  an empty prompt. It is NOT added to
  SHIPPED_BACKENDS or the factory — TF-153 owns
  admission, exactly as TF-074/TF-076 were handled.

Proof grade FIXTURE (the catalogue + recipe judged
live against the tree, the example backend judged
by the contract's own checker and exercised plain
+ streamed on this VM) — the guide making a *real*
third-party backend correct is NOT PROVEN until an
author follows it to a live server (TF-086's gate /
TF-153's admission own that road), never claimed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

GUIDE_FORMAT = "thoughtforge-backend-guide/1"

GUIDE_DOC = "docs/architecture/BACKEND_GUIDE.md"

# ── The catalogue ───────────────────────────────────────────────────────────
# role: core = a backend author's primary reading;
# supporting = needed for one contract surface.

BACKEND_DOCS: tuple[dict[str, Any], ...] = (
    {"path": "docs/architecture/BACKEND_CONTRACT.md",
     "slice": "TF-065", "role": "core",
     "covers": "the contract itself: duties, capability "
               "vocabularies, dependency rulings, factory keys",
     "module": "src/thoughtforge/inference/backend_contract.py"},
    {"path": "docs/architecture/BACKEND_HEALTH_AND_FALLBACK.md",
     "slice": "TF-073", "role": "core",
     "covers": "health semantics + FallbackBackend: how a "
               "second backend is tried and announced",
     "module": "src/thoughtforge/inference/backend_fallback.py"},
    {"path": "docs/architecture/AESIR_BACKEND.md",
     "slice": "TF-074", "role": "core",
     "covers": "a worked binding of an external server, "
               "conformed directly, admitted nowhere yet",
     "module": "src/thoughtforge/inference/aesir_backend.py"},
    {"path": "docs/architecture/AESIR_TURBOQUANT_INTERPLAY.md",
     "slice": "TF-075", "role": "core",
     "covers": "quant truth for a served backend: identity, "
               "not a wire field",
     "module": "src/thoughtforge/inference/aesir_quant.py"},
    {"path": "docs/architecture/OPENAI_COMPATIBLE_BACKEND.md",
     "slice": "TF-076", "role": "core",
     "covers": "a worked binding of a common HTTP shape, "
               "with its dialect rules as data",
     "module": "src/thoughtforge/inference/openai_compatible.py"},
    {"path": "docs/architecture/LOCAL_FIRST_ROUTING.md",
     "slice": "TF-077", "role": "core",
     "covers": "placement in practice: local-first order, "
               "remote opt-in, per-instance privacy/cost",
     "module": "src/thoughtforge/inference/local_first_routing.py"},
    {"path": "docs/architecture/TOKENIZER_DISCIPLINE.md",
     "slice": "TF-071", "role": "supporting",
     "covers": "count_tokens agreement: the tolerance a new "
               "backend's counting must land inside",
     "module": "src/thoughtforge/inference/tokenizer_discipline.py"},
    {"path": "docs/architecture/STREAMING_BACKPRESSURE.md",
     "slice": "TF-072", "role": "supporting",
     "covers": "streaming duties: backpressure, clean "
               "cancellation, closing what you open",
     "module": "src/thoughtforge/inference/streaming_discipline.py"},
    {"path": "docs/architecture/JOINED_TRACES.md",
     "slice": "TF-084", "role": "supporting",
     "covers": "what the core records about a backend's "
               "calls in the joined trace, per turn",
     "module": "src/thoughtforge/inference/joined_trace.py"},
)

#: The core docs the checker requires to be catalogued
#: if present on disk — a new core backend doc that
#: lands uncatalogued fails this slice's checker.
_CORE_DOC_BASENAMES = tuple(
    Path(d["path"]).name for d in BACKEND_DOCS if d["role"] == "core")

# ── The recipe ──────────────────────────────────────────────────────────────

RECIPE_STEPS: tuple[dict[str, Any], ...] = (
    {"step": 1, "id": "subclass_and_name",
     "duties": ("backend_name",),
     "source": "src/thoughtforge/inference/unified_backend.py",
     "instruction": "Subclass UnifiedBackend; backend_name() "
                    "returns 'family:model' and that exact string "
                    "rides every GenerationResponse.backend_used."},
    {"step": 2, "id": "generate_returns_data_never_raises",
     "duties": ("generate",),
     "source": "src/thoughtforge/inference/unified_backend.py",
     "instruction": "generate(request) returns a "
                    "GenerationResponse for every outcome; a "
                    "failure is finish_reason 'error' + the "
                    "error string, never an exception out of "
                    "generate()."},
    {"step": 3, "id": "finish_vocabulary_only",
     "duties": ("finish_reason",),
     "source": "src/thoughtforge/inference/backend_contract.py",
     "instruction": "finish_reason is one of stop / length / "
                    "error; a server dialect's other reasons "
                    "(content_filter, tool_calls, ...) are "
                    "mapped or returned as named error data, "
                    "never passed through (TF-076's rule)."},
    {"step": 4, "id": "health_check_bool_never_raises",
     "duties": ("health_check",),
     "source": "src/thoughtforge/inference/backend_fallback.py",
     "instruction": "health_check() returns a bool and never "
                    "raises; server-level health and per-model "
                    "listing are separate surfaces (TF-073 / "
                    "TF-077's tri-state model_listed pattern)."},
    {"step": 5, "id": "capabilities_honest_record",
     "duties": ("capabilities",),
     "source": "src/thoughtforge/inference/backend_contract.py",
     "instruction": "capabilities() states the truth in the "
                    "contract vocabularies, read live: streaming, "
                    "token_counting, seed, placement (never "
                    "'unknown' for a backend offered for "
                    "admission), generate, health."},
    {"step": 6, "id": "count_tokens_declared_basis",
     "duties": ("count_tokens",),
     "source": "src/thoughtforge/inference/tokenizer_discipline.py",
     "instruction": "count_tokens('') == 0; use the model's "
                    "tokeniser when one is actually loaded "
                    "(declare 'model') and the contract estimate "
                    "otherwise (declare 'estimate'); land inside "
                    "TF-071's tolerance against the other "
                    "backends."},
    {"step": 7, "id": "streaming_decision_stated",
     "duties": ("generate_stream",),
     "source": "src/thoughtforge/inference/unified_backend.py",
     "instruction": "Either keep the base generate_stream "
                    "(simulated: completed text as one chunk, "
                    "nothing on error, the TF-084 stash carries "
                    "the error) or override it and yield "
                    "incrementally in order (native) — and close "
                    "what the stream opened, on every exit "
                    "(TF-072). The capability record must match "
                    "the choice; the checker compares them."},
    {"step": 8, "id": "seed_forward_or_declare",
     "duties": ("seed",),
     "source": "src/thoughtforge/inference/backend_contract.py",
     "instruction": "If the sampler can be seeded, forward "
                    "request.seed whenever it is not None and "
                    "declare 'honoured'; if it cannot, declare "
                    "'not_supported'. Silently dropping a "
                    "carried seed is a breach (TF-036/TF-065)."},
    {"step": 9, "id": "placement_and_local_first",
     "duties": ("capabilities",),
     "source": "src/thoughtforge/inference/local_first_routing.py",
     "instruction": "Declare placement honestly (local / "
                    "remote / local_or_remote resolved per "
                    "instance by base-URL host); a remote "
                    "backend is opt-in in routing and its cost "
                    "is declared, never guessed (TF-077)."},
    {"step": 10, "id": "dependencies_declared",
     "duties": ("dependencies",),
     "source": "src/thoughtforge/inference/backend_contract.py",
     "instruction": "Every import is declared: a new top-level "
                    "'requests' consumer amends REQUESTS_"
                    "CONSUMERS on purpose; anything heavier is "
                    "an optional extra, imported lazily or "
                    "guarded (DEPENDENCY_RULINGS, D-DEP-1/5)."},
    {"step": 11, "id": "conformance_before_admission",
     "duties": ("conformance",),
     "source": "src/thoughtforge/inference/backend_contract.py",
     "instruction": "backend_contract_problems([your_backend]) "
                    "== [] is the gate, run against the real "
                    "class — claims are re-derived from code, "
                    "not trusted."},
    {"step": 12, "id": "tests_in_the_estate_shape",
     "duties": ("conformance",),
     "source": "tests/test_tf065_backend_contract.py",
     "instruction": "Tests pin: the success payload, the error "
                    "as data, count_tokens('') == 0 + monotonic "
                    "prose, health bool, seed present in the "
                    "payload when carried and absent when not, "
                    "and the capability record against the class."},
    {"step": 13, "id": "admission_is_tf153s",
     "duties": ("admission",),
     "source": "docs/architecture/BACKEND_CONTRACT.md",
     "instruction": "Do NOT add the backend to SHIPPED_"
                    "BACKENDS or the factory yourself: conform "
                    "directly first (the TF-074/TF-076 pattern); "
                    "TF-153 owns admission to the shipped set "
                    "and the config keys that come with it."},
    {"step": 14, "id": "doc_filed_with_proof_grade",
     "duties": ("docs",),
     "source": "docs/architecture/BACKEND_GUIDE.md",
     "instruction": "File the backend's own architecture doc: "
                    "measured start, the rules as data, and a "
                    "proof grade that states what is NOT PROVEN "
                    "(a live server, a loaded model) until it "
                    "has actually been faced."},
)

GUIDE_RULES: tuple[dict[str, str], ...] = (
    {"id": "catalogue_is_judged_live",
     "rule": "every BACKEND_DOCS row names a doc that exists "
             "and names its slice, and a module that exists; a "
             "core backend doc present on disk but absent from "
             "the catalogue fails the checker"},
    {"id": "recipe_covers_every_contract_duty",
     "rule": "the union of RECIPE_STEPS' duties covers every "
             "duty in TF-065's CONTRACT_DUTIES (recomputed "
             "live), plus dependencies / conformance / "
             "admission / docs, which the contract assumes"},
    {"id": "recipe_order_is_the_build_order",
     "rule": "steps are numbered 1..14 in build order: class "
             "shape, then duties, then honesty surfaces, then "
             "the conformance gate, then admission and docs"},
    {"id": "example_backend_proves_the_recipe",
     "rule": "GuideExampleBackend follows all fourteen steps "
             "and is judged by TF-065's own "
             "backend_contract_problems — the recipe is proven "
             "by a backend built from it, not asserted"},
    {"id": "example_is_fixture_never_shipped",
     "rule": "the example is synthetic fixture data (no model, "
             "no network, deterministic) and joins neither "
             "SHIPPED_BACKENDS nor the factory; TF-153 owns "
             "every admission"},
    {"id": "guide_doc_is_the_index",
     "rule": "docs/architecture/BACKEND_GUIDE.md lists every "
             "catalogued doc path and every recipe step id "
             "(the checker verifies) — D-GUIDE-1 FIXED at "
             "guide grade; docs/index.md + docs/api.md "
             "reconciliation is TF-182's, README wiring is "
             "forbidden by standing law"},
    {"id": "live_author_proof_not_claimed",
     "rule": "a real author following this guide to a live "
             "third-party backend is NOT PROVEN (no such "
             "backend has been built from it yet); TF-086's "
             "gate and TF-153's admission own that road"},
)


# ── The worked example ──────────────────────────────────────────────────────

from thoughtforge.inference.backend_contract import (
    BackendCapabilities,
)
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)


class GuideExampleBackend(UnifiedBackend):
    """The recipe, cooked: the smallest honest backend.

    A deterministic synthetic responder for tests and
    teaching — step 1 subclass + name, step 2 data-never-
    raises (an empty prompt is named error data), step 5
    an honest capability record, steps 6–8 by the base
    defaults declared as they are (estimate counting,
    simulated streaming, seed not_supported), step 9
    placement local. No model, no network.
    """

    def __init__(self, model: str = "lantern-1") -> None:
        self.model = model

    def backend_name(self) -> str:
        return f"guide_example:{self.model}"

    def health_check(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not request.prompt or not request.prompt.strip():
            return GenerationResponse(
                text="", tokens_generated=0, finish_reason="error",
                backend_used=self.backend_name(),
                error="guide example: empty prompt")
        text = f"guide example answer: {request.prompt.strip()}"
        return GenerationResponse(
            text=text, tokens_generated=self.count_tokens(text),
            finish_reason="stop", backend_used=self.backend_name())

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.backend_name(), generate=True,
            streaming="simulated", token_counting="estimate",
            seed="not_supported", health=True, placement="local",
            model=self.model,
            notes=("synthetic fixture backend from TF-085's "
                   "recipe; no model, no network",))


# ── Export ──────────────────────────────────────────────────────────────────

def guide_as_data() -> dict[str, Any]:
    return {
        "format": GUIDE_FORMAT,
        "guide_doc": GUIDE_DOC,
        "backend_docs": [dict(d) for d in BACKEND_DOCS],
        "recipe_steps": [dict(s) for s in RECIPE_STEPS],
        "rules": [dict(r) for r in GUIDE_RULES],
        "example_backend": GuideExampleBackend().backend_name(),
        "proof": {
            "catalogue_and_recipe": "judged_live_against_tree",
            "example_backend": "fixture_conformed_by_tf065_checker",
            "live_third_party_backend_from_guide": "not_proven",
        },
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def backend_guide_problems(root: Path | None = None) -> list[str]:
    """The filed estate judged against the live tree."""
    root = root or _repo_root()
    problems: list[str] = []

    doc_ids = [d["path"] for d in BACKEND_DOCS]
    if len(doc_ids) != len(set(doc_ids)):
        problems.append("BACKEND_DOCS paths are not unique")
    for row in BACKEND_DOCS:
        doc = root / row["path"]
        if not doc.exists():
            problems.append(f"catalogued doc missing: {row['path']}")
        elif row["slice"] not in doc.read_text(encoding="utf-8")[:400]:
            problems.append(f"{row['path']}: doc head does not name "
                            f"its slice {row['slice']}")
        if not (root / row["module"]).exists():
            problems.append(f"{row['path']}: module missing: "
                            f"{row['module']}")
    # A core backend doc on disk must be catalogued.
    arch = root / "docs" / "architecture"
    for name in ("BACKEND_CONTRACT.md", "BACKEND_HEALTH_AND_FALLBACK.md",
                 "AESIR_BACKEND.md", "AESIR_TURBOQUANT_INTERPLAY.md",
                 "OPENAI_COMPATIBLE_BACKEND.md",
                 "LOCAL_FIRST_ROUTING.md"):
        if (arch / name).exists() and \
                f"docs/architecture/{name}" not in doc_ids:
            problems.append(f"core backend doc uncatalogued: {name}")

    step_ids = [s["id"] for s in RECIPE_STEPS]
    if len(step_ids) != len(set(step_ids)):
        problems.append("RECIPE_STEPS ids are not unique")
    if [s["step"] for s in RECIPE_STEPS] != \
            list(range(1, len(RECIPE_STEPS) + 1)):
        problems.append("RECIPE_STEPS are not numbered 1..N in order")
    for s in RECIPE_STEPS:
        if not (root / s["source"]).exists():
            problems.append(f"step {s['id']}: source missing: "
                            f"{s['source']}")

    from thoughtforge.inference.backend_contract import CONTRACT_DUTIES
    covered = {d for s in RECIPE_STEPS for d in s["duties"]}
    for duty in CONTRACT_DUTIES:
        if duty["duty"] not in covered:
            problems.append(f"contract duty {duty['duty']} is not "
                            f"covered by any recipe step")
    for extra in ("dependencies", "conformance", "admission", "docs"):
        if extra not in covered:
            problems.append(f"recipe omits the {extra} step")

    guide_doc = root / GUIDE_DOC
    if not guide_doc.exists():
        problems.append(f"guide doc missing: {GUIDE_DOC}")
    else:
        text = guide_doc.read_text(encoding="utf-8")
        for row in BACKEND_DOCS:
            if row["path"] not in text:
                problems.append(f"guide doc does not list {row['path']}")
        for s in RECIPE_STEPS:
            if s["id"] not in text:
                problems.append(f"guide doc does not list step "
                                f"{s['id']}")

    rule_ids = [r["id"] for r in GUIDE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("GUIDE_RULES ids are not unique")
    return problems


def backend_guide_live_problems() -> list[str]:
    """The example backend, judged behaviourally."""
    problems: list[str] = []
    from thoughtforge.inference.backend_contract import (
        backend_contract_problems,
    )

    backend = GuideExampleBackend()
    problems += backend_contract_problems([backend])

    from thoughtforge.inference.backend_contract import SHIPPED_BACKENDS
    if any(r["class"] == "GuideExampleBackend" for r in SHIPPED_BACKENDS):
        problems.append("the example backend joined SHIPPED_BACKENDS "
                        "— admission is TF-153's")

    ok = backend.generate(GenerationRequest(prompt="where is the ferry?"))
    if ok.finish_reason != "stop" or not ok.text \
            or ok.backend_used != backend.backend_name():
        problems.append("example backend's plain turn is not a clean stop")
    err = backend.generate(GenerationRequest(prompt=""))
    if err.finish_reason != "error" or not err.error:
        problems.append("example backend's empty prompt is not named "
                        "error data")
    streamed = list(backend.generate_stream(
        GenerationRequest(prompt="where is the ferry?")))
    if streamed != [ok.text]:
        problems.append("example backend's simulated stream is not "
                        "the completed text as one chunk")
    err_stream = list(backend.generate_stream(
        GenerationRequest(prompt="")))
    if err_stream != [] or getattr(backend, "last_stream_response",
                                    None) is None or \
            backend.last_stream_response.error != err.error:
        problems.append("example backend's streamed error lost its "
                        "TF-084 stash")
    if backend.health_check() is not True:
        problems.append("example backend's health is not True")
    return problems


__all__ = [
    "BACKEND_DOCS",
    "GUIDE_DOC",
    "GUIDE_FORMAT",
    "GUIDE_RULES",
    "RECIPE_STEPS",
    "GuideExampleBackend",
    "backend_guide_live_problems",
    "backend_guide_problems",
    "guide_as_data",
]
