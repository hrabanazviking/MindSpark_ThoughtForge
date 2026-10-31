"""
Embedding discipline — models per profile, offline proof (TF-053).

The roadmap's rule: *the embeddings module's models per
profile, with offline availability proven per profile.*

The measured starting state, before this slice (probed
this tick, keeper 07:22):

  * **No profile named an embedding model at all.** The six
    hardware profiles carry ``model`` sections for the LLM
    and ``retrieval`` sections with vector top-k values,
    but no embedding assignment — every profile silently
    inherited the one global default,
    ``all-MiniLM-L6-v2`` (~91 MB, plus the torch stack
    sentence-transformers rides on), from
    ``etl/embeddings.DEFAULT_MODEL``,
    ``KnowledgeForge(model_name=...)``'s default, and
    ``configs/default.yaml``'s ``embedding_model`` key
    alike. A pi_zero with 512 MB of RAM was assigned, in
    effect, the same embedding stack as a server.
  * **The model's name was split in two.** The ETL, the
    forge, and the config spell it ``all-MiniLM-L6-v2``;
    ``inference/onnx_export`` spells it
    ``sentence-transformers/all-MiniLM-L6-v2``. Both
    spellings resolve to the same hub model, but the
    loader's cache is keyed by the string it is handed
    and the knowledge DB's ``model_name`` column is
    filtered by exact match — embeddings built under one
    spelling are invisible to a search under the other.
  * **Offline availability was assumed, never proven.**
    Nothing checked the package or the local cache before
    ``SentenceTransformer(model_name)`` was constructed;
    on an uncached machine that constructor reaches for
    the network, on a machine without the package it
    raises — and ``KnowledgeForge``'s own docstring
    claimed "Fully offline — no network calls" over the
    top of both facts. On this VM the package is absent
    (docket D-ENV-1) and the model is in no local cache,
    and the measured consequence was worse than a failed
    vector search: ``retrieve(path="hybrid")`` — the
    core's default path — *raised* ``RuntimeError`` out of
    the vector leg and took the working SQL leg down with
    it. Only the core's turn-level catch stood between
    that and a failed turn, and it still lost every SQL
    result the turn should have had.
  * **The dimension was a bare constant with one silent
    assumption.** ``EMBEDDING_DIM = 384`` is
    all-MiniLM-L6-v2's dimension; nothing tied the
    constant to the model, so a different model would
    have produced mismatched blobs with no check anywhere.
  * **The model's licence was recorded nowhere**
    (docket D-LIC-2's embedding share, TF-008 finding 7).

What this slice builds:

  * ``EMBEDDING_MODELS`` — the model registry as data:
    every embedding model the tree names (exactly one
    today), with its canonical name, hub id, dimension,
    size, and licence recorded from its upstream model
    card. ``canonical_model_name()`` folds every spelling
    of a registered model to the canonical name; the ETL
    loader, both embedding builders, both vector
    searches, and the forge now normalise through it, so
    the split spelling can no longer split the cache or
    the DB.
  * ``PROFILE_EMBEDDINGS`` — the per-profile assignment
    as data. sentence-transformers runtimes for the
    profiles that can carry the torch stack beside their
    LLM (server_gpu / desktop_gpu / desktop_cpu / pi_5,
    with batch sizes stepped down as RAM steps down);
    phone_low is assigned the same model through the
    ONNX runtime its own profile already declares
    (``deployment.onnx_export: true``); pi_zero is
    assigned **no embedding model** — 512 MB cannot
    honestly carry one beside anything else, so its
    vector leg is SQL-only by rule, not by accident.
    Every profile's fallback on unavailability is the
    same stated rule: degrade to SQL retrieval, announced
    in the log, never a raised error and never a silent
    empty.
  * ``availability_report(profile_id)`` — the offline
    proof, per profile, from filesystem and import facts
    only (it never loads a model and never touches the
    network): for a sentence-transformers profile, the
    package must be importable *and* the model must be
    present in a local cache (hub cache or legacy
    sentence-transformers home); for the ONNX profile, a
    model artefact must exist in the embeddings directory
    (``utils/paths.get_embeddings_dir()`` — docket
    D-DEAD-3's never-called helper, now the named home
    this report checks). The verdict is a status from
    ``AVAILABILITY_STATUSES`` with the reasons named —
    ``available_offline`` is claimed only when both facts
    hold.
  * **The forge degrades.** ``KnowledgeForge.
    vector_retrieve`` now catches a failing vector leg,
    logs it, and returns what it has; ``retrieve`` on the
    hybrid path keeps its SQL results. The class
    docstring's offline claim is corrected to the truth.
  * ``stored_embeddings_problems(db_path)`` — the stored
    estate checked against the registry: every embedding
    row's model name must be canonical for its model and
    every blob's length must be the registered dimension
    × 4 bytes (float32). TF-111 (embedding at ingestion)
    consumes this.
  * Docket D-DEAD-2 resolved by retirement: the scalar
    ``cosine_similarity`` in ``etl/embeddings.py`` had no
    caller anywhere (TF-011) while its batch sibling is
    the live path — it is removed, and the batch function
    is what this module's checker exercises.

Stated boundaries:

  * The batch sizes in ``PROFILE_EMBEDDINGS`` are the
    discipline's assignment; the builders still take
    their ``batch_size`` arguments from their callers
    (TF-111 wires ingestion to the assignment).
  * The registry records the one model the tree names.
    Registering a second model is a deliberate act —
    registry entry, dimension, licence, and a
    per-profile assignment — never a string change at a
    call site.
  * Availability is proven for *this* machine's caches at
    call time. Another machine proves its own by running
    the same report; no verdict travels.
  * On this VM today the honest verdict for every
    sentence-transformers profile is
    ``unavailable_package_missing`` (D-ENV-1 stands OPEN —
    the package is still not installed; this slice makes
    the system behave truthfully in its absence and prove
    the absence per profile, it does not install the
    torch stack), for phone_low
    ``unavailable_onnx_artefact_missing`` (no ONNX
    artefact has been exported into the embeddings
    directory), and for pi_zero ``not_assigned`` by rule.
    Vector/hybrid retrieval *quality* remains NOT PROVEN.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thoughtforge.utils.config import list_hardware_profiles
from thoughtforge.utils.paths import get_embeddings_dir

# ── The model registry ────────────────────────────────────────────────────────

EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "all-MiniLM-L6-v2": {
        "canonical_name": "all-MiniLM-L6-v2",
        "hub_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "parameters_m": 22.7,
        "size_mb": 91,
        "licence": "Apache-2.0",
        "licence_source": (
            "upstream model card, huggingface.co/"
            "sentence-transformers/all-MiniLM-L6-v2 "
            "(recorded TF-053; final licence pass TF-190)"
        ),
    },
}

#: Every spelling that names a registered model → its canonical name.
MODEL_ALIASES: dict[str, str] = {}
for _canonical, _row in EMBEDDING_MODELS.items():
    MODEL_ALIASES[_canonical] = _canonical
    MODEL_ALIASES[_row["hub_id"]] = _canonical

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Runtimes, statuses ────────────────────────────────────────────────────────

RUNTIME_SENTENCE_TRANSFORMERS = "sentence_transformers"
RUNTIME_ONNX = "onnx"
RUNTIME_NONE = "none"
EMBEDDING_RUNTIMES = (
    RUNTIME_SENTENCE_TRANSFORMERS,
    RUNTIME_ONNX,
    RUNTIME_NONE,
)

#: The package each runtime needs importable on the machine.
RUNTIME_PACKAGES = {
    RUNTIME_SENTENCE_TRANSFORMERS: "sentence_transformers",
    RUNTIME_ONNX: "onnxruntime",
    RUNTIME_NONE: None,
}

STATUS_AVAILABLE = "available_offline"
STATUS_PACKAGE_MISSING = "unavailable_package_missing"
STATUS_NOT_CACHED = "unavailable_model_not_cached"
STATUS_NO_ONNX = "unavailable_onnx_artefact_missing"
STATUS_NOT_ASSIGNED = "not_assigned"
AVAILABILITY_STATUSES = (
    STATUS_AVAILABLE,
    STATUS_PACKAGE_MISSING,
    STATUS_NOT_CACHED,
    STATUS_NO_ONNX,
    STATUS_NOT_ASSIGNED,
)

#: The one fallback rule, for every profile: the SQL leg.
EMBEDDING_FALLBACK = "sql"

# ── The per-profile assignment ────────────────────────────────────────────────

PROFILE_EMBEDDINGS: dict[str, dict[str, Any]] = {
    "server_gpu": {
        "model": "all-MiniLM-L6-v2",
        "runtime": RUNTIME_SENTENCE_TRANSFORMERS,
        "batch_size": 128,
        "fallback": EMBEDDING_FALLBACK,
        "reason": "64 GB RAM carries the torch stack trivially; "
                  "the estate's largest batches.",
    },
    "desktop_gpu": {
        "model": "all-MiniLM-L6-v2",
        "runtime": RUNTIME_SENTENCE_TRANSFORMERS,
        "batch_size": 128,
        "fallback": EMBEDDING_FALLBACK,
        "reason": "16 GB RAM; same stack, same batches as server_gpu.",
    },
    "desktop_cpu": {
        "model": "all-MiniLM-L6-v2",
        "runtime": RUNTIME_SENTENCE_TRANSFORMERS,
        "batch_size": 64,
        "fallback": EMBEDDING_FALLBACK,
        "reason": "8 GB RAM shared with a CPU LLM; batches halved.",
    },
    "pi_5": {
        "model": "all-MiniLM-L6-v2",
        "runtime": RUNTIME_SENTENCE_TRANSFORMERS,
        "batch_size": 32,
        "fallback": EMBEDDING_FALLBACK,
        "reason": "4 GB RAM on ARM; the model fits, batches step "
                  "down to leave the LLM its room.",
    },
    "phone_low": {
        "model": "all-MiniLM-L6-v2",
        "runtime": RUNTIME_ONNX,
        "batch_size": 16,
        "fallback": EMBEDDING_FALLBACK,
        "reason": "2 GB RAM cannot carry the torch stack beside a "
                  "phone LLM; the profile's own deployment section "
                  "already declares onnx_export, so the model is "
                  "assigned through the ONNX runtime (TF-070 owns "
                  "the export path's truth).",
    },
    "pi_zero": {
        "model": None,
        "runtime": RUNTIME_NONE,
        "batch_size": 0,
        "fallback": EMBEDDING_FALLBACK,
        "reason": "512 MB RAM: no embedding model is honestly "
                  "assignable beside anything else. Vector "
                  "retrieval is not assigned; the profile runs "
                  "SQL-only by rule, not by accident.",
    },
}


def canonical_model_name(name: str) -> str:
    """Fold any spelling of a registered model to its canonical name.

    Registered hub ids (``sentence-transformers/all-MiniLM-L6-v2``)
    and any other org-prefixed spelling of a registered bare name
    fold to the bare canonical name. A name that is an existing
    local path, and any unregistered name, passes through
    unchanged — registration is the registry's act, not a guess
    made here.
    """
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    if isinstance(name, str) and "/" in name:
        if Path(name).exists():
            return name
        tail = name.rsplit("/", 1)[-1]
        if tail in EMBEDDING_MODELS:
            return tail
    return name


def embedding_assignment(profile_id: str) -> dict[str, Any]:
    """The embedding assignment for one shipped profile.

    Raises ValueError naming the assigned profiles for an
    unknown id (the TF-044 weights_for_profile shape).
    """
    try:
        return PROFILE_EMBEDDINGS[profile_id]
    except KeyError:
        raise ValueError(
            f"unknown hardware profile for embeddings: {profile_id!r} "
            f"(assigned: {', '.join(sorted(PROFILE_EMBEDDINGS))})"
        ) from None


# ── Machine facts (no model is loaded, no network is touched) ────────────────

def package_installed(package: str) -> bool:
    """True when *package* is importable on this machine."""
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def cache_roots(env: dict[str, str] | None = None) -> list[Path]:
    """The local model-cache roots, in search order."""
    env = os.environ if env is None else env
    roots: list[Path] = []
    if env.get("HF_HUB_CACHE"):
        roots.append(Path(env["HF_HUB_CACHE"]))
    if env.get("HF_HOME"):
        roots.append(Path(env["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    if env.get("SENTENCE_TRANSFORMERS_HOME"):
        roots.append(Path(env["SENTENCE_TRANSFORMERS_HOME"]))
    roots.append(Path.home() / ".cache" / "torch" / "sentence_transformers")
    seen: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.append(root)
    return seen


_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _dir_holds_model(directory: Path) -> bool:
    return (directory / "config.json").exists() and any(
        (directory / name).exists() for name in _WEIGHT_FILES
    )


def model_cache_status(
    model: str = DEFAULT_EMBEDDING_MODEL,
    roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Is *model* present in a local cache, and where?

    Recognises the Hugging Face hub layout
    (``models--<org>--<name>/snapshots/<rev>/…``) and the legacy
    sentence-transformers home layout (a folder named for the
    model directly under the root). Filesystem facts only.
    """
    canonical = canonical_model_name(model)
    row = EMBEDDING_MODELS.get(canonical)
    hub_id = row["hub_id"] if row else canonical
    roots = cache_roots() if roots is None else roots
    searched: list[str] = []
    for root in roots:
        searched.append(str(root))
        hub_dir = root / ("models--" + hub_id.replace("/", "--"))
        snapshots = hub_dir / "snapshots"
        if snapshots.is_dir():
            for snapshot in sorted(snapshots.iterdir()):
                if snapshot.is_dir() and _dir_holds_model(snapshot):
                    return {"cached": True, "location": str(snapshot),
                            "searched": searched}
        for legacy in (canonical, hub_id.replace("/", "_"),
                       hub_id.replace("/", "--")):
            candidate = root / legacy
            if candidate.is_dir() and _dir_holds_model(candidate):
                return {"cached": True, "location": str(candidate),
                        "searched": searched}
    return {"cached": False, "location": None, "searched": searched}


def onnx_artefact_status(onnx_dir: Path | None = None) -> dict[str, Any]:
    """Is an exported ONNX embedding model in the embeddings dir?

    The default directory is ``get_embeddings_dir()`` — docket
    D-DEAD-3's helper, called here for the first time: the
    embeddings directory is the named home an export lands in
    (TF-070's path writes where its caller says; this report
    checks the estate's own home).
    """
    directory = Path(onnx_dir) if onnx_dir is not None else get_embeddings_dir()
    for filename in ("model.onnx", "model_quantized.onnx"):
        if (directory / filename).exists():
            return {"present": True, "dir": str(directory),
                    "file": filename}
    return {"present": False, "dir": str(directory), "file": None}


# ── The per-profile offline proof ─────────────────────────────────────────────

@dataclass
class AvailabilityReport:
    profile_id: str
    model: str | None
    runtime: str
    status: str
    package_installed: bool | None = None
    model_cached: bool | None = None
    cache_location: str | None = None
    onnx_present: bool | None = None
    onnx_dir: str | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.status == STATUS_AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model": self.model,
            "runtime": self.runtime,
            "status": self.status,
            "available": self.available,
            "package_installed": self.package_installed,
            "model_cached": self.model_cached,
            "cache_location": self.cache_location,
            "onnx_present": self.onnx_present,
            "onnx_dir": self.onnx_dir,
            "reasons": list(self.reasons),
        }


def availability_report(
    profile_id: str,
    *,
    roots: list[Path] | None = None,
    onnx_dir: Path | None = None,
    package: bool | None = None,
) -> AvailabilityReport:
    """Prove (or disprove) offline embedding availability per profile.

    ``package`` overrides the importable check (tests and
    what-if readings); the cache and artefact checks always
    read the filesystem they are given. No model is loaded
    and no network is touched, in any branch.
    """
    assignment = embedding_assignment(profile_id)
    runtime = assignment["runtime"]
    model = assignment["model"]
    report = AvailabilityReport(
        profile_id=profile_id, model=model, runtime=runtime,
        status=STATUS_NOT_ASSIGNED,
        reasons=[assignment["reason"]],
    )
    if runtime == RUNTIME_NONE:
        return report

    package_name = RUNTIME_PACKAGES[runtime]
    installed = package_installed(package_name) if package is None else package
    report.package_installed = installed
    if not installed:
        report.status = STATUS_PACKAGE_MISSING
        report.reasons.append(
            f"package {package_name!r} is not importable on this "
            f"machine; the {runtime} runtime cannot load at all — "
            f"retrieval falls back to {EMBEDDING_FALLBACK}."
        )
        return report

    if runtime == RUNTIME_ONNX:
        artefact = onnx_artefact_status(onnx_dir)
        report.onnx_present = artefact["present"]
        report.onnx_dir = artefact["dir"]
        if artefact["present"]:
            report.status = STATUS_AVAILABLE
            report.reasons.append(
                f"ONNX artefact {artefact['file']} present in "
                f"{artefact['dir']}."
            )
        else:
            report.status = STATUS_NO_ONNX
            report.reasons.append(
                f"no ONNX artefact in {artefact['dir']} — the "
                f"export (TF-070) has not landed one there; "
                f"retrieval falls back to {EMBEDDING_FALLBACK}."
            )
        return report

    cache = model_cache_status(model, roots)
    report.model_cached = cache["cached"]
    report.cache_location = cache["location"]
    if cache["cached"]:
        report.status = STATUS_AVAILABLE
        report.reasons.append(
            f"model cached at {cache['location']} — loading it "
            f"needs no network."
        )
    else:
        report.status = STATUS_NOT_CACHED
        report.reasons.append(
            "model is in no local cache (searched: "
            + ", ".join(cache["searched"])
            + ") — a first load would reach for the network, so "
              "offline availability is NOT proven; retrieval "
              f"falls back to {EMBEDDING_FALLBACK} until the "
              "model is cached."
        )
    return report


# ── The stored estate, checked against the registry ───────────────────────────

_EMBEDDING_TABLES = (
    ("embeddings", "embedding"),
    ("reference_embeddings", "embedding"),
)


def stored_embeddings_problems(
    db_path: Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[str]:
    """Check a knowledge DB's stored embeddings against the registry.

    Every row filed under *model* (either spelling) must carry
    the canonical model name and a blob of exactly the
    registered dimension × 4 bytes. Rows filed under other
    models are named, not judged (their registry entry is
    another act).
    """
    from thoughtforge.etl.schema import get_connection

    canonical = canonical_model_name(model)
    row = EMBEDDING_MODELS.get(canonical)
    if row is None:
        return [f"model {model!r} is not in the embedding registry"]
    expected_bytes = row["dimension"] * 4
    problems: list[str] = []
    conn = get_connection(Path(db_path), read_only=True)
    try:
        for table, blob_column in _EMBEDDING_TABLES:
            try:
                rows = conn.execute(
                    f"SELECT model_name, {blob_column} FROM {table}"
                ).fetchall()
            except Exception:
                continue  # table absent — nothing stored, nothing to judge
            for stored_name, blob in rows:
                if canonical_model_name(stored_name) != canonical:
                    continue
                if stored_name != canonical:
                    problems.append(
                        f"{table}: a row is filed under the alias "
                        f"{stored_name!r}, not the canonical "
                        f"{canonical!r}"
                    )
                if blob is None or len(blob) != expected_bytes:
                    got = 0 if blob is None else len(blob)
                    problems.append(
                        f"{table}: a {canonical} blob is {got} bytes, "
                        f"not the registered {expected_bytes} "
                        f"({row['dimension']} dims × float32)"
                    )
    finally:
        conn.close()
    return problems


# ── The live checker ──────────────────────────────────────────────────────────

def _functions_calling(source: str, callee: str) -> set[str]:
    """Names of functions in *source* whose bodies call *callee*."""
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == callee):
                    found.add(node.name)
    return found


def embedding_problems() -> list[str]:
    """Re-derive TF-053's claims from the live tree; [] = all hold."""
    import tempfile

    import numpy as np

    from thoughtforge.etl import embeddings
    from thoughtforge.etl.schema import initialize_schema
    from thoughtforge.inference import onnx_export
    from thoughtforge.knowledge.forge import KnowledgeForge

    problems: list[str] = []

    # The registry describes the tree's actual default model.
    canonical = canonical_model_name(embeddings.DEFAULT_MODEL)
    if canonical != embeddings.DEFAULT_MODEL:
        problems.append(
            f"etl DEFAULT_MODEL {embeddings.DEFAULT_MODEL!r} is not "
            f"the canonical spelling"
        )
    registry_row = EMBEDDING_MODELS.get(canonical)
    if registry_row is None:
        problems.append(f"default model {canonical!r} is not registered")
    else:
        if registry_row["dimension"] != embeddings.EMBEDDING_DIM:
            problems.append(
                f"registry dimension {registry_row['dimension']} != "
                f"EMBEDDING_DIM {embeddings.EMBEDDING_DIM}"
            )
        if not registry_row["licence"]:
            problems.append("the default model's licence is unrecorded")

    # Every other home of the model's name agrees, or folds to it.
    import inspect

    forge_default = inspect.signature(
        KnowledgeForge.__init__).parameters["model_name"].default
    if canonical_model_name(forge_default) != canonical:
        problems.append(
            f"KnowledgeForge default {forge_default!r} does not "
            f"fold to {canonical!r}"
        )
    if canonical_model_name(onnx_export._DEFAULT_MODEL) != canonical:
        problems.append(
            f"onnx_export default {onnx_export._DEFAULT_MODEL!r} "
            f"does not fold to {canonical!r}"
        )
    config_text = (
        Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
    ).read_text(encoding="utf-8")
    config_model = None
    for line in config_text.splitlines():
        if "embedding_model:" in line:
            config_model = line.split(":", 1)[1].strip()
    if config_model != canonical:
        problems.append(
            f"configs/default.yaml embedding_model is "
            f"{config_model!r}, not {canonical!r}"
        )

    # The assignment covers exactly the shipped profiles.
    shipped = sorted(list_hardware_profiles())
    assigned = sorted(PROFILE_EMBEDDINGS)
    if shipped != assigned:
        problems.append(
            f"assigned profiles {assigned} != shipped {shipped}"
        )
    for profile_id, assignment in PROFILE_EMBEDDINGS.items():
        model = assignment["model"]
        if model is not None and model not in EMBEDDING_MODELS:
            problems.append(
                f"{profile_id}: model {model!r} is not registered"
            )
        if assignment["runtime"] not in EMBEDDING_RUNTIMES:
            problems.append(
                f"{profile_id}: runtime {assignment['runtime']!r} "
                f"is not a named runtime"
            )
        if assignment["runtime"] == RUNTIME_NONE and model is not None:
            problems.append(f"{profile_id}: runtime none with a model")
        if assignment["runtime"] != RUNTIME_NONE and model is None:
            problems.append(f"{profile_id}: a runtime with no model")
        if assignment["fallback"] != EMBEDDING_FALLBACK:
            problems.append(
                f"{profile_id}: fallback is not the stated rule"
            )

    # The ETL entry points normalise through the canonical fold.
    etl_source = Path(embeddings.__file__).read_text(encoding="utf-8")
    wired = _functions_calling(etl_source, "canonical_model_name")
    for function in ("_get_model", "build_entity_embeddings",
                     "build_reference_embeddings",
                     "vector_search_entities",
                     "vector_search_reference"):
        if function not in wired:
            problems.append(
                f"etl.embeddings.{function} does not normalise "
                f"through canonical_model_name"
            )

    # The batch sibling is the live cosine path (D-DEAD-2's ground).
    unit = np.array([1.0, 0.0], dtype=np.float32)
    scores = embeddings.batch_cosine_similarity(unit, unit)
    if float(scores[0]) != 1.0:
        problems.append("batch_cosine_similarity lost its identity case")

    # The forge degrades instead of raising (measured TF-053 defect):
    # with the vector leg forced to fail, hybrid retrieval keeps
    # its SQL results on TF-043's fixture corpus.
    from thoughtforge.knowledge import forge as forge_module
    from thoughtforge.knowledge.retrieval_precision import build_fixture_db

    with tempfile.TemporaryDirectory() as tmp:
        db_path = build_fixture_db(Path(tmp) / "fixture.db")
        forge = KnowledgeForge(db_path=db_path)
        original = forge_module.vector_search_entities

        def _raising(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("embedding backend unavailable (probe)")

        forge_module.vector_search_entities = _raising  # type: ignore[misc]
        try:
            bundle = forge.retrieve("ferry", path="hybrid")
        except Exception as exc:  # noqa: BLE001 — the check IS the raise
            problems.append(
                f"hybrid retrieval raised {type(exc).__name__} with "
                f"the vector leg down, instead of degrading to SQL"
            )
            bundle = None
        finally:
            forge_module.vector_search_entities = original  # type: ignore[misc]
        if bundle is not None and bundle.total_retrieved < 1:
            problems.append(
                "hybrid retrieval returned no SQL records with the "
                "vector leg down"
            )

        # The stored-estate check runs clean on a disciplined DB…
        clean_db = Path(tmp) / "clean.db"
        initialize_schema(clean_db)
        from thoughtforge.etl.schema import get_connection

        conn = get_connection(clean_db)
        try:
            conn.execute(
                "INSERT INTO embeddings (qid, text_for_embedding, "
                "embedding, model_name) VALUES (?, ?, ?, ?)",
                ("Q1", "ferry", np.zeros(384, dtype=np.float32).tobytes(),
                 canonical),
            )
            conn.commit()
        finally:
            conn.close()
        stored = stored_embeddings_problems(clean_db)
        if stored:
            problems.append(
                f"stored_embeddings_problems on a clean DB: {stored}"
            )

    # The live availability verdict is consistent with the facts
    # it is derived from (whatever this machine's facts are).
    report = availability_report("desktop_cpu")
    if report.status not in AVAILABILITY_STATUSES:
        problems.append(f"availability status {report.status!r} unnamed")
    installed = package_installed("sentence_transformers")
    if report.package_installed != installed:
        problems.append("availability report disagrees on the package")
    if installed and report.status == STATUS_PACKAGE_MISSING:
        problems.append("report claims a missing package that imports")
    if report.available:
        cache = model_cache_status(canonical)
        if not cache["cached"]:
            problems.append(
                "report claims offline availability with no cached model"
            )

    # The ONNX home is the paths helper's embeddings directory.
    artefact = onnx_artefact_status()
    if artefact["dir"] != str(get_embeddings_dir()):
        problems.append("onnx artefact home is not get_embeddings_dir()")

    return problems
