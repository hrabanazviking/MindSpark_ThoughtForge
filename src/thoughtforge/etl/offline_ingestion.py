"""
Offline ingestion — everything except the fetch works
with the network cut, as data (TF-117).

The roadmap's rule: *offline ingestion: everything
except the fetch itself works with the network cut
(sovereignty, proven).*

The measured starting state, probed live this tick on
this VM before anything was built:

  * **Offline was an assumption, never a proof
    (D-OFFLINE-1).** The five builders read local files
    by their signatures, so the pipeline *looked*
    offline — but no module, test or checker ever ran
    ingestion with the network cut. No socket was ever
    blocked, no attempt ever recorded, and nothing
    could answer "did anything try to reach out?"
    Sovereignty was a reading, not a measurement.
  * **The embedding load was an unguarded fetch inside
    ingestion (D-OFFLINE-2).** ``etl/embeddings.
    _get_model`` constructed ``SentenceTransformer(name)``
    with no ``local_files_only`` and no consultation of
    TF-053's cache proof: on a machine with the package
    installed but the model uncached, the embedding step
    of an otherwise local build (``forge_memory all``
    step 6, the ``embeddings`` command, TF-111's default
    encoder) silently became a Hugging Face *fetch* —
    and offline it failed with a hub/network error, not
    a named local verdict. TF-053 already knew how to
    prove a model cached (``model_cache_status``); no
    ingestion path used that knowledge at load time.
  * **The fetch boundary was undefined (D-OFFLINE-3).**
    Nothing filed which operations ARE the fetch and
    which are ingestion: source downloads exist only as
    docstring URLs in ``etl/sources.py`` (no downloader
    exists in the tree), model download existed only as
    D-OFFLINE-2's hidden side effect, and package
    installation is a pip act no doc named as separate.
    "Except the fetch itself" therefore named a thing
    nobody had ever drawn.

What this slice builds, filed as data in
``FETCH_STEPS`` / ``OFFLINE_STEPS`` / ``OFFLINE_RULES``
and enforced here:

  * **The fetch boundary, as data.** Exactly three
    fetch steps, all OUTSIDE ingestion: downloading a
    source file, downloading an embedding model, and
    installing packages. No code in ``etl/`` performs
    any of them after this slice (the loader fix below
    removes the last hidden one). Everything else in
    Part VI is an offline step, bound by name to
    TF-105's builders and stages.
  * ``NetworkCut`` — a hard, recording network block:
    socket connect / connect_ex, ``create_connection``,
    ``getaddrinfo`` / ``gethostbyname`` and
    ``urllib.request.urlopen`` all raise
    ``OfflineAttemptError`` (an ``OSError``) and are
    recorded in the cut's ``attempts``. Creating a
    socket object is not an attempt; connecting is.
    The cut restores every patched name exactly on
    exit, nested or not.
  * ``run_offline_pipeline(root)`` — TF-105's whole
    synthetic pipeline (schema, all five builders,
    embeddings through the cached fake model, subset)
    plus a poison ``scan_store`` over the result, all
    run inside a ``NetworkCut``, returning a report
    (format ``thoughtforge-offline-ingestion/1``): the
    pipeline's own counts verbatim, the scan's counts,
    and ``network_attempts`` — which must be empty for
    the run to file ``offline: True``. An attempt does
    not abort the report; it is filed in it, named.
  * **The loader fix (D-OFFLINE-2).** ``_get_model``
    now loads with ``local_files_only=True``, falling
    back to a plain construction only when the
    constructor rejects that argument (test fakes do;
    the real constructor accepts it). An uncached real
    model therefore fails locally instead of fetching.

Filed limits, stated plainly:

  * The proof is at FIXTURE grade: synthetic inputs
    through the real builders on this VM under a
    process-local socket block. It does not pull the
    machine's real network cable, and a library that
    reached the network without sockets / urllib (none
    is in the ingestion path; the static checker pins
    the etl estate's imports) is outside its reach.
  * The embedding leg is proven offline in its cached
    / injected shapes (TF-105's fake model in the
    loader cache; TF-111's injected encoder). A real
    sentence-transformers encode is NOT PROVEN on this
    VM — the package is absent (D-ENV-1) — and is
    never claimed here. What IS proven is that the
    loader can no longer turn that leg into a fetch.
  * ``forge_memory.py`` gains no ``--offline`` flag in
    this slice: the CLI's build steps are the same
    builders proven here, and TF-119 owns CLI wiring.

Docket (TF-016): D-OFFLINE-1, D-OFFLINE-2 and
D-OFFLINE-3 are added and FIXED by this slice.

The human-readable write-up is
``docs/architecture/OFFLINE_INGESTION.md``. Proof
grade: FIXTURE — synthetic sources through the real
builders under a hard socket block on this VM; a
real dump ingested with the machine's network
physically cut is NOT PROVEN, never claimed.
"""

from __future__ import annotations

import socket
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

OFFLINE_FORMAT = "thoughtforge-offline-ingestion/1"
OFFLINE_DOC = "docs/architecture/OFFLINE_INGESTION.md"


# ── The fetch boundary, as data ─────────────────────────────────────────────
#
# A fetch is any act that must reach outside the
# machine. Exactly three exist for this estate, and
# none of them is an ingestion step: ingestion starts
# when the files are already here.

FETCH_STEPS: tuple[dict[str, Any], ...] = (
    {"id": "source_file_download",
     "what": "downloading a source dump / CSV / text file "
             "(wikidata, conceptnet, geonames, dbpedia)",
     "where": "outside the tree, by hand — the builders take "
              "local paths; their docstrings in etl/sources.py "
              "carry the upstream URLs and no downloader "
              "exists in etl/",
     "ingestion_code_performs_it": False},
    {"id": "embedding_model_download",
     "what": "downloading an embedding model into a local "
             "cache (TF-053's cache roots)",
     "where": "an explicit act before ingestion — after "
              "TF-117 the loader is local-files-only, so "
              "ingestion itself never performs it",
     "ingestion_code_performs_it": False},
    {"id": "package_install",
     "what": "installing packages (pip: sentence-transformers, "
             "ijson, tqdm, ...)",
     "where": "outside the tree, at setup time — no ingestion "
              "module installs anything",
     "ingestion_code_performs_it": False},
)

#: The offline steps: every Part VI operation that must
#: work with the network cut. ``builder`` binds the
#: first five to TF-105's SOURCE_BUILDERS by id.
OFFLINE_STEPS: tuple[dict[str, Any], ...] = (
    {"id": "schema_init", "builder": None,
     "module": "etl/schema.py",
     "what": "initialising the store schema"},
    {"id": "wikidata_build", "builder": "wikidata",
     "module": "etl/wikidata.py",
     "what": "streaming a local wikidata dump into the store"},
    {"id": "conceptnet_build", "builder": "conceptnet",
     "module": "etl/sources.py",
     "what": "ingesting a local conceptnet CSV / reference edges"},
    {"id": "geonames_build", "builder": "geonames",
     "module": "etl/sources.py",
     "what": "ingesting a local geonames text file"},
    {"id": "dbpedia_build", "builder": "dbpedia",
     "module": "etl/sources.py",
     "what": "ingesting a local dbpedia JSON-lines file"},
    {"id": "reference_build", "builder": "reference",
     "module": "etl/sources.py",
     "what": "ingesting the local reference directory"},
    {"id": "embeddings_cached_or_injected", "builder": None,
     "module": "etl/embeddings.py + etl/embedding_ingestion.py",
     "what": "building embeddings from a cached model or an "
             "injected encoder (an uncached real model fails "
             "locally — it does not fetch)"},
    {"id": "subset_build", "builder": None,
     "module": "etl/subset.py + etl/subset_craft.py",
     "what": "building a profile subset from a local store"},
    {"id": "fts_rebuild", "builder": None,
     "module": "etl/schema.py",
     "what": "rebuilding the FTS indexes of a local store"},
    {"id": "dry_run_metrics_canon_scan", "builder": None,
     "module": "etl/ingestion_dry_run.py + "
               "etl/ingestion_metrics.py + "
               "etl/canon_ingestion.py + "
               "etl/poisoned_source.py",
     "what": "TF-112 dry-runs, TF-115 metrics runs, TF-113 "
             "canon ingestion and TF-116 store scans over "
             "local files and stores"},
)


# ── The rules ───────────────────────────────────────────────────────────────

OFFLINE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "ingestion_starts_when_the_files_are_here",
     "rule": "every builder takes a local file or directory and ingestion is the work done on files already on the machine; downloading a source is the fetch step, outside ingestion and outside this tree (no downloader exists in etl/)"},
    {"id": "the_fetch_boundary_is_filed_data",
     "rule": "exactly the FETCH_STEPS are fetches — source download, embedding-model download, package install — and every other Part VI operation is an OFFLINE_STEPS entry bound to its module (and, for builders, to TF-105's map); a new operation is filed on one side of the boundary before it ships"},
    {"id": "offline_is_proven_by_a_cut_never_assumed",
     "rule": "offline is claimed only from a run under NetworkCut: the whole synthetic pipeline (all five builders, cached/injected embeddings, subset, store scan) completing with zero recorded network attempts — a builder's local-file signature alone proves nothing (D-OFFLINE-1)"},
    {"id": "the_cut_blocks_and_records",
     "rule": "NetworkCut blocks socket connect/connect_ex, create_connection, getaddrinfo/gethostbyname and urllib urlopen, raising OfflineAttemptError (an OSError) and recording each attempt with its target; creating a socket object is not an attempt"},
    {"id": "the_cut_restores_exactly",
     "rule": "leaving a NetworkCut restores every patched name to the exact object it replaced, including on exception and under nesting — a cut never leaks its block into the caller's process"},
    {"id": "an_attempt_is_filed_never_hidden",
     "rule": "run_offline_pipeline files every recorded attempt in its report and files offline True only when attempts is empty; an attempt never silently passes and never converts the report into a bare exception"},
    {"id": "the_embedding_loader_never_fetches",
     "rule": "etl/embeddings._get_model loads with local_files_only=True (a plain construction only as the fallback for a constructor that rejects the argument, as test fakes do): an uncached real model fails locally, and downloading the model is the filed fetch step, proven cached beforehand by TF-053's model_cache_status (D-OFFLINE-2)"},
    {"id": "cached_or_injected_embeddings_are_the_offline_leg",
     "rule": "the embedding step is an offline step only in its cached-model and injected-encoder shapes; TF-111's driver with an injected encoder and TF-105's cached fake model are the proven shapes, and no real-model encode is claimed on a machine whose package or cache is absent (D-ENV-1)"},
    {"id": "no_etl_module_reaches_for_the_network",
     "rule": "no module under etl/ imports requests, urllib (outside this module, which imports it only to block it), http.client or socket (outside this module, which imports it only to block it) at any level the static checker scans — a new network import in the ingestion estate turns the checker red until the boundary is re-filed in the open"},
    {"id": "local_files_and_stores_only",
     "rule": "every offline step reads and writes only local files and local SQLite stores: no step resolves a hostname, opens a URL, or shells out to a network tool, and the reference/canon sources are local directories by construction"},
    {"id": "no_cli_offline_flag_claimed",
     "rule": "forge_memory.py gains no offline flag in this slice — its build steps are the builders proven here, and TF-119 owns any CLI wiring of this report"},
)


# ── The cut ─────────────────────────────────────────────────────────────────

class OfflineAttemptError(OSError):
    """A network attempt made inside a NetworkCut."""


class NetworkCut:
    """Block and record every network reach in a scope.

    Use as a context manager. ``attempts`` holds one
    ``{"kind": ..., "target": ...}`` dict per blocked
    reach, in order. Socket *creation* is allowed (local
    libraries create unused sockets); *connecting* is
    the attempt.
    """

    def __init__(self) -> None:
        self.attempts: list[dict[str, Any]] = []
        self._saved: list[tuple[Any, str, Any]] = []
        self._depth = 0

    def _record(self, kind: str, target: Any) -> OfflineAttemptError:
        self.attempts.append({"kind": kind, "target": str(target)})
        return OfflineAttemptError(
            f"network is cut (TF-117): {kind} to {target!r} blocked")

    def __enter__(self) -> NetworkCut:
        if self._depth:
            self._depth += 1
            return self
        self._depth = 1
        cut = self
        real_socket = socket.socket

        class _BlockedSocket(real_socket):  # type: ignore[valid-type,misc]
            def connect(self, address: Any) -> None:
                raise cut._record("socket.connect", address)

            def connect_ex(self, address: Any) -> int:
                raise cut._record("socket.connect_ex", address)

        def _blocked_create(*args: Any, **kwargs: Any) -> Any:
            raise cut._record(
                "socket.create_connection", args[0] if args else kwargs)

        def _blocked_gai(*args: Any, **kwargs: Any) -> Any:
            raise cut._record(
                "socket.getaddrinfo", args[0] if args else kwargs)

        def _blocked_ghn(*args: Any, **kwargs: Any) -> Any:
            raise cut._record(
                "socket.gethostbyname", args[0] if args else kwargs)

        def _blocked_urlopen(*args: Any, **kwargs: Any) -> Any:
            raise cut._record(
                "urllib.request.urlopen",
                getattr(args[0], "full_url", args[0]) if args else kwargs)

        for module, name, replacement in (
                (socket, "socket", _BlockedSocket),
                (socket, "create_connection", _blocked_create),
                (socket, "getaddrinfo", _blocked_gai),
                (socket, "gethostbyname", _blocked_ghn),
                (urllib.request, "urlopen", _blocked_urlopen)):
            self._saved.append((module, name, getattr(module, name)))
            setattr(module, name, replacement)
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._depth -= 1
        if self._depth:
            return False
        for module, name, original in reversed(self._saved):
            setattr(module, name, original)
        self._saved.clear()
        return False


# ── The offline run ─────────────────────────────────────────────────────────

def run_offline_pipeline(root: Path | str) -> dict[str, Any]:
    """Run the whole synthetic pipeline with the network cut.

    TF-105's ``run_synthetic_pipeline`` (schema, all five
    builders, embeddings through its cached fake model,
    subset) plus a TF-116 ``scan_store`` over the result,
    all inside one ``NetworkCut``. The report files the
    pipeline's counts verbatim and ``offline`` — True
    only when no attempt was recorded.
    """
    from thoughtforge.etl.etl_map import run_synthetic_pipeline
    from thoughtforge.etl.poisoned_source import scan_store

    root = Path(root)
    with NetworkCut() as cut:
        pipeline = run_synthetic_pipeline(root)
        scan = scan_store(root / "synthetic.db")
    return {
        "format": OFFLINE_FORMAT,
        "doc": OFFLINE_DOC,
        "pipeline": pipeline,
        "scan": {
            "scanned_rows": scan["scanned_rows"],
            "flagged": len(scan["flagged"]),
            "by_table": scan["by_table"],
        },
        "network_attempts": [dict(a) for a in cut.attempts],
        "offline": not cut.attempts,
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def offline_problems() -> list[str]:
    """Static truth: the boundary + rules against the live tree."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in OFFLINE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate OFFLINE_RULES ids")
    for needed in ("ingestion_starts_when_the_files_are_here",
                   "offline_is_proven_by_a_cut_never_assumed",
                   "the_embedding_loader_never_fetches",
                   "no_etl_module_reaches_for_the_network"):
        if needed not in rule_ids:
            problems.append(f"OFFLINE_RULES lost {needed}")

    fetch_ids = [s["id"] for s in FETCH_STEPS]
    if len(fetch_ids) != len(set(fetch_ids)):
        problems.append("duplicate FETCH_STEPS ids")
    if any(s["ingestion_code_performs_it"] for s in FETCH_STEPS):
        problems.append("a fetch step claims ingestion performs it")

    from thoughtforge.etl.etl_map import ORCHESTRATION, SOURCE_BUILDERS

    builder_ids = [b["id"] for b in SOURCE_BUILDERS]
    bound = [s["builder"] for s in OFFLINE_STEPS if s["builder"]]
    if bound != builder_ids:
        problems.append(
            f"OFFLINE_STEPS builders {bound} != TF-105 builders "
            f"{builder_ids}")
    if "offline_ingestion" not in ORCHESTRATION.get("offline", ""):
        problems.append("TF-105's map lost its offline orchestration "
                        "entry")

    root = Path(__file__).resolve().parents[3]
    embeddings_text = (
        root / "src/thoughtforge/etl/embeddings.py").read_text(
            encoding="utf-8")
    if "local_files_only" not in embeddings_text:
        problems.append("etl/embeddings.py lost its local-files-only "
                        "loader guard (D-OFFLINE-2)")

    # No network imports in the ingestion estate (this
    # module is the filed exception: it imports socket /
    # urllib only to block them).
    etl_dir = root / "src/thoughtforge/etl"
    for path in sorted(etl_dir.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("import requests", "import urllib",
                       "import http.client", "import socket",
                       "from urllib", "urlopen("):
            if marker in text:
                problems.append(
                    f"etl/{path.name} carries a network marker "
                    f"{marker!r} (TF-117 boundary)")

    doc = root / OFFLINE_DOC
    if not doc.exists():
        problems.append(f"the write-up {OFFLINE_DOC} is missing")
    return problems


def offline_live_problems() -> list[str]:
    """Behavioural truth: the pipeline under a real cut."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # The cut itself: a deliberate reach is blocked,
        # recorded, and the world is restored after.
        real_gai = socket.getaddrinfo
        cut = NetworkCut()
        with cut:
            try:
                socket.create_connection(("example.invalid", 443))
                problems.append("create_connection was not blocked")
            except OfflineAttemptError:
                pass
            try:
                urllib.request.urlopen("https://example.invalid/")
                problems.append("urlopen was not blocked")
            except OfflineAttemptError:
                pass
        if len(cut.attempts) != 2:
            problems.append(f"cut recorded {cut.attempts}, expected 2")
        if socket.getaddrinfo is not real_gai:
            problems.append("the cut did not restore getaddrinfo")

        # The whole pipeline, cut: counts as TF-105 files
        # them, and no attempt at all.
        report = run_offline_pipeline(root / "run")
        if not report["offline"]:
            problems.append(
                f"pipeline recorded network attempts: "
                f"{report['network_attempts']}")
        counts = report["pipeline"]["store_counts"]
        expected = {"entities": 1, "statements": 1, "labels": 1,
                    "embeddings": 1, "conceptnet_edges": 1,
                    "geonames_places": 1, "dbpedia_entities": 1,
                    "reference_chunks": 1}
        if counts != expected:
            problems.append(f"offline store counts drifted: {counts}")
        if report["pipeline"]["entity_embeddings"] != 1 or \
                report["pipeline"]["reference_embeddings"] != 1:
            problems.append(
                f"offline embeddings drifted: {report['pipeline']}")
        subset = report["pipeline"]["subset"]
        if not subset["success"] or subset["entities"] != 1:
            problems.append(f"offline subset drifted: {subset}")
        if report["scan"]["scanned_rows"] < 1:
            problems.append(f"offline scan drifted: {report['scan']}")
    return problems


def offline_as_data() -> dict[str, Any]:
    """The boundary + rules as plain data."""
    return {
        "format": OFFLINE_FORMAT,
        "doc": OFFLINE_DOC,
        "fetch_steps": [dict(s) for s in FETCH_STEPS],
        "offline_steps": [dict(s) for s in OFFLINE_STEPS],
        "rules": [dict(r) for r in OFFLINE_RULES],
    }


__all__ = [
    "FETCH_STEPS",
    "OFFLINE_DOC",
    "OFFLINE_FORMAT",
    "OFFLINE_RULES",
    "OFFLINE_STEPS",
    "NetworkCut",
    "OfflineAttemptError",
    "offline_as_data",
    "offline_live_problems",
    "offline_problems",
    "run_offline_pipeline",
]
