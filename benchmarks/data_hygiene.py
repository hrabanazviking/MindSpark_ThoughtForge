"""
Benchmark data hygiene, as data (TF-131).

The roadmap's rule: *benchmark data hygiene: all
fixtures synthetic or licensed; canon never a
benchmark input.* TF-017 set the fixture policy
for tests — synthetic stand-ins under
``tests/fixtures/``, guarded by tests that scan
the test estate for canon access. But the
benchmark estate grew past that guard: TF-121..
TF-130 filed eleven harness fixtures and a ring
of suite / panel corpora across
``benchmarks/*.py`` plus two bound neighbour
corpora in ``src/`` (TF-043's retrieval fixture,
TF-067's quant fixture), and no filing anywhere
said, per benchmark input, where it came from
or proved it was not canon. TF-017's guards
scan ``tests/`` only; a benchmark corpus could
have drifted to real material and nothing in
the tree would have named it.

The measured starting state, probed live this
tick on this tree before anything was built:

  * **The estate stood clean, and unfiled.**
    ``harness_problems()`` returned [],
    ``baseline_problems()`` returned [] over
    the eleven pinned baselines,
    ``regression_gate_problems()`` returned []
    with the live gate PASS (9 baselines / 83
    metrics / 0 failures), and
    ``spot_checks_problems()`` returned [] —
    and a grep for any hygiene / provenance /
    licence registry across the benchmark
    estate found nothing: no per-fixture
    provenance filing existed for any
    benchmark input.
  * **The canon scan was clean but unguarded.**
    A grep of ``benchmarks/*.py`` for the
    canon-access shapes TF-017 forbids (the
    repo-anchored canon-tree path, the canon
    path helpers, the reference-library name)
    found no hits — true today, proven by
    nothing that runs.
  * **The gap was never numbered.** The number
    **D-BENCH-16** was verified unused anywhere
    in benchmarks / docs / tests / src before
    this slice filed it (the D-BENCH-9..15
    pattern: a gap assigned by the roadmap
    without ever being numbered).

What this module files, as verifiable data:

  * ``HYGIENE_ROWS`` — the full inventory of
    benchmark inputs: every harness fixture,
    every suite / panel corpus bound through
    ``benchmarks/*.py``, the bound neighbour
    corpora in ``src/`` the suites consume
    (TF-043, TF-067, TF-030's persona fixture),
    the shipped hardware-profile files the
    profile panel reads, and the
    ``tests/fixtures/`` estate (filed for
    completeness: no benchmark module imports
    it — the suites carry their own corpora in
    TF-017's same invented world). Each row
    files where the input lives (module +
    symbol, or filed paths), its origin class
    (synthetic-invented / licensed-or-public-
    domain with the licence or source named /
    derived-live-from-another-filed-corpus),
    and its proof. **No corpus is copied**:
    rows bind the real modules / symbols /
    files and the checker inspects them live.
  * The measured classification on this tree:
    21 inputs — 15 synthetic-invented, 6
    derived-live from other filed rows, and
    **0 licensed**: no benchmark input in the
    estate is licensed or public-domain
    material, so the licensed class files no
    rows here; the checker still enforces its
    rule (a licensed row must name its licence
    or source), proven by a filed probe.
  * ``run_data_hygiene()`` files the verdict:
    PASS only on the FULL filed inventory with
    every input classified and clean; a
    partial selection files REVIEW however it
    scores; an empty selection files 0 inputs
    / REVIEW — no vacuous pass. Canon
    contamination, an unregistered fixture, or
    a classification that cannot be proven
    FAILs by name.
  * ``HYGIENE_PROBES`` — filed probes against
    the pure checker parts with synthetic
    perturbed filings (a canon-origin row, a
    licensed row naming no licence, a synthetic
    row with no proof, an unresolvable binding,
    a source carrying a canon-access shape, an
    unregistered fixture, and one clean row
    that must pass), proving the checker can
    fail. Real filings are never perturbed.

The canon scan reuses TF-017's guard shapes.
TF-017's lists live in a test module the
benchmark estate cannot import, so the shapes
are re-filed here **openly, as a duplication**:
the same repo-anchored canon path pattern, the
same canon path helpers, the same
reference-library and picture-prefix markers,
and the same canon persona name list. The
pattern strings are assembled from parts so
this module's own source — which the scan also
covers — does not itself carry the forbidden
literals (the shape TF-130's tests use for the
same reason).

D-BENCH-16 is FIXED by this slice in the
measuring sense: every benchmark input now
carries a filed hygiene row, judged live. What
is NOT fixed, and is stated in the filed
limits: the proof is marker / path /
provenance-based over the estate's own filings
— canon content itself is never opened
(TF-017 / TF-007); a synthetic corpus whose
text quietly paraphrased canon without using
any forbidden shape would not be caught by
this scan, and no such claim is made; and
live-model benchmark data is NOT PROVEN
(TF-173's road). This slice's own harness
baseline is judged by this module's checker,
NOT gated by TF-129's regression gate (the
gate's gated set stays the nine TF-121..
TF-128 baselines exactly as filed; the
exclusion is filed in the open in
``benchmarks/regression_gate.py``, the shape
TF-129 / TF-130 filed for their own baselines).

**TF-132 update, in the open:** TF-132 filed
one new benchmark input after this inventory
was built — the WYRD-shaped panel's corpus
(``benchmarks/wyrd_panel.py``: the invented
Emberwick world + its 18 filed cases). It is
filed below as the row ``wyrd_panel_corpus``
(synthetic-invented, harness fixture
``wyrd_panel_corpus``), so the inventory grows
20 -> 21 inputs and the classification moves
14 -> 15 synthetic-invented (0 licensed, 6
derived-live unchanged), all 21 clean. The
pinned hygiene baseline in the harness is
re-pinned in the open for exactly that reason
— the estate grew by one filed synthetic
input; no classification of any earlier row
changed.

Filed limits, stated plainly:

  * Proof is marker / path / provenance-based
    over the estate's own filings: bindings
    resolve live, TF-017 markers are checked
    where the policy files them, module
    filings are checked for their synthetic
    claims, and benchmark code is scanned for
    canon-access shapes. Canon content itself
    is never opened, copied, or quoted.
  * The scan covers benchmark code
    (``benchmarks/*.py`` and the bound
    neighbour modules in ``src/``), not
    fixture data files and not the whole
    ``src/`` tree — TF-017's guards cover
    ``tests/``; other ``src/`` modules are
    other slices' filings.
  * No licensed or public-domain input exists
    in the estate today (0 rows): the class is
    enforced by rule + probe, not exercised
    by a real row.
  * Live-model benchmark data is NOT PROVEN
    (TF-173): every input filed here is a
    fixture for the harness's scorers, never
    a live model's output or a real
    conversation.

The human-readable write-up is
``docs/architecture/BENCHMARK_DATA_HYGIENE.md``.
Proof grade: FIXTURE — the estate's own filed
fixtures inspected live on this VM; no canon
opened, no live model, never claimed.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_HYGIENE_FORMAT = "thoughtforge-data-hygiene/1"
DATA_HYGIENE_DOC = "docs/architecture/BENCHMARK_DATA_HYGIENE.md"

#: The new baseline this slice registers in the
#: harness — the inventory's own pin. It is NOT
#: one of TF-129's gated baselines (the gate
#: gates the nine TF-121..TF-128 baselines, as
#: filed; this baseline is judged by this
#: module's checker, the shape TF-129 filed for
#: its own baseline and TF-130 for its).
DATA_HYGIENE_BASELINE_ID = "data_hygiene_synthetic_v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The origin classes a row may file. A row in
#: none of these — canon above all — fails.
ORIGIN_CLASSES: tuple[str, ...] = (
    "synthetic", "licensed", "derived")

#: The proof kinds a row may file. A synthetic
#: row must carry one; a derived row's proof is
#: the live cleanliness of the rows it derives
#: from, filed as ``derived_live``.
PROOF_KINDS: tuple[str, ...] = (
    "harness_fixture", "module_filing",
    "tf017_marker", "files_filed", "derived_live")


# ── TF-017's guard shapes, re-filed openly ──────────────────────────────────
#
# Duplication, filed in the open (see the module
# docstring): TF-017's CANON_ACCESS_PATTERNS and
# CANON_PERSONA_NAMES live in
# tests/test_tf017_fixture_policy.py, which the
# benchmark estate does not import. The shapes
# below are the same shapes, assembled from
# parts so this module's own source stays clean
# under its own scan.

_D = "da" + "ta"
_KR = "knowledge_" + "reference"
_PIC = "pictures_" + "of_"

CANON_ACCESS_PATTERNS: tuple[tuple[str, Any], ...] = (
    ("repo_anchored_canon_path", re.compile(
        r"(?:REPO_ROOT|_REPO_ROOT|_REPO|REPO)\s*/\s*[\"']"
        + _D + r"[\"']")),
    ("canon_reference_path", re.compile(
        r"[\"']" + _D + r"/(?:" + _KR + r"|" + _D + r"\.db)")),
    ("canon_data_dir_helper", re.compile(
        r"get_" + _D + r"_dir\(\)")),
    ("canon_reference_dir_helper", re.compile(
        r"get_" + _KR + r"_dir\(\)")),
    ("canon_db_path_helper", re.compile(
        r"get_" + _KR + r"_db_path\(\)")),
    ("canon_reference_marker", re.compile(_KR)),
    ("canon_picture_marker", re.compile(_PIC)),
)

#: TF-017's canon persona name list, re-filed
#: openly (assembled from parts, as above).
CANON_PERSONA_NAMES: tuple[str, ...] = ("Sig" + "rid",)


def scan_source_text(text: str) -> list[str]:
    """Name every canon-access shape a source
    text carries (TF-017's guard shapes, plus
    the canon persona names as data). Pure."""
    hits: list[str] = []
    for name, pattern in CANON_ACCESS_PATTERNS:
        if pattern.search(text):
            hits.append(name)
    for persona in CANON_PERSONA_NAMES:
        if persona in text:
            hits.append(f"canon_persona_name:{persona}")
    return hits


# ── The filed inventory: every benchmark input ──────────────────────────────
#
# Each row binds the real thing: a module +
# symbol (resolved live, with any sibling
# symbols the corpus spans), or filed paths on
# disk. ``harness_fixture`` names the harness
# FIXTURES entry the row covers, where one
# exists — the checker requires the rows'
# harness fixtures to equal the harness's
# FIXTURES exactly, so an unregistered fixture
# fails by name.

_SYNTHETIC_SRC = (
    "invented in-tree for this estate (TF-017's "
    "synthetic-only fixture policy)")

HYGIENE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "persona_fixture_responses",
     "benchmark": "persona_consistency",
     "bind_kind": "module",
     "bind_module": "benchmarks.harness",
     "bind_attribute": "PERSONA_FIXTURE_RESPONSES",
     "also_attributes": (),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC,
     "derived_from": (),
     "proof_kind": "harness_fixture",
     "harness_fixture": "persona_fixture_responses",
     "subject": "TF-121's six synthetic response "
                "lines scored by the persona "
                "scorer (3 in-voice, 1 plain, 2 "
                "deliberately generic)"},
    {"id": "profile_default_queries",
     "benchmark": "profile_benchmark",
     "bind_kind": "module",
     "bind_module": "benchmarks.benchmark_profiles",
     "bind_attribute": "_DEFAULT_QUERIES",
     "also_attributes": ("WARM_UP_QUERY",),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC,
     "derived_from": (),
     "proof_kind": "harness_fixture",
     "harness_fixture": "profile_default_queries",
     "subject": "the profile benchmark's own 10 "
                "standard Norse-topic queries "
                "plus its dedicated warm-up query"},
    {"id": "persona_suite_corpus",
     "benchmark": "persona_suite",
     "bind_kind": "module",
     "bind_module": "benchmarks.persona_suite",
     "bind_attribute": "ALL_SUITE_TURNS",
     "also_attributes": ("IDENTITY_TURNS", "VOICE_TURNS",
                         "PRESSURE_TURNS", "IDENTITY_FACTS"),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "scripted turns in TF-030's "
                "invented Lantern Hollow world",
     "derived_from": (),
     "proof_kind": "harness_fixture",
     "harness_fixture": "persona_suite_corpus",
     "subject": "TF-122's scripted persona "
                "corpora: 108 scored turns over "
                "the identity / voice / pressure "
                "axes"},
    {"id": "memory_suite_corpus",
     "benchmark": "memory_suite",
     "bind_kind": "module",
     "bind_module": "benchmarks.memory_suite",
     "bind_attribute": "ALL_SUITE_CASES",
     "also_attributes": ("PRESENT_CASES", "TRUE_CASES",
                         "EXPLAIN_CASES", "MEMORY_FACTS"),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "records built over TF-043's "
                "filed synthetic corpus (the "
                "tf043_retrieval_fixture_corpus "
                "row below)",
     "derived_from": (),
     "proof_kind": "harness_fixture",
     "harness_fixture": "memory_suite_corpus",
     "subject": "TF-123's scripted memory "
                "corpora: 36 scored cases over "
                "the present / true / explain axes"},
    {"id": "depth_suite_corpus",
     "benchmark": "depth_suite",
     "bind_kind": "module",
     "bind_module": "benchmarks.depth_suite",
     "bind_attribute": "ALL_SUITE_CASES",
     "also_attributes": ("CHAIN_CASES", "PLAN_CASES",
                         "CHECK_CASES"),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC,
     "derived_from": (),
     "proof_kind": "harness_fixture",
     "harness_fixture": "depth_suite_corpus",
     "subject": "TF-124's scripted depth "
                "corpora: 36 scored multi-step "
                "problems, each on both roads"},
    {"id": "presence_suite_corpus",
     "benchmark": "presence_suite",
     "bind_kind": "module",
     "bind_module": "benchmarks.presence_suite",
     "bind_attribute": "ALL_SUITE_CASES",
     "also_attributes": ("COHERENCE_SAGAS", "STEERING_SAGAS",
                         "CLOSURE_SAGAS"),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC,
     "derived_from": (),
     "proof_kind": "harness_fixture",
     "harness_fixture": "presence_suite_corpus",
     "subject": "TF-125's scripted presence "
                "corpora: 36 scored sagas over "
                "coherence / will steering / "
                "will closure"},
    {"id": "honesty_panel_corpus",
     "benchmark": "honesty_panel",
     "bind_kind": "module",
     "bind_module": "benchmarks.honesty_panel",
     "bind_attribute": "MODEL_TIERS",
     "also_attributes": ("PANEL_ROWS", "TIER_MEANS"),
     "files": (),
     "origin_class": "derived",
     "licence_or_source": "",
     "derived_from": ("persona_suite_corpus",
                      "memory_suite_corpus",
                      "depth_suite_corpus",
                      "presence_suite_corpus"),
     "proof_kind": "derived_live",
     "harness_fixture": "honesty_panel_corpus",
     "subject": "TF-126's filed size tiers: the "
                "four suites' own filed corpora "
                "degraded by 4 filed tiers — no "
                "corpus of its own, derived live "
                "from the suite rows"},
    {"id": "quant_panel_corpus",
     "benchmark": "quant_panel",
     "bind_kind": "module",
     "bind_module": "benchmarks.quant_panel",
     "bind_attribute": "PANEL_ROWS",
     "also_attributes": ("KIND_ROWS",),
     "files": (),
     "origin_class": "derived",
     "licence_or_source": "",
     "derived_from": ("tf067_quant_fixture_corpus",),
     "proof_kind": "derived_live",
     "harness_fixture": "quant_panel_corpus",
     "subject": "TF-127's published panel rows "
                "over TF-067's synthetic fixture "
                "corpus, consumed mechanically — "
                "derived live from the TF-067 row"},
    {"id": "profile_panel_corpus",
     "benchmark": "profile_panel",
     "bind_kind": "module",
     "bind_module": "benchmarks.profile_panel",
     "bind_attribute": "PANEL_ROWS",
     "also_attributes": (),
     "files": (),
     "origin_class": "derived",
     "licence_or_source": "",
     "derived_from": ("profile_default_queries",
                      "persona_suite_corpus",
                      "memory_suite_corpus",
                      "depth_suite_corpus",
                      "presence_suite_corpus",
                      "hardware_profile_files"),
     "proof_kind": "derived_live",
     "harness_fixture": "profile_panel_corpus",
     "subject": "TF-128's published panel rows: "
                "per shipped profile, the profile "
                "benchmark's filed queries plus "
                "the four suites' filed corpora — "
                "derived live from those rows"},
    {"id": "regression_gate_corpus",
     "benchmark": "regression_gate",
     "bind_kind": "module",
     "bind_module": "benchmarks.regression_gate",
     "bind_attribute": "NOISE_ROWS",
     "also_attributes": ("GATE_PROBES", "TOLERANCES"),
     "files": (),
     "origin_class": "derived",
     "licence_or_source": "",
     "derived_from": ("persona_fixture_responses",
                      "profile_default_queries",
                      "persona_suite_corpus",
                      "memory_suite_corpus",
                      "depth_suite_corpus",
                      "presence_suite_corpus",
                      "honesty_panel_corpus",
                      "quant_panel_corpus",
                      "profile_panel_corpus"),
     "proof_kind": "derived_live",
     "harness_fixture": "regression_gate_corpus",
     "subject": "TF-129's filed noise rows over "
                "the nine gated pinned baselines, "
                "themselves re-measurements of the "
                "rows above — derived live from "
                "the harness baselines"},
    {"id": "spot_checks_corpus",
     "benchmark": "spot_checks",
     "bind_kind": "module",
     "bind_module": "benchmarks.spot_checks",
     "bind_attribute": "TRANSCRIPT_SPECS",
     "also_attributes": (),
     "files": (),
     "origin_class": "derived",
     "licence_or_source": "",
     "derived_from": ("persona_suite_corpus",
                      "memory_suite_corpus",
                      "depth_suite_corpus",
                      "presence_suite_corpus",
                      "tf067_quant_fixture_corpus"),
     "proof_kind": "derived_live",
     "harness_fixture": "spot_checks_corpus",
     "subject": "TF-130's filed transcript "
                "specs locating real cases in the "
                "suites' and TF-067's filed "
                "corpora — derived live from "
                "those rows, assembled by "
                "build_spot_checks()"},
    {"id": "tf043_retrieval_fixture_corpus",
     "benchmark": "retrieval_precision",
     "bind_kind": "module",
     "bind_module": "thoughtforge.knowledge.retrieval_precision",
     "bind_attribute": "FIXTURE_ENTITIES",
     "also_attributes": ("FIXTURE_REFERENCE_CHUNKS",
                         "FIXTURE_CONCEPTNET_EDGES",
                         "FIXTURE_GEONAMES",
                         "FIXTURE_QUERIES"),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "TF-043's module filing: a "
                "synthetic corpus in TF-017's "
                "invented Lantern Hollow world "
                "(the ConceptNet / GeoNames "
                "shapes are record shapes the "
                "forge emits; every row is "
                "invented Lantern Hollow material, "
                "no external dataset is read)",
     "derived_from": (),
     "proof_kind": "module_filing",
     "harness_fixture": None,
     "subject": "TF-043's known-answer retrieval "
                "fixture: entities, reference "
                "chunks, concept edges, place rows "
                "and 9 known-answer queries — "
                "bound by the harness as the "
                "retrieval_precision neighbour and "
                "consumed by TF-123's memory suite"},
    {"id": "tf030_persona_fixture",
     "benchmark": "persona_suite",
     "bind_kind": "module",
     "bind_module": "thoughtforge.cognition.persona",
     "bind_attribute": "FIXTURE_EXPECTATION",
     "also_attributes": ("PERSONA_FIXTURE_TURNS",),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "TF-030's module filing: synthetic "
                "ground truth mirroring the "
                "tests/fixtures synthetic persona "
                "core, in the same invented world",
     "derived_from": (),
     "proof_kind": "module_filing",
     "harness_fixture": None,
     "subject": "TF-030's persona fixture "
                "expectation + fixture turns — "
                "the ground truth TF-122's "
                "persona suite scores against"},
    {"id": "tf067_quant_fixture_corpus",
     "benchmark": "quant_quality",
     "bind_kind": "module",
     "bind_module": "thoughtforge.inference.quant_quality",
     "bind_attribute": "QUANT_PROMPTS",
     "also_attributes": ("FIXTURE_RESPONSES", "CURVE"),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "TF-067's module filing: eight "
                "synthetic prompts in TF-017's "
                "Lantern Hollow world plus two "
                "arithmetic prompts, with "
                "authored fixture responses",
     "derived_from": (),
     "proof_kind": "module_filing",
     "harness_fixture": None,
     "subject": "TF-067's quant fixture estate: "
                "8 prompts x 6 quant levels with "
                "filed fixture responses — bound "
                "by the harness as the "
                "quant_quality neighbour and "
                "consumed by TF-127's panel and "
                "TF-130's spot checks"},
    {"id": "hardware_profile_files",
     "benchmark": "profile_panel",
     "bind_kind": "files",
     "bind_module": "",
     "bind_attribute": "",
     "also_attributes": (),
     "files": ("hardware_profiles/*.json",),
     "origin_class": "synthetic",
     "licence_or_source": "authored in-tree as "
                "project configuration (the "
                "shipped hardware profiles' "
                "budgets); no external data",
     "derived_from": (),
     "proof_kind": "files_filed",
     "harness_fixture": None,
     "subject": "the six shipped hardware "
                "profile JSONs the profile panel "
                "reads live (ids checked against "
                "the harness's KNOWN_PROFILES)"},
    {"id": "tests_fixtures_synthetic_persona_core",
     "benchmark": "tests_fixtures",
     "bind_kind": "files",
     "bind_module": "",
     "bind_attribute": "",
     "also_attributes": (),
     "files": ("tests/fixtures/synthetic_personality_core.yaml",),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "TF-017's stand-in persona core, "
                "carrying its marker",
     "derived_from": (),
     "proof_kind": "tf017_marker",
     "harness_fixture": None,
     "subject": "TF-017's synthetic personality "
                "core YAML — filed for the "
                "tests/fixtures census; no "
                "benchmark module imports it "
                "(the suites carry their own "
                "corpora in the same world)"},
    {"id": "tests_fixtures_synthetic_reference",
     "benchmark": "tests_fixtures",
     "bind_kind": "files",
     "bind_module": "",
     "bind_attribute": "",
     "also_attributes": (),
     "files": ("tests/fixtures/synthetic_reference/*",),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "TF-017's stand-in reference "
                "library, every file carrying "
                "its marker or structured flag",
     "derived_from": (),
     "proof_kind": "tf017_marker",
     "harness_fixture": None,
     "subject": "TF-017's synthetic reference "
                "directory (.md / .jsonl / .yaml) "
                "— filed for the tests/fixtures "
                "census; not benchmark-consumed"},
    {"id": "tests_fixtures_synthetic_module",
     "benchmark": "tests_fixtures",
     "bind_kind": "files",
     "bind_module": "",
     "bind_attribute": "",
     "also_attributes": (),
     "files": ("tests/fixtures/synthetic.py",),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "TF-017's loader module, filing "
                "the marker constant and the "
                "invented turns",
     "derived_from": (),
     "proof_kind": "tf017_marker",
     "harness_fixture": None,
     "subject": "TF-017's synthetic fixture "
                "loader + SYNTHETIC_TURNS — filed "
                "for the tests/fixtures census; "
                "not benchmark-consumed"},
    {"id": "tests_fixtures_chaos_corpus",
     "benchmark": "tests_fixtures",
     "bind_kind": "files",
     "bind_module": "",
     "bind_attribute": "",
     "also_attributes": (),
     "files": ("tests/fixtures/chaos_corpus.json",),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "the corpus files its own "
                "SYNTHETIC_FIXTURE world line "
                "(TF-017's invented world)",
     "derived_from": (),
     "proof_kind": "tf017_marker",
     "harness_fixture": None,
     "subject": "TF-100's chaos corpus JSON — "
                "filed for the tests/fixtures "
                "census; consumed by the chaos "
                "tests, NOT by any benchmark"},
    {"id": "wyrd_panel_corpus",
     "benchmark": "wyrd_panel",
     "bind_kind": "module",
     "bind_module": "benchmarks.wyrd_panel",
     "bind_attribute": "PANEL_CASES",
     "also_attributes": ("WORLD_ENTITIES", "WORLD_EVENTS",
                         "PANEL_PROBES"),
     "files": (),
     "origin_class": "synthetic",
     "licence_or_source": _SYNTHETIC_SRC + " — "
                "TF-132's invented Emberwick "
                "world, written for this estate "
                "in TF-043's Lantern Hollow style",
     "derived_from": (),
     "proof_kind": "harness_fixture",
     "harness_fixture": "wyrd_panel_corpus",
     "subject": "TF-132's filed WYRD panel "
                "corpus: a synthetic invented "
                "world (8 entities, 6 timeline "
                "changes) and 18 filed "
                "world-state consistency cases "
                "scored by the panel's own scorer"},
    {"id": "data_hygiene_corpus",
     "benchmark": "data_hygiene",
     "bind_kind": "module",
     "bind_module": "benchmarks.data_hygiene",
     "bind_attribute": "HYGIENE_ROWS",
     "also_attributes": ("HYGIENE_PROBES",),
     "files": (),
     "origin_class": "derived",
     "licence_or_source": "",
     "derived_from": (),
     "proof_kind": "derived_live",
     "harness_fixture": "data_hygiene_corpus",
     "subject": "TF-131's own filed inventory "
                "(this filing): the registry "
                "itself, derived live from every "
                "other row in it — its "
                "derived_from is the whole rest "
                "of the inventory, filed by the "
                "checker against ALL_ROW_IDS"},
)

#: Every row id except this filing's own — the
#: self row's derivation set, checked live.
ALL_ROW_IDS: tuple[str, ...] = tuple(
    r["id"] for r in HYGIENE_ROWS
    if r["id"] != "data_hygiene_corpus")

TOTAL_INPUTS = len(HYGIENE_ROWS)


# ── The pure checker parts (probes run here) ────────────────────────────────

def row_problems(
    row: dict[str, Any],
    *,
    resolved_empty: bool = False,
    source_text: str = "",
    harness_synthetic: bool | None = None,
) -> list[str]:
    """Judge ONE filed row against its claimed
    classification, from filed + probed facts
    only. Pure: the probes below perturb copies
    of real filings and run here; the live
    evaluator feeds it the real facts."""
    problems: list[str] = []
    rid = row.get("id", "?")
    origin = row.get("origin_class")
    if origin not in ORIGIN_CLASSES:
        problems.append(
            f"{rid}: origin class {origin!r} is not a "
            f"filed class {ORIGIN_CLASSES} — canon or "
            f"unclassified material is never a "
            f"benchmark input")
    if origin == "synthetic":
        if row.get("proof_kind") not in PROOF_KINDS \
                or row.get("proof_kind") == "derived_live":
            problems.append(
                f"{rid}: a synthetic row files no "
                f"proof (proof_kind "
                f"{row.get('proof_kind')!r})")
        if not str(row.get("licence_or_source", "")).strip():
            problems.append(
                f"{rid}: a synthetic row names no "
                f"origin in licence_or_source")
        if harness_synthetic is False:
            problems.append(
                f"{rid}: the harness files its "
                f"fixture as NOT synthetic")
    if origin == "licensed":
        if not str(row.get("licence_or_source", "")).strip():
            problems.append(
                f"{rid}: a licensed row names no "
                f"licence or source — an unproven "
                f"licence claim is not a "
                f"classification")
    if origin == "derived":
        if row.get("proof_kind") != "derived_live":
            problems.append(
                f"{rid}: a derived row files "
                f"proof_kind "
                f"{row.get('proof_kind')!r}, not "
                f"derived_live")
        if rid != "data_hygiene_corpus" \
                and not row.get("derived_from"):
            problems.append(
                f"{rid}: a derived row names no "
                f"rows it derives from")
    if resolved_empty:
        problems.append(
            f"{rid}: its binding resolves empty or "
            f"not at all — the classification "
            f"cannot be proven")
    for hit in scan_source_text(source_text):
        problems.append(
            f"{rid}: canon contamination — its "
            f"source carries {hit} (TF-017)")
    return problems


def coverage_problems(
    filed_fixture_ids: Any, harness_fixture_ids: Any,
) -> list[str]:
    """The registry must cover the harness's
    fixtures exactly: an unregistered fixture,
    or a row covering a fixture that does not
    exist, fails by name. Pure."""
    problems: list[str] = []
    filed = set(filed_fixture_ids)
    harness = set(harness_fixture_ids)
    for missing in sorted(harness - filed):
        problems.append(
            f"unregistered fixture: harness fixture "
            f"{missing} has no hygiene row")
    for extra in sorted(filed - harness):
        problems.append(
            f"hygiene rows cover fixture {extra}, "
            f"which the harness does not file")
    return problems


# ── The filed probes ────────────────────────────────────────────────────────
#
# Synthetic perturbed filings run against the
# pure parts above. Real filings are never
# perturbed; a probe passes when the pure
# checker files exactly the expected shape.

def _probe_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "probe_row", "benchmark": "probe",
        "bind_kind": "module",
        "bind_module": "benchmarks.harness",
        "bind_attribute": "PERSONA_FIXTURE_RESPONSES",
        "also_attributes": (), "files": (),
        "origin_class": "synthetic",
        "licence_or_source": "invented in-tree (probe)",
        "derived_from": (),
        "proof_kind": "harness_fixture",
        "harness_fixture": "probe_fixture",
        "subject": "a probe filing",
    }
    row.update(overrides)
    return row


HYGIENE_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "canon_origin_row_fails",
     "kind": "row",
     "row": _probe_row(origin_class="canon"),
     "facts": {},
     "expect_problem": True,
     "detail": "a row claiming canon as its "
               "origin is refused by class"},
    {"id": "licensed_without_licence_fails",
     "kind": "row",
     "row": _probe_row(origin_class="licensed",
                       licence_or_source=""),
     "facts": {},
     "expect_problem": True,
     "detail": "a licensed row naming no licence "
               "or source cannot be proven"},
    {"id": "synthetic_without_proof_fails",
     "kind": "row",
     "row": _probe_row(proof_kind=""),
     "facts": {},
     "expect_problem": True,
     "detail": "a synthetic row with no proof "
               "kind filed is unproven"},
    {"id": "unresolvable_binding_fails",
     "kind": "row",
     "row": _probe_row(),
     "facts": {"resolved_empty": True},
     "expect_problem": True,
     "detail": "a binding that resolves empty "
               "cannot carry a classification"},
    {"id": "canon_access_source_fails",
     "kind": "row",
     "row": _probe_row(),
     "facts": {"source_text": "x = " + _KR},
     "expect_problem": True,
     "detail": "a source carrying a canon-access "
               "shape is contamination, named"},
    {"id": "unregistered_fixture_fails",
     "kind": "coverage",
     "filed": ("persona_fixture_responses",),
     "harness": ("persona_fixture_responses",
                 "profile_default_queries"),
     "expect_problem": True,
     "detail": "a harness fixture with no "
               "hygiene row fails coverage by name"},
    {"id": "clean_synthetic_row_passes",
     "kind": "row",
     "row": _probe_row(),
     "facts": {"harness_synthetic": True},
     "expect_problem": False,
     "detail": "a clean synthetic filing with "
               "its proof in order files no "
               "problem"},
)


def run_hygiene_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    checker parts; each returns its verdict."""
    results: list[dict[str, Any]] = []
    for probe in HYGIENE_PROBES:
        if probe["kind"] == "coverage":
            problems = coverage_problems(
                probe["filed"], probe["harness"])
        else:
            problems = row_problems(
                probe["row"], **probe["facts"])
        passed = bool(problems) == probe["expect_problem"]
        results.append({
            "id": probe["id"], "passed": passed,
            "problems": problems,
            "detail": probe["detail"]})
    return results


# ── The live evaluation ─────────────────────────────────────────────────────

_MARKER = "SYNTHETIC_" + "FIXTURE"
_STRUCTURED = '"synthetic_fixture": true'


def _resolve_files(row: dict[str, Any]) -> tuple[list[Path], bool]:
    """Resolve a files row's filed paths /
    globs under the repo root."""
    paths: list[Path] = []
    for entry in row["files"]:
        if any(ch in entry for ch in "*?["):
            paths.extend(sorted(_REPO_ROOT.glob(entry)))
        else:
            paths.append(_REPO_ROOT / entry)
    empty = not paths or any(not p.is_file() for p in paths)
    return paths, empty


def _evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one filed row live: resolve its
    binding, gather its proof facts, and judge
    it with the pure checker."""
    problems: list[str] = []
    source_text = ""
    resolved_empty = False
    harness_synthetic: bool | None = None
    if row["harness_fixture"] is not None:
        from benchmarks.harness import FIXTURES

        filed = [f for f in FIXTURES
                 if f["id"] == row["harness_fixture"]]
        if not filed:
            problems.append(
                f"{row['id']}: harness fixture "
                f"{row['harness_fixture']} is not "
                f"filed in the harness")
        else:
            harness_synthetic = bool(filed[0]["synthetic"])
    if row["bind_kind"] == "module":
        try:
            module = importlib.import_module(row["bind_module"])
        except Exception as exc:
            return {**row, "problems": [
                f"{row['id']}: cannot import "
                f"{row['bind_module']}: {exc!r}"]}
        value = getattr(module, row["bind_attribute"], None)
        if not value:
            resolved_empty = True
        for attr in row["also_attributes"]:
            if not getattr(module, attr, None):
                problems.append(
                    f"{row['id']}: sibling symbol "
                    f"{attr} on {row['bind_module']} "
                    f"resolves empty or not at all")
        module_file = getattr(module, "__file__", None)
        if module_file:
            source_text = Path(module_file).read_text(
                encoding="utf-8")
        if row["proof_kind"] == "module_filing" \
                and "synthetic" not in source_text.lower():
            problems.append(
                f"{row['id']}: its module files no "
                f"synthetic claim — the "
                f"classification cannot be proven "
                f"from its filing")
    elif row["bind_kind"] == "files":
        paths, resolved_empty = _resolve_files(row)
        if row["proof_kind"] == "tf017_marker":
            for path in paths:
                text = path.read_text(encoding="utf-8") \
                    if path.is_file() else ""
                if _MARKER not in text and _STRUCTURED not in text:
                    problems.append(
                        f"{row['id']}: {path.relative_to(_REPO_ROOT)} "
                        f"carries neither the TF-017 "
                        f"marker nor the structured "
                        f"synthetic flag")
        if row["id"] == "hardware_profile_files":
            from benchmarks.harness import KNOWN_PROFILES

            ids: list[str] = []
            for path in paths:
                try:
                    ids.append(json.loads(
                        path.read_text(encoding="utf-8"))["profile_id"])
                except Exception as exc:
                    problems.append(
                        f"{row['id']}: {path.name} does "
                        f"not parse as a profile: {exc!r}")
            if sorted(ids) != sorted(KNOWN_PROFILES):
                problems.append(
                    f"{row['id']}: profile ids "
                    f"{sorted(ids)} != the harness's "
                    f"KNOWN_PROFILES")
    problems.extend(row_problems(
        row, resolved_empty=resolved_empty,
        source_text=source_text,
        harness_synthetic=harness_synthetic))
    return {**row, "problems": problems}


@dataclass
class DataHygieneResult:
    """One hygiene run: the evaluated inventory
    + its verdict data."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_inputs(self) -> int:
        return len(self.rows)

    @property
    def synthetic_inputs(self) -> int:
        return sum(1 for r in self.rows
                   if r["origin_class"] == "synthetic")

    @property
    def licensed_inputs(self) -> int:
        return sum(1 for r in self.rows
                   if r["origin_class"] == "licensed")

    @property
    def derived_inputs(self) -> int:
        return sum(1 for r in self.rows
                   if r["origin_class"] == "derived")

    @property
    def failures(self) -> list[str]:
        return [p for r in self.rows for p in r["problems"]]

    @property
    def clean_inputs(self) -> int:
        return sum(1 for r in self.rows if not r["problems"])

    @property
    def verdict(self) -> str:
        """PASS only on the full filed inventory
        with every input classified and clean;
        FAIL on a full inventory with any
        failure; REVIEW for any partial or
        empty selection, however it scores."""
        if [r["id"] for r in self.rows] != [
                r["id"] for r in HYGIENE_ROWS]:
            return "REVIEW"
        return "FAIL" if self.failures else "PASS"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_data_hygiene(
    row_ids: Any = None,
) -> DataHygieneResult:
    """Evaluate the selection live and file its
    verdict. Any explicit row selection may be
    given — including an empty one, which
    evaluates 0 inputs and files REVIEW (no
    vacuous pass). A derived row also fails by
    name when a row it derives from is itself
    unclean in the same run."""
    if row_ids is None:
        rows = HYGIENE_ROWS
    else:
        wanted = tuple(row_ids)
        rows = tuple(r for r in HYGIENE_ROWS
                     if r["id"] in wanted)
    evaluated = [_evaluate_row(r) for r in rows]
    by_id = {r["id"]: r for r in evaluated}
    full = {r["id"]: r for r in
            (_evaluate_row(r) for r in HYGIENE_ROWS)} \
        if row_ids is not None else by_id
    for row in evaluated:
        sources = row["derived_from"] or (
            ALL_ROW_IDS if row["id"] == "data_hygiene_corpus"
            else ())
        for source_id in sources:
            source = full.get(source_id)
            if source is not None and source["problems"]:
                row["problems"].append(
                    f"{row['id']}: derived from "
                    f"unclean row {source_id} — the "
                    f"derivation inherits the failure")
    return DataHygieneResult(rows=evaluated)


# ── The filing's rules ──────────────────────────────────────────────────────

HYGIENE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "every_benchmark_input_carries_a_hygiene_row",
     "rule": "every harness fixture and every "
             "suite / panel corpus bound through "
             "benchmarks/*.py, plus the bound "
             "neighbour corpora in src/ and the "
             "tests/fixtures estate, files one "
             "HYGIENE_ROWS row — the checker "
             "requires the rows' harness fixtures "
             "to equal the harness's FIXTURES "
             "exactly, so an unregistered fixture "
             "fails by name"},
    {"id": "rows_bind_real_things_never_copy_corpora",
     "rule": "a row binds the real module + "
             "symbol or the real filed paths and "
             "the checker resolves them live — an "
             "empty or missing binding cannot "
             "carry a classification"},
    {"id": "canon_is_never_a_benchmark_input",
     "rule": "no row may file canon as an "
             "origin, and every bound module's "
             "source is scanned live with "
             "TF-017's guard shapes (re-filed "
             "openly in this module): the "
             "repo-anchored canon path, the canon "
             "path helpers, the reference-library "
             "and picture markers, and the canon "
             "persona names — a hit is "
             "contamination, named"},
    {"id": "synthetic_rows_carry_their_proof",
     "rule": "a synthetic-invented row files "
             "its proof kind and its origin: the "
             "harness's synthetic flag, the "
             "module's own synthetic filing, "
             "TF-017's marker on the filed paths, "
             "or the shipped files themselves — "
             "an unproven classification fails"},
    {"id": "licensed_rows_name_their_licence",
     "rule": "a licensed-or-public-domain row "
             "must name the licence or source in "
             "its filing; on this tree the class "
             "files 0 rows (no benchmark input "
             "is licensed material) and the rule "
             "is enforced by probe"},
    {"id": "derived_rows_derive_from_clean_rows",
     "rule": "a derived-live row names the "
             "filed rows it derives from and "
             "inherits their failures in the same "
             "run: a panel over an unclean corpus "
             "is itself unclean, by name"},
    {"id": "pass_needs_the_full_inventory_clean",
     "rule": "the verdict is PASS only when "
             "all 21 filed inputs evaluate "
             "classified and clean; a partial "
             "selection files REVIEW however it "
             "scores and an empty selection "
             "files 0 inputs / REVIEW"},
    {"id": "probes_prove_the_checker_can_fail",
     "rule": "HYGIENE_PROBES files perturbed "
             "filings against the pure checker "
             "parts (canon origin, unproven "
             "licence, missing proof, empty "
             "binding, canon-access source, "
             "unregistered fixture) plus one "
             "clean row that must pass — real "
             "filings are never perturbed"},
    {"id": "the_data_hygiene_baseline_is_not_gated_by_tf129",
     "rule": "TF-129's regression gate gates "
             "the nine TF-121..TF-128 pinned "
             "baselines exactly as filed; this "
             "slice's own baseline "
             "(data_hygiene_synthetic_v1) is "
             "judged by this module's checker, "
             "not gated by the gate — the shape "
             "TF-129 / TF-130 filed for their "
             "own baselines, with the exclusion "
             "filed in the open in "
             "regression_gate.py"},
)


# ── The live checker ────────────────────────────────────────────────────────

def data_hygiene_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # Shape: the filed inventory, its classes, its rules.
    row_ids = [r["id"] for r in HYGIENE_ROWS]
    if len(row_ids) != len(set(row_ids)):
        problems.append("duplicate HYGIENE_ROWS ids")
    if TOTAL_INPUTS != 21 or len(HYGIENE_ROWS) != 21:
        problems.append(
            f"the filed inventory files "
            f"{len(HYGIENE_ROWS)} inputs, not 21")
    classes = [r["origin_class"] for r in HYGIENE_ROWS]
    if (classes.count("synthetic"), classes.count("licensed"),
            classes.count("derived")) != (15, 0, 6):
        problems.append(
            f"the filed classification is "
            f"{classes.count('synthetic')} synthetic / "
            f"{classes.count('licensed')} licensed / "
            f"{classes.count('derived')} derived, "
            f"not 15 / 0 / 6")
    known = set(row_ids)
    for row in HYGIENE_ROWS:
        for source_id in row["derived_from"]:
            if source_id not in known:
                problems.append(
                    f"{row['id']}: derives from "
                    f"unfiled row {source_id}")
    rule_ids = [r["id"] for r in HYGIENE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate HYGIENE_RULES ids")
    for needed in ("every_benchmark_input_carries_a_hygiene_row",
                   "canon_is_never_a_benchmark_input",
                   "synthetic_rows_carry_their_proof",
                   "licensed_rows_name_their_licence",
                   "derived_rows_derive_from_clean_rows",
                   "pass_needs_the_full_inventory_clean",
                   "probes_prove_the_checker_can_fail",
                   "the_data_hygiene_baseline_is_not_gated_by_tf129"):
        if needed not in rule_ids:
            problems.append(f"HYGIENE_RULES lost {needed}")

    # Coverage: the rows' harness fixtures equal
    # the harness's FIXTURES exactly.
    from benchmarks.harness import FIXTURES, baseline_problems

    problems.extend(coverage_problems(
        [r["harness_fixture"] for r in HYGIENE_ROWS
         if r["harness_fixture"] is not None],
        [f["id"] for f in FIXTURES]))

    # The whole benchmark code estate scans
    # clean, this module included.
    benchmarks_dir = Path(__file__).resolve().parent
    for path in sorted(benchmarks_dir.glob("*.py")):
        hits = scan_source_text(path.read_text(encoding="utf-8"))
        if hits:
            problems.append(
                f"benchmarks/{path.name}: canon "
                f"contamination — {hits} (TF-017)")

    # The live run: full PASS, partial / empty REVIEW.
    result = run_data_hygiene()
    if result.verdict != "PASS" or not result.passes:
        problems.append(
            f"the live inventory does not file PASS: "
            f"{result.verdict} ({len(result.failures)} "
            f"failures: {result.failures[:3]})")
    if result.total_inputs != 21 or result.clean_inputs != 21:
        problems.append(
            f"the live inventory files "
            f"{result.total_inputs} inputs / "
            f"{result.clean_inputs} clean, not 21 / 21")
    empty = run_data_hygiene(row_ids=())
    if empty.total_inputs != 0 or empty.verdict != "REVIEW" \
            or empty.passes:
        problems.append("an empty selection does not file 0 / REVIEW")
    if row_ids:
        subset = run_data_hygiene(row_ids=(row_ids[0],))
        if subset.total_inputs != 1 or subset.verdict != "REVIEW":
            problems.append(
                "a single-input selection does not "
                "evaluate its input / REVIEW mechanically")

    # The probes all behave.
    probes = run_hygiene_probes()
    if len(probes) != 7 or not all(p["passed"] for p in probes):
        problems.append(
            f"the hygiene probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The doc files the inventory.
    doc = _REPO_ROOT / DATA_HYGIENE_DOC
    if not doc.exists():
        problems.append(f"the write-up {DATA_HYGIENE_DOC} is missing")
    else:
        doc_text = doc.read_text(encoding="utf-8")
        for rid in row_ids:
            if rid not in doc_text:
                problems.append(
                    f"the doc {DATA_HYGIENE_DOC} lost "
                    f"the inventory row {rid}")

    # Harness consistency as filed + every pinned
    # baseline preserved exactly (this slice's
    # included), and the gate's gated set untouched.
    from benchmarks.harness import BASELINES, BENCHMARKS
    from benchmarks.regression_gate import GATED_BASELINE_IDS

    if "data_hygiene" not in [b["id"] for b in BENCHMARKS]:
        problems.append("BENCHMARKS lost the data_hygiene entry")
    if "data_hygiene_corpus" not in [f["id"] for f in FIXTURES]:
        problems.append("FIXTURES lost the data_hygiene_corpus entry")
    if DATA_HYGIENE_BASELINE_ID not in [b["id"] for b in BASELINES]:
        problems.append("BASELINES lost the data-hygiene baseline")
    if DATA_HYGIENE_BASELINE_ID in GATED_BASELINE_IDS:
        problems.append(
            "the data-hygiene baseline entered TF-129's "
            "gated set (it is judged by this checker, "
            "not gated)")
    if len(GATED_BASELINE_IDS) != 9:
        problems.append("TF-129's gated set changed shape")
    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"harness baselines moved under the data "
            f"hygiene filing: {earlier}")
    return problems


def data_hygiene_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_data_hygiene()
    probes = run_hygiene_probes()
    return {
        "format": DATA_HYGIENE_FORMAT,
        "doc": DATA_HYGIENE_DOC,
        "rows": [{k: v for k, v in r.items()
                  if k != "problems"}
                 for r in result.rows],
        "measured": {
            "total_inputs": result.total_inputs,
            "synthetic_inputs": result.synthetic_inputs,
            "licensed_inputs": result.licensed_inputs,
            "derived_inputs": result.derived_inputs,
            "clean_inputs": result.clean_inputs,
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
            "verdict": result.verdict,
        },
        "probes": probes,
        "rules": [dict(r) for r in HYGIENE_RULES],
    }


__all__ = [
    "ALL_ROW_IDS",
    "CANON_ACCESS_PATTERNS",
    "CANON_PERSONA_NAMES",
    "DATA_HYGIENE_BASELINE_ID",
    "DATA_HYGIENE_DOC",
    "DATA_HYGIENE_FORMAT",
    "HYGIENE_PROBES",
    "HYGIENE_ROWS",
    "HYGIENE_RULES",
    "ORIGIN_CLASSES",
    "PROOF_KINDS",
    "TOTAL_INPUTS",
    "DataHygieneResult",
    "coverage_problems",
    "data_hygiene_as_data",
    "data_hygiene_problems",
    "row_problems",
    "run_data_hygiene",
    "run_hygiene_probes",
    "scan_source_text",
]
