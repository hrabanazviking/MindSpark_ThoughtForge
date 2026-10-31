"""
Scoring honesty: every scoring constant listed, checked, and
tunable per profile (TF-044).

Before this slice the scoring module's own docstring claimed
"no magic numbers — all weights are named constants". That was
half true. The retrieval / prune / survival tables and the
thresholds were named, but:

  * the three writeback formulas carried their weights as bare
    literals inside the function bodies (0.30/0.20/0.25/…),
  * the genericness penalty's 0.15 / 0.6 / 0.4 were bare
    literals too, and the recency fallback half-life was a
    bare 30.0,
  * the knowledge forge's fusion arithmetic — the scoring that
    actually runs on every retrieval — carried its constants
    inline in ``forge.py`` (fts/20, per-source recency proxies,
    the 0.5 neutrals, GeoNames' fixed 0.5, the vector floor
    0.3), and its docstring quoted the BASE weight row while
    the code scored with the ``user_fact`` row (TF-041 caught
    the arithmetic; the docstring still lied),
  * ``configs/default.yaml`` carried a *shadow* copy of the
    retrieval weights with two name-drifted keys
    (``tone_match`` / ``user_preference`` for the code's
    ``tone_similarity`` / ``preference_relevance``, D-CFG-1 /
    TF-009) that no code read — so the one place a user would
    go to tune the weights tuned nothing,
  * and no profile could tune anything: every hardware
    profile scored identically (the retrieval share of
    D-PROF-3).

What this slice does, in order:

  1. **Names everything.** The writeback tables, genericness
     constants, recency fallback, and all forge fusion
     constants now live as named constants in
     ``knowledge/scoring.py`` (values unchanged — TF-041's
     pinned arithmetic and TF-043's pinned baseline are the
     regression proof), the forge imports them, and the
     forge's docstring tells the truth about which weight row
     it scores with. Every scoring function takes an optional
     ``weights`` table (default: the shipped table).
  2. **Lists everything.** ``SCORING_CENSUS`` names every
     weight table, threshold, and scoring constant in the
     knowledge layer, with its role; ``census()`` reads the
     values live. Scoring-shaped constants that live OUTSIDE
     this layer are listed too, with their owners, so nothing
     hides by being elsewhere: the models.py composite
     properties and the core's own fragment/genericness
     formulas are cognition's (Part II territory, TF-022
     ruled the dual fragment systems a recorded duality, not
     a defect to merge here); salvage's fragment weights are
     already named in ``refinement/salvage.py`` and match
     the config file exactly.
  3. **Checks it, mechanically.** ``scoring_problems()``
     re-proves the census live: every weight table sums to
     1.0, the thresholds stand at their pinned values, an AST
     scan finds no numeric literal in any scoring function
     body outside the stated structural set (0 / 0.0 / the
     0.5 decay base / 1 / 1.0 — definitions, not weights),
     the forge's fusion code references the named constants,
     the config file's drift is exactly the recorded drift,
     and the models.py composites still sum as censused.
  4. **Makes it tunable per profile.** ``weights_for_profile``
     resolves a full per-profile weight set: the shipped
     tables with the profile's stated shift applied. The
     shift moves weight from ``semantic_similarity`` to
     ``importance`` on the two smallest profiles only
     (phone_low 0.05, pi_zero 0.10; every other profile 0.0 —
     the shipped tables stand). The hypothesis behind the
     shift: on the SQL path the forge actually runs, the
     semantic dimension is an FTS proxy (``min(fts/20, 1)``)
     while importance is the corpus's own popularity/weight
     signal, and the smallest profiles return the fewest
     records, so leaning on the corpus signal might rank
     better there. **Measured on TF-043's fixtures, the
     hypothesis fails**: ``PROFILE_BASELINE_IMPACT`` records
     precision 0.8519 for both shifted profiles against the
     baseline 0.8889, recall and MRR unchanged — the shift
     costs precision and buys nothing on the only corpus
     with known answers. The shifts stay recorded as the
     tuning surface's first experiment, labelled with their
     measured cost, NOT as an improvement; whether any
     nonzero shift earns its place is TF-064's gate call,
     with this table as evidence. The zero-shift profiles —
     including this VM's desktop_cpu — measure exactly the
     TF-043 baseline, which is also this slice's proof that
     the shipped scoring did not move. A forge built
     with ``retrieval_weights=weights_for_profile(pid,
     "user_fact")`` scores with that profile's table; a forge
     built without one scores exactly as before.
     ``ThoughtForgeCore`` does not select a scoring profile
     yet — the core's retrieval still runs the flat default
     (the retrieval-top_k share of D-PROF-3 stays with
     TF-098; this slice builds the tuning surface and wires
     it at the forge, the layer that owns the arithmetic).
  5. **Closes the TF-043 findings' scoring half honestly.**
     With every constant named, TF-043's two findings are
     arithmetic, not mysteries: a GeoNames record's composite
     is ``GEONAMES_COMPOSITE`` (0.285) at the shipped
     ``user_fact`` weights — under ``MIN_MEMORY_SCORE`` by
     construction, because its fixed fts score (0.5) passes
     through the semantic proxy at 0.025. Naming it does not
     fix it: whether GeoNames should score differently is a
     retrieval-design decision, and TF-064's gate (with
     TF-059 freshness and the Part III remainder) owns
     judging it against the pinned baseline. The ConceptNet
     single-term firing is likewise unchanged: its composite
     is corpus weight through the same named arithmetic.
     Nothing in this slice retunes a single shipped value.

Docket: D-CFG-1's scoring share is CLOSED by this slice in
TF-016 (the weights are explicit, checked, and tunable
through code; the yaml file remains a shadow copy by the
TF-032 live-keys ruling — its drift is now pinned by a
checker instead of discovered by an audit). D-PROF-3 is
narrowed again (retrieval *scoring* is per-profile tunable;
retrieval top_k, history, and temperatures stay flat for
TF-098).

Proof grade: FIXTURE (TF-043's synthetic corpus, SQL path,
this VM) for the measured impact table; the census/checker
proofs are live-tree proofs. Live-corpus tuning NOT PROVEN,
never claimed.

The human-readable write-up is
``docs/architecture/SCORING_HONESTY.md``.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

from thoughtforge.knowledge import scoring as _scoring

# ── The census: every scoring constant in the knowledge layer ───────────────
#
# (name in knowledge/scoring.py, kind, role). Values are read live
# by census(); this table is the *list* — the promise that nothing
# scoring-shaped exists in the layer without appearing here.

SCORING_CENSUS: tuple[tuple[str, str, str], ...] = (
    ("RETRIEVAL_WEIGHTS_BASE", "weights",
     "retrieval weights for a record type with no by-type row"),
    ("RETRIEVAL_WEIGHTS_BY_TYPE", "weights",
     "retrieval weights per memory record type (5 rows)"),
    ("PRUNE_SCORE_WEIGHTS", "weights", "prune-score weights (lifecycle)"),
    ("SURVIVAL_SCORE_WEIGHTS", "weights", "survival-score weights (lifecycle)"),
    ("PREFERENCE_WRITEBACK_WEIGHTS", "weights",
     "preference writeback weights — named by TF-044 (were inline literals)"),
    ("EPISODE_WRITEBACK_WEIGHTS", "weights",
     "episode writeback weights — named by TF-044 (were inline literals)"),
    ("PATTERN_LEARNING_WEIGHTS", "weights",
     "pattern learning weights — named by TF-044 (were inline literals)"),
    ("MIN_MEMORY_SCORE", "threshold", "forge default min_score / bundle gate"),
    ("MIN_FRAGMENT_SCORE", "threshold", "fragment keep gate (core/salvage)"),
    ("MIN_FINAL_RESPONSE_SCORE", "threshold", "final response quality gate"),
    ("MIN_PREFERENCE_WRITEBACK_SCORE", "threshold", "preference writeback gate"),
    ("MIN_EPISODE_WRITEBACK_SCORE", "threshold", "episode writeback gate"),
    ("MIN_PATTERN_LEARNING_SCORE", "threshold", "pattern learning gate"),
    ("HALF_LIFE_DAYS", "decay", "recency half-lives per record type (days)"),
    ("DEFAULT_HALF_LIFE_DAYS", "decay",
     "recency fallback half-life — named by TF-044 (was a bare 30.0)"),
    ("REINFORCE_CONFIDENCE_DELTA", "reinforcement", "per-reinforcement confidence step"),
    ("REINFORCE_IMPORTANCE_DELTA", "reinforcement", "per-reinforcement importance step"),
    ("REINFORCE_QUALITY_DELTA", "reinforcement", "per-reinforcement quality step"),
    ("GENERICNESS_PHRASE_PENALTY", "genericness",
     "penalty per matched generic phrase — named by TF-044"),
    ("GENERICNESS_PHRASE_PENALTY_CAP", "genericness",
     "phrase share cap — named by TF-044"),
    ("GENERICNESS_REPETITION_WEIGHT", "genericness",
     "repetition ratio's share — named by TF-044"),
    ("GENERICNESS_PHRASES", "genericness",
     "the phrase set the genericness penalty matches (11 phrases)"),
    ("FTS_SCORE_DIVISOR", "fusion",
     "semantic proxy divisor: min(fts / this, 1) — named by TF-044"),
    ("SOURCE_RECENCY_PROXIES", "fusion",
     "per-source recency proxies for static KG records — named by TF-044"),
    ("DEFAULT_SOURCE_RECENCY", "fusion", "recency proxy for an unlisted source"),
    ("NEUTRAL_TONE_SIMILARITY", "fusion",
     "tone dimension for DB records (no tone metadata) — named by TF-044"),
    ("NEUTRAL_PREFERENCE_RELEVANCE", "fusion",
     "preference dimension for DB records — named by TF-044"),
    ("IMPORTANCE_WEIGHT_DEFAULT", "fusion",
     "importance proxy's record.get('weight', this) default — named by TF-044"),
    ("GEONAMES_FIXED_FTS_SCORE", "fusion",
     "the GeoNames pass's fixed fts score — named by TF-044; see GEONAMES_COMPOSITE"),
    ("VECTOR_MIN_SCORE", "fusion",
     "the vector pass's similarity floor — named by TF-044"),
)

# Scoring-shaped constants OUTSIDE the knowledge layer, listed so
# the census is complete, with the owner that keeps them. Values
# are checked live where a checker can reach them (models
# composites); the core's formulas are Part II cognition, listed
# here as a pointer, not re-pinned (TF-039 reconciled that doc).
EXTERNAL_SCORING: tuple[tuple[str, str, str], ...] = (
    ("models.CandidateScores.composite", "cognition composite",
     "0.28/0.18/0.16/0.16/0.14/0.08 − 0.10·genericness (knowledge/models.py)"),
    ("models.FragmentScores.composite", "cognition composite",
     "0.30/0.18/0.18/0.20/0.14 − 0.10·genericness (knowledge/models.py)"),
    ("models.FinalResponseScores.composite", "cognition composite",
     "0.24/0.18/0.18/0.14/0.16/0.10 − 0.10·genericness (knowledge/models.py)"),
    ("salvage._LENGTH_WEIGHT/_CITATION_WEIGHT", "fragment scoring",
     "0.45/0.55, already named in refinement/salvage.py; matches default.yaml"),
    ("core._genericness_penalty", "cognition heuristic",
     "the core's own 0.2/phrase genericness (Part II; a different formula "
     "from scoring.compute_genericness_penalty — TF-022's recorded duality)"),
    ("lifecycle.ARCHIVE/DELETE_THRESHOLDS + STORE_CAPS", "lifecycle tables",
     "named tables in knowledge/lifecycle.py, mapped by TF-041, ruled by TF-045"),
)

# The only numeric literals allowed inside scoring.py function
# bodies. These are definitions and structural values, never
# weights: 0 / 0.0 (zero defaults, counts, empty results), the
# 0.5 decay base (the half-life formula IS 0.5 ** (age/half)),
# 1 / 1.0 (unit caps, full scores, counting). Any other literal
# found by the AST scan is an unnamed scoring constant.
STRUCTURAL_LITERALS: dict[float, str] = {
    0: "zero default / count / empty result",
    0.0: "zero default / empty result",
    0.5: "the recency decay base — the half-life definition itself",
    1: "counting unit",
    1.0: "unit cap / full score",
}

# The forge fusion functions must reference these named constants
# (AST-checked) — the wiring proof that the forge scores with the
# census, not with a second private copy of the numbers.
FORGE_FUSION_NAMES: dict[str, tuple[str, ...]] = {
    "_score_records": (
        "FTS_SCORE_DIVISOR", "SOURCE_RECENCY_PROXIES",
        "DEFAULT_SOURCE_RECENCY", "NEUTRAL_TONE_SIMILARITY",
        "NEUTRAL_PREFERENCE_RELEVANCE", "IMPORTANCE_WEIGHT_DEFAULT",
    ),
    "sql_retrieve": ("GEONAMES_FIXED_FTS_SCORE",),
    "vector_retrieve": ("VECTOR_MIN_SCORE",),
}

# configs/default.yaml's scoring sections, as TF-009 measured
# them: the pruning/survival/fragment tables equal the code's
# exactly; the retrieval row carries exactly two name-drifted
# keys. scoring_problems() pins this drift by name — the file
# stays a shadow copy (TF-032's live-keys ruling), but its drift
# can never again be discovered by accident.
CONFIG_DRIFT_KEYS: dict[str, str] = {
    "tone_match": "tone_similarity",
    "user_preference": "preference_relevance",
}

# ── Per-profile tuning ───────────────────────────────────────────────────────
#
# The stated shift per shipped profile: weight moved from
# semantic_similarity to importance in every retrieval row.
# Zero for the desktop/server profiles — the shipped tables are
# their tuning, stated explicitly rather than inherited silently.

PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT: dict[str, float] = {
    "server_gpu": 0.0,
    "desktop_gpu": 0.0,
    "desktop_cpu": 0.0,
    "pi_5": 0.0,
    "phone_low": 0.05,
    "pi_zero": 0.10,
}

# What each profile's resolved weights measure on TF-043's
# known-answer fixtures (SQL path, top_k 10, min_score 0.45,
# this VM at TF-044 banking). Zero-shift profiles reproduce
# TF-043's pinned baseline exactly; the two shifted profiles
# measure precision 0.8519 — a real, recorded cost, not an
# improvement (see the module docstring). TF-064's gate owns
# judging the shifts against the full Part III evidence.
PROFILE_BASELINE_IMPACT: dict[str, dict[str, float]] = {
    "desktop_cpu": {"mean_precision": 0.8889, "mean_recall": 0.9630,
                    "mean_f1": 0.9111, "mean_reciprocal_rank": 1.0},
    "desktop_gpu": {"mean_precision": 0.8889, "mean_recall": 0.9630,
                    "mean_f1": 0.9111, "mean_reciprocal_rank": 1.0},
    "pi_5": {"mean_precision": 0.8889, "mean_recall": 0.9630,
             "mean_f1": 0.9111, "mean_reciprocal_rank": 1.0},
    "server_gpu": {"mean_precision": 0.8889, "mean_recall": 0.9630,
                   "mean_f1": 0.9111, "mean_reciprocal_rank": 1.0},
    "phone_low": {"mean_precision": 0.8519, "mean_recall": 0.9630,
                  "mean_f1": 0.8889, "mean_reciprocal_rank": 1.0},
    "pi_zero": {"mean_precision": 0.8519, "mean_recall": 0.9630,
                "mean_f1": 0.8889, "mean_reciprocal_rank": 1.0},
}

RETRIEVAL_DIMENSIONS: tuple[str, ...] = (
    "semantic_similarity", "tone_similarity", "preference_relevance",
    "recency", "importance", "thread_boost", "pattern_quality", "priority",
)

# TF-043's GeoNames finding as arithmetic over the named
# constants, at the shipped user_fact row: the fixed fts score
# through the semantic proxy, plus the neutral/proxy dims.
def geonames_composite(weights: dict[str, float] | None = None) -> float:
    """A GeoNames record's composite under a retrieval row.

    At the shipped ``user_fact`` row this is exactly 0.285 —
    TF-043's pinned GEONAMES_DEFAULT_SCORE — which is why the
    GeoNames pass can never clear MIN_MEMORY_SCORE (0.45).
    """
    row = _scoring.RETRIEVAL_WEIGHTS_BY_TYPE["user_fact"] if weights is None else weights
    dims = _scoring.RetrievalDimensions(
        semantic_similarity=min(
            _scoring.GEONAMES_FIXED_FTS_SCORE / _scoring.FTS_SCORE_DIVISOR, 1.0
        ),
        tone_similarity=_scoring.NEUTRAL_TONE_SIMILARITY,
        preference_relevance=_scoring.NEUTRAL_PREFERENCE_RELEVANCE,
        recency=_scoring.SOURCE_RECENCY_PROXIES["geonames"],
        importance=min(0.0 + _scoring.IMPORTANCE_WEIGHT_DEFAULT, 1.0),
    )
    return _scoring.score_retrieval(dims, "user_fact", weights=row)


GEONAMES_COMPOSITE: float = 0.285  # pinned; geonames_composite() re-derives it


def weight_table_problems(table: Any, *, dimensions: tuple[str, ...]) -> list[str]:
    """Validate one retrieval weight table: exact dimension keys,
    finite non-negative values, sums to 1.0."""
    problems: list[str] = []
    if not isinstance(table, dict):
        return [f"weight table is not a dict: {type(table).__name__}"]
    if set(table) != set(dimensions):
        problems.append(
            f"weight table keys {sorted(table)} != dimensions {sorted(dimensions)}"
        )
    for key, value in table.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            problems.append(f"weight {key!r} is not a finite number: {value!r}")
        elif value < 0:
            problems.append(f"weight {key!r} is negative: {value}")
    if not problems and not math.isclose(
        sum(table.values()), 1.0, abs_tol=1e-9
    ):
        problems.append(f"weight table sums to {sum(table.values())}, not 1.0")
    return problems


def weights_for_profile(
    profile_id: str,
    record_type: str | None = None,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Resolve a shipped profile's retrieval weights.

    Returns the full by-type mapping (plus a ``"base"`` row), or
    the single row when ``record_type`` is given (``"base"``
    selects RETRIEVAL_WEIGHTS_BASE). The profile's stated shift
    moves weight from semantic_similarity to importance in every
    row; sums stay 1.0 by construction. An unknown profile raises
    ``ValueError`` naming the valid ids — a tuning never silently
    defaults. The shipped tables are never mutated: every row
    returned is a fresh dict.
    """
    from thoughtforge.utils.config import list_hardware_profiles

    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile: {profile_id!r} "
            f"(valid: {', '.join(sorted(valid))})"
        )
    if profile_id not in PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT:
        raise ValueError(
            f"no scoring shift recorded for profile: {profile_id!r} "
            f"(recorded: {', '.join(sorted(PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT))})"
        )
    shift = PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT[profile_id]

    def _shifted(row: dict[str, float]) -> dict[str, float]:
        out = dict(row)
        if shift:
            out["semantic_similarity"] = row["semantic_similarity"] - shift
            out["importance"] = row["importance"] + shift
        return out

    tables: dict[str, dict[str, float]] = {
        "base": _shifted(_scoring.RETRIEVAL_WEIGHTS_BASE),
    }
    # RETRIEVAL_WEIGHTS_BASE carries only the 5 core dimensions;
    # pad the base row to the full dimension set with the zeros
    # score_retrieval's .get semantics already imply.
    for dim in RETRIEVAL_DIMENSIONS:
        tables["base"].setdefault(dim, 0.0)
    for rtype, row in _scoring.RETRIEVAL_WEIGHTS_BY_TYPE.items():
        tables[rtype] = _shifted(row)

    if record_type is None:
        return tables
    if record_type not in tables:
        raise ValueError(
            f"unknown record type: {record_type!r} "
            f"(valid: {', '.join(sorted(tables))})"
        )
    return tables[record_type]


def census() -> list[dict[str, Any]]:
    """The scoring census with values read live from the tree."""
    rows: list[dict[str, Any]] = []
    for name, kind, role in SCORING_CENSUS:
        value = getattr(_scoring, name)
        row: dict[str, Any] = {"name": name, "kind": kind, "role": role}
        if isinstance(value, dict) and value and all(
            isinstance(v, (int, float)) for v in value.values()
        ):
            row["sum"] = sum(value.values())
        elif isinstance(value, dict):
            row["rows"] = {
                k: (sum(v.values()) if isinstance(v, dict) else v)
                for k, v in value.items()
            }
        else:
            row["value"] = value
        rows.append(row)
    return rows


def _ast_problems() -> list[str]:
    problems: list[str] = []
    src_root = Path(_scoring.__file__).resolve().parent

    scoring_tree = ast.parse((src_root / "scoring.py").read_text())
    for node in ast.walk(scoring_tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, (int, float))
                    and not isinstance(sub.value, bool)
                    and float(sub.value) not in STRUCTURAL_LITERALS
                ):
                    problems.append(
                        f"scoring.{node.name} holds unnamed literal "
                        f"{sub.value!r} (line {sub.lineno}) — name it in "
                        f"scoring.py and list it in SCORING_CENSUS"
                    )

    forge_tree = ast.parse((src_root / "forge.py").read_text())
    forge_funcs = {
        n.name: n for n in ast.walk(forge_tree) if isinstance(n, ast.FunctionDef)
    }
    for func_name, names in FORGE_FUSION_NAMES.items():
        fn = forge_funcs.get(func_name)
        if fn is None:
            problems.append(f"forge.{func_name} is missing")
            continue
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
        }
        for name in names:
            if name not in used:
                problems.append(
                    f"forge.{func_name} does not reference {name} — "
                    f"the fusion constant is not wired"
                )
    return problems


def _config_problems() -> list[str]:
    problems: list[str] = []
    try:
        import yaml
    except ImportError:
        return ["pyyaml unavailable — config drift cannot be checked"]
    cfg_path = (
        Path(_scoring.__file__).resolve().parents[3] / "configs" / "default.yaml"
    )
    if not cfg_path.exists():
        return [f"configs/default.yaml missing at {cfg_path}"]
    cfg = yaml.safe_load(cfg_path.read_text())

    retrieval = cfg.get("retrieval", {}).get("scoring_weights", {})
    base = _scoring.RETRIEVAL_WEIGHTS_BASE
    for key, value in retrieval.items():
        code_key = CONFIG_DRIFT_KEYS.get(key, key)
        if code_key not in base:
            problems.append(f"default.yaml retrieval weight {key!r} has no code row")
        elif base[code_key] != value:
            problems.append(
                f"default.yaml retrieval weight {key!r} = {value} "
                f"!= code {code_key} = {base[code_key]}"
            )
    drifted = {k for k in retrieval if k in CONFIG_DRIFT_KEYS}
    if drifted != set(CONFIG_DRIFT_KEYS):
        problems.append(
            f"default.yaml drift keys {sorted(drifted)} != recorded "
            f"{sorted(CONFIG_DRIFT_KEYS)}"
        )

    memory = cfg.get("memory", {})
    if memory.get("pruning_score_weights") != _scoring.PRUNE_SCORE_WEIGHTS:
        problems.append("default.yaml pruning weights != PRUNE_SCORE_WEIGHTS")
    if memory.get("survival_score_weights") != _scoring.SURVIVAL_SCORE_WEIGHTS:
        problems.append("default.yaml survival weights != SURVIVAL_SCORE_WEIGHTS")

    from thoughtforge.refinement import salvage as _salvage

    fragment = cfg.get("refinement", {}).get("fragment_scoring_weights", {})
    if fragment.get("length_score") != _salvage._LENGTH_WEIGHT:
        problems.append("default.yaml fragment length weight != salvage._LENGTH_WEIGHT")
    if fragment.get("citation_score") != _salvage._CITATION_WEIGHT:
        problems.append("default.yaml fragment citation weight != salvage._CITATION_WEIGHT")
    return problems


def scoring_problems() -> list[str]:
    """Re-prove the census against the live tree. ``[]`` = honest."""
    problems: list[str] = []

    # Every censused name exists in the scoring module.
    for name, _kind, _role in SCORING_CENSUS:
        if not hasattr(_scoring, name):
            problems.append(f"census name {name} missing from knowledge/scoring.py")

    # Every weight table sums to 1.0 (the five-dimension base row
    # is judged on its own keys; by-type rows on the full set).
    tables: list[tuple[str, dict[str, float], tuple[str, ...]]] = [
        ("RETRIEVAL_WEIGHTS_BASE", _scoring.RETRIEVAL_WEIGHTS_BASE,
         tuple(_scoring.RETRIEVAL_WEIGHTS_BASE)),
        ("PRUNE_SCORE_WEIGHTS", _scoring.PRUNE_SCORE_WEIGHTS,
         tuple(_scoring.PRUNE_SCORE_WEIGHTS)),
        ("SURVIVAL_SCORE_WEIGHTS", _scoring.SURVIVAL_SCORE_WEIGHTS,
         tuple(_scoring.SURVIVAL_SCORE_WEIGHTS)),
        ("PREFERENCE_WRITEBACK_WEIGHTS", _scoring.PREFERENCE_WRITEBACK_WEIGHTS,
         tuple(_scoring.PREFERENCE_WRITEBACK_WEIGHTS)),
        ("EPISODE_WRITEBACK_WEIGHTS", _scoring.EPISODE_WRITEBACK_WEIGHTS,
         tuple(_scoring.EPISODE_WRITEBACK_WEIGHTS)),
        ("PATTERN_LEARNING_WEIGHTS", _scoring.PATTERN_LEARNING_WEIGHTS,
         tuple(_scoring.PATTERN_LEARNING_WEIGHTS)),
    ]
    for rtype, row in _scoring.RETRIEVAL_WEIGHTS_BY_TYPE.items():
        tables.append((f"RETRIEVAL_WEIGHTS_BY_TYPE[{rtype}]", row, RETRIEVAL_DIMENSIONS))
    for label, table, dims in tables:
        for problem in weight_table_problems(table, dimensions=dims):
            problems.append(f"{label}: {problem}")

    # The thresholds stand where TF-041's map pinned them.
    expected_thresholds = {
        "MIN_MEMORY_SCORE": 0.45, "MIN_FRAGMENT_SCORE": 0.54,
        "MIN_FINAL_RESPONSE_SCORE": 0.68, "MIN_PREFERENCE_WRITEBACK_SCORE": 0.72,
        "MIN_EPISODE_WRITEBACK_SCORE": 0.58, "MIN_PATTERN_LEARNING_SCORE": 0.78,
    }
    for name, value in expected_thresholds.items():
        if getattr(_scoring, name) != value:
            problems.append(f"threshold {name} moved from {value}")

    # The genericness constants are the values the formula shipped with.
    if (
        _scoring.GENERICNESS_PHRASE_PENALTY != 0.15
        or _scoring.GENERICNESS_PHRASE_PENALTY_CAP != 0.6
        or _scoring.GENERICNESS_REPETITION_WEIGHT != 0.4
    ):
        problems.append("genericness constants moved from 0.15 / 0.6 / 0.4")

    # TF-043's GeoNames arithmetic, re-derived from the named constants.
    if not math.isclose(geonames_composite(), GEONAMES_COMPOSITE, abs_tol=1e-9):
        problems.append(
            f"geonames composite re-derives to {geonames_composite()}, "
            f"not the pinned {GEONAMES_COMPOSITE}"
        )
    if not geonames_composite() < _scoring.MIN_MEMORY_SCORE:
        problems.append("geonames composite is no longer under MIN_MEMORY_SCORE")

    # Every shipped profile resolves and validates.
    from thoughtforge.utils.config import list_hardware_profiles

    for profile_id in list_hardware_profiles():
        if profile_id not in PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT:
            problems.append(f"profile {profile_id} has no recorded scoring shift")
            continue
        resolved = weights_for_profile(profile_id)
        assert isinstance(resolved, dict)
        for rtype, row in resolved.items():
            for problem in weight_table_problems(row, dimensions=RETRIEVAL_DIMENSIONS):
                problems.append(f"profile {profile_id}/{rtype}: {problem}")

    # The models.py composites sum as censused (live instantiation).
    from thoughtforge.knowledge.models import (
        CandidateScores,
        FinalResponseScores,
        FragmentScores,
    )

    for cls, fields in (
        (CandidateScores, ("relevance", "clarity", "coherence",
                           "personality_fit", "specificity", "goal_fit")),
        (FragmentScores, ("relevance", "clarity", "specificity",
                          "usefulness", "personality_fit")),
        (FinalResponseScores, ("relevance", "clarity", "coherence",
                               "personality_fit", "usefulness", "goal_fit")),
    ):
        total = sum(cls(**{f: 1.0}).composite for f in fields)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            problems.append(f"{cls.__name__}.composite weights sum to {total}, not 1.0")
        if not math.isclose(cls(genericness_penalty=1.0).composite, -0.10, abs_tol=1e-9):
            problems.append(f"{cls.__name__}.composite genericness share != -0.10")

    problems.extend(_ast_problems())
    problems.extend(_config_problems())
    return problems
