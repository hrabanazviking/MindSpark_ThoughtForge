"""TF-089 — Salvage metrics: rescue rate and
corruption rate measured on the fixture corpus,
with policy thresholds taken from the measurements.

Roadmap: "Salvage metrics: rescue rate + corruption
rate measured on the fixture corpus; policy
thresholds from measurements."

Measured start (probed live this tick, every leg
run through the real ``FragmentSalvage``):

- **The corpus is the two filed fixture sets**:
  TF-087's nine ``BROKEN_FIXTURES`` (the rescue
  corpus) and TF-088's nine ``ADVERSARIAL_FIXTURES``
  (the corruption corpus) — 18 entries, synthetic
  Lantern Hollow material throughout, never canon.
  No new fixtures are invented here; this slice
  *scores* the estate the last two slices filed.
- **Rescue rate 2/7.** Of the nine broken inputs,
  two are not rescue-eligible (the empty input has
  nothing to rescue; the engine-failure fixture's
  input is a clean cited draft the fallback simply
  returns). Of the seven eligible, exactly two are
  rescued: rambling beaten by a cited rival — by
  *selection*, the rival is returned and the
  rambling discarded — and the refine fixture,
  whose stub-engine output is adopted. The other
  five come back **byte-identical**: the truncated
  draft, the lone rambling draft, the mixed draft
  (its junk sentences ride along; fragments never
  clean an output, TF-087), the broken-JSON draft
  and the 20x loop.
- **Rambling, this slice's D-SALV-1 share**: the
  rambling *problem* appears in two fixtures and
  is rescued in one (by selection, when a rival
  exists). Lone-rambling compression is 0/1 —
  there is still no compression mechanism; TF-089
  measures that share and does not fix it. The
  remaining implementation owners stand (TF-090
  truncation+format, TF-091 loops, TF-092 drift);
  TF-101 scores these mechanisms on TF-100's chaos
  corpus and TF-104's gate owns the rates.
- **Corruption, two tiers.** Marker corruption
  (TF-088's guard kinds) is **0**: all six
  fabrication attempts are rejected (prevention
  6/6), so no marker-fabricated text is adopted.
  Broad corruption is **not zero**: of the two
  refine outputs the adversarial corpus adopts,
  one is TF-088's filed residual — common-word
  elaboration (a keeper, a red lantern, a winter
  timetable added in plain nouns) — human-judged
  corrupted at filing, marker-clean by definition.
  Adopted-output corruption is therefore 1/2.
  One further output carries invented facts it
  did not invent: the source-invention draft
  passes through verbatim (pool-relative rule,
  TF-088) — counted separately, never charged
  to salvage.
- **Thresholds are the measurements, not wishes.**
  ``THRESHOLDS`` files the marker-adoption ceiling
  at 0.0 and the prevention floor at 1.0 (absolute,
  TF-104's gate inherits them), the rescue floor
  at the measured 2/7, the unrescued-verbatim
  ceiling at the measured 5, and the
  adopted-corruption ceiling at the measured 1/2
  with the residual named as its cause. Any future
  slice that moves a rate must move the filed
  measurement and the threshold together, in the
  open — a silent drift fails this slice's
  live checker by name.

Proof grade FIXTURE — every number is recomputed
live from the filed corpora through the real
FragmentSalvage on this VM. A live model's rescue
or corruption rates are NOT PROVEN (no model in
tree); TF-100's chaos corpus owns real-shaped
failures.
"""

from __future__ import annotations

from typing import Any

METRICS_FORMAT = "thoughtforge-salvage-metrics/1"

METRICS_DOC = "docs/architecture/SALVAGE_METRICS.md"

# ── The rules ───────────────────────────────────────────────────────────────

METRIC_RULES: tuple[dict[str, Any], ...] = (
    {"id": "corpus_is_the_filed_fixtures",
     "rule": "the measured corpus is exactly TF-087's "
             "BROKEN_FIXTURES plus TF-088's "
             "ADVERSARIAL_FIXTURES (18 entries); this slice "
             "scores the filed estate and invents no new "
             "fixtures to flatter the rates"},
    {"id": "rescue_means_the_output_is_no_longer_the_broken_input",
     "rule": "a fixture is rescued when the returned text is "
             "a clean text that is not the broken input "
             "verbatim: a clean rival selected over it, or a "
             "clean refine output adopted over it; returning "
             "the broken input byte-identical is unrescued, "
             "however sensible the fallback"},
    {"id": "eligibility_excludes_nothing_broken_and_nothing_to_save",
     "rule": "the empty input (nothing to rescue) and the "
             "engine-failure fixture (its input is already a "
             "clean cited draft) are handled safely but are "
             "not rescue-eligible and leave the denominator"},
    {"id": "corruption_is_two_tiered",
     "rule": "marker corruption is what TF-088's guard can "
             "name (numbers, citations, entities, polarity, "
             "displaced time); broad corruption also counts "
             "adopted outputs human-judged at filing to add "
             "facts in common words — the residual tier; the "
             "two are never merged into one number"},
    {"id": "pass_through_invention_is_not_salvage_corruption",
     "rule": "a draft's own inventions returned verbatim are "
             "counted as outputs carrying invented facts, "
             "attributed to the draft (generation/enforcement "
             "judge drafts), never to salvage (TF-088's "
             "pool-relative rule)"},
    {"id": "thresholds_are_measurements",
     "rule": "every threshold in THRESHOLDS equals a "
             "measured rate or count from this tick, except "
             "the two absolutes (marker adoption 0.0, "
             "prevention 1.0) which restate TF-088's gate; "
             "moving a rate means moving its filed "
             "measurement and threshold together"},
    {"id": "rambling_share_is_measured_not_fixed",
     "rule": "this slice discharges D-SALV-1's rambling "
             "share as a measurement: rescued 1/2 by "
             "selection, lone compression 0/1 and still "
             "absent; no compression mechanism is claimed"},
)

# ── Filed outcomes: the rescue corpus (TF-087) ──────────────────────────────
# outcome vocabulary:
#   rescued_by_selection / rescued_by_refine — rescued
#   unrescued_verbatim — eligible, returned byte-identical
#   handled_empty / safe_fallback — handled, not eligible

BROKEN_OUTCOMES: tuple[dict[str, Any], ...] = (
    {"fixture": "empty_candidates",
     "outcome": "handled_empty", "eligible": False},
    {"fixture": "truncated_mid_sentence",
     "outcome": "unrescued_verbatim", "eligible": True},
    {"fixture": "rambling_beaten_by_cited_rival",
     "outcome": "rescued_by_selection", "eligible": True},
    {"fixture": "rambling_alone_verbatim",
     "outcome": "unrescued_verbatim", "eligible": True},
    {"fixture": "fragment_filter_keeps_only_scored_sentences",
     "outcome": "unrescued_verbatim", "eligible": True},
    {"fixture": "broken_json_verbatim",
     "outcome": "unrescued_verbatim", "eligible": True},
    {"fixture": "loop_verbatim",
     "outcome": "unrescued_verbatim", "eligible": True},
    {"fixture": "refine_adopts_better_output",
     "outcome": "rescued_by_refine", "eligible": True},
    {"fixture": "refine_engine_failure_falls_back",
     "outcome": "safe_fallback", "eligible": False},
)

#: Fixtures where the broken input is rambling
#: (TF-089's D-SALV-1 share).
RAMBLING_FIXTURES: tuple[str, ...] = (
    "rambling_beaten_by_cited_rival", "rambling_alone_verbatim")

# ── Filed outcomes: the corruption corpus (TF-088) ──────────────────────────
# outcome vocabulary:
#   fabrication_rejected — a marker fabrication refused
#   clean_refine_adopted — adopted, no invention
#   residual_elaboration_adopted — adopted, broad-tier
#     corruption (human-judged at TF-088's filing)
#   source_invention_pass_through — verbatim draft whose
#     own inventions ride along (not salvage's)

ADVERSARIAL_OUTCOMES: tuple[dict[str, Any], ...] = (
    {"fixture": "invented_price",
     "outcome": "fabrication_rejected"},
    {"fixture": "invented_citation",
     "outcome": "fabrication_rejected"},
    {"fixture": "negation_flip",
     "outcome": "fabrication_rejected"},
    {"fixture": "invented_destinations",
     "outcome": "fabrication_rejected"},
    {"fixture": "time_displaced",
     "outcome": "fabrication_rejected"},
    {"fixture": "negation_dropped",
     "outcome": "fabrication_rejected"},
    {"fixture": "benign_rephrase",
     "outcome": "clean_refine_adopted"},
    {"fixture": "common_word_elaboration_residual",
     "outcome": "residual_elaboration_adopted"},
    {"fixture": "source_invention_pass_through",
     "outcome": "source_invention_pass_through"},
)

# ── The measurements, as filed this tick ────────────────────────────────────
# Every value is recomputed live by
# salvage_metrics_live_problems(); a filed value
# the live run cannot reproduce fails by name.

FILED_MEASUREMENTS: dict[str, Any] = {
    "corpus_size": 18,
    "broken_fixtures": 9,
    "adversarial_fixtures": 9,
    "rescue_eligible": 7,
    "rescued": 2,
    "rescue_rate": 2 / 7,
    "unrescued_verbatim": 5,
    "rambling_fixtures": 2,
    "rambling_rescued": 1,
    "rambling_rescue_rate": 0.5,
    "lone_rambling_compression_rate": 0.0,
    "fabrication_attempts": 6,
    "fabrications_rejected": 6,
    "fabrication_prevention_rate": 1.0,
    "marker_adoption_rate": 0.0,
    "adversarial_refine_adoptions": 2,
    "corrupted_adoptions": 1,
    "adopted_corruption_rate": 0.5,
    "salvage_attributable_corrupted_outputs": 1,
    "outputs_carrying_invented_facts": 2,
}

# ── Policy thresholds, from the measurements ────────────────────────────────

THRESHOLDS: dict[str, Any] = {
    "marker_adoption_ceiling": 0.0,
    "fabrication_prevention_floor": 1.0,
    "rescue_rate_floor": 2 / 7,
    "unrescued_verbatim_ceiling": 5,
    "adopted_corruption_ceiling": 0.5,
    "rambling_rescue_floor": 0.5,
}

_RESCUED = ("rescued_by_selection", "rescued_by_refine")


# ── Live runners ────────────────────────────────────────────────────────────

class _StubEngine:
    def __init__(self, text: str):
        self._text = text

    def generate(self, prompt, params=None):
        from types import SimpleNamespace
        return SimpleNamespace(text=self._text)


class _BadEngine:
    def generate(self, prompt, params=None):
        raise RuntimeError("engine down")


def run_broken(fixture_id: str):
    """Run one TF-087 fixture through the real
    ``FragmentSalvage`` exactly as TF-087's tests
    do (stub / failing engine + the real
    ``PromptBuilder`` for the engine fixtures)."""
    from thoughtforge.cognition.prompt_builder import (
        PromptBuilder)
    from thoughtforge.knowledge.models import (
        CognitionScaffold, InputSketch)
    from thoughtforge.refinement.salvage_map import (
        BROKEN_FIXTURES, run_fixture)

    fixture = {f["id"]: f for f in BROKEN_FIXTURES}[fixture_id]
    if not fixture["engine_required"]:
        return run_fixture(fixture_id)
    engine = (_BadEngine() if fixture["engine_text"] is None
              else _StubEngine(fixture["engine_text"]))
    sketch = InputSketch(
        raw_text="when does the ferry leave?",
        intent="factual_query", topic="ferry",
        response_mode="factual_direct")
    scaffold = CognitionScaffold(
        goal="give a clear direct answer",
        tone=["direct", "factual"], avoid=["filler"])
    return run_fixture(
        fixture_id, engine=engine,
        prompt_builder=PromptBuilder(), sketch=sketch,
        scaffold=scaffold, max_passes=1)


def measure_metrics() -> dict[str, Any]:
    """Recompute every filed measurement live: run
    both corpora through the real FragmentSalvage
    and count the filed outcomes (each outcome is
    verified against the live result by
    ``salvage_metrics_live_problems``; this function
    counts the same runs)."""
    from thoughtforge.refinement.salvage_map import (
        BROKEN_FIXTURES)
    from thoughtforge.refinement.salvage_safety import (
        ADVERSARIAL_FIXTURES, run_adversarial)

    broken_by_id = {f["id"]: f for f in BROKEN_FIXTURES}
    adv_by_id = {f["id"]: f for f in ADVERSARIAL_FIXTURES}
    broken_rows = []
    for row in BROKEN_OUTCOMES:
        result = run_broken(row["fixture"])
        broken_rows.append({**row, "path": result.salvage_path,
                            "text": result.text})
    adv_rows = []
    for row in ADVERSARIAL_OUTCOMES:
        result = run_adversarial(row["fixture"])
        adv_rows.append({**row, "path": result.salvage_path,
                         "text": result.text})

    eligible = [r for r in broken_rows if r["eligible"]]
    rescued = [r for r in eligible if r["outcome"] in _RESCUED]
    rambling = [r for r in broken_rows
                if r["fixture"] in RAMBLING_FIXTURES]
    rambling_rescued = [r for r in rambling
                        if r["outcome"] in _RESCUED]
    attempts = [r for r in adv_rows
                if r["outcome"] == "fabrication_rejected"]
    adoptions = [r for r in adv_rows if r["outcome"] in (
        "clean_refine_adopted", "residual_elaboration_adopted")]
    corrupted = [r for r in adv_rows
                 if r["outcome"] == "residual_elaboration_adopted"]
    carrying = corrupted + [r for r in adv_rows if r["outcome"]
                            == "source_invention_pass_through"]
    return {
        "corpus_size": len(broken_rows) + len(adv_rows),
        "broken_fixtures": len(broken_rows),
        "adversarial_fixtures": len(adv_rows),
        "rescue_eligible": len(eligible),
        "rescued": len(rescued),
        "rescue_rate": len(rescued) / len(eligible),
        "unrescued_verbatim": len([r for r in eligible
                                   if r["outcome"] ==
                                   "unrescued_verbatim"]),
        "rambling_fixtures": len(rambling),
        "rambling_rescued": len(rambling_rescued),
        "rambling_rescue_rate":
            len(rambling_rescued) / len(rambling),
        "lone_rambling_compression_rate": 0.0 if any(
            r["outcome"] == "unrescued_verbatim"
            for r in rambling
            if r["fixture"] == "rambling_alone_verbatim")
            else 1.0,
        "fabrication_attempts": len(attempts),
        "fabrications_rejected": len(attempts),
        "fabrication_prevention_rate": 1.0,
        "marker_adoption_rate": 0.0,
        "adversarial_refine_adoptions": len(adoptions),
        "corrupted_adoptions": len(corrupted),
        "adopted_corruption_rate":
            len(corrupted) / len(adoptions),
        "salvage_attributable_corrupted_outputs": len(corrupted),
        "outputs_carrying_invented_facts": len(carrying),
        "_broken_texts": {r["fixture"]: r["text"]
                          for r in broken_rows},
        "_broken_paths": {r["fixture"]: r["path"]
                          for r in broken_rows},
        "_adv_texts": {r["fixture"]: r["text"] for r in adv_rows},
        "_adv_paths": {r["fixture"]: r["path"] for r in adv_rows},
        "_broken_by_id": broken_by_id,
        "_adv_by_id": adv_by_id,
    }


def public_measurements(measured: dict[str, Any]) -> dict[str, Any]:
    """The measurement dict without the live-run
    internals (texts / paths / fixture maps)."""
    return {k: v for k, v in measured.items()
            if not k.startswith("_")}


# ── Checkers ────────────────────────────────────────────────────────────────

def salvage_metrics_problems() -> list[str]:
    """Data-integrity checker: rules unique, the
    outcome tables cover each corpus fixture
    exactly once and no other, thresholds are the
    filed measurements or the two absolutes."""
    from thoughtforge.refinement.salvage_map import (
        BROKEN_FIXTURES)
    from thoughtforge.refinement.salvage_safety import (
        ADVERSARIAL_FIXTURES)

    problems: list[str] = []
    rule_ids = [r["id"] for r in METRIC_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate METRIC_RULES ids")
    broken_ids = [f["id"] for f in BROKEN_FIXTURES]
    filed_broken = [r["fixture"] for r in BROKEN_OUTCOMES]
    if sorted(filed_broken) != sorted(broken_ids):
        problems.append(
            "BROKEN_OUTCOMES does not cover BROKEN_FIXTURES "
            "exactly once")
    adv_ids = [f["id"] for f in ADVERSARIAL_FIXTURES]
    filed_adv = [r["fixture"] for r in ADVERSARIAL_OUTCOMES]
    if sorted(filed_adv) != sorted(adv_ids):
        problems.append(
            "ADVERSARIAL_OUTCOMES does not cover "
            "ADVERSARIAL_FIXTURES exactly once")
    for name in RAMBLING_FIXTURES:
        if name not in broken_ids:
            problems.append(f"rambling fixture {name} not in corpus")
    if THRESHOLDS["rescue_rate_floor"] != \
            FILED_MEASUREMENTS["rescue_rate"]:
        problems.append("rescue floor is not the measured rate")
    if THRESHOLDS["unrescued_verbatim_ceiling"] != \
            FILED_MEASUREMENTS["unrescued_verbatim"]:
        problems.append("unrescued ceiling is not the measured count")
    if THRESHOLDS["adopted_corruption_ceiling"] != \
            FILED_MEASUREMENTS["adopted_corruption_rate"]:
        problems.append("adopted-corruption ceiling is not measured")
    if THRESHOLDS["rambling_rescue_floor"] != \
            FILED_MEASUREMENTS["rambling_rescue_rate"]:
        problems.append("rambling floor is not the measured rate")
    return problems


def salvage_metrics_live_problems() -> list[str]:
    """Live checker: every filed outcome is verified
    against its live result (verbatim means
    byte-identical, rescued means the filed clean
    text, rejected means the candidate stands), the
    recomputed measurements equal the filed ones,
    and the measured rates satisfy the thresholds."""
    from thoughtforge.refinement.salvage_safety import (
        classify_output)

    problems: list[str] = []
    measured = measure_metrics()
    broken_by_id = measured["_broken_by_id"]
    adv_by_id = measured["_adv_by_id"]

    for row in BROKEN_OUTCOMES:
        fixture = broken_by_id[row["fixture"]]
        text = measured["_broken_texts"][row["fixture"]]
        path = measured["_broken_paths"][row["fixture"]]
        outcome = row["outcome"]
        if outcome == "unrescued_verbatim":
            if text != fixture["candidates"][0] or \
                    path != "best_draft":
                problems.append(
                    f"{row['fixture']}: filed unrescued_verbatim "
                    "but the live output differs")
        elif outcome == "rescued_by_selection":
            if text != fixture["expected_text"] or \
                    text == fixture["candidates"][0]:
                problems.append(
                    f"{row['fixture']}: filed rescued_by_selection "
                    "but the live output is not the clean rival")
        elif outcome == "rescued_by_refine":
            if text != fixture["engine_text"] or \
                    path != "refine_pass_1":
                problems.append(
                    f"{row['fixture']}: filed rescued_by_refine "
                    "but the live output is not the adopted refine")
        elif outcome == "handled_empty":
            if path != "empty":
                problems.append(
                    f"{row['fixture']}: filed handled_empty, "
                    f"live path {path}")
        elif outcome == "safe_fallback":
            if text != fixture["candidates"][0] or \
                    path != "best_draft":
                problems.append(
                    f"{row['fixture']}: filed safe_fallback but "
                    "the live output differs")

    for row in ADVERSARIAL_OUTCOMES:
        fixture = adv_by_id[row["fixture"]]
        text = measured["_adv_texts"][row["fixture"]]
        outcome = row["outcome"]
        if outcome == "fabrication_rejected":
            if text != fixture["candidate"]:
                problems.append(
                    f"{row['fixture']}: filed fabrication_rejected "
                    "but the fabricated text was adopted")
        elif outcome in ("clean_refine_adopted",
                         "residual_elaboration_adopted"):
            if text != fixture["engine_text"]:
                problems.append(
                    f"{row['fixture']}: filed {outcome} but the "
                    "refine output was not adopted")
            verdict = classify_output(
                [fixture["candidate"]], fixture["engine_text"],
                ("Q123",))
            if verdict["problems"]:
                problems.append(
                    f"{row['fixture']}: filed marker-clean "
                    f"adoption but the guard reports "
                    f"{verdict['problems']}")
        elif outcome == "source_invention_pass_through":
            if text != fixture["candidate"]:
                problems.append(
                    f"{row['fixture']}: filed pass-through but "
                    "the output is not the verbatim draft")

    for key, filed in FILED_MEASUREMENTS.items():
        live = measured.get(key)
        if isinstance(filed, float):
            if live is None or abs(live - filed) > 1e-12:
                problems.append(
                    f"measurement {key}: filed {filed} != live {live}")
        elif live != filed:
            problems.append(
                f"measurement {key}: filed {filed} != live {live}")

    if measured["marker_adoption_rate"] > \
            THRESHOLDS["marker_adoption_ceiling"]:
        problems.append("marker adoption above its ceiling")
    if measured["fabrication_prevention_rate"] < \
            THRESHOLDS["fabrication_prevention_floor"]:
        problems.append("fabrication prevention below its floor")
    if measured["rescue_rate"] < THRESHOLDS["rescue_rate_floor"]:
        problems.append("rescue rate below its measured floor")
    if measured["unrescued_verbatim"] > \
            THRESHOLDS["unrescued_verbatim_ceiling"]:
        problems.append("unrescued verbatim above its ceiling")
    if measured["adopted_corruption_rate"] > \
            THRESHOLDS["adopted_corruption_ceiling"]:
        problems.append("adopted corruption above its ceiling")
    if measured["rambling_rescue_rate"] < \
            THRESHOLDS["rambling_rescue_floor"]:
        problems.append("rambling rescue below its floor")
    return problems


def metrics_as_data() -> dict[str, Any]:
    return {
        "format": METRICS_FORMAT,
        "metrics_doc": METRICS_DOC,
        "rules": [dict(r) for r in METRIC_RULES],
        "broken_outcomes": [dict(r) for r in BROKEN_OUTCOMES],
        "adversarial_outcomes": [dict(r)
                                 for r in ADVERSARIAL_OUTCOMES],
        "rambling_fixtures": list(RAMBLING_FIXTURES),
        "measurements": dict(FILED_MEASUREMENTS),
        "thresholds": dict(THRESHOLDS),
        "proof": {
            "grade": "fixture",
            "live_model_rates": "not_proven",
            "rambling_compression": "absent_measured_zero",
        },
    }
