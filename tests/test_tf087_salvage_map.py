"""
TF-087 — Salvage mapped: salvage.py's strategies
catalogued, each with fixtures of broken input.

The estate lives in
``src/thoughtforge/refinement/salvage_map.py``
(``STRATEGIES`` / ``ABSENT_STRATEGIES`` /
``BROKEN_FIXTURES`` / ``SALVAGE_RULES`` as data,
the checkers ``salvage_map_problems`` /
``salvage_map_live_problems``, format
``thoughtforge-salvage-map/1``).
Write-up: ``docs/architecture/SALVAGE_MAPPED.md``.

Measured start: salvage.py does five things —
draft scoring/selection, fragment extraction,
engine refine passes, best-draft fallback, empty
fallback — and none of the roadmap's named
rescues (truncation repair, rambling compression,
format rescue, loop cutting, language drift).
Probed live: truncated, format-broken, looping
and lone rambling drafts are all returned
byte-identical.

Pinned here:

  * the filed estate: both checkers clean, ids
    unique, constants judged against salvage.py's
    live values, proof grade stated,
  * the catalogue: five present strategies with
    source anchors, five absent strategies with
    Part V owners (D-SALV-1),
  * the fixtures: every no-engine fixture runs
    through the real FragmentSalvage exactly as
    filed (path, verbatim text, fragments kept),
  * the refine legs: a stub engine + the real
    PromptBuilder — a better output is adopted as
    refine_pass_1, an engine failure falls back
    to best_draft, and a non-improving pass still
    names refine_pass_1 over the original text
    (paths name attempts, TF-097 owns the record).
"""

from __future__ import annotations

from types import SimpleNamespace

from thoughtforge.cognition.prompt_builder import PromptBuilder
from thoughtforge.knowledge.models import CognitionScaffold, InputSketch
from thoughtforge.refinement import salvage as salvage_module
from thoughtforge.refinement.salvage_map import (
    ABSENT_STRATEGIES,
    BROKEN_FIXTURES,
    SALVAGE_CONSTANTS,
    SALVAGE_DOC,
    SALVAGE_FORMAT,
    SALVAGE_RULES,
    STRATEGIES,
    run_fixture,
    salvage_as_data,
    salvage_map_live_problems,
    salvage_map_problems,
)

_BY_ID = {f["id"]: f for f in BROKEN_FIXTURES}


def _sketch() -> InputSketch:
    return InputSketch(
        raw_text="when does the ferry leave?",
        intent="factual_query", topic="ferry",
        response_mode="factual_direct")


def _scaffold() -> CognitionScaffold:
    return CognitionScaffold(
        goal="give a clear direct answer",
        tone=["direct", "factual"], avoid=["filler"])


class _Engine:
    def __init__(self, text):
        self._text = text

    def generate(self, prompt, params=None):
        return SimpleNamespace(text=self._text)


class _BadEngine:
    def generate(self, prompt, params=None):
        raise RuntimeError("engine down")


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert salvage_map_problems() == []
        assert salvage_map_live_problems() == []

    def test_format_doc_and_proof(self):
        data = salvage_as_data()
        assert data["format"] == SALVAGE_FORMAT
        assert data["salvage_doc"] == SALVAGE_DOC
        assert data["proof"]["live_model_broken_outputs"] == "not_proven"

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in SALVAGE_RULES]
        assert len(ids) == len(set(ids)) == 7

    def test_constants_match_live_module(self):
        assert SALVAGE_CONSTANTS["length_weight"] == \
            salvage_module._LENGTH_WEIGHT == 0.45
        assert SALVAGE_CONSTANTS["citation_weight"] == \
            salvage_module._CITATION_WEIGHT == 0.55
        assert SALVAGE_CONSTANTS["min_fragment_score"] == \
            salvage_module._MIN_FRAGMENT_SCORE == 0.54
        assert SALVAGE_CONSTANTS["max_refine_passes"] == \
            salvage_module._MAX_REFINE_PASSES == 2


# ── The catalogue ──────────────────────────────────────────────


class TestCatalogue:
    def test_five_present_strategies(self):
        assert [s["id"] for s in STRATEGIES] == [
            "draft_scoring_selection", "fragment_extraction",
            "refine_passes", "best_draft_fallback",
            "empty_candidates_fallback"]

    def test_five_absent_strategies_owned_in_part_v(self):
        assert [a["id"] for a in ABSENT_STRATEGIES] == [
            "truncation_repair", "rambling_compression",
            "format_rescue", "repetition_loop_cut",
            "language_drift_rescue"]
        owners = {a["id"]: a["owner"] for a in ABSENT_STRATEGIES}
        assert owners == {
            "truncation_repair": "TF-090",
            "rambling_compression": "TF-089",
            "format_rescue": "TF-090",
            "repetition_loop_cut": "TF-091",
            "language_drift_rescue": "TF-092"}

    def test_no_strategy_is_both_present_and_absent(self):
        present = {s["id"] for s in STRATEGIES}
        absent = {a["id"] for a in ABSENT_STRATEGIES}
        assert not present & absent


# ── The fixtures, run live ─────────────────────────────────────


class TestNoEngineFixtures:
    def test_empty_candidates(self):
        result = run_fixture("empty_candidates")
        assert result.salvage_path == "empty"
        assert result.confidence == 0.0
        assert result.text == "The forge finds no candidates to salvage."

    def test_truncated_draft_returned_verbatim_no_repair(self):
        fixture = _BY_ID["truncated_mid_sentence"]
        result = run_fixture(fixture["id"])
        assert result.text == fixture["candidates"][0]
        assert result.text.endswith("across the")

    def test_cited_concise_beats_rambling_by_selection(self):
        result = run_fixture("rambling_beaten_by_cited_rival")
        assert result.text == _BY_ID[
            "rambling_beaten_by_cited_rival"]["expected_text"]
        assert "Q123" in result.citations

    def test_rambling_alone_returned_verbatim_no_compression(self):
        fixture = _BY_ID["rambling_alone_verbatim"]
        result = run_fixture(fixture["id"])
        assert result.text == fixture["candidates"][0]

    def test_fragment_filter_keeps_one_and_output_stays_whole(self):
        result = run_fixture("fragment_filter_keeps_only_scored_sentences")
        assert result.fragments_kept == 1
        # Fragments never clean the no-engine output:
        assert result.text.startswith("Ok. Yes indeed.")

    def test_broken_json_returned_verbatim_no_rescue(self):
        fixture = _BY_ID["broken_json_verbatim"]
        result = run_fixture(fixture["id"])
        assert result.text == fixture["candidates"][0]

    def test_loop_returned_verbatim_no_cut(self):
        fixture = _BY_ID["loop_verbatim"]
        result = run_fixture(fixture["id"])
        assert result.text == fixture["candidates"][0]
        assert result.text.count("The bell rings.") == 20


# ── The refine legs (stub engine, real PromptBuilder) ──────────


class TestRefineFixtures:
    def test_better_output_adopted_as_refine_pass_1(self):
        fixture = _BY_ID["refine_adopts_better_output"]
        result = run_fixture(
            fixture["id"], engine=_Engine(fixture["engine_text"]),
            prompt_builder=PromptBuilder(), sketch=_sketch(),
            scaffold=_scaffold(), max_passes=1)
        assert result.salvage_path == "refine_pass_1"
        assert result.passes_used == 1
        assert result.text == fixture["expected_text"]
        assert result.confidence > result.best_draft_score

    def test_engine_failure_falls_back_to_best_draft(self):
        result = run_fixture(
            "refine_engine_failure_falls_back", engine=_BadEngine(),
            prompt_builder=PromptBuilder(), sketch=_sketch(),
            scaffold=_scaffold(), max_passes=1)
        assert result.salvage_path == "best_draft"
        assert result.passes_used == 0

    def test_non_improving_pass_still_names_refine_pass_1(self):
        # The path names passes attempted, not a rescue
        # adopted: the original text stands, the path
        # still says refine_pass_1 (TF-097 owns this).
        fixture = _BY_ID["refine_adopts_better_output"]
        result = run_fixture(
            fixture["id"], engine=_Engine("ok"),
            prompt_builder=PromptBuilder(), sketch=_sketch(),
            scaffold=_scaffold(), max_passes=1)
        assert result.text == fixture["candidates"][0]
        assert result.salvage_path == "refine_pass_1"
        assert result.passes_used == 1
