"""
TF-088 — Salvage safety: rescue never invents
content — the line between repair and fabrication
specified + tested with adversarial broken
outputs.

The estate lives in
``src/thoughtforge/refinement/salvage_safety.py``
(``SAFETY_RULES`` / ``PROBLEM_KINDS`` /
``ADVERSARIAL_FIXTURES`` as data, the guard
``fabrication_problems`` / ``classify_output``,
the checkers ``salvage_safety_problems`` /
``salvage_safety_live_problems``, format
``thoughtforge-salvage-safety/1``), wired into
``refinement/salvage.py``'s refine adoption rule
(D-SALV-2 FIXED). Write-up:
``docs/architecture/SALVAGE_SAFETY.md``.

Measured start: the adoption rule was score-only,
and five adversarial stub-engine outputs probed
live were ALL adopted as refine_pass_1 — an
invented price, an invented citation, a negation
flip, invented destinations, a displaced time.

Pinned here:

  * the filed estate: both checkers clean, ids
    unique, kinds known, the guard's wiring anchor
    present in salvage.py, proof grade stated,
  * the guard itself: each marker family judged
    directly (numbers, number words, citations,
    named entities, polarity both ways, time
    displacement vs mere addition), verbatim
    pass-through classified as the source's own,
  * the adversarial fixtures run through the real
    FragmentSalvage: fabrications rejected with
    the candidate text standing and the attempt
    still counted, clean legs adopted,
  * the residual filed, not hidden: common-word
    elaboration (TF-087's own adopted fixture)
    carries no marker and is not judged —
    TF-089/TF-104 own it.
"""

from __future__ import annotations

from thoughtforge.refinement.salvage_safety import (
    ADVERSARIAL_FIXTURES,
    PROBLEM_KINDS,
    SAFETY_DOC,
    SAFETY_FORMAT,
    SAFETY_RULES,
    classify_output,
    fabrication_problems,
    run_adversarial,
    safety_as_data,
    salvage_safety_live_problems,
    salvage_safety_problems,
)

_BY_ID = {f["id"]: f for f in ADVERSARIAL_FIXTURES}
_BASE = _BY_ID["invented_price"]["candidate"]


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert salvage_safety_problems() == []
        assert salvage_safety_live_problems() == []

    def test_format_doc_and_proof(self):
        data = safety_as_data()
        assert data["format"] == SAFETY_FORMAT
        assert data["safety_doc"] == SAFETY_DOC
        assert data["proof"]["live_model_fabrications"] == \
            "not_proven"
        assert data["proof"]["common_word_elaboration"] == \
            "residual_not_judged"

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in SAFETY_RULES]
        assert len(ids) == len(set(ids)) == 10

    def test_problem_kinds(self):
        assert PROBLEM_KINDS == (
            "new_number", "new_number_word", "new_citation",
            "new_named_entity", "negation_introduced",
            "negation_dropped", "time_marker_displaced")


# ── The guard, judged directly ─────────────────────────────────


class TestGuard:
    def test_verbatim_output_is_clean(self):
        assert fabrication_problems([_BASE], _BASE, ("Q123",)) == []
        verdict = classify_output([_BASE], _BASE, ("Q123",))
        assert verdict["verdict"] == "verbatim_pass_through"

    def test_new_number_and_number_word(self):
        out = _BASE + " It costs 50 coins, fifty in silver."
        kinds = {p.split(":")[0] for p in
                 fabrication_problems([_BASE], out, ("Q123",))}
        assert kinds == {"new_number", "new_number_word"}

    def test_source_number_may_be_reused(self):
        source = _BASE + " The crossing takes 2 hours at dawn."
        out = _BASE + " The crossing takes 2 hours."
        assert fabrication_problems([source], out, ("Q123",)) == []

    def test_new_citation_flagged_retrieved_one_fine(self):
        out = _BASE.replace("Q123", "Q123 and Q999")
        problems = fabrication_problems([_BASE], out, ("Q123",))
        assert problems == ["new_citation:Q999"]

    def test_new_named_entity_mid_sentence(self):
        out = _BASE + " It sails to Bergen tonight."
        problems = fabrication_problems([_BASE], out, ("Q123",))
        assert problems == ["new_named_entity:Bergen"]

    def test_time_addition_alone_is_not_displacement(self):
        out = _BASE + " The harbour is quiet at night."
        assert fabrication_problems([_BASE], out, ("Q123",)) == []

    def test_time_displacement_flagged(self):
        out = _BASE.replace("at dawn", "at midnight")
        problems = fabrication_problems([_BASE], out, ("Q123",))
        assert problems == ["time_marker_displaced:dawn->midnight"]

    def test_negation_introduced_and_dropped(self):
        flipped = _BASE.replace("leaves at dawn",
                                 "does not leave at dawn")
        assert fabrication_problems([_BASE], flipped, ("Q123",)) == \
            ["negation_introduced"]
        negated = _BY_ID["negation_dropped"]["candidate"]
        cleaned = _BY_ID["negation_dropped"]["engine_text"]
        kinds = {p.split(":")[0] for p in
                 fabrication_problems([negated], cleaned, ("Q123",))}
        assert kinds == {"negation_dropped"}

    def test_bundle_cue_content_is_in_the_pool(self):
        # Content grounded in a retrieved record's cue is
        # not salvage invention.
        out = _BASE + " The keeper tends the Bergen light."
        assert fabrication_problems(
            [_BASE], out, ("Q123",)) == \
            ["new_named_entity:Bergen"]
        assert fabrication_problems(
            [_BASE, "the Bergen light keeper cue"],
            out, ("Q123",)) == []


# ── The adversarial fixtures, run live ─────────────────────────


class TestAdversarialFixtures:
    def test_all_five_probed_fabrications_now_rejected(self):
        for fixture_id in (
                "invented_price", "invented_citation",
                "negation_flip", "invented_destinations",
                "time_displaced"):
            fixture = _BY_ID[fixture_id]
            result = run_adversarial(fixture_id)
            assert result.text == fixture["candidate"]
            assert result.text != fixture["engine_text"]
            # The attempt still counts (TF-087 naming):
            assert result.salvage_path == "refine_pass_1"
            assert result.passes_used == 1

    def test_negation_dropped_rejected(self):
        fixture = _BY_ID["negation_dropped"]
        result = run_adversarial("negation_dropped")
        assert result.text == fixture["candidate"]

    def test_benign_rephrase_still_adopted(self):
        fixture = _BY_ID["benign_rephrase"]
        result = run_adversarial("benign_rephrase")
        assert result.text == fixture["engine_text"]
        assert result.salvage_path == "refine_pass_1"

    def test_residual_elaboration_adopted_and_unjudged(self):
        # TF-087's adopted fixture: common-word elaboration
        # with no new marker — the guard does not judge it
        # and this slice does not claim it safe.
        fixture = _BY_ID["common_word_elaboration_residual"]
        result = run_adversarial(
            "common_word_elaboration_residual")
        assert result.text == fixture["engine_text"]
        verdict = classify_output(
            [fixture["candidate"]], fixture["engine_text"],
            ("Q123",))
        assert verdict["verdict"] == "repair_within_markers"

    def test_source_invention_passes_through_unjudged(self):
        fixture = _BY_ID["source_invention_pass_through"]
        result = run_adversarial("source_invention_pass_through")
        assert result.text == fixture["candidate"]
        assert result.salvage_path == "best_draft"
        verdict = classify_output(
            [fixture["candidate"]], result.text, ("Q123",))
        assert verdict["verdict"] == "verbatim_pass_through"
