"""
TF-078 — Model cards in-repo: per supported model, the
measured truth (size, quant, profiles proven, scaffold
behaviour).

The cards live in
``src/thoughtforge/inference/model_cards.py``
(``CARD_RULES`` / ``RECOMMENDED_AUDIT`` /
``MODEL_CARD_DISCREPANCIES`` as data, ``build_cards()`` /
``scaffold_behaviour()`` / ``cards_as_data()``, format
``thoughtforge-model-cards/1``).
Write-up: ``docs/architecture/MODEL_CARDS.md``.

Measured start: no per-model card existed — the
catalogue (15 entries), the profiles' recommended
names (15 distinct) and the root MODEL_CARD.md table
(6 rows) named models on three unjoined surfaces.

Pinned here:

  * the filed estate: both checkers clean, one card
    per catalogue entry in order, rule ids unique,
  * quant truth: 13 proven + exactly TF-066's 2
    generic gaps; a proven card's curve row is
    TF-067's row + TF-068's class for its level; a
    generic card carries no curve row,
  * size/params honesty: sizes are the catalogue's
    declared figures; the Phi-3 pair's params stay
    None (unprovable, never guessed),
  * the fit matrix: the filed 28/53/9 summary, the
    named sharp rows (Qwen-72B exceeds server_gpu,
    Qwen-14B exceeds desktop_gpu, TinyLlama exceeds
    only pi_zero),
  * proof honesty: every card's profiles_proven is
    empty and live_proof is not_proven,
  * scaffold behaviour: model-blind builder signature
    + the profiles' generation shapes resolved live
    (pi_zero runs zero refinement passes),
  * the recommended audit: every distinct recommended
    name covered exactly once, 6/5/4 verdict counts,
    targets catalogued,
  * the root card reconciled: 6 rows parsed, the
    filed discrepancy set recomputed exactly.
"""

from __future__ import annotations

from thoughtforge.inference.model_browser import ModelBrowser
from thoughtforge.inference.model_cards import (
    CARD_RULES,
    CARDS_FORMAT,
    FIT_SUMMARY,
    MODEL_CARD_DISCREPANCIES,
    RECOMMENDED_AUDIT,
    build_cards,
    card_for_filename,
    cards_as_data,
    model_card_discrepancies,
    model_card_rows,
    model_cards_live_problems,
    model_cards_problems,
    scaffold_behaviour,
)
from thoughtforge.inference.quant_policy import LEVEL_CLASSES
from thoughtforge.inference.quant_quality import CURVE


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert model_cards_problems() == []
        assert model_cards_live_problems() == []

    def test_one_card_per_catalogue_entry_in_order(self):
        cards = build_cards()
        assert len(cards) == 15
        assert [c.filename for c in cards] == [
            e.filename for e in ModelBrowser.CURATED_MODELS]

    def test_rules_and_format(self):
        ids = [r["id"] for r in CARD_RULES]
        assert len(ids) == len(set(ids)) == 10
        assert cards_as_data()["format"] == CARDS_FORMAT


# ── Quant + curve truth ────────────────────────────────────────


class TestQuantTruth:
    def test_grades_split_13_proven_2_generic(self):
        cards = build_cards()
        proven = [c for c in cards if c.quant_grade == "proven"]
        generic = [c for c in cards if c.quant_grade == "generic"]
        assert len(proven) == 13 and len(generic) == 2
        assert {c.filename for c in generic} == {
            "Phi-3-mini-4k-instruct-q4.gguf",
            "Phi-3-mini-4k-instruct-q5.gguf"}

    def test_proven_curve_rows_are_the_levels_rows(self):
        curve = {row[0]: row for row in CURVE}
        for card in build_cards():
            if card.quant_grade != "proven":
                assert card.curve is None
                continue
            row = curve[card.filename_quant]
            assert card.curve["mean"] == row[1]
            assert card.curve["loss"] == row[2]
            assert card.curve["curve_class"] == LEVEL_CLASSES[row[0]]
            assert card.curve["basis"] == \
                "tf067_fixture_corpus_per_quant_level"

    def test_generic_never_rounded_up(self):
        phi = card_for_filename("Phi-3-mini-4k-instruct-q4.gguf")
        assert phi.declared_quant == "q4_k_m"
        assert phi.filename_quant == "q4"
        assert phi.quant_grade == "generic"


# ── Size / params honesty ──────────────────────────────────────


class TestSizeAndParams:
    def test_sizes_are_catalogue_declared(self):
        declared = {e.filename: e.size_gb
                    for e in ModelBrowser.CURATED_MODELS}
        for card in build_cards():
            assert card.size_gb == declared[card.filename]
            assert card.to_dict()["size_basis"] == "catalogue_declared"

    def test_phi_params_unprovable(self):
        for name in ("Phi-3-mini-4k-instruct-q4.gguf",
                     "Phi-3-mini-4k-instruct-q5.gguf"):
            assert card_for_filename(name).params_b is None

    def test_params_parsed_where_named(self):
        assert card_for_filename(
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf").params_b == 1.1
        assert card_for_filename(
            "Qwen2.5-72B-Instruct-Q4_K_M.gguf").params_b == 72.0


# ── The fit matrix ─────────────────────────────────────────────


class TestFitMatrix:
    def test_filed_summary(self):
        assert FIT_SUMMARY == {"fits": 28, "exceeds": 53,
                               "unprovable": 9}
        assert sum(FIT_SUMMARY.values()) == 90

    def test_sharp_rows(self):
        for name in ("Qwen2.5-72B-Instruct-Q4_K_M.gguf",
                     "Qwen2.5-72B-Instruct-IQ2_XS.gguf"):
            assert card_for_filename(name).fits["server_gpu"] == "exceeds"
        qwen14 = card_for_filename("Qwen2.5-14B-Instruct-Q4_K_M.gguf")
        assert qwen14.fits["desktop_gpu"] == "exceeds"
        assert qwen14.fits["server_gpu"] == "fits"
        tiny = card_for_filename("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        assert tiny.fits_profiles == (
            "desktop_cpu", "desktop_gpu", "phone_low", "pi_5",
            "server_gpu")

    def test_phi_cells_unprovable_where_size_passes(self):
        phi = card_for_filename("Phi-3-mini-4k-instruct-q4.gguf")
        assert phi.fits["pi_5"] == "unprovable"
        assert phi.fits["pi_zero"] == "exceeds"


# ── Proof honesty ──────────────────────────────────────────────


class TestProofHonesty:
    def test_no_card_claims_live_proof(self):
        for card in build_cards():
            assert card.profiles_proven == ()
            assert card.live_proof == "not_proven"
            assert card.to_dict()["profiles_proven"] == []


# ── Scaffold behaviour ─────────────────────────────────────────


class TestScaffoldBehaviour:
    def test_shapes_resolved_live(self):
        behaviour = scaffold_behaviour()
        assert behaviour["phone_low"] == {
            "draft_count": 2, "max_refinement_passes": 1,
            "max_response_tokens": 180, "token_budget_per_turn": 180,
            "max_context_tokens": 512}
        assert behaviour["pi_zero"]["max_refinement_passes"] == 0
        assert behaviour["server_gpu"]["draft_count"] == 5

    def test_builder_is_model_blind(self):
        import inspect
        from thoughtforge.cognition.scaffold import ScaffoldBuilder
        assert list(inspect.signature(
            ScaffoldBuilder.build).parameters) == [
                "self", "sketch", "bundle", "personality"]


# ── The recommended audit ──────────────────────────────────────


class TestRecommendedAudit:
    def test_verdict_counts(self):
        verdicts = [r["verdict"] for r in RECOMMENDED_AUDIT]
        assert verdicts.count("carded") == 6
        assert verdicts.count("family_only") == 5
        assert verdicts.count("no_card") == 4

    def test_no_card_names(self):
        assert {r["name"] for r in RECOMMENDED_AUDIT
                if r["verdict"] == "no_card"} == {
                    "Granite-1B-Instruct", "Gemma-2B-it-q4_k_m",
                    "LLaMA-3-13B-Instruct-q4_k_m",
                    "Mixtral-8x7B-Instruct-v0.1-fp16"}

    def test_audit_profiles_filed_in_export(self):
        data = cards_as_data()
        by_name = {r["name"]: r for r in data["recommended_audit"]}
        assert by_name["TinyLlama-1.1B-Chat"]["profiles"] == ["phone_low"]
        assert by_name["Mixtral-8x7B-Instruct-v0.1-fp16"]["profiles"] == [
            "server_gpu"]


# ── The root card, reconciled ──────────────────────────────────


class TestRootCard:
    def test_six_rows_parsed(self):
        rows = model_card_rows()
        assert [r["profile"] for r in rows] == [
            "phone_low", "pi_zero", "pi_5", "desktop_cpu",
            "desktop_gpu", "server_gpu"]

    def test_discrepancies_recomputed(self):
        assert model_card_discrepancies() == set(MODEL_CARD_DISCREPANCIES)
        assert MODEL_CARD_DISCREPANCIES == frozenset({
            ("phone_low", "quant_not_in_catalogue"),
            ("pi_zero", "model_not_in_catalogue"),
            ("pi_zero", "quant_not_in_catalogue"),
            ("pi_5", "model_variant_not_in_catalogue"),
            ("server_gpu", "model_not_in_catalogue"),
            ("server_gpu", "quant_differs_from_profile_default"),
        })

    def test_clean_rows_stay_clean(self):
        assert not any(p in ("desktop_cpu", "desktop_gpu")
                       for p, _ in MODEL_CARD_DISCREPANCIES)
