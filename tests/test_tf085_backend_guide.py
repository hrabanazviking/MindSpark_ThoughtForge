"""
TF-085 — Backend docs + the new-backend guide (the
contract as a recipe).

The estate lives in
``src/thoughtforge/inference/backend_guide.py``
(``BACKEND_DOCS`` / ``RECIPE_STEPS`` / ``GUIDE_RULES``
as data, ``GuideExampleBackend``, the checkers
``backend_guide_problems`` /
``backend_guide_live_problems``, format
``thoughtforge-backend-guide/1``).
Write-up: ``docs/architecture/BACKEND_GUIDE.md``.

Measured start: six core + three supporting backend
docs stood unindexed, no new-backend recipe existed
anywhere (only TF-153's admission ownership was
named), and docs/index.md + docs/api.md contain no
mention of "backend" at all (D-GUIDE-1, fixed in
scope at guide grade).

Pinned here:

  * the filed estate: both checkers clean, rule ids
    unique, the export's proof grades stated,
  * the catalogue: nine docs, six core + three
    supporting, each judged live (exists, names its
    slice, module exists),
  * the recipe: fourteen steps numbered 1..14,
    covering every CONTRACT_DUTIES duty plus
    dependencies / conformance / admission / docs,
    each step's source file present,
  * the worked example: conformed by TF-065's own
    checker, plain turn clean, empty prompt as named
    error data plain AND streamed (the TF-084 stash
    carries the exact error), simulated stream ==
    the completed text as one chunk, health True,
  * admission honesty: the example is in neither
    SHIPPED_BACKENDS nor the factory, and D-GUIDE-1's
    remainder is stated (live-author proof
    not_proven).
"""

from __future__ import annotations

from pathlib import Path

from thoughtforge.inference.backend_contract import (
    CONTRACT_DUTIES,
    SHIPPED_BACKENDS,
)
from thoughtforge.inference.backend_guide import (
    BACKEND_DOCS,
    GUIDE_DOC,
    GUIDE_FORMAT,
    GUIDE_RULES,
    RECIPE_STEPS,
    GuideExampleBackend,
    backend_guide_live_problems,
    backend_guide_problems,
    guide_as_data,
)
from thoughtforge.inference.unified_backend import GenerationRequest

ROOT = Path(__file__).resolve().parents[1]


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert backend_guide_problems() == []
        assert backend_guide_live_problems() == []

    def test_rules_format_and_proof(self):
        ids = [r["id"] for r in GUIDE_RULES]
        assert len(ids) == len(set(ids)) == 7
        data = guide_as_data()
        assert data["format"] == GUIDE_FORMAT
        assert data["guide_doc"] == GUIDE_DOC
        assert data["proof"]["live_third_party_backend_from_guide"] \
            == "not_proven"


# ── The catalogue ──────────────────────────────────────────────


class TestCatalogue:
    def test_nine_docs_six_core_three_supporting(self):
        assert len(BACKEND_DOCS) == 9
        roles = [d["role"] for d in BACKEND_DOCS]
        assert roles.count("core") == 6
        assert roles.count("supporting") == 3

    def test_each_row_judged_live(self):
        for row in BACKEND_DOCS:
            doc = ROOT / row["path"]
            assert doc.exists(), row["path"]
            assert row["slice"] in doc.read_text(
                encoding="utf-8")[:400]
            assert (ROOT / row["module"]).exists()

    def test_slices_are_the_backend_parts_owners(self):
        by_slice = {d["slice"]: d for d in BACKEND_DOCS}
        assert by_slice["TF-065"]["path"].endswith(
            "BACKEND_CONTRACT.md")
        assert by_slice["TF-077"]["path"].endswith(
            "LOCAL_FIRST_ROUTING.md")
        assert by_slice["TF-084"]["role"] == "supporting"


# ── The recipe ─────────────────────────────────────────────────


class TestRecipe:
    def test_fourteen_steps_numbered_in_order(self):
        assert len(RECIPE_STEPS) == 14
        assert [s["step"] for s in RECIPE_STEPS] == list(range(1, 15))
        ids = [s["id"] for s in RECIPE_STEPS]
        assert len(ids) == len(set(ids))

    def test_covers_every_contract_duty(self):
        covered = {d for s in RECIPE_STEPS for d in s["duties"]}
        for duty in CONTRACT_DUTIES:
            assert duty["duty"] in covered

    def test_covers_the_assumed_duties(self):
        covered = {d for s in RECIPE_STEPS for d in s["duties"]}
        assert {"dependencies", "conformance",
                "admission", "docs"} <= covered

    def test_admission_step_names_tf153(self):
        step = next(s for s in RECIPE_STEPS
                    if s["id"] == "admission_is_tf153s")
        assert "TF-153" in step["instruction"]
        assert "SHIPPED_BACKENDS" in step["instruction"]

    def test_each_step_source_exists(self):
        for s in RECIPE_STEPS:
            assert (ROOT / s["source"]).exists(), s["id"]

    def test_guide_doc_lists_every_doc_and_step(self):
        text = (ROOT / GUIDE_DOC).read_text(encoding="utf-8")
        for row in BACKEND_DOCS:
            assert row["path"] in text
        for s in RECIPE_STEPS:
            assert s["id"] in text


# ── The worked example ─────────────────────────────────────────


class TestExampleBackend:
    def test_conformed_by_tf065s_own_checker(self):
        from thoughtforge.inference.backend_contract import (
            backend_contract_problems,
        )
        assert backend_contract_problems(
            [GuideExampleBackend()]) == []

    def test_plain_turn(self):
        b = GuideExampleBackend()
        resp = b.generate(GenerationRequest(
            prompt="where is the ferry?"))
        assert resp.finish_reason == "stop"
        assert "where is the ferry?" in resp.text
        assert resp.backend_used == "guide_example:lantern-1"
        assert resp.tokens_generated == b.count_tokens(resp.text)

    def test_empty_prompt_is_named_error_data(self):
        b = GuideExampleBackend()
        resp = b.generate(GenerationRequest(prompt=""))
        assert resp.finish_reason == "error"
        assert resp.error == "guide example: empty prompt"
        assert resp.text == ""

    def test_stream_simulated_and_error_stash(self):
        b = GuideExampleBackend()
        plain = b.generate(GenerationRequest(prompt="hello"))
        chunks = list(b.generate_stream(
            GenerationRequest(prompt="hello")))
        assert chunks == [plain.text]
        assert b.capabilities().streaming == "simulated"
        assert list(b.generate_stream(
            GenerationRequest(prompt=""))) == []
        assert b.last_stream_response.error == \
            "guide example: empty prompt"

    def test_capabilities_honest(self):
        caps = GuideExampleBackend().capabilities().to_dict()
        assert caps["placement"] == "local"
        assert caps["token_counting"] == "estimate"
        assert caps["seed"] == "not_supported"
        assert caps["name"] == "guide_example:lantern-1"

    def test_health_and_counting(self):
        b = GuideExampleBackend()
        assert b.health_check() is True
        assert b.count_tokens("") == 0
        assert 0 < b.count_tokens("the ferry") <= \
            b.count_tokens("the ferry " * 8)

    def test_example_never_shipped(self):
        assert all(r["class"] != "GuideExampleBackend"
                   for r in SHIPPED_BACKENDS)
        from thoughtforge.inference import unified_backend
        import inspect
        assert "GuideExampleBackend" not in inspect.getsource(
            unified_backend.load_backend_from_config)

    def test_example_is_deterministic(self):
        b = GuideExampleBackend()
        req = GenerationRequest(prompt="same prompt")
        assert b.generate(req).text == b.generate(req).text
