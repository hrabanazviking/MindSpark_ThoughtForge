"""TF-071 — Tokeniser discipline.

Pins: the discipline format + live checker; the
agreement rule and its derivation (the estimate inside
tolerance of the filed BPE counts on every corpus row,
the pre-TF-071 flat formula provably outside it on the
CJK row); per-backend agreement (all four shipped
backends count identically on the corpus); the stub
model leg (reference tokeniser + BOS agrees with the
estimate on every row; a loaded engine IS the stub
count, the wrapper still guards "" to 0); D-TOK-1
(the unloaded engine delegates to the contract
estimate, "" included — no private // 4 copy survives);
D-TOK-2 (the estimate is script-aware, Latin counts
byte-identical to TF-065's); D-TOK-3 (both HF legs
report count_tokens — no word-split count survives);
and the word discipline's separation (the cognition
word count breaches the BPE tolerance on exactly the
filed rows and is a different function object). A
loaded GGUF tokeniser is NOT exercised — no llama_cpp
or GGUF exists on this VM — and nothing here claims it.
"""

from __future__ import annotations

from pathlib import Path

from thoughtforge.cognition.history_discipline import (
    estimate_tokens as word_estimate,
)
from thoughtforge.inference.backend_contract import (
    SCRIPT_TOKEN_RANGES,
    estimate_tokens,
    shipped_backends,
)
from thoughtforge.inference.tokenizer_discipline import (
    ABS_TOKENS,
    CORPUS,
    COUNTING_SURFACES,
    DISCIPLINE_FORMAT,
    MEASURED_BPE,
    REL_TOLERANCE,
    TOKENIZER_RULES,
    ReferenceStubLlamaModel,
    corpus_counts,
    counts_agree,
    discipline_as_data,
    reference_stub_tokens,
    relative_error,
    tokenizer_problems,
)
from thoughtforge.inference.turboquant import TurboQuantEngine
from thoughtforge.inference.turboquant_backend import TurboQuantBackend


# ── The format + checker ────────────────────────────────────────────────────

class TestDisciplineEstate:
    def test_format_and_checker(self):
        assert DISCIPLINE_FORMAT == "thoughtforge-tokenizer-discipline/1"
        assert tokenizer_problems() == []

    def test_as_data_shape(self):
        data = discipline_as_data()
        assert data["format"] == DISCIPLINE_FORMAT
        assert data["abs_tokens"] == ABS_TOKENS == 2
        assert data["rel_tolerance"] == REL_TOLERANCE == 0.35
        assert {r["rule"] for r in data["rules"]} == {
            "empty_is_zero", "one_estimate_one_home",
            "script_aware_estimate", "basis_is_stated",
            "agreement_within_tolerance",
            "special_tokens_are_a_basis",
            "word_counts_are_not_tokens",
            "reported_counts_match_their_backend",
        }
        assert len(data["surfaces"]) == len(COUNTING_SURFACES) == 9
        assert set(data["corpus_counts"]) == set(CORPUS)

    def test_rules_are_the_filed_rules(self):
        assert len(TOKENIZER_RULES) == 8
        for row in COUNTING_SURFACES:
            assert row["basis"] and row["special_tokens"]


# ── The agreement rule ──────────────────────────────────────────────────────

class TestAgreementRule:
    def test_absolute_floor_absorbs_special_tokens(self):
        # BOS (+1) and a boundary token: honest counters
        # 2 apart agree even when the counts are tiny.
        assert counts_agree(1, 3)
        assert counts_agree(0, 1)   # loaded engine on ""
        assert counts_agree(0, 2)

    def test_relative_band(self):
        assert counts_agree(100, 130)
        assert not counts_agree(100, 160)
        assert relative_error(0, 0) == 0.0
        assert relative_error(10, 20) == 0.5

    def test_estimate_agrees_with_filed_bpe_everywhere(self):
        rows = corpus_counts()
        for name, row in rows.items():
            assert row["bpe_filed"] == MEASURED_BPE[name]
            assert counts_agree(row["estimate"], row["bpe_filed"]), name
            assert relative_error(
                row["estimate"], row["bpe_filed"]) <= 0.35 or \
                abs(row["estimate"] - row["bpe_filed"]) <= ABS_TOKENS

    def test_estimate_agrees_with_stub_model_everywhere(self):
        for name, row in corpus_counts().items():
            assert counts_agree(row["estimate"], row["stub_model"]), name


# ── Per-backend agreement ───────────────────────────────────────────────────

class TestPerBackendAgreement:
    def test_all_shipped_backends_count_identically(self):
        backends = shipped_backends()
        assert len(backends) == 4
        for text in CORPUS.values():
            counts = {b.count_tokens(text) for b in backends}
            assert len(counts) == 1

    def test_backend_counts_agree_with_bpe_within_tolerance(self):
        backends = shipped_backends()
        for name, text in CORPUS.items():
            for backend in backends:
                assert counts_agree(
                    backend.count_tokens(text), MEASURED_BPE[name]), \
                    (backend.backend_name(), name)


# ── D-TOK-2 — the script-aware estimate ─────────────────────────────────────

class TestScriptAwareEstimate:
    def test_latin_counts_unchanged_from_tf065(self):
        for text in ("abcd", "a" * 40, "the ferry leaves at dawn",
                     CORPUS["prose"], CORPUS["code"], CORPUS["numbers"]):
            assert estimate_tokens(text) == max(1, len(text) // 4)

    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_cjk_counts_per_character(self):
        assert estimate_tokens(CORPUS["cjk"]) == len(CORPUS["cjk"]) == 11

    def test_pre_fix_flat_formula_breaches_on_cjk(self):
        flat = max(1, len(CORPUS["cjk"]) // 4)
        assert flat == 2
        assert not counts_agree(flat, MEASURED_BPE["cjk"])
        assert relative_error(flat, MEASURED_BPE["cjk"]) > 0.8

    def test_script_ranges_cover_the_filed_scripts(self):
        assert len(SCRIPT_TOKEN_RANGES) == 5
        for ch in ("灯", "は", "한"):
            assert estimate_tokens(ch) == 1


# ── D-TOK-1 — the engine delegates ──────────────────────────────────────────

class TestEngineDelegation:
    def test_unloaded_engine_is_the_contract_estimate(self):
        engine = TurboQuantEngine(model_path=None, profile_id="desktop_cpu")
        for text in CORPUS.values():
            assert engine.estimate_tokens(text) == estimate_tokens(text)

    def test_unloaded_engine_empty_is_zero(self):
        engine = TurboQuantEngine(model_path=None, profile_id="desktop_cpu")
        assert engine.estimate_tokens("") == 0  # was 1 before TF-071

    def test_stub_loaded_engine_is_the_stub_count_bos_included(self):
        engine = TurboQuantEngine(model_path=None, profile_id="desktop_cpu")
        engine._model = ReferenceStubLlamaModel()
        for name, text in CORPUS.items():
            expected = len(reference_stub_tokens(text)) + 1  # BOS
            assert engine.estimate_tokens(text) == expected, name
        # The BOS basis, stated by measurement: a loaded
        # engine counts "" as 1 — its own truth, not a
        # contract breach (the wrapper guards the surface).
        assert engine.estimate_tokens("") == 1

    def test_wrapper_guards_empty_with_loaded_engine(self):
        wrapper = TurboQuantBackend(
            model_path=Path("/nonexistent/tf071.gguf"))
        engine = TurboQuantEngine(model_path=None, profile_id="desktop_cpu")
        engine._model = ReferenceStubLlamaModel()
        wrapper._engine = engine
        assert wrapper.count_tokens("") == 0
        assert wrapper.count_tokens(CORPUS["prose"]) == \
            len(reference_stub_tokens(CORPUS["prose"])) + 1

    def test_no_private_heuristic_copy_survives(self):
        import inspect

        from thoughtforge.inference import turboquant

        assert "len(text) // 4" not in inspect.getsource(turboquant)


# ── D-TOK-3 — HF reports its own basis ──────────────────────────────────────

class TestHfReportedCounts:
    def test_both_legs_report_count_tokens(self):
        import inspect

        from thoughtforge.inference import hf_backend

        src = inspect.getsource(hf_backend)
        assert src.count("tokens_generated=self.count_tokens(text)") == 2
        assert "len(text.split())" not in src

    def test_client_leg_reports_the_estimate(self):
        from thoughtforge.inference.hf_backend import HuggingFaceBackend
        from thoughtforge.inference.unified_backend import GenerationRequest

        class FakeClient:
            def text_generation(self, prompt, **kwargs):
                return "The ferry leaves at dawn over the harbour."

        backend = HuggingFaceBackend(model="fake")
        backend._client = FakeClient()
        resp = backend.generate(GenerationRequest(prompt="hi"))
        assert resp.text.startswith("The ferry")
        assert resp.tokens_generated == backend.count_tokens(resp.text)
        assert resp.tokens_generated != len(resp.text.split())


# ── The word discipline stays separate ──────────────────────────────────────

class TestWordDisciplineSeparation:
    def test_word_breaches_are_exactly_the_filed_rows(self):
        breaches = {
            name for name, row in corpus_counts().items()
            if not counts_agree(row["word"], row["bpe_filed"])
        }
        assert breaches == {"numbers", "code", "punct", "cjk", "mixed"}

    def test_the_two_estimate_tokens_are_not_one_function(self):
        assert word_estimate is not estimate_tokens
        # Behaviourally distinct where it matters: the
        # long-prose budget count is a word count.
        assert word_estimate(CORPUS["long_prose"]) == 102
        assert estimate_tokens(CORPUS["long_prose"]) == 147
