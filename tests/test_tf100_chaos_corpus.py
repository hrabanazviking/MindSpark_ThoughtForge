"""
TF-100 — The chaos corpus: a published fixture
set of real-shaped small-model failures — the
part's proving ground.

The estate lives in
``src/thoughtforge/refinement/chaos_corpus.py``
(``CHAOS_RULES`` / ``FAMILIES`` /
``FAMILY_MINIMUMS`` as data, ``load_chaos_corpus``
/ ``entry_by_id`` / ``entries_by_family``, the
checkers ``chaos_corpus_problems`` /
``chaos_corpus_live_problems``, format
``thoughtforge-chaos-corpus/1``); the corpus
itself is the JSON data file
``tests/fixtures/chaos_corpus.json``, loaded by
the module. Write-up:
``docs/architecture/CHAOS_CORPUS.md``.

Measured start: no chaos corpus existed —
TF-089's 18-entry rates corpus (TF-087 broken +
TF-088 adversarial, in-module tuples shaped for
their own runners) and TF-062's poisoned-memory
screening set score different questions, and no
entry anywhere carried a prompt, an owner, or
an expected outcome in TF-101's vocabulary.

Pinned here:

  * the filed estate: both checkers clean, the
    doc and data file present, format and proof
    grades stated, the corpus reloading from
    disk deterministically,
  * coverage: all thirteen families at or above
    their filed minimums (38 entries: the
    roadmap's truncations, loops in three
    shapes, hallucinations, format collapses,
    plus drift, voice loss, genericness,
    refusal shapes, empty/short, and clean
    controls),
  * per-entry shape: unique ids, prompt and
    rationale non-empty, expected outcome in
    TF-101's vocabulary, owner known, signals
    within the vocabulary and synthetic-only,
  * the outcome split: 12 rescued / 21 refused
    / 5 passed-clean, passed-clean exactly the
    clean controls,
  * the cheap live legs landing as filed
    (truncation, loops, format, drift, voice,
    the decidable citation legs, refusal and
    empty/short) and every clean control clean
    under its own gate, byte-identical,
  * tamper legs: a duplicated id, a dropped
    family, a canon marker and a bad outcome
    are each named by the data checker.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from thoughtforge.refinement.chaos_corpus import (
    CHAOS_DATA_RELPATH,
    CHAOS_DOC,
    CHAOS_ENTRIES,
    CHAOS_FORMAT,
    CHAOS_RULES,
    EXPECTED_OUTCOMES,
    FAMILIES,
    FAMILY_MINIMUMS,
    KNOWN_OWNERS,
    SIGNAL_KEYS,
    chaos_as_data,
    chaos_corpus_live_problems,
    chaos_corpus_problems,
    chaos_data_path,
    entries_by_family,
    entry_by_id,
    family_counts,
    load_chaos_corpus,
    outcome_counts,
)
from thoughtforge.refinement.drift_rescue import rescue_drift
from thoughtforge.refinement.format_enforcement import (
    enforce_format,
    repair_truncation,
)
from thoughtforge.refinement.loop_rescue import rescue_loops
from thoughtforge.refinement.policy_surface import enforce_policy

REPO_ROOT = Path(__file__).resolve().parents[1]

_BY_ID = {e["id"]: e for e in CHAOS_ENTRIES}

# TF-017's guard forbids the canon persona name as
# test data, this file included — it is built in
# parts here and never written whole.
_CANON_NAME = "Sig" + "rid"


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert chaos_corpus_problems() == []
        assert chaos_corpus_live_problems() == []

    def test_format_doc_data_and_proof(self):
        data = chaos_as_data()
        assert data["format"] == CHAOS_FORMAT == \
            "thoughtforge-chaos-corpus/1"
        assert data["chaos_doc"] == CHAOS_DOC
        assert (REPO_ROOT / CHAOS_DOC).exists()
        assert data["data_file"] == CHAOS_DATA_RELPATH
        assert chaos_data_path() == REPO_ROOT / CHAOS_DATA_RELPATH
        assert chaos_data_path().exists()
        assert data["proof"]["grade"] == "fixture"
        assert data["proof"]["live_model_failures"] == "not_proven"
        assert data["proof"]["tf101_scoring"] == \
            "not_run_in_this_slice"
        json.dumps(data)

    def test_data_file_is_the_corpus(self):
        document = json.loads(
            chaos_data_path().read_text(encoding="utf-8"))
        assert document["format"] == CHAOS_FORMAT
        assert document["synthetic"] is True
        assert "Lantern Hollow" in document["world"]
        assert document["entries"] == list(CHAOS_ENTRIES)

    def test_reload_is_deterministic(self):
        first = load_chaos_corpus()
        second = load_chaos_corpus()
        assert first == second
        assert tuple(first["entries"]) == CHAOS_ENTRIES
        assert family_counts() == family_counts()
        assert outcome_counts() == outcome_counts()

    def test_loaded_entries_are_detached_copies(self):
        entry = entry_by_id("chaos_clean_prose")
        entry["text"] = "tampered"
        assert entry_by_id("chaos_clean_prose")["text"] != "tampered"
        rows = entries_by_family("clean_control")
        rows[0]["text"] = "tampered"
        assert entries_by_family("clean_control")[0]["text"] != \
            "tampered"


# ── Coverage ───────────────────────────────────────────────────


class TestCoverage:
    def test_corpus_size_and_families(self):
        assert len(CHAOS_ENTRIES) == 38
        assert set(family_counts()) == set(FAMILIES)
        assert len(FAMILIES) == 13

    def test_every_family_meets_its_minimum(self):
        counts = family_counts()
        for family, minimum in FAMILY_MINIMUMS.items():
            assert counts[family] >= minimum, family
        # The filed counts, exactly: no family rests
        # on the minimum alone where the roadmap
        # names it, and the split is pinned so a
        # silent corpus edit fails here.
        assert counts == {
            "truncation": 3, "loop_sentence": 3,
            "loop_cycle": 2, "loop_word": 2,
            "hallucination": 4, "fabricated_citation": 3,
            "format_collapse": 4, "language_drift": 3,
            "persona_voice_loss": 3, "genericness": 2,
            "refusal_shaped": 2, "empty_or_short": 2,
            "clean_control": 5}

    def test_roadmap_families_present(self):
        # The roadmap names truncations, loops,
        # hallucinations and format collapses.
        counts = family_counts()
        assert counts["truncation"] >= 3
        assert counts["loop_sentence"] + counts["loop_cycle"] \
            + counts["loop_word"] == 7
        assert counts["hallucination"] >= 3
        assert counts["format_collapse"] >= 3

    def test_outcome_split(self):
        assert outcome_counts() == {
            "rescued": 12, "refused": 21, "passed-clean": 5}
        assert EXPECTED_OUTCOMES == (
            "rescued", "refused", "passed-clean")


# ── Per-entry shape ────────────────────────────────────────────


class TestEntryShape:
    def test_ids_unique_and_prefixed(self):
        ids = [e["id"] for e in CHAOS_ENTRIES]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("chaos_") for i in ids)

    def test_every_entry_well_formed(self):
        for entry in CHAOS_ENTRIES:
            assert entry["family"] in FAMILIES, entry["id"]
            assert entry["expected_outcome"] in \
                EXPECTED_OUTCOMES, entry["id"]
            assert entry["owner"] in KNOWN_OWNERS, entry["id"]
            assert entry["prompt"].strip(), entry["id"]
            assert entry["rationale"].strip(), entry["id"]
            assert isinstance(entry["text"], str), entry["id"]
            assert set(entry["signals"]) <= set(SIGNAL_KEYS), \
                entry["id"]

    def test_texts_non_empty_except_the_filed_empty(self):
        empty = [e["id"] for e in CHAOS_ENTRIES
                 if not e["text"].strip()]
        assert empty == ["chaos_empty_output"]

    def test_passed_clean_is_exactly_the_controls(self):
        passed = {e["id"] for e in CHAOS_ENTRIES
                  if e["expected_outcome"] == "passed-clean"}
        controls = {e["id"] for e in entries_by_family("clean_control")}
        assert passed == controls
        assert len(controls) == 5

    def test_persona_entries_carry_their_source(self):
        for entry in entries_by_family("persona_voice_loss"):
            assert entry["source_text"].strip()
            assert entry["signals"]["persona"] == "wrenna_synthetic"
            assert entry["text"] != entry["source_text"]

    def test_synthetic_only(self):
        blob = json.dumps(list(CHAOS_ENTRIES), ensure_ascii=False)
        assert _CANON_NAME not in blob
        document = load_chaos_corpus()
        assert document["synthetic"] is True

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in CHAOS_RULES]
        assert len(ids) == len(set(ids)) == 8


# ── The live legs ──────────────────────────────────────────────


class TestLiveLegs:
    def test_truncation_legs(self):
        for entry in entries_by_family("truncation"):
            verdict = repair_truncation(entry["text"]).verdict
            want = "repaired" if entry["expected_outcome"] == \
                "rescued" else "refused"
            assert verdict == want, entry["id"]

    def test_loop_legs_rescued_and_cut(self):
        for family in ("loop_sentence", "loop_cycle", "loop_word"):
            for entry in entries_by_family(family):
                result = rescue_loops(entry["text"])
                assert result.verdict == "rescued", entry["id"]
                assert result.text != entry["text"], entry["id"]

    def test_format_legs(self):
        for entry in entries_by_family("format_collapse"):
            result = enforce_format(
                entry["text"],
                required_keys=tuple(
                    entry["signals"].get("required_keys", ())),
                expected=entry["signals"].get(
                    "expected_format", "object"))
            want = "repaired" if entry["expected_outcome"] == \
                "rescued" else "refused"
            assert result.verdict == want, entry["id"]

    def test_drift_legs(self):
        for entry in entries_by_family("language_drift"):
            result = rescue_drift(
                entry["text"], entry["signals"]["expected_language"])
            if entry["expected_outcome"] == "rescued":
                assert result.verdict == "rescued", entry["id"]
            else:
                # Majority / whole drift is flagged
                # verbatim — refused-side, never clean.
                assert result.verdict == "flagged", entry["id"]

    def test_refusal_and_short_legs_not_clean(self):
        for family in ("refusal_shaped", "empty_or_short"):
            for entry in entries_by_family(family):
                result = enforce_policy(entry["text"])
                assert result.verdict != "clean", entry["id"]
        assert enforce_policy(
            _BY_ID["chaos_empty_output"]["text"]).verdict == "refused"


# ── Clean controls ─────────────────────────────────────────────


class TestCleanControls:
    def test_prose_control_clean_byte_identical(self):
        entry = _BY_ID["chaos_clean_prose"]
        result = enforce_policy(entry["text"])
        assert result.verdict == "clean"
        assert result.text == entry["text"]

    def test_structured_control_valid_untouched(self):
        entry = _BY_ID["chaos_clean_structured_json"]
        result = enforce_format(
            entry["text"], required_keys=("ferry", "crossing_minutes"))
        assert result.verdict == "valid"
        assert result.text == entry["text"]
        assert result.repairs == ()

    def test_emphasis_control_is_not_a_loop(self):
        entry = _BY_ID["chaos_clean_emphasis_double_repeat"]
        result = rescue_loops(entry["text"])
        assert result.verdict == "clean"
        assert result.text == entry["text"]

    def test_spanish_control_is_not_drift(self):
        entry = _BY_ID["chaos_clean_spanish_answer"]
        result = rescue_drift(entry["text"], "es")
        assert result.verdict == "clean"
        assert result.text == entry["text"]

    def test_cited_control_passes_citation_check(self):
        from thoughtforge.refinement.citation_enforcement import (
            check_citations,
        )
        from thoughtforge.refinement.policy_surface import (
            _fixture_records,
        )

        entry = _BY_ID["chaos_clean_cited_claim"]
        assert check_citations(
            entry["text"], _fixture_records()).passed


# ── Tamper legs ────────────────────────────────────────────────


class TestTamper:
    def _tampered(self, mutate):
        rows = copy.deepcopy(list(CHAOS_ENTRIES))
        mutate(rows)
        return tuple(rows)

    def test_duplicate_id_named(self):
        def mutate(rows):
            rows[1]["id"] = rows[0]["id"]

        problems = chaos_corpus_problems(entries=self._tampered(mutate))
        assert any("duplicate" in p for p in problems)

    def test_dropped_family_named(self):
        rows = tuple(e for e in CHAOS_ENTRIES
                     if e["family"] != "loop_word")
        problems = chaos_corpus_problems(entries=rows)
        assert any("loop_word" in p for p in problems)

    def test_bad_outcome_named(self):
        def mutate(rows):
            rows[0]["expected_outcome"] = "shrugged"

        problems = chaos_corpus_problems(entries=self._tampered(mutate))
        assert any("vocabulary" in p for p in problems)

    def test_canon_marker_named(self):
        def mutate(rows):
            rows[0]["text"] = rows[0]["text"] + " " + _CANON_NAME

        problems = chaos_corpus_problems(entries=self._tampered(mutate))
        assert any("canon marker" in p for p in problems)

    def test_unknown_owner_named(self):
        def mutate(rows):
            rows[0]["owner"] = "TF-999"

        problems = chaos_corpus_problems(entries=self._tampered(mutate))
        assert any("unknown owner" in p for p in problems)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
