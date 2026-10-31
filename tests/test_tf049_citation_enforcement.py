"""TF-049 — cite-or-explain II (enforcement side): a
post-generation per-claim check — claims matching stored
facts must cite that record's TF-048 marker inside the
claim itself, or the answer is flagged per policy.

Pins refinement/citation_enforcement.py, the records leg
of EnforcementGate.check, and the core wiring (compose
passes the bundle records, the check retained on
core.last_citation_check, the repair gate delegating to
the one checker, _extract_citations speaking every kind)
against the live tree. The legacy any-QID floor is pinned
unchanged, in both of its homes.

Proof grade: FIXTURE (synthetic records + the TF-043
corpus with one real-shaped-QID entity + stub-backend
core turns, VM). A live model's citation behaviour is
NOT PROVEN.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from thoughtforge.cognition.core import (
    ThoughtForgeCore,
    _check_citation_gate,
    _extract_citations,
)
from thoughtforge.etl.schema import get_connection, initialize_schema
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.knowledge.models import (
    ActivatedRecord,
    MemoryActivationBundle,
)
from thoughtforge.knowledge.retrieval_precision import build_fixture_db
from thoughtforge.refinement.citation_enforcement import (
    CITATION_POLICY,
    MATCH_RULES,
    MIN_SHARED_TERMS,
    CitationCheck,
    check_citations,
    citation_enforcement_problems,
    cited_markers_in,
    distinctive_terms,
    match_claim_to_record,
)
from thoughtforge.refinement.enforcement import EnforcementGate

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "architecture" / "CITATION_ENFORCEMENT.md"


def _entity() -> ActivatedRecord:
    return ActivatedRecord(
        "Q123", "entity", 0.9, "Odin chief god of Norse mythology",
        raw={"qid": "Q123", "label_en": "Odin",
             "description_en": "chief god of Norse mythology",
             "source": "wikidata"},
    )


def _chunk() -> ActivatedRecord:
    return ActivatedRecord(
        "chunk_7", "reference_chunk", 0.8,
        "lantern oil kept in the hollow stores",
        raw={"chunk_id": "chunk_7", "source_file": "hollow.md",
             "title": "The Hollow",
             "content": "lantern oil is kept in the hollow stores",
             "source": "reference"},
    )


def _place() -> ActivatedRecord:
    return ActivatedRecord(
        "Hollow Creek", "place", 0.7, "Hollow Creek village by the river",
        raw={"geonames_id": 5128581, "label_en": "Hollow Creek",
             "description_en": "village by the river", "source": "geonames"},
    )


def _edge() -> ActivatedRecord:
    return ActivatedRecord(
        "ferry", "concept_edge", 0.6, "ferry UsedFor crossing rivers",
        raw={"concept1": "ferry", "relation": "UsedFor",
             "concept2": "crossing rivers", "source": "conceptnet",
             "content": "ferry UsedFor crossing rivers"},
    )


RECORDS = [_entity(), _chunk()]


# ── Self-check, policy, rules ────────────────────────────────────────────────


class TestSelfCheck:
    def test_problems_empty(self) -> None:
        assert citation_enforcement_problems() == []

    def test_doc_exists(self) -> None:
        assert DOC.exists()

    def test_policy_table(self) -> None:
        assert CITATION_POLICY["on_uncited_claim"] == "flag"
        assert CITATION_POLICY["on_foreign_marker"] == "flag"
        assert "regenerate" in CITATION_POLICY["actions"]
        # Regeneration is named, not wired — the policy says so.
        assert "not wired" in CITATION_POLICY["regeneration"]

    def test_match_rules_table(self) -> None:
        assert [r["name"] for r in MATCH_RULES] == [
            "label_phrase", "label_plus_term", "term_overlap",
        ]
        assert MIN_SHARED_TERMS == 3


class TestTerms:
    def test_distinctive_terms_fold_and_stopwords(self) -> None:
        terms = distinctive_terms("The stores are kept in the Hollow")
        assert "store" in terms  # singular fold
        assert "hollow" in terms
        assert "the" not in terms and "are" not in terms

    def test_cited_markers_brackets_and_bare_qids(self) -> None:
        markers = cited_markers_in("See [ref:chunk_7] and Q123 for details.")
        assert "ref:chunk_7" in markers and "Q123" in markers


# ── Matching rules ───────────────────────────────────────────────────────────


class TestMatching:
    def test_label_phrase(self) -> None:
        ferry = ActivatedRecord(
            "QF1", "entity", 0.9, "ferry",
            raw={"qid": "QF1", "label_en": "Lantern Hollow Ferry",
                 "description_en": "a boat", "source": "wikidata"},
        )
        match = match_claim_to_record(
            "The Lantern Hollow Ferry departs at dawn.", ferry,
        )
        assert match is not None and match["rule"] == "label_phrase"

    def test_label_plus_term(self) -> None:
        match = match_claim_to_record("Odin is a god of wisdom.", _entity())
        assert match is not None and match["rule"] == "label_plus_term"
        assert "odin" in match["shared_terms"]

    def test_term_overlap_without_label(self) -> None:
        # No label term ("hollow") in the claim at all — the
        # match rides on shared distinctive terms alone.
        match = match_claim_to_record(
            "Lantern oil is kept in the stores.", _chunk(),
        )
        assert match is not None
        assert match["rule"] == "term_overlap"
        assert len(match["shared_terms"]) >= MIN_SHARED_TERMS

    def test_no_match_for_unrelated_claim(self) -> None:
        assert match_claim_to_record(
            "The weather over the fjord is calm today.", _entity(),
        ) is None


# ── The per-claim check ──────────────────────────────────────────────────────


class TestCheckCitations:
    def test_cited_claims_pass_for_every_kind(self) -> None:
        check = check_citations(
            "Odin is the chief god of Norse mythology [Q123]. "
            "Lantern oil is kept in the hollow stores [ref:chunk_7].",
            RECORDS,
        )
        assert check.passed and check.action == "pass"
        assert check.matched_record_ids == ["Q123", "chunk_7"]
        assert check.uncited_record_ids == []

    def test_bare_qid_counts_as_citation(self) -> None:
        check = check_citations(
            "Odin, Q123, is the chief god of Norse mythology.", RECORDS,
        )
        assert "Q123" not in check.uncited_record_ids

    def test_uncited_claims_flagged_per_record(self) -> None:
        check = check_citations(
            "Odin is the chief god of Norse mythology. "
            "Lantern oil is kept in the hollow stores.",
            RECORDS,
        )
        assert not check.passed
        assert check.uncited_record_ids == ["Q123", "chunk_7"]
        assert check.uncited_markers == ["Q123", "ref:chunk_7"]
        assert check.action == "flag"
        assert check.regeneration_required is False

    def test_one_cited_one_uncited_flags_only_the_uncited(self) -> None:
        check = check_citations(
            "Odin is the chief god of Norse mythology [Q123]. "
            "Lantern oil is kept in the hollow stores.",
            RECORDS,
        )
        assert not check.passed
        assert check.uncited_record_ids == ["chunk_7"]

    def test_citation_in_another_sentence_does_not_cover(self) -> None:
        check = check_citations(
            "Lantern oil is kept in the hollow stores. "
            "Odin [Q123] is a god.",
            RECORDS,
        )
        assert "chunk_7" in check.uncited_record_ids
        assert "Q123" not in check.uncited_record_ids

    def test_non_qid_kinds_are_enforced(self) -> None:
        # The legacy gate could not see these kinds at all.
        place_check = check_citations(
            "Hollow Creek is a village by the river.", [_place()],
        )
        assert place_check.uncited_record_ids == ["Hollow Creek"]
        cited = check_citations(
            "Hollow Creek is a village by the river [geonames:5128581].",
            [_place()],
        )
        assert cited.passed
        edge_check = check_citations(
            "A ferry is used for crossing rivers.", [_edge()],
        )
        assert edge_check.uncited_record_ids == ["ferry"]
        edge_cited = check_citations(
            "A ferry is used for crossing rivers "
            "[conceptnet:ferry|UsedFor|crossing rivers].",
            [_edge()],
        )
        assert edge_cited.passed

    def test_no_matching_claims_passes(self) -> None:
        check = check_citations(
            "The weather over the fjord is calm and bright today.", RECORDS,
        )
        assert check.passed and check.matched_record_ids == []

    def test_foreign_qid_flagged(self) -> None:
        check = check_citations(
            "Odin is the chief god of Norse mythology [Q999].", RECORDS,
        )
        assert not check.passed
        assert check.foreign_markers == ["Q999"]
        # And the real record's claim is still uncited.
        assert check.uncited_record_ids == ["Q123"]

    def test_foreign_reference_marker_flagged(self) -> None:
        check = check_citations(
            "Lantern oil is kept in the hollow stores [ref:chunk_999].",
            RECORDS,
        )
        assert check.foreign_markers == ["ref:chunk_999"]

    def test_forge_note_is_not_a_foreign_marker(self) -> None:
        check = check_citations(
            "Odin is the chief god of Norse mythology [Q123]."
            "\n\n[Forge: limited verified citations available for "
            "this response.]",
            RECORDS,
        )
        assert check.foreign_markers == []

    def test_empty_text_and_empty_records(self) -> None:
        assert check_citations("", RECORDS).passed
        assert check_citations("Anything at all here.", []).passed

    def test_to_dict_shape(self) -> None:
        check = check_citations(
            "Odin is the chief god of Norse mythology.", RECORDS,
        )
        assert isinstance(check, CitationCheck)
        data = check.to_dict()
        assert data["passed"] is False
        assert data["uncited_markers"] == ["Q123"]
        assert data["claims"][0]["matched"][0]["record_id"] == "Q123"


# ── The gate ─────────────────────────────────────────────────────────────────


class TestGateWiring:
    def test_gate_flags_uncited_claim_and_names_the_marker(self) -> None:
        result = EnforcementGate().check(
            "Odin is the chief god of Norse mythology, a god of wisdom.",
            [], {"Q123"}, records=RECORDS,
        )
        assert result.citation_check is False and result.passed is False
        assert result.claim_check is not None
        assert result.claim_check.uncited_record_ids == ["Q123"]
        assert "Q123" in result.notes
        assert "[Forge:" in result.text

    def test_gate_passes_a_cited_claim(self) -> None:
        result = EnforcementGate().check(
            "Odin [Q123] is the chief god of Norse mythology and wisdom.",
            ["Q123"], {"Q123"}, records=RECORDS,
        )
        assert result.citation_check is True and result.passed is True
        assert result.claim_check is not None and result.claim_check.passed

    def test_gate_enforces_a_bundle_with_no_qids(self) -> None:
        # Pre-TF-049 this bundle skipped citation enforcement
        # entirely (the legacy floor keys on QIDs only).
        result = EnforcementGate().check(
            "Lantern oil is kept in the hollow stores near the dock.",
            [], set(), records=[_chunk()],
        )
        assert result.citation_check is False and result.passed is False
        assert "ref:chunk_7" in result.notes

    def test_gate_without_records_is_the_legacy_gate(self) -> None:
        gate = EnforcementGate()
        failed = gate.check("No citations in this response.", [], {"Q123"})
        assert failed.citation_check is False and failed.claim_check is None
        passed = gate.check(
            "According to Q123, this is a true and stated fact.",
            ["Q123"], {"Q123"},
        )
        assert passed.citation_check is True and passed.claim_check is None


class TestCoreHelpers:
    def test_check_citation_gate_delegates(self) -> None:
        bundle = MemoryActivationBundle(activated_records=[_chunk()])
        # A claim matching the chunk, uncited: the old gate
        # passed this (no Wikidata records); the delegated
        # per-claim check does not.
        assert _check_citation_gate(
            "Lantern oil is kept in the hollow stores.", bundle,
        ) is False
        assert _check_citation_gate(
            "Lantern oil is kept in the hollow stores [ref:chunk_7].",
            bundle,
        ) is True

    def test_check_citation_gate_legacy_pins_hold(self) -> None:
        assert _check_citation_gate("any text", MemoryActivationBundle()) is True
        bundle = MemoryActivationBundle(activated_records=[
            ActivatedRecord("Q123", "entity", 0.9, "Odin", raw={"qid": "Q123"}),
            ActivatedRecord("Q456", "entity", 0.8, "Yggdrasil", raw={"qid": "Q456"}),
        ])
        assert _check_citation_gate("Odin (Q123) is the chief god.", bundle) is True
        assert _check_citation_gate(
            "Odin is the chief god of Norse mythology.", bundle,
        ) is False

    def test_extract_citations_speaks_every_kind(self) -> None:
        bundle = MemoryActivationBundle(activated_records=RECORDS)
        cited = _extract_citations(
            "Odin [Q123] and the stores [ref:chunk_7], not [Q999].", bundle,
        )
        assert set(cited) == {"Q123", "ref:chunk_7"}


# ── End to end through real core turns ───────────────────────────────────────

COMET_QID = "Q123456"
COMET_TEXT = "The Zorblax Comet has a tail of ice and dust"


def _comet_db(path: Path) -> Path:
    """The TF-043 fixture corpus plus one entity with a
    real-shaped QID (the corpus's QF-ids do not match the
    legacy floor's QID regex — TF-048's fixture note) whose
    vocabulary is distinctive to it alone."""
    build_fixture_db(path)
    conn = get_connection(path)
    try:
        conn.execute(
            "INSERT INTO entities (qid, label_en, description_en,"
            " aliases_en, instance_of, popularity_score, source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (COMET_QID, "Zorblax Comet",
             "zorblax comet tail ice zorblax comet orbits sun "
             "zorblax comet tail ice dust comet", "Zorblax Comet",
             "comet", 1.0, "wikidata"),
        )
        conn.commit()
        conn.execute("INSERT INTO entities_fts(entities_fts) VALUES('rebuild')")
        conn.commit()
    finally:
        conn.close()
    return path


class _FixedBackend(UnifiedBackend):
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, request):
        return GenerationResponse(
            text=self._text, tokens_generated=len(self._text.split()),
        )

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf049-stub"


@pytest.fixture(scope="module")
def comet_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _comet_db(tmp_path_factory.mktemp("tf049") / "comet.db")


def _core(db: Path, text: str) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        config={}, memory_dir=Path(tmp) / "mem", db_path=db,
        backend=_FixedBackend(text),
    )


class TestCoreEndToEnd:
    def test_uncited_claim_is_flagged_and_retained(self, comet_db: Path) -> None:
        core = _core(comet_db, COMET_TEXT + " in the night sky above us.")
        final = core.think("Zorblax comet", retrieval_path="sql")
        assert core.last_citation_check is not None
        assert COMET_QID in core.last_citation_check.uncited_record_ids
        assert final.enforcement_passed is False
        assert COMET_QID in final.enforcement_notes

    def test_cited_claim_passes_end_to_end(self, comet_db: Path) -> None:
        core = _core(comet_db, COMET_TEXT + f" [{COMET_QID}] in the sky.")
        final = core.think("Zorblax comet", retrieval_path="sql")
        assert core.last_citation_check is not None
        assert core.last_citation_check.passed
        assert core.last_citation_check.matched_record_ids == [COMET_QID]
        assert final.enforcement_passed is True

    def test_knowledge_only_turn_runs_no_claim_check(
        self, comet_db: Path,
    ) -> None:
        tmp = tempfile.mkdtemp()
        core = ThoughtForgeCore(
            config={}, memory_dir=Path(tmp) / "mem", db_path=comet_db,
        )
        assert core.last_citation_check is None
        core.think("Zorblax comet", retrieval_path="sql")
        # No candidates composed — nothing was generated, so
        # no claim check ran and none is retained.
        assert core.last_citation_check is None

    def test_freeze_additions_record_stands(self) -> None:
        from thoughtforge.cognition.core_docs import additions_problems

        assert additions_problems() == []
