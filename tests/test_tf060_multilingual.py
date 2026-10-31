"""TF-060 — multilingual memory: non-English facts stored
and retrieved without translation loss (fixtures in two
languages: Spanish + German, TF-017's invented world).

Pins knowledge/multilingual.py against the live tree:
verbatim round-trips (store files + the TF-052 bundle),
folding (casefold ß -> ss, NFC), diacritic-stripped
auxiliary matching, key-term extraction (es/de stopwords
filtered, a two-character spaceless word survives, the
standing English example reproduces the pre-slice rule),
search in each fixture language, accent-stripped and
shouting-case retrieval, the no-translation rule (an
English translation query retrieves neither fixture
fact), spaceless-script bigram retrieval, language
hints + the language report, and the wiring (the forge
delegates; TF-059 freshness search and TF-058 reference
search retrieve accent-stripped; the TF-057 toolset's
text filter folds). All fixtures are synthetic; no canon
is read.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.models import (
    EpisodicMemoryRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.multilingual import (
    ALL_STOPWORDS,
    EN_STOPWORDS_ORIGINAL,
    FIXTURE_LANGUAGES,
    MIN_TOKEN_LEN,
    MULTILINGUAL_RULES,
    STOPWORDS_BY_LANGUAGE,
    detect_language,
    fold_text,
    key_terms,
    language_report,
    match_tokens,
    multilingual_problems,
    relevance_share,
    search_store,
    strip_diacritics,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.knowledge.store_map import bundle_problems

ES_FACT = UserFactRecord(
    record_id="ml_es", tags=["faro", "travesía"],
    summary="el faro rojo guía la travesía nocturna",
    fact_value={"lugar": "biblioteca del faro"},
    source="synthetic_fixture")
DE_FACT = UserFactRecord(
    record_id="ml_de", tags=["straße", "hafen"],
    summary="die straße führt zum hafen und zur fähre",
    fact_value={"ort": "hafenstraße"},
    source="synthetic_fixture")
DE_EPISODE = EpisodicMemoryRecord(
    record_id="ml_ep_de", tags=["fähre"],
    summary="die fähre fuhr bei sonnenaufgang über den fluss",
    source="synthetic_fixture")
ES_PREF = UserPreferenceRecord(
    record_id="ml_pref_es", category="tone", tags=["respuestas"],
    summary="la viajera prefiere las respuestas claras y breves",
    value="claras y breves", source="synthetic_fixture")
JA_FACT = UserFactRecord(
    record_id="ml_ja", tags=["toudai"],
    summary="赤い灯台が夜の渡航を導く",
    source="synthetic_fixture")


def _store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(memory_dir=tmp_path / "mem")
    for record in (ES_FACT, DE_FACT, ES_PREF):
        store.append_profile_record(record)
    store.append_episode(DE_EPISODE)
    store.append_profile_record(JA_FACT)
    return store


# ── The rules and tables, as data ────────────────────────────────────────────


class TestDataTables:
    def test_problems_empty_on_live_tree(self) -> None:
        assert multilingual_problems() == []

    def test_rules_and_languages(self) -> None:
        assert [r["name"] for r in MULTILINGUAL_RULES] == [
            "verbatim_storage", "derived_matching_keys",
            "no_translation_retrieval", "stopwords_as_union",
            "hint_not_field", "read_only"]
        assert FIXTURE_LANGUAGES == ("es", "de")
        assert set(STOPWORDS_BY_LANGUAGE) == {"en", "es", "de"}
        assert MIN_TOKEN_LEN == 2

    def test_english_stopwords_preserve_the_preslice_list(self) -> None:
        en = STOPWORDS_BY_LANGUAGE["en"]
        for word in EN_STOPWORDS_ORIGINAL:
            assert word in en
        assert "está" in ALL_STOPWORDS and "die" in ALL_STOPWORDS


# ── Folding ──────────────────────────────────────────────────────────────────


class TestFolding:
    def test_casefold_maps_eszett(self) -> None:
        assert fold_text("Die Straße") == "die strasse"
        assert fold_text("STRASSE") == fold_text("straße")

    def test_fold_is_nfc_and_case_insensitive(self) -> None:
        assert fold_text("GUÍA") == "guía"
        # decomposed input (a + combining acute) folds to the
        # same NFC form as the precomposed character
        assert fold_text("guía") == "guía"

    def test_strip_diacritics(self) -> None:
        assert strip_diacritics("guía") == "guia"
        assert strip_diacritics("travesía") == "travesia"
        assert strip_diacritics("año") == "ano"  # the stated conflation
        assert strip_diacritics("ø") == "ø"      # not mark-separable

    def test_match_tokens_spaceless_bigrams(self) -> None:
        tokens = match_tokens("赤い灯台が夜の渡航を導く")
        assert "灯台" in tokens and "渡航" in tokens
        assert "fähre" in match_tokens("Die Fähre fährt")


# ── Key terms ────────────────────────────────────────────────────────────────


class TestKeyTerms:
    def test_spanish_stopwords_filtered(self) -> None:
        assert key_terms("¿dónde está la biblioteca del faro") == \
            ["biblioteca", "faro"]
        assert key_terms("el faro y la travesía") == ["faro", "travesía"]

    def test_german_stopwords_filtered_and_folded(self) -> None:
        assert key_terms("wo ist die fähre nach lantern") == \
            ["fähre", "lantern"]
        assert key_terms("die straße zum hafen") == ["strasse", "hafen"]

    def test_two_character_spaceless_word_survives(self) -> None:
        assert key_terms("灯台") == ["灯台"]

    def test_standing_english_behaviour_preserved(self) -> None:
        query = "what is the ferry crossing time at the hollow"
        clean = re.sub(r"[^\w\s]", " ", query.lower())
        legacy = [w for w in clean.split()
                  if w not in set(EN_STOPWORDS_ORIGINAL) and len(w) > 2][:5]
        assert key_terms(query) == legacy == \
            ["ferry", "crossing", "time", "hollow"]

    def test_forge_delegates(self) -> None:
        assert KnowledgeForge._extract_key_terms(
            "¿dónde está la biblioteca del faro", max_terms=8) == \
            ["biblioteca", "faro"]
        assert KnowledgeForge._extract_key_terms("straße") == ["strasse"]


# ── Verbatim storage ─────────────────────────────────────────────────────────


class TestVerbatimStorage:
    def test_store_round_trip_byte_for_byte(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        loaded = {r.record_id: r for r in store.load_profile_records()}
        assert loaded["ml_es"].to_dict() == ES_FACT.to_dict()
        assert loaded["ml_de"].to_dict() == DE_FACT.to_dict()
        assert loaded["ml_pref_es"].to_dict() == ES_PREF.to_dict()
        assert loaded["ml_ja"].to_dict() == JA_FACT.to_dict()
        episodes = {r.record_id: r for r in store.load_episodes()}
        assert episodes["ml_ep_de"].to_dict() == DE_EPISODE.to_dict()
        raw = (tmp_path / "mem" / "user_profile_store.jsonl").read_text(
            encoding="utf-8")
        assert "travesía" in raw and "straße" in raw and "\\u" not in raw

    def test_portability_bundle_keeps_text(self, tmp_path: Path) -> None:
        from thoughtforge.knowledge.portability import (
            export_store,
            import_store,
        )

        store = _store(tmp_path)
        target = MemoryStore(memory_dir=tmp_path / "imported")
        import_store(target, export_store(store), mode="replace")
        second = export_store(target)  # TF-052: stable from the 2nd export
        summaries = {row["record_id"]: row["summary"]
                     for row in second["sections"]["profile_records"]}
        assert summaries["ml_es"] == ES_FACT.summary
        assert summaries["ml_de"] == DE_FACT.summary

    def test_tag_index_keeps_non_english_tags(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        index = store.rebuild_tag_index()
        assert index["travesía"] == ["ml_es"]
        assert index["straße"] == ["ml_de"]


# ── Search in each fixture language ──────────────────────────────────────────


class TestSearch:
    def test_spanish_query_retrieves_spanish_fact_verbatim(
            self, tmp_path: Path) -> None:
        bundle, report = search_store(_store(tmp_path), "faro travesía")
        assert [r.record_id for r in bundle.activated_records] == ["ml_es"]
        record = bundle.activated_records[0]
        assert record.raw["summary"] == ES_FACT.summary
        assert record.raw["language"]["language"] == "es"
        assert record.score == 1.0
        assert report["query_language"] == "es"
        assert bundle_problems(bundle, top_k=10, min_score=0.0) == []

    def test_german_query_retrieves_german_fact(self, tmp_path: Path) -> None:
        bundle, report = search_store(_store(tmp_path), "straße hafen")
        assert [r.record_id for r in bundle.activated_records] == ["ml_de"]
        assert bundle.activated_records[0].raw["language"]["language"] == "de"
        assert report["query_language"] == "de"

    def test_accent_stripped_query_still_finds_the_fact(
            self, tmp_path: Path) -> None:
        bundle, _ = search_store(_store(tmp_path), "guia travesia")
        assert [r.record_id for r in bundle.activated_records] == ["ml_es"]
        assert bundle.activated_records[0].score == 1.0

    def test_shouting_casefold_query_finds_eszett_fact(
            self, tmp_path: Path) -> None:
        bundle, _ = search_store(_store(tmp_path), "STRASSE HAFEN")
        assert [r.record_id for r in bundle.activated_records] == ["ml_de"]

    def test_no_translation_retrieval(self, tmp_path: Path) -> None:
        bundle, _ = search_store(
            _store(tmp_path), "lighthouse guides the night crossing")
        ids = [r.record_id for r in bundle.activated_records]
        assert "ml_es" not in ids and "ml_de" not in ids

    def test_shared_token_is_the_only_crossing(self, tmp_path: Path) -> None:
        # "hafen" is genuinely in the German fact's text and
        # tags — a query sharing that token crosses by token,
        # not by translation.
        assert relevance_share(["hafen"], DE_FACT.summary) == 1.0
        assert relevance_share(["harbour"], DE_FACT.summary) == 0.0

    def test_spaceless_query_retrieves_by_bigram(
            self, tmp_path: Path) -> None:
        bundle, _ = search_store(_store(tmp_path), "灯台")
        assert [r.record_id for r in bundle.activated_records] == ["ml_ja"]

    def test_empty_and_stopword_only_queries(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert search_store(store, "")[0].total_retrieved == 0
        assert search_store(store, "el la die das")[0].total_retrieved == 0

    def test_search_is_read_only(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        mem = tmp_path / "mem"
        before = {p.name: p.read_bytes()
                  for p in sorted(mem.iterdir()) if p.is_file()}
        search_store(store, "faro travesía")
        search_store(store, "STRASSE HAFEN")
        language_report(store)
        after = {p.name: p.read_bytes()
                 for p in sorted(mem.iterdir()) if p.is_file()}
        assert after == before


# ── Language hints & report ──────────────────────────────────────────────────


class TestLanguageHints:
    def test_hints(self) -> None:
        assert detect_language(ES_FACT.summary).language == "es"
        assert detect_language(DE_FACT.summary).language == "de"
        assert detect_language(
            "the ferry crosses the river").language == "en"
        assert detect_language("zzqx").language == "unknown"
        assert detect_language("赤い灯台").language == "cjk"

    def test_report_counts_and_names(self, tmp_path: Path) -> None:
        report = language_report(_store(tmp_path))
        assert report["total"] == 5
        assert report["by_language"]["es"] == 2
        assert report["by_language"]["de"] == 2
        assert report["by_language"]["cjk"] == 1
        rows = {r["record_id"]: r for r in report["records"]}
        assert rows["ml_es"]["language"] == "es"
        assert rows["ml_de"]["language"] == "de"
        assert "heuristic" in report["note"]


# ── Wiring into the Part III searches ────────────────────────────────────────


class TestWiring:
    def test_freshness_search_matches_accent_stripped(
            self, tmp_path: Path) -> None:
        from thoughtforge.knowledge.freshness import (
            search_store as freshness_search,
        )

        bundle, _ = freshness_search(_store(tmp_path), "guia travesia")
        assert "ml_es" in [r.record_id for r in bundle.activated_records]

    def test_reference_search_matches_accent_stripped(
            self, tmp_path: Path) -> None:
        from thoughtforge.knowledge.reference_library import ReferenceLibrary

        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "faro.md").write_text(
            "# El faro\n\nEl faro rojo guía la travesía nocturna "
            "en Lantern Hollow.\n", encoding="utf-8")
        bundle = ReferenceLibrary(lib).search("guia travesia")
        assert bundle.total_retrieved == 1
        assert bundle.activated_records[0].score == 1.0

    def test_toolset_text_filter_folds(self, tmp_path: Path) -> None:
        from thoughtforge.knowledge.forge_toolset import inspect_records

        store = _store(tmp_path)
        rows = inspect_records(store, text="STRASSE")
        assert [r["record_id"] for r in rows] == ["ml_de"]
        rows = inspect_records(store, text="travesía")
        assert [r["record_id"] for r in rows] == ["ml_es"]
