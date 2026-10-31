"""
TF-041 — Knowledge store mapped: models/lifecycle/scoring/store
documented from code; the store's invariants listed + tested.

Part III's first slice. The map lives as verifiable data in
``src/thoughtforge/knowledge/store_map.py``
(``KNOWLEDGE_DB_TABLES`` / ``KNOWLEDGE_DB_FTS`` /
``MEMORY_FILES`` / ``PERSISTED_RECORDS`` / ``TRANSIENT_RECORDS``
/ ``RETRIEVAL_MAP`` / ``INVARIANTS``, checked by
``store_map_problems()`` and ``bundle_problems()``), the
write-up is ``docs/architecture/KNOWLEDGE_STORE_MAP.md``, and
CORE_LOOP §8-F2 is FIXED here: ``core._retrieve`` gates on the
configured ``self._knowledge.db_path``, never the default path
(the TF-039 findings record was flipped to closed with this
evidence, as its checker instructed).

Pinned here:

  * the map is complete and true: 10 DB tables + 3 FTS indexes,
    7 memory files, 6 persisted + 9 transient record types,
    invariants K1–K16 all stated, and the live checker finds
    no drift in the machine-checkable ones (K1–K10),
  * the behavioural invariants hold on synthetic stores
    (K11–K16): bundle bookkeeping on a populated synthetic DB,
    round-trips, rewrite-as-deletion, malformed-line tolerance,
    tag index / recent queue shape, protected records surviving
    every prune mode, and the forge's fusion arithmetic equal
    to the mapped proxy formula exactly,
  * the threshold arithmetic is honest: on a one-row FTS
    corpus the mapped proxies score real SQL hits BELOW the
    0.45 default threshold (bundle empty at the default,
    populated at min_score=0) — mapped for TF-043/TF-044,
    not retuned here,
  * F2 is fixed behaviourally, both ways: a populated custom
    DB is retrieved while the DEFAULT path is absent, and a
    missing custom DB is not even created while the default
    path exists and is populated.

Proof grade: FIXTURE (synthetic DB + synthetic memory stores
on this VM). The vector path is NOT exercised — sentence-
transformers is absent on this VM (D-ENV-1); TF-053 owns
embedding availability per profile.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from thoughtforge.cognition import core as core_module
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.etl.schema import get_connection, initialize_schema
from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.lifecycle import MemoryLifecycle
from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    EpisodicMemoryRecord,
    InputSketch,
    PersonalityCoreRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.scoring import (
    MIN_MEMORY_SCORE,
    RetrievalDimensions,
    score_retrieval,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.knowledge.store_map import (
    INVARIANTS,
    KNOWLEDGE_DB_FTS,
    KNOWLEDGE_DB_TABLES,
    MEMORY_FILES,
    PERSISTED_RECORDS,
    RETRIEVAL_MAP,
    TRANSIENT_RECORDS,
    bundle_problems,
    store_map_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_DOC = REPO_ROOT / "docs" / "architecture" / "KNOWLEDGE_STORE_MAP.md"
CORE_LOOP_DOC = REPO_ROOT / "docs" / "architecture" / "CORE_LOOP.md"


def build_corpus_db(path: Path, *, fillers: int = 50) -> Path:
    """A synthetic SQLite knowledge DB with one known right answer.

    One Lantern Hollow entity (the invented TF-017 world) plus
    filler entities so FTS bm25 scores land in a realistic range
    (a one-row corpus scores near zero — see the threshold test).
    """
    initialize_schema(path)
    conn = get_connection(path)
    try:
        conn.execute(
            "INSERT INTO entities (qid, label_en, description_en,"
            " aliases_en, instance_of, popularity_score, source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "Q_SYN1",
                "Lantern Hollow",
                "ferry ferry ferry crossing river settlement "
                "lantern hollow ferry",
                "Lantern Hollow",
                "settlement",
                0.9,
                "wikidata",
            ),
        )
        for i in range(fillers):
            conn.execute(
                "INSERT INTO entities (qid, label_en, description_en,"
                " aliases_en, instance_of, popularity_score, source)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"QF{i}", f"filler {i}",
                 "unrelated filler text about mountains and deserts",
                 "x", "place", 0.1, "wikidata"),
            )
        conn.commit()
        conn.execute("INSERT INTO entities_fts(entities_fts) VALUES('rebuild')")
        conn.commit()
    finally:
        conn.close()
    return path


def make_core(tmp_path: Path, db_path: Path) -> ThoughtForgeCore:
    return ThoughtForgeCore(
        config={"cognition": {}},
        memory_dir=tmp_path / "memory",
        db_path=db_path,
    )


# ── The map itself ───────────────────────────────────────────────────────────


class TestMapCompleteness:
    def test_db_tables_and_fts(self) -> None:
        assert len(KNOWLEDGE_DB_TABLES) == 10
        assert len(KNOWLEDGE_DB_FTS) == 3
        assert {r["name"] for r in KNOWLEDGE_DB_FTS} == {
            "entities_fts", "reference_fts", "conceptnet_fts",
        }

    def test_memory_files(self) -> None:
        assert [r["file"] for r in MEMORY_FILES] == [
            "personality_core.yaml",
            "user_profile_store.jsonl",
            "episodic_store.jsonl",
            "response_patterns.jsonl",
            "active_thread_state.json",
            "tag_index.json",
            "recent_queue.json",
        ]

    def test_record_census(self) -> None:
        assert len(PERSISTED_RECORDS) == 6
        assert len(TRANSIENT_RECORDS) == 9
        assert "WritebackRecord" in TRANSIENT_RECORDS

    def test_invariants_k1_to_k16_all_stated(self) -> None:
        assert [row["id"] for row in INVARIANTS] == [
            f"K{i}" for i in range(1, 17)
        ]
        assert {row["check"] for row in INVARIANTS} == {"problems", "tested"}

    def test_retrieval_map_defaults(self) -> None:
        assert RETRIEVAL_MAP["paths"] == ("sql", "vector", "hybrid")
        assert RETRIEVAL_MAP["scoring"]["weights_row_for_every_record"] == "user_fact"
        assert RETRIEVAL_MAP["scoring"]["tone_similarity"] == 0.5

    def test_live_checker_finds_no_drift(self) -> None:
        assert store_map_problems() == []

    def test_doc_names_every_invariant(self) -> None:
        text = MAP_DOC.read_text(encoding="utf-8")
        for row in INVARIANTS:
            assert row["id"] in text


# ── Bundle invariants on a synthetic DB (K11, K16) ───────────────────────────


class TestBundleInvariants:
    def test_populated_bundle_honours_its_contract(self, tmp_path: Path) -> None:
        db = build_corpus_db(tmp_path / "corpus.db")
        forge = KnowledgeForge(db_path=db)
        bundle = forge.retrieve("Lantern Hollow ferry", path="sql", top_k=10)
        assert bundle.total_retrieved >= 1
        assert bundle_problems(bundle, top_k=10, min_score=MIN_MEMORY_SCORE) == []
        top = bundle.activated_records[0]
        assert top.record_id == "Q_SYN1"
        # Measured on this VM at map time: the known answer's
        # fused score (bm25 14.92 → sim 0.746 through the mapped
        # proxies). Pinned as a range — bm25 is corpus-shaped.
        assert 0.60 < top.score < 0.72

    def test_top_k_caps_the_bundle(self, tmp_path: Path) -> None:
        db = build_corpus_db(tmp_path / "corpus.db")
        forge = KnowledgeForge(db_path=db)
        bundle = forge.retrieve(
            "Lantern Hollow ferry filler", path="sql", top_k=2, min_score=0.0
        )
        assert bundle.total_retrieved <= 2
        assert bundle_problems(bundle, top_k=2, min_score=0.0) == []

    def test_threshold_arithmetic_on_a_tiny_corpus(self, tmp_path: Path) -> None:
        # The mapped proxies on a one-row corpus: FTS bm25 is
        # near zero, so semantic similarity ~0 and even a real
        # hit scores below the 0.45 default threshold. This is
        # TF-043/TF-044's surface — pinned here as arithmetic,
        # not retuned.
        db = build_corpus_db(tmp_path / "tiny.db", fillers=0)
        forge = KnowledgeForge(db_path=db)
        sql_hits = forge.sql_retrieve("Lantern Hollow ferry", top_k=15)
        assert any(r.get("qid") == "Q_SYN1" for r in sql_hits)
        default_bundle = forge.retrieve("Lantern Hollow ferry", path="sql")
        assert default_bundle.total_retrieved == 0
        open_bundle = forge.retrieve(
            "Lantern Hollow ferry", path="sql", min_score=0.0
        )
        assert open_bundle.total_retrieved >= 1
        assert bundle_problems(open_bundle, top_k=10, min_score=0.0) == []

    def test_fusion_arithmetic_is_the_mapped_proxy(self, tmp_path: Path) -> None:
        forge = KnowledgeForge(db_path=tmp_path / "any.db")
        record = {"source": "reference", "popularity_score": 0.0, "fts_score": 10.0}
        scored = forge._score_records([dict(record)], "any query")
        dims = RetrievalDimensions(
            semantic_similarity=0.5,       # min(10/20, 1)
            tone_similarity=0.5,           # neutral proxy
            preference_relevance=0.5,      # neutral proxy
            recency=0.9,                   # the reference source proxy
            importance=0.5,                # min(0.0 + default 0.5, 1)
        )
        assert scored[0]["_composite_score"] == score_retrieval(dims, "user_fact")
        assert scored[0]["_composite_score"] == pytest.approx(0.49)

    def test_nonsense_query_retrieves_nothing(self, tmp_path: Path) -> None:
        db = build_corpus_db(tmp_path / "corpus.db")
        forge = KnowledgeForge(db_path=db)
        bundle = forge.retrieve(
            "zzzzqqqq nonexistentterm", path="sql", min_score=0.0
        )
        assert bundle.total_retrieved == 0
        assert bundle.retrieval_confidence == 0.0


# ── F2: the retrieval gate reads the configured path ─────────────────────────


class TestRetrievalGateFix:
    def test_custom_db_retrieved_while_default_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = build_corpus_db(tmp_path / "custom.db")
        core = make_core(tmp_path, db)
        monkeypatch.setattr(
            core_module, "get_knowledge_db_path",
            lambda: tmp_path / "default_missing.db",
        )
        sketch = InputSketch(
            raw_text="Lantern Hollow ferry", retrieval_path="sql"
        )
        bundle = core._retrieve(sketch)
        assert bundle.total_retrieved >= 1
        assert bundle.activated_records[0].record_id == "Q_SYN1"

    def test_missing_custom_db_not_created_while_default_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        default_db = build_corpus_db(tmp_path / "default.db")
        custom_db = tmp_path / "custom_missing.db"
        core = make_core(tmp_path, custom_db)
        monkeypatch.setattr(
            core_module, "get_knowledge_db_path", lambda: default_db
        )
        sketch = InputSketch(
            raw_text="Lantern Hollow ferry", retrieval_path="sql"
        )
        bundle = core._retrieve(sketch)
        assert bundle.total_retrieved == 0
        # The pre-fix gate passed (default exists) and retrieval
        # then touched the custom path, creating the file; the
        # fixed gate returns before retrieval ever runs.
        assert not custom_db.exists()

    def test_core_loop_doc_records_f2_closed(self) -> None:
        text = CORE_LOOP_DOC.read_text(encoding="utf-8")
        assert "TF-041 update: FIXED" in text


# ── MemoryStore behavioural invariants (K12–K14) ─────────────────────────────


class TestMemoryStoreInvariants:
    def test_round_trip_every_persisted_type(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        store.save_personality_core(
            PersonalityCoreRecord(record_id="pers_syn", traits=["plain-spoken"])
        )
        store.append_profile_record(
            UserPreferenceRecord(
                record_id="pref_1", category="tone", tags=["calm"],
                summary="prefers calm answers", value="calm",
            )
        )
        store.append_profile_record(
            UserFactRecord(
                record_id="fact_1", tags=["ferry"],
                summary="the ferry takes eleven minutes",
                fact_value={"crossing_minutes": 11},
            )
        )
        store.append_episode(
            EpisodicMemoryRecord(
                record_id="ep_1", tags=["river"], summary="the river ran high"
            )
        )
        store.append_pattern(
            ResponsePatternRecord(
                record_id="pat_1", tags=["shape"], pattern_shape=["answer"]
            )
        )
        store.save_thread_state(
            ActiveThreadStateRecord(record_id="thr_1", topic="ferry times")
        )
        loaded = store.load_all()
        assert loaded["personality_core"].traits == ["plain-spoken"]
        by_id = {r.record_id: r for r in loaded["profile_records"]}
        assert by_id["fact_1"].fact_value == {"crossing_minutes": 11}
        assert by_id["pref_1"].value == "calm"
        assert loaded["episodes"][0].summary == "the river ran high"
        assert loaded["patterns"][0].pattern_shape == ["answer"]
        assert loaded["thread_state"].topic == "ferry times"

    def test_from_dict_ignores_unknown_keys(self) -> None:
        record = UserFactRecord.from_dict(
            {"record_id": "f1", "summary": "s", "bogus_key": 1}
        )
        assert record.record_id == "f1"
        assert not hasattr(record, "bogus_key")

    def test_missing_files_load_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "fresh")
        assert store.load_personality_core() is None
        assert store.load_profile_records() == []
        assert store.load_episodes() == []
        assert store.load_patterns() == []
        assert store.load_thread_state() is None
        assert store.load_tag_index() == {}
        assert store.load_recent_queue() == []

    def test_malformed_jsonl_lines_are_skipped(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        store.append_episode(
            EpisodicMemoryRecord(record_id="ep_good", summary="good line")
        )
        with store._episodic_path.open("a", encoding="utf-8") as f:
            f.write("{not json\n")
            f.write("\n")
        episodes = store.load_episodes()
        assert [e.record_id for e in episodes] == ["ep_good"]

    def test_rewrite_drops_archived_and_deleted(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        keep = UserFactRecord(record_id="keep_1", summary="stays")
        gone = UserFactRecord(record_id="gone_1", summary="goes")
        gone.status = "deleted"
        store.rewrite_profile_store([keep, gone])
        assert [r.record_id for r in store.load_profile_records()] == ["keep_1"]
        raw = store._profile_path.read_text(encoding="utf-8")
        assert "gone_1" not in raw

    def test_tag_index_shape(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        store.save_personality_core(
            PersonalityCoreRecord(record_id="pers_x", traits=["t"])
        )
        store.append_profile_record(
            UserFactRecord(record_id="f1", tags=["river", "ferry"], summary="s")
        )
        store.append_episode(
            EpisodicMemoryRecord(record_id="e1", tags=["river"], summary="s")
        )
        store.append_episode(
            EpisodicMemoryRecord(record_id="e2", tags=["river"], summary="s")
        )
        index = store.rebuild_tag_index()
        assert index["river"] == ["f1", "e1", "e2"]  # profile, then episodes
        assert index["ferry"] == ["f1"]
        assert all("pers_x" not in ids for ids in index.values())

    def test_recent_queue_shape_and_cap(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        for i in range(70):
            store.push_to_recent_queue(f"rec_{i}")
        queue = store.load_recent_queue()
        assert len(queue) == 64
        assert queue[0] == "rec_69"  # most recent first
        store.push_to_recent_queue("rec_5")
        queue = store.load_recent_queue()
        assert queue[0] == "rec_5"
        assert queue.count("rec_5") == 1  # deduped


# ── Lifecycle invariants (K15) ───────────────────────────────────────────────


class TestLifecycleInvariants:
    def _ancient(self) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()

    def test_protected_record_survives_every_mode(self, tmp_path: Path) -> None:
        old = self._ancient()
        for mode in ("light", "routine", "heavy", "emergency"):
            store = MemoryStore(memory_dir=tmp_path / mode)
            store.append_profile_record(
                UserPreferenceRecord(
                    record_id="prot_1", category="tone", tags=["x"],
                    summary="protected pref", retention_class="protected",
                    last_used_at=old, created_at=old, updated_at=old,
                )
            )
            MemoryLifecycle(store=store).prune(mode)  # type: ignore[arg-type]
            survivors = store.load_profile_records()
            assert [r.record_id for r in survivors] == ["prot_1"]
            assert survivors[0].status == "created"  # untouched entirely

    def test_ancient_episode_pruned_by_emergency(self, tmp_path: Path) -> None:
        old = self._ancient()
        store = MemoryStore(memory_dir=tmp_path / "mem")
        store.append_episode(
            EpisodicMemoryRecord(
                record_id="ep_old", tags=["x"], summary="ancient episode",
                last_used_at=old, created_at=old, updated_at=old,
            )
        )
        counts = MemoryLifecycle(store=store).prune("emergency")
        assert counts["archived"] + counts["deleted"] >= 1
        # K13: once marked, the rewrite leaves it out of the file.
        assert store.load_episodes() == []

    def test_thread_cooling_clears_a_stale_thread(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        thread = ActiveThreadStateRecord(
            record_id="thr_old", topic="done topic", priority=0.25,
            expires_after_turns=1,
        )
        store.save_thread_state(thread)
        MemoryLifecycle(store=store).cool_thread()
        assert store.load_thread_state() is None
