"""TF-017 — fixture policy: synthetic canon stand-ins, guarded.

Policy (docs/audit/TF-017_FIXTURE_POLICY.md): no test ever needs real
data. These tests prove the stand-ins work through the real code and
guard the estate: no test file may reach into the repo's data/ tree,
and no canon persona name may appear as test data.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TESTS_DIR = REPO_ROOT / "tests"


def _load_synthetic():
    spec = importlib.util.spec_from_file_location(
        "tf_synthetic_fixtures", TESTS_DIR / "fixtures" / "synthetic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


syn = _load_synthetic()

# Patterns that would mean a test is reading the real canon tree.
# (Asserting the string "data/" inside a .dockerignore-content test is
# not a read of data/ and is deliberately not matched here.)
CANON_ACCESS_PATTERNS = [
    # Repo-root-anchored construction into the canon tree, e.g.
    # REPO_ROOT / "data" / ... — the one shape that reads real data.
    re.compile(r"(?:REPO_ROOT|_REPO_ROOT|_REPO|REPO)\s*/\s*[\"']data[\"']"),
    re.compile(r"[\"']data/(?:knowledge_reference|knowledge\.db)"),
    re.compile(r"get_data_dir\(\)"),
    re.compile(r"get_knowledge_reference_dir\(\)"),
    re.compile(r"get_knowledge_db_path\(\)"),
    re.compile(r"knowledge_reference"),
    re.compile(r"pictures_of_"),
]

CANON_PERSONA_NAMES = ["Sigrid"]


def _test_sources() -> list[Path]:
    return sorted(TESTS_DIR.glob("test_*.py"))


class TestSyntheticStandIns:
    def test_personality_loads_through_real_shape(self):
        core = syn.load_synthetic_personality()
        assert core.record_id == "syn_wrenna_001"
        assert "steady" in core.traits
        assert core.retention_class == "protected"

    def test_personality_yaml_carries_marker(self):
        text = syn.PERSONALITY_PATH.read_text(encoding="utf-8")
        assert syn.SYNTHETIC_MARKER in text

    def test_reference_dir_files_all_marked_or_structured_synthetic(self):
        files = sorted(p for p in syn.REFERENCE_DIR.iterdir() if p.is_file())
        assert {p.suffix for p in files} == {".md", ".jsonl", ".yaml"}
        for p in files:
            text = p.read_text(encoding="utf-8")
            assert syn.SYNTHETIC_MARKER in text or '"synthetic_fixture": true' in text, p.name
            assert syn.PLACE_NAME in text or syn.PERSONA_NAME in text, p.name

    def test_reference_text_covers_the_invented_facts(self):
        text = syn.synthetic_reference_text()
        assert "eleven minutes" in text
        assert "second tide bell" in text
        assert "red lantern" in text

    def test_reference_etl_ingests_synthetic_dir_only(self, tmp_path):
        from thoughtforge.etl.sources import ReferenceDataETL

        db = tmp_path / "synthetic.db"
        stats = ReferenceDataETL(db_path=db).build(ref_dir=syn.REFERENCE_DIR)
        assert stats["files"] == 3
        assert stats["chunks"] >= 3
        assert stats["skipped"] == 0

    def test_store_built_from_stand_ins_round_trips(self, tmp_path):
        store = syn.build_synthetic_store(tmp_path)
        core = store.load_personality_core()
        assert core is not None and core.record_id == "syn_wrenna_001"
        records = store.load_profile_records()
        assert {r.record_type for r in records} == {"user_preference", "user_fact"}
        fact = next(r for r in records if r.record_type == "user_fact")
        assert fact.fact_value["crossing_minutes_calm"] == 11
        episodes = store.load_episodes()
        assert len(episodes) == 1 and "red lantern" in episodes[0].summary

    def test_turns_are_invented_world_only(self):
        assert len(syn.SYNTHETIC_TURNS) >= 4
        joined = " ".join(q + a for q, a in syn.SYNTHETIC_TURNS)
        assert syn.PLACE_NAME in joined


class TestFixturePolicyGuards:
    def test_no_test_reaches_into_canon_data(self):
        offenders: list[str] = []
        for path in _test_sources():
            if path.name == Path(__file__).name:
                continue  # this file names the patterns to forbid them
            text = path.read_text(encoding="utf-8")
            for pat in CANON_ACCESS_PATTERNS:
                if pat.search(text):
                    offenders.append(f"{path.name}: {pat.pattern}")
        assert offenders == [], f"canon access in tests: {offenders}"

    def test_no_canon_persona_name_as_test_data(self):
        offenders: list[str] = []
        for path in _test_sources():
            text = path.read_text(encoding="utf-8")
            for name in CANON_PERSONA_NAMES:
                if name in text and path.name != Path(__file__).name:
                    offenders.append(f"{path.name}: {name}")
        # The guard file itself lists the name in order to forbid it.
        assert offenders == [], f"canon persona name in tests: {offenders}"

    def test_fixtures_live_under_tests_fixtures(self):
        assert syn.FIXTURES_DIR == TESTS_DIR / "fixtures"
        assert syn.REFERENCE_DIR.is_dir()
        # Nothing synthetic may masquerade as canon: fixtures never
        # live under the repo's data/ tree.
        assert "data" not in syn.FIXTURES_DIR.relative_to(REPO_ROOT).parts
