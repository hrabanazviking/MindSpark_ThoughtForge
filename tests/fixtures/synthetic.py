"""TF-017 — synthetic canon stand-ins for tests.

The fixture policy (docs/audit/TF-017_FIXTURE_POLICY.md): no test ever
needs real data. Every stand-in a test might want — a persona, a
reference library, a populated memory store, conversation turns —
lives here or beside this module under tests/fixtures/, is invented
whole, and carries the SYNTHETIC marker. Canon (the repo's data/
tree, its persona material, and its knowledge_reference library)
is never read by a test.

The invented world is Lantern Hollow on the River Sarn, kept by
Wrenna the Lantern-Keeper. It resembles the real shapes (a persona
core, reference documents in .md/.jsonl/.yaml, memory records) so
the code paths are exercised honestly, and resembles nothing real.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from thoughtforge.knowledge.models import (
    EpisodicMemoryRecord,
    PersonalityCoreRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.store import MemoryStore

FIXTURES_DIR = Path(__file__).parent
REFERENCE_DIR = FIXTURES_DIR / "synthetic_reference"
PERSONALITY_PATH = FIXTURES_DIR / "synthetic_personality_core.yaml"

SYNTHETIC_MARKER = "SYNTHETIC_FIXTURE"
PERSONA_NAME = "Wrenna the Lantern-Keeper"
PLACE_NAME = "Lantern Hollow"

# Conversation turns in the invented world, for cognition/chat tests.
SYNTHETIC_TURNS: list[tuple[str, str]] = [
    ("How long does the ferry crossing take?",
     "Eleven minutes in calm water, between the willow narrows and "
     "the mill landing at Lantern Hollow."),
    ("When does the ferry stop running?",
     "After the second tide bell at dusk. No ferry runs after that."),
    ("What does Wrenna do when the river runs high?",
     "She hangs a red lantern at the ferry post so travellers know "
     "the crossing is closed."),
    ("Who records the river level?",
     "Wrenna records it each morning in chalk on the mill door."),
]


def load_synthetic_personality() -> PersonalityCoreRecord:
    """The synthetic persona core, loaded through the real shape."""
    data = yaml.safe_load(PERSONALITY_PATH.read_text(encoding="utf-8"))
    return PersonalityCoreRecord.from_dict(data)


def synthetic_reference_text() -> str:
    """All synthetic reference documents concatenated (md+jsonl+yaml)."""
    parts: list[str] = []
    for path in sorted(REFERENCE_DIR.iterdir()):
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    parts.append(json.loads(line)["fact"])
        else:
            parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def build_synthetic_store(memory_dir: Path) -> MemoryStore:
    """A MemoryStore populated only with synthetic records.

    Persona core + one preference + one fact + one episode, all in
    the invented world — enough for retrieval/writeback tests to run
    against known content with known right answers.
    """
    store = MemoryStore(memory_dir=memory_dir)
    store.save_personality_core(load_synthetic_personality())
    store.append_profile_record(UserPreferenceRecord(
        record_id="syn_pref_001",
        category="tone",
        tags=["plain-spoken", "synthetic"],
        summary="prefers plain-spoken answers about Lantern Hollow",
        value="plain-spoken",
        importance=0.8,
    ))
    store.append_profile_record(UserFactRecord(
        record_id="syn_fact_001",
        tags=["lantern-hollow", "ferry", "synthetic"],
        summary="the River Sarn ferry crossing takes eleven minutes",
        fact_value={"place": PLACE_NAME, "crossing_minutes_calm": 11},
    ))
    store.append_episode(EpisodicMemoryRecord(
        record_id="syn_ep_001",
        tags=["river", "synthetic"],
        summary="the river ran high and the red lantern was hung",
        importance=0.7,
        quality=0.8,
    ))
    return store
