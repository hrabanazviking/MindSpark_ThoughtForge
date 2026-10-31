"""TF-055 — persona memory separation.

Two synthetic Lantern Hollow personae (knowledge/persona_separation
fixtures) must never bleed: not through loads, indices, queues,
writeback, forgetting, lifecycle or construction — and not under
adversarial prompts run through real cores. The one bleed the
slice's probes found (D-PER-1: a core's startup SelfHealer
repaired the DEFAULT store whatever core was built) is pinned
fixed; the config personality assignment (D-PER-2) is pinned at
its ruled precedence: store core > config assignment > none.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import thoughtforge.utils.self_heal as self_heal_mod
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.knowledge.forgetting import forget_record
from thoughtforge.knowledge.models import WritebackRecord
from thoughtforge.knowledge.persona_separation import (
    ADVERSARIAL_PROMPTS,
    PERSONA_FIXTURES,
    SEPARATION_SURFACES,
    SHARED_RECORD_ID,
    STORE_FILES,
    build_persona_store,
    inspect_fixture_separation,
    persona_separation_problems,
    raw_scan,
)
from thoughtforge.knowledge.portability import export_store, import_store
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.knowledge.writeback import apply_writeback
from thoughtforge.utils.self_heal import SelfHealer

FX_A, FX_B = PERSONA_FIXTURES


def _root() -> Path:
    return Path(tempfile.mkdtemp(prefix="tf055-"))


def _pair(root: Path):
    a = build_persona_store(root / FX_A["key"], FX_A)
    b = build_persona_store(root / FX_B["key"], FX_B)
    return a, b


def _dir_bytes(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()}


def _core(memory_dir: Path, root: Path, **kw) -> ThoughtForgeCore:
    kw.setdefault("config", {"cognition": {}})
    return ThoughtForgeCore(
        memory_dir=memory_dir, db_path=root / "x.db",
        model_path=None, **kw,
    )


# ── The surfaces as data ───────────────────────────────────────────────────────


class TestSurfaces:
    def test_scopes_are_the_ruled_vocabulary(self) -> None:
        scopes = {row["scope"] for row in SEPARATION_SURFACES}
        assert scopes == {"per_persona", "shared_by_rule", "explicit_transfer"}

    def test_every_store_file_is_a_per_persona_surface_home(self) -> None:
        homes = " ".join(row["home"] for row in SEPARATION_SURFACES)
        for name in STORE_FILES:
            assert name in homes

    def test_only_crossing_road_is_the_named_import(self) -> None:
        transfers = [
            row for row in SEPARATION_SURFACES
            if row["scope"] == "explicit_transfer"
        ]
        assert [row["surface"] for row in transfers] == ["store bundle import"]

    def test_store_files_match_the_live_store(self) -> None:
        store = MemoryStore(memory_dir=_root() / "s")
        actual = {
            getattr(store, attr).name
            for attr in (
                "_personality_path", "_profile_path", "_episodic_path",
                "_patterns_path", "_thread_path", "_tag_index_path",
                "_recent_queue_path",
            )
        }
        assert actual == set(STORE_FILES)

    def test_problems_empty_on_live_tree(self) -> None:
        assert persona_separation_problems() == []


# ── Fixture separation ─────────────────────────────────────────────────────────


class TestFixtureSeparation:
    def test_two_personae_separate_on_every_check(self) -> None:
        report = inspect_fixture_separation(_root())
        assert report.separated, report.to_dict()

    def test_raw_scan_is_a_real_detector(self) -> None:
        # Negative control: plant persona A's secret in B's home —
        # the scan must find it (a scan that cannot fail proves
        # nothing).
        root = _root()
        _, b = _pair(root)
        (b.dir / "episodic_store.jsonl").write_text(
            FX_A["secret"] + "\n", encoding="utf-8"
        )
        hits = raw_scan(b.dir, [FX_A["secret"]])
        assert hits == {"episodic_store.jsonl": [FX_A["secret"]]}

    def test_same_directory_is_the_same_persona(self) -> None:
        # The rule, stated as behaviour: a directory IS the home.
        root = _root()
        a, _ = _pair(root)
        twin = MemoryStore(memory_dir=a.dir)
        core = twin.load_personality_core()
        assert core is not None and core.record_id == FX_A["core_id"]
        assert [r.record_id for r in twin.load_profile_records()] == [
            r.record_id for r in a.load_profile_records()
        ]

    def test_colliding_id_holds_each_store_own_content(self) -> None:
        root = _root()
        a, b = _pair(root)
        for store, fx in ((a, FX_A), (b, FX_B)):
            (shared,) = [
                r for r in store.load_profile_records()
                if r.record_id == SHARED_RECORD_ID
            ]
            assert shared.summary == fx["secret"]


# ── Write-side legs stay in their own store ────────────────────────────────────


class TestWriteSideIsolation:
    def test_writeback_lands_in_own_store_only(self) -> None:
        root = _root()
        a, b = _pair(root)
        before = _dir_bytes(b.dir)
        outcome = apply_writeback(
            a,
            WritebackRecord(
                turn_id="t1",
                new_facts=[{"summary": "a brand new alpha-only fact"}],
            ),
            enforcement_passed=True,
        )
        assert outcome.refused == []
        assert any(
            r.summary == "a brand new alpha-only fact"
            for r in a.load_profile_records()
        )
        assert _dir_bytes(b.dir) == before

    def test_forgetting_in_one_store_leaves_the_other_byte_identical(
        self,
    ) -> None:
        root = _root()
        a, b = _pair(root)
        before = _dir_bytes(b.dir)
        report = forget_record(a, FX_A["fact_id"])
        assert report.status == "forgotten"
        assert _dir_bytes(b.dir) == before
        assert b.load_profile_records()  # B's records all still live

    def test_import_is_the_explicit_crossing_and_it_is_stamped(
        self,
    ) -> None:
        root = _root()
        a, b = _pair(root)
        # Nothing crosses by itself…
        assert FX_A["secret"] not in " ".join(
            r.summary for r in b.load_profile_records()
            if r.record_id != SHARED_RECORD_ID
        )
        # …the operator's import crosses, by name, stamped.
        bundle = export_store(a)
        outcome = import_store(b, bundle, mode="merge")
        assert outcome.status == "imported"
        landed = [
            r for r in b.load_profile_records()
            if r.record_id == FX_A["fact_id"]
        ]
        assert len(landed) == 1
        assert landed[0].source == "import"


# ── D-PER-1: the healer heals only its owner's store ──────────────────────────


def _plant_corrupt_line(store: MemoryStore) -> Path:
    path = store.dir / "episodic_store.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write("NOT JSON AT ALL\n")
    return path


class TestHealTarget:
    def test_targeted_healer_leaves_other_stores_alone(self) -> None:
        root = _root()
        a, b = _pair(root)
        path_a = _plant_corrupt_line(a)
        _plant_corrupt_line(b)
        before_a = path_a.read_bytes()
        SelfHealer(memory_dir=b.dir, knowledge_db_path=root / "b.db").heal_all()
        assert path_a.read_bytes() == before_a  # A untouched
        assert "NOT JSON" not in (b.dir / "episodic_store.jsonl").read_text()
        assert list(b.dir.glob("corrupt_backup_*.jsonl"))  # healed at home

    def test_core_construction_heals_its_own_store_not_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _root()
        # A fake "default deployment" persona home, planted with a
        # corrupt line — the pre-fix core rewrote THIS file when
        # any other persona's core was constructed (probe P3).
        default_dir = root / "default_home"
        default_store = build_persona_store(default_dir, FX_A)
        default_path = _plant_corrupt_line(default_store)
        before = default_path.read_bytes()
        monkeypatch.setattr(
            self_heal_mod, "get_memory_dir", lambda: default_dir
        )
        monkeypatch.setattr(
            self_heal_mod, "get_knowledge_db_path", lambda: root / "d.db"
        )
        monkeypatch.setattr(
            self_heal_mod, "get_configs_dir", lambda: root / "cfgs"
        )
        # Persona B's own store also carries a corrupt line.
        b = build_persona_store(root / "b_home", FX_B)
        _plant_corrupt_line(b)
        _core(b.dir, root)
        # The default home is byte-identical; B healed its own.
        assert default_path.read_bytes() == before
        assert list(default_dir.glob("corrupt_backup_*.jsonl")) == []
        assert "NOT JSON" not in (b.dir / "episodic_store.jsonl").read_text()
        assert list(b.dir.glob("corrupt_backup_*.jsonl"))


# ── D-PER-2: identity precedence ───────────────────────────────────────────────


class TestIdentityPrecedence:
    def test_store_core_wins_over_config_assignment(self) -> None:
        root = _root()
        _pair(root)
        cfg = {"cognition": {"personality_core_path": str(
            root / FX_A["key"] / "personality_core.yaml"
        )}}
        core_b = _core(root / FX_B["key"], root, config=cfg)
        assert core_b.persona_expectation is not None
        assert core_b.persona_expectation.record_id == FX_B["core_id"]

    def test_empty_store_takes_the_config_assignment(self) -> None:
        root = _root()
        _pair(root)
        cfg = {"cognition": {"personality_core_path": str(
            root / FX_A["key"] / "personality_core.yaml"
        )}}
        core = _core(root / "empty", root, config=cfg)
        assert core.persona_expectation is not None
        assert core.persona_expectation.record_id == FX_A["core_id"]

    def test_two_stores_two_configs_two_personae(self) -> None:
        root = _root()
        _pair(root)
        cores = {}
        for fx in (FX_A, FX_B):
            cfg = {"cognition": {"personality_core_path": str(
                root / fx["key"] / "personality_core.yaml"
            )}}
            # An empty store per core: identity comes from each
            # core's OWN config assignment, not the other's.
            cores[fx["key"]] = _core(root / f"empty_{fx['key']}", root, config=cfg)
        assert cores[FX_A["key"]].persona_expectation.record_id == FX_A["core_id"]
        assert cores[FX_B["key"]].persona_expectation.record_id == FX_B["core_id"]


# ── Adversarial prompts through real cores ─────────────────────────────────────


def _markers(fx: dict) -> list[str]:
    return [
        fx["secret"], fx["trait"], fx["core_id"],
        fx["thread_topic"], fx["preference_value"],
    ]


class TestAdversarialPrompts:
    def test_prompts_themselves_carry_no_secret(self) -> None:
        for prompt in ADVERSARIAL_PROMPTS:
            for fx in (FX_A, FX_B):
                for marker in _markers(fx):
                    assert marker not in prompt

    @pytest.mark.parametrize("prompt", ADVERSARIAL_PROMPTS)
    def test_knowledge_only_turns_never_surface_the_other(
        self, prompt: str
    ) -> None:
        root = _root()
        _pair(root)
        for fx, other in ((FX_A, FX_B), (FX_B, FX_A)):
            core = _core(root / fx["key"], root)
            result = core.think(prompt)
            turn = core.last_turn_state
            assert turn is not None and turn.scaffold is not None
            surfaces = result.text + "\n" + turn.scaffold.fact_block
            for marker in _markers(other):
                assert marker not in surfaces
            # …and the persona actually in force is its own.
            assert fx["trait"] in turn.scaffold.fact_block

    def test_stub_backend_requests_carry_only_own_persona(self) -> None:
        seen: list[str] = []

        class CaptureBackend(UnifiedBackend):
            def generate(self, request):
                seen.append(repr(request))
                return GenerationResponse(
                    text="The ferry crossing takes eleven minutes.",
                    tokens_generated=7,
                )

            def health_check(self) -> bool:
                return True

            def backend_name(self) -> str:
                return "tf055-capture"

        root = _root()
        _pair(root)
        for fx, other in ((FX_A, FX_B), (FX_B, FX_A)):
            seen.clear()
            core = _core(root / fx["key"], root, backend=CaptureBackend())
            for prompt in ADVERSARIAL_PROMPTS:
                core.think(prompt)
            joined = "\n".join(seen)
            assert joined  # the backend was really exercised
            for marker in _markers(other):
                assert marker not in joined
