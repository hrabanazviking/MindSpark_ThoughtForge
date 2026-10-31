"""
TF-034 — Core API freeze v1: the cognition layer's public
interface versioned + documented.

The freeze lives in ``src/thoughtforge/cognition/api_surface.py``
(``API_VERSION``, the frozen tables, ``api_manifest()``,
``verify_api_surface()``); the write-up is
``docs/architecture/CORE_API_FREEZE_V1.md``.

Pinned here:

  * the version is 1.0.0 and the manifest is plain JSON-safe data
    built from the frozen tables — mutating a returned manifest
    cannot rewrite the freeze,
  * the live surface matches the freeze exactly (the verifier
    returns no problems), and the verifier CATCHES drift: a
    tampered manifest (renamed parameter, missing export,
    missing module name, property demoted) reports each problem
    by name,
  * the package exports are exactly the freeze's list, in
    ``thoughtforge.cognition.__all__`` and importable, each the
    same object as in its home module,
  * the core's constructor / methods / properties match the
    frozen signatures read live via inspect,
  * no frozen name is private (underscore-prefixed),
  * CORE_LOOP §8-F1 is settled by wiring, not removal:
    ``num_drafts`` caps candidate drafts — a stub-engine core
    drafts twice by default and once with num_drafts=1, on the
    plain AND streamed paths; num_drafts=0 raises ValueError;
    a unified-backend core still drafts exactly once under a
    cap (the cap cannot raise its by-construction single draft).

Proof grade: FIXTURE engines/backends only, as throughout
Part II. No real model ran for this slice.
"""

from __future__ import annotations

import copy
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import thoughtforge.cognition as cognition
from thoughtforge.cognition.api_surface import (
    API_VERSION,
    API_VERSION_TUPLE,
    CORE_CONSTRUCTOR_PARAMS,
    CORE_METHODS,
    CORE_PROPERTIES,
    MODULE_SURFACES,
    PACKAGE_EXPORTS,
    api_manifest,
    verify_api_surface,
)
from thoughtforge.cognition.core import ThoughtForgeCore

TEXT = (
    "Yggdrasil is the world tree that joins the nine realms. "
    "Its roots reach into deep wells of wisdom and fate. "
    "Odin hung upon its branches for nine nights seeking runes."
)


def _core(backend=None) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    kwargs: dict = dict(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    if backend is not None:
        kwargs["backend"] = backend
    return ThoughtForgeCore(**kwargs)


class _CountingEngine:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, params=None):
        self.calls += 1
        return SimpleNamespace(
            text=TEXT, tokens_generated=len(TEXT.split())
        )


# ── The version and the manifest ─────────────────────────────────────────────


class TestVersionAndManifest:
    def test_version_is_1_0_0(self) -> None:
        assert API_VERSION == "1.0.0"
        assert API_VERSION_TUPLE == (1, 0, 0)
        assert cognition.API_VERSION == "1.0.0"

    def test_manifest_shape(self) -> None:
        manifest = api_manifest()
        assert manifest["api"] == "thoughtforge.cognition"
        assert manifest["version"] == API_VERSION
        assert manifest["core"]["class"] == (
            "thoughtforge.cognition.core.ThoughtForgeCore"
        )
        assert manifest["core"]["constructor_params"] == list(
            CORE_CONSTRUCTOR_PARAMS
        )
        assert manifest["package_exports"] == list(PACKAGE_EXPORTS)
        assert set(manifest["modules"]) == set(MODULE_SURFACES)

    def test_manifest_is_json_safe(self) -> None:
        assert json.loads(json.dumps(api_manifest())) == api_manifest()

    def test_manifest_is_a_fresh_copy_each_call(self) -> None:
        manifest = api_manifest()
        manifest["core"]["methods"]["think"].append("bogus")
        manifest["package_exports"].clear()
        fresh = api_manifest()
        assert fresh["core"]["methods"]["think"] == list(CORE_METHODS["think"])
        assert fresh["package_exports"] == list(PACKAGE_EXPORTS)

    def test_no_frozen_name_is_private(self) -> None:
        manifest = api_manifest()
        names = (
            list(manifest["package_exports"])
            + list(manifest["core"]["methods"])
            + list(manifest["core"]["properties"])
            + [n for names in manifest["modules"].values() for n in names]
        )
        assert names and all(not n.startswith("_") for n in names)


# ── The verifier: live surface == freeze, drift is caught ────────────────────


class TestVerify:
    def test_live_surface_matches_the_freeze(self) -> None:
        assert verify_api_surface() == []

    def test_renamed_parameter_is_caught(self) -> None:
        manifest = copy.deepcopy(api_manifest())
        manifest["core"]["methods"]["think"] = ["user_text"]
        problems = verify_api_surface(manifest)
        assert len(problems) == 1 and "think" in problems[0]

    def test_constructor_drift_is_caught(self) -> None:
        manifest = copy.deepcopy(api_manifest())
        manifest["core"]["constructor_params"] = ["config"]
        problems = verify_api_surface(manifest)
        assert len(problems) == 1 and "__init__" in problems[0]

    def test_missing_package_export_is_caught(self) -> None:
        manifest = copy.deepcopy(api_manifest())
        manifest["package_exports"].append("NoSuchExport")
        problems = verify_api_surface(manifest)
        assert any("NoSuchExport" in p for p in problems)

    def test_unlisted_all_entry_is_caught(self) -> None:
        manifest = copy.deepcopy(api_manifest())
        manifest["package_exports"].remove("StreamEvent")
        problems = verify_api_surface(manifest)
        assert any("StreamEvent" in p for p in problems)

    def test_missing_module_name_is_caught(self) -> None:
        manifest = copy.deepcopy(api_manifest())
        manifest["modules"]["thoughtforge.cognition.will"].append("NoSuchName")
        problems = verify_api_surface(manifest)
        assert any("NoSuchName" in p for p in problems)

    def test_property_demoted_is_caught(self) -> None:
        manifest = copy.deepcopy(api_manifest())
        manifest["core"]["properties"].append("think")
        problems = verify_api_surface(manifest)
        assert any("think" in p and "property" in p for p in problems)


# ── Package exports and live signatures ──────────────────────────────────────


class TestPackageAndSignatures:
    def test_package_all_is_exactly_the_freeze(self) -> None:
        assert set(cognition.__all__) == set(PACKAGE_EXPORTS)
        assert len(cognition.__all__) == len(PACKAGE_EXPORTS)

    def test_every_export_importable_from_the_package(self) -> None:
        for name in PACKAGE_EXPORTS:
            assert hasattr(cognition, name), name

    def test_constructor_signature_matches_freeze(self) -> None:
        params = list(inspect.signature(ThoughtForgeCore.__init__).parameters)
        assert params[1:] == list(CORE_CONSTRUCTOR_PARAMS)

    def test_method_signatures_match_freeze(self) -> None:
        for name, frozen in CORE_METHODS.items():
            member = getattr(ThoughtForgeCore, name)
            assert callable(member), name
            params = list(inspect.signature(member).parameters)
            assert (params[1:] if params[:1] == ["self"] else params) == list(
                frozen
            ), name

    def test_properties_are_properties(self) -> None:
        for name in CORE_PROPERTIES:
            assert isinstance(
                getattr(ThoughtForgeCore, name), property
            ), name

    def test_record_types_exist(self) -> None:
        from thoughtforge.knowledge.models import (
            FinalResponseRecord,
            RuntimeTurnState,
        )
        from thoughtforge.cognition.streaming import StreamEvent

        manifest = api_manifest()
        assert manifest["core"]["record_types"] == {
            "think_return": "thoughtforge.knowledge.models.FinalResponseRecord",
            "resume_turn_return": (
                "thoughtforge.knowledge.models.FinalResponseRecord"
            ),
            "last_turn_state_type": (
                "thoughtforge.knowledge.models.RuntimeTurnState"
            ),
            "think_stream_event": "thoughtforge.cognition.streaming.StreamEvent",
        }
        assert FinalResponseRecord and RuntimeTurnState and StreamEvent


# ── CORE_LOOP §8-F1: num_drafts wired ────────────────────────────────────────


class TestNumDraftsWired:
    def test_default_drafts_once_per_mode(self) -> None:
        core = _core()
        engine = _CountingEngine()
        core._engine = engine
        core.think("What is Yggdrasil?")
        assert engine.calls == 2
        assert core.last_turn_state is not None
        assert len(core.last_turn_state.candidates) == 2

    def test_cap_one_drafts_once(self) -> None:
        core = _core()
        engine = _CountingEngine()
        core._engine = engine
        core.think("What is Yggdrasil?", num_drafts=1)
        assert engine.calls == 1
        assert core.last_turn_state is not None
        assert len(core.last_turn_state.candidates) == 1

    def test_cap_above_construction_is_harmless(self) -> None:
        # The phase-6 adversarial shape: a large cap must not
        # crash and cannot raise the by-construction maximum.
        core = _core()
        engine = _CountingEngine()
        core._engine = engine
        core.think("What is Yggdrasil?", num_drafts=5)
        assert engine.calls == 2

    def test_cap_applies_on_the_streamed_path(self) -> None:
        core = _core()
        engine = _CountingEngine()
        core._engine = engine
        events = list(core.think_stream("What is Yggdrasil?", num_drafts=1))
        starts = [e for e in events if e.kind == "candidate_start"]
        assert len(starts) == 1
        assert events[-1].kind == "final"

    def test_zero_drafts_raises(self) -> None:
        core = _core()
        with pytest.raises(ValueError, match="num_drafts"):
            core.think("What is Yggdrasil?", num_drafts=0)

    def test_negative_drafts_raises(self) -> None:
        core = _core()
        with pytest.raises(ValueError, match="num_drafts"):
            core.think("What is Yggdrasil?", num_drafts=-2)

    def test_unified_backend_drafts_once_with_or_without_cap(self) -> None:
        from thoughtforge.inference.unified_backend import (
            GenerationResponse,
            UnifiedBackend,
        )

        class _Backend(UnifiedBackend):
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, request):
                self.calls += 1
                return GenerationResponse(
                    text=TEXT, tokens_generated=len(TEXT.split())
                )

            def health_check(self) -> bool:
                return True

            def backend_name(self) -> str:
                return "stub-api-freeze"

        backend = _Backend()
        core = _core(backend=backend)
        core.think("What is Yggdrasil?", num_drafts=1)
        assert backend.calls == 1
        core.think("What is Yggdrasil?")
        assert backend.calls == 2
