"""
TF-094 — Policy configuration: enforcement
strictness per conversation (strict / standard /
relaxed) with the defaults stated + tested.

The estate lives in
``src/thoughtforge/refinement/policy_config.py``
(``CONFIG_RULES`` / ``DISPOSITIONS`` /
``CONFIG_FIXTURES`` as data,
``resolve_policy_config`` / ``enforce_configured``,
the checkers ``policy_config_problems`` /
``policy_config_live_problems``, format
``thoughtforge-policy-configuration/1``).
Write-up: ``docs/architecture/POLICY_CONFIGURATION.md``.

Measured start: TF-093's surface had exactly one
disposition — a flagged answer always left
annotated, a refusal always refused — and no
level, default, or per-conversation strictness
existed anywhere in the refinement estate.

Pinned here:

  * the filed estate: both checkers clean, the
    levels exactly strict / standard / relaxed,
    the default stated as standard, proof grade
    stated, the doc present,
  * standard is TF-093 exactly, leg by leg,
  * strict blocks every flag: no text, flags
    preserved and carried as problems, check
    outcomes unchanged,
  * relaxed downgrades flags to verbatim
    warnings: the answer leaves as judged
    underneath them, nothing hidden,
  * a refusal is never relaxed, and a rescue or
    clean verdict is identical at every level,
  * the level is caller-supplied per
    conversation: unknown names raise by name,
    a config / strictness disagreement raises,
    nothing is guessed,
  * TF-093's surface, EnforcementGate and
    FragmentSalvage are unchanged, and
    validate_config is ruled not wired
    (D-DEAD-1, TF-094's share).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thoughtforge.refinement.policy_config import (
    CONFIG_DOC,
    CONFIG_FIXTURES,
    CONFIG_FORMAT,
    CONFIG_RULES,
    DEFAULT_STRICTNESS,
    DISPOSITIONS,
    STRICTNESS_LEVELS,
    PolicyConfig,
    enforce_configured,
    policy_config_as_data,
    policy_config_live_problems,
    policy_config_problems,
    resolve_policy_config,
)
from thoughtforge.refinement.policy_surface import enforce_policy
from thoughtforge.refinement.salvage_map import run_fixture
from thoughtforge.utils.validators import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]

_FIXTURE = {f["id"]: f for f in CONFIG_FIXTURES}


def _records():
    from thoughtforge.knowledge.models import ActivatedRecord

    return [
        ActivatedRecord(
            "Q123", "entity", 0.9, "Odin chief god of Norse mythology",
            raw={"qid": "Q123", "label_en": "Odin",
                 "description_en": "chief god of Norse mythology",
                 "source": "wikidata"}),
        ActivatedRecord(
            "chunk_7", "reference_chunk", 0.8,
            "lantern oil kept in the hollow stores",
            raw={"chunk_id": "chunk_7", "source_file": "hollow.md",
                 "title": "The Hollow",
                 "content": "lantern oil is kept in the hollow stores",
                 "source": "reference"}),
    ]


def _signals(fixture_id: str) -> dict:
    signals = dict(_FIXTURE[fixture_id]["signals"])
    if signals.get("records") == "lantern":
        signals["records"] = _records()
    return signals


def _run(fixture_id: str, level: str):
    return enforce_configured(
        _FIXTURE[fixture_id]["text"], strictness=level,
        **_signals(fixture_id))


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert policy_config_problems() == []
        assert policy_config_live_problems() == []

    def test_format_doc_and_proof(self):
        data = policy_config_as_data()
        assert data["format"] == CONFIG_FORMAT
        assert data["config_doc"] == CONFIG_DOC
        assert (REPO_ROOT / CONFIG_DOC).exists()
        assert data["proof"]["live_model_behaviour"] == "not_proven"
        assert data["proof"]["policy_surface"] == "unchanged"
        json.dumps(data)

    def test_levels_and_default_stated(self):
        assert STRICTNESS_LEVELS == ("strict", "standard", "relaxed")
        assert DEFAULT_STRICTNESS == "standard"
        assert [d["level"] for d in DISPOSITIONS] == \
            list(STRICTNESS_LEVELS)

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in CONFIG_RULES]
        assert len(ids) == len(set(ids)) == 11

    def test_fixture_counts(self):
        assert len(CONFIG_FIXTURES) == 12
        bases = [f["base_verdict"] for f in CONFIG_FIXTURES]
        assert bases.count("clean") == 1
        assert bases.count("rescued") == 2
        assert bases.count("flagged") == 5
        assert bases.count("refused") == 4

    def test_no_level_relaxes_a_refusal_in_the_table(self):
        for entry in DISPOSITIONS:
            assert entry["on_refused"] == "refused"


# ── Resolution: default, canonicalisation, refusal by name ────


class TestResolution:
    def test_default_is_standard(self):
        config = resolve_policy_config()
        assert config == PolicyConfig("standard")
        assert config.is_default

    def test_names_canonicalised(self):
        assert resolve_policy_config(" Strict ").strictness == "strict"
        assert resolve_policy_config("RELAXED").strictness == "relaxed"

    def test_unknown_level_raises_by_name(self):
        with pytest.raises(ValueError, match="severe"):
            resolve_policy_config("severe")
        with pytest.raises(ValueError, match="severe"):
            enforce_configured("Any honest answer text here.",
                               strictness="severe")

    def test_config_and_strictness_disagreement_raises(self):
        with pytest.raises(ValueError, match="disagrees"):
            enforce_configured(
                "Any honest answer text here.", strictness="strict",
                config=PolicyConfig("relaxed"))

    def test_config_object_accepted(self):
        result = enforce_configured(
            _FIXTURE["clean_prose_all_levels"]["text"],
            config=PolicyConfig("strict"))
        assert result.strictness == "strict"
        assert result.verdict == "clean"


# ── Dispositions ───────────────────────────────────────────────


class TestDispositions:
    def test_standard_is_tf093_exactly(self):
        for fixture_id, fixture in _FIXTURE.items():
            base = enforce_policy(fixture["text"], **_signals(fixture_id))
            result = _run(fixture_id, "standard")
            assert result.verdict == base.verdict
            assert result.text == base.text
            assert result.flags == base.flags
            assert result.problems == base.problems
            assert result.warnings == ()
            assert result.disposition == "as_judged"
            assert [o.status for o in result.checks] == \
                [o.status for o in base.checks]

    def test_no_level_named_means_standard(self):
        fixture = _FIXTURE["citation_flagged_blocked_or_warned"]
        result = enforce_configured(
            fixture["text"], **_signals(
                "citation_flagged_blocked_or_warned"))
        assert result.strictness == "standard"
        assert result.verdict == "flagged"

    @pytest.mark.parametrize("fixture_id", [
        "drift_flagged_blocked_or_warned",
        "citation_flagged_blocked_or_warned",
        "legacy_floor_flagged_blocked_or_warned",
        "too_short_flagged_blocked_or_warned",
        "generic_flagged_blocked_or_warned",
    ])
    def test_strict_blocks_every_flag(self, fixture_id):
        fixture = _FIXTURE[fixture_id]
        result = _run(fixture_id, "strict")
        assert result.verdict == "refused"
        assert result.base_verdict == "flagged"
        assert result.disposition == "blocked_flagged"
        assert result.text is None
        assert result.flags == tuple(fixture["flags"])
        assert result.problems == tuple(fixture["flags"])
        # The checks stand as judged — a blocked flag is
        # still recorded flagged, never rewritten.
        base = enforce_policy(fixture["text"], **_signals(fixture_id))
        assert [o.status for o in result.checks] == \
            [o.status for o in base.checks]

    @pytest.mark.parametrize("fixture_id", [
        "drift_flagged_blocked_or_warned",
        "citation_flagged_blocked_or_warned",
        "legacy_floor_flagged_blocked_or_warned",
        "too_short_flagged_blocked_or_warned",
        "generic_flagged_blocked_or_warned",
    ])
    def test_relaxed_downgrades_flags_to_warnings(self, fixture_id):
        fixture = _FIXTURE[fixture_id]
        result = _run(fixture_id, "relaxed")
        assert result.verdict == "clean"
        assert result.base_verdict == "flagged"
        assert result.disposition == "downgraded_to_warnings"
        assert result.text == fixture["expected_text"]
        assert result.flags == ()
        assert result.warnings == tuple(fixture["flags"])

    @pytest.mark.parametrize("fixture_id", [
        "format_refused_never_relaxed",
        "truncation_refused_never_relaxed",
        "empty_refused_never_relaxed",
        "unsupported_language_refused_never_relaxed",
    ])
    def test_refusal_is_never_relaxed(self, fixture_id):
        fixture = _FIXTURE[fixture_id]
        for level in STRICTNESS_LEVELS:
            result = _run(fixture_id, level)
            assert result.verdict == "refused"
            assert result.text is None
            assert result.problems == tuple(fixture["problems"])
            assert result.warnings == ()

    @pytest.mark.parametrize("fixture_id", [
        "clean_prose_all_levels",
        "loop_rescued_all_levels",
        "format_repaired_all_levels",
    ])
    def test_clean_and_rescued_identical_at_every_level(self, fixture_id):
        fixture = _FIXTURE[fixture_id]
        for level in STRICTNESS_LEVELS:
            result = _run(fixture_id, level)
            assert result.verdict == fixture["base_verdict"]
            assert result.text == fixture["expected_text"]
            assert result.disposition == "as_judged"


# ── What this slice does not touch ─────────────────────────────


class TestUntouched:
    def test_salvage_itself_is_unchanged(self):
        result = run_fixture("loop_verbatim")
        assert result.salvage_path == "best_draft"

    def test_validate_config_still_validates_backends(self):
        # D-DEAD-1 (TF-094 share): the function is ruled
        # not wired here — and it is itself unmoved: it
        # still judges a backend dict, the shape that
        # makes it the wrong tool for a strictness level.
        assert validate_config({}) == ["Missing required key: 'backend'"]
        assert validate_config({"backend": "ollama",
                                "ollama_url": "http://localhost"}) == []
