"""
TF-106 — Source registry: every source carries provenance
+ licence (TF-008's audit consumed); unlicensed-in-fact
sources quarantined by rule.

The registry lives as verifiable data in
``src/thoughtforge/etl/source_registry.py`` (``SOURCES`` /
``REGISTRY_RULES``, the gate ``check_source_use()``, the
proposal judge ``classify_candidate()``, checked by
``registry_problems()`` and ``registry_live_problems()``),
the write-up is ``docs/architecture/SOURCE_REGISTRY.md``,
the TF-106 shares of D-LIC-2 and D-DATA-3 are FIXED here,
and D-ETL-2 is filed (the builders do not yet consult the
registry at build time).

Pinned here:

  * every TF-105 builder has exactly one primary registry
    entry, every entry carries provenance + a named
    upstream-terms source, and the static checker is clean,
  * the four external sources are licensed in fact, with
    the licences their upstream terms state (Wikidata CC0,
    ConceptNet CC BY-SA 4.0, GeoNames CC BY 4.0, DBpedia
    dual CC BY-SA 3.0 + GFDL) and their obligations filed,
  * the reference library is the one quarantined source:
    licence None by construction, private ingestion allowed
    by the gate, redistribution refused by name,
  * the SRD rides as a sub-source on its own verified
    CC BY 4.0 (SRD 5.1) without lifting its parent's
    quarantine,
  * an unregistered source and an unknown use are refused
    by name, and classify_candidate quarantines proposals
    missing provenance or a licence, naming what is missing.

All fixtures are synthetic (TF-017): candidate proposals
are invented names, never real sources or canon.

Proof grade: FIXTURE (registry + gate exercised on this
VM; upstream terms read from the publishers' own pages
2026-11-01). No source was fetched or ingested at scale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.etl.etl_map import SOURCE_BUILDERS
from thoughtforge.etl.source_registry import (
    REGISTRY_DOC,
    REGISTRY_FORMAT,
    REGISTRY_RULES,
    SOURCES,
    STATUS_LICENSED,
    STATUS_OBLIGATIONS,
    STATUS_QUARANTINED,
    QuarantinedSourceError,
    check_source_use,
    classify_candidate,
    get_source,
    registry_as_data,
    registry_live_problems,
    registry_problems,
)


class TestRegistryData:
    def test_format_and_doc(self) -> None:
        assert REGISTRY_FORMAT == "thoughtforge-source-registry/1"
        assert REGISTRY_DOC == "docs/architecture/SOURCE_REGISTRY.md"
        assert Path(REGISTRY_DOC).exists()

    def test_every_builder_one_primary_entry(self) -> None:
        primary = [e for e in SOURCES if e["sub_path"] is None]
        assert sorted(e["builder_id"] for e in primary) == \
            sorted(b["id"] for b in SOURCE_BUILDERS)

    def test_entries_complete(self) -> None:
        for entry in SOURCES:
            assert entry["name"] and entry["provenance"]
            assert entry["upstream_terms"] and entry["notes"]
            assert entry["status"] in (
                STATUS_LICENSED, STATUS_OBLIGATIONS, STATUS_QUARANTINED)

    def test_rules_unique(self) -> None:
        ids = [r["id"] for r in REGISTRY_RULES]
        assert len(ids) == len(set(ids)) == 8

    def test_as_data_shape(self) -> None:
        data = registry_as_data()
        assert data["format"] == REGISTRY_FORMAT
        assert len(data["sources"]) == len(SOURCES) == 6
        assert data["uses"] == ["ingest_private", "redistribute"]


class TestLicensedSources:
    def test_wikidata_cc0_no_obligations(self) -> None:
        entry = get_source("wikidata")
        assert entry["status"] == STATUS_LICENSED
        assert entry["licence"].startswith("CC0 1.0")
        assert entry["obligations"] == ()

    def test_conceptnet_by_sa(self) -> None:
        entry = get_source("conceptnet")
        assert entry["status"] == STATUS_OBLIGATIONS
        assert "CC BY-SA 4.0" in entry["licence"]
        assert entry["obligations"] == ("attribution", "share_alike")

    def test_geonames_by(self) -> None:
        entry = get_source("geonames")
        assert entry["licence"] == "CC BY 4.0"
        assert entry["obligations"] == ("attribution",)

    def test_dbpedia_dual(self) -> None:
        entry = get_source("dbpedia")
        assert "CC BY-SA 3.0" in entry["licence"]
        assert "Free Documentation License" in entry["licence"]
        assert entry["obligations"] == ("attribution", "share_alike")

    def test_licensed_pass_both_uses(self) -> None:
        for source_id in ("wikidata", "conceptnet", "geonames",
                          "dbpedia", "dnd_5e_srd"):
            assert check_source_use(
                source_id, "ingest_private")["id"] == source_id
            assert check_source_use(
                source_id, "redistribute")["id"] == source_id


class TestQuarantine:
    def test_reference_library_quarantined_no_licence(self) -> None:
        entry = get_source("reference_library")
        assert entry["status"] == STATUS_QUARANTINED
        assert entry["licence"] is None
        assert "no_redistribution" in entry["obligations"]

    def test_quarantined_private_ingest_allowed(self) -> None:
        assert check_source_use(
            "reference_library", "ingest_private")["id"] == \
            "reference_library"

    def test_quarantined_redistribution_refused(self) -> None:
        with pytest.raises(QuarantinedSourceError):
            check_source_use("reference_library", "redistribute")

    def test_exactly_one_quarantined_entry(self) -> None:
        assert [e["id"] for e in SOURCES
                if e["status"] == STATUS_QUARANTINED] == \
            ["reference_library"]

    def test_srd_sub_source_own_licence(self) -> None:
        entry = get_source("dnd_5e_srd")
        assert entry["builder_id"] == "reference"
        assert entry["sub_path"] == "dnd-5e-srd-markdown/"
        assert "CC BY 4.0" in entry["licence"]
        # The sub-source's licence does not lift the parent.
        assert get_source("reference_library")["status"] == \
            STATUS_QUARANTINED


class TestGateRefusals:
    def test_unregistered_source_raises_by_name(self) -> None:
        with pytest.raises(ValueError, match="unregistered source"):
            check_source_use("no_such_source_tf106", "ingest_private")
        with pytest.raises(ValueError, match="unregistered source"):
            get_source("no_such_source_tf106")

    def test_unknown_use_raises_by_name(self) -> None:
        with pytest.raises(ValueError, match="unknown source use"):
            check_source_use("wikidata", "publish_everywhere")


class TestClassifyCandidate:
    def test_bare_candidate_quarantined(self) -> None:
        verdict = classify_candidate("bare_source", None, None)
        assert verdict["status"] == STATUS_QUARANTINED
        assert verdict["missing"] == ("provenance", "licence")

    def test_licenceless_candidate_quarantined(self) -> None:
        verdict = classify_candidate(
            "half_source", "a named publisher's scrape", "")
        assert verdict["status"] == STATUS_QUARANTINED
        assert verdict["missing"] == ("licence",)

    def test_provenanceless_candidate_quarantined(self) -> None:
        verdict = classify_candidate("half_source", "  ", "CC0 1.0")
        assert verdict["status"] == STATUS_QUARANTINED
        assert verdict["missing"] == ("provenance",)

    def test_complete_candidate_admissible_for_filing(self) -> None:
        verdict = classify_candidate(
            "full_source", "a named publisher", "CC BY 4.0")
        assert verdict["status"] == "admissible_for_filing"
        assert verdict["missing"] == ()


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert registry_problems() == []

    def test_live_checker_clean(self) -> None:
        assert registry_live_problems() == []
