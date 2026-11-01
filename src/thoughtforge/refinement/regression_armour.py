"""TF-103 — Regression armour: every historical
bad output named in the repo's docs becomes a
fixture before its fix is trusted.

Roadmap: "Regression armour: every historical
bad output named in the repo's docs becomes a
fixture before its fix is trusted."

Measured start (censused this tick from
``docs/architecture/*.md`` and
``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`` —
the campaign's own docs name the bad outputs,
slice by slice, in their measured-start
passages):

- **The bad outputs are named, but nowhere
  registered together.** Each fixing slice
  filed its own fixtures in its own module
  (TF-088's nine adversarial fabrications,
  TF-090's format set, TF-091's loops, TF-092's
  drift, TF-093's policy set, TF-076's parser
  legs, TF-020's router guard, TF-050's
  contradiction pairs, TF-031's stated-failure
  pins, TF-028's contaminated-prompt pin,
  TF-065's contract legs, TF-100's chaos
  corpus). Nothing in the tree answers the
  question "which historical bad outputs exist,
  where is each one fixtured, and is its fix
  re-proven today?" — a fixture set in an
  unvisited module is armour nobody checks
  for rust.
- **Three named bad outputs have fixtures but
  no trusted fix.** TF-101's scored run filed
  9 mismatches in 3 named gaps (draft-side
  hallucination, persona voice at the policy
  surface, genericness beyond hard phrases).
  The chaos corpus fixtures exist; the fixes
  belong to their owners and have NOT landed.
  Armour must register these as gaps, never
  dress them as covered.
- **Two named surfaces have no historical bad
  output to fixture at all.** D-AESIR-1 and
  D-OAI-4 record that no live Aesir server and
  no real third-party OpenAI-shaped server has
  ever faced the new backends — there is no
  output, good or bad, from those legs. They
  are registered NOT PROVEN / not fixtureable,
  plainly, not silently omitted.

What this slice adds: **one registry**
(``ARMOUR_CASES`` as data) of every historical
bad output the docs name, each case carrying
its source doc, its naming slice, the bad
output's shape, its fix owner, and its armour
kind:

- ``rerun`` — a fixture set already exists in
  the fixing slice's module; this slice's live
  checker re-runs that module's own live
  checker today and requires it green. The
  fix is trusted only while the re-run passes.
- ``direct`` — this module files a NEW fixture
  (``ARMOUR_FIXTURES``) for a doc-named bad
  output whose fix surface is cheap to judge
  directly (the router's four dead-pattern
  misroutes, TF-050's silent overwrite,
  TF-076's double-/v1 + parser outcome legs,
  TF-091's 20x loop at text level), and the
  live checker runs it against the real fix.
- ``reference`` — the fixture lives in the
  fixing slice's test file; the data checker
  verifies from disk that the named test file
  carries the named pin (needle), so the
  registry cannot claim a fixture that is
  not there.
- ``gap_open`` — a fixture exists (TF-100's
  corpus) but the fix has NOT landed; the
  live checker pins the gap's exact shape
  (TF-101's 9 mismatches, in exactly the 3
  named families) so a future fix — or a
  future regression — changes this registry
  in the open.
- ``not_proven`` — no live output exists to
  fixture (D-AESIR-1, D-OAI-4); registered
  with the reason, never counted as covered.

No gate, surface, or fixing module is changed
by this slice; no expectation in any existing
fixture set is edited. TF-016: no docket entry
is owned by TF-103 and none is added — the
totals stand unchanged at 90.

Proof grade FIXTURE — registry judged from
disk (source docs and test needles read live),
rerun legs recomputed through the real live
checkers, direct fixtures run against the real
fix surfaces, and the gap shape recomputed by
TF-101's real runner, all on this VM. A live
model's regression behaviour is NOT PROVEN
(no model in tree), never claimed.

The human-readable write-up is
``docs/architecture/REGRESSION_ARMOUR.md``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

#: This armour's estate format.
ARMOUR_FORMAT = "thoughtforge-regression-armour/1"

ARMOUR_DOC = "docs/architecture/REGRESSION_ARMOUR.md"

#: The armour kinds, in registry order of trust.
ARMOUR_KINDS: tuple[str, ...] = (
    "rerun", "direct", "reference", "gap_open", "not_proven")

#: Kinds whose fix is re-proven by this slice.
COVERED_KINDS: tuple[str, ...] = ("rerun", "direct", "reference")

#: Strings that would mean canon leaked into a
#: case (TF-017's guard vocabulary).
FORBIDDEN_MARKERS: tuple[str, ...] = ("Sigrid", "data/")

# ── The rules ───────────────────────────────────────────────────────────────

ARMOUR_RULES: tuple[dict[str, Any], ...] = (
    {"id": "every_named_bad_output_is_registered",
     "rule": "every historical bad output named in "
             "docs/architecture or the TF-016 docket "
             "carries exactly one registry case with "
             "its source doc and naming slice; a bad "
             "output with no case is a registry gap, "
             "and a case with no doc is an invention — "
             "both fail the data checker"},
    {"id": "a_fix_is_trusted_only_while_its_fixture_passes",
     "rule": "a covered case names the fixture that "
             "pins the bad output and the live leg that "
             "re-runs it; trust is the re-run passing "
             "today, not the fixing slice's banking "
             "line — a red re-run names the case"},
    {"id": "rerun_never_reimplements",
     "rule": "rerun cases call the fixing module's own "
             "live checker unchanged; this slice adds "
             "no copy of any gate, parser, or guard, so "
             "the armour cannot drift from the fix it "
             "claims to prove"},
    {"id": "gaps_are_armour_too",
     "rule": "a bad output whose fixture exists but "
             "whose fix has not landed is registered "
             "gap_open with its owner, and the live "
             "checker pins the gap's exact current shape "
             "— the gap is guarded in the open, never "
             "counted as covered and never hidden"},
    {"id": "no_output_no_fixture_no_claim",
     "rule": "a surface no live output has ever come "
             "from is registered not_proven with its "
             "reason; it is never fixtured synthetically "
             "and dressed as a historical case"},
    {"id": "references_are_verified_from_disk",
     "rule": "a reference case's test file and pin "
             "needle are read from disk by the data "
             "checker on every run; a missing file or "
             "needle fails the checker, so the registry "
             "cannot outlive the fixture it names"},
)

# ── The registry ────────────────────────────────────────────────────────────
#
# One case per historical bad output (or named
# family of them) in the docs. ``armour`` names
# the proving leg per kind: rerun → ``checker``
# (dotted path to the fixing module's live
# checker); direct → ``fixture`` (an id in
# ARMOUR_FIXTURES below); reference →
# ``test_file`` + ``needle``; gap_open →
# ``gap`` (a key in GAP_FAMILIES); not_proven →
# ``reason``.

ARMOUR_CASES: tuple[dict[str, Any], ...] = (
    # ── rerun: Part V fixing slices' own fixtures ──
    {"id": "hist_salvage_fabrications",
     "name": "refine pass invents content (price of "
             "50 silver coins, citation Q999, negation "
             "flip, Bergen/Reykjavik, dawn->midnight) "
             "and salvage adopts it on score",
     "source_doc": "docs/architecture/SALVAGE_SAFETY.md",
     "source_slice": "TF-088", "fix_owner": "TF-088",
     "docket": "D-SALV-2",
     "kind": "rerun",
     "checker": "thoughtforge.refinement.salvage_safety"
                ":salvage_safety_live_problems"},
    {"id": "hist_salvage_map_gaps",
     "name": "salvage gaps at the map: empty "
             "candidates, truncated draft, rambling, "
             "fragment filter, broken JSON returned "
             "as-is by the pre-Part-V salvage",
     "source_doc": "docs/architecture/SALVAGE_MAPPED.md",
     "source_slice": "TF-087", "fix_owner": "TF-087",
     "docket": None,
     "kind": "rerun",
     "checker": "thoughtforge.refinement.salvage_map"
                ":salvage_map_live_problems"},
    {"id": "hist_loop_verbatim",
     "name": "a 20x sentence loop returned "
             "byte-identical (dedup touched fragments "
             "only, never the answer text)",
     "source_doc": "docs/architecture/SALVAGE_MAPPED.md",
     "source_slice": "TF-091", "fix_owner": "TF-091",
     "docket": None,
     "kind": "rerun",
     "checker": "thoughtforge.refinement.loop_rescue"
                ":loop_rescue_live_problems"},
    {"id": "hist_format_truncation",
     "name": "structured answers repaired by "
             "inventing values / truncated text "
             "passed or misrepaired at the format "
             "and truncation gates",
     "source_doc": "docs/architecture/FORMAT_ENFORCEMENT.md",
     "source_slice": "TF-090", "fix_owner": "TF-090",
     "docket": None,
     "kind": "rerun",
     "checker": "thoughtforge.refinement.format_enforcement"
                ":format_enforcement_live_problems"},
    {"id": "hist_drift",
     "name": "answers wandering off-language "
             "(minority / majority / whole drift) "
             "delivered unrescued and unflagged",
     "source_doc": "docs/architecture/DRIFT_RESCUE.md",
     "source_slice": "TF-092", "fix_owner": "TF-092",
     "docket": None,
     "kind": "rerun",
     "checker": "thoughtforge.refinement.drift_rescue"
                ":drift_rescue_live_problems"},
    {"id": "hist_policy_surface",
     "name": "uncited claims and foreign citation "
             "markers passing the pre-TF-093 "
             "scattered enforcement ungated",
     "source_doc": "docs/architecture/ENFORCEMENT_POLICY_SURFACE.md",
     "source_slice": "TF-093", "fix_owner": "TF-093",
     "docket": None,
     "kind": "rerun",
     "checker": "thoughtforge.refinement.policy_surface"
                ":policy_surface_live_problems"},
    {"id": "hist_persona_voice_sanding",
     "name": "salvage sanding a persona's voice "
             "flat (register / signature / avoid "
             "rules broken by the rescue)",
     "source_doc": "docs/architecture/PERSONA_VOICE_PRESERVATION.md",
     "source_slice": "TF-096", "fix_owner": "TF-096",
     "docket": "D-SALV-3",
     "kind": "rerun",
     "checker": "thoughtforge.refinement.persona_voice"
                ":persona_voice_live_problems"},
    {"id": "hist_rescue_provenance_blindspot",
     "name": "a fabrication-rejected refine pass "
             "counted as the adoption (path said "
             "refine_pass_2; the adoption was pass 1) "
             "— the rescue record misnamed its source",
     "source_doc": "docs/architecture/RESCUE_PROVENANCE.md",
     "source_slice": "TF-097", "fix_owner": "TF-097",
     "docket": "D-TRACE-1",
     "kind": "rerun",
     "checker": "thoughtforge.refinement.rescue_provenance"
                ":rescue_provenance_live_problems"},
    {"id": "hist_refusal_silence",
     "name": "a blocked answer told the user "
             "nothing — no that, no why, no path",
     "source_doc": "docs/architecture/REFUSAL_HONESTY.md",
     "source_slice": "TF-095", "fix_owner": "TF-095",
     "docket": None,
     "kind": "rerun",
     "checker": "thoughtforge.refinement.refusal_honesty"
                ":refusal_honesty_live_problems"},
    {"id": "hist_quality_marks_scatter",
     "name": "quality evidence scattered or "
             "invented: a no-claim pass dressed as "
             "cited, flags leaving no visible trace",
     "source_doc": "docs/architecture/QUALITY_MARKS.md",
     "source_slice": "TF-102", "fix_owner": "TF-102",
     "docket": None,
     "kind": "rerun",
     "checker": "thoughtforge.refinement.quality_marks"
                ":quality_marks_live_problems"},
    {"id": "hist_oai_parser_legs",
     "name": "OpenAI-shaped server answers mangled "
             "at the LM Studio leg: finish_reason "
             "passthrough, content:null as text=None, "
             "empty choices as IndexError, silent "
             "zero token counts, HTML error pages",
     "source_doc": "docs/architecture/OPENAI_COMPATIBLE_BACKEND.md",
     "source_slice": "TF-076", "fix_owner": "TF-076",
     "docket": "D-OAI-1",
     "kind": "rerun",
     "checker": "thoughtforge.inference.openai_compatible"
                ":openai_live_problems"},
    {"id": "hist_chaos_corpus_legs",
     "name": "the chaos corpus's cheap legs "
             "(truncation, loop, format, drift, "
             "citation, refusal/empty, clean controls) "
             "decided other than as filed",
     "source_doc": "docs/architecture/CHAOS_CORPUS.md",
     "source_slice": "TF-100", "fix_owner": "TF-100",
     "docket": None,
     "kind": "rerun",
     "checker": "thoughtforge.refinement.chaos_corpus"
                ":chaos_corpus_live_problems"},
    # ── direct: new fixtures filed by this slice ──
    {"id": "hist_router_dead_patterns",
     "name": "router misroutes from dead "
             "upper-case patterns: 'the api' read as "
             "factual, 'how do i plan' missing its "
             "planning signal, 'i don't know' missing "
             "emotional, 'i've been thinking' missing "
             "curious tone",
     "source_doc": "docs/architecture/ROUTER_DECISION_TABLE.md",
     "source_slice": "TF-020", "fix_owner": "TF-020",
     "docket": None,
     "kind": "direct", "fixture": "router_dead_patterns"},
    {"id": "hist_contradiction_silent_overwrite",
     "name": "a new fact silently overwriting the "
             "stored fact it contradicts (ferry at "
             "midnight over ferry at dawn), no "
             "conflict surfaced",
     "source_doc": "docs/architecture/CONTRADICTIONS.md",
     "source_slice": "TF-050", "fix_owner": "TF-050",
     "docket": None,
     "kind": "direct", "fixture": "contradiction_overwrite"},
    {"id": "hist_oai_double_v1",
     "name": "a base URL ending /v1 producing "
             "/v1/v1/chat/completions against the "
             "server",
     "source_doc": "docs/architecture/OPENAI_COMPATIBLE_BACKEND.md",
     "source_slice": "TF-076", "fix_owner": "TF-076",
     "docket": "D-OAI-2",
     "kind": "direct", "fixture": "oai_double_v1"},
    {"id": "hist_oai_outcome_legs",
     "name": "parser outcome legs at text level: "
             "content_filter dressed as a completion, "
             "null content passed on as a None text, "
             "empty choices crashing the parse",
     "source_doc": "docs/architecture/OPENAI_COMPATIBLE_BACKEND.md",
     "source_slice": "TF-076", "fix_owner": "TF-076",
     "docket": "D-OAI-1",
     "kind": "direct", "fixture": "oai_outcome_legs"},
    {"id": "hist_loop_20x_text_level",
     "name": "the 20x loop at text level, judged "
             "directly (not through TF-091's fixture "
             "set): the cut must leave one telling",
     "source_doc": "docs/architecture/LOOP_RESCUE.md",
     "source_slice": "TF-091", "fix_owner": "TF-091",
     "docket": None,
     "kind": "direct", "fixture": "loop_20x"},
    # ── reference: fixtures in the fixing tests ──
    {"id": "hist_silent_error_string",
     "name": "a failed turn answered with a bare "
             "'Error generating response.' / raw "
             "exception text, stating nothing",
     "source_doc": "docs/architecture/ERROR_HONESTY.md",
     "source_slice": "TF-031", "fix_owner": "TF-031",
     "docket": "D-HON-1",
     "kind": "reference",
     "test_file": "tests/test_tf031_error_honesty.py",
     "needle": "stated failure"},
    {"id": "hist_contaminated_prompt",
     "name": "whole-prompt keyword routing "
             "misrouting a turn on words quoted in "
             "its own history (turn 4 sent to the "
             "red-lantern answer)",
     "source_doc": "docs/architecture/MULTI_TURN_COHERENCE.md",
     "source_slice": "TF-028", "fix_owner": "TF-028",
     "docket": None,
     "kind": "reference",
     "test_file": "tests/test_tf028_multi_turn_coherence.py",
     "needle": "extract_request"},
    {"id": "hist_ollama_chat_crash",
     "name": "OllamaBackend unusable through the "
             "common surface (chat() missing — an "
             "AttributeError where an answer was "
             "asked)",
     "source_doc": "docs/architecture/BACKEND_CONTRACT.md",
     "source_slice": "TF-065", "fix_owner": "TF-065",
     "docket": "D-DEP-1",
     "kind": "reference",
     "test_file": "tests/test_tf065_backend_contract.py",
     "needle": "OllamaBackend().generate"},
    # ── gap_open: fixtured, fix not landed (TF-101) ──
    {"id": "gap_draft_side_hallucination",
     "name": "draft-side hallucination (invented "
             "facts, negation flip) passing the "
             "policy stack clean, filed refused",
     "source_doc": "docs/architecture/CHAOS_RUN.md",
     "source_slice": "TF-101", "fix_owner": "TF-088",
     "docket": None,
     "kind": "gap_open", "gap": "hallucination"},
    {"id": "gap_persona_voice_at_policy",
     "name": "persona-voice loss passing the "
             "policy stack clean (the voice guard "
             "is not a policy-surface check)",
     "source_doc": "docs/architecture/CHAOS_RUN.md",
     "source_slice": "TF-101", "fix_owner": "TF-096",
     "docket": None,
     "kind": "gap_open", "gap": "persona_voice_loss"},
    {"id": "gap_genericness_beyond_hard_phrases",
     "name": "fluent generic filler and the "
             "restated question passing clean (only "
             "hard phrases are checked)",
     "source_doc": "docs/architecture/CHAOS_RUN.md",
     "source_slice": "TF-101", "fix_owner": "TF-093",
     "docket": None,
     "kind": "gap_open", "gap": "genericness"},
    # ── not_proven: no live output exists ──
    {"id": "live_aesir_output_not_proven",
     "name": "a live Aesir server's output through "
             "the TF-074 backend — none has ever "
             "been produced to fixture, good or bad",
     "source_doc": "docs/architecture/AESIR_BACKEND.md",
     "source_slice": "TF-074", "fix_owner": "TF-086",
     "docket": "D-AESIR-1",
     "kind": "not_proven",
     "reason": "no live Aesir server has ever faced "
               "the backend (TF-016 D-AESIR-1); there "
               "is no historical output to fixture"},
    {"id": "live_third_party_oai_not_proven",
     "name": "a real third-party OpenAI-shaped "
             "server's output through the TF-076 "
             "backend — none has ever been produced "
             "to fixture, good or bad",
     "source_doc": "docs/architecture/OPENAI_COMPATIBLE_BACKEND.md",
     "source_slice": "TF-076", "fix_owner": "TF-086",
     "docket": "D-OAI-4",
     "kind": "not_proven",
     "reason": "no real third-party shape-speaking "
               "server has faced the backend (TF-016 "
               "D-OAI-4); the mocked legs are covered "
               "by hist_oai_parser_legs, a live one "
               "cannot be invented"},
)

_CASE_BY_ID = {c["id"]: c for c in ARMOUR_CASES}

#: The gap families TF-101 filed, with the exact
#: mismatch counts the live checker pins.
GAP_FAMILIES: dict[str, dict[str, Any]] = {
    "hallucination": {"entries": 4, "owner": "TF-088",
                      "gap_name": "draft_side_hallucination"
                                  "_not_judged_by_the_policy_surface"},
    "persona_voice_loss": {"entries": 3, "owner": "TF-096",
                           "gap_name": "persona_voice_not_a"
                                       "_policy_surface_check"},
    "genericness": {"entries": 2, "owner": "TF-093",
                    "gap_name": "genericness_checks_hard"
                                "_phrases_only"},
}

#: Total TF-101 mismatches the gap shape pins.
GAP_MISMATCH_TOTAL = 9

# ── Direct fixtures ─────────────────────────────────────────────────────────

ARMOUR_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "router_dead_patterns",
     "legs": (
         {"text": "tell me about the api",
          "field": "intent", "expected": "technical_spec_request"},
         {"text": "how do i plan the phases",
          "field": "intent", "expected": "planning_request"},
         {"text": "i don't know what to do anymore",
          "field": "intent", "expected": "emotional_support"},
         {"text": "i've been thinking about the old hall",
          "field": "tone_in", "expected": "curious"},
     )},
    {"id": "contradiction_overwrite",
     "slot": "ferry_departure",
     "stored_value": "dawn", "new_value": "midnight",
     "expected_rule": "fact_value_slot",
     "expected_resolutions": (
         "preserve_both_flagged", "supersede",
         "stored_stands")},
    {"id": "oai_double_v1",
     "base_url": "http://127.0.0.1:1234/v1",
     "expected_normalized": "http://127.0.0.1:1234"},
    {"id": "oai_outcome_legs",
     "legs": (
         {"name": "content_filter_not_a_completion",
          "payload": {"choices": [{
              "message": {"content": "partial"},
              "finish_reason": "content_filter"}]},
          "expected_ok": False,
          "error_mentions": "content_filter"},
         {"name": "null_content_not_a_none_text",
          "payload": {"choices": [{
              "message": {"content": None},
              "finish_reason": "stop"}]},
          "expected_ok": False,
          "error_mentions": "NoneType"},
         {"name": "empty_choices_not_a_crash",
          "payload": {"choices": []},
          "expected_ok": False,
          "error_mentions": "no choices"},
         {"name": "plain_completion_still_passes",
          "payload": {"choices": [{
              "message": {"content": "The ferry leaves "
                                     "at dawn."},
              "finish_reason": "stop"}]},
          "expected_ok": True,
          "expected_text": "The ferry leaves at dawn."},
     )},
    {"id": "loop_20x",
     "sentence": "The bell rings.", "repeats": 20,
     "expected_verdict": "rescued",
     "expected_occurrences": 1},
)

_FIXTURE_BY_ID = {f["id"]: f for f in ARMOUR_FIXTURES}


def case_by_id(case_id: str) -> dict[str, Any]:
    """One registry case by id (a detached copy)."""
    return copy.deepcopy(_CASE_BY_ID[case_id])


def kind_counts() -> dict[str, int]:
    """Live case count per armour kind."""
    counts = {kind: 0 for kind in ARMOUR_KINDS}
    for case in ARMOUR_CASES:
        counts[case["kind"]] = counts.get(case["kind"], 0) + 1
    return counts


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ── Checkers ────────────────────────────────────────────────────────────────


def regression_armour_problems() -> list[str]:
    """Data-integrity checker: registry ids unique;
    every case names an existing source doc, a
    naming slice and a fix owner; kinds in the
    vocabulary; per-kind armour fields present and
    resolving — rerun checkers importable, direct
    fixtures filed, reference test files carrying
    their needles on disk, gap families exactly
    GAP_FAMILIES with TF-101's counts, not_proven
    cases carrying their reasons; no canon marker
    anywhere."""
    problems: list[str] = []
    root = _repo_root()
    rule_ids = [r["id"] for r in ARMOUR_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate ARMOUR_RULES ids")
    ids = [c.get("id") for c in ARMOUR_CASES]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ARMOUR_CASES ids")
    fixture_ids = [f.get("id") for f in ARMOUR_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate ARMOUR_FIXTURES ids")
    used_fixtures: set[str] = set()
    used_gaps: set[str] = set()
    for case in ARMOUR_CASES:
        cid = case.get("id", "<no id>")
        if case.get("kind") not in ARMOUR_KINDS:
            problems.append(f"{cid}: unknown kind {case.get('kind')!r}")
        if not str(case.get("name", "")).strip():
            problems.append(f"{cid}: empty name")
        if not str(case.get("source_slice", "")).strip():
            problems.append(f"{cid}: no naming slice")
        if not str(case.get("fix_owner", "")).strip():
            problems.append(f"{cid}: no fix owner")
        doc = case.get("source_doc", "")
        if not (root / doc).is_file():
            problems.append(f"{cid}: source doc {doc!r} not on disk")
        kind = case.get("kind")
        if kind == "rerun":
            dotted = case.get("checker", "")
            module_name, _, attr = dotted.partition(":")
            try:
                module = __import__(module_name, fromlist=[attr])
                if not callable(getattr(module, attr, None)):
                    problems.append(
                        f"{cid}: checker {dotted!r} not callable")
            except Exception as exc:  # noqa: BLE001 — named, not hidden
                problems.append(f"{cid}: checker import failed: {exc}")
        elif kind == "direct":
            fixture_id = case.get("fixture")
            if fixture_id not in _FIXTURE_BY_ID:
                problems.append(f"{cid}: fixture {fixture_id!r} not filed")
            used_fixtures.add(fixture_id)
        elif kind == "reference":
            test_file = case.get("test_file", "")
            path = root / test_file
            if not path.is_file():
                problems.append(f"{cid}: test file {test_file!r} missing")
            elif case.get("needle", "") not in \
                    path.read_text(encoding="utf-8"):
                problems.append(
                    f"{cid}: needle {case.get('needle')!r} not in "
                    f"{test_file}")
        elif kind == "gap_open":
            gap = case.get("gap")
            if gap not in GAP_FAMILIES:
                problems.append(f"{cid}: unknown gap family {gap!r}")
            else:
                used_gaps.add(gap)
                if GAP_FAMILIES[gap]["owner"] != case.get("fix_owner"):
                    problems.append(
                        f"{cid}: gap owner {case.get('fix_owner')!r} "
                        f"!= filed {GAP_FAMILIES[gap]['owner']!r}")
        elif kind == "not_proven":
            if not str(case.get("reason", "")).strip():
                problems.append(f"{cid}: not_proven without a reason")
        blob = repr(case)
        for marker in FORBIDDEN_MARKERS:
            if marker in blob:
                problems.append(f"{cid}: canon marker {marker!r} present")
    if used_fixtures != set(_FIXTURE_BY_ID):
        problems.append(
            f"filed fixtures {sorted(_FIXTURE_BY_ID)} != used "
            f"{sorted(used_fixtures)}")
    if used_gaps != set(GAP_FAMILIES):
        problems.append("GAP_FAMILIES not fully registered as gap_open")
    if sum(g["entries"] for g in GAP_FAMILIES.values()) \
            != GAP_MISMATCH_TOTAL:
        problems.append("gap entry counts do not sum to the filed total")
    return problems


def _direct_problems() -> list[str]:
    problems: list[str] = []
    from thoughtforge.cognition.router import InputRouter
    from thoughtforge.inference.openai_compatible import (
        chat_completion_outcome,
        normalize_base_url,
    )
    from thoughtforge.knowledge.contradictions import (
        assess_contradiction,
    )
    from thoughtforge.knowledge.models import UserFactRecord
    from thoughtforge.refinement.loop_rescue import rescue_loops

    router_fixture = _FIXTURE_BY_ID["router_dead_patterns"]
    router = InputRouter()
    for leg in router_fixture["legs"]:
        sketch = router.route(leg["text"])
        actual = getattr(sketch, leg["field"], None)
        if actual != leg["expected"]:
            problems.append(
                f"router_dead_patterns: {leg['text']!r} "
                f"{leg['field']} {actual!r} != {leg['expected']!r}")

    contra = _FIXTURE_BY_ID["contradiction_overwrite"]
    stored = UserFactRecord(
        record_id="armour_stored", summary="The ferry leaves at dawn",
        fact_value={contra["slot"]: contra["stored_value"]},
        confidence=0.6, source="", tags=[])
    new = UserFactRecord(
        record_id="armour_new", summary="The ferry leaves at midnight",
        fact_value={contra["slot"]: contra["new_value"]},
        confidence=0.6, source="", tags=[])
    report = assess_contradiction(new, [stored])
    if len(report.conflicts) != 1:
        problems.append(
            f"contradiction_overwrite: {len(report.conflicts)} "
            "conflicts, filed 1 (a silent overwrite surfaces none)")
    else:
        conflict = report.conflicts[0]
        if conflict.rule != contra["expected_rule"]:
            problems.append(
                f"contradiction_overwrite: rule {conflict.rule!r} "
                f"!= {contra['expected_rule']!r}")
        if conflict.resolution not in contra["expected_resolutions"]:
            problems.append(
                f"contradiction_overwrite: resolution "
                f"{conflict.resolution!r} not a stated resolution")

    v1 = _FIXTURE_BY_ID["oai_double_v1"]
    normalized = normalize_base_url(v1["base_url"])
    if normalized != v1["expected_normalized"]:
        problems.append(
            f"oai_double_v1: normalized {normalized!r} != "
            f"{v1['expected_normalized']!r}")

    outcomes = _FIXTURE_BY_ID["oai_outcome_legs"]
    for leg in outcomes["legs"]:
        outcome = chat_completion_outcome(copy.deepcopy(leg["payload"]))
        if outcome.get("ok") is not leg["expected_ok"]:
            problems.append(
                f"oai_outcome_legs/{leg['name']}: ok "
                f"{outcome.get('ok')!r} != {leg['expected_ok']!r}")
        elif leg["expected_ok"] and \
                outcome.get("text") != leg["expected_text"]:
            problems.append(
                f"oai_outcome_legs/{leg['name']}: text changed")
        elif not leg["expected_ok"] and \
                leg["error_mentions"] not in outcome.get("error", ""):
            problems.append(
                f"oai_outcome_legs/{leg['name']}: error "
                f"{outcome.get('error')!r} does not name "
                f"{leg['error_mentions']!r}")

    loop = _FIXTURE_BY_ID["loop_20x"]
    text = " ".join([loop["sentence"]] * loop["repeats"])
    result = rescue_loops(text)
    if result.verdict != loop["expected_verdict"]:
        problems.append(
            f"loop_20x: verdict {result.verdict!r} != "
            f"{loop['expected_verdict']!r}")
    if result.text.count(loop["sentence"]) != \
            loop["expected_occurrences"]:
        problems.append(
            f"loop_20x: {result.text.count(loop['sentence'])} "
            "occurrences survive, filed "
            f"{loop['expected_occurrences']}")
    return problems


def _gap_problems() -> list[str]:
    """Pin TF-101's gap shape exactly: the scored
    run's mismatches are precisely the filed 9,
    in precisely the 3 gap families, each naming
    its filed gap in TF-101's NAMED_GAPS."""
    problems: list[str] = []
    from thoughtforge.refinement.chaos_run import (
        NAMED_GAPS,
        run_chaos,
    )

    run = run_chaos()
    mismatches = run["mismatches"]
    if len(mismatches) != GAP_MISMATCH_TOTAL:
        problems.append(
            f"gap shape: {len(mismatches)} mismatches, filed "
            f"{GAP_MISMATCH_TOTAL}")
    by_family: dict[str, int] = {}
    for row in mismatches:
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1
    expected = {name: g["entries"]
                for name, g in GAP_FAMILIES.items()}
    if by_family != expected:
        problems.append(
            f"gap shape: families {by_family} != filed {expected}")
    filed_gap_names = {g["gap_name"] for g in GAP_FAMILIES.values()}
    named = {g["id"] for g in NAMED_GAPS}
    if not filed_gap_names <= named:
        problems.append(
            f"gap shape: registry gap names "
            f"{sorted(filed_gap_names - named)} not in TF-101's "
            "NAMED_GAPS")
    for row in mismatches:
        if row.get("gap") not in filed_gap_names:
            problems.append(
                f"gap shape: mismatch {row.get('id')!r} names gap "
                f"{row.get('gap')!r}, not a filed registry gap")
    return problems


def regression_armour_live_problems() -> list[str]:
    """Live checker: every covered case re-proven
    today — rerun cases through their fixing
    modules' own live checkers, direct fixtures
    against the real fix surfaces, and the
    gap_open shape recomputed through TF-101's
    real runner (exactly the filed 9 mismatches
    in the filed 3 families). Reference cases are
    proven by the data checker's on-disk needle
    verification; not_proven cases are claims of
    absence and are never run."""
    problems: list[str] = []
    for case in ARMOUR_CASES:
        if case["kind"] != "rerun":
            continue
        module_name, _, attr = case["checker"].partition(":")
        module = __import__(module_name, fromlist=[attr])
        for problem in getattr(module, attr)():
            problems.append(f"{case['id']}: {problem}")
    problems.extend(_direct_problems())
    problems.extend(_gap_problems())
    return problems


def armour_as_data() -> dict[str, Any]:
    counts = kind_counts()
    return {
        "format": ARMOUR_FORMAT,
        "armour_doc": ARMOUR_DOC,
        "rules": [dict(r) for r in ARMOUR_RULES],
        "kinds": list(ARMOUR_KINDS),
        "kind_counts": counts,
        "covered": sum(counts[k] for k in COVERED_KINDS),
        "total_cases": len(ARMOUR_CASES),
        "cases": copy.deepcopy(list(ARMOUR_CASES)),
        "fixtures": copy.deepcopy(list(ARMOUR_FIXTURES)),
        "gap_families": copy.deepcopy(GAP_FAMILIES),
        "gap_mismatch_total": GAP_MISMATCH_TOTAL,
        "proof": {
            "grade": "fixture",
            "live_model_regressions": "not_proven",
            "not_proven_cases": counts["not_proven"],
            "gap_open_cases": counts["gap_open"],
        },
    }
