"""TF-097 — Provenance of rescue: a salvaged answer
carries its rescue record into the trace (TF-084).

Roadmap: "Provenance of rescue: a salvaged answer
carries its rescue record into the trace (TF-084)."

Measured start (probed live this tick, through the
real ``FragmentSalvage``):

- **A result knew only its path string.** TF-096's
  benign adopted fixture and its voice-rejected
  fixture both returned ``salvage_path ==
  "refine_pass_1"`` with ``passes_used == 1`` —
  adopted and rejected were indistinguishable in
  the result. TF-087 had already filed the naming
  debt: ``refine_pass_N`` names passes *attempted*,
  not adopted, and "TF-097's rescue record must
  carry that distinction".
- **The refine passes were invisible to the
  trace.** TF-084's joined trace records the
  core's own leaf calls, but the passes
  ``FragmentSalvage`` makes through the engine
  inside its own ``forge()`` were recorded
  nowhere — the core hands the engine over and
  cannot see them. TF-084 filed that as
  D-TRACE-1, OPEN, assigned here: "an aggregate
  invented from the salvage result would be
  fabrication", so the passes had to be recorded
  *by salvage itself*, at the moment each runs.

What this slice builds:

- **The rescue record** (format
  ``thoughtforge-rescue-record/1``), built inside
  ``FragmentSalvage.forge()`` and carried on
  every ``SalvageResult``: the candidates as
  scored (id / chars / length / citation /
  composite / citations found — sizes and scores,
  never texts), the selection (selected candidate
  id, whether TF-096's persona filter applied and
  how many candidates were consistent), fragments
  kept, and **each refine pass individually** —
  pass number, outcome (``adopted`` /
  ``rejected_fabrication`` / ``rejected_voice`` /
  ``not_improved`` / ``failed``), whether it
  counted in ``passes_used``, its score and output
  size, the guards' own problem lists verbatim
  (TF-088's named codes, TF-096's avoid entries),
  the failure string when it failed, and two
  latencies named never conflated (the wall
  salvage measured around the engine call, and
  what the engine itself reported, ``None`` when
  it reported nothing).
- **The final block answers the naming debt**:
  beside ``salvage_path`` (attempted naming,
  unchanged) it states ``final_source`` —
  ``selected_draft`` or ``refine_pass`` — and
  ``final_pass``, the adopted pass the final text
  actually came from, or ``None``.
- **A failed attempt is provenance too.** An
  engine exception on pass 1 makes ``forge()``
  fall back to the best draft with
  ``passes_used == 0``; before this slice that
  attempt left no trace anywhere. The record
  carries it (``outcome == "failed"``, the error
  string, ``counted_in_passes_used == False`` —
  ``passes_used`` still counts only completed
  generations, exactly as before).
- **Into the trace**: the core's
  ``_compose_final`` passes the record onto the
  ``FinalResponseRecord`` (a new defaulted field),
  and TF-038's trace export carries it in the
  final section — so TF-084's joined trace, which
  rides the cognition trace whole, carries the
  passes inside its cognition section.
  ``trace_problems`` judges a carried record by
  this module's checker. The inference section is
  unchanged: the recorder records the core's own
  leaf calls, and salvage's passes are not the
  core's calls — their provenance is this
  record's job, stated in TF-084's rewritten
  ``salvage_refine_passes_are_a_named_gap`` rule
  (kept id, D-TRACE-1 now FIXED by this slice).

Proof grade FIXTURE — stub-engine outputs and
candidate sets run through the real
FragmentSalvage on this VM, plus a real core turn
whose joined trace is checked whole. A live
model's rescue record is NOT PROVEN (no model in
tree); the record's shape does not depend on who
wrote the candidates.

The human-readable write-up is
``docs/architecture/RESCUE_PROVENANCE.md``.
"""

from __future__ import annotations

from typing import Any

#: This module's estate format.
RESCUE_FORMAT = "thoughtforge-rescue-provenance/1"

#: The record format salvage builds and the trace carries.
RESCUE_RECORD_FORMAT = "thoughtforge-rescue-record/1"

RESCUE_DOC = "docs/architecture/RESCUE_PROVENANCE.md"

#: The outcome vocabulary of one refine pass.
PASS_OUTCOMES: tuple[str, ...] = (
    "adopted",
    "rejected_fabrication",
    "rejected_voice",
    "not_improved",
    "failed",
)

#: Where a final text can have come from.
FINAL_SOURCES: tuple[str, ...] = (
    "empty",
    "selected_draft",
    "refine_pass",
)

_SCORE_TOL = 1e-3

# ── The rules ───────────────────────────────────────────────────────────────

RESCUE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "every_forge_carries_its_record",
     "rule": "every FragmentSalvage.forge() result carries a "
             "rescue record in RESCUE_RECORD_FORMAT — empty, "
             "best-draft, and refine outcomes alike; a record "
             "is never present only for the interesting turns"},
    {"id": "attempted_is_not_adopted",
     "rule": "salvage_path keeps TF-087's attempted naming "
             "(refine_pass_N = N passes attempted); the record's "
             "final block states final_source and final_pass — "
             "which adopted pass the final text came from, or "
             "the selected draft — so the distinction TF-087 "
             "filed is carried, not renamed away"},
    {"id": "every_pass_recorded_individually",
     "rule": "D-TRACE-1 (FIXED by TF-097): each refine pass "
             "salvage runs is recorded individually as it runs "
             "— outcome, score, output size, rejection reasons, "
             "failure string — by salvage itself, never as an "
             "aggregate invented afterwards from the result"},
    {"id": "rejection_reasons_are_the_guards_own",
     "rule": "a rejected pass carries the guard's own lists "
             "verbatim — TF-088's fabrication problem codes and "
             "TF-096's new avoid entries — never a paraphrase "
             "invented by the record"},
    {"id": "latencies_named_never_conflated",
     "rule": "latency_ms is the wall salvage measured around "
             "the engine call; reported_latency_ms is what the "
             "engine itself reported (its duration_ms), filed "
             "beside it and None when the engine reported "
             "nothing — neither is substituted for the other "
             "(TF-084's rule, applied inside salvage)"},
    {"id": "a_failed_attempt_is_provenance",
     "rule": "an engine failure is recorded (outcome failed, "
             "the error string) even when forge() then falls "
             "back to the best draft; counted_in_passes_used is "
             "False for it — passes_used still counts only "
             "completed generations, exactly as before this slice"},
    {"id": "sizes_and_scores_never_texts",
     "rule": "the record carries candidate ids, char counts "
             "and scores, pass output sizes and scores, and the "
             "guards' problem codes — never candidate, fragment "
             "or output texts (the texts live once, in the "
             "trace's candidates and final; TF-084's ruling)"},
    {"id": "the_record_is_derived_never_trusted",
     "rule": "rescue_record_problems recomputes the record's "
             "internal identities — candidate composites from "
             "their filed scores, passes_used from the counted "
             "passes, the final's confidence from the selected "
             "candidate or the adopted pass — and names any "
             "disagreement; TF-038's trace_problems judges a "
             "carried record by the same checker"},
    {"id": "the_surfaces_are_unchanged",
     "rule": "the scoring formula, TF-088's and TF-096's "
             "guards, the adoption rule, passes_used semantics "
             "and salvage_path naming are unchanged; this slice "
             "adds recording beside them and one defaulted "
             "field on SalvageResult and FinalResponseRecord"},
)

# ── The checker ─────────────────────────────────────────────────────────────


def rescue_record_problems(record: Any) -> list[str]:
    """Problems with a rescue record, as data ([] = sound).

    Judges the record as a self-contained document:
    format, candidate rows whose composites recompute
    from their filed scores, a selection that names a
    filed candidate, per-pass records in sequence
    with outcomes in the vocabulary and their
    required companions (a rejection carries its
    guard's list, a failure its error), and the
    final block's identities — passes_used is the
    counted passes, final_pass is the last adopted
    pass, and the filed confidence is the selected
    candidate's or the adopted pass's score.
    """
    if not isinstance(record, dict):
        return ["rescue record is not a mapping"]
    problems: list[str] = []
    if record.get("format") != RESCUE_RECORD_FORMAT:
        problems.append(
            f"rescue record format is {record.get('format')!r}, "
            f"not {RESCUE_RECORD_FORMAT!r}")

    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        problems.append("rescue record candidates are not a list")
        candidates = []
    candidate_scores: dict[str, float] = {}
    for row in candidates:
        cid = row.get("candidate_id")
        if cid in candidate_scores:
            problems.append(f"candidate {cid!r} is filed twice")
        composite = row.get("composite")
        expected = (0.45 * (row.get("length_score") or 0.0)
                    + 0.55 * (row.get("citation_score") or 0.0))
        if isinstance(composite, (int, float)):
            candidate_scores[cid] = composite
            if abs(composite - expected) > _SCORE_TOL:
                problems.append(
                    f"candidate {cid!r}: composite {composite} "
                    f"disagrees with its filed scores ({expected:.6f})")
        if not isinstance(row.get("text_chars"), int) \
                or row.get("text_chars") < 0:
            problems.append(f"candidate {cid!r}: text_chars is not "
                            "a non-negative int")

    selection = record.get("selection") or {}
    selected = selection.get("selected_candidate_id")
    if candidates and selected not in candidate_scores:
        problems.append(
            f"selection names candidate {selected!r} absent "
            "from the record")
    if not candidates and selected is not None:
        problems.append("an empty record selects a candidate")
    if selection.get("persona_filter_applied"):
        count = selection.get("persona_consistent_count")
        if not isinstance(count, int) \
                or not 0 <= count <= len(candidates):
            problems.append("persona_consistent_count is not an "
                            "int within the candidate count")
    elif selection.get("persona_consistent_count") is not None:
        problems.append("persona_consistent_count is filed "
                        "without the persona filter")

    passes = record.get("refine_passes")
    if not isinstance(passes, list):
        problems.append("rescue record refine_passes are not a list")
        passes = []
    counted = 0
    adopted_scores: dict[int, float] = {}
    for idx, entry in enumerate(passes):
        label = f"pass {idx + 1}"
        if entry.get("pass") != idx + 1:
            problems.append(f"{label}: filed as pass "
                            f"{entry.get('pass')!r}")
        outcome = entry.get("outcome")
        if outcome not in PASS_OUTCOMES:
            problems.append(f"{label}: outcome {outcome!r} is "
                            "not a known pass outcome")
        if entry.get("counted_in_passes_used") != (outcome != "failed"):
            problems.append(f"{label}: counted_in_passes_used "
                            "disagrees with its outcome")
        if outcome != "failed":
            counted += 1
            if not isinstance(entry.get("score"), (int, float)):
                problems.append(f"{label}: a completed pass "
                                "carries no score")
            if not isinstance(entry.get("text_chars"), int):
                problems.append(f"{label}: a completed pass "
                                "carries no text_chars")
        elif not entry.get("error"):
            problems.append(f"{label}: a failed pass carries "
                            "no error detail")
        if outcome == "rejected_fabrication" \
                and not entry.get("fabrication_problems"):
            problems.append(f"{label}: a fabrication rejection "
                            "carries no fabrication problems")
        if outcome == "rejected_voice" \
                and not entry.get("voice_problems"):
            problems.append(f"{label}: a voice rejection "
                            "carries no voice problems")
        if outcome == "adopted" \
                and isinstance(entry.get("score"), (int, float)):
            adopted_scores[idx + 1] = entry["score"]
    if passes and not record.get("refine_attempted"):
        problems.append("passes are filed but refine_attempted "
                        "is false")

    final = record.get("final") or {}
    if final.get("passes_used") != counted:
        problems.append(
            f"final passes_used {final.get('passes_used')!r} "
            f"disagrees with the counted passes ({counted})")
    last_adopted = max(adopted_scores) if adopted_scores else None
    if final.get("final_pass") != last_adopted:
        problems.append(
            f"final_pass {final.get('final_pass')!r} is not the "
            f"last adopted pass ({last_adopted!r})")
    source = final.get("final_source")
    if source not in FINAL_SOURCES:
        problems.append(f"final_source {source!r} is not a "
                        "known final source")
    if (source == "refine_pass") != (last_adopted is not None):
        problems.append("final_source disagrees with the "
                        "adopted passes")
    path = final.get("salvage_path")
    if source == "empty":
        if path != "empty" or candidates:
            problems.append("an empty final disagrees with its "
                            "path or candidates")
    elif path == "best_draft":
        if source != "selected_draft" or counted != 0:
            problems.append("a best_draft final disagrees with "
                            "its source or counted passes")
    elif isinstance(path, str) and path.startswith("refine_pass_"):
        if int(path.rsplit("_", 1)[1]) != counted or counted == 0:
            problems.append("a refine path disagrees with the "
                            "counted passes")
    else:
        problems.append(f"final salvage_path {path!r} is not a "
                        "known salvage path")
    confidence = final.get("confidence")
    if isinstance(confidence, (int, float)):
        if last_adopted is not None:
            if abs(confidence - adopted_scores[last_adopted]) > _SCORE_TOL:
                problems.append("final confidence disagrees with "
                                "the adopted pass's score")
        elif selected in candidate_scores:
            if abs(confidence - candidate_scores[selected]) > _SCORE_TOL:
                problems.append("final confidence disagrees with "
                                "the selected candidate's composite")
    best = final.get("best_draft_score")
    if candidates and isinstance(best, (int, float)):
        if abs(best - max(candidate_scores.values())) > _SCORE_TOL:
            problems.append("best_draft_score disagrees with the "
                            "filed candidates")
    return problems


def rescue_record_result_problems(record: Any, result: Any) -> list[str]:
    """A record judged against the SalvageResult it
    rode on: the final block must be the result's own
    numbers, and the record's fragments_kept the
    result's."""
    problems = rescue_record_problems(record)
    if not isinstance(record, dict):
        return problems
    final = record.get("final") or {}
    if final.get("salvage_path") != result.salvage_path:
        problems.append("final salvage_path disagrees with "
                        "the result")
    if final.get("passes_used") != result.passes_used:
        problems.append("final passes_used disagrees with "
                        "the result")
    if abs((final.get("confidence") or 0.0)
           - round(result.confidence, 6)) > _SCORE_TOL:
        problems.append("final confidence disagrees with "
                        "the result")
    if record.get("fragments_kept") != result.fragments_kept:
        problems.append("fragments_kept disagrees with the result")
    return problems


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic throughout (TF-017's Lantern Hollow, never
# canon). The refine fixtures' candidate cites Q123 so
# its fragments clear salvage's keep threshold and the
# refine legs actually run.

_BASE = (
    "According to Q123, the ferry crossing takes eleven "
    "minutes in calm water. Wrenna records the river level "
    "each morning in chalk on the mill door.")
_SHORT = "The ferry is quick."
_BENIGN = (
    "According to Q123, the ferry crossing takes eleven "
    "minutes in calm water. Each morning Wrenna records "
    "the river level in chalk on the mill door, and after "
    "the second tide bell the ferry stops running until "
    "morning, whatever the weather.")
_FABRICATED = (
    "According to Q123, the ferry crossing takes eleven "
    "minutes in calm water and costs 50 silver coins for "
    "every traveller. Wrenna records the river level each "
    "morning in chalk on the mill door, and the boats wait "
    "in a row by the landing steps for everyone who comes "
    "down to the river in the morning light today.")
_FABRICATED_2 = (
    "According to Q123, the ferry crossing takes eleven "
    "minutes in calm water and costs 50 silver coins. Each "
    "morning Wrenna records the river level in chalk on "
    "the mill door, and after the second tide bell the "
    "ferry stops running until morning, whatever the "
    "weather brings for the travellers waiting there.")
_FLAT = (
    "According to Q123, it is hard to say — really just "
    "vague handwaving on my part, but the river does what "
    "it does, and the ferry crossing takes eleven minutes "
    "in calm water while Wrenna records the river level "
    "each morning in chalk on the mill door, and travellers "
    "should simply wait and see what the morning brings "
    "for everyone who waits.")

RESCUE_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "empty_candidates",
     "kind": "empty",
     "candidates": (),
     "engine_outputs": None,
     "with_expectation": False,
     "max_passes": 2,
     "expected_path": "empty",
     "expected_final_source": "empty",
     "expected_final_pass": None,
     "expected_outcomes": ()},
    {"id": "best_draft_no_engine",
     "kind": "selection",
     "candidates": (_SHORT, _BASE),
     "engine_outputs": None,
     "with_expectation": False,
     "max_passes": 2,
     "expected_path": "best_draft",
     "expected_final_source": "selected_draft",
     "expected_final_pass": None,
     "expected_outcomes": (),
     "expected_text": _BASE},
    {"id": "refine_adopted",
     "kind": "refine",
     "candidates": (_BASE,),
     "engine_outputs": (_BENIGN,),
     "with_expectation": False,
     "max_passes": 1,
     "expected_path": "refine_pass_1",
     "expected_final_source": "refine_pass",
     "expected_final_pass": 1,
     "expected_outcomes": ("adopted",),
     "expected_text": _BENIGN},
    {"id": "refine_rejected_fabrication",
     "kind": "refine",
     "candidates": (_BASE,),
     "engine_outputs": (_FABRICATED,),
     "with_expectation": False,
     "max_passes": 1,
     "expected_path": "refine_pass_1",
     "expected_final_source": "selected_draft",
     "expected_final_pass": None,
     "expected_outcomes": ("rejected_fabrication",),
     "expected_text": _BASE},
    {"id": "refine_rejected_voice",
     "kind": "refine",
     "candidates": (_BASE,),
     "engine_outputs": (_FLAT,),
     "with_expectation": True,
     "max_passes": 1,
     "expected_path": "refine_pass_1",
     "expected_final_source": "selected_draft",
     "expected_final_pass": None,
     "expected_outcomes": ("rejected_voice",),
     "expected_text": _BASE},
    {"id": "refine_not_improved",
     "kind": "refine",
     "candidates": (_BASE,),
     "engine_outputs": (_SHORT,),
     "with_expectation": False,
     "max_passes": 1,
     "expected_path": "refine_pass_1",
     "expected_final_source": "selected_draft",
     "expected_final_pass": None,
     "expected_outcomes": ("not_improved",),
     "expected_text": _BASE},
    {"id": "refine_engine_failure_falls_back",
     "kind": "refine",
     "candidates": (_BASE,),
     "engine_outputs": ("RAISE:engine blew up",),
     "with_expectation": False,
     "max_passes": 1,
     "expected_path": "best_draft",
     "expected_final_source": "selected_draft",
     "expected_final_pass": None,
     "expected_outcomes": ("failed",),
     "expected_text": _BASE,
     "note": "the attempt is recorded even though "
             "forge() falls back and passes_used "
             "stays 0 — the pre-TF-097 result left "
             "this attempt nowhere"},
    {"id": "refine_adopt_then_fabricate",
     "kind": "refine",
     "candidates": (_BASE,),
     "engine_outputs": (_BENIGN, _FABRICATED_2),
     "with_expectation": False,
     "max_passes": 2,
     "expected_path": "refine_pass_2",
     "expected_final_source": "refine_pass",
     "expected_final_pass": 1,
     "expected_outcomes": ("adopted", "rejected_fabrication"),
     "expected_text": _BENIGN,
     "note": "the naming debt in one fixture: the "
             "path says pass 2 (attempted), the "
             "record says the text came from pass 1"},
)

_FIXTURE_BY_ID = {f["id"]: f for f in RESCUE_FIXTURES}

#: The stub engines' reported duration — filed so the
#: tests can prove the reported latency rides beside
#: the wall, never in its place.
STUB_DURATION_MS = 3.5


class _StubEngine:
    def __init__(self, outputs: tuple[str, ...]):
        self._outputs = list(outputs)

    def generate(self, prompt, params=None):
        from types import SimpleNamespace
        item = self._outputs.pop(0)
        if item.startswith("RAISE:"):
            raise RuntimeError(item[len("RAISE:"):])
        return SimpleNamespace(text=item, duration_ms=STUB_DURATION_MS)


def run_rescue_fixture(fixture_id: str):
    """Run one filed fixture through the real
    ``FragmentSalvage`` and return its SalvageResult
    (whose ``rescue_record`` the checkers judge)."""
    from thoughtforge.cognition.persona import FIXTURE_EXPECTATION
    from thoughtforge.cognition.prompt_builder import PromptBuilder
    from thoughtforge.knowledge.models import (
        ActivatedRecord,
        CandidateRecord,
        CandidateScores,
        CognitionScaffold,
        InputSketch,
        MemoryActivationBundle,
    )
    from thoughtforge.refinement.salvage import FragmentSalvage

    fixture = _FIXTURE_BY_ID[fixture_id]
    candidates = [
        CandidateRecord(
            candidate_id=f"cand_{i}", mode="practical", text=text,
            token_estimate=len(text.split()), scores=CandidateScores())
        for i, text in enumerate(fixture["candidates"])
    ]
    bundle = MemoryActivationBundle(
        activated_records=[ActivatedRecord(
            "Q123", "entity", 0.8, "cue for Q123",
            raw={"qid": "Q123"})],
        total_retrieved=1, retrieval_confidence=0.8)
    expectation = FIXTURE_EXPECTATION if fixture[
        "with_expectation"] else None
    if fixture["kind"] != "refine":
        return FragmentSalvage().forge(
            candidates, bundle, persona_expectation=expectation)
    sketch = InputSketch(
        raw_text="when does the ferry leave?",
        intent="factual_query", topic="ferry",
        response_mode="factual_direct")
    scaffold = CognitionScaffold(
        goal="give a clear direct answer",
        tone=["direct", "factual"], avoid=["filler"])
    return FragmentSalvage().forge(
        candidates, bundle,
        engine=_StubEngine(fixture["engine_outputs"]),
        prompt_builder=PromptBuilder(), sketch=sketch,
        scaffold=scaffold, max_passes=fixture["max_passes"],
        persona_expectation=expectation)


# ── Live checker + estate as data ───────────────────────────────────────────


def rescue_provenance_live_problems() -> list[str]:
    """Live checker: every filed fixture is run
    through the real FragmentSalvage and its record
    must land exactly as filed — path, final source,
    final pass, per-pass outcomes — and judge clean
    against both checkers and its own result."""
    problems: list[str] = []
    for fixture in RESCUE_FIXTURES:
        result = run_rescue_fixture(fixture["id"])
        record = result.rescue_record
        if record is None:
            problems.append(
                f"fixture {fixture['id']}: no rescue record")
            continue
        problems.extend(
            f"fixture {fixture['id']}: {p}"
            for p in rescue_record_result_problems(record, result))
        final = record["final"]
        if final["salvage_path"] != fixture["expected_path"]:
            problems.append(
                f"fixture {fixture['id']}: path "
                f"{final['salvage_path']} != "
                f"{fixture['expected_path']}")
        if final["final_source"] != fixture["expected_final_source"]:
            problems.append(
                f"fixture {fixture['id']}: final_source "
                f"{final['final_source']} != "
                f"{fixture['expected_final_source']}")
        if final["final_pass"] != fixture["expected_final_pass"]:
            problems.append(
                f"fixture {fixture['id']}: final_pass "
                f"{final['final_pass']} != "
                f"{fixture['expected_final_pass']}")
        outcomes = tuple(p["outcome"]
                         for p in record["refine_passes"])
        if outcomes != tuple(fixture["expected_outcomes"]):
            problems.append(
                f"fixture {fixture['id']}: outcomes "
                f"{outcomes} != {fixture['expected_outcomes']}")
        if fixture.get("expected_text") is not None \
                and result.text != fixture["expected_text"]:
            problems.append(
                f"fixture {fixture['id']}: text differs "
                "from filed")
        for entry in record["refine_passes"]:
            if entry["outcome"] != "failed" and entry[
                    "reported_latency_ms"] != STUB_DURATION_MS:
                problems.append(
                    f"fixture {fixture['id']}: pass "
                    f"{entry['pass']} reported latency is "
                    f"{entry['reported_latency_ms']}, not "
                    "the stub's filed duration")
    return problems


def rescue_provenance_problems() -> list[str]:
    """Data-integrity checker: rule and fixture ids
    unique, fixtures well-formed, and the record
    wired into salvage.py + the core + the trace
    (source anchors)."""
    import inspect

    problems: list[str] = []
    rule_ids = [r["id"] for r in RESCUE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate RESCUE_RULES ids")
    fixture_ids = [f["id"] for f in RESCUE_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate RESCUE_FIXTURES ids")
    for fixture in RESCUE_FIXTURES:
        if (fixture["kind"] == "refine") != (
                fixture["engine_outputs"] is not None):
            problems.append(
                f"fixture {fixture['id']}: kind/engine_outputs "
                "disagree")
        if len(fixture["expected_outcomes"]) not in (0, 1, 2):
            problems.append(
                f"fixture {fixture['id']}: outcomes length "
                "outside the filed shapes")
    from thoughtforge.cognition import traces as traces_module
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.refinement import salvage as salvage_module

    salvage_source = inspect.getsource(salvage_module)
    if "rescue_record" not in salvage_source:
        problems.append("salvage.py builds no rescue record")
    if "RESCUE_RECORD_FORMAT" not in salvage_source:
        problems.append("salvage.py does not file the record "
                        "format")
    if "rescue_record" not in inspect.getsource(
            ThoughtForgeCore._compose_final):
        problems.append("core._compose_final does not carry "
                        "the rescue record")
    if "rescue_record" not in inspect.getsource(
            traces_module._final_section):
        problems.append("traces._final_section does not carry "
                        "the rescue record into the trace")
    return problems


def rescue_provenance_as_data() -> dict[str, Any]:
    return {
        "format": RESCUE_FORMAT,
        "record_format": RESCUE_RECORD_FORMAT,
        "rescue_doc": RESCUE_DOC,
        "rules": [dict(r) for r in RESCUE_RULES],
        "pass_outcomes": list(PASS_OUTCOMES),
        "final_sources": list(FINAL_SOURCES),
        "fixtures": [
            {k: v for k, v in f.items()} for f in RESCUE_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_rescue_record": "not_proven",
            "d_trace_1": "fixed",
            "joined_trace": "carried_in_cognition_final",
            "inference_section": "unchanged",
        },
    }
