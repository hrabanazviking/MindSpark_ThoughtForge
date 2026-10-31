"""TF-087 — Salvage mapped: the strategies in
``refinement/salvage.py`` catalogued as data, each
with fixtures of broken input, and the rescue
strategies the roadmap names that do NOT exist
filed as gaps with owners.

Roadmap: "Salvage mapped: salvage.py's strategies
catalogued (truncation repair, rambling compression,
format rescue, …) each with fixtures of broken
input."

Measured start (probed live this tick, every leg
run through the real ``FragmentSalvage``):

- **What salvage.py actually does is narrower
  than its name suggests.** It scores drafts
  (length 0.45 + citation 0.55, ideal 550 chars),
  selects the best, extracts sentence fragments
  from the top 2 drafts (≥4 words, deduped, scored
  ≥0.54, at most 5), optionally runs up to 2
  engine refine passes (adopting a pass only when
  its score beats the current best), and otherwise
  returns the best draft *verbatim*. An empty
  candidate list returns one fixed sentence at
  confidence 0. That is the whole estate:
  ``STRATEGIES`` files its five real strategies.
- **The roadmap's parenthetical strategies do
  not exist.** A truncated draft ending
  mid-sentence ("…rings across the") was returned
  byte-identical; a broken JSON-shaped draft
  ('{"ferry": "dawn", "tide": ') was returned
  byte-identical; a 20× repeated sentence loop
  was returned byte-identical (dedup touches the
  fragment list, never the output text); a lone
  rambling draft was returned byte-identical (a
  cited concise rival beats rambling only by
  *selection*, 0.627 vs 0.431 — nothing compresses
  the rambling itself). No language-drift check
  exists anywhere in the module. Filed OPEN as
  D-SALV-1, owners in ``ABSENT_STRATEGIES``
  (TF-089 / TF-090 / TF-091 / TF-092).
- **One naming honesty point.** When a refine
  pass runs but its output scores no better, the
  result still reports ``salvage_path=
  refine_pass_1`` with ``passes_used=1`` over the
  *original* text (probed): the path names passes
  attempted, not a rescue adopted. TF-097's
  rescue record must carry that distinction.
- **Fragments do not alter a no-engine output.**
  The mixed fixture (junk short sentences + one
  long cited sentence) keeps exactly 1 fragment,
  yet the returned text is the whole best draft,
  junk included — fragments only feed refine
  passes. Catalogued so no later slice mistakes
  ``fragments_kept`` for a cleaned answer.

The fixtures (``BROKEN_FIXTURES``) are synthetic
Lantern Hollow material (TF-017's world, never
canon), each run live by
``salvage_map_live_problems`` (the two refine
fixtures need an engine + prompt builder, so the
checker marks them engine-required and the tests
exercise them with a stub engine and the real
``PromptBuilder``).

Proof grade FIXTURE — every filed behaviour was
run through the real FragmentSalvage on this VM
this tick. A live model's broken outputs are NOT
PROVEN here (no engine/model in tree); TF-100's
chaos corpus owns real-shaped failures.
"""

from __future__ import annotations

from typing import Any

SALVAGE_FORMAT = "thoughtforge-salvage-map/1"

SALVAGE_DOC = "docs/architecture/SALVAGE_MAPPED.md"

#: The scoring/keeping constants of salvage.py,
#: filed here and judged live against the module
#: by the checker — a silent constant change in
#: salvage.py fails this slice's tests by name.
SALVAGE_CONSTANTS: dict[str, Any] = {
    "length_weight": 0.45,
    "citation_weight": 0.55,
    "ideal_length_chars": 550,
    "fragment_ideal_length_chars": 200,
    "max_refine_passes": 2,
    "min_fragment_score": 0.54,
    "min_fragment_words": 4,
    "max_fragments_in_refine": 5,
    "top_drafts_for_fragments": 2,
    "no_qids_citation_score": 0.5,
    "fragment_cited_score": 0.8,
    "fragment_uncited_score": 0.4,
}

# ── The rules ───────────────────────────────────────────────────────────────

SALVAGE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "selection_is_scoring_not_repair",
     "rule": "the no-engine path selects the best-scoring draft and "
             "returns it verbatim; selection never edits text"},
    {"id": "citation_outweighs_length",
     "rule": "composite = 0.45*length + 0.55*citation, so a short "
             "cited draft beats a long uncited one (fixture: "
             "0.627 cited-concise vs 0.431 rambling)"},
    {"id": "fragments_feed_refine_only",
     "rule": "extracted fragments are inputs to refine passes; "
             "they never clean the returned text on their own"},
    {"id": "refine_adopts_only_on_improvement",
     "rule": "a refine pass replaces the current best only when "
             "its composite score is strictly higher"},
    {"id": "refine_path_names_attempts",
     "rule": "salvage_path refine_pass_N / passes_used count "
             "passes attempted, including a pass whose output "
             "was not adopted (TF-097 owns the record fix)"},
    {"id": "refine_failure_falls_back",
     "rule": "an engine exception breaks the refine loop; with "
             "zero completed passes forge() falls back to the "
             "best draft instead of raising"},
    {"id": "absent_strategies_are_gaps_not_claims",
     "rule": "truncation repair, rambling compression, format "
             "rescue, loop cutting and language-drift rescue do "
             "not exist in salvage.py; they are filed as "
             "D-SALV-1 with Part V owners, never implied"},
)

# ── The strategies that exist ───────────────────────────────────────────────
# anchor = a literal the checker requires in
# salvage.py's source, so a strategy cannot stay
# catalogued after its code is removed.

STRATEGIES: tuple[dict[str, Any], ...] = (
    {"id": "draft_scoring_selection",
     "where": "FragmentSalvage._score_draft / forge step 1",
     "anchor": "def _score_draft",
     "mechanism": "each candidate scored length (chars/550, "
                  "capped 1.0) * 0.45 + citation (share of the "
                  "bundle's QIDs present in the text) * 0.55; "
                  "with no retrieved QIDs citation is neutral 0.5; "
                  "drafts sorted by composite, best first",
     "output": "the ranked drafts; forge proceeds with the best",
     "paths": ("best_draft", "refine_pass_1", "refine_pass_2")},
    {"id": "fragment_extraction",
     "where": "FragmentSalvage._extract_fragments / _score_fragment",
     "anchor": "def _extract_fragments",
     "mechanism": "sentences from the top 2 drafts only; a "
                  "sentence needs >=4 words, is deduped by exact "
                  "text, and is kept when its fragment score "
                  "(length chars/200 * 0.45 + citation 0.8 cited / "
                  "0.4 uncited / 0.5 no-QIDs * 0.55) is >= 0.54; "
                  "at most 5 kept, best first",
     "output": "the kept fragment texts, feeding refine passes "
               "only — the returned text is unchanged by them",
     "paths": ()},
    {"id": "refine_passes",
     "where": "FragmentSalvage._refine_with_engine",
     "anchor": "def _refine_with_engine",
     "mechanism": "only when engine AND prompt_builder AND "
                  "sketch AND scaffold are all present; up to "
                  "max_passes (default 2, profile-resolved by "
                  "TF-037) generations at temperature 0.50 - "
                  "0.05*(pass-1), seeded per TF-036 when a seed "
                  "is given; a pass is adopted only when its "
                  "score beats the current best, and adopted "
                  "text re-seeds the next pass's fragments; an "
                  "exception breaks the loop",
     "output": "the best text seen (a pass's or the original "
               "draft's), path refine_pass_<passes attempted>",
     "paths": ("refine_pass_1", "refine_pass_2")},
    {"id": "best_draft_fallback",
     "where": "FragmentSalvage.forge step 4",
     "anchor": 'salvage_path="best_draft"',
     "mechanism": "no engine surface, or refine returned None "
                  "(zero passes completed): the best-scoring "
                  "draft is returned byte-identical, with its "
                  "citations and composite as confidence",
     "output": "the best draft verbatim",
     "paths": ("best_draft",)},
    {"id": "empty_candidates_fallback",
     "where": "FragmentSalvage.forge, candidates empty",
     "anchor": 'salvage_path="empty"',
     "mechanism": "no candidates at all: one fixed sentence, "
                  "no citations, confidence 0.0, no scoring",
     "output": "'The forge finds no candidates to salvage.'",
     "paths": ("empty",)},
)

# ── The strategies that do NOT exist (D-SALV-1) ─────────────────────────────

ABSENT_STRATEGIES: tuple[dict[str, Any], ...] = (
    {"id": "truncation_repair",
     "owner": "TF-090",
     "evidence": "a draft ending mid-sentence ('…rings across "
                 "the') is returned byte-identical as best_draft "
                 "(fixture truncated_mid_sentence)"},
    {"id": "rambling_compression",
     "owner": "TF-089",
     "evidence": "a lone rambling draft is returned "
                 "byte-identical; a cited rival beats rambling "
                 "only by selection, never by compressing it "
                 "(fixtures rambling_beaten_by_cited_rival, "
                 "rambling_alone_verbatim)"},
    {"id": "format_rescue",
     "owner": "TF-090",
     "evidence": "a broken JSON-shaped draft is returned "
                 "byte-identical; no parse or repair is "
                 "attempted (fixture broken_json_verbatim)"},
    {"id": "repetition_loop_cut",
     "owner": "TF-091",
     "evidence": "a sentence repeated 20x is returned "
                 "byte-identical; dedup applies to the fragment "
                 "list only, never to output text (fixture "
                 "loop_verbatim)"},
    {"id": "language_drift_rescue",
     "owner": "TF-092",
     "evidence": "no language detection or drift check exists "
                 "anywhere in salvage.py (source sweep this "
                 "tick); no fixture can exercise an absent "
                 "check, so none is filed for it"},
)

# ── Fixtures of broken input ────────────────────────────────────────────────
# candidates: the drafts handed to forge(), in
# order. expected_text: the exact text forge()
# must return on the no-engine path (all fixtures
# here were measured live this tick).

_CONCISE_CITED = (
    "According to Q123, the ferry leaves at dawn when the "
    "tide bell rings across the harbour mouth.")
_RAMBLING = " ".join(
    ["The river is nice and the river is nice and the river "
     "is nice and"] * 8)
_MIXED = (
    "Ok. Yes indeed. According to Q123, the ferry leaves at "
    "dawn when the tide bell rings across the harbour mouth "
    "and the keeper lights the red lantern for the last "
    "crossing tonight. Short.")
_BASE_CITED = (
    "According to Q123, the ferry leaves at dawn when the "
    "tide bell rings.")
_BETTER = (
    "According to Q123, the ferry leaves at dawn when the "
    "tide bell rings across the harbour mouth and the keeper "
    "lights the red lantern for the last crossing of the "
    "night before the winter timetable begins at first light.")

BROKEN_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "empty_candidates",
     "strategy": "empty_candidates_fallback",
     "candidates": (), "qids": (),
     "expected_path": "empty",
     "expected_text": "The forge finds no candidates to salvage.",
     "expected_fragments_kept": 0,
     "engine_required": False},
    {"id": "truncated_mid_sentence",
     "strategy": "truncation_repair",
     "candidates": ("The ferry leaves at dawn and the tide "
                    "bell rings across the",),
     "qids": ("Q123",),
     "expected_path": "best_draft",
     "expected_text": "The ferry leaves at dawn and the tide "
                      "bell rings across the",
     "expected_fragments_kept": 0,
     "engine_required": False},
    {"id": "rambling_beaten_by_cited_rival",
     "strategy": "draft_scoring_selection",
     "candidates": (_RAMBLING, _CONCISE_CITED),
     "qids": ("Q123",),
     "expected_path": "best_draft",
     "expected_text": _CONCISE_CITED,
     "expected_fragments_kept": 2,
     "engine_required": False},
    {"id": "rambling_alone_verbatim",
     "strategy": "rambling_compression",
     "candidates": (_RAMBLING,),
     "qids": ("Q123",),
     "expected_path": "best_draft",
     "expected_text": _RAMBLING,
     "expected_fragments_kept": 1,
     "engine_required": False},
    {"id": "fragment_filter_keeps_only_scored_sentences",
     "strategy": "fragment_extraction",
     "candidates": (_MIXED,),
     "qids": ("Q123",),
     "expected_path": "best_draft",
     "expected_text": _MIXED,
     "expected_fragments_kept": 1,
     "engine_required": False},
    {"id": "broken_json_verbatim",
     "strategy": "format_rescue",
     "candidates": ('{"ferry": "dawn", "tide": ',),
     "qids": ("Q123",),
     "expected_path": "best_draft",
     "expected_text": '{"ferry": "dawn", "tide": ',
     "expected_fragments_kept": 0,
     "engine_required": False},
    {"id": "loop_verbatim",
     "strategy": "repetition_loop_cut",
     "candidates": (" ".join(["The bell rings."] * 20),),
     "qids": ("Q123",),
     "expected_path": "best_draft",
     "expected_text": " ".join(["The bell rings."] * 20),
     "expected_fragments_kept": 0,
     "engine_required": False},
    {"id": "refine_adopts_better_output",
     "strategy": "refine_passes",
     "candidates": (_BASE_CITED,),
     "qids": ("Q123",),
     "expected_path": "refine_pass_1",
     "expected_text": _BETTER,
     "expected_fragments_kept": 1,
     "engine_required": True,
     "engine_text": _BETTER},
    {"id": "refine_engine_failure_falls_back",
     "strategy": "refine_passes",
     "candidates": (_BASE_CITED,),
     "qids": ("Q123",),
     "expected_path": "best_draft",
     "expected_text": _BASE_CITED,
     "expected_fragments_kept": 1,
     "engine_required": True,
     "engine_text": None},
)

_FIXTURE_BY_ID = {f["id"]: f for f in BROKEN_FIXTURES}


def run_fixture(fixture_id: str, *, engine: Any = None,
                prompt_builder: Any = None, sketch: Any = None,
                scaffold: Any = None, max_passes: int = 1):
    """Run one filed fixture through the real
    ``FragmentSalvage`` and return its SalvageResult.
    Engine fixtures need the caller's engine /
    prompt_builder / sketch / scaffold."""
    from thoughtforge.knowledge.models import (
        ActivatedRecord,
        CandidateRecord,
        CandidateScores,
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
        activated_records=[
            ActivatedRecord(qid, "entity", 0.8, f"cue for {qid}",
                            raw={"qid": qid})
            for qid in fixture["qids"]
        ],
        total_retrieved=len(fixture["qids"]),
        retrieval_confidence=0.8,
    )
    return FragmentSalvage().forge(
        candidates, bundle, engine=engine,
        prompt_builder=prompt_builder, sketch=sketch,
        scaffold=scaffold, max_passes=max_passes)


# ── Checkers ────────────────────────────────────────────────────────────────

def salvage_map_problems() -> list[str]:
    """Data-integrity checker: ids unique, every
    strategy judged against salvage.py's live
    constants and source anchors, every absent
    strategy owned, every fixture well-formed and
    pointing at a catalogued (present or absent)
    strategy."""
    from thoughtforge.refinement import salvage as salvage_module

    problems: list[str] = []
    rule_ids = [r["id"] for r in SALVAGE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate SALVAGE_RULES ids")
    strat_ids = [s["id"] for s in STRATEGIES]
    if len(strat_ids) != len(set(strat_ids)):
        problems.append("duplicate STRATEGIES ids")
    absent_ids = [a["id"] for a in ABSENT_STRATEGIES]
    if len(absent_ids) != len(set(absent_ids)):
        problems.append("duplicate ABSENT_STRATEGIES ids")
    if set(strat_ids) & set(absent_ids):
        problems.append("a strategy is filed as both present and absent")

    live_constants = {
        "length_weight": salvage_module._LENGTH_WEIGHT,
        "citation_weight": salvage_module._CITATION_WEIGHT,
        "ideal_length_chars": salvage_module._IDEAL_LENGTH_CHARS,
        "max_refine_passes": salvage_module._MAX_REFINE_PASSES,
        "min_fragment_score": salvage_module._MIN_FRAGMENT_SCORE,
        "min_fragment_words": salvage_module._MIN_FRAGMENT_WORDS,
        "max_fragments_in_refine":
            salvage_module._MAX_FRAGMENTS_IN_REFINE,
    }
    for key, live in live_constants.items():
        if SALVAGE_CONSTANTS.get(key) != live:
            problems.append(
                f"constant {key}: filed {SALVAGE_CONSTANTS.get(key)} "
                f"!= live {live}")

    import inspect
    source = inspect.getsource(salvage_module)
    for row in STRATEGIES:
        if row["anchor"] not in source:
            problems.append(
                f"strategy {row['id']}: anchor {row['anchor']!r} "
                "not in salvage.py")
    for row in ABSENT_STRATEGIES:
        if not row["owner"].startswith("TF-"):
            problems.append(
                f"absent strategy {row['id']}: owner {row['owner']!r} "
                "is not a slice")
    known = set(strat_ids) | set(absent_ids)
    fixture_ids = [f["id"] for f in BROKEN_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate BROKEN_FIXTURES ids")
    for fixture in BROKEN_FIXTURES:
        if fixture["strategy"] not in known:
            problems.append(
                f"fixture {fixture['id']}: strategy "
                f"{fixture['strategy']!r} not catalogued")
        if fixture["expected_path"] not in (
                "empty", "best_draft", "refine_pass_1",
                "refine_pass_2"):
            problems.append(
                f"fixture {fixture['id']}: unexpected path "
                f"{fixture['expected_path']!r}")
    return problems


def salvage_map_live_problems() -> list[str]:
    """Live checker: every no-engine fixture is run
    through the real FragmentSalvage and must land
    exactly as filed (path, text, fragments kept).
    Engine fixtures are skipped here — the tests
    exercise them with a stub engine."""
    problems: list[str] = []
    for fixture in BROKEN_FIXTURES:
        if fixture["engine_required"]:
            continue
        result = run_fixture(fixture["id"])
        if result.salvage_path != fixture["expected_path"]:
            problems.append(
                f"fixture {fixture['id']}: path "
                f"{result.salvage_path} != {fixture['expected_path']}")
        if result.text != fixture["expected_text"]:
            problems.append(
                f"fixture {fixture['id']}: text differs from filed")
        if result.fragments_kept != fixture["expected_fragments_kept"]:
            problems.append(
                f"fixture {fixture['id']}: fragments_kept "
                f"{result.fragments_kept} != "
                f"{fixture['expected_fragments_kept']}")
    return problems


def salvage_as_data() -> dict[str, Any]:
    return {
        "format": SALVAGE_FORMAT,
        "salvage_doc": SALVAGE_DOC,
        "constants": dict(SALVAGE_CONSTANTS),
        "strategies": [dict(s) for s in STRATEGIES],
        "absent_strategies": [dict(a) for a in ABSENT_STRATEGIES],
        "fixtures": [dict(f) for f in BROKEN_FIXTURES],
        "rules": [dict(r) for r in SALVAGE_RULES],
        "proof": {
            "grade": "fixture",
            "live_model_broken_outputs": "not_proven",
        },
    }
