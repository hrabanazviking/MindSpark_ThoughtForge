"""
Human-shaped spot checks, as data (TF-130).

The roadmap's rule: *human-shaped spot checks: a
small set of transcripts Volmarr can read and
judge himself (the numbers' sanity anchor).*
Part VII's suites and panels (TF-122..TF-129)
score hundreds of scripted turns and file their
numbers as pinned baselines — but no artefact
in the estate ever put a handful of those very
transcripts in front of a human reader, in
readable form, with the machine's score beside
the question a human judgement answers. The
numbers had no sanity anchor: a scorer could
drift away from what a reader would say and
nothing in the tree would ever notice.

The measured starting state, probed live this
tick on this tree before anything was built:

  * **The estate stood clean, and nothing was
    readable in it.** ``baseline_problems()``
    returned [] and ``harness_problems()``
    returned [] over the ten pinned baselines —
    and a grep for any spot-check / transcript
    artefact across benchmarks / docs / tests /
    src found nothing: no module assembles suite
    cases into transcripts for a human judge,
    no doc publishes transcripts beside their
    machine scores. Every suite doc publishes
    *numbers about* its corpus; none publishes
    the corpus turns themselves for reading.
  * **The gap was never numbered.** The number
    **D-BENCH-15** was verified unused anywhere
    in benchmarks / docs / tests / src before
    this slice filed it (the D-BENCH-9..14
    pattern: a gap assigned by the roadmap
    without ever being numbered).

What this module files, as verifiable data:

  * ``TRANSCRIPT_SPECS`` — eight filed specs,
    each naming ONE real case from a real
    suite's filed corpus by locator (collection
    + index / case id), the suite and axis it
    anchors, the human-judgement prompt (what
    Volmarr should look for — the question his
    judgement answers), and the measured
    expectation (score + held) the suite's real
    scorer files for that case alone. **No
    corpus and no scorer is copied**: the turns
    are pulled live from the suites' own filed
    data and every score is recomputed by that
    suite's real per-case scorer, so editing a
    suite's corpus drifts this set and the
    checker names it.
  * The set spans five anchors — persona_suite
    (an identity turn that holds; a voice turn
    that slips into generic AI-speak), the
    memory_suite (a citation present and true;
    a citation present but FALSE — the value
    contradicts the record it cites), the
    depth_suite (a chain problem the direct road
    misses and the scaffolded road lands), the
    presence_suite (a 20-turn coherence saga
    that breaks from turn 15; a will-steering
    saga that holds) and TF-067's quant estate
    through TF-127's panel (a q4_k_m answer
    whose working looks plausible and whose
    total is wrong). Four transcripts hold and
    four do not, as filed — a sanity anchor that
    only ever showed passes would anchor
    nothing.
  * ``build_spot_checks()`` assembles the
    transcripts live; ``run_spot_checks()``
    files the verdict: PASS only on the FULL
    filed set with every transcript's
    recomputed score and held-shape matching
    its filed expectation; a partial selection
    files REVIEW however it scores; an empty
    selection files 0 transcripts / REVIEW —
    no vacuous pass.
  * ``render_spot_checks_markdown()`` renders
    the set as readable markdown — the very
    text filed in
    ``docs/architecture/SPOT_CHECKS.md``, which
    IS the spot-check set (Volmarr reads prose,
    not code), checked live by
    ``spot_checks_problems()``: every rendered
    turn must appear in the doc verbatim.

D-BENCH-15 is FIXED by this slice in the
measuring sense: the estate's numbers now have
a human-shaped anchor — a small, filed set of
real transcripts a human can read and judge,
each recomputed live by its own suite's real
scorer. What is NOT fixed, and is stated in
the filed limits: no human judgement has been
recorded yet (the set is the instrument, the
judgements are Volmarr's to make when he reads
it); the transcripts are the suites' scripted
fixture turns, not a live model's words; and
this slice's own harness baseline is judged by
this module's checker, NOT gated by TF-129's
regression gate (the gate's gated set is the
nine TF-121..TF-128 baselines, exactly as
TF-129 filed it — the spot-check baseline is
excluded from that set in the open there, the
same shape as the gate's own baseline).

Filed limits, stated plainly:

  * The transcripts are scripted synthetic
    fixture turns (TF-017) drawn verbatim from
    the suites' filed corpora — a live model's
    transcripts are NOT PROVEN and never
    claimed (TF-173).
  * Some source suites file scripted assistant
    answers without a user line (persona /
    memory): those transcripts carry the
    suite's filed fact as context instead of an
    invented user turn — no user text is
    fabricated anywhere in the set.
  * The set is small on purpose (8 transcripts,
    56 turns): it anchors judgement, it does not
    re-measure the suites; the suites' own
    baselines remain the measurements.
  * The mean score across the set mixes five
    different scorers and is filed as a
    descriptive summary only — the anchor is
    the per-transcript reading, never the mean.
  * No human judgement is recorded in this
    slice: the doc files the judgement prompts
    and a blank judgement line per transcript.

The human-readable write-up — which is itself
the spot-check set — is
``docs/architecture/SPOT_CHECKS.md``.
Proof grade: FIXTURE — synthetic scripted
transcripts through the real suite scorers on
this VM; no live model, never claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPOT_CHECKS_FORMAT = "thoughtforge-spot-checks/1"
SPOT_CHECKS_DOC = "docs/architecture/SPOT_CHECKS.md"

#: The new baseline this slice registers in the
#: harness — the set's own pin. It is NOT one of
#: TF-129's gated baselines (the gate gates the
#: nine TF-121..TF-128 baselines, as filed; this
#: baseline is judged by this module's checker,
#: the shape TF-129 filed for its own baseline).
SPOT_CHECKS_BASELINE_ID = "spot_checks_synthetic_v1"

# The real machinery, reused — never copied: the
# suites' filed corpora and their real per-case
# scorers, and TF-067's fixture generation +
# scorer through its filed prompt set.
from benchmarks import depth_suite as _depth  # noqa: E402
from benchmarks import memory_suite as _memory  # noqa: E402
from benchmarks import persona_suite as _persona  # noqa: E402
from benchmarks import presence_suite as _presence  # noqa: E402
from thoughtforge.inference.quant_quality import (  # noqa: E402
    QUANT_PROMPTS,
    fixture_generate,
    score_answer,
)


# ── The filed set: eight specs over real cases ──────────────────────────────
#
# Each spec locates ONE case in a suite's own
# filed corpus and files the judgement prompt +
# the measured expectation (score / held for
# that case alone, through the suite's real
# scorer, measured live at banking 2026-11-01).
# The turns themselves are never copied here —
# build_spot_checks() pulls them live, and the
# checker re-derives them from the source
# collections and names any drift.

TRANSCRIPT_SPECS: tuple[dict[str, Any], ...] = (
    {"id": "sc01_persona_identity_holds",
     "title": "The crossing time, held plainly",
     "suite": "persona_suite", "axis": "identity_facts",
     "source": {"kind": "persona_identity",
                "collection": "IDENTITY_TURNS", "index": 0},
     "case_ref": "IDENTITY_TURNS[0] (fact crossing_minutes)",
     "judgement_prompt":
         "Read the answer against the filed fact above it. "
         "Does it say the same thing, in the persona's own "
         "plain voice — nothing added, nothing softened?",
     "expected_score": 1.0, "expected_held": True},
    {"id": "sc02_persona_voice_generic_slip",
     "title": "The voice slips into generic AI-speak",
     "suite": "persona_suite", "axis": "voice",
     "source": {"kind": "persona_voice",
                "collection": "VOICE_TURNS", "index": 30},
     "case_ref": "VOICE_TURNS[30] (a deliberately generic line)",
     "judgement_prompt":
         "Does this line sound like the skald of the earlier "
         "suites, or like a generic assistant apologising? "
         "The machine flagged it — do your ears agree?",
     "expected_score": 0.3, "expected_held": False},
    {"id": "sc03_memory_citation_present_and_true",
     "title": "A memory answer that cites its record",
     "suite": "memory_suite", "axis": "citations_present",
     "source": {"kind": "memory_present",
                "collection": "PRESENT_CASES",
                "case_id": "p01_ferry_chunk_cited"},
     "case_ref": "PRESENT_CASES p01_ferry_chunk_cited",
     "judgement_prompt":
         "Is the citation actually attached to the claim it "
         "backs, and does the claim match the filed fact? "
         "Would you trust this answer in a real conversation?",
     "expected_score": 1.0, "expected_held": True},
    {"id": "sc04_memory_citation_present_but_false",
     "title": "A citation that is present — and false",
     "suite": "memory_suite", "axis": "citations_true",
     "source": {"kind": "memory_true",
                "collection": "TRUE_CASES",
                "case_id": "t10_cited_but_false_value"},
     "case_ref": "TRUE_CASES t10_cited_but_false_value",
     "judgement_prompt":
         "The citation looks right at a glance. Check the "
         "value it backs against the filed fact: is the "
         "cited claim actually true?",
     "expected_score": 0.0, "expected_held": False},
    {"id": "sc05_depth_scaffold_lands_the_sum",
     "title": "A sum the direct road misses, the scaffolded road lands",
     "suite": "depth_suite", "axis": "chain_of_thought",
     "source": {"kind": "depth_chain",
                "collection": "CHAIN_CASES",
                "case_id": "c01_precedence_chain"},
     "case_ref": "CHAIN_CASES c01_precedence_chain",
     "judgement_prompt":
         "Follow the scaffolded working step by step. Is "
         "each step right, and does the final answer "
         "follow — or did the working just look busy?",
     "expected_score": 1.0, "expected_held": True},
    {"id": "sc06_presence_coherence_breaks_at_15",
     "title": "A long conversation that drifts from turn 15",
     "suite": "presence_suite",
     "axis": "long_conversation_coherence",
     "source": {"kind": "presence_coherence",
                "collection": "COHERENCE_SAGAS",
                "case_id": "k07_drift_from_15_break"},
     "case_ref": "COHERENCE_SAGAS k07_drift_from_15_break "
                 "(TF-028's filed 20-turn script)",
     "judgement_prompt":
         "Read the whole twenty turns. When do you first "
         "feel the answers stop matching what was "
         "established — and is that where the machine says "
         "the drift starts (turn 15)?",
     "expected_score": 0.7, "expected_held": False},
    {"id": "sc07_presence_goal_keeps_steering",
     "title": "A standing goal that keeps steering",
     "suite": "presence_suite", "axis": "will_steering",
     "source": {"kind": "presence_steering",
                "collection": "STEERING_SAGAS",
                "case_id": "w01_declared_goal_steers"},
     "case_ref": "STEERING_SAGAS w01_declared_goal_steers",
     "judgement_prompt":
         "Does each answer keep the standing goal in view "
         "without letting it swallow the actual question "
         "being asked?",
     "expected_score": 1.0, "expected_held": True},
    {"id": "sc08_quant_plausible_working_wrong_total",
     "title": "Plausible working, wrong total (quant q4_k_m)",
     "suite": "quant_quality", "axis": "quant_fixture",
     "source": {"kind": "quant_fixture",
                "collection": "QUANT_PROMPTS",
                "case_id": "fare_maths", "quant": "q4_k_m"},
     "case_ref": "QUANT_PROMPTS fare_maths at q4_k_m "
                 "(TF-067's fixture corpus, via TF-127)",
     "judgement_prompt":
         "The working looks confident. Do the numbers in "
         "it agree with each other — and with the final "
         "total?",
     "expected_score": 0.0, "expected_held": False},
)

#: The anchors the filed set covers, in spec order.
SPOT_CHECK_SUITES: tuple[str, ...] = tuple(
    dict.fromkeys(s["suite"] for s in TRANSCRIPT_SPECS))

TOTAL_TRANSCRIPTS = len(TRANSCRIPT_SPECS)


def _case_by_id(cases: Any, case_id: str) -> dict[str, Any]:
    return [c for c in cases if c["id"] == case_id][0]


# ── Per-kind builders: turns + score from the real scorers ──────────────────

def _build_persona_identity(spec: dict[str, Any]) -> dict[str, Any]:
    turn = _persona.IDENTITY_TURNS[spec["source"]["index"]]
    fact = [f for f in _persona.IDENTITY_FACTS
            if f["id"] == turn["fact"]][0]
    scored = _persona.score_identity_turns([turn])
    return {
        "context": f"Filed fact this answer must hold: {fact['fact']}.",
        "turns": [{"speaker": "Assistant", "text": turn["text"]}],
        "score": scored["score"],
        "held": scored["held"] == 1,
        "detail": {"axis_held": scored["held"],
                   "axis_turns": scored["turns"],
                   "fact": turn["fact"],
                   "filed_holds": turn["holds"]},
    }


def _build_persona_voice(spec: dict[str, Any]) -> dict[str, Any]:
    text = _persona.VOICE_TURNS[spec["source"]["index"]]
    scored = _persona.score_voice_turns([text])
    return {
        "context": "One scripted voice line, scored alone by the "
                   "suite's real persona scorer (the full voice "
                   "axis runs 36 lines).",
        "turns": [{"speaker": "Assistant", "text": text}],
        "score": scored["score"],
        "held": scored["score"] >= _persona.AXIS_PASS_THRESHOLD,
        "detail": {"flagged_turns": scored["flagged_turns"],
                   "generic_phrase_hits": scored["generic_phrase_hits"],
                   "norse_tone_hits": scored["norse_tone_hits"],
                   "skald_quality_hits": scored["skald_quality_hits"],
                   "pass_threshold": _persona.AXIS_PASS_THRESHOLD},
    }


def _build_memory_present(spec: dict[str, Any]) -> dict[str, Any]:
    case = _case_by_id(_memory.PRESENT_CASES, spec["source"]["case_id"])
    fact = [f for f in _memory.MEMORY_FACTS
            if f["id"] == case["fact"]][0]
    scored = _memory.score_present_cases([case])
    return {
        "context": f"Filed fact this answer must cite: {fact['fact']}.",
        "turns": [{"speaker": "Assistant", "text": case["text"]}],
        "score": scored["score"],
        "held": scored["held"] == 1,
        "detail": {"axis_held": scored["held"],
                   "records": list(case["records"]),
                   "filed_holds": case["holds"]},
    }


def _build_memory_true(spec: dict[str, Any]) -> dict[str, Any]:
    case = _case_by_id(_memory.TRUE_CASES, spec["source"]["case_id"])
    fact = [f for f in _memory.MEMORY_FACTS
            if f["id"] == case["fact"]][0]
    scored = _memory.score_true_cases([case])
    truth = _memory.citation_truth(
        case["text"], _memory._records(case["records"]))
    return {
        "context": f"Filed fact this answer must cite truly: "
                   f"{fact['fact']}.",
        "turns": [{"speaker": "Assistant", "text": case["text"]}],
        "score": scored["score"],
        "held": scored["held"] == 1,
        "detail": {"axis_held": scored["held"],
                   "records": list(case["records"]),
                   "filed_holds": case["holds"],
                   "citation_true": truth["true"],
                   "cited_markers": truth["cited_markers"]},
    }


def _build_depth_chain(spec: dict[str, Any]) -> dict[str, Any]:
    case = _case_by_id(_depth.CHAIN_CASES, spec["source"]["case_id"])
    scored = _depth.score_chain_cases([case])
    return {
        "context": f"Known answer: {case['expected']} "
                   f"(strategy {case['strategy']}).",
        "turns": [
            {"speaker": "User", "text": case["question"]},
            {"speaker": "Assistant (direct, scaffolds off)",
             "text": case["off_answer"]},
            {"speaker": "Assistant (scaffolded working)",
             "text": case["working"]},
            {"speaker": "Assistant (scaffolded answer)",
             "text": case["on_answer"]},
        ],
        "score": scored["on_score"],
        "held": scored["on_held"] == 1,
        "detail": {"off_score": scored["off_score"],
                   "on_score": scored["on_score"],
                   "delta": scored["delta"],
                   "expected": case["expected"],
                   "filed_off_holds": case["off_holds"],
                   "filed_on_holds": case["on_holds"]},
    }


def _build_presence_coherence(spec: dict[str, Any]) -> dict[str, Any]:
    saga = _case_by_id(
        _presence.COHERENCE_SAGAS, spec["source"]["case_id"])
    transcript = _presence.transcript_for_saga(saga)
    report = _presence.coherence_saga_report(saga)
    turns: list[dict[str, Any]] = []
    for user, assistant in transcript:
        turns.append({"speaker": "User", "text": user})
        turns.append({"speaker": "Assistant", "text": assistant})
    return {
        "context": f"TF-028's filed 20-turn script; this saga "
                   f"answers with the script's drifted text from "
                   f"turn {saga['drift_onset']} on.",
        "turns": turns,
        "score": report["mean_score"],
        "held": report["held"],
        "detail": {"turns": report["turns"],
                   "early_mean": report["early_mean"],
                   "late_mean": report["late_mean"],
                   "drift": report["drift"],
                   "contradiction_turns": report["contradiction_turns"],
                   "tf028_verdict": report["verdict"],
                   "filed_holds": saga["holds"]},
    }


def _build_presence_steering(spec: dict[str, Any]) -> dict[str, Any]:
    saga = _case_by_id(
        _presence.STEERING_SAGAS, spec["source"]["case_id"])
    judged = _presence.judge_will_saga(saga)
    turns: list[dict[str, Any]] = []
    for turn in saga["turns"]:
        turns.append({"speaker": "User", "text": turn["user"]})
        turns.append({"speaker": "Assistant", "text": turn["assistant"]})
    return {
        "context": f"Filed end state: the goal "
                   f"{saga['end_active']!r} still stands at the "
                   f"end, nothing closed.",
        "turns": turns,
        "score": 1.0 if judged["held"] else 0.0,
        "held": judged["held"],
        "detail": {"violations": judged["violations"],
                   "end_active": judged["end_active"],
                   "end_closed": judged["end_closed"],
                   "filed_holds": saga["holds"]},
    }


def _build_quant_fixture(spec: dict[str, Any]) -> dict[str, Any]:
    prompt = [p for p in QUANT_PROMPTS
              if p.prompt_id == spec["source"]["case_id"]][0]
    answer = fixture_generate(spec["source"]["quant"], prompt)
    scored = score_answer(prompt, answer)
    return {
        "context": f"TF-067 fixture prompt (kind {prompt.kind}) "
                   f"answered at quant level "
                   f"{spec['source']['quant']}; the reference "
                   f"answer scores 1.0 by construction.",
        "turns": [
            {"speaker": "User", "text": prompt.text},
            {"speaker": "Assistant", "text": answer},
        ],
        "score": scored.score,
        "held": scored.score == 1.0,
        "detail": {"quant": spec["source"]["quant"],
                   "kind": prompt.kind,
                   "retention": scored.retention,
                   "contradictions": list(scored.contradictions),
                   "reference": prompt.reference},
    }


_BUILDERS = {
    "persona_identity": _build_persona_identity,
    "persona_voice": _build_persona_voice,
    "memory_present": _build_memory_present,
    "memory_true": _build_memory_true,
    "depth_chain": _build_depth_chain,
    "presence_coherence": _build_presence_coherence,
    "presence_steering": _build_presence_steering,
    "quant_fixture": _build_quant_fixture,
}


# ── The set, built live ─────────────────────────────────────────────────────

def build_spot_checks(
    specs: Any = None,
) -> list[dict[str, Any]]:
    """Assemble the filed transcripts live: every
    turn pulled from its suite's filed corpus,
    every score recomputed by that suite's real
    per-case scorer. Nothing is copied."""
    specs = TRANSCRIPT_SPECS if specs is None else tuple(specs)
    transcripts: list[dict[str, Any]] = []
    for spec in specs:
        built = _BUILDERS[spec["source"]["kind"]](spec)
        transcripts.append({
            "id": spec["id"],
            "title": spec["title"],
            "suite": spec["suite"],
            "axis": spec["axis"],
            "case_ref": spec["case_ref"],
            "context": built["context"],
            "turns": built["turns"],
            "score": built["score"],
            "held": built["held"],
            "verdict": "HOLDS" if built["held"] else "DOES NOT HOLD",
            "expected_score": spec["expected_score"],
            "expected_held": spec["expected_held"],
            "judgement_prompt": spec["judgement_prompt"],
            "detail": built["detail"],
        })
    return transcripts


@dataclass
class SpotChecksResult:
    """One full spot-check run: the set + verdict data."""

    transcripts: list[dict[str, Any]] = field(default_factory=list)
    total_turns: int = 0
    holding: int = 0
    failing: int = 0
    mean_score: float = 0.0

    @property
    def total_transcripts(self) -> int:
        return len(self.transcripts)

    @property
    def suites_covered(self) -> int:
        return len({t["suite"] for t in self.transcripts})

    @property
    def mismatches(self) -> int:
        return sum(
            1 for t in self.transcripts
            if t["held"] != t["expected_held"]
            or t["score"] != t["expected_score"])

    @property
    def verdict(self) -> str:
        """PASS only on the full filed set with
        every transcript reproducing its filed
        expectation; FAIL on a full set with a
        mismatch; REVIEW for any partial or empty
        selection, however it scores."""
        if [t["id"] for t in self.transcripts] != [
                s["id"] for s in TRANSCRIPT_SPECS]:
            return "REVIEW"
        return "FAIL" if self.mismatches else "PASS"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_spot_checks(
    transcript_ids: Any = None,
) -> SpotChecksResult:
    """Build the selection live and file its verdict.

    Any explicit transcript selection may be
    given — including an empty one, which builds
    0 transcripts and files REVIEW (no vacuous
    pass).
    """
    if transcript_ids is None:
        specs = TRANSCRIPT_SPECS
    else:
        wanted = tuple(transcript_ids)
        specs = tuple(
            s for s in TRANSCRIPT_SPECS if s["id"] in wanted)
    transcripts = build_spot_checks(specs)
    total_turns = sum(len(t["turns"]) for t in transcripts)
    holding = sum(1 for t in transcripts if t["held"])
    mean = (
        sum(t["score"] for t in transcripts) / len(transcripts)
        if transcripts else 0.0)
    return SpotChecksResult(
        transcripts=transcripts, total_turns=total_turns,
        holding=holding, failing=len(transcripts) - holding,
        mean_score=mean)


# ── The readable render ─────────────────────────────────────────────────────

def render_spot_checks_markdown(
    transcripts: Any = None,
) -> str:
    """Render the set as readable markdown — the
    text filed in SPOT_CHECKS.md. Every turn is
    rendered verbatim, one blockquote line per
    source line."""
    transcripts = build_spot_checks() if transcripts is None \
        else transcripts
    lines: list[str] = []
    for index, t in enumerate(transcripts, start=1):
        lines.append(f"### {index}. {t['title']} ({t['id']})")
        lines.append("")
        lines.append(
            f"**Anchors:** {t['suite']}, {t['axis']} axis — "
            f"case {t['case_ref']}.")
        lines.append("")
        lines.append(f"**Filed context:** {t['context']}")
        lines.append("")
        lines.append(
            f"**Machine score:** {t['score']} — "
            f"**verdict:** {t['verdict']}.")
        lines.append("")
        lines.append(f"**Judge this:** {t['judgement_prompt']}")
        lines.append("")
        for turn in t["turns"]:
            turn_lines = str(turn["text"]).splitlines() or [""]
            lines.append(
                f"> **{turn['speaker']}:** {turn_lines[0]}")
            for extra in turn_lines[1:]:
                lines.append(f"> {extra}")
        lines.append("")
        lines.append("**Your judgement:** _(not yet recorded)_")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ── The filing's rules ──────────────────────────────────────────────────────

SPOT_CHECK_RULES: tuple[dict[str, Any], ...] = (
    {"id": "transcripts_are_the_suites_own_cases_verbatim",
     "rule": "every transcript's turns are pulled "
             "live from the named suite's filed "
             "corpus by locator (TRANSCRIPT_SPECS) "
             "— no corpus text is copied into this "
             "module, and the checker re-derives "
             "every turn from the source collection "
             "and names any drift"},
    {"id": "scores_are_recomputed_by_the_real_scorers",
     "rule": "each transcript's machine score is "
             "recomputed by its own suite's real "
             "per-case scorer over that case alone "
             "(the persona / memory / depth axis "
             "scorers, TF-028's conversation scorer "
             "via the presence suite, TF-029's "
             "ledger judge, TF-067's score_answer) "
             "and must reproduce the filed "
             "expectation exactly"},
    {"id": "the_set_holds_passes_and_failures_side_by_side",
     "rule": "the filed set carries holding and "
             "non-holding transcripts in equal "
             "number (4 and 4): a sanity anchor "
             "that only showed passes would anchor "
             "nothing, so the failing shapes are "
             "filed by name, never hidden"},
    {"id": "every_transcript_carries_a_judgement_prompt",
     "rule": "each transcript files what the "
             "human judge should look for — the "
             "question their judgement answers "
             "(does it sound like the persona, is "
             "the citation actually true, does the "
             "working actually follow) — beside "
             "the machine score and verdict"},
    {"id": "no_user_text_is_fabricated",
     "rule": "suites that file scripted assistant "
             "answers without a user line (persona "
             "/ memory) render the assistant turn "
             "with the suite's filed fact as "
             "context; no user turn is invented "
             "anywhere in the set"},
    {"id": "pass_needs_the_full_set_reproducing",
     "rule": "the verdict is PASS only when all "
             "eight filed transcripts build and "
             "every one reproduces its filed score "
             "+ held expectation; a partial "
             "selection files REVIEW however it "
             "scores and an empty selection files "
             "0 transcripts / REVIEW"},
    {"id": "the_doc_is_the_set",
     "rule": "docs/architecture/SPOT_CHECKS.md "
             "files the rendered transcripts in "
             "full — the render is generated from "
             "this module and the checker requires "
             "every transcript id and every turn's "
             "text to appear in the doc verbatim; "
             "the doc is the spot-check set, not a "
             "description of it"},
    {"id": "fixtures_are_synthetic_only",
     "rule": "every transcript is a scripted "
             "synthetic fixture turn from the "
             "suites' TF-017 corpora — canon is "
             "never a benchmark input and data/ "
             "was never opened; a live model's "
             "transcripts are NOT PROVEN (TF-173)"},
    {"id": "the_spot_check_baseline_is_not_gated_by_tf129",
     "rule": "TF-129's regression gate gates the "
             "nine TF-121..TF-128 pinned baselines "
             "exactly as filed; this slice's own "
             "baseline (spot_checks_synthetic_v1) "
             "is judged by this module's checker, "
             "not gated by the gate — the shape "
             "TF-129 filed for its own baseline, "
             "with the exclusion filed in the open "
             "in regression_gate.py"},
)


# ── The live checker ────────────────────────────────────────────────────────

def _source_texts(spec: dict[str, Any]) -> list[str]:
    """Every source text a spec's turns must be
    drawn from, re-derived straight from the
    source collections (the fidelity leg)."""
    source = spec["source"]
    kind = source["kind"]
    if kind == "persona_identity":
        return [_persona.IDENTITY_TURNS[source["index"]]["text"]]
    if kind == "persona_voice":
        return [_persona.VOICE_TURNS[source["index"]]]
    if kind == "memory_present":
        return [_case_by_id(
            _memory.PRESENT_CASES, source["case_id"])["text"]]
    if kind == "memory_true":
        return [_case_by_id(
            _memory.TRUE_CASES, source["case_id"])["text"]]
    if kind == "depth_chain":
        case = _case_by_id(_depth.CHAIN_CASES, source["case_id"])
        return [case["question"], case["off_answer"],
                case["working"], case["on_answer"]]
    if kind == "presence_coherence":
        saga = _case_by_id(
            _presence.COHERENCE_SAGAS, source["case_id"])
        texts: list[str] = []
        for user, assistant in _presence.transcript_for_saga(saga):
            texts.extend([user, assistant])
        return texts
    if kind == "presence_steering":
        saga = _case_by_id(
            _presence.STEERING_SAGAS, source["case_id"])
        texts = []
        for turn in saga["turns"]:
            texts.extend([turn["user"], turn["assistant"]])
        return texts
    if kind == "quant_fixture":
        prompt = [p for p in QUANT_PROMPTS
                  if p.prompt_id == source["case_id"]][0]
        return [prompt.text,
                fixture_generate(source["quant"], prompt)]
    return []


def spot_checks_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # Shape: the filed set, its anchors, its balance.
    spec_ids = [s["id"] for s in TRANSCRIPT_SPECS]
    if len(spec_ids) != len(set(spec_ids)):
        problems.append("duplicate TRANSCRIPT_SPECS ids")
    if TOTAL_TRANSCRIPTS != 8 or len(TRANSCRIPT_SPECS) != 8:
        problems.append(
            f"the filed set files {len(TRANSCRIPT_SPECS)} "
            f"transcripts, not 8")
    if list(SPOT_CHECK_SUITES) != [
            "persona_suite", "memory_suite", "depth_suite",
            "presence_suite", "quant_quality"]:
        problems.append(
            f"SPOT_CHECK_SUITES drifted: {list(SPOT_CHECK_SUITES)}")
    filed_holding = sum(
        1 for s in TRANSCRIPT_SPECS if s["expected_held"])
    if filed_holding != 4 or len(TRANSCRIPT_SPECS) - filed_holding != 4:
        problems.append(
            f"the filed balance is {filed_holding} holding / "
            f"{len(TRANSCRIPT_SPECS) - filed_holding} not, not 4 / 4")
    rule_ids = [r["id"] for r in SPOT_CHECK_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate SPOT_CHECK_RULES ids")
    for needed in ("transcripts_are_the_suites_own_cases_verbatim",
                   "scores_are_recomputed_by_the_real_scorers",
                   "the_set_holds_passes_and_failures_side_by_side",
                   "every_transcript_carries_a_judgement_prompt",
                   "no_user_text_is_fabricated",
                   "pass_needs_the_full_set_reproducing",
                   "the_doc_is_the_set",
                   "fixtures_are_synthetic_only",
                   "the_spot_check_baseline_is_not_gated_by_tf129"):
        if needed not in rule_ids:
            problems.append(f"SPOT_CHECK_RULES lost {needed}")
    for spec in TRANSCRIPT_SPECS:
        if not spec["judgement_prompt"].strip():
            problems.append(f"{spec['id']}: empty judgement prompt")
        if spec["source"]["kind"] not in _BUILDERS:
            problems.append(
                f"{spec['id']}: unknown source kind "
                f"{spec['source']['kind']}")

    # Build live: fidelity, scores, expectations.
    result = run_spot_checks()
    if result.verdict != "PASS" or not result.passes:
        problems.append(
            f"the live set does not file PASS: {result.verdict} "
            f"({result.mismatches} mismatches)")
    if result.total_transcripts != 8 or result.total_turns != 56:
        problems.append(
            f"the live set files {result.total_transcripts} "
            f"transcripts / {result.total_turns} turns, not 8 / 56")
    if result.holding != 4 or result.failing != 4:
        problems.append(
            f"the live set files {result.holding} holding / "
            f"{result.failing} not, not 4 / 4")
    if result.suites_covered != 5:
        problems.append(
            f"the live set covers {result.suites_covered} suites, "
            f"not 5")
    for transcript in result.transcripts:
        spec = [s for s in TRANSCRIPT_SPECS
                if s["id"] == transcript["id"]][0]
        if [t["text"] for t in transcript["turns"]] != \
                _source_texts(spec):
            problems.append(
                f"{transcript['id']}: turns drifted from the "
                f"source corpus (verbatim fidelity)")
        if transcript["score"] != spec["expected_score"]:
            problems.append(
                f"{transcript['id']}: recomputed score "
                f"{transcript['score']} != filed "
                f"{spec['expected_score']}")
        if transcript["held"] != spec["expected_held"]:
            problems.append(
                f"{transcript['id']}: recomputed held "
                f"{transcript['held']} != filed "
                f"{spec['expected_held']}")

    # Partial + empty selections file REVIEW.
    empty = run_spot_checks(transcript_ids=())
    if empty.total_transcripts != 0 or empty.verdict != "REVIEW" \
            or empty.passes:
        problems.append("an empty selection does not file 0 / REVIEW")
    if spec_ids:
        subset = run_spot_checks(transcript_ids=(spec_ids[0],))
        if subset.total_transcripts != 1 \
                or subset.verdict != "REVIEW":
            problems.append(
                "a single-transcript selection does not build "
                "its transcript / REVIEW mechanically")

    # The render + the doc are the set, verbatim.
    rendered = render_spot_checks_markdown(result.transcripts)
    for transcript in result.transcripts:
        if transcript["id"] not in rendered:
            problems.append(
                f"the render lost {transcript['id']}")
        for turn in transcript["turns"]:
            if turn["text"] not in rendered:
                problems.append(
                    f"the render lost a turn of {transcript['id']}")
    doc = Path(__file__).resolve().parent.parent / SPOT_CHECKS_DOC
    if not doc.exists():
        problems.append(f"the write-up {SPOT_CHECKS_DOC} is missing")
    else:
        doc_text = doc.read_text(encoding="utf-8")
        for transcript in result.transcripts:
            if transcript["id"] not in doc_text:
                problems.append(
                    f"the doc {SPOT_CHECKS_DOC} lost "
                    f"{transcript['id']}")
            for turn in transcript["turns"]:
                if turn["text"] not in doc_text:
                    problems.append(
                        f"the doc {SPOT_CHECKS_DOC} lost a turn "
                        f"of {transcript['id']} verbatim")

    # Harness consistency as filed + every pinned
    # baseline preserved exactly (this slice's
    # included: baseline_problems judges all
    # eleven), and the gate's gated set untouched.
    from benchmarks.harness import (
        BASELINES,
        BENCHMARKS,
        FIXTURES,
        baseline_problems,
    )
    from benchmarks.regression_gate import GATED_BASELINE_IDS

    if "spot_checks" not in [b["id"] for b in BENCHMARKS]:
        problems.append("BENCHMARKS lost the spot_checks entry")
    if "spot_checks_corpus" not in [f["id"] for f in FIXTURES]:
        problems.append("FIXTURES lost the spot_checks_corpus entry")
    if SPOT_CHECKS_BASELINE_ID not in [b["id"] for b in BASELINES]:
        problems.append("BASELINES lost the spot-check baseline")
    if SPOT_CHECKS_BASELINE_ID in GATED_BASELINE_IDS:
        problems.append(
            "the spot-check baseline entered TF-129's gated set "
            "(it is judged by this checker, not gated)")
    if len(GATED_BASELINE_IDS) != 9:
        problems.append("TF-129's gated set changed shape")
    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"harness baselines moved under the spot checks: "
            f"{earlier}")
    return problems


def spot_checks_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_spot_checks()
    return {
        "format": SPOT_CHECKS_FORMAT,
        "doc": SPOT_CHECKS_DOC,
        "suites": list(SPOT_CHECK_SUITES),
        "specs": [dict(s) for s in TRANSCRIPT_SPECS],
        "transcripts": result.transcripts,
        "rendered_markdown": render_spot_checks_markdown(
            result.transcripts),
        "measured": {
            "total_transcripts": result.total_transcripts,
            "total_turns": result.total_turns,
            "suites_covered": result.suites_covered,
            "holding_transcripts": result.holding,
            "failing_transcripts": result.failing,
            "mean_score": result.mean_score,
            "verdict": result.verdict,
        },
        "rules": [dict(r) for r in SPOT_CHECK_RULES],
    }


__all__ = [
    "SPOT_CHECKS_BASELINE_ID",
    "SPOT_CHECKS_DOC",
    "SPOT_CHECKS_FORMAT",
    "SPOT_CHECK_RULES",
    "SPOT_CHECK_SUITES",
    "TOTAL_TRANSCRIPTS",
    "TRANSCRIPT_SPECS",
    "SpotChecksResult",
    "build_spot_checks",
    "render_spot_checks_markdown",
    "run_spot_checks",
    "spot_checks_as_data",
    "spot_checks_problems",
]
