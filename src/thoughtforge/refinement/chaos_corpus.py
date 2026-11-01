"""TF-100 — The chaos corpus: a published fixture
set of real-shaped small-model failures — the
proving ground of Part V.

Roadmap: "The chaos corpus: a published fixture
set of real-shaped small-model failures
(truncations, loops, hallucinations, format
collapses) — the part's proving ground."
Consumed by TF-101 (the full policy stack run
over it, outcomes scored rescued / refused /
passed-clean), TF-142 and TF-104's gate.

Measured start (probed live this tick):

- **No chaos corpus existed.** Nothing in the
  tree is named chaos, and no single corpus
  collects the Part V failure modes in one
  scored shape. What existed was per-gate,
  per-slice, and smaller:
- **TF-089's measured corpus is 18 entries** —
  TF-087's nine ``BROKEN_FIXTURES`` (one broken
  shape per strategy gap, scored for salvage
  *rates*) plus TF-088's nine
  ``ADVERSARIAL_FIXTURES`` (refine-output
  fabrications, scored for corruption). Both
  live as tuples inside their modules, each
  entry shaped for its own slice's runner
  (candidate lists, engine texts), not as
  (prompt, broken output, expected outcome)
  rows a whole stack can be scored against.
- **TF-062's adversarial set is a different
  estate**: poisoned *memory records* (refuse /
  quarantine screening at the store boundary),
  not broken model *outputs* at all.
- **Each Part V gate filed its own fixtures**
  (TF-090 format + truncation, TF-091 loops,
  TF-092 drift, TF-096 voice, TF-093 policy) —
  deep per mechanism, but no entry anywhere
  carries the TF-101 outcome vocabulary, no
  corpus states which strategy owns each
  failure beside the prompt that produced it,
  and hallucination as a *draft-side* failure,
  generic filler non-answers, refusal-shaped
  non-answers and empty/short outputs have no
  shared fixture home at all.

What this slice adds beyond them, honestly:
**one published, versioned corpus of 38
entries** — the data filed as JSON at
``tests/fixtures/chaos_corpus.json`` (TF-017's
fixture home; the reference-library binding
already loads its estate from there), loaded
by this module, never hardcoded in tests.
Every entry is a (prompt, broken output) pair
in TF-017's synthetic Lantern Hollow world with
its expected outcome in TF-101's vocabulary —
``rescued`` / ``refused`` / ``passed-clean`` —
and the strategy/policy that owns it. Thirteen
families: the roadmap's four (truncations;
loops in sentence, cycle and word shapes;
hallucinations, with fabricated citations
filed as their own family; format collapses)
plus the other failure modes the Part V stack
handles (language drift, persona-voice loss,
genericness, refusal-shaped non-answers,
empty-or-short outputs) and clean controls
that must pass clean. TF-089's and TF-062's
corpora are untouched and unmerged: they
score different questions (rates, screening)
and this corpus does not restate them.

The outcome vocabulary, filed so TF-101 scores
against a stated standard, not a mood:

- ``rescued`` — an honest repair (deletion or
  re-delimiting only, TF-088's line) leaves a
  usable answer from the text already present.
- ``refused`` — the output must not be
  delivered as an answer: no honest repair
  exists, or what repair exists would invent
  content. A stack verdict of ``refused`` *or*
  a blocking ``flagged`` (whole/majority drift,
  length, genericness — the surface will not
  pass the text clean) satisfies this class in
  TF-101's scoring.
- ``passed-clean`` — the output is already
  sound and every gate must leave it
  byte-identical; a rescue applied to a clean
  control is corruption, not rescue.

An expected outcome is the corpus's *demand*,
not a claim that today's stack meets it.
TF-100 proves, live and cheaply, where the
existing gates already decide the filed way:
both truncations legs, every loop leg, the
format legs, the drift legs, the persona
legs' new violations, two citation legs, the
refusal/empty/short legs, and that every clean
control passes clean under its gate. The legs
no cheap gate decides — draft-side
hallucination beyond the citation-signalled
entries, generic filler, the invented-
reference citation — are filed with their
rationales and left for TF-101's scored stack
run to judge; this slice claims nothing about
them beyond their shape. TF-016: no docket
entry is owned by TF-100 and none is added —
the totals stand unchanged at 90.

Proof grade FIXTURE — synthetic real-shaped
failures loaded from the filed JSON and judged
by the real gates on this VM. A live model's
failures are NOT PROVEN (no model in tree),
never claimed: the corpus is shaped like real
failures, it is not a recording of one.

The human-readable write-up is
``docs/architecture/CHAOS_CORPUS.md``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

#: This corpus's estate format.
CHAOS_FORMAT = "thoughtforge-chaos-corpus/1"

CHAOS_DOC = "docs/architecture/CHAOS_CORPUS.md"

#: The corpus data file, repo-relative (TF-017's
#: fixture home — synthetic data lives there).
CHAOS_DATA_RELPATH = "tests/fixtures/chaos_corpus.json"

#: The outcome vocabulary TF-101 scores.
EXPECTED_OUTCOMES: tuple[str, ...] = (
    "rescued", "refused", "passed-clean")

#: The failure families, in corpus order.
FAMILIES: tuple[str, ...] = (
    "truncation",
    "loop_sentence",
    "loop_cycle",
    "loop_word",
    "hallucination",
    "fabricated_citation",
    "format_collapse",
    "language_drift",
    "persona_voice_loss",
    "genericness",
    "refusal_shaped",
    "empty_or_short",
    "clean_control",
)

#: Minimum entries per family, as data: no family
#: rests on a single example (minimum 2), the
#: roadmap-named families and the controls carry
#: more — truncations and format show both sides
#: of the repair/refuse line, loops cover three
#: shapes, hallucination covers both the marker
#: tier and TF-088's residual, and the controls
#: cover each gate family that could over-fire.
FAMILY_MINIMUMS: dict[str, int] = {
    "truncation": 3,
    "loop_sentence": 2,
    "loop_cycle": 2,
    "loop_word": 2,
    "hallucination": 3,
    "fabricated_citation": 2,
    "format_collapse": 3,
    "language_drift": 2,
    "persona_voice_loss": 2,
    "genericness": 2,
    "refusal_shaped": 2,
    "empty_or_short": 2,
    "clean_control": 4,
}

#: The owners an entry may name (the Part V
#: strategy/policy that owns its disposition).
KNOWN_OWNERS: tuple[str, ...] = (
    "TF-088", "TF-090", "TF-091", "TF-092",
    "TF-093", "TF-093/TF-095", "TF-096",
    "TF-049/TF-093",
)

#: Caller signals an entry may carry (TF-093's
#: surface vocabulary, plus the corpus's named
#: synthetic record/persona sets TF-101 resolves).
SIGNAL_KEYS: tuple[str, ...] = (
    "truncation_suspect", "structured",
    "required_keys", "expected_format",
    "expected_language", "records",
    "retrieved_qids", "persona",
)

#: Named synthetic signal targets (never canon).
SIGNAL_RECORD_SETS: tuple[str, ...] = (
    "lantern_hollow_synthetic",)
SIGNAL_PERSONAS: tuple[str, ...] = ("wrenna_synthetic",)

#: Strings that would mean canon leaked into the
#: corpus (TF-017's guard vocabulary: the canon
#: persona name and the repo canon tree).
FORBIDDEN_MARKERS: tuple[str, ...] = ("Sigrid", "data/")

# ── The rules ───────────────────────────────────────────────────────────────

CHAOS_RULES: tuple[dict[str, Any], ...] = (
    {"id": "synthetic_only_real_shaped",
     "rule": "every entry is invented in TF-017's "
             "Lantern Hollow world and marked synthetic "
             "in the data file; shapes are real-shaped "
             "(the failures small models actually make) "
             "but no text is a recording of a live model "
             "and no canon or persona material appears"},
    {"id": "one_row_per_failure_with_its_prompt",
     "rule": "each entry carries the prompt/ask it "
             "answers beside the broken output — a "
             "failure without its ask cannot be judged "
             "(a refusal shape is only a failure against "
             "a benign ask; drift only against the ask's "
             "language)"},
    {"id": "expected_outcome_is_tf101s_vocabulary",
     "rule": "every entry's expected_outcome is exactly "
             "one of rescued / refused / passed-clean; "
             "refused is satisfied in TF-101's scoring by "
             "a refused verdict or a blocking flagged "
             "verdict, never by a pass-clean"},
    {"id": "every_entry_names_its_owner",
     "rule": "each entry names the Part V strategy or "
             "policy surface that owns its disposition "
             "in KNOWN_OWNERS, with a rationale stating "
             "the mechanism — an unowned failure is a "
             "gap, filed in the open, not an entry"},
    {"id": "expectations_are_demands_not_claims",
     "rule": "an expected outcome states what the stack "
             "must do, not what it is proven to do; "
             "TF-100's live checker proves only the legs "
             "the existing gates decide cheaply, and "
             "TF-101's scored run owns the rest"},
    {"id": "clean_controls_are_part_of_the_corpus",
     "rule": "clean controls carry expected passed-clean "
             "and must survive every gate byte-identical; "
             "over-rescue is scored as failure, so the "
             "corpus guards both directions — broken "
             "text passed, and clean text cut"},
    {"id": "the_corpus_is_versioned_data",
     "rule": "the corpus lives in one JSON data file "
             "under the format id thoughtforge-chaos-"
             "corpus/1, loaded by this module; changing "
             "an entry's text or expectation is a corpus "
             "change in the open (TF-103's regression "
             "armour and TF-104's gate consume this "
             "version, never a copy)"},
    {"id": "prior_corpora_are_not_restated",
     "rule": "TF-087/TF-088's fixture sets (TF-089's "
             "rates corpus) and TF-062's poisoned-memory "
             "set score different questions and are "
             "neither merged nor recounted here; overlap "
             "of a failure shape is a family resemblance, "
             "not a duplicate entry"},
)

# ── Loading ─────────────────────────────────────────────────────────────────


def chaos_data_path() -> Path:
    """The corpus data file's absolute path
    (repo root = three parents above ``src/``)."""
    return Path(__file__).resolve().parents[3] / CHAOS_DATA_RELPATH


def load_chaos_corpus(
        path: Path | None = None) -> dict[str, Any]:
    """Load the filed corpus document fresh from
    disk (no cache: a reload must see the file
    exactly as it stands)."""
    source = path if path is not None else chaos_data_path()
    return json.loads(source.read_text(encoding="utf-8"))


def _load_entries() -> tuple[dict[str, Any], ...]:
    document = load_chaos_corpus()
    return tuple(document["entries"])


#: The corpus entries, loaded from the data file
#: at import — data, not code.
CHAOS_ENTRIES: tuple[dict[str, Any], ...] = _load_entries()

_ENTRY_BY_ID = {e["id"]: e for e in CHAOS_ENTRIES}


def entry_by_id(entry_id: str) -> dict[str, Any]:
    """One entry by id (a detached copy)."""
    return copy.deepcopy(_ENTRY_BY_ID[entry_id])


def entries_by_family(
        family: str) -> tuple[dict[str, Any], ...]:
    """The entries of one family, in corpus order
    (detached copies)."""
    return tuple(copy.deepcopy(e) for e in CHAOS_ENTRIES
                 if e["family"] == family)


def family_counts() -> dict[str, int]:
    """Live entry count per family, in FAMILIES
    order."""
    counts = {family: 0 for family in FAMILIES}
    for entry in CHAOS_ENTRIES:
        counts[entry["family"]] = counts.get(entry["family"], 0) + 1
    return counts


def outcome_counts() -> dict[str, int]:
    """Live entry count per expected outcome."""
    counts = {outcome: 0 for outcome in EXPECTED_OUTCOMES}
    for entry in CHAOS_ENTRIES:
        counts[entry["expected_outcome"]] = \
            counts.get(entry["expected_outcome"], 0) + 1
    return counts


# ── Checkers ────────────────────────────────────────────────────────────────


def chaos_corpus_problems(
        entries: tuple[dict[str, Any], ...] | None = None,
        document: dict[str, Any] | None = None) -> list[str]:
    """Data-integrity checker: the document is the
    filed format and marked synthetic; entry ids
    unique; every family in FAMILIES present at or
    above its filed minimum and no other family
    present; expected outcomes in the vocabulary,
    with clean controls exactly the passed-clean
    family and every other family carrying none;
    prompts and rationales non-empty, texts
    non-empty except the one filed empty entry;
    owners known; signals within the vocabulary
    and naming only the synthetic targets; the
    persona family carrying its source text; no
    canon marker anywhere."""
    problems: list[str] = []
    doc = document if document is not None else load_chaos_corpus()
    rows = CHAOS_ENTRIES if entries is None else entries
    if doc.get("format") != CHAOS_FORMAT:
        problems.append(
            f"corpus format {doc.get('format')!r} != {CHAOS_FORMAT}")
    if doc.get("synthetic") is not True:
        problems.append("corpus document is not marked synthetic")
    if not str(doc.get("world", "")).strip():
        problems.append("corpus document names no world")
    if tuple(doc.get("entries", ())) != tuple(CHAOS_ENTRIES) \
            and entries is None and document is None:
        problems.append("loaded entries differ from the data file")
    rule_ids = [r["id"] for r in CHAOS_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate CHAOS_RULES ids")
    if set(FAMILY_MINIMUMS) != set(FAMILIES):
        problems.append("FAMILY_MINIMUMS does not cover FAMILIES")
    ids = [e.get("id") for e in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate entry ids")
    counts: dict[str, int] = {}
    for entry in rows:
        counts[entry.get("family")] = \
            counts.get(entry.get("family"), 0) + 1
    if set(counts) != set(FAMILIES):
        problems.append(
            f"families present {sorted(counts)} != FAMILIES")
    for family, minimum in FAMILY_MINIMUMS.items():
        if counts.get(family, 0) < minimum:
            problems.append(
                f"family {family}: {counts.get(family, 0)} "
                f"entries below its minimum {minimum}")
    for entry in rows:
        eid = entry.get("id", "<no id>")
        if entry.get("family") not in FAMILIES:
            problems.append(f"{eid}: unknown family")
        if entry.get("expected_outcome") not in EXPECTED_OUTCOMES:
            problems.append(
                f"{eid}: expected_outcome "
                f"{entry.get('expected_outcome')!r} not in vocabulary")
        if entry.get("family") == "clean_control" and \
                entry.get("expected_outcome") != "passed-clean":
            problems.append(f"{eid}: a clean control must be passed-clean")
        if entry.get("family") != "clean_control" and \
                entry.get("expected_outcome") == "passed-clean":
            problems.append(
                f"{eid}: passed-clean outside the clean controls")
        if not str(entry.get("prompt", "")).strip():
            problems.append(f"{eid}: empty prompt")
        if not str(entry.get("rationale", "")).strip():
            problems.append(f"{eid}: empty rationale")
        text = entry.get("text")
        if not isinstance(text, str):
            problems.append(f"{eid}: text is not a string")
        elif not text.strip() and eid != "chaos_empty_output":
            problems.append(f"{eid}: empty text outside the filed empty entry")
        if entry.get("owner") not in KNOWN_OWNERS:
            problems.append(f"{eid}: unknown owner {entry.get('owner')!r}")
        signals = entry.get("signals")
        if not isinstance(signals, dict):
            problems.append(f"{eid}: signals is not an object")
        else:
            for key in signals:
                if key not in SIGNAL_KEYS:
                    problems.append(f"{eid}: unknown signal {key!r}")
            if "records" in signals and \
                    signals["records"] not in SIGNAL_RECORD_SETS:
                problems.append(f"{eid}: records names a non-synthetic set")
            if "persona" in signals and \
                    signals["persona"] not in SIGNAL_PERSONAS:
                problems.append(f"{eid}: persona names a non-synthetic persona")
            if "required_keys" in signals and \
                    not signals.get("structured"):
                problems.append(
                    f"{eid}: required_keys without a structured signal")
        if entry.get("family") == "persona_voice_loss":
            if not str(entry.get("source_text", "")).strip():
                problems.append(f"{eid}: persona entry carries no source_text")
            if signals.get("persona") not in SIGNAL_PERSONAS:
                problems.append(f"{eid}: persona entry names no synthetic persona")
        blob = json.dumps(entry, ensure_ascii=False)
        for marker in FORBIDDEN_MARKERS:
            if marker in blob:
                problems.append(f"{eid}: canon marker {marker!r} present")
    return problems


def chaos_corpus_live_problems() -> list[str]:
    """Live checker: the legs the existing gates
    decide cheaply land exactly as filed — the
    truncation, loop, format and drift gates on
    their families, the persona guard's new
    violations, the two mechanically decidable
    citation legs, the refusal/empty/short legs
    under TF-093's surface, and every clean
    control passed clean under its own gate
    (byte-identical where the gate returns text).
    The legs no cheap gate decides (draft-side
    hallucination beyond the signalled citation
    entries, generic filler, the invented
    reference) are deliberately not judged here —
    TF-101's scored stack run owns them."""
    problems: list[str] = []
    from thoughtforge.refinement.drift_rescue import rescue_drift
    from thoughtforge.refinement.format_enforcement import (
        enforce_format,
        repair_truncation,
    )
    from thoughtforge.refinement.loop_rescue import rescue_loops
    from thoughtforge.refinement.policy_surface import enforce_policy

    for entry in CHAOS_ENTRIES:
        eid, family = entry["id"], entry["family"]
        text, signals = entry["text"], entry["signals"]
        expected = entry["expected_outcome"]
        if family == "truncation":
            verdict = repair_truncation(text).verdict
            want = "repaired" if expected == "rescued" else "refused"
            if verdict != want:
                problems.append(
                    f"{eid}: truncation gate {verdict} != filed {want}")
        elif family in ("loop_sentence", "loop_cycle", "loop_word"):
            result = rescue_loops(text)
            if result.verdict != "rescued" or result.text == text:
                problems.append(
                    f"{eid}: loop gate {result.verdict}, filed rescued")
        elif family == "format_collapse":
            result = enforce_format(
                text,
                required_keys=tuple(signals.get("required_keys", ())),
                expected=signals.get("expected_format", "object"))
            want = "repaired" if expected == "rescued" else "refused"
            if result.verdict != want:
                problems.append(
                    f"{eid}: format gate {result.verdict} != filed {want}")
        elif family == "language_drift":
            result = rescue_drift(
                text, signals.get("expected_language", "en"))
            if expected == "rescued" and result.verdict != "rescued":
                problems.append(
                    f"{eid}: drift gate {result.verdict}, filed rescued")
            if expected == "refused" and result.verdict != "flagged":
                problems.append(
                    f"{eid}: drift gate {result.verdict}, filed "
                    "refused-side (flagged)")
        elif family == "persona_voice_loss":
            from thoughtforge.cognition.persona import (
                FIXTURE_EXPECTATION,
            )
            from thoughtforge.refinement.persona_voice import check_voice

            verdict = check_voice(
                FIXTURE_EXPECTATION, entry["source_text"], text)
            if verdict.verdict != "voice_broken" or \
                    not verdict.new_violations:
                problems.append(
                    f"{eid}: voice guard found no new violation "
                    f"({verdict.verdict})")
        elif family in ("refusal_shaped", "empty_or_short"):
            result = enforce_policy(text)
            if expected == "refused" and result.verdict == "clean":
                problems.append(
                    f"{eid}: policy surface passed a filed-refused "
                    "output clean")
            if eid == "chaos_empty_output" and \
                    result.verdict != "refused":
                problems.append(
                    f"{eid}: empty output not refused "
                    f"({result.verdict})")
        elif eid in ("chaos_citation_foreign_qid",
                     "chaos_citation_uncited_claim",
                     "chaos_clean_cited_claim"):
            from thoughtforge.refinement.citation_enforcement import (
                check_citations,
            )
            from thoughtforge.refinement.policy_surface import (
                _fixture_records,
            )

            check = check_citations(text, _fixture_records())
            if eid == "chaos_citation_foreign_qid" and \
                    "Q999999" not in check.foreign_markers:
                problems.append(f"{eid}: foreign marker not named")
            if eid == "chaos_citation_uncited_claim" and \
                    not check.uncited_markers:
                problems.append(f"{eid}: uncited claim not named")
            if eid == "chaos_clean_cited_claim" and not check.passed:
                problems.append(f"{eid}: cited control did not pass")
        elif family == "clean_control":
            if eid == "chaos_clean_structured_json":
                result = enforce_format(
                    text,
                    required_keys=tuple(signals.get("required_keys", ())),
                    expected=signals.get("expected_format", "object"))
                if result.verdict != "valid" or result.text != text:
                    problems.append(
                        f"{eid}: structured control not valid/untouched")
            elif eid == "chaos_clean_spanish_answer":
                result = rescue_drift(text, signals["expected_language"])
                if result.verdict != "clean" or result.text != text:
                    problems.append(
                        f"{eid}: Spanish control not clean/untouched")
            elif eid == "chaos_clean_emphasis_double_repeat":
                result = rescue_loops(text)
                if result.verdict != "clean" or result.text != text:
                    problems.append(
                        f"{eid}: emphasis control not clean/untouched")
            elif eid == "chaos_clean_prose":
                result = enforce_policy(text)
                if result.verdict != "clean" or result.text != text:
                    problems.append(
                        f"{eid}: prose control not clean/untouched")
    return problems


def chaos_as_data() -> dict[str, Any]:
    return {
        "format": CHAOS_FORMAT,
        "chaos_doc": CHAOS_DOC,
        "data_file": CHAOS_DATA_RELPATH,
        "rules": [dict(r) for r in CHAOS_RULES],
        "families": list(FAMILIES),
        "family_minimums": dict(FAMILY_MINIMUMS),
        "family_counts": family_counts(),
        "outcome_counts": outcome_counts(),
        "expected_outcomes": list(EXPECTED_OUTCOMES),
        "known_owners": list(KNOWN_OWNERS),
        "entries": copy.deepcopy(list(CHAOS_ENTRIES)),
        "proof": {
            "grade": "fixture",
            "live_model_failures": "not_proven",
            "tf101_scoring": "not_run_in_this_slice",
            "prior_corpora": "tf089_rates_and_tf062_screening_unchanged",
        },
    }
