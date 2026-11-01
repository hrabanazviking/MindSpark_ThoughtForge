"""TF-094 — Policy configuration: enforcement
strictness per conversation (strict / standard /
relaxed) with the defaults stated + tested.

Roadmap: "Policy configuration: enforcement
strictness per conversation (strict / standard /
relaxed) with the defaults stated + tested."

Measured start (probed live this tick):

- **TF-093's surface had exactly one disposition.**
  ``enforce_policy`` composes the seven checks and
  combines them refused > flagged > rescued >
  clean, and that was the whole policy: a flagged
  answer always left (annotated where the gate
  annotates), a refusal always refused. Nothing
  anywhere in the refinement estate named a
  strictness level, a default, or a way for one
  conversation to be held tighter or looser than
  another (greps for strict / standard / relaxed
  over refinement/ find only unrelated uses).
- **The gates' thresholds are not the config
  surface.** Length's five-word floor and the
  generic-phrase list live inside
  ``EnforcementGate``; TF-090/091/092's repair
  rules live inside their gates. A strictness
  level that silently retuned those would rewrite
  filed gates mid-Part — so this slice governs
  **disposition** (what a verdict *does*), never
  detection (what a check *finds*).

What this slice builds — ``enforce_configured``
in this module, a thin disposition layer over
TF-093's real surface:

- **THREE LEVELS, AS DATA (STRICTNESS_LEVELS /
  DISPOSITIONS).** ``strict`` / ``standard`` /
  ``relaxed``, each filed with what it does to
  each base verdict TF-093 can return.
- **THE DEFAULT IS STANDARD, STATED.**
  ``DEFAULT_STRICTNESS = "standard"``, and
  standard is TF-093 exactly — same verdict,
  same text, same flags, same problems, same
  check outcomes, proven leg-by-leg against
  ``enforce_policy`` itself in the live checker.
  A caller who names no level gets the shipped
  TF-093 behaviour, never a guess at another.
- **STRICTNESS IS PER CONVERSATION AND
  CALLER-SUPPLIED, NEVER GUESSED** (TF-093's
  signal rule, carried to configuration). One
  level applies to a whole conversation's run
  through the surface; the layer never infers a
  level from the draft, the ask, the user, or a
  hardware profile. An unknown level name is a
  caller bug and raises ``ValueError`` naming
  the value and the three real levels (the
  TF-015 precedent for a bad configuration
  value) — it is never silently mapped to the
  default.
- **STRICT BLOCKS EVERY FLAG.** Under strict, a
  base ``flagged`` verdict becomes ``refused``:
  no usable text leaves, the flags are preserved
  verbatim and carried as the refusal's named
  problems, and every check outcome stands
  exactly as TF-093 recorded it (a blocked flag
  is still recorded flagged, never rewritten to
  refused at check level). TF-095 owns telling
  the user that and why.
- **RELAXED DOWNGRADES FLAGS TO WARNINGS,
  NEVER HIDES THEM.** Under relaxed, a base
  ``flagged`` verdict takes the verdict the run
  earned underneath the flags — ``rescued`` if
  any check rescued, else ``clean`` — the text
  leaves exactly as TF-093 returned it (gate
  annotations included), and the flags move to
  the result's ``warnings``, verbatim. Nothing
  is deleted from the record: the check outcomes
  still say flagged, by name.
- **A REFUSAL IS NEVER RELAXED.** Empty output,
  an unrepairable structured answer, a truncation
  with no complete sentence, an unsupported
  expected language — there is no honest text
  to pass, at any level. Refused is refused
  under strict, standard and relaxed alike,
  with the same problems. Likewise a rescue is
  allowed at every level: a base ``rescued`` or
  ``clean`` verdict is identical at all three.
- **ONE LEVEL PER CONVERSATION, NO PER-CHECK
  OVERRIDES** (filed limit). A caller cannot
  relax citation while tightening length here;
  that fragmentation is not offered. TF-098
  owns cost caps, TF-095 refusal telling — this
  slice owns only the level and its disposition
  table.
- **D-DEAD-1, TF-094's share ruled:**
  ``utils.validators.validate_config`` is NOT
  wired into this configuration surface — it
  validates a backend dict (it requires the
  ``backend`` key), the wrong shape for a
  strictness level; this surface validates its
  own selections itself (TF-032's ruling shape
  for its own surface). The validators module
  is unchanged; the remaining functions' verdicts
  stand with their filed owners (TF-065 and the
  ingestion line), and D-DEAD-1 stays OPEN.

Proof grade FIXTURE — every filed case is run
live through the real TF-093 surface at all
three levels on this VM, and standard is proven
identical to the surface itself. A live model's
behaviour under any level is NOT PROVEN (no
model in tree); TF-101 runs the full stack over
TF-100's chaos corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thoughtforge.refinement.policy_surface import (
    POLICY_CHECKS,
    PolicyResult,
    enforce_policy,
)

CONFIG_FORMAT = "thoughtforge-policy-configuration/1"

CONFIG_DOC = "docs/architecture/POLICY_CONFIGURATION.md"

STRICTNESS_LEVELS: tuple[str, ...] = ("strict", "standard", "relaxed")

DEFAULT_STRICTNESS = "standard"

# ── The rules ───────────────────────────────────────────────────────────────

CONFIG_RULES: tuple[dict[str, Any], ...] = (
    {"id": "default_is_standard",
     "rule": "a caller who names no strictness level gets "
             "standard, which is TF-093's shipped behaviour "
             "exactly — the default is stated here, never "
             "inferred and never environment-dependent"},
    {"id": "strictness_is_per_conversation_caller_supplied",
     "rule": "one level applies to a whole conversation's "
             "run through the policy surface; the level is "
             "supplied by the caller and never guessed from "
             "the draft, the ask, the user, or a hardware "
             "profile (TF-093's signal rule, carried to "
             "configuration)"},
    {"id": "unknown_strictness_raises_by_name",
     "rule": "an unknown level name raises ValueError naming "
             "the value and the three real levels; it is "
             "never silently mapped to the default"},
    {"id": "standard_is_tf093_exactly",
     "rule": "under standard the configured result is "
             "TF-093's result — same verdict, text, flags, "
             "problems, notes and check outcomes"},
    {"id": "strict_blocks_every_flag",
     "rule": "under strict a base flagged verdict becomes "
             "refused: no usable text leaves, the flags are "
             "preserved verbatim and carried as the "
             "refusal's problems, and the check outcomes "
             "stand exactly as judged"},
    {"id": "relaxed_downgrades_flags_to_warnings",
     "rule": "under relaxed a base flagged verdict takes "
             "the verdict earned underneath the flags "
             "(rescued if any check rescued, else clean), "
             "the text leaves exactly as TF-093 returned "
             "it, and the flags move verbatim to warnings — "
             "recorded, never hidden, check outcomes "
             "unchanged"},
    {"id": "refusal_is_never_relaxed",
     "rule": "a base refused verdict is refused at every "
             "level with the same problems and no usable "
             "text — a refusal means there is no honest "
             "text to pass, and no level invents one"},
    {"id": "rescue_is_allowed_at_every_level",
     "rule": "a base rescued or clean verdict is identical "
             "at all three levels — strictness governs "
             "flagged and refused dispositions, not whether "
             "an honest repair may stand"},
    {"id": "strictness_governs_disposition_not_detection",
     "rule": "the levels change what a verdict does, never "
             "what a check finds: every underlying gate, "
             "threshold and repair rule is unchanged, and "
             "TF-093's surface itself is unchanged"},
    {"id": "one_level_no_per_check_overrides",
     "rule": "there are no per-check overrides in this "
             "slice: a conversation carries exactly one "
             "level, so the policy a conversation runs "
             "under can always be named in one word"},
    {"id": "validate_config_not_wired_d_dead_1",
     "rule": "D-DEAD-1 (TF-094 share): utils.validators."
             "validate_config is not wired into this "
             "surface — it validates a backend dict, the "
             "wrong shape for a strictness level; this "
             "surface validates its own selections itself, "
             "and the validators module is unchanged"},
)

# ── The disposition table ───────────────────────────────────────────────────

DISPOSITIONS: tuple[dict[str, Any], ...] = (
    {"level": "strict",
     "on_clean": "clean",
     "on_rescued": "rescued",
     "on_flagged": "refused",
     "on_refused": "refused",
     "summary": "every flag blocks: the answer does not "
                "leave, its flags are the refusal's named "
                "problems"},
    {"level": "standard",
     "on_clean": "clean",
     "on_rescued": "rescued",
     "on_flagged": "flagged",
     "on_refused": "refused",
     "summary": "the default — TF-093 exactly: a flagged "
                "answer leaves annotated, with every flag "
                "named"},
    {"level": "relaxed",
     "on_clean": "clean",
     "on_rescued": "rescued",
     "on_flagged": "warning",
     "on_refused": "refused",
     "summary": "flags become warnings: the answer leaves "
                "as judged underneath them, the flags "
                "recorded verbatim as warnings"},
)

_DISPOSITION = {d["level"]: d for d in DISPOSITIONS}

_VERDICTS = ("clean", "rescued", "flagged", "refused")

# ── Configuration + result types ────────────────────────────────────────────


@dataclass(frozen=True)
class PolicyConfig:
    """One conversation's enforcement configuration."""
    strictness: str = DEFAULT_STRICTNESS

    @property
    def is_default(self) -> bool:
        return self.strictness == DEFAULT_STRICTNESS


@dataclass
class ConfiguredResult:
    """TF-093's verdict, disposed under one level."""
    strictness: str                 # the level applied
    verdict: str                    # effective verdict
    base_verdict: str               # TF-093's verdict, before disposition
    disposition: str                # as_judged | blocked_flagged | downgraded_to_warnings
    text: str | None = None         # usable text; None when refused
    checks: tuple = ()              # TF-093 CheckOutcomes, unchanged
    flags: tuple[str, ...] = ()     # named, on flagged / blocked verdicts
    warnings: tuple[str, ...] = ()  # flags downgraded under relaxed
    problems: tuple[str, ...] = ()  # named, on refused verdicts
    notes: str = ""                 # the gate's notes, when the gate ran
    claim_check: Any = None         # TF-049 CitationCheck, when records ran

    def check(self, check_id: str):
        for outcome in self.checks:
            if outcome.id == check_id:
                return outcome
        raise KeyError(check_id)


def resolve_policy_config(
    strictness: str | None = None,
) -> PolicyConfig:
    """Resolve a caller-supplied level name to a
    ``PolicyConfig``. ``None`` is the stated
    default (standard). Names are matched
    case-insensitively with surrounding whitespace
    ignored, and the canonical level is recorded.
    An unknown name raises ``ValueError`` naming
    the value and the real levels — a caller bug
    is never silently defaulted."""
    if strictness is None:
        return PolicyConfig(DEFAULT_STRICTNESS)
    canonical = str(strictness).strip().lower()
    if canonical not in STRICTNESS_LEVELS:
        raise ValueError(
            f"unknown strictness {strictness!r}: expected one of "
            f"{', '.join(STRICTNESS_LEVELS)}")
    return PolicyConfig(canonical)


# ── The configured surface ──────────────────────────────────────────────────


def enforce_configured(
    text: str,
    *,
    strictness: str | None = None,
    config: PolicyConfig | None = None,
    **signals: Any,
) -> ConfiguredResult:
    """Judge one output through TF-093's real
    surface (``enforce_policy``, with the caller's
    signals passed through untouched) and dispose
    the verdict under one conversation's level.

    Give the level as ``strictness=`` or as a
    resolved ``config=``; giving both and having
    them disagree raises ``ValueError``."""
    if config is not None and strictness is not None:
        resolved = resolve_policy_config(strictness)
        if resolved.strictness != config.strictness:
            raise ValueError(
                f"strictness {strictness!r} disagrees with "
                f"config {config.strictness!r}: name one level")
        level = config
    elif config is not None:
        level = config
    else:
        level = resolve_policy_config(strictness)

    base: PolicyResult = enforce_policy(text, **signals)
    disposition_of = _DISPOSITION[level.strictness]

    if base.verdict == "refused":
        return ConfiguredResult(
            level.strictness, "refused", base.verdict, "as_judged",
            text=None, checks=base.checks, flags=base.flags,
            problems=base.problems, notes=base.notes,
            claim_check=base.claim_check)
    if base.verdict == "flagged":
        if disposition_of["on_flagged"] == "refused":
            return ConfiguredResult(
                level.strictness, "refused", base.verdict,
                "blocked_flagged", text=None, checks=base.checks,
                flags=base.flags, problems=tuple(base.flags),
                notes=base.notes, claim_check=base.claim_check)
        if disposition_of["on_flagged"] == "warning":
            earned = "rescued" if any(
                o.status == "rescued" for o in base.checks) else "clean"
            return ConfiguredResult(
                level.strictness, earned, base.verdict,
                "downgraded_to_warnings", text=base.text,
                checks=base.checks, warnings=tuple(base.flags),
                notes=base.notes, claim_check=base.claim_check)
        return ConfiguredResult(
            level.strictness, "flagged", base.verdict, "as_judged",
            text=base.text, checks=base.checks, flags=base.flags,
            notes=base.notes, claim_check=base.claim_check)
    # clean / rescued: identical at every level.
    return ConfiguredResult(
        level.strictness, base.verdict, base.verdict, "as_judged",
        text=base.text, checks=base.checks, flags=base.flags,
        notes=base.notes, claim_check=base.claim_check)


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic material throughout (TF-093's filed
# cases, reused verbatim — never canon).
# ``levels`` files the effective verdict at each
# level; ``expected_text`` is the text standard
# and relaxed return (strict returns the same
# text except on a blocked flag, where it
# returns None, exactly as a refusal does).

_EN1 = "The ferry leaves at dawn and the tide is low."
_EN2 = "The keeper lights the lantern when the ferry is late."
_ES1 = "El faro está en la costa y la marea es alta."
_ES2 = "La campana suena cuando el barco llega a la costa."
_BELL = "The bell rings across the harbour at dawn."
_ODIN_UNCITED = "Odin is the chief god of Norse mythology, a god of wisdom and war."
_WEATHER = "The weather over the fjord is calm and bright today."
_NO_CITE_NOTE = "\n\n[Forge: limited verified citations available for this response.]"
_TOO_SHORT_NOTE = "\n\n[Forge: response is shorter than expected — knowledge may be sparse.]"

def _levels(verdict: str, **overrides: str) -> dict[str, str]:
    levels = {"strict": verdict, "standard": verdict, "relaxed": verdict}
    levels.update(overrides)
    return levels


CONFIG_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "clean_prose_all_levels",
     "text": f"{_EN1} {_EN2}",
     "signals": {},
     "base_verdict": "clean",
     "levels": _levels("clean"),
     "expected_text": f"{_EN1} {_EN2}"},
    {"id": "loop_rescued_all_levels",
     "text": " ".join([_BELL] * 4),
     "signals": {},
     "base_verdict": "rescued",
     "levels": _levels("rescued"),
     "expected_text": _BELL},
    {"id": "drift_flagged_blocked_or_warned",
     "text": f"{_ES1} {_ES2}",
     "signals": {"expected_language": "en"},
     "base_verdict": "flagged",
     "levels": _levels("flagged", strict="refused", relaxed="clean"),
     "expected_text": f"{_ES1} {_ES2}",
     "flags": ("drift:whole_answer_off_language",)},
    {"id": "citation_flagged_blocked_or_warned",
     "text": _ODIN_UNCITED,
     "signals": {"records": "lantern"},
     "base_verdict": "flagged",
     "levels": _levels("flagged", strict="refused", relaxed="clean"),
     "expected_text": _ODIN_UNCITED + _NO_CITE_NOTE,
     "flags": ("citation:uncited_claim",)},
    {"id": "legacy_floor_flagged_blocked_or_warned",
     "text": _WEATHER,
     "signals": {"retrieved_qids": ["Q123"]},
     "base_verdict": "flagged",
     "levels": _levels("flagged", strict="refused", relaxed="clean"),
     "expected_text": _WEATHER + _NO_CITE_NOTE,
     "flags": ("citation:no_retrieved_qids_cited",)},
    {"id": "too_short_flagged_blocked_or_warned",
     "text": "Too short.",
     "signals": {},
     "base_verdict": "flagged",
     "levels": _levels("flagged", strict="refused", relaxed="clean"),
     "expected_text": "Too short." + _TOO_SHORT_NOTE,
     "flags": ("length:too_short",)},
    {"id": "generic_flagged_blocked_or_warned",
     "text": "As an AI language model I cannot provide that information today.",
     "signals": {},
     "base_verdict": "flagged",
     "levels": _levels("flagged", strict="refused", relaxed="clean"),
     "expected_text": "As an AI language model I cannot provide that information today.",
     "flags": ("genericness:hard_generic_phrase",)},
    {"id": "format_repaired_all_levels",
     "text": '{"ferry": "dawn",}',
     "signals": {"structured": True},
     "base_verdict": "rescued",
     "levels": _levels("rescued"),
     "expected_text": '{"ferry": "dawn"}'},
    {"id": "format_refused_never_relaxed",
     "text": "The ferry leaves at dawn.",
     "signals": {"structured": True},
     "base_verdict": "refused",
     "levels": _levels("refused"),
     "expected_text": None,
     "problems": ("no_json_span",)},
    {"id": "truncation_refused_never_relaxed",
     "text": "The keeper lights the lantern when",
     "signals": {"truncation_suspect": True},
     "base_verdict": "refused",
     "levels": _levels("refused"),
     "expected_text": None,
     "problems": ("no_complete_sentence",)},
    {"id": "empty_refused_never_relaxed",
     "text": "   ",
     "signals": {},
     "base_verdict": "refused",
     "levels": _levels("refused"),
     "expected_text": None,
     "problems": ("empty_output",)},
    {"id": "unsupported_language_refused_never_relaxed",
     "text": _EN1,
     "signals": {"expected_language": "fr"},
     "base_verdict": "refused",
     "levels": _levels("refused"),
     "expected_text": None,
     "problems": ("unsupported_expected_language",)},
)


def _fixture_records() -> list:
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


def _run_fixture(
    fixture: dict[str, Any], strictness: str,
) -> ConfiguredResult:
    signals = dict(fixture["signals"])
    if signals.get("records") == "lantern":
        signals["records"] = _fixture_records()
    return enforce_configured(
        fixture["text"], strictness=strictness, **signals)


# ── Checkers ────────────────────────────────────────────────────────────────


def policy_config_problems() -> list[str]:
    """Data-integrity checker: the levels are
    exactly the three named, the default is one
    of them and is standard, the disposition
    table covers every level and every base
    verdict, rule ids are unique, fixtures are
    well-formed (every level filed, refusals
    naming problems, flagged cases naming
    flags)."""
    problems: list[str] = []
    if STRICTNESS_LEVELS != ("strict", "standard", "relaxed"):
        problems.append("STRICTNESS_LEVELS drifted from the three named levels")
    if DEFAULT_STRICTNESS not in STRICTNESS_LEVELS:
        problems.append("DEFAULT_STRICTNESS is not a named level")
    if DEFAULT_STRICTNESS != "standard":
        problems.append("the stated default is no longer standard")
    if [d["level"] for d in DISPOSITIONS] != list(STRICTNESS_LEVELS):
        problems.append("DISPOSITIONS do not cover every level in order")
    for entry in DISPOSITIONS:
        for key in ("on_clean", "on_rescued", "on_flagged", "on_refused"):
            if entry.get(key) not in _VERDICTS + ("warning",):
                problems.append(
                    f"disposition {entry['level']}: {key} "
                    f"{entry.get(key)!r} not in the vocabulary")
        if entry["on_refused"] != "refused":
            problems.append(
                f"disposition {entry['level']}: a refusal is relaxed")
    rule_ids = [r["id"] for r in CONFIG_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate CONFIG_RULES ids")
    fixture_ids = [f["id"] for f in CONFIG_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate CONFIG_FIXTURES ids")
    for fixture in CONFIG_FIXTURES:
        if set(fixture["levels"]) != set(STRICTNESS_LEVELS):
            problems.append(
                f"fixture {fixture['id']}: levels do not cover "
                "every strictness level")
        for verdict in fixture["levels"].values():
            if verdict not in _VERDICTS:
                problems.append(
                    f"fixture {fixture['id']}: verdict "
                    f"{verdict!r} not in the vocabulary")
        if fixture["base_verdict"] == "refused" and not fixture.get("problems"):
            problems.append(
                f"fixture {fixture['id']}: a refusal must name "
                "its problems")
        if fixture["base_verdict"] == "flagged" and not fixture.get("flags"):
            problems.append(
                f"fixture {fixture['id']}: a flagged case must "
                "name its flags")
    return problems


def policy_config_live_problems() -> list[str]:
    """Live checker: every filed fixture is run
    through the real TF-093 surface at every
    level and must land exactly as filed —
    standard identical to ``enforce_policy``
    itself, strict blocking every filed flag
    (no text, flags carried as problems, check
    outcomes unchanged), relaxed downgrading
    every filed flag to a verbatim warning,
    refusals identical at every level — and the
    default resolution is standard while an
    unknown level raises by name."""
    problems: list[str] = []
    check_ids = tuple(c["id"] for c in POLICY_CHECKS)
    for fixture in CONFIG_FIXTURES:
        signals = dict(fixture["signals"])
        if signals.get("records") == "lantern":
            signals["records"] = _fixture_records()
        base = enforce_policy(fixture["text"], **signals)
        if base.verdict != fixture["base_verdict"]:
            problems.append(
                f"fixture {fixture['id']}: base verdict "
                f"{base.verdict} != {fixture['base_verdict']}")
            continue
        for level in STRICTNESS_LEVELS:
            result = _run_fixture(fixture, level)
            want = fixture["levels"][level]
            if result.verdict != want:
                problems.append(
                    f"fixture {fixture['id']} at {level}: verdict "
                    f"{result.verdict} != {want}")
            if result.strictness != level or \
                    result.base_verdict != base.verdict:
                problems.append(
                    f"fixture {fixture['id']} at {level}: level "
                    "or base verdict misrecorded")
            if tuple(o.id for o in result.checks) != check_ids or \
                    [o.status for o in result.checks] != \
                    [o.status for o in base.checks]:
                problems.append(
                    f"fixture {fixture['id']} at {level}: check "
                    "outcomes drifted from TF-093's")
            if level == "standard":
                if (result.verdict, result.text, result.flags,
                        result.problems) != (base.verdict, base.text,
                                             base.flags, base.problems):
                    problems.append(
                        f"fixture {fixture['id']}: standard is not "
                        "TF-093 exactly")
            if fixture["base_verdict"] == "flagged":
                if level == "strict":
                    if result.text is not None or \
                            result.problems != tuple(fixture["flags"]) or \
                            result.flags != tuple(fixture["flags"]) or \
                            result.disposition != "blocked_flagged":
                        problems.append(
                            f"fixture {fixture['id']}: strict does "
                            "not block the flag as filed")
                if level == "relaxed":
                    if result.text != fixture["expected_text"] or \
                            result.warnings != tuple(fixture["flags"]) or \
                            result.flags != () or \
                            result.disposition != "downgraded_to_warnings":
                        problems.append(
                            f"fixture {fixture['id']}: relaxed does "
                            "not downgrade the flag as filed")
            if fixture["base_verdict"] == "refused":
                if result.text is not None or \
                        result.problems != tuple(fixture["problems"]):
                    problems.append(
                        f"fixture {fixture['id']} at {level}: a "
                        "refusal was relaxed or renamed")
            if fixture["base_verdict"] in ("clean", "rescued"):
                if result.text != fixture["expected_text"]:
                    problems.append(
                        f"fixture {fixture['id']} at {level}: text "
                        "drifted across levels")
    if resolve_policy_config().strictness != DEFAULT_STRICTNESS:
        problems.append("the default resolution is not the stated default")
    if resolve_policy_config(" Strict ").strictness != "strict":
        problems.append("level names are not canonicalised as filed")
    try:
        resolve_policy_config("severe")
        problems.append("an unknown strictness level did not raise")
    except ValueError as exc:
        if "severe" not in str(exc) or "strict" not in str(exc):
            problems.append("the unknown-level error does not name its value")
    return problems


def policy_config_as_data() -> dict[str, Any]:
    return {
        "format": CONFIG_FORMAT,
        "config_doc": CONFIG_DOC,
        "levels": list(STRICTNESS_LEVELS),
        "default": DEFAULT_STRICTNESS,
        "rules": [dict(r) for r in CONFIG_RULES],
        "dispositions": [dict(d) for d in DISPOSITIONS],
        "fixtures": [
            {k: v for k, v in f.items()} for f in CONFIG_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_behaviour": "not_proven",
            "policy_surface": "unchanged",
            "enforcement_gate": "unchanged",
            "validate_config": "not_wired_by_ruling_tf094_d_dead_1",
        },
    }
