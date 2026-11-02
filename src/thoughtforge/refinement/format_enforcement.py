"""TF-090 — Format enforcement: structured outputs
(JSON-shaped asks) repaired or refused honestly,
per policy; prose truncation cut to the last
complete sentence or refused.

Roadmap: "Format enforcement: structured outputs
(JSON-shaped asks) repaired or refused honestly,
per policy." TF-016 assigns TF-090 D-SALV-1's
truncation-repair and format-rescue shares.

Measured start (probed live this tick):

- **Nothing in the refinement package could
  judge a structured answer at all.** TF-087
  probed the broken-JSON draft
  ``{"ferry": "dawn", "tide": `` returned
  byte-identical by ``FragmentSalvage`` — and
  TF-087's filings pin that behaviour: salvage
  has no signal that an ask was JSON-shaped, so
  it must not guess. The enforcement therefore
  lives here, as a gate a caller invokes *when
  the ask was structured* — the same shape as
  TF-049's citation enforcement beside
  ``EnforcementGate``. TF-093 owns unifying the
  surfaces; TF-095 owns telling the user.
  ``FragmentSalvage`` is untouched by this
  slice, so TF-089's filed metrics (rescue 2/7,
  unrescued verbatim 5) stand unchanged.
- **The repair line is TF-088's line, applied
  to structure**: a repair may only remove or
  re-delimit what is already there — extract a
  JSON span out of prose or a fence, drop a
  trailing comma, re-encode a Python literal,
  close containers a truncation left open, and
  drop an *incomplete* trailing element (a cut
  key, a key with no value, a cut string). A
  repair may never complete a value, invent a
  key, or guess a missing required field: the
  moment honesty would require invention, the
  verdict is **refused**, with the reason named.
- **Truncation of prose gets the same rule.**
  A draft ending mid-sentence is cut back to
  its last complete sentence (the dropped tail
  is recorded verbatim); a draft with no
  complete sentence at all is refused — ending
  it would mean inventing its ending.

Proof grade FIXTURE — every filed case is run
live through the real gate functions on this
VM. A live model's structured outputs are NOT
PROVEN (no model in tree); TF-101 scores this
gate in the full stack over TF-100's corpus.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

FORMAT_FORMAT = "thoughtforge-format-enforcement/1"

FORMAT_DOC = "docs/architecture/FORMAT_ENFORCEMENT.md"

# ── The rules ───────────────────────────────────────────────────────────────

FORMAT_RULES: tuple[dict[str, Any], ...] = (
    {"id": "repair_never_invents",
     "rule": "a repair only removes or re-delimits content "
             "already present (extraction, trailing commas, "
             "literal re-encoding, container closure, dropping "
             "an incomplete trailing element); completing a "
             "cut value or inventing a key is fabrication "
             "(TF-088's line) and is never done"},
    {"id": "refusal_is_a_verdict_not_a_shrug",
     "rule": "when no honest repair exists the gate refuses: "
             "verdict 'refused', text None, and every problem "
             "named (no JSON span, multiple spans, unparseable, "
             "missing required key, wrong top-level shape) — "
             "the broken text is never passed through as if "
             "it were structured output"},
    {"id": "every_repair_is_recorded",
     "rule": "a repaired verdict carries the ordered names of "
             "the repairs applied and, for a truncation drop, "
             "the dropped tail verbatim — a consumer can see "
             "exactly what the gate changed (TF-097 owns "
             "carrying the record into the trace)"},
    {"id": "schema_is_a_gate_not_a_repair_target",
     "rule": "required keys and the expected top-level shape "
             "are checked after any repair; a missing required "
             "key refuses the output — the gate never invents "
             "the missing field to make the schema pass"},
    {"id": "valid_passes_untouched",
     "rule": "output that already parses and satisfies the "
             "schema is returned byte-identical with verdict "
             "'valid' and no repairs"},
    {"id": "prose_truncation_cut_or_refuse",
     "rule": "a prose draft ending mid-sentence is cut to its "
             "last complete sentence (dropped tail recorded); "
             "with no complete sentence the verdict is refused "
             "— an ending is never invented"},
    {"id": "the_gate_is_caller_invoked",
     "rule": "the gate runs when the ask was structured; "
             "FragmentSalvage has no such signal and is "
             "unchanged — TF-093 owns the unified policy "
             "surface, TF-095 the user-facing refusal"},
)

# ── Result types ────────────────────────────────────────────────────────────


@dataclass
class FormatResult:
    """Verdict of the format gate on one output."""
    verdict: str                    # "valid" | "repaired" | "refused"
    text: str | None = None         # usable JSON text; None when refused
    parsed: Any = None              # the parsed structure, when usable
    repairs: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    dropped_tail: str = ""          # verbatim, when a truncation drop cut text


@dataclass
class TruncationResult:
    """Verdict on one prose draft's ending."""
    verdict: str                    # "intact" | "repaired" | "refused"
    text: str | None = None         # usable prose; None when refused
    dropped_tail: str = ""
    problems: tuple[str, ...] = ()


# ── JSON plumbing ───────────────────────────────────────────────────────────


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-JSON constant {name}")


def _loads(text: str) -> Any:
    """Strict JSON parse (NaN/Infinity are not JSON)."""
    return json.loads(text, parse_constant=_reject_constant)


def _try_loads(text: str) -> tuple[bool, Any]:
    try:
        return True, _loads(text)
    except (ValueError, TypeError):
        return False, None


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _drop_trailing_commas(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = _TRAILING_COMMA_RE.sub(r"\1", current)
    return current


def _python_literal(text: str) -> str | None:
    """Re-encode a Python-literal structure as JSON,
    when its content is exactly JSON-safe data."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(value, (dict, list)):
        return None

    def safe(v: Any) -> bool:
        if v is None or isinstance(v, (str, int, float, bool)):
            return True
        if isinstance(v, list):
            return all(safe(item) for item in v)
        if isinstance(v, dict):
            return all(isinstance(k, str) and safe(val)
                       for k, val in v.items())
        return False

    if not safe(value):
        return None
    return json.dumps(value, ensure_ascii=False)


# ── The truncation scanner ──────────────────────────────────────────────────
# Walks the text once, tracking the container stack
# and each frame's expectation (key / colon / value /
# comma). A *safe point* is a position where the text
# ends exactly at a complete value (or a just-opened
# container): cutting there and closing the recorded
# stack yields valid JSON that keeps every complete
# element and drops only the incomplete tail.


@dataclass
class _Frame:
    opener: str                     # "{" or "["
    expecting: str                  # key | colon | value | comma


def _scan(text: str) -> dict[str, Any] | None:
    """Scan once; None when the structure is invalid
    (mismatched closer, value where a key belongs…)
    rather than merely truncated."""
    stack: list[_Frame] = []
    safe: list[tuple[int, tuple[str, ...]]] = []
    in_string = False
    escaped = False
    token_start: int | None = None
    i = 0
    n = len(text)

    def closers(frames: list[_Frame]) -> tuple[str, ...]:
        return tuple("}" if f.opener == "{" else "]"
                     for f in reversed(frames))

    def note_value_end(pos: int) -> None:
        safe.append((pos, closers(stack)))
        if stack:
            stack[-1].expecting = "comma"

    while i < n:
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                if stack and stack[-1].opener == "{" and \
                        stack[-1].expecting == "key":
                    stack[-1].expecting = "colon"
                else:
                    note_value_end(i + 1)
            i += 1
            continue
        if token_start is not None:
            if ch in ",}] \t\r\n":
                token = text[token_start:i]
                if token not in ("true", "false", "null") and \
                        not _is_number(token):
                    return None
                note_value_end(i)
                token_start = None
                # fall through: the delimiter is processed below
            else:
                i += 1
                continue
        if ch == '"':
            if stack and stack[-1].expecting in ("colon", "comma"):
                return None
            in_string = True
            i += 1
            continue
        if ch in "{[":
            if stack and stack[-1].expecting in ("colon", "comma"):
                return None
            stack.append(_Frame(
                ch, "key" if ch == "{" else "value"))
            safe.append((i + 1, closers(stack)))
            i += 1
            continue
        if ch in "}]":
            if not stack:
                return None
            frame = stack[-1]
            if (ch == "}") != (frame.opener == "{"):
                return None
            if frame.expecting in ("colon",):
                return None
            stack.pop()
            note_value_end(i + 1)
            i += 1
            continue
        if ch == ",":
            if not stack or stack[-1].expecting != "comma":
                return None
            stack[-1].expecting = \
                "key" if stack[-1].opener == "{" else "value"
            i += 1
            continue
        if ch == ":":
            if not stack or stack[-1].expecting != "colon":
                return None
            stack[-1].expecting = "value"
            i += 1
            continue
        if ch in " \t\r\n":
            i += 1
            continue
        # a bare scalar token begins
        if stack and stack[-1].expecting != "value":
            return None
        if not stack:
            return None
        token_start = i
        i += 1

    end_state: dict[str, Any] = {
        "in_string": in_string,
        "stack": stack,
        "safe": safe,
        "token_start": token_start,
    }
    if token_start is not None and not in_string:
        token = text[token_start:n]
        if token in ("true", "false", "null") or _is_number(token):
            note_value_end(n)
            end_state["token_complete"] = True
        else:
            end_state["token_complete"] = False
    return end_state


def _is_number(token: str) -> bool:
    try:
        json.loads(token)
        return isinstance(json.loads(token), (int, float))
    except ValueError:
        return False


def _close_truncated(text: str) -> tuple[str, list[str], str] | None:
    """Repair a truncated JSON text, or None.

    Returns (repaired_text, repair_names, dropped_tail).
    Only the incomplete tail is dropped; only open
    containers are closed. Nothing is completed."""
    state = _scan(text)
    if state is None:
        return None
    in_string = state["in_string"]
    stack: list[_Frame] = state["stack"]
    token_start = state["token_start"]
    truncated = (
        in_string or bool(stack)
        or (token_start is not None and not state.get("token_complete"))
        or (stack and stack[-1].expecting in ("colon", "value")))
    if not truncated:
        return None
    if not state["safe"]:
        return None
    pos, closing = state["safe"][-1]
    repaired = text[:pos] + "".join(closing)
    ok, _ = _try_loads(repaired)
    if not ok:
        return None
    repairs: list[str] = []
    dropped = text[pos:]
    if dropped:
        repairs.append("dropped_incomplete_tail")
    if closing:
        repairs.append("closed_open_containers")
    if not repairs:
        return None
    return repaired, repairs, dropped


def _top_level_spans(text: str) -> int:
    """Count disjoint top-level JSON spans (objects /
    arrays) in the text, string-aware."""
    count = 0
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            if depth == 0:
                count += 1
            depth += 1
        elif ch in "}]":
            depth = max(0, depth - 1)
    return count


# ── Schema ──────────────────────────────────────────────────────────────────


def _schema_problems(parsed: Any, required_keys: tuple[str, ...],
                     expected: str) -> list[str]:
    problems: list[str] = []
    if expected == "object" and not isinstance(parsed, dict):
        problems.append(
            f"wrong_top_level_shape: expected object, "
            f"got {type(parsed).__name__}")
    elif expected == "array" and not isinstance(parsed, list):
        problems.append(
            f"wrong_top_level_shape: expected array, "
            f"got {type(parsed).__name__}")
    if isinstance(parsed, dict):
        for key in required_keys:
            if key not in parsed:
                problems.append(f"missing_required_key: {key}")
    elif required_keys:
        problems.append("missing_required_key: " +
                        ", ".join(required_keys))
    return problems


# ── The gate ────────────────────────────────────────────────────────────────


def _attempt_repairs(text: str) -> tuple[str, list[str], str] | None:
    """The ordered repair ladder on one candidate
    string. Returns (usable_text, repairs, dropped)
    or None when no honest repair parses."""
    repairs: list[str] = []

    no_commas = _drop_trailing_commas(text)
    if no_commas != text:
        ok, _ = _try_loads(no_commas)
        if ok:
            return no_commas, ["dropped_trailing_comma"], ""
        text = no_commas
        repairs.append("dropped_trailing_comma")

    literal = _python_literal(text)
    if literal is not None:
        ok, _ = _try_loads(literal)
        if ok:
            return literal, repairs + ["converted_python_literal"], ""

    closed = _close_truncated(text)
    if closed is not None:
        repaired, names, dropped = closed
        return repaired, repairs + names, dropped
    return None


def enforce_format(text: str, *,
                   required_keys: tuple[str, ...] = (),
                   expected: str = "object") -> FormatResult:
    """Judge one structured (JSON-shaped) output.

    Verdicts: ``valid`` (parses + schema, untouched),
    ``repaired`` (an honest repair made it usable;
    every repair named, any dropped tail verbatim),
    ``refused`` (no honest repair — problems named,
    text None; the broken output is never passed
    through as structured)."""
    raw = (text or "").strip()
    if not raw:
        return FormatResult("refused", problems=("empty_output",))

    ok, parsed = _try_loads(raw)
    if ok:
        problems = _schema_problems(parsed, required_keys, expected)
        if problems:
            return FormatResult("refused", parsed=parsed,
                                problems=tuple(problems))
        return FormatResult("valid", text=raw, parsed=parsed)

    if _top_level_spans(raw) > 1:
        return FormatResult("refused",
                            problems=("multiple_json_spans",))

    # Candidate strings, in order: the whole text,
    # then an extraction (fence first, else the
    # outermost span) when one exists.
    candidates: list[tuple[str, str | None]] = [(raw, None)]
    fence = _FENCE_RE.search(raw)
    if fence and fence.group(1).strip() != raw:
        candidates.append((fence.group(1).strip(),
                           "extracted_from_fence"))
    else:
        first = min((idx for idx in (raw.find("{"), raw.find("["))
                     if idx >= 0), default=-1)
        last = max(raw.rfind("}"), raw.rfind("]"))
        if first > 0 and last > first:
            candidates.append((raw[first:last + 1],
                               "extracted_from_prose"))
        elif first == 0 and 0 <= last < len(raw) - 1:
            candidates.append((raw[:last + 1],
                               "extracted_from_prose"))

    for candidate, extraction in candidates:
        if extraction is not None:
            ok_direct, parsed_direct = _try_loads(candidate)
            if ok_direct:
                problems = _schema_problems(
                    parsed_direct, required_keys, expected)
                if problems:
                    return FormatResult(
                        "refused", parsed=parsed_direct,
                        repairs=(extraction,),
                        problems=tuple(problems))
                return FormatResult(
                    "repaired", text=candidate, parsed=parsed_direct,
                    repairs=(extraction,))
        attempt = _attempt_repairs(candidate)
        if attempt is None:
            continue
        usable, repairs, dropped = attempt
        ok, parsed = _try_loads(usable)
        if not ok:
            continue
        names = ([extraction] if extraction else []) + repairs
        problems = _schema_problems(parsed, required_keys, expected)
        if problems:
            return FormatResult("refused", parsed=parsed,
                                repairs=tuple(names),
                                problems=tuple(problems),
                                dropped_tail=dropped)
        return FormatResult("repaired", text=usable, parsed=parsed,
                            repairs=tuple(names),
                            dropped_tail=dropped)

    if "{" not in raw and "[" not in raw:
        problem = "no_json_span"
    elif _scan(raw) is not None:
        problem = "truncated_beyond_repair"
    else:
        problem = "unparseable_json"
    return FormatResult("refused", problems=(problem,))


# ── Prose truncation ────────────────────────────────────────────────────────

_TERMINAL_CHARS = ".!?\u2026"
_SENTENCE_END_RE = re.compile(
    r"[.!?\u2026][\"\'\)\]]*(?=\s|$)")


def repair_truncation(text: str) -> TruncationResult:
    """Judge a prose draft's ending: intact when it
    ends on a terminal mark; repaired by cutting to
    the last complete sentence (dropped tail
    verbatim); refused when no complete sentence
    exists to cut to."""
    raw = (text or "").strip()
    if not raw:
        return TruncationResult("refused",
                                problems=("empty_output",))
    tail = raw.rstrip("\"')\u201d\u2019] ")
    if tail and tail[-1] in _TERMINAL_CHARS:
        return TruncationResult("intact", text=raw)
    ends = [m.end() for m in _SENTENCE_END_RE.finditer(raw)]
    if not ends:
        return TruncationResult(
            "refused", problems=("no_complete_sentence",))
    cut = ends[-1]
    return TruncationResult("repaired", text=raw[:cut],
                            dropped_tail=raw[cut:].strip())


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic Lantern Hollow material throughout
# (TF-017's world, never canon). expected_parsed is
# the exact structure a usable verdict must carry;
# None on a refusal.

FORMAT_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "valid_object",
     "text": '{"ferry": "dawn", "tide": "low"}',
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "valid", "repairs": (),
     "expected_parsed": {"ferry": "dawn", "tide": "low"}},
    {"id": "fenced_json",
     "text": "```json\n{\"ferry\": \"dawn\"}\n```",
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "repaired", "repairs": ("extracted_from_fence",),
     "expected_parsed": {"ferry": "dawn"}},
    {"id": "prose_wrapped_json",
     "text": 'Here is the answer: {"ferry": "dawn"} — hope it helps',
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "repaired", "repairs": ("extracted_from_prose",),
     "expected_parsed": {"ferry": "dawn"}},
    {"id": "trailing_comma",
     "text": '{"ferry": "dawn", "tide": "low",}',
     "required_keys": (), "expected": "object",
     "verdict": "repaired", "repairs": ("dropped_trailing_comma",),
     "expected_parsed": {"ferry": "dawn", "tide": "low"}},
    {"id": "missing_closers",
     "text": '{"ferry": "dawn", "tide": "low"',
     "required_keys": (), "expected": "object",
     "verdict": "repaired", "repairs": ("closed_open_containers",),
     "expected_parsed": {"ferry": "dawn", "tide": "low"}},
    {"id": "truncated_after_complete_pair",
     "text": '{"ferry": "dawn", "tide": ',
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "repaired",
     "repairs": ("dropped_incomplete_tail", "closed_open_containers"),
     "expected_parsed": {"ferry": "dawn"}},
    {"id": "truncated_mid_string_value",
     "text": '{"ferry": "dawn", "tide": "ris',
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "repaired",
     "repairs": ("dropped_incomplete_tail", "closed_open_containers"),
     "expected_parsed": {"ferry": "dawn"}},
    {"id": "truncated_mid_key",
     "text": '{"ferry": "dawn", "ti',
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "repaired",
     "repairs": ("dropped_incomplete_tail", "closed_open_containers"),
     "expected_parsed": {"ferry": "dawn"}},
    {"id": "truncated_array_mid_string",
     "text": '["dawn", "low',
     "required_keys": (), "expected": "array",
     "verdict": "repaired",
     "repairs": ("dropped_incomplete_tail", "closed_open_containers"),
     "expected_parsed": ["dawn"]},
    {"id": "unclosed_array_of_numbers",
     "text": "[1, 2, 3",
     "required_keys": (), "expected": "array",
     "verdict": "repaired", "repairs": ("closed_open_containers",),
     "expected_parsed": [1, 2, 3]},
    {"id": "python_literal",
     "text": "{'ferry': 'dawn', 'bell': True, 'tide': None}",
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "repaired",
     "repairs": ("converted_python_literal",),
     "expected_parsed": {"ferry": "dawn", "bell": True,
                         "tide": None}},
    {"id": "nested_truncation",
     "text": '{"route": {"from": "haven", "to": "isle"}, "bell": "si',
     "required_keys": ("route",), "expected": "object",
     "verdict": "repaired",
     "repairs": ("dropped_incomplete_tail", "closed_open_containers"),
     "expected_parsed": {"route": {"from": "haven", "to": "isle"}}},
    {"id": "tf087_broken_json_fixture",
     "text": '{"ferry": "dawn", "tide": ',
     "required_keys": (), "expected": "object",
     "verdict": "repaired",
     "repairs": ("dropped_incomplete_tail", "closed_open_containers"),
     "expected_parsed": {"ferry": "dawn"}},
    {"id": "refuse_prose_when_json_asked",
     "text": "The ferry leaves at dawn when the tide bell rings.",
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "refused", "repairs": (),
     "expected_parsed": None,
     "problems": ("no_json_span",)},
    {"id": "refuse_missing_required_key",
     "text": '{"tide": "low"}',
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "refused", "repairs": (),
     "expected_parsed": {"tide": "low"},
     "problems": ("missing_required_key: ferry",)},
    {"id": "refuse_missing_key_after_repair",
     "text": '{"tide": "ris',
     "required_keys": ("ferry",), "expected": "object",
     "verdict": "refused",
     "repairs": ("dropped_incomplete_tail", "closed_open_containers"),
     "expected_parsed": {},
     "problems": ("missing_required_key: ferry",)},
    {"id": "refuse_wrong_top_level",
     "text": '["dawn", "low"]',
     "required_keys": (), "expected": "object",
     "verdict": "refused", "repairs": (),
     "expected_parsed": ["dawn", "low"],
     "problems": ("wrong_top_level_shape: expected object, got list",)},
    {"id": "refuse_bare_word_value",
     "text": '{"ferry": dawn, "tide": "low"}',
     "required_keys": (), "expected": "object",
     "verdict": "refused", "repairs": (),
     "expected_parsed": None,
     "problems": ("unparseable_json",)},
    {"id": "refuse_multiple_spans",
     "text": '{"ferry": "dawn"} and {"tide": "low"}',
     "required_keys": (), "expected": "object",
     "verdict": "refused", "repairs": (),
     "expected_parsed": None,
     "problems": ("multiple_json_spans",)},
    {"id": "refuse_empty",
     "text": "   ",
     "required_keys": (), "expected": "object",
     "verdict": "refused", "repairs": (),
     "expected_parsed": None,
     "problems": ("empty_output",)},
)

TRUNCATION_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "intact_sentence",
     "text": "The ferry leaves at dawn.",
     "verdict": "intact",
     "expected_text": "The ferry leaves at dawn.",
     "dropped_tail": ""},
    {"id": "intact_question_with_quote",
     "text": 'The keeper asked, "When does the ferry leave?"',
     "verdict": "intact",
     "expected_text": 'The keeper asked, "When does the ferry leave?"',
     "dropped_tail": ""},
    {"id": "cut_to_last_complete_sentence",
     "text": "The ferry leaves at dawn. The tide bell rings across the",
     "verdict": "repaired",
     "expected_text": "The ferry leaves at dawn.",
     "dropped_tail": "The tide bell rings across the"},
    {"id": "tf087_truncated_fixture_refused",
     "text": "The ferry leaves at dawn and the tide bell rings "
             "across the",
     "verdict": "refused",
     "expected_text": None,
     "dropped_tail": ""},
    {"id": "refuse_empty_prose",
     "text": "",
     "verdict": "refused",
     "expected_text": None,
     "dropped_tail": ""},
)


# ── Checkers ────────────────────────────────────────────────────────────────


def format_enforcement_problems() -> list[str]:
    """Data-integrity checker: rule and fixture ids
    unique, every fixture well-formed, refusal
    fixtures naming their problems."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in FORMAT_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate FORMAT_RULES ids")
    fixture_ids = [f["id"] for f in FORMAT_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate FORMAT_FIXTURES ids")
    trunc_ids = [f["id"] for f in TRUNCATION_FIXTURES]
    if len(trunc_ids) != len(set(trunc_ids)):
        problems.append("duplicate TRUNCATION_FIXTURES ids")
    for fixture in FORMAT_FIXTURES:
        if fixture["verdict"] not in ("valid", "repaired", "refused"):
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{fixture['verdict']!r} not in the vocabulary")
        if fixture["verdict"] == "refused" and \
                not fixture.get("problems"):
            problems.append(
                f"fixture {fixture['id']}: a refusal must "
                "name its problems")
        if fixture["verdict"] == "repaired" and \
                not fixture["repairs"]:
            problems.append(
                f"fixture {fixture['id']}: a repair must "
                "name its repairs")
    return problems


def format_enforcement_live_problems() -> list[str]:
    """Live checker: every filed fixture is run
    through the real gate and must land exactly as
    filed (verdict, repairs, parsed structure,
    problems); every usable verdict's text must
    re-parse to the filed structure (a repair is
    content-preserving by construction — the kept
    keys and values are the source's own)."""
    problems: list[str] = []
    for fixture in FORMAT_FIXTURES:
        result = enforce_format(
            fixture["text"],
            required_keys=tuple(fixture["required_keys"]),
            expected=fixture["expected"])
        if result.verdict != fixture["verdict"]:
            problems.append(
                f"fixture {fixture['id']}: verdict "
                f"{result.verdict} != {fixture['verdict']}")
            continue
        if tuple(result.repairs) != tuple(fixture["repairs"]):
            problems.append(
                f"fixture {fixture['id']}: repairs "
                f"{result.repairs} != {fixture['repairs']}")
        if result.parsed != fixture["expected_parsed"]:
            problems.append(
                f"fixture {fixture['id']}: parsed "
                f"{result.parsed!r} != {fixture['expected_parsed']!r}")
        if fixture["verdict"] == "refused":
            if result.text is not None:
                problems.append(
                    f"fixture {fixture['id']}: a refusal "
                    "carries usable text")
            if tuple(result.problems) != tuple(fixture["problems"]):
                problems.append(
                    f"fixture {fixture['id']}: problems "
                    f"{result.problems} != {fixture['problems']}")
        else:
            ok, reparsed = _try_loads(result.text or "")
            if not ok or reparsed != fixture["expected_parsed"]:
                problems.append(
                    f"fixture {fixture['id']}: usable text "
                    "does not re-parse to the filed structure")
    for fixture in TRUNCATION_FIXTURES:
        result = repair_truncation(fixture["text"])
        if result.verdict != fixture["verdict"]:
            problems.append(
                f"truncation fixture {fixture['id']}: verdict "
                f"{result.verdict} != {fixture['verdict']}")
            continue
        if result.text != fixture["expected_text"]:
            problems.append(
                f"truncation fixture {fixture['id']}: text "
                f"{result.text!r} != {fixture['expected_text']!r}")
        if result.dropped_tail != fixture["dropped_tail"]:
            problems.append(
                f"truncation fixture {fixture['id']}: dropped "
                f"{result.dropped_tail!r} != "
                f"{fixture['dropped_tail']!r}")
        if fixture["verdict"] == "repaired":
            kept, dropped = fixture["expected_text"], \
                fixture["dropped_tail"]
            if kept + " " + dropped != fixture["text"]:
                problems.append(
                    f"truncation fixture {fixture['id']}: kept "
                    "text plus dropped tail is not the source")
    return problems


def format_as_data() -> dict[str, Any]:
    return {
        "format": FORMAT_FORMAT,
        "format_doc": FORMAT_DOC,
        "rules": [dict(r) for r in FORMAT_RULES],
        "format_fixtures": [dict(f) for f in FORMAT_FIXTURES],
        "truncation_fixtures": [dict(f)
                                 for f in TRUNCATION_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_structured_outputs": "not_proven",
            "salvage_py": "unchanged_by_design",
        },
    }
