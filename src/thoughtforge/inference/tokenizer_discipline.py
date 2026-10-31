"""TF-071 — Tokeniser discipline: per-backend token
counting agreed within tolerance.

Roadmap: "per-backend token counting agreed within
tolerance (the Aesir tokenizer lessons applied)".

Measured start (probed this tick, VM):

- TF-065 gave every backend a ``count_tokens`` and a
  declared basis (``model`` / ``estimate``), but no
  agreement between backends was ever measured, and no
  tolerance existed anywhere: TF-065's doc closed with
  "token-count agreement between backends NOT PROVEN
  (TF-071 owns the tolerance discipline)".
- Three different counting conventions coexisted under
  the one word "token": the contract estimate
  (chars / 4), the cognition word count
  (``len(text.split())`` — history budgets, prompt
  costs, final ``token_count``), and server-reported
  completion counts (ollama ``eval_count``, lmstudio
  ``usage.completion_tokens``, TurboQuant's llama
  ``completion_tokens`` — real model counts) — plus one
  impostor: HF's ``tokens_generated`` was a word-split
  count wearing a token count's name.
- Two measured defects hid in the estimate's copies:
  the TurboQuant engine kept a private copy of the
  heuristic that counted "" as 1 even with no model
  loaded (the TF-065 contract says empty counts 0; the
  backend wrapper guarded it, the engine did not), and
  the flat chars/4 heuristic is script-blind — measured
  against a real BPE tokeniser (tiktoken cl100k_base,
  present on this VM as an environment tool, not a
  shipped dependency), Latin prose landed within 25%
  but an 11-character CJK sentence counted 2 against
  the tokeniser's 16.

THE AESIR LESSONS, applied (from Project Aesir's
comparator-artifact verdicts): most apparent token
divergences are artefacts of the comparison, not of
either counter —

- **Special tokens are a basis, not an error.** A
  llama.cpp ``tokenize`` call includes BOS by default,
  so a loaded engine counts "" as 1 and every text as
  +1 against a special-free counter. Counts state
  whether special tokens are included; the tolerance's
  absolute allowance absorbs the difference instead of
  "fixing" either side.
- **Prompt-boundary / short-text effects dominate at
  small counts.** "Yes." is 1 by the estimate, 2 by
  BPE, 3 by a BOS-carrying stub — all three honest.
  Agreement is therefore judged with an absolute floor,
  never purely relatively.
- **Never compare across tokenisers expecting
  equality.** Different vocabularies split differently;
  the discipline is a stated tolerance on a stated
  corpus, re-derivable — not a hope that two counters
  coincide.
- **Script is part of the basis.** A chars-per-token
  density measured on Latin prose does not transfer to
  scripts that tokenise per character; an estimate
  that does not say so is silently wrong there.

THE DISCIPLINE (``TOKENIZER_RULES`` as data):

- ``empty_is_zero`` — on the contract surface, "" counts
  0 on every backend and on the unloaded engine. (A
  *loaded* model's count of "" is its own truth — BOS —
  and is stated as such, never compared raw.)
- ``one_estimate_one_home`` — the contract estimate
  lives in ``backend_contract.estimate_tokens`` alone;
  the engine delegates, the backends inherit or
  delegate. No private copies to drift.
- ``script_aware_estimate`` — script-dense characters
  (CJK / kana / Hangul) count one token each in the
  estimate; Latin behaviour is byte-identical to
  TF-065's.
- ``basis_is_stated`` — every counting surface files
  its basis in ``COUNTING_SURFACES`` (model / estimate /
  word / server-model) and its special-token stance;
  a surface whose basis changes amends the table.
- ``agreement_within_tolerance`` — two counters agree
  on a text iff ``|a - b| <= max(ABS_TOKENS,
  REL_TOLERANCE * max(a, b))`` with ABS_TOKENS = 2 and
  REL_TOLERANCE = 0.35, derived from this tick's
  measurements (see ``CORPUS`` / ``MEASURED_BPE``):
  Latin estimate-vs-BPE relative error peaked at 0.25
  (long prose), post-fix CJK at 0.31 — a 0.35 band with
  a 2-token floor covers every measured honest pair
  and still catches the pre-fix CJK breach (0.88).
- ``word_counts_are_not_tokens`` — the cognition word
  count is a *budget* convention (TF-021/TF-024), kept
  deliberately; it breaches the tolerance against BPE
  on 5 of the 9 corpus rows and must never be
  substituted for, or compared as, a backend token
  count. The shared function name ``estimate_tokens``
  in two modules is filed here as a standing hazard
  with this rule as its guard.
- ``reported_counts_match_their_backend`` — a
  response's ``tokens_generated`` is either the
  server's own model count (ollama / lmstudio /
  TurboQuant) or the backend's declared count basis
  (HF: the contract estimate, TF-071).

FIXED in scope:

- **D-TOK-1** — ``TurboQuantEngine.estimate_tokens``
  counted "" as 1 with no model loaded (private
  heuristic copy). The fallback now delegates to the
  contract estimate; the model path's BOS basis is
  documented on the method.
- **D-TOK-2** — the contract estimate was script-blind
  (CJK counted at Latin density: 2 vs a real
  tokeniser's 16). ``backend_contract.estimate_tokens``
  is now script-aware; Latin counts unchanged.
- **D-TOK-3** — HF's ``tokens_generated`` was a
  word-split count under a token name on both legs.
  Both legs now report ``count_tokens(text)`` — the
  backend's own declared basis — and the capability
  note says so (TF-065's pin updated in scope).

Proof grade FIXTURE (the shipped backends and the real
engine on this VM, a deterministic stub llama model
whose tokeniser is the filed reference rule + BOS, and
BPE counts measured this tick from tiktoken's
cl100k_base — an environment tool, re-derived live by
the checker when it is importable, filed as data when
it is not) — agreement with a *loaded GGUF model's*
tokeniser is NOT PROVEN (no llama_cpp, no GGUF on this
VM; TF-074's Aesir backend and the live-model slices
own that road), never claimed.
"""

from __future__ import annotations

from typing import Any

DISCIPLINE_FORMAT = "thoughtforge-tokenizer-discipline/1"

#: The agreement band, derived from this tick's
#: measurements (module docstring): the absolute floor
#: absorbs special tokens (BOS = +1) and short-text
#: boundary effects; the relative band covers the
#: measured estimate-vs-BPE spread (Latin peak 0.25,
#: post-fix CJK 0.31) with headroom, and is still tight
#: enough to catch the pre-fix CJK breach (0.88).
ABS_TOKENS = 2
REL_TOLERANCE = 0.35


def counts_agree(a: int, b: int) -> bool:
    """The agreement rule: two honest counters agree on
    a text iff their counts are within the band."""
    return abs(a - b) <= max(ABS_TOKENS, REL_TOLERANCE * max(a, b))


def relative_error(a: int, b: int) -> float:
    """|a - b| relative to the larger count (0 when both 0)."""
    hi = max(a, b)
    return abs(a - b) / hi if hi else 0.0


#: The probe corpus: synthetic Lantern Hollow / neutral
#: texts (never canon), one row per text class the
#: discipline must hold across.
CORPUS: dict[str, str] = {
    "prose": "The ferry leaves at dawn when the tide bell rings "
             "twice over the harbour.",
    "long_prose": (
        "The keeper lit the red lantern and checked the chalk "
        "board by the boathouse door. Winter crossings take "
        "forty minutes longer, and the last boat returns "
        "before the ice forms on the channel markers. "
    ) * 3,
    "numbers": "In 2026 the crossing carried 12,480 passengers, "
               "3,914 cars and 217 bicycles in 96 days.",
    "code": "def select_window(messages, max_turns=20):\n"
            "    kept = messages[-(max_turns * 2):]\n"
            "    return kept, messages[:len(messages)-len(kept)]",
    "punct": "Wait — what?! 'Tokens,' she said, \"aren't just "
             "words...\" (not really.)",
    "cjk": "灯台は夜の渡し船を導く",
    "mixed": "The guía said the travesía takes 40 Minuten bei "
             "gutem Wetter.",
    "short": "Yes.",
    "empty": "",
}

#: BPE counts for the corpus, MEASURED 2026-10-31 on this
#: VM with tiktoken's cl100k_base encoding (an environment
#: tool — tiktoken is not a shipped dependency and no
#: shipped code imports it). Filed as data; the checker
#: re-derives them live when tiktoken is importable and
#: reports a drift as a problem, and otherwise judges the
#: estimate against these filed rows.
MEASURED_BPE: dict[str, int] = {
    "prose": 15,
    "long_prose": 118,
    "numbers": 27,
    "code": 38,
    "punct": 20,
    "cjk": 16,
    "mixed": 18,
    "short": 2,
    "empty": 0,
}

#: Every counting surface in the tree, as data: what it
#: counts, on what basis, and its special-token stance.
#: Judged live by the checker (anchors must exist).
COUNTING_SURFACES: tuple[dict[str, Any], ...] = (
    {
        "surface": "backend_contract.estimate_tokens",
        "kind": "count_tokens basis",
        "basis": "estimate",
        "special_tokens": "none",
        "note": "the one contract estimate: ~4 chars/token, "
                "script-aware since TF-071 (D-TOK-2)",
    },
    {
        "surface": "UnifiedBackend.count_tokens (default)",
        "kind": "count_tokens",
        "basis": "estimate",
        "special_tokens": "none",
        "note": "inherited by ollama / lmstudio / hf — the "
                "three server backends count identically",
    },
    {
        "surface": "TurboQuantBackend.count_tokens",
        "kind": "count_tokens",
        "basis": "model when the engine is loaded, else estimate",
        "special_tokens": "included when model (llama BOS)",
        "note": "empty guarded to 0 before the engine, per "
                "the contract's empty rule",
    },
    {
        "surface": "TurboQuantEngine.estimate_tokens",
        "kind": "engine count",
        "basis": "model when loaded, else the contract estimate "
                 "by delegation (D-TOK-1)",
        "special_tokens": "included when model (llama BOS — "
                          "a loaded engine counts '' as 1)",
        "note": "the name says estimate; with a model loaded "
                "it is a model count, basis stated here and "
                "on the method",
    },
    {
        "surface": "ollama tokens_generated (eval_count)",
        "kind": "tokens_generated",
        "basis": "server-model",
        "special_tokens": "server's own",
        "note": "completion tokens only, counted by the "
                "server's model tokeniser",
    },
    {
        "surface": "lmstudio tokens_generated (usage.completion_tokens)",
        "kind": "tokens_generated",
        "basis": "server-model",
        "special_tokens": "server's own",
        "note": "completion tokens only, counted by the "
                "server's model tokeniser — when the "
                "response reports usage; TF-076's shared "
                "family parser falls back to the contract "
                "estimate when it does not (D-OAI-3)",
    },
    {
        "surface": "turboquant tokens_generated (completion_tokens)",
        "kind": "tokens_generated",
        "basis": "server-model",
        "special_tokens": "server's own",
        "note": "completion tokens from the llama usage block",
    },
    {
        "surface": "hf tokens_generated",
        "kind": "tokens_generated",
        "basis": "estimate",
        "special_tokens": "none",
        "note": "the backend's own count_tokens basis since "
                "TF-071 (D-TOK-3); pre-TF-071 a word-split "
                "count under a token name",
    },
    {
        "surface": "cognition history_discipline.estimate_tokens",
        "kind": "budget count — NOT a token count",
        "basis": "word",
        "special_tokens": "n/a",
        "note": "the cognition word-token convention "
                "(TF-021/TF-024): history budgets, prompt "
                "costs, final token_count. A different "
                "discipline by ruling — never substituted "
                "for a backend count (it breaches the "
                "tolerance against BPE on the numbers / "
                "code / punct / cjk / mixed corpus rows)",
    },
)

#: The rules, as data.
TOKENIZER_RULES: tuple[dict[str, Any], ...] = (
    {"rule": "empty_is_zero",
     "text": "on the contract surface '' counts 0 on every "
             "backend and on the unloaded engine; a loaded "
             "model's BOS count of '' is its own stated "
             "basis, never compared raw"},
    {"rule": "one_estimate_one_home",
     "text": "the contract estimate lives in "
             "backend_contract.estimate_tokens alone; "
             "backends inherit it, the engine delegates "
             "to it — no private copies to drift"},
    {"rule": "script_aware_estimate",
     "text": "script-dense characters (CJK / kana / "
             "Hangul) count one token each in the "
             "estimate; Latin counts are TF-065's, "
             "unchanged"},
    {"rule": "basis_is_stated",
     "text": "every counting surface files its basis "
             "and special-token stance in "
             "COUNTING_SURFACES; a basis change amends "
             "the table in the same slice"},
    {"rule": "agreement_within_tolerance",
     "text": "|a - b| <= max(ABS_TOKENS, REL_TOLERANCE * "
             "max(a, b)) on the filed corpus — equality "
             "across different tokenisers is never "
             "demanded (the Aesir comparator lesson)"},
    {"rule": "special_tokens_are_a_basis",
     "text": "a model count includes the special tokens "
             "its tokeniser adds (llama BOS = +1); the "
             "absolute floor absorbs them — neither side "
             "is 'fixed' to match the other"},
    {"rule": "word_counts_are_not_tokens",
     "text": "the cognition word count is a budget "
             "convention, kept deliberately; it is never "
             "substituted for, or compared as, a backend "
             "token count, despite the shared "
             "estimate_tokens name"},
    {"rule": "reported_counts_match_their_backend",
     "text": "tokens_generated is the server's own model "
             "count or the backend's declared count "
             "basis — never a third convention under a "
             "token name"},
)


def reference_stub_tokens(text: str) -> list[str]:
    """The fixture reference tokeniser: a deterministic
    subword rule standing in for a model tokeniser in
    tests (letter runs in 4-char chunks, digit runs in
    3-digit groups, symbol runs in 4-char chunks, one
    token per script-dense character, whitespace free).

    It is a *fixture*, not a model: its job is to be a
    second, independent-ish counter whose disagreements
    with the estimate are of the size real tokenisers
    produce, so the tolerance is exercised honestly.
    """
    from thoughtforge.inference.backend_contract import _is_script_char

    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if _is_script_char(ch):
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            for k in range(i, j, 3):
                tokens.append(text[k:k + 3])
            i = j
            continue
        if ch.isalpha():
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
            for k in range(i, j, 4):
                tokens.append(text[k:k + 4])
            i = j
            continue
        j = i
        while j < len(text) and not text[j].isspace() \
                and not _is_script_char(text[j]) \
                and not text[j].isdigit() and not text[j].isalpha():
            j += 1
        for k in range(i, j, 4):
            tokens.append(text[k:k + 4])
        i = j
    return tokens


class ReferenceStubLlamaModel:
    """A stub of the llama.cpp model object the engine
    calls: ``tokenize(bytes)`` returns the reference
    tokens PLUS a BOS marker at the front — the
    llama-cpp-python default (``add_bos=True``) that the
    special-tokens rule exists to state."""

    BOS = "<bos>"

    def tokenize(self, data: bytes) -> list[str]:
        return [self.BOS] + reference_stub_tokens(data.decode("utf-8"))


def corpus_counts() -> dict[str, dict[str, int]]:
    """Every counter's count for every corpus row,
    measured live: the contract estimate, the cognition
    word count, the filed BPE row, and the stub model
    (reference + BOS) the way a loaded engine reports it.
    """
    from thoughtforge.cognition.history_discipline import (
        estimate_tokens as word_estimate,
    )
    from thoughtforge.inference.backend_contract import estimate_tokens

    rows: dict[str, dict[str, int]] = {}
    stub = ReferenceStubLlamaModel()
    for name, text in CORPUS.items():
        rows[name] = {
            "estimate": estimate_tokens(text),
            "word": word_estimate(text),
            "bpe_filed": MEASURED_BPE[name],
            "stub_model": len(stub.tokenize(text.encode("utf-8"))),
        }
    return rows


def discipline_as_data() -> dict[str, Any]:
    """The whole estate as JSON-safe data."""
    return {
        "format": DISCIPLINE_FORMAT,
        "abs_tokens": ABS_TOKENS,
        "rel_tolerance": REL_TOLERANCE,
        "rules": [dict(r) for r in TOKENIZER_RULES],
        "surfaces": [dict(s) for s in COUNTING_SURFACES],
        "corpus_counts": corpus_counts(),
    }


# ── Live checker ────────────────────────────────────────────────────────────

def _bpe_live_counts() -> dict[str, int] | None:
    """Re-derive the BPE rows live when tiktoken is
    importable; None when it is not (the filed rows then
    stand as this slice's measurement). Any failure —
    import, encoding load, encode — also yields None:
    the checker never lets an environment tool's absence
    masquerade as a discipline breach."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return {name: len(enc.encode(text))
                for name, text in CORPUS.items()}
    except Exception:  # noqa: BLE001 — environment tool, judged
        return None


def tokenizer_problems() -> list[str]:
    """Re-derive the discipline live; [] means true."""
    import inspect

    from thoughtforge.cognition.history_discipline import (
        estimate_tokens as word_estimate,
    )
    from thoughtforge.inference import backend_contract, hf_backend
    from thoughtforge.inference.backend_contract import (
        estimate_tokens,
        shipped_backends,
    )
    from thoughtforge.inference.turboquant import TurboQuantEngine

    problems: list[str] = []
    rows = corpus_counts()

    # The tolerance covers the estimate against the filed
    # BPE rows on every corpus row — and the pre-TF-071
    # flat formula would breach it on the CJK row (the
    # D-TOK-2 defect, kept provable).
    for name, row in rows.items():
        if not counts_agree(row["estimate"], row["bpe_filed"]):
            problems.append(
                f"{name}: estimate {row['estimate']} outside "
                f"tolerance of filed BPE {row['bpe_filed']}")
        if not counts_agree(row["estimate"], row["stub_model"]):
            problems.append(
                f"{name}: estimate {row['estimate']} outside "
                f"tolerance of stub model {row['stub_model']}")
    cjk = CORPUS["cjk"]
    flat = max(1, len(cjk) // 4)
    if counts_agree(flat, MEASURED_BPE["cjk"]):
        problems.append("pre-TF-071 flat estimate no longer "
                        "breaches on the CJK row — D-TOK-2's "
                        "evidence changed")
    live_bpe = _bpe_live_counts()
    if live_bpe is not None and live_bpe != MEASURED_BPE:
        problems.append(f"live BPE counts {live_bpe} != filed "
                        f"{MEASURED_BPE}")

    # The word discipline is a different discipline: it
    # must breach the BPE tolerance on the filed rows —
    # if it ever agreed everywhere, the two conventions
    # would be silently converging and the guard rule
    # would be moot.
    word_breaches = {
        name for name, row in rows.items()
        if not counts_agree(row["word"], row["bpe_filed"])
    }
    if word_breaches != {"numbers", "code", "punct", "cjk", "mixed"}:
        problems.append(f"word-count BPE breaches {sorted(word_breaches)} "
                        f"!= filed set")

    # Per-backend agreement: every shipped backend counts
    # every corpus row identically today (all estimate
    # based, TurboQuant unloaded) — and therefore within
    # tolerance, which is the form the rule takes when a
    # model tokeniser joins (TF-074).
    backends = shipped_backends()
    for name, text in CORPUS.items():
        counts = {b.backend_name(): b.count_tokens(text)
                  for b in backends}
        values = list(counts.values())
        if len(set(values)) != 1:
            problems.append(f"{name}: shipped backends disagree "
                            f"{counts}")
        for value in values:
            if not counts_agree(value, rows[name]["bpe_filed"]):
                problems.append(
                    f"{name}: backend count {value} outside "
                    f"tolerance of filed BPE {rows[name]['bpe_filed']}")

    # D-TOK-1 stays fixed: the unloaded engine IS the
    # contract estimate, empty included; the loaded
    # (stub) engine is the stub count, BOS included, and
    # the wrapper still guards its surface to 0 on empty.
    engine = TurboQuantEngine(model_path=None, profile_id="desktop_cpu")
    for name, text in CORPUS.items():
        if engine.estimate_tokens(text) != estimate_tokens(text):
            problems.append(
                f"{name}: unloaded engine count "
                f"{engine.estimate_tokens(text)} != contract "
                f"estimate {estimate_tokens(text)} (D-TOK-1)")
    engine._model = ReferenceStubLlamaModel()
    for name, text in CORPUS.items():
        if engine.estimate_tokens(text) != rows[name]["stub_model"]:
            problems.append(
                f"{name}: stub-loaded engine count "
                f"{engine.estimate_tokens(text)} != stub "
                f"{rows[name]['stub_model']}")
    from pathlib import Path

    from thoughtforge.inference.turboquant_backend import TurboQuantBackend

    wrapper = TurboQuantBackend(model_path=Path("/nonexistent/tf071.gguf"))
    wrapper._engine = engine
    if wrapper.count_tokens("") != 0:
        problems.append("wrapper count_tokens('') != 0 with a "
                        "loaded engine — the contract empty "
                        "rule is breached at the surface")
    engine_src = inspect.getsource(TurboQuantEngine.estimate_tokens)
    if "backend_contract" not in engine_src:
        problems.append("engine estimate no longer delegates "
                        "to the contract estimate (D-TOK-1)")

    # D-TOK-2 stays fixed: the estimate's one home is
    # script-aware, and no second // 4 copy survives in
    # the engine.
    if not backend_contract.SCRIPT_TOKEN_RANGES:
        problems.append("script token ranges missing (D-TOK-2)")
    from thoughtforge.inference import turboquant

    tq_src = inspect.getsource(turboquant)
    if "len(text) // 4" in tq_src:
        problems.append("a private // 4 heuristic copy survives "
                        "in turboquant.py (D-TOK-1)")

    # D-TOK-3 stays fixed: both HF legs report the
    # backend's own count basis.
    hf_src = inspect.getsource(hf_backend)
    if hf_src.count("tokens_generated=self.count_tokens(text)") != 2:
        problems.append("HF tokens_generated is not the "
                        "contract estimate on both legs (D-TOK-3)")
    if "len(text.split())" in hf_src:
        problems.append("a word-split count survives in "
                        "hf_backend.py (D-TOK-3)")

    # The two estimate_tokens functions stay two: the
    # cognition word convention is not the contract
    # estimate, by identity and by behaviour.
    if word_estimate is estimate_tokens:
        problems.append("cognition estimate_tokens IS the "
                        "contract estimate — disciplines merged")
    return problems


__all__ = [
    "ABS_TOKENS",
    "CORPUS",
    "COUNTING_SURFACES",
    "DISCIPLINE_FORMAT",
    "MEASURED_BPE",
    "REL_TOLERANCE",
    "TOKENIZER_RULES",
    "ReferenceStubLlamaModel",
    "corpus_counts",
    "counts_agree",
    "discipline_as_data",
    "reference_stub_tokens",
    "relative_error",
    "tokenizer_problems",
]
