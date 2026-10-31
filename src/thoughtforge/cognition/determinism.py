"""
Determinism mode (TF-036): seeded runs reproducible for tests
and benchmarks.

Before this slice the cognition layer was *mostly* deterministic
by accident and reproducible by nobody: the router, scaffold
builder, prompt builder, scoring, salvage and enforcement are
pure functions of their inputs (no RNG anywhere in them — the
census below is grep-provable), but two things made two runs of
the same conversation impossible to compare exactly:

  * **every id was minted from ``uuid.uuid4()``** — turn ids,
    candidate ids, fragment ids, response ids all differed run
    to run, so no two turn records could ever be diffed whole;
  * **generation sampling had no seed to carry** —
    ``GenerationParams`` / ``GenerationRequest`` had no seed
    field at all, so even a backend that *wanted* to be seeded
    could not be told to be.

Determinism mode is what a conversation selects (config section
``determinism``, key ``seed``) when its runs must be reproducible
— tests and benchmarks are the named consumers (TF-121/TF-135
inherit this; TF-135's flake discipline cites it). With a seed
selected, the core:

  * mints every id it creates from a dedicated seeded stream
    (``DeterministicIds``) instead of uuid4 — two fresh cores
    with the same seed running the same script mint the same
    ids in the same order;
  * draws one generation seed per generation call from a second
    dedicated stream and carries it on the generation params /
    request (``seed=``) — candidate generation (plain, streamed,
    and via the unified backend), the quality-gate repair pass,
    and one draw per compose for the salvage refine passes
    (pass *i* carries that draw + *i*). A generation source
    that honours the seed then samples reproducibly; one that
    ignores it is NOT deterministic and the mode does not
    pretend otherwise (see the honesty table below);
  * leaves everything else exactly as it was. No seed selected
    = the pre-TF-036 behaviour, byte for byte: uuid4 ids, no
    seed on any generation call.

**What reproducibility claims, exactly.** ``turn_fingerprint``
defines it: a turn's fingerprint carries its ids, its scaffold
(intent / depth / modes / goal / fact_block), its candidates
(mode + text + token estimate), its kept fragments, and its
final (text, mode, scores, token count, enforcement verdict,
salvage path, source ids). Two runs are reproducible when their
fingerprints are EQUAL — ``reproducibility_problems`` names any
field that differs. Deliberately EXCLUDED from the fingerprint,
named here so nobody has to guess: the turn's timestamps
(``started_at`` / ``completed_at`` — wall clock by nature) and
its stage timings in ms (``retrieval_ms`` / ``generation_ms`` /
``salvage_ms`` — measurements, TF-033's subject). Wall clock is
not content and no seed can or should pin it.

**The source census** (``DETERMINISM_SOURCES``, stated as data):

  * *pure layers* — router / scaffold / prompt builder /
    scoring / salvage / enforcement: deterministic by
    construction, no RNG, no clock, no global state read that
    varies between runs of the same inputs.
  * *generation sampling* — seeded in this mode, as above; the
    one content-bearing source that can vary, and the honesty
    boundary: seed-honouring is a property of the generation
    source. The local TurboQuant engine forwards the seed to
    the llama call when one is set (and omits it entirely when
    not, so unseeded behaviour is unchanged); unified backends
    receive it on the request — TF-065's backend contract owns
    requiring them to honour it.
  * *record ids* — seeded in this mode (uuid4 otherwise).
    Salvage's internal fragment ids are NOT among them: they
    are minted inside ``FragmentSalvage.forge()``, never leave
    it (``SalvageResult`` carries no ids), and never reach a
    turn record — excluded with that evidence, not by hope.
  * *timestamps & timings* — excluded from reproducibility, as
    above.
  * *chat history ids/timestamps* — caller-owned
    (``ChatMessage`` defaults): they never reach a prompt (the
    render carries role + content only, TF-024) and are not in
    the fingerprint; a caller who needs a reproducible history
    object supplies explicit ids.
  * *the process-global RNG* — deliberately NOT seeded by this
    mode. Seeding ``random`` globally would couple every
    conversation in the process (and every thread, TF-035) to
    one stream; the mode's streams are dedicated per core.
    Nothing in the core loop reads the global RNG today (the
    one ``random`` user in the tree, ``utils/retry``'s jitter,
    is test-only and invoked by nothing in the loop — TF-011's
    D-ORPHAN-2).

**Re-running on one core.** A core's seeded streams advance as
it runs (that is what makes turn 2's ids differ from turn 1's
inside one run). A benchmark that re-runs a script on the SAME
core calls ``reset_deterministic_state(core)`` first, which
re-derives both streams from the mode's seed AND restarts the
will ledger — the one piece of cross-turn state the core owns
(TF-029): the first probe of this slice showed a re-run whose
finals matched but whose scaffolds carried the previous run's
standing goal, so streams-only reset was a half-reset and is
not what is shipped. A re-run is a new conversation; the
caller-owned chat history is the caller's to renew likewise.
On a core with no seed selected the reset raises — there is
nothing to reset, and a silent no-op would let a benchmark
believe it was seeded when it was not.

Proof grade: FIXTURE — a seed-honouring stub backend (samples
among phrasings with ``random.Random(request.seed)``) and a
seed-honouring stub engine, plus knowledge-only cores, on this
VM. A *live* model's reproducibility under a carried seed is
that model's own property and is NOT PROVEN here (the quant /
backend panels, TF-067/TF-126, re-measure it where a real model
runs).

The human-readable write-up is
``docs/architecture/DETERMINISM_MODE.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# ── Rulings ───────────────────────────────────────────────────────────────────

RULING_DETERMINISTIC = "deterministic"
RULING_SEEDED = "seeded-in-mode"
RULING_EXCLUDED = "excluded-from-reproducibility"

VALID_RULINGS: frozenset[str] = frozenset(
    {RULING_DETERMINISTIC, RULING_SEEDED, RULING_EXCLUDED}
)


@dataclass(frozen=True)
class DeterminismSource:
    """One surface that could vary between runs, and its ruling."""

    surface: str
    ruling: str
    detail: str


DETERMINISM_SOURCES: tuple[DeterminismSource, ...] = (
    DeterminismSource(
        "router_scaffold_prompts",
        RULING_DETERMINISTIC,
        "InputRouter, ScaffoldBuilder and PromptBuilder are pure "
        "functions of (text, bundle, personality): regex tables "
        "and assembly rules, no RNG, no clock.",
    ),
    DeterminismSource(
        "scoring_salvage_enforcement",
        RULING_DETERMINISTIC,
        "Candidate/fragment/final scoring, FragmentSalvage and "
        "EnforcementGate compute from their inputs only; ties "
        "break by stable sort order, never by chance.",
    ),
    DeterminismSource(
        "generation_sampling",
        RULING_SEEDED,
        "The one content-bearing source that can vary. In this "
        "mode the core draws one seed per generation call "
        "(candidates, repair, and one per compose for salvage "
        "refine passes) from its seeded generation stream and "
        "carries it on the params/request; a source that honours "
        "the seed samples reproducibly, one that ignores it is "
        "not deterministic and is not claimed to be.",
    ),
    DeterminismSource(
        "record_ids",
        RULING_SEEDED,
        "Turn / candidate / fragment / response ids are minted "
        "from the core's seeded id stream in this mode (uuid4 "
        "when no seed is selected). Salvage-internal fragment "
        "ids never leave FragmentSalvage.forge() and are not "
        "record ids.",
    ),
    DeterminismSource(
        "timestamps_and_timings",
        RULING_EXCLUDED,
        "started_at / completed_at are wall clock and the *_ms "
        "stage timings are measurements; neither is content, "
        "neither is in the fingerprint, no seed pins them.",
    ),
    DeterminismSource(
        "chat_history_ids",
        RULING_EXCLUDED,
        "ChatMessage ids/timestamps are caller-owned defaults; "
        "they never reach a prompt (the render is role + content "
        "only) and are not in the fingerprint.",
    ),
    DeterminismSource(
        "global_rng",
        RULING_EXCLUDED,
        "The process-global random module is deliberately never "
        "seeded by this mode (it would couple every conversation "
        "and thread to one stream); nothing in the core loop "
        "reads it.",
    ),
)


def source_ruling(surface: str) -> DeterminismSource:
    """The ruling for one named surface.

    Raises:
        ValueError: the surface is not in the census; the
            message names the valid surfaces.
    """
    for source in DETERMINISM_SOURCES:
        if source.surface == surface:
            return source
    valid = ", ".join(s.surface for s in DETERMINISM_SOURCES)
    raise ValueError(f"unknown determinism surface {surface!r}; valid: {valid}")


# ── The mode ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DeterminismMode:
    """One conversation's determinism selection.

    ``seed`` is None when no seed was selected — determinism
    mode is then OFF and the core behaves exactly as before
    TF-036. A selected seed (0 included — 0 is a real seed,
    never treated as "unset") turns the mode on.
    """

    seed: int | None = None

    @property
    def enabled(self) -> bool:
        return self.seed is not None

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def new_id_stream(self) -> "DeterministicIds":
        """A fresh id stream for this mode's seed."""
        if self.seed is None:
            raise ValueError("determinism mode is off — no seed selected")
        return DeterministicIds(self.seed)

    def new_generation_rng(self) -> random.Random:
        """A fresh generation-seed stream for this mode's seed.

        A dedicated ``random.Random`` (never the global module),
        seeded from a string so the derivation is stable across
        processes and Python runs.
        """
        if self.seed is None:
            raise ValueError("determinism mode is off — no seed selected")
        return random.Random(f"thoughtforge-determinism-gen-{self.seed}")


def resolve_determinism_mode(
    config: dict[str, Any] | None,
) -> DeterminismMode:
    """
    Resolve one core's determinism selection from its config
    dict: the ``determinism`` section's ``seed`` key — the only
    key this surface reads. No section / no key = mode off.

    Raises:
        ValueError: the seed is present but is not a
            non-negative int (bools included — ``True`` is not
            a seed a benchmark chose).
    """
    config = config or {}
    section = config.get("determinism") or {}
    if not isinstance(section, dict):
        raise ValueError("config 'determinism' must be a mapping")
    if "seed" not in section or section.get("seed") is None:
        return DeterminismMode()
    seed = section["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            f"determinism seed must be a non-negative int; got {seed!r}"
        )
    return DeterminismMode(seed=seed)


# ── The seeded id stream ──────────────────────────────────────────────────────


class DeterministicIds:
    """Mints record ids from a seeded stream instead of uuid4.

    Ids keep the standing shapes (``turn_`` + 8 hex,
    ``cand_`` / ``frag_`` + 6 hex, ``resp_`` + 8 hex — the
    caller names prefix and hex length); the hex comes from a
    dedicated ``random.Random`` seeded from a string, so the
    sequence is identical in every process for the same seed.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = random.Random(f"thoughtforge-determinism-ids-{seed}")
        self._minted = 0

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def minted(self) -> int:
        """How many ids this stream has minted so far."""
        return self._minted

    def new_id(self, prefix: str, hex_len: int) -> str:
        bits = self._rng.getrandbits(hex_len * 4)
        self._minted += 1
        return f"{prefix}_{bits:0{hex_len}x}"


def reset_deterministic_state(core: Any) -> None:
    """
    Restart one core's determinism state from its mode's seed,
    so the same script re-run on the SAME core reproduces the
    first run: the id and generation-seed streams re-derive
    from the top, and the will ledger (the core's one piece of
    cross-turn mutable state) is replaced with a fresh one —
    a re-run is a new conversation.

    Raises:
        ValueError: the core has no seed selected — there is
            nothing to reset, and a silent no-op would let a
            caller believe an unseeded run was seeded.
    """
    mode: DeterminismMode = core._determinism
    if not mode.enabled:
        raise ValueError(
            "core has no determinism seed selected — nothing to reset"
        )
    core._det_ids = mode.new_id_stream()
    core._det_gen_rng = mode.new_generation_rng()
    from thoughtforge.cognition.will import GoalLedger

    core._will = GoalLedger()


# ── The fingerprint: what "reproducible" means, as data ──────────────────────


def turn_fingerprint(turn: Any) -> dict[str, Any]:
    """
    One finished turn's reproducibility fingerprint: everything
    two runs of the same conversation must agree on, and nothing
    they cannot (timestamps and stage timings are excluded —
    see the census). Built from the retained ``RuntimeTurnState``
    (TF-025); raises ValueError on an unfinished turn, whose
    fingerprint would be a lie about a run still in flight.
    """
    if turn is None or turn.final_response is None:
        raise ValueError("turn has no final response — not a finished turn")
    scaffold = turn.scaffold
    final = turn.final_response
    return {
        "turn_id": turn.turn_id,
        "scaffold": {
            "intent": turn.input_sketch.intent if turn.input_sketch else None,
            "depth": scaffold.depth if scaffold else None,
            "candidate_modes": (
                list(scaffold.candidate_modes) if scaffold else None
            ),
            "goal": scaffold.goal if scaffold else None,
            "fact_block": scaffold.fact_block if scaffold else None,
        },
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "mode": c.mode,
                "text": c.text,
                "token_estimate": c.token_estimate,
            }
            for c in turn.candidates
        ],
        "fragments_kept": [
            {"fragment_id": f.fragment_id, "text": f.text}
            for f in turn.fragments
            if f.keep
        ],
        "final": {
            "response_id": final.response_id,
            "text": final.text,
            "mode": final.mode,
            "token_count": final.token_count,
            "scores": {
                "composite": final.scores.composite,
                "quality_tier": final.scores.quality_tier,
            },
            "enforcement_passed": final.enforcement_passed,
            "enforcement_notes": final.enforcement_notes,
            "salvage_path": final.salvage_path,
            "source_candidate_ids": list(final.source_candidate_ids),
            "source_fragment_ids": list(final.source_fragment_ids),
            "citations": list(final.citations),
        },
    }


def reproducibility_problems(turn_a: Any, turn_b: Any) -> list[str]:
    """
    Name every fingerprint field on which two finished turns
    differ. Empty = the two runs reproduced each other exactly,
    on the fingerprint's terms. Pure comparison of the two
    fingerprints — the field path of each difference is named
    (``final.text``, ``candidates[1].candidate_id``, …) so a
    flake report says WHAT drifted, not just that something did
    (TF-135 consumes this).
    """
    fp_a = turn_fingerprint(turn_a)
    fp_b = turn_fingerprint(turn_b)
    problems: list[str] = []

    def walk(path: str, a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
                walk(f"{path}.{key}" if path else key, a[key], b.get(key))
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                problems.append(
                    f"{path}: {len(a)} item(s) in run A vs {len(b)} in run B"
                )
                return
            for idx, (item_a, item_b) in enumerate(zip(a, b)):
                walk(f"{path}[{idx}]", item_a, item_b)
            return
        if a != b:
            problems.append(f"{path}: run A {a!r} != run B {b!r}")

    walk("", fp_a, fp_b)
    return problems


# ── The seeded runner ─────────────────────────────────────────────────────────


@dataclass
class SeededRun:
    """One script run on one core: the finals in order, and the
    per-turn fingerprints that judge reproducibility."""

    finals: list[Any]
    fingerprints: list[dict[str, Any]]


def run_seeded_script(
    core: Any,
    texts: list[str],
    *,
    reset: bool = True,
) -> SeededRun:
    """
    Run ``texts`` as sequential turns on ``core`` and return
    their finals + fingerprints. With ``reset`` (the default)
    the core's seeded streams are re-derived first, so calling
    this twice on one seeded core is the reproducibility proof
    itself. On an unseeded core the reset raises (see
    ``reset_deterministic_state``) — a benchmark that forgot
    its seed finds out here, loudly, before it runs anything.
    """
    if reset:
        reset_deterministic_state(core)
    finals: list[Any] = []
    fingerprints: list[dict[str, Any]] = []
    for text in texts:
        finals.append(core.think(text))
        fingerprints.append(turn_fingerprint(core.last_turn_state))
    return SeededRun(finals=finals, fingerprints=fingerprints)


__all__ = [
    "DETERMINISM_SOURCES",
    "RULING_DETERMINISTIC",
    "RULING_EXCLUDED",
    "RULING_SEEDED",
    "VALID_RULINGS",
    "DeterminismMode",
    "DeterminismSource",
    "DeterministicIds",
    "SeededRun",
    "reproducibility_problems",
    "reset_deterministic_state",
    "resolve_determinism_mode",
    "run_seeded_script",
    "source_ruling",
    "turn_fingerprint",
]
