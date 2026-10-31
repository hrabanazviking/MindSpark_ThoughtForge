"""TF-068 — Quant selection policy: the ADR that chooses each
profile's default quant *from TF-067's measured curve*, and
the enforcement that ends the silent load (D-QUANT-1).

ADR shape (the full note is
``docs/architecture/QUANT_SELECTION_POLICY.md``):

- **Status**: accepted, this slice. Supersedes: nothing — the
  profile JSONs' ``model.quantization`` values were shipped
  recommendations with no stated basis (TF-066's map).
  Superseded by: nobody yet; re-review triggers are named
  per exception below and TF-127's live panel can reopen
  any row.
- **Context**: TF-067 measured the first curve in the tree
  (fixture grade): iq2_xs 0.0000 (loss 1.0000, 7 of 8 prompts
  contradicted), q2_k 0.4375 (loss 0.5625), q4_k_m 0.8125
  (loss 0.1875), q5_k_m 0.7083 (loss 0.2917), q8_0 1.0000 and
  fp16 1.0000 (parity). The curve is *not monotone*: q4_k_m
  out-scores q5_k_m on the eight-prompt corpus, a step
  TF-067 filed as data (``NON_MONOTONE_STEPS``) — an
  eight-prompt corpus shows the curve's *shape*, it cannot
  rank neighbouring levels. No size or fit measurement for
  any quant at any profile exists in-tree (TF-066: no
  bits-per-weight figure is stated anywhere), so this policy
  raises no default on a fit guess it cannot prove.
- **Decision**: every shipped default is re-derived from
  the curve's shape and ratified or excepted, per profile,
  in ``POLICY`` below — no default changes this slice,
  because the curve gives no proven basis to change one:
  the four defaults at or above the quality floor are
  *kept on the curve's evidence*; the two below it are
  *kept only as named constrained exceptions* with their
  measured cost stated and a re-review trigger owned by a
  named later slice. The rules (``POLICY_RULES``):

  1. ``quality_floor`` — a default must sit in the working
     band or at parity (curve classes below), unless its
     row is a recorded constrained exception.
  2. ``never_catastrophic`` — iq2_xs (mean 0.0, a
     contradiction on 7 of 8 prompts) is never a default,
     for any profile, at any size pressure.
  3. ``neighbours_not_ranked`` — inside the working band
     the fixture corpus does not rank q4_k_m against
     q5_k_m (the non-monotone step); q5_k_m is therefore
     not promoted over a shipped q4_k_m default, and no
     row cites this corpus as a neighbour ranking.
  4. ``parity_needs_no_fp16`` — where q8_0 measures at
     parity with fp16 (loss 0.0), the curve gives no
     quality reason to prefer fp16; desktop_gpu's shipped
     q8_0 default stands on that evidence.
  5. ``constrained_exception`` — a below-floor default is
     a size-class exception, never a quality claim: the
     row states the measured fixture cost and the trigger
     that forces re-review (TF-069's budget-enforced
     profile runner, TF-082's real-Pi proof, TF-127's
     live panel). If a trigger proves a floor-level quant
     fits the profile's class, the default must rise.
  6. ``announced_not_silent`` (D-QUANT-1, FIXED here) —
     a recommendation nobody reads is not a policy. The
     engine now checks, at load time, the loaded file's
     filename-provable quant against the profile's policy
     default (:func:`check_quant_for_profile`, wired in
     ``TurboQuantEngine.load``): a match is recorded, a
     mismatch is recorded *and* announced (warning log +
     the verdict on the engine), and a filename that
     proves no quant — or only a generic class, TF-066's
     under-specification gap — is recorded as
     ``unprovable``, never rounded up and never called a
     mismatch. Enforcement is announcement, not refusal:
     the operator may deliberately load another quant, but
     never again unknowingly.

- **Consequences**: D-QUANT-1 closes (TF-016: 60 entries —
  38 OPEN / 1 OPEN-ENVIRONMENT / 19 FIXED / 2
  CLOSED-BY-VERDICT). TF-066's duties table moves
  ``enforce_profile_quant`` to does, with the honest scope:
  the check reads the *filename* (TF-066's
  ``quant_of_filename``), not the file's header — header
  inspection stays a does-not. The two constrained
  exceptions (phone_low, pi_zero at q2_k, fixture loss
  0.5625 each) are the policy's open debt, owned by the
  trigger slices named in their rows. Proof grade stays
  FIXTURE, exactly TF-067's: this policy reasons from a
  fixture curve's shape, and TF-127's live panel is the
  road that can overturn any row.
"""

from __future__ import annotations

from dataclasses import dataclass

from thoughtforge.inference.quant_quality import CURVE, QUANT_LEVELS
from thoughtforge.inference.turboquant_map import (
    GENERIC_QUANTS,
    profile_quants,
    quant_of_filename,
)

POLICY_FORMAT = "thoughtforge-quant-selection-policy/1"

# ── Curve classes (from TF-067's published curve) ───────────────────────────
# Filed as data against the CURVE rows; the checker recomputes
# each class from the curve's own numbers under these rules:
# mean 0.0 -> catastrophic; loss 0.0 -> parity; mean >= 0.70 ->
# working_band; anything else -> steep_loss.

WORKING_BAND_MIN_MEAN = 0.70


def curve_class(mean_score: float, loss: float) -> str:
    if mean_score == 0.0:
        return "catastrophic"
    if loss == 0.0:
        return "parity"
    if mean_score >= WORKING_BAND_MIN_MEAN:
        return "working_band"
    return "steep_loss"


_CURVE_BY_QUANT = {row[0]: row for row in CURVE}

LEVEL_CLASSES: dict[str, str] = {
    quant: curve_class(mean, loss)
    for quant, mean, loss, _perfect, _contra in CURVE
}

AT_OR_ABOVE_FLOOR = frozenset({"working_band", "parity"})

# ── The rules ───────────────────────────────────────────────────────────────

POLICY_RULES: tuple[dict, ...] = (
    {"id": "quality_floor",
     "rule": "a profile's default quant sits in the working band or at "
             "parity on TF-067's curve, unless its row is a recorded "
             "constrained exception"},
    {"id": "never_catastrophic",
     "rule": "iq2_xs (curve mean 0.0, contradictions on 7 of 8 prompts) "
             "is never a default for any profile"},
    {"id": "neighbours_not_ranked",
     "rule": "the fixture corpus does not rank neighbours inside the "
             "working band (TF-067's non-monotone q4_k_m/q5_k_m step); "
             "q5_k_m is not promoted over a shipped q4_k_m default on "
             "this evidence"},
    {"id": "parity_needs_no_fp16",
     "rule": "where a level measures at parity with the fp16 reference "
             "(loss 0.0), the curve gives no quality reason to prefer "
             "fp16 over it"},
    {"id": "constrained_exception",
     "rule": "a below-floor default is a size-class exception, never a "
             "quality claim: the row states the measured fixture cost "
             "and a named re-review trigger; a trigger that proves a "
             "floor-level quant fits forces the default up"},
    {"id": "announced_not_silent",
     "rule": "at load time the engine checks the loaded file's "
             "filename-provable quant against the profile's policy "
             "default and records the verdict; a mismatch is announced, "
             "an unprovable filename is recorded as unprovable — never "
             "silently passed, never a hard refusal (D-QUANT-1)"},
)

# ── The policy table ────────────────────────────────────────────────────────
# One row per shipped hardware profile. ``shipped`` is the
# profile JSON's own value (the checker pins it live against
# TF-066's profile_quants()); ``chosen`` is this ADR's decision
# from the curve. Curve figures are TF-067's published rows,
# pinned by the checker against CURVE itself.

POLICY: tuple[dict, ...] = (
    {"profile": "phone_low", "shipped": "q2_k", "chosen": "q2_k",
     "decision": "keep_constrained_exception",
     "basis": "below the floor on the curve (steep_loss) — kept only as "
              "a size-class exception for the ~2 GB / 1.1B-parameter "
              "phone class the profile declares; not a quality claim",
     "review_trigger": "TF-069 profile runner budgets + TF-127 live "
                       "panel: if q4_k_m-class is proven to fit this "
                       "class, the default rises to the floor"},
    {"profile": "pi_zero", "shipped": "q2_k", "chosen": "q2_k",
     "decision": "keep_constrained_exception",
     "basis": "below the floor on the curve (steep_loss) — kept only as "
              "a size-class exception for the 0.5 GB / 0.5B-parameter "
              "class the profile declares (its knowledge_subset "
              "profile); not a quality claim",
     "review_trigger": "TF-082 Pi profile proof on real hardware + "
                       "TF-127 live panel: if q4_k_m-class is proven "
                       "to fit this class, the default rises to the "
                       "floor"},
    {"profile": "pi_5", "shipped": "q4_k_m", "chosen": "q4_k_m",
     "decision": "keep",
     "basis": "the working band's floor on the curve; q5_k_m is not "
              "promoted over it (neighbours_not_ranked)",
     "review_trigger": "TF-127 live panel may re-rank the band"},
    {"profile": "desktop_cpu", "shipped": "q4_k_m", "chosen": "q4_k_m",
     "decision": "keep",
     "basis": "the working band's floor on the curve; q5_k_m is not "
              "promoted over it (neighbours_not_ranked)",
     "review_trigger": "TF-127 live panel may re-rank the band"},
    {"profile": "desktop_gpu", "shipped": "q8_0", "chosen": "q8_0",
     "decision": "keep",
     "basis": "parity with the fp16 reference on the curve (loss 0.0) "
              "— parity_needs_no_fp16",
     "review_trigger": "TF-127 live panel may break the parity"},
    {"profile": "server_gpu", "shipped": "fp16", "chosen": "fp16",
     "decision": "keep",
     "basis": "the reference level itself: no loss, and this class "
              "carries no size pressure that buys anything by "
              "compressing",
     "review_trigger": "TF-127 live panel is run against this row "
                       "first (it is the panel's reference)"},
)

POLICY_BY_PROFILE: dict[str, dict] = {row["profile"]: row for row in POLICY}

CONSTRAINED_EXCEPTIONS = frozenset(
    row["profile"] for row in POLICY
    if row["decision"] == "keep_constrained_exception"
)


def default_for_profile(profile_id: str) -> str:
    """The policy's chosen default quant for a profile."""
    try:
        return POLICY_BY_PROFILE[profile_id]["chosen"]
    except KeyError:
        raise ValueError(f"no quant policy row for profile {profile_id}") from None


def policy_as_data() -> dict:
    """The whole policy as JSON-safe data (fresh copy per call)."""
    rows = []
    for row in POLICY:
        _q, mean, loss, _p, _c = _CURVE_BY_QUANT[row["chosen"]]
        rows.append({
            **row,
            "curve_mean": mean,
            "curve_loss": loss,
            "curve_class": LEVEL_CLASSES[row["chosen"]],
        })
    return {
        "format": POLICY_FORMAT,
        "rules": [dict(r) for r in POLICY_RULES],
        "level_classes": dict(LEVEL_CLASSES),
        "profiles": rows,
        "constrained_exceptions": sorted(CONSTRAINED_EXCEPTIONS),
    }


# ── Enforcement: the load-time check (D-QUANT-1) ────────────────────────────

@dataclass(frozen=True)
class QuantVerdict:
    """One load's quant check, recorded on the engine."""

    profile_id: str
    expected: str               # the policy default for the profile
    found: str | None           # the filename-provable quant, if any
    status: str                 # match | mismatch | unprovable | no_model
    gap: str | None = None      # why unprovable, when it is


def check_quant_for_profile(profile_id: str, filename: str | None) -> QuantVerdict:
    """Judge a model file's filename-provable quant against
    the profile's policy default. A generic filename token
    (TF-066's ``-q4`` class) proves a class, not the variant —
    unprovable, never rounded up to the default, never a
    mismatch. ``filename=None`` is the no-model load."""
    expected = default_for_profile(profile_id)
    if filename is None:
        return QuantVerdict(profile_id, expected, None, "no_model")
    found = quant_of_filename(filename)
    if found is None:
        return QuantVerdict(profile_id, expected, None, "unprovable",
                            gap="filename_proves_no_quant")
    if found in GENERIC_QUANTS:
        return QuantVerdict(profile_id, expected, found, "unprovable",
                            gap="filename_under_specifies_variant")
    if found == expected:
        return QuantVerdict(profile_id, expected, found, "match")
    return QuantVerdict(profile_id, expected, found, "mismatch")


# ── Live checker ────────────────────────────────────────────────────────────

def quant_policy_problems() -> list[str]:
    """Re-derive the whole policy live; [] means it is true."""
    problems: list[str] = []

    live = profile_quants()
    if {row["profile"] for row in POLICY} != set(live):
        problems.append(
            f"policy profiles {sorted(r['profile'] for r in POLICY)} "
            f"!= shipped profiles {sorted(live)}")
    for row in POLICY:
        pid = row["profile"]
        if pid in live and row["shipped"] != live[pid]:
            problems.append(f"{pid}: shipped filed {row['shipped']} "
                            f"!= live profile value {live[pid]}")
        if row["chosen"] not in QUANT_LEVELS:
            problems.append(f"{pid}: chosen {row['chosen']} is not a "
                            f"panel level")
        if row["chosen"] not in _CURVE_BY_QUANT:
            problems.append(f"{pid}: chosen {row['chosen']} has no curve row")
            continue
        cls = LEVEL_CLASSES[row["chosen"]]
        if row["decision"] == "keep" and cls not in AT_OR_ABOVE_FLOOR:
            problems.append(f"{pid}: kept default {row['chosen']} is "
                            f"{cls}, below the quality floor")
        if row["decision"] == "keep_constrained_exception" \
                and cls in AT_OR_ABOVE_FLOOR:
            problems.append(f"{pid}: recorded as an exception but "
                            f"{row['chosen']} is {cls}, at/above the floor")
        if row["chosen"] == "iq2_xs":
            problems.append(f"{pid}: catastrophic level chosen as a "
                            f"default (never_catastrophic)")

    expected_exceptions = {
        row["profile"] for row in POLICY
        if LEVEL_CLASSES.get(row["chosen"]) not in AT_OR_ABOVE_FLOOR
    }
    if set(CONSTRAINED_EXCEPTIONS) != expected_exceptions:
        problems.append(
            f"constrained exceptions {sorted(CONSTRAINED_EXCEPTIONS)} != "
            f"the below-floor rows {sorted(expected_exceptions)}")
    for pid in CONSTRAINED_EXCEPTIONS:
        row = POLICY_BY_PROFILE[pid]
        _q, _mean, loss, _p, _c = _CURVE_BY_QUANT[row["chosen"]]
        if loss <= 0.0 or not row.get("review_trigger"):
            problems.append(f"{pid}: exception without a stated cost "
                            f"or review trigger")
    if any(row["chosen"] == "q5_k_m" for row in POLICY):
        problems.append("a row chose q5_k_m — the fixture corpus cannot "
                        "rank it over q4_k_m (neighbours_not_ranked)")

    # Curve classes recomputed from the raw curve numbers:
    for quant, mean, loss, _p, _c in CURVE:
        if LEVEL_CLASSES[quant] != curve_class(mean, loss):
            problems.append(f"level class for {quant} drifted")

    # The enforcement wiring is live in the engine:
    from pathlib import Path

    import thoughtforge.inference.turboquant as tq

    source = Path(tq.__file__).read_text(encoding="utf-8")
    if "check_quant_for_profile" not in source:
        problems.append("engine no longer calls the quant policy check "
                        "— D-QUANT-1 reopened")

    return problems
