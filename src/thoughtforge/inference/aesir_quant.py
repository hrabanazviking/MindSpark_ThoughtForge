"""TF-075 — Aesir + TurboQuant interplay: quantised models
served by Aesir, measured through ThoughtForge's fixtures.

Roadmap: "quantised models served by Aesir measured through
ThoughtForge's fixtures."

Measured start (probed this tick, VM):

- TF-074 bound Aesir as a backend, but nothing connected it
  to the quant estate: TF-067's panel ran only against its
  fixture corpus directly, TF-068's policy check was wired
  only into the *local* engine's load path
  (``TurboQuantEngine.load``), and the served path carried
  no quant check at all — filed as **D-AESIR-2** and fixed
  here, at identity grade (below).
- The TF-074 wire payload contains **no quant field**:
  the served chat subset is model / messages / stream /
  max_tokens / temperature / seed. Quant cannot be sent
  to Aesir; it can only be *the resident model Aesir is
  serving*. Aesir's own docs name that resident by its
  GGUF filename (``gemma-4-E2B-it-Q4_K_M.gguf``, Aesir
  ``docs/GEMMA4_CUDA.md`` @ ``1eed9706``), so the filename
  vocabulary TF-066 built for local files is the same
  vocabulary that proves a served resident's quant.
- Aesir serves exactly **one resident model, serialized**
  (TF-074's ledger): a quant panel through Aesir is
  therefore a serial sweep — one resident at a time, in
  panel order — never the parallel multi-model run a
  local harness might imagine.

THE INTERPLAY (``INTERPLAY_RULES`` as data):

- ``quant_is_identity_not_a_wire_field`` — a level reaches
  Aesir only as the resident model's identity; the payload
  never carries a quant key (pinned against the real
  backend's payload builder and against every payload the
  fixture server receives).
- ``one_resident_so_panel_is_serial`` — the panel runs
  levels in ``QUANT_LEVELS`` order, one resident at a
  time; the fixture server's call log proves the grouping
  (each level's eight calls contiguous, levels in order).
- ``same_panel_same_scorer_same_curve`` — the measurement
  is TF-067's own runner, prompts, and scorer, with only
  the generation source changed to a real
  :class:`AesirBackend` talking to a fake Aesir transport.
  The curve through Aesir is recomputed every run and
  must equal TF-067's published ``CURVE`` exactly: the
  binding transports answers, it does not bend scores.
- ``policy_check_at_identity_grade`` (D-AESIR-2, FIXED) —
  TF-068's ``check_quant_for_profile`` now also judges
  *served* residents: :func:`resident_verdict` applies it
  to the resident model's id, and :func:`profile_residents`
  files the expected resident for every shipped profile
  (status ``match`` for all six). The grade is honestly
  weaker than the local check: locally the engine loads
  the file whose name it checks; served, ThoughtForge
  sees only the identity string Aesir reports — a
  misnamed file on the server would pass this check and
  fail nothing here. Identity grade, never load grade.
- ``unprovable_resident_never_assumed`` — a resident id
  carrying no quant token (Aesir's catalog-style
  ``gemma4-e2b:latest``) proves no quant: TF-068's verdict
  is ``unprovable`` with its gap named, never rounded to
  a profile default and never called a match.
- ``serving_does_not_change_the_curve_class`` — TF-068's
  per-level classes carry through the served path
  unchanged: phone_low's and pi_zero's q2_k residents are
  still constrained exceptions (steep_loss, fixture loss
  0.5625) — being served by Aesir buys distribution, not
  quality.
- ``error_is_panel_failure_not_a_zero`` — if Aesir
  answers a panel prompt with an error (404 non-resident
  is the live shape), the panel run raises naming the
  resident and the server's error; no level is scored
  0.0 for a serve that never happened, and no curve is
  published from a partial sweep.
- ``usage_counts_come_from_the_server`` — panel answers
  carry Aesir's own ``usage.completion_tokens`` as
  ``tokens_generated`` when the server reports it
  (TF-074's rule, exercised here per level); the fixture
  server reports the contract estimate of each answer,
  so the counts are checkable, not decorative.

THE FIXTURE RESIDENTS: one synthetic resident per panel
level, named in Aesir's own GGUF-filename shape over
TF-017's Lantern Hollow fixture world
(``lantern-hollow-<QUANT>.gguf``); each id round-trips
through ``quant_of_filename`` to exactly its level, and
the set covers exactly ``QUANT_LEVELS``.

Proof grade FIXTURE (a fake Aesir transport speaking
Aesir's v1 shapes, serving TF-067's recorded corpus
through the real :class:`AesirBackend` and the real
panel runner, VM) — a live Aesir serve of quantised
models is NOT PROVEN (D-AESIR-1 stands, owned by
TF-086; no Aesir server answers on this VM), and this
slice measures the *transport's* fidelity to the
fixture curve, never a real model's quantisation loss,
never claimed.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import patch

from thoughtforge.inference.aesir_backend import (
    AesirBackend,
    _FakeResponse,
)
from thoughtforge.inference.backend_contract import estimate_tokens
from thoughtforge.inference.quant_policy import (
    LEVEL_CLASSES,
    POLICY_BY_PROFILE,
    QuantVerdict,
    check_quant_for_profile,
)
from thoughtforge.inference.quant_quality import (
    CURVE,
    FIXTURE_RESPONSES,
    QUANT_LEVELS,
    QUANT_PROMPTS,
    QuantPanel,
    QuantPrompt,
    run_quant_panel,
)
from thoughtforge.inference.turboquant_map import quant_of_filename
from thoughtforge.inference.unified_backend import GenerationRequest

AESIR_QUANT_FORMAT = "thoughtforge-aesir-quant-interplay/1"

# ── The fixture residents ──────────────────────────────────────
# One synthetic resident model per panel level, named in
# Aesir's own GGUF-filename shape (model-QUANT.gguf, as in
# Aesir's docs) over TF-017's synthetic fixture world.

RESIDENT_MODEL_IDS: dict[str, str] = {
    quant: f"lantern-hollow-{quant.upper()}.gguf"
    for quant in QUANT_LEVELS
}

PROMPTS_BY_TEXT: dict[str, QuantPrompt] = {
    p.text: p for p in QUANT_PROMPTS
}

# ── The interplay rules ────────────────────────────────────────

INTERPLAY_RULES: tuple[dict[str, Any], ...] = (
    {"id": "quant_is_identity_not_a_wire_field",
     "rule": "a quant level reaches Aesir only as the "
             "resident model's identity; the TF-074 wire "
             "payload carries no quant field of any name",
     "measured_by": "the real backend's payload keys pinned; "
                    "every payload the fixture server receives "
                    "carries the level only in its model id"},
    {"id": "one_resident_so_panel_is_serial",
     "rule": "Aesir serves one resident model, serialized "
             "(TF-074 ledger), so the panel sweeps levels "
             "one resident at a time in QUANT_LEVELS order",
     "measured_by": "the fixture server's call log: 48 calls, "
                    "each level's 8 contiguous, in panel order"},
    {"id": "same_panel_same_scorer_same_curve",
     "rule": "the measurement is TF-067's runner, prompts "
             "and scorer with only the generation source "
             "changed to the real AesirBackend; the curve "
             "through Aesir equals TF-067's CURVE exactly",
     "measured_by": "CURVE_THROUGH_AESIR recomputed from a "
                    "live fixture sweep on every checker run"},
    {"id": "policy_check_at_identity_grade",
     "rule": "TF-068's check_quant_for_profile judges served "
             "residents by their model id (D-AESIR-2 fixed); "
             "the grade is identity-only — ThoughtForge does "
             "not load the served file and a misnamed server "
             "file would pass, stated never hidden",
     "measured_by": "profile_residents(): all six profiles' "
                    "expected residents verdict match; the "
                    "grade stated in this module and the doc"},
    {"id": "unprovable_resident_never_assumed",
     "rule": "a resident id with no filename-provable quant "
             "(catalog/tag shape) yields verdict unprovable "
             "with its gap named — never a match, never a "
             "default assumed",
     "measured_by": "gemma4-e2b:latest (TF-074's own catalog "
                    "example) -> quant None, verdict "
                    "unprovable / filename_proves_no_quant"},
    {"id": "serving_does_not_change_the_curve_class",
     "rule": "TF-068's level classes and per-profile "
             "decisions carry through the served path "
             "unchanged; q2_k residents for phone_low / "
             "pi_zero remain constrained exceptions",
     "measured_by": "profile_residents() rows carry "
                    "LEVEL_CLASSES + the policy decision "
                    "verbatim from TF-068's POLICY"},
    {"id": "error_is_panel_failure_not_a_zero",
     "rule": "a backend error during the sweep (the live "
             "shape: 404 model-not-resident) raises naming "
             "the resident and the error; a partial sweep "
             "publishes no curve and scores no false zero",
     "measured_by": "the fixture server failing q2_k with "
                    "404 makes the run raise; no panel "
                    "object is produced"},
    {"id": "usage_counts_come_from_the_server",
     "rule": "tokens_generated on a panel answer is Aesir's "
             "reported usage.completion_tokens when the "
             "server reports it (TF-074), per level",
     "measured_by": "the fixture server reports the "
                    "contract estimate of each answer; a "
                    "direct generate per level returns "
                    "exactly that count"},
    {"id": "live_proof_not_claimed",
     "rule": "no live Aesir serve has carried a quant panel; "
             "every proof here is fixture grade and D-AESIR-1 "
             "stands (TF-086 owns the live smoke)",
     "measured_by": "this module, its doc, and the docket "
                    "carry the NOT PROVEN line"},
)


def quant_of_resident(model_id: str | None) -> str | None:
    """The filename-provable quant of a served resident's
    model id (TF-066's parser, unchanged), or None when the
    id proves none — the tag/catalog shape."""
    if not model_id:
        return None
    return quant_of_filename(model_id)


def resident_verdict(profile_id: str, model_id: str | None) -> QuantVerdict:
    """TF-068's policy check applied to a *served* resident:
    the same verdict type and rules as the local load check,
    at identity grade (see INTERPLAY_RULES)."""
    return check_quant_for_profile(profile_id, model_id)


def profile_residents() -> list[dict[str, Any]]:
    """The expected Aesir resident for every shipped
    profile: TF-068's chosen default as a fixture resident
    id, its verdict, its curve class, and the policy
    decision carried verbatim."""
    rows: list[dict[str, Any]] = []
    for profile_id in sorted(POLICY_BY_PROFILE):
        policy = POLICY_BY_PROFILE[profile_id]
        quant = policy["chosen"]
        resident = RESIDENT_MODEL_IDS[quant]
        verdict = resident_verdict(profile_id, resident)
        rows.append({
            "profile": profile_id,
            "quant": quant,
            "resident": resident,
            "verdict": verdict.status,
            "curve_class": LEVEL_CLASSES[quant],
            "decision": policy["decision"],
        })
    return rows


# ── Generation through Aesir ───────────────────────────────────


def generate_through_aesir(backend: AesirBackend,
                           prompt: QuantPrompt) -> str:
    """One panel prompt through one Aesir backend. A backend
    error is a panel failure (INTERPLAY_RULES): it raises
    naming the resident and the server's error rather than
    returning text a scorer could mistake for an answer."""
    resp = backend.generate(GenerationRequest(prompt=prompt.text))
    if resp.finish_reason == "error" or resp.error:
        raise RuntimeError(
            f"aesir panel: resident {backend.model!r} failed "
            f"prompt {prompt.prompt_id}: {resp.error}")
    return resp.text


def run_aesir_quant_panel(
    make_backend: Callable[[str], AesirBackend] | None = None,
) -> QuantPanel:
    """TF-067's panel with the generation source swapped to
    Aesir: one backend (one resident) per level, the
    identical prompt set at every level, loss against the
    panel's own fp16 row — the runner is TF-067's, unmodified."""
    if make_backend is None:
        make_backend = lambda quant: AesirBackend(  # noqa: E731
            model=RESIDENT_MODEL_IDS[quant])
    return run_quant_panel(
        lambda quant, prompt: generate_through_aesir(
            make_backend(quant), prompt))


# ── The fixture Aesir server ───────────────────────────────────


class FakeAesirQuantServer:
    """A fake Aesir serve at panel scale: it holds ONE
    resident per request (the request's own model id, as
    Aesir's serialized serve would after a resident swap),
    answers with TF-067's recorded corpus cell for
    (prompt, the resident's quant), and reports usage as
    the contract estimate of the answer. Levels named in
    ``fail_levels`` answer Aesir's native 404 shape — the
    model-not-resident error. Every call is recorded."""

    def __init__(self, fail_levels: frozenset[str] = frozenset()):
        self.fail_levels = fail_levels
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, **kwargs):
        payload = kwargs.get("json") or {}
        self.calls.append({"url": url, "json": payload})
        model = payload.get("model") or ""
        quant = quant_of_resident(model)
        if quant is None or quant in self.fail_levels:
            return _FakeResponse(404, {"error": {
                "code": 404,
                "message": f"model {model} is not resident"}})
        messages = payload.get("messages") or []
        text = messages[-1].get("content") if messages else None
        prompt = PROMPTS_BY_TEXT.get(text or "")
        if prompt is None:
            return _FakeResponse(422, {"error": {
                "code": 422,
                "message": "unknown prompt for this fixture serve"}})
        answer = FIXTURE_RESPONSES[(prompt.prompt_id, quant)]
        return _FakeResponse(200, {
            "id": "chatcmpl-aesir-quant",
            "object": "chat.completion", "created": 1,
            "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": estimate_tokens(prompt.text),
                      "completion_tokens": estimate_tokens(answer),
                      "total_tokens": estimate_tokens(prompt.text)
                      + estimate_tokens(answer)},
        })


def fixture_panel_through_aesir(
    fail_levels: frozenset[str] = frozenset(),
) -> tuple[QuantPanel, list[dict[str, Any]]]:
    """The published sweep: the panel run through the real
    AesirBackend against the fixture server. Returns the
    panel and the server's call log."""
    server = FakeAesirQuantServer(fail_levels=fail_levels)
    with patch("thoughtforge.inference.aesir_backend.requests.post",
               side_effect=server):
        panel = run_aesir_quant_panel()
    return panel, server.calls


# ── The published curve through Aesir ──────────────────────────
# Literal rows in TF-067's CURVE shape (quant, mean, loss,
# perfect, contradiction_prompts), recomputed by the checker
# from a live fixture sweep on every run.

CURVE_THROUGH_AESIR: tuple[tuple[str, float, float, int, int], ...] = (
    ("iq2_xs", 0.0, 1.0, 0, 7),
    ("q2_k", 0.4375, 0.5625, 2, 2),
    ("q4_k_m", 0.8125, 0.1875, 6, 1),
    ("q5_k_m", 0.7083, 0.2917, 4, 1),
    ("q8_0", 1.0, 0.0, 8, 0),
    ("fp16", 1.0, 0.0, 8, 0),
)


def aesir_quant_as_data() -> dict[str, Any]:
    return {
        "format": AESIR_QUANT_FORMAT,
        "rules": [dict(r) for r in INTERPLAY_RULES],
        "residents": dict(RESIDENT_MODEL_IDS),
        "profiles": profile_residents(),
        "curve_through_aesir": [list(r) for r in CURVE_THROUGH_AESIR],
    }


# ── Checkers ───────────────────────────────────────────────────


def aesir_quant_problems() -> list[str]:
    """The filed estate judged live (no panel sweep)."""
    problems: list[str] = []

    ids = [r["id"] for r in INTERPLAY_RULES]
    if len(ids) != len(set(ids)):
        problems.append("INTERPLAY_RULES ids are not unique")
    if set(RESIDENT_MODEL_IDS) != set(QUANT_LEVELS):
        problems.append("resident levels != QUANT_LEVELS")
    for quant, resident in RESIDENT_MODEL_IDS.items():
        if quant_of_resident(resident) != quant:
            problems.append(
                f"resident {resident!r} proves "
                f"{quant_of_resident(resident)!r}, not {quant!r}")

    # Quant is identity, never a wire field: the real
    # backend's payload for a resident request has no
    # quant-shaped key.
    payload = AesirBackend(
        model=RESIDENT_MODEL_IDS["q4_k_m"])._payload(
        GenerationRequest(prompt="x"), stream=False)
    quant_keys = [k for k in payload
                  if "quant" in k.lower()]
    if quant_keys:
        problems.append(f"payload carries quant keys {quant_keys}")
    if payload.get("model") != RESIDENT_MODEL_IDS["q4_k_m"]:
        problems.append("payload does not carry the resident "
                        "model id (the quant's only channel)")

    # The unprovable shape (TF-074's catalog example):
    if quant_of_resident("gemma4-e2b:latest") is not None:
        problems.append("tag-shaped resident id proved a quant")
    verdict = resident_verdict("desktop_cpu", "gemma4-e2b:latest")
    if verdict.status != "unprovable" or \
            verdict.gap != "filename_proves_no_quant":
        problems.append(f"unprovable resident verdict is {verdict!r}")

    for row in profile_residents():
        if row["verdict"] != "match":
            problems.append(
                f"{row['profile']}: expected resident verdict "
                f"{row['verdict']}, not match")
        if row["quant"] != POLICY_BY_PROFILE[row["profile"]]["chosen"]:
            problems.append(
                f"{row['profile']}: resident quant is not the "
                f"TF-068 policy default")
        if row["curve_class"] != LEVEL_CLASSES[row["quant"]]:
            problems.append(
                f"{row['profile']}: curve class drifted from "
                f"TF-068's LEVEL_CLASSES")
    exceptions = {r["profile"] for r in profile_residents()
                  if r["decision"] == "keep_constrained_exception"}
    if exceptions != {"phone_low", "pi_zero"}:
        problems.append(
            f"served constrained exceptions {sorted(exceptions)} "
            f"!= TF-068's phone_low/pi_zero")

    if CURVE_THROUGH_AESIR != CURVE:
        problems.append("CURVE_THROUGH_AESIR filed != TF-067's CURVE")
    return problems


def aesir_quant_live_problems() -> list[str]:
    """The sweep itself, re-derived live over the fixture
    server (this VM)."""
    problems: list[str] = []

    panel, calls = fixture_panel_through_aesir()
    recomputed = {
        (r.quant, round(r.mean_score, 4),
         round(r.loss_vs_reference, 4), r.perfect,
         r.contradiction_prompts)
        for r in panel.rows
    }
    filed = {(q, round(m, 4), round(l, 4), p, c)
             for q, m, l, p, c in CURVE_THROUGH_AESIR}
    if recomputed != filed:
        problems.append(
            f"curve through Aesir drifted: {sorted(recomputed)} "
            f"!= {sorted(filed)}")

    # Serial sweep shape: 48 calls, levels grouped
    # contiguously in QUANT_LEVELS order, 8 per level, the
    # level present only as the model id.
    if len(calls) != len(QUANT_LEVELS) * len(QUANT_PROMPTS):
        problems.append(f"sweep made {len(calls)} calls, not 48")
    sequence = [quant_of_resident(c["json"].get("model") or "")
                for c in calls]
    expected = [q for q in QUANT_LEVELS for _ in QUANT_PROMPTS]
    if sequence != expected:
        problems.append("call sequence is not the serial "
                        "QUANT_LEVELS sweep")
    for c in calls:
        if any("quant" in k.lower() for k in c["json"]):
            problems.append("a sweep payload carried a quant key")
            break

    # Determinism: a second sweep, identical rows.
    again, _ = fixture_panel_through_aesir()
    if [(r.quant, r.mean_score) for r in again.rows] != [
            (r.quant, r.mean_score) for r in panel.rows]:
        problems.append("the Aesir sweep is not deterministic")

    # Usage leg: a direct generate per level reports the
    # server's completion_tokens (the fixture server's
    # contract estimate of the answer).
    server = FakeAesirQuantServer()
    prompt = QUANT_PROMPTS[0]
    with patch("thoughtforge.inference.aesir_backend.requests.post",
               side_effect=server):
        for quant in QUANT_LEVELS:
            backend = AesirBackend(model=RESIDENT_MODEL_IDS[quant])
            resp = backend.generate(
                GenerationRequest(prompt=prompt.text))
            expected_tokens = estimate_tokens(
                FIXTURE_RESPONSES[(prompt.prompt_id, quant)])
            if resp.tokens_generated != expected_tokens:
                problems.append(
                    f"{quant}: tokens_generated "
                    f"{resp.tokens_generated} != the server's "
                    f"reported {expected_tokens}")
                break

    # Error leg: a 404 level fails the whole run, naming it.
    try:
        fixture_panel_through_aesir(fail_levels=frozenset({"q2_k"}))
        problems.append("a 404 resident did not fail the panel run")
    except RuntimeError as exc:
        if "q2_k" not in str(exc) and \
                RESIDENT_MODEL_IDS["q2_k"] not in str(exc):
            problems.append(f"panel failure did not name the "
                            f"resident: {exc}")
    return problems


__all__ = [
    "AESIR_QUANT_FORMAT",
    "CURVE_THROUGH_AESIR",
    "INTERPLAY_RULES",
    "RESIDENT_MODEL_IDS",
    "FakeAesirQuantServer",
    "aesir_quant_as_data",
    "aesir_quant_live_problems",
    "aesir_quant_problems",
    "fixture_panel_through_aesir",
    "generate_through_aesir",
    "profile_residents",
    "quant_of_resident",
    "resident_verdict",
    "run_aesir_quant_panel",
]
