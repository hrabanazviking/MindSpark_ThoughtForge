"""TF-078 — Model cards in-repo: per supported model, the
measured truth (size, quant, profiles proven, scaffold
behaviour).

Roadmap: "Model cards in-repo: per supported model, the
measured truth (size, quant, profiles proven, scaffold
behaviour)."

Measured start (probed this tick, VM):

- **No per-model card existed.** Three surfaces named
  models and none agreed as one estate: the curated
  catalogue (``ModelBrowser.CURATED_MODELS``, 15 entries
  with model_id / filename / size_gb / tier /
  description / quantization), the hardware profiles'
  ``recommended_models`` (15 distinct names across six
  JSONs, no sizes anywhere), and the root
  ``MODEL_CARD.md`` compatible-models table (6 rows —
  a framework card, not per-model cards). Nothing in
  the tree joins them.
- **The catalogue is the only supported set with
  truth to file**: it alone carries a size and a quant
  per model. Its sizes are *declared* — no GGUF file
  exists anywhere in this tree (rglob: none) and
  ``llama_cpp`` is not installed, so no catalogue size
  has ever been weighed and no catalogue model has
  ever been loaded here (TF-066/TF-067's standing).
- **Quant truth splits 13 / 2**: TF-066's
  ``quant_of_filename`` proves the declared quant
  exactly for 13 of 15 filenames; the Phi-3 pair
  (``-q4`` / ``-q5``) proves only the generic class —
  TF-066's filed ``FILENAME_QUANT_GAPS`` — and a card
  must not round a generic up to the declared K_M
  variant.
- **Profiles proven, honestly**: no model is proven
  on any profile live. What exists is *simulated*
  truth: TF-069 boots each profile's stack in
  simulation (no specific model loaded) and TF-037
  proves the core at the phone_low floor with a
  fixture backend. A card therefore files
  ``profiles_proven`` as empty for every model and
  files the *fit* matrix instead (below) under its
  own name — fit is not proof.
- **Scaffold behaviour is model-blind**: the scaffold
  builder's signature is (sketch, bundle, personality)
  — no model enters it; the registry varies by intent
  alone (TF-021). What varies per deployment is the
  *profile's* generation shape (draft count,
  refinement passes, response/token budgets, context)
  — TF-032/TF-037 made the profile's token cap and
  refinement cap live in the core, and the engine's
  draft count comes from the profile (TF-069's
  resolver). A card's scaffold behaviour is therefore
  its fitting profiles' shapes, filed as such.

THE CARDS (``CARD_RULES`` as data; one
:class:`ModelCard` per catalogue entry, built live):

- ``supported_means_catalogued`` — a card exists for
  exactly the 15 curated entries; a name with no
  catalogue entry gets no card and no invented size.
- ``size_is_catalogue_declared`` — ``size_gb`` is the
  catalogue's declared figure, basis stated on the
  card; it is never presented as weighed.
- ``quant_is_filename_proven_or_generic`` — the
  card's quant truth is TF-066's filename verdict:
  ``proven`` (filename == declared, 13 entries),
  ``generic`` (filename proves only q4/q5, the Phi-3
  pair), never rounded up.
- ``curve_is_per_quant_not_per_model`` — the card
  carries TF-067's fixture curve row + TF-068's curve
  class *for its proven quant level*: a property of
  the level over an 8-prompt fixture corpus, not a
  measurement of this model. Generic-grade cards
  carry no curve row.
- ``fit_is_simulated_not_proven`` — per profile,
  ``fits`` / ``exceeds`` / ``unprovable`` from two
  declared figures only: the catalogue size against
  the profile's declared RAM+VRAM ceiling, and the
  name-parsed parameter count against the profile's
  ``max_params_b`` (TF-069's parser; the Phi-3 pair
  carries no parseable count, so every size-passing
  cell for them is ``unprovable``). Filed summary
  over the 90 cells: 28 fits / 53 exceeds /
  9 unprovable — including both Qwen-72B entries
  exceeding even server_gpu (72B > its 70B cap) and
  Qwen-14B exceeding desktop_gpu's 13B cap despite
  its gpu tier.
- ``profiles_proven_is_empty`` — every card's
  ``profiles_proven`` is ``()`` and its
  ``live_proof`` is ``not_proven``: no GGUF in-tree,
  no llama_cpp on this VM. TF-082/TF-126/TF-173 own
  the roads that could change a card.
- ``scaffold_is_model_blind`` — ``scaffold_behaviour``
  files each shipped profile's generation shape
  resolved live from its JSON; the checker pins the
  builder's model-free signature so the claim is
  re-proven, not asserted.
- ``recommended_is_not_carded`` — the profiles'
  15 distinct recommended names are audited against
  the catalogue through the filed ``RECOMMENDED_AUDIT``
  alias table: 6 ``carded`` (family + quant carried),
  5 ``family_only`` (family carried, this quant or
  variant not), 4 ``no_card`` (Granite-1B, Gemma-2B,
  LLaMA-3-13B, Mixtral-8x7B — recommended with no
  size truth anywhere in-tree). 8 catalogue entries
  are recommended by no profile. D-CARD-2 (TF-016).
- ``root_card_reconciled_not_edited`` — the root
  ``MODEL_CARD.md`` table is parsed live and judged
  (``MODEL_CARD_DISCREPANCIES``): server_gpu's row
  pairs Q4_K_M against the profile's TF-068-ratified
  fp16 default and names Mixtral, which has no card;
  phone_low's Q2_K TinyLlama pairing exists in the
  catalogue only at Q4_K_M; pi_zero's "subset" model
  and pi_5's "128k" variant name no catalogue entry.
  3 of 6 rows drift. The root card itself is not
  edited here — TF-136's claims audit + TF-183's doc
  care own its correction, with these cards as their
  evidence. D-CARD-1 (TF-016).
- ``live_proof_not_claimed`` — below.

Proof grade FIXTURE/STATIC (the real catalogue,
profile JSONs, TF-066/067/068 surfaces and root card
read live on this VM) — no model loaded, no size
weighed, no per-model quality measured; a live
model's behaviour is NOT PROVEN for every card,
never claimed.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thoughtforge.inference.model_browser import ModelBrowser
from thoughtforge.inference.profile_runner import params_from_name
from thoughtforge.inference.quant_policy import LEVEL_CLASSES
from thoughtforge.inference.quant_quality import CURVE
from thoughtforge.inference.turboquant_map import (
    FILENAME_QUANT_GAPS,
    GENERIC_QUANTS,
    quant_of_filename,
)
from thoughtforge.utils.config import (
    list_hardware_profiles,
    load_hardware_profile,
)

CARDS_FORMAT = "thoughtforge-model-cards/1"

# ── The rules ───────────────────────────────────────────────────────────────

CARD_RULES: tuple[dict, ...] = (
    {"id": "supported_means_catalogued",
     "rule": "a card exists for exactly the curated catalogue's "
             "entries; a name with no catalogue entry gets no card "
             "and no invented size"},
    {"id": "size_is_catalogue_declared",
     "rule": "size_gb is the catalogue's declared figure, its basis "
             "stated on the card; no size in-tree has been weighed "
             "(no GGUF exists in the tree)"},
    {"id": "quant_is_filename_proven_or_generic",
     "rule": "the card's quant truth is TF-066's filename verdict — "
             "proven when the filename proves the declared level, "
             "generic when it proves only q4/q5 (the Phi-3 pair); "
             "a generic is never rounded up to the declared variant"},
    {"id": "curve_is_per_quant_not_per_model",
     "rule": "the TF-067 fixture curve row + TF-068 class a card "
             "carries belong to its proven quant LEVEL over an "
             "8-prompt fixture corpus, not to this model; "
             "generic-grade cards carry no curve row"},
    {"id": "fit_is_simulated_not_proven",
     "rule": "per-profile fit (fits / exceeds / unprovable) is "
             "judged from declared figures only — catalogue size vs "
             "the profile's RAM+VRAM ceiling, name-parsed params vs "
             "its max_params_b; an unparseable param count makes a "
             "size-passing cell unprovable, never a fit"},
    {"id": "profiles_proven_is_empty",
     "rule": "every card's profiles_proven is empty and its "
             "live_proof is not_proven: no catalogue model has "
             "been loaded on any profile in this tree"},
    {"id": "scaffold_is_model_blind",
     "rule": "scaffolds vary by intent and profile, never by model "
             "(the builder takes sketch + bundle + personality); a "
             "card's scaffold behaviour is its fitting profiles' "
             "generation shapes, resolved live"},
    {"id": "recommended_is_not_carded",
     "rule": "a profile's recommended name is audited, not assumed "
             "carded: carded / family_only / no_card through the "
             "filed alias table, recomputed live"},
    {"id": "root_card_reconciled_not_edited",
     "rule": "the root MODEL_CARD.md table is parsed and judged "
             "live; its correction belongs to TF-136/TF-183 with "
             "these cards as evidence — this slice edits no part "
             "of it"},
    {"id": "live_proof_not_claimed",
     "rule": "no card claims a loaded model, a weighed size, or a "
             "per-model quality figure; TF-082/TF-126/TF-173 own "
             "the roads that could prove one"},
)

# ── Filed summaries (recomputed by the checker) ─────────────────────────────

#: Fit verdicts over all card x profile cells (15 x 6 = 90).
FIT_SUMMARY: dict[str, int] = {"fits": 28, "exceeds": 53, "unprovable": 9}

#: (profile, kind) pairs where the root MODEL_CARD.md table
#: drifts from the profiles / catalogue it names.
MODEL_CARD_DISCREPANCIES: frozenset[tuple[str, str]] = frozenset({
    ("phone_low", "quant_not_in_catalogue"),
    ("pi_zero", "model_not_in_catalogue"),
    ("pi_zero", "quant_not_in_catalogue"),
    ("pi_5", "model_variant_not_in_catalogue"),
    ("server_gpu", "model_not_in_catalogue"),
    ("server_gpu", "quant_differs_from_profile_default"),
})

#: Family aliases: a family is recognised in a model text
#: (a recommended name, a root-card row, a catalogue id or
#: filename) when any of its spellings appears, lowercased.
FAMILY_SPELLINGS: dict[str, tuple[str, ...]] = {
    "tinyllama": ("tinyllama",),
    "phi-3-mini": ("phi-3-mini", "phi-3 mini"),
    "llama-3.2-3b": ("llama-3.2-3b",),
    "mistral-7b": ("mistral-7b",),
    "llama-3-8b": ("llama-3-8b", "llama-3.1-8b"),
    "qwen-14b": ("qwen2.5-14b",),
    "mistral-nemo": ("mistral-nemo",),
    "llama-3-70b": ("llama-3-70b", "llama-3.1-70b"),
    "qwen-72b": ("qwen2.5-72b",),
    "granite-1b": ("granite",),
    "gemma-2b": ("gemma-2b", "gemma-2b-it"),
    "llama-3-13b": ("llama-3-13b",),
    "mixtral": ("mixtral",),
}

#: The recommended-names audit: one row per distinct name the
#: shipped profiles recommend. ``card`` names the catalogue
#: filename carrying the family at the named quant (verdict
#: carded), the family's nearest catalogue filename
#: (family_only), or None (no_card). Version drift inside a
#: carded row (v0.3 recommended vs v0.2 catalogued; LLaMA-3
#: vs 3.1 naming) is filed in the note, never hidden.
RECOMMENDED_AUDIT: tuple[dict, ...] = (
    {"name": "TinyLlama-1.1B-Chat", "verdict": "family_only",
     "card": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
     "note": "no quant named; the catalogue carries TinyLlama "
             "only at Q4_K_M"},
    {"name": "Granite-1B-Instruct", "verdict": "no_card",
     "card": None,
     "note": "no Granite entry anywhere in the catalogue"},
    {"name": "Phi-3-mini-4k-instruct-q4_k_m", "verdict": "carded",
     "card": "Phi-3-mini-4k-instruct-q4.gguf",
     "note": "carded at declared grade; the filename proves "
             "only generic q4 (TF-066 gap)"},
    {"name": "Gemma-2B-it-q4_k_m", "verdict": "no_card",
     "card": None,
     "note": "no Gemma entry anywhere in the catalogue"},
    {"name": "TinyLlama-1.1B-Chat-q4_k_m", "verdict": "carded",
     "card": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
     "note": "family + quant carried"},
    {"name": "TinyLlama-1.1B-Chat-q2_k_subset", "verdict": "family_only",
     "card": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
     "note": "the q2_k subset variant is not catalogued"},
    {"name": "Mistral-7B-Instruct-v0.3-q4_k_m", "verdict": "carded",
     "card": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
     "note": "family + quant carried; version drift v0.3 "
             "recommended vs v0.2 catalogued"},
    {"name": "LLaMA-3-8B-Instruct-q4_k_m", "verdict": "carded",
     "card": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "note": "family + quant carried under the catalogue's "
             "3.1 naming"},
    {"name": "Phi-3-mini-128k-instruct-q8_0", "verdict": "family_only",
     "card": "Phi-3-mini-4k-instruct-q4.gguf",
     "note": "the 128k variant and q8_0 are not catalogued "
             "for Phi-3"},
    {"name": "LLaMA-3-8B-Instruct-q8_0", "verdict": "carded",
     "card": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
     "note": "family + quant carried under the catalogue's "
             "3.1 naming"},
    {"name": "Mistral-7B-Instruct-v0.3-fp16", "verdict": "family_only",
     "card": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
     "note": "no fp16 Mistral is catalogued"},
    {"name": "LLaMA-3-13B-Instruct-q4_k_m", "verdict": "no_card",
     "card": None,
     "note": "no 13B entry anywhere in the catalogue"},
    {"name": "LLaMA-3-70B-Instruct-fp16", "verdict": "family_only",
     "card": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
     "note": "70B is catalogued at Q4_K_M / IQ2_XS, not fp16"},
    {"name": "LLaMA-3-70B-Instruct-q4_k_m", "verdict": "carded",
     "card": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
     "note": "family + quant carried under the catalogue's "
             "3.1 naming"},
    {"name": "Mixtral-8x7B-Instruct-v0.1-fp16", "verdict": "no_card",
     "card": None,
     "note": "no Mixtral entry anywhere in the catalogue"},
)

# ── The card ────────────────────────────────────────────────────────────────

_CURVE_BY_QUANT = {row[0]: row for row in CURVE}


@dataclass(frozen=True)
class ModelCard:
    """One supported model's measured truth, as filed data."""

    model_id: str
    filename: str
    tier: str
    description: str
    declared_quant: str           # catalogue spelling, lowercased
    filename_quant: str | None    # TF-066's filename verdict
    quant_grade: str              # proven | generic
    size_gb: float                # catalogue-declared, never weighed
    params_b: float | None        # name-parsed; None = unprovable
    curve: dict | None            # TF-067 row for the proven level
    fits: dict[str, str]          # profile -> fits|exceeds|unprovable
    quant_default_profiles: tuple[str, ...]
    profiles_proven: tuple[str, ...] = ()
    live_proof: str = "not_proven"

    @property
    def fits_profiles(self) -> tuple[str, ...]:
        return tuple(p for p, v in self.fits.items() if v == "fits")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "filename": self.filename,
            "tier": self.tier, "description": self.description,
            "declared_quant": self.declared_quant,
            "filename_quant": self.filename_quant,
            "quant_grade": self.quant_grade,
            "size_gb": self.size_gb, "size_basis": "catalogue_declared",
            "params_b": self.params_b,
            "curve": self.curve,
            "fits": dict(self.fits),
            "fits_profiles": list(self.fits_profiles),
            "quant_default_profiles": list(self.quant_default_profiles),
            "profiles_proven": list(self.profiles_proven),
            "live_proof": self.live_proof,
        }


def _profiles() -> dict[str, dict[str, Any]]:
    return {pid: load_hardware_profile(pid)
            for pid in list_hardware_profiles()}


def _fit_verdict(size_gb: float, params_b: float | None,
                 profile: dict[str, Any]) -> str:
    ceiling = (float(profile["hardware"]["ram_gb"])
               + float(profile["hardware"]["vram_gb"]))
    if size_gb > ceiling:
        return "exceeds"
    if params_b is not None and params_b > float(
            profile["model"]["max_params_b"]):
        return "exceeds"
    if params_b is None:
        return "unprovable"
    return "fits"


def build_cards() -> list[ModelCard]:
    """One card per curated catalogue entry, built live."""
    profiles = _profiles()
    cards: list[ModelCard] = []
    for entry in ModelBrowser.CURATED_MODELS:
        declared = entry.quantization.lower()
        filename_quant = quant_of_filename(entry.filename)
        grade = "proven" if filename_quant == declared else "generic"
        params = (params_from_name(f"{entry.model_id} {entry.filename}")
                  or params_from_name(entry.description))
        curve = None
        if grade == "proven" and filename_quant in _CURVE_BY_QUANT:
            q, mean, loss, perfect, contra = _CURVE_BY_QUANT[filename_quant]
            curve = {"quant": q, "mean": mean, "loss": loss,
                     "perfect": perfect, "contradictions": contra,
                     "curve_class": LEVEL_CLASSES[q],
                     "basis": "tf067_fixture_corpus_per_quant_level"}
        fits = {pid: _fit_verdict(entry.size_gb, params, profile)
                for pid, profile in sorted(profiles.items())}
        defaults = tuple(
            pid for pid, profile in sorted(profiles.items())
            if grade == "proven"
            and profile["model"]["quantization"].lower() == filename_quant)
        cards.append(ModelCard(
            model_id=entry.model_id, filename=entry.filename,
            tier=entry.tier, description=entry.description,
            declared_quant=declared, filename_quant=filename_quant,
            quant_grade=grade, size_gb=entry.size_gb,
            params_b=params, curve=curve, fits=fits,
            quant_default_profiles=defaults))
    return cards


def card_for_filename(filename: str) -> ModelCard | None:
    for card in build_cards():
        if card.filename == filename:
            return card
    return None


# ── Scaffold behaviour (profile-shaped, model-blind) ────────────────────────

def scaffold_behaviour() -> dict[str, dict[str, int]]:
    """Each shipped profile's generation shape, resolved live
    from its JSON: the behaviour any model runs under there."""
    out: dict[str, dict[str, int]] = {}
    for pid, profile in sorted(_profiles().items()):
        model = profile["model"]
        gen = profile["generation"]
        out[pid] = {
            "draft_count": int(model["draft_count"]),
            "max_refinement_passes": int(gen["max_refinement_passes"]),
            "max_response_tokens": int(gen["max_response_tokens"]),
            "token_budget_per_turn": int(model["token_budget_per_turn"]),
            "max_context_tokens": int(model["max_context_tokens"]),
        }
    return out


# ── The root card, reconciled ───────────────────────────────────────────────

def _root_card_path() -> Path:
    return Path(__file__).resolve().parents[3] / "MODEL_CARD.md"


def model_card_rows() -> list[dict[str, str]]:
    """The root MODEL_CARD.md compatible-models rows, parsed
    live: one dict per row naming a shipped profile."""
    rows: list[dict[str, str]] = []
    profile_ids = set(list_hardware_profiles())
    for line in _root_card_path().read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "`" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        match = re.search(r"`([a-z_0-9]+)`", cells[0])
        if match and match.group(1) in profile_ids:
            rows.append({"profile": match.group(1),
                         "model": cells[1], "quant": cells[2]})
    return rows


def _families_in(text: str) -> set[str]:
    lowered = text.lower()
    return {family for family, spellings in FAMILY_SPELLINGS.items()
            if any(s in lowered for s in spellings)}


def _catalogue_families() -> dict[str, set[str]]:
    """family -> the declared quants the catalogue carries it at."""
    out: dict[str, set[str]] = {}
    for entry in ModelBrowser.CURATED_MODELS:
        for family in _families_in(f"{entry.model_id} {entry.filename}"):
            out.setdefault(family, set()).add(entry.quantization.lower())
    return out


def model_card_discrepancies() -> set[tuple[str, str]]:
    """The root card's drift, recomputed live under stated rules:
    a row's quant differing from its profile's shipped default;
    a named family the catalogue does not carry; a named
    family's quant the catalogue does not carry it at; a
    context-size variant token (``<n>k``) no catalogue filename
    of that family carries."""
    problems: set[tuple[str, str]] = set()
    profiles = _profiles()
    catalogue = _catalogue_families()
    catalogue_text = " ".join(
        f"{e.model_id} {e.filename}".lower()
        for e in ModelBrowser.CURATED_MODELS)
    for row in model_card_rows():
        pid = row["profile"]
        if row["quant"].lower() != profiles[pid]["model"][
                "quantization"].lower():
            problems.add((pid, "quant_differs_from_profile_default"))
        families = _families_in(row["model"])
        for family in families:
            if family not in catalogue:
                problems.add((pid, "model_not_in_catalogue"))
            elif row["quant"].lower() not in catalogue[family]:
                problems.add((pid, "quant_not_in_catalogue"))
        for token in re.findall(r"\d+k\b", row["model"].lower()):
            if token not in catalogue_text:
                problems.add((pid, "model_variant_not_in_catalogue"))
        if "subset" in row["model"].lower() \
                and "subset" not in catalogue_text:
            problems.add((pid, "model_not_in_catalogue"))
    return problems


# ── Export ──────────────────────────────────────────────────────────────────

def cards_as_data() -> dict[str, Any]:
    cards = build_cards()
    audit = []
    profiles = _profiles()
    for row in RECOMMENDED_AUDIT:
        named_by = sorted(
            pid for pid, profile in profiles.items()
            if row["name"] in profile["model"]["recommended_models"])
        audit.append({**row, "profiles": named_by})
    return {
        "format": CARDS_FORMAT,
        "rules": [dict(r) for r in CARD_RULES],
        "cards": [c.to_dict() for c in cards],
        "fit_summary": dict(FIT_SUMMARY),
        "scaffold_behaviour": scaffold_behaviour(),
        "recommended_audit": audit,
        "model_card_discrepancies": sorted(model_card_discrepancies()),
    }


# ── Checkers ────────────────────────────────────────────────────────────────

def model_cards_problems() -> list[str]:
    """The filed estate judged against the live tree."""
    problems: list[str] = []
    cards = build_cards()
    entries = ModelBrowser.CURATED_MODELS

    if len(cards) != len(entries):
        problems.append("card count != catalogue count")
    if [c.filename for c in cards] != [e.filename for e in entries]:
        problems.append("cards are not the catalogue in order")
    rule_ids = [r["id"] for r in CARD_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("CARD_RULES ids are not unique")

    # Quant grades: exactly TF-066's filed gaps are generic.
    generic = {c.filename for c in cards if c.quant_grade == "generic"}
    gap_files = {g["filename"] for g in FILENAME_QUANT_GAPS}
    if generic != gap_files:
        problems.append(f"generic-grade cards {sorted(generic)} != "
                        f"TF-066 gaps {sorted(gap_files)}")
    for card in cards:
        if card.quant_grade == "proven":
            if card.filename_quant != card.declared_quant:
                problems.append(f"{card.filename}: proven grade but "
                                f"filename {card.filename_quant} != "
                                f"declared {card.declared_quant}")
            row = _CURVE_BY_QUANT.get(card.filename_quant or "")
            if row is None or card.curve is None \
                    or card.curve["mean"] != row[1] \
                    or card.curve["curve_class"] != LEVEL_CLASSES[row[0]]:
                problems.append(f"{card.filename}: curve row drifted "
                                f"from TF-067/TF-068")
        elif card.filename_quant not in GENERIC_QUANTS \
                or card.curve is not None:
            problems.append(f"{card.filename}: generic grade without "
                            f"a generic filename quant, or carrying "
                            f"a curve row")
        if card.profiles_proven != () or card.live_proof != "not_proven":
            problems.append(f"{card.filename}: live proof claimed")

    # Fit summary recomputed.
    summary = {"fits": 0, "exceeds": 0, "unprovable": 0}
    for card in cards:
        for verdict in card.fits.values():
            summary[verdict] += 1
    if summary != FIT_SUMMARY:
        problems.append(f"fit summary {summary} != filed {FIT_SUMMARY}")
    if sum(summary.values()) != len(cards) * len(_profiles()):
        problems.append("fit matrix is not cards x profiles")

    # Scaffold behaviour judged against the live JSONs + the
    # builder's model-free signature.
    behaviour = scaffold_behaviour()
    if set(behaviour) != set(_profiles()):
        problems.append("scaffold behaviour profiles != shipped")
    from thoughtforge.cognition.scaffold import ScaffoldBuilder
    params = list(inspect.signature(ScaffoldBuilder.build).parameters)
    if params != ["self", "sketch", "bundle", "personality"]:
        problems.append(f"scaffold builder signature {params} — "
                        f"model-blindness no longer provable")

    # The recommended audit covers every distinct recommended
    # name exactly once; its card targets exist.
    recommended = {name for profile in _profiles().values()
                   for name in profile["model"]["recommended_models"]}
    audited = [r["name"] for r in RECOMMENDED_AUDIT]
    if set(audited) != recommended or len(audited) != len(set(audited)):
        problems.append("audit does not cover the recommended "
                        "names exactly once")
    known_files = {e.filename for e in entries}
    for row in RECOMMENDED_AUDIT:
        if row["verdict"] == "no_card" and row["card"] is not None:
            problems.append(f"{row['name']}: no_card with a target")
        if row["verdict"] != "no_card" and row["card"] not in known_files:
            problems.append(f"{row['name']}: audit target "
                            f"{row['card']} is not catalogued")

    # The root card's discrepancies, recomputed.
    if model_card_discrepancies() != set(MODEL_CARD_DISCREPANCIES):
        problems.append(
            f"root-card discrepancies "
            f"{sorted(model_card_discrepancies())} != filed "
            f"{sorted(MODEL_CARD_DISCREPANCIES)}")
    if len(model_card_rows()) != len(_profiles()):
        problems.append("root card rows != shipped profiles")

    # Live-proof honesty: the checker itself re-probes the
    # grounds (no GGUF in the sparse tree, no llama_cpp here).
    root = Path(__file__).resolve().parents[3]
    if list(root.rglob("*.gguf")):
        problems.append("a GGUF now exists in-tree — the "
                        "not_proven filing must be revisited")
    if importlib.util.find_spec("llama_cpp") is not None:
        problems.append("llama_cpp is now importable — the "
                        "not_proven filing must be revisited")
    return problems


def model_cards_live_problems() -> list[str]:
    """The cards re-derived behaviourally on named cases."""
    problems: list[str] = []
    if build_cards() != build_cards():
        problems.append("card building is not deterministic")

    tiny = card_for_filename("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    if tiny is None or tiny.fits["phone_low"] != "fits" \
            or tiny.fits["pi_zero"] != "exceeds" \
            or tiny.quant_default_profiles != ("desktop_cpu", "pi_5"):
        problems.append("TinyLlama card truth drifted (fits "
                        "phone_low, exceeds pi_zero, and its "
                        "Q4_K_M is the desktop_cpu + pi_5 default "
                        "— but not phone_low's, whose own pairing "
                        "claims Q2_K)")
    qwen72 = card_for_filename("Qwen2.5-72B-Instruct-Q4_K_M.gguf")
    if qwen72 is None or qwen72.fits["server_gpu"] != "exceeds":
        problems.append("Qwen-72B must exceed even server_gpu "
                        "(72B > its 70B cap)")
    phi = card_for_filename("Phi-3-mini-4k-instruct-q5.gguf")
    if phi is None or phi.quant_grade != "generic" \
            or phi.curve is not None \
            or phi.fits["phone_low"] != "exceeds" \
            or phi.fits["pi_5"] != "unprovable":
        problems.append("Phi-3 q5 card truth drifted (generic, "
                        "no curve, size-exceeds phone_low, "
                        "params-unprovable on pi_5)")
    behaviour = scaffold_behaviour()
    if behaviour["phone_low"] != {
            "draft_count": 2, "max_refinement_passes": 1,
            "max_response_tokens": 180, "token_budget_per_turn": 180,
            "max_context_tokens": 512}:
        problems.append("phone_low scaffold behaviour drifted")
    if behaviour["pi_zero"]["max_refinement_passes"] != 0:
        problems.append("pi_zero must run zero refinement passes")
    return problems


__all__ = [
    "CARD_RULES",
    "CARDS_FORMAT",
    "FAMILY_SPELLINGS",
    "FIT_SUMMARY",
    "MODEL_CARD_DISCREPANCIES",
    "RECOMMENDED_AUDIT",
    "ModelCard",
    "build_cards",
    "card_for_filename",
    "cards_as_data",
    "model_card_discrepancies",
    "model_card_rows",
    "model_cards_live_problems",
    "model_cards_problems",
    "scaffold_behaviour",
]
