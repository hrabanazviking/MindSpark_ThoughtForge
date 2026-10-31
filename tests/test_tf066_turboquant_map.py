"""TF-066 — TurboQuant mapped.

Pins: the engine quantises nothing (no quantisation function
by AST, llama kwargs exactly the five non-quant keys; the
profile's resolved quantization was read nowhere in the
engine at TF-066's banking — D-QUANT-1, measured then and
FIXED by TF-068, whose load-time policy check the duties
table now records); the duties table (does / does-not-do);
the quant vocabulary consumed live from
QUANT_SUFFIXES plus the browser-only iq2_xs (D-QUANT-2: the
suffix map's first consumer); filename inference across every
catalogue entry (dot forms resolve, the two Phi-3 generic
names under-specify and are named as gaps, never rounded up);
profile recommendations read live and all in-vocabulary; and
the loss claims — every one qualitative/inventory/size, none
measured in-tree.
"""

from __future__ import annotations

from thoughtforge.inference.backends import QUANT_SUFFIXES
from thoughtforge.inference.model_browser import ModelBrowser
from thoughtforge.inference.turboquant_map import (
    ENGINE_DUTIES,
    FILENAME_QUANT_GAPS,
    GENERIC_QUANTS,
    KNOWN_QUANTS,
    LOSS_CLAIMS,
    MAP_FORMAT,
    QUANT_TYPES,
    filename_quant_gap,
    profile_quants,
    quant_of_filename,
    turboquant_map_problems,
)


def test_map_format_and_live_checker():
    assert MAP_FORMAT == "thoughtforge-turboquant-map/1"
    assert turboquant_map_problems() == []


def test_engine_duties_split():
    does = {r["id"] for r in ENGINE_DUTIES if r["does"]}
    does_not = {r["id"] for r in ENGINE_DUTIES if not r["does"]}
    assert does == {
        "load_gguf", "profile_resolution", "backend_selection",
        "token_budget", "generate_stream_drafts", "token_estimation",
        "enforce_profile_quant",  # TF-068: the load-time policy check
    }
    assert does_not == {
        "quantise_weights", "inspect_loaded_quant",
    }


def test_quant_vocabulary_consumes_suffix_map():
    suffix_rows = [r for r in QUANT_TYPES if r["source"] == "backends.QUANT_SUFFIXES"]
    assert {r["id"] for r in suffix_rows} == set(QUANT_SUFFIXES)
    assert "iq2_xs" in KNOWN_QUANTS  # browser-only, absent from the suffix map
    assert "q3_k_m" in KNOWN_QUANTS  # the inventory claim's only 3-bit class


def test_filename_inference():
    assert quant_of_filename("Llama-3.2-3B-Instruct-Q4_K_M.gguf") == "q4_k_m"
    assert quant_of_filename("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") == "q4_k_m"  # dot form
    assert quant_of_filename("model-q2_k.gguf") == "q2_k"
    assert quant_of_filename("model-F16.gguf") == "fp16"
    assert quant_of_filename("Qwen2.5-72B-Instruct-IQ2_XS.gguf") == "iq2_xs"
    assert quant_of_filename("Phi-3-mini-4k-instruct-q4.gguf") in GENERIC_QUANTS
    assert quant_of_filename("notes.txt") is None
    assert quant_of_filename("model.gguf") is None


def test_catalogue_filename_gaps_are_exactly_the_phi3_pair():
    assert {(g["filename"], g["declared"]) for g in FILENAME_QUANT_GAPS} == {
        ("Phi-3-mini-4k-instruct-q4.gguf", "Q4_K_M"),
        ("Phi-3-mini-4k-instruct-q5.gguf", "Q5_K_M"),
    }
    assert all(g["gap"] == "filename_under_specifies_variant" for g in FILENAME_QUANT_GAPS)
    # Every other catalogue entry's filename proves its declared quant.
    for entry in ModelBrowser.CURATED_MODELS:
        if entry.filename not in {g["filename"] for g in FILENAME_QUANT_GAPS}:
            assert filename_quant_gap(entry) is None, entry.filename


def test_profile_quants_live():
    quants = profile_quants()
    assert quants["phone_low"] == "q2_k"
    assert quants["pi_zero"] == "q2_k"
    assert quants["server_gpu"] == "fp16"
    assert quants["desktop_gpu"] == "q8_0"
    assert set(quants.values()) <= KNOWN_QUANTS


def test_loss_claims_all_unmeasured():
    assert LOSS_CLAIMS
    assert all(not r["measured_in_tree"] for r in LOSS_CLAIMS)
    assert {r["kind"] for r in LOSS_CLAIMS} <= {
        "inventory", "qualitative_loss", "size",
    }
    # No quantitative loss figure is filed as a claim anywhere.
    assert all("%" not in r["claim"] for r in LOSS_CLAIMS)
