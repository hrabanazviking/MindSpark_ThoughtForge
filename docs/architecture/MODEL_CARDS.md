# TF-078 — Model Cards In-Repo

*Per supported model, the measured truth: size, quant,
profiles proven, scaffold behaviour. The estate lives in
`src/thoughtforge/inference/model_cards.py` (`CARD_RULES` /
`RECOMMENDED_AUDIT` / `MODEL_CARD_DISCREPANCIES` as data,
`build_cards()` / `scaffold_behaviour()` /
`cards_as_data()`, format `thoughtforge-model-cards/1`).*

## Measured start

No per-model card existed. Three surfaces named models and
none agreed as one estate: the curated catalogue
(`ModelBrowser.CURATED_MODELS` — 15 entries with model_id /
filename / size_gb / tier / description / quantization),
the hardware profiles' `recommended_models` (15 distinct
names across six JSONs, no sizes anywhere), and the root
`MODEL_CARD.md` compatible-models table (6 rows — a
framework card, not per-model cards). Nothing joined them.

The catalogue is the only supported set with truth to
file: it alone carries a size and a quant per model. Its
sizes are **declared** — no GGUF file exists anywhere in
this tree and `llama_cpp` is not installed, so no
catalogue size has ever been weighed and no catalogue
model has ever been loaded here (TF-066/TF-067's
standing).

## The cards

One card per catalogue entry, built live, each carrying:

- **size** — the catalogue's declared `size_gb`, basis
  stated (`catalogue_declared`); never presented as
  weighed.
- **quant** — TF-066's filename verdict: `proven` when
  the filename proves the declared level (13 of 15),
  `generic` when it proves only q4/q5 — the Phi-3 pair,
  TF-066's filed `FILENAME_QUANT_GAPS` — never rounded
  up to the declared K_M variant.
- **the quant's curve** — TF-067's fixture curve row +
  TF-068's class *for the proven level* (a property of
  the level over an 8-prompt fixture corpus, not a
  measurement of this model). Generic-grade cards carry
  no curve row.
- **fit per profile** — `fits` / `exceeds` / `unprovable`
  from two declared figures only: catalogue size vs the
  profile's declared RAM+VRAM ceiling, and the
  name-parsed parameter count (TF-069's parser) vs the
  profile's `max_params_b`. The Phi-3 pair carries no
  parseable count, so every size-passing cell for them
  is `unprovable`, never a fit.
- **profiles proven** — empty, for every card (below).

Fit summary over the 90 card × profile cells:
**28 fits / 53 exceeds / 9 unprovable**. The sharp rows:
both Qwen-72B entries exceed even `server_gpu` (72B >
its 70B cap — the Q4_K_M by cap, the IQ2_XS by cap
despite its 19 GB declared size); Qwen-14B exceeds
`desktop_gpu`'s 13B cap despite its gpu tier and its
"fits 12 GB VRAM" description (its size fits; the cap
excludes it); TinyLlama fits everywhere except
`pi_zero` (0.67 GB declared > the 0.5 GB ceiling).

## Profiles proven: none — stated, not softened

No model is proven on any profile live. TF-069 boots
each profile's stack in *simulation* with no specific
model loaded; TF-037 proves the core at the phone_low
floor with a *fixture* backend. Fit is therefore filed
under its own name and never as proof: every card's
`profiles_proven` is `()` and its `live_proof` is
`not_proven`. TF-082 (real Pi), TF-126 (small-model
honesty panel) and TF-173 (real-hardware lab) own the
roads that could change a card.

## Scaffold behaviour is model-blind

The scaffold builder takes (sketch, bundle, personality)
— no model enters it; the registry varies by intent
alone (TF-021). What varies per deployment is the
profile's generation shape, filed as each card's
scaffold context for its fitting profiles:

| profile | drafts | refine passes | max response tokens | token budget / turn | context |
|---|---|---|---|---|---|
| phone_low | 2 | 1 | 180 | 180 | 512 |
| pi_zero | 1 | 0 | 120 | 120 | 256 |
| pi_5 | 3 | 2 | 250 | 200 | 2048 |
| desktop_cpu | 3 | 2 | 400 | 250 | 4096 |
| desktop_gpu | 3 | 3 | 600 | 500 | 8192 |
| server_gpu | 5 | 3 | 2000 | 2000 | 32768 |

The checker pins the builder's model-free signature, so
the claim is re-proven every run, not asserted.

## Recommended is not carded (D-CARD-2)

The profiles' 15 distinct recommended names, audited
against the catalogue through the filed alias table:
**6 carded** (family + quant carried — with version
drift noted where it stands: Mistral v0.3 recommended
vs v0.2 catalogued; LLaMA-3 names carried under the
catalogue's 3.1 entries), **5 family_only** (TinyLlama
unnamed-quant and q2_k-subset, Phi-3 128k/q8_0, Mistral
fp16, LLaMA-3-70B fp16), **4 no_card** — Granite-1B,
Gemma-2B, LLaMA-3-13B and Mixtral-8x7B are recommended
with no size truth anywhere in-tree. 8 catalogue entries
are recommended by no profile.

## The root card, reconciled (D-CARD-1)

`MODEL_CARD.md`'s table is parsed and judged live; 3 of
its 6 rows drift from the profiles/catalogue they name:

- `server_gpu` — pairs Q4_K_M against the profile's
  TF-068-ratified fp16 default, and names Mixtral,
  which has no card.
- `phone_low` — the Q2_K TinyLlama pairing exists in
  the catalogue only at Q4_K_M.
- `pi_zero` — the "TinyLlama-1.1B subset" names no
  catalogue entry, at a quant the catalogue does not
  carry for the family.
- `pi_5` — the "Phi-3-mini-128k" variant names no
  catalogue entry (the catalogue carries the 4k
  variant; the profiles recommend 128k only for
  desktop_cpu).

The root card itself is **not edited here** — TF-136's
claims audit and TF-183's doc care own its correction,
with these cards as their evidence.

## D-ORPHAN-3 — the model_browser share, narrowed

TF-011 filed `inference/model_browser.py` as test-only;
its catalogue share was assigned here. The measured
verdict: the catalogue **data** is no longer orphaned —
`CURATED_MODELS` is read live by production modules
TF-066, TF-067, TF-069 and now these cards. The class's
behaviour surface (`list_curated` / `download` /
`detect_existing_ggufs`) still has no production caller
— no tool or engine path instantiates `ModelBrowser` —
and wiring a downloader is an install-surface act,
reassigned to TF-179. The docket entry is narrowed,
not closed (its other three shares stand).

## Proof grade

FIXTURE/STATIC — the real catalogue, profile JSONs,
TF-066/067/068 surfaces and root card read live on this
VM. No model loaded, no size weighed, no per-model
quality measured: a live model's behaviour is
**NOT PROVEN** for every card, never claimed.
