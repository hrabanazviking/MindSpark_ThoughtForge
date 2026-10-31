# TF-040 — Part II Gate: the Cognition Core Stands Proven at Its Floor (Part II Gate)

Closed 2026-10-31 on branch `yrsa/thoughtforge-road-build`.
Roadmap wording: "Part II gate: the drift fixtures (TF-028) +
floor test (TF-037) + API freeze all green; plain-language
summary filed."

## The gate verdict

**Part II is COMPLETE. All 21 build slices (TF-019–TF-039)
are banked with evidence in `docs/architecture/`, and the
three gate proofs were re-measured fresh at this gate — not
quoted from their slices:**

1. **Drift fixtures (TF-028) — GREEN.** The coherent 20-turn
   run scores PASS with mean / early / late 1.000 / 1.000 /
   1.000, drift 0.000, 0 contradiction turns (54 engine calls,
   history exactly 40 messages, 0 dropped, 20 unique turn
   ids). The deliberately drifting run is caught: REVIEW,
   mean 0.700, late 0.000, drift 1.000, 6 contradiction turns.
   The scorer both passes a conversation that holds its facts
   and fails one that loses them.
2. **Floor test (TF-037) — GREEN.** The same 20-turn script
   through a `phone_low`-selected core holds the floor on all
   20 turns: within floor on every dimension (max response
   30 tokens against the 180 cap, max prompt 338 word-tokens
   against 512, max 2 candidates, max 1 refinement pass with
   exactly 7 refining turns, 47 engine calls), and its drift
   report is PASS 1.000 / drift 0.000 / 0 contradictions.
3. **API freeze (TF-034) — GREEN.** `verify_api_surface()`
   returns `[]` against the live tree at this gate: the frozen
   v1.0.0 surface (core constructor, drivers, properties,
   18 package exports, 18 module surfaces) matches what ships.
   TF-039's reconciliation stands: the minor additions since
   the freeze (TF-035–TF-038) are recorded as data in
   `core_docs.py`, the frozen tables are untouched at 1.0.0,
   and a fold-in to 1.1.0 is a named future act (TF-192
   verifies none is unrecorded).

The targeted gate tests were also re-run as one set at this
gate: `test_tf028` + `test_tf037` + `test_tf034` =
**70 tests, 70 passed** — exactly the three slices' own counts
(22 + 23 + 25).

## Plain-language summary (the Volmarr-facing half of this gate)

Here is what Part II did, in plain words:

- **The thinking engine is now mapped, tested, and frozen.**
  At the start of Part II the cognition core was real code
  nobody had charted: a turn's path through routing, memory,
  scaffolds, generation, repair, and enforcement existed but
  was documented nowhere and guarded by few tests aimed at it
  directly. It is now all of those things — every stage has a
  written contract, tests that prove the contract, and a
  frozen public interface (version 1.0.0) that the later
  bindings (Rúnmál, Samtal, Hermes, Ember) will build on.
- **A conversation can now be interrupted and resumed,
  streamed, run two at once, and replayed exactly.** A cut
  turn keeps its state and can finish later under the same
  turn id. Two conversations on one machine do not bleed into
  each other (a real race that mixed up two callers' answers
  was found and fixed). With a seed set, a conversation
  replays turn-for-turn identically.
- **The engine remembers what it is trying to do, and who it
  is being.** A standing goal set in one turn steers later
  turns until it is finished or dropped (the "will" layer),
  and a persona's traits, speech style, and behaviour rules
  now actually steer every prompt instead of sitting dormant
  in a settings file (the "presence" half).
- **When the model fails, the answer says so.** A half-failed
  turn delivers its real answer with the failure named; a
  fully failed turn delivers a plainly stated failure — or
  the retrieved knowledge, labelled as knowledge only, never
  dressed up as a model-written answer. Nothing is silently
  fabricated.
- **Small hardware is no longer just a promise.** A
  conversation can select a hardware profile, and the profile
  now governs real things: how long an answer may be, how
  many repair passes may run, and the latency budget a turn
  is judged against. At the smallest profile (`phone_low`),
  the whole core was run end-to-end and held every budget
  while keeping its facts straight across 20 turns. One real
  defect fell out of that proof: the smallest profiles
  promised fewer repair passes than the engine ran anyway —
  found by measurement, fixed, and re-measured clean.
- **The honest limits, named plainly:** everything in Part
  II is proven with fixture models on this machine — small
  deterministic stand-ins, not a live language model, and
  not a real phone. A live small model keeping its facts
  (TF-126), the suite panels per profile (TF-128), and real
  phone hardware (TF-173) remain NOT PROVEN, exactly as the
  roadmap roads them. The one known hole inside the core's
  own map — retrieval still checks the default database path
  rather than the configured one (CORE_LOOP §8-F2) — is
  recorded open, not hidden, and its owner is the very next
  slice, TF-041, which opens Part III.

## The evidence, slice by slice

| Slice | Artefact | Headline |
|---|---|---|
| TF-019 | `CORE_LOOP.md` | Turn lifecycle mapped from code; write-back leg found invoked by nothing (→ TF-047); findings F1–F8 owned |
| TF-020 | `ROUTER_DECISION_TABLE.md` | Router decision table explicit; 5 never-matching patterns fixed (dead `\bAPI\b` family); +92 tests |
| TF-021 | `SCAFFOLD_REGISTRY.md` + `scaffold_registry.py` | Live tables assembled into one registry; brainstorming's duplicated mode fixed (F8 closed) |
| TF-022 | `SCAFFOLD_STRATEGIES.md` + `scaffold_strategies.py` | Chain-of-thought / plan-then-answer / self-check on one interface; F6 ruled not-a-defect (0.25 vs 0.54 are different instruments) |
| TF-023 | `PROMPT_BUILDER_CONTRACT.md` + `prompt_contract.py` | Assembly order golden-tested for all 6 prompt kinds; defect fixed — persona avoids and tone overlay had never reached the model; F7 closed by removal |
| TF-024 | `CHAT_HISTORY_DISCIPLINE.md` + `history_discipline.py` | History budgets enforced on storage, not just render; empty-render defect fixed; summariser hook registered, none shipped (→ TF-047) |
| TF-025 | `TURN_STATE.md` + `turn_state.py` | Turn state phased, retained, and measured (`salvage_ms` real); D-DUP-1 FIXED (one shared utils home) |
| TF-026 | `STREAMING_TURNS.md` + `streaming.py` | One pipeline, two drivers; streamed tokens are never final until salvage/enforcement ran |
| TF-027 | `INTERRUPTION_AND_RESUME.md` + `interruption.py` | Cooperative cut points; resume finishes the same turn id, generating only what is missing |
| TF-028 | `MULTI_TURN_COHERENCE.md` + `coherence.py` | 20-turn drift fixtures + scorer — gate proof 1, re-measured at this gate |
| TF-029 | `WILL_LAYER.md` + `will.py` | Standing-goal ledger: set by declaration, steers later turns' scaffolds, closes on completion/abandonment words |
| TF-030 | `PERSONA_CONSISTENCY_HOOKS.md` + `persona.py` | Persona steering block from the first turn; avoid-checkable responses; cognition half of the persona benchmark (other half TF-122) |
| TF-031 | `ERROR_HONESTY.md` + `error_honesty.py` | Every core swallow point records a failure; partial = answer + disclosure, total = stated failure or labelled knowledge; D-HON-1 docked (→ TF-084) |
| TF-032 | `COGNITION_CONFIG_SURFACE.md` + `config_surface.py` | Per-conversation profile/depth/strategy/modes, live keys only; profile token cap replaces the hard-coded 512; D-CFG-1/5, D-PROF-3 narrowed |
| TF-033 | `LATENCY_BUDGETS.md` + `latency_budgets.py` | Per-profile turn latency budgets, retrieval ceiling read live from each profile; enforced on the VM profile |
| TF-034 | `CORE_API_FREEZE_V1.md` + `api_surface.py` | API frozen at v1.0.0 with a tamper-proven verifier; `num_drafts` wired as a real cap (F1 closed) — gate proof 3 |
| TF-035 | `CONCURRENCY.md` + `concurrency.py` | Two-conversation isolation proven; same-core race that handed callers each other's answers found and fixed (per-core turn lock) |
| TF-036 | `DETERMINISM_MODE.md` + `determinism.py` | Seeded conversations replay exactly (ids + generation); will-ledger leak into replays found and fixed |
| TF-037 | `SMALL_MODEL_FLOOR.md` + `small_model_floor.py` | Floor proven at `phone_low`; profile refinement cap found dead, fixed — gate proof 2, re-measured at this gate |
| TF-038 | `LOGGING_AND_TRACES.md` + `traces.py` | A turn's full cognition trace exportable as detached JSON in one named format; export half of F3 closed; D-COV-3 narrowed |
| TF-039 | `CORE_DOCS_RECONCILED.md` + `core_docs.py` | Docs reconciled with shipped behaviour as verifiable data; F1–F8 recorded 5 closed / 2 ruled / F2 open (→ TF-041); 5 runnable examples pinned verbatim |

## What Part II leaves open — carried forward, each with its owner

Nothing found in Part II is left unfixed *in scope* or
unowned. The gate records the honest remainders exactly:

- **CORE_LOOP §8-F2 (open):** `_retrieve`'s existence gate
  still checks the default DB path, not the configured
  `KnowledgeForge.db_path`. Owner: **TF-041**, the first
  slice of Part III — recorded open in TF-039's data and
  checked live by its checker, so a fix landing without
  flipping the record is itself named.
- **Docket (TF-016) at the gate:** 47 entries — the Part II
  slices fixed D-DUP-1 outright (TF-025) and narrowed
  D-CFG-1, D-CFG-2, D-CFG-5, D-COV-3, D-PROF-3, and D-PROF-4
  with evidence; each remainder already had its later owner
  before Part II began (TF-044 scoring weights, TF-069 the
  `paths` share of D-COV-3, TF-094/TF-065 the validators
  verdicts, TF-141 the persona tuning weights, TF-153 the
  system-wide merged config view, TF-098 refinement cost
  across all profiles). D-HON-1, found by TF-031 in Part II,
  is owned by TF-084. TF-192 verifies at the end that no
  entry is left open without a reason.
- **Proof grades stay as labelled:** fixture generation
  sources on this VM throughout. Live-model coherence
  (TF-125/TF-126), per-profile panels (TF-128), the Pi and
  real-device legs (TF-082/TF-173/TF-174), and concurrent
  load truth (TF-081) are NOT PROVEN and were never claimed
  by any Part II slice.

## Road re-cut assessment

Part II changed no Part III–X slice's wording or order. Two
facts shape what comes next without re-cutting it: Part III
inherits exactly one open core finding (F2, owned by its
first slice, TF-041) and the unbuilt write-back leg TF-019
found (owned by TF-047, already roaded). The frozen API
(TF-034) plus TF-039's recorded post-freeze additions give
Parts III–IX the stable surface the roadmap assumed.

## Suite at the gate (measured this slice, TF-014 method)

`PYTHONPATH=src python3 -m pytest tests/ -q` on this VM:
**1210 collected / 1208 passed / 1 skipped / 1 failed
(D-ENV-1, unchanged — `test_yaml_present`, missing
sentence-transformers on this VM, an environment entry not a
code defect)** — exactly TF-039's banking line. The gate's
targeted re-run (TF-028 + TF-037 + TF-034) is 70 / 70 passed,
and the three gate proofs above were re-measured by direct
runs of `verify_api_surface()`, TF-028's `run_conversation`
(coherent and drifting), and TF-037's `run_floor_conversation`
at this gate, matching their slices' banked numbers exactly.
This slice adds one gate doc and no code or tests, so the
count does not move at banking.

## What opens next

Part III — Memory & Knowledge: the Sovereign Mind
(TF-041–TF-064), starting with TF-041: the knowledge store
mapped, its invariants listed and tested — and F2, the
retrieval gate's default-path defect, fixed in its scope.
