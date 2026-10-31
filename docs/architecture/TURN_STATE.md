# ThoughtForge Architecture — Turn State (TF-025)

Turn state: a typed turn-state object replaces ad-hoc dicts and
argument bundles where the census found them. Built 2026-10-31 on
branch `yrsa/thoughtforge-road-build`.

The object is `RuntimeTurnState` (`src/thoughtforge/knowledge/
models.py` §14); its phase law lives in `src/thoughtforge/cognition/
turn_state.py` (phases as data, `advance_phase`); the tests are
`tests/test_tf025_turn_state.py` (23 tests). This slice also closes
TF-011's **D-DUP-1** (the copied private helpers) and the turn-state
half of TF-019's **F3** (the state was built, mistimed, and thrown
away). Trace *export* remains TF-038's slice, as CORE_LOOP assigned.

## 1. What was true before

- `think()` built a `RuntimeTurnState`, filled most of it — and
  **discarded it**. Callers received only the `FinalResponseRecord`;
  no log sink, property, or return path exposed the state (F3).
- **`salvage_ms` was never assigned.** The field existed on the
  dataclass and stayed 0 through every turn: compose/salvage time
  was unmeasured, and 0 was indistinguishable from a fast salvage.
- The state's parts travelled **twice**: once on the object, and
  again as loose positional argument bundles — `_compose_final`
  took (sketch, scaffold, bundle, candidates, fragments, turn_id),
  `_repair` took (turn_id, sketch, scaffold, bundle, previous). Two
  copies of the same plumbing, free to disagree.
- The state had **no phase**: nothing on the object said how far a
  turn had travelled, so an interrupted or partially-built turn
  (TF-027's territory) had no coherent story to resume from.
- TF-011's census (§3) found the tree's only real duplication:
  three private helpers copied between modules — `_now_iso` ×3
  (one byte-identical pair), `_split_sentences` ×2, `_build_messages`
  ×2 — private, zero-risk that day, drift-prone every day after.
  The docket (TF-016) assigned the one-shared-home fix to
  "TF-025-era hygiene"; this slice is that era.

## 2. The typed object, completed

`RuntimeTurnState` gains two fields and one property; everything
else about its shape is unchanged:

| Field | Meaning |
|---|---|
| `phase` | one of the seven phases (§3); starts `"created"` |
| `repair_attempted` | True once the quality gate fires a repair pass (was previously visible only in the final's `salvage_path`) |
| `is_complete` (property) | `phase == "completed"` **and** `completed_at` set — both, so a hand-advanced state cannot fake completion |

`_compose_final` and `_repair` now take the turn state itself
(`_compose_final(turn)`, `_repair(turn, previous)`) and read
sketch / scaffold / bundle / candidates / fragments / turn_id from
it. They are private; no caller outside `core.py` existed (grep-
verified across src + tests + benchmarks before the change).

## 3. The phase law

Phases are data — `PHASE_ORDER` in `turn_state.py`:

```
created → routed → retrieved → scaffolded → generated → composed → completed
```

`advance_phase(turn, phase)` is the only way the core moves a
turn's phase, and it enforces the law:

- forward moves apply (skipping is allowed — a turn's phases are
  milestones, not a checklist it must narrate one by one);
- re-asserting the current phase is a no-op returning False;
- a backward move or an unknown phase raises `ValueError`, and a
  rejected move leaves the phase untouched.

Repair is deliberately **not** a phase: it is a branch inside
composed → completed, recorded on `repair_attempted`. Giving it a
phase would make the order lie (most turns never repair, and a
repaired turn still composes first).

`think()` advances the phase after each step it names, so the
retained state of any turn — including one a future TF-027
interruption cuts mid-flight — always states exactly how far it got.

## 4. Timings, all three real

| Field | Measured around | Knowledge-only turn |
|---|---|---|
| `retrieval_ms` | `_retrieve` (unchanged from before) | measured (usually an existence check) |
| `generation_ms` | candidate generation + fragment scoring (unchanged) | measured (empty candidate list) |
| `salvage_ms` | `_compose_final` — salvage + enforcement (**new**) | **exactly 0** |

The 0 is a ruling, not an accident: on the knowledge-only path no
salvage runs, so `salvage_ms = 0` means *did not run*, never
*unmeasured*. Before TF-025 the same 0 meant *unmeasured* on every
path; that ambiguity is what F3 named and what is now closed.
`think()` also records a `think.salvage` perf sample per turn,
beside the existing total / retrieval / generation samples.

## 5. Retention

The core keeps the finished object on
`ThoughtForgeCore.last_turn_state` (None before the first
`think()`; replaced — never mutated — by the next turn). It is the
same object `think()` built: its `final_response` **is** the record
`think()` returned (identity-pinned in the tests). Callers treat it
as read-only. What TF-038 builds on this is export; what TF-027
builds on it is interruption/resume against a phased, retained
state instead of a discarded one.

## 6. D-DUP-1 closed: one shared home

`src/thoughtforge/utils/common.py` now holds the only
implementations:

- `now_iso()` — was `_now_iso` in models.py, chat_history.py,
  core.py;
- `split_sentences(text)` — was `_split_sentences` in core.py,
  salvage.py;
- `build_messages(request)` — was `_build_messages` in
  ollama_backend.py, lmstudio_backend.py (duck-typed on the shared
  `GenerationRequest` shape: pre-built `messages` pass through
  unchanged, else system_prompt + prompt).

The old private names survive as **aliases** (`_now_iso = now_iso`
and kin) in every module that had one, so existing importers —
including the tests importing `_split_sentences` from core and from
salvage — did not change, while exactly one body per helper remains
to drift. The tests pin identity (`is`), not merely equal
behaviour, so a fourth copy reappearing anywhere fails the suite.

## 7. What this slice did not do

- **Trace export** (serialising the state out of the core) is
  TF-038, per CORE_LOOP §8-F3's split ownership.
- **Interruption/resume** against the phased state is TF-027.
- `WritebackRecord`'s dict-typed lists (`new_facts:
  list[dict[str, Any]]` and kin) stay dict-typed: they are the
  not-yet-wired write-back payload (TF-047 owns conversation →
  memory and will type them against the ingestion contract,
  TF-042). Typing them here would invent a shape Part III has not
  ruled on.
