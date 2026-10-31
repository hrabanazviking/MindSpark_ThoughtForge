# TF-034 — Core API Freeze v1

*Part II, slice TF-034. Code:
`src/thoughtforge/cognition/api_surface.py` (`API_VERSION`,
`CORE_*` frozen tables, `PACKAGE_EXPORTS`, `MODULE_SURFACES`,
`api_manifest()`, `verify_api_surface()`), the package
`src/thoughtforge/cognition/__init__.py` re-exporting exactly
the frozen list, and the `num_drafts` wiring in
`src/thoughtforge/cognition/core.py`. Tests:
`tests/test_tf034_api_freeze.py`.*

## 1. Why a freeze, and why now

TF-019–TF-033 built the cognition layer's surface slice by
slice — turn drivers, streaming, interruption/resume, the will
ledger, the persona expectation, the config surface, latency
budgets — but nothing named that surface as one interface,
stated its version, or said what a consumer may rely on. The
package `__init__` still exported only the six names it was
born with (core, router, scaffold, prompt builder, chat
history), while the real surface had grown to span eighteen
modules. Part IX's bindings (Rúnmál TF-156, Samtal TF-157,
Hermes TF-158, Ember TF-159) bind to this layer; the roadmap
says they bind HERE, at the freeze — so the freeze comes
first, while changing it is still cheap.

## 2. The frozen surface (v1.0.0)

**The core** — `ThoughtForgeCore(config, model_path,
memory_dir, db_path, backend)`:

| Kind | Names |
|---|---|
| Turn drivers | `think(user_text, retrieval_path, num_drafts, history, interrupt)` → FinalResponseRecord; `think_stream(same)` → Iterator[StreamEvent]; `resume_turn(interrupt)` → FinalResponseRecord |
| Turn inspection | `last_turn_state` (property, RuntimeTurnState \| None), `last_turn_latency(total_ms)` → LatencyVerdict \| None |
| Conversation state | `conversation_config`, `selected_strategy`, `latency_budget`, `persona_expectation`, `will_ledger` (all properties) |
| Model lifecycle | `load_model(model_path)`, `unload_model()` |

**The package** — `from thoughtforge.cognition import …`
guarantees 18 names: the original six (`ThoughtForgeCore`,
`InputRouter`, `ScaffoldBuilder`, `PromptBuilder`,
`ChatHistory`, `ChatMessage`), the freeze machinery
(`API_VERSION`, `api_manifest`, `verify_api_surface`), and
the TF-026–TF-033 additions (`StreamEvent`, `InterruptToken`,
`TurnInterrupted`, `ConversationConfig`, `GoalLedger`,
`StandingGoal`, `PersonaExpectation`, `LatencyBudget`,
`LatencyVerdict`). The package `__all__` is exactly this
list; the verifier enforces it in both directions.

**Per-module surfaces** — `MODULE_SURFACES` in the code names
the frozen public names of 18 cognition modules (the
registries, contracts, checkers and record types the earlier
Part II slices specified). A module may carry more names for
the layer's own internal use; only the frozen ones are
promised to consumers. The coherence module is frozen as the
campaign harness's fixture/scorer surface (its consumers are
Part VII's suites, not Part IX's bindings).

## 3. How the freeze is enforced

The frozen tables are written by hand from the measured live
surface — never derived from it, so a silent change to the
real surface cannot rewrite the contract to match itself.
`verify_api_surface(manifest=None)` compares a manifest (the
frozen one, or one a consumer carries) against the live
objects and returns every problem it finds: the core's
constructor and method parameter names read live via
`inspect`, properties still properties, package exports
importable, in `__all__`, and identical (`is`) to the object
in their home module, every per-module frozen name present.
Extra live names are not problems — additions are
minor-version business, judged by humans; an unlisted name in
`__all__`, however, IS a problem, because `__all__` is itself
part of the promise.

Tests prove the check has teeth: tampered manifests (think's
parameters truncated, constructor truncated, a bogus export,
an unlisted `__all__` entry exposed by removing a name, a
bogus module name, a method listed as a property) each
produce exactly the named problem.

## 4. Versioning rules

- Removing/renaming a frozen name or parameter, or changing
  a frozen return contract: **breaking** — major bump + a
  note in this doc's successor.
- Adding a name, or a parameter with a default: **minor**.
- Behaviour fixes keeping names, parameters, and contracts:
  **patch**.
- Underscore-prefixed names are NOT public and may change
  without a bump: `_run_turn`, `_generate_candidates`, and
  kin are the pipeline's private machinery.
- A consumer on an incompatible major version must receive a
  truthful refusal, never silent corruption — TF-162 owns
  enforcing that at the service shape; this slice gives it
  the version and the manifest to check against.

## 5. CORE_LOOP §8-F1 settled: `num_drafts` wired

CORE_LOOP found `think()`'s `num_drafts` documented as
"Override draft count (default from hardware profile)" and
read by no code — draft count was fixed by construction
(≤2 local modes, exactly 1 unified), and no hardware profile
was ever consulted. TF-019 proposed this slice either wire
or remove it.

**Ruling: wired.** Removal would break pre-freeze callers
that already pass it (the phase-6 adversarial tests pass 1
and 5), and a draft cap is a genuine small-hardware need.
Frozen v1 semantics, implemented in `core.py`:

- The cap selects the first `num_drafts` candidate modes on
  the local-engine path (plain and streamed alike — pinned by
  counting stub-engine calls: default 2 drafts, cap 1 →
  exactly 1, cap 5 → the by-construction maximum of 2, no
  crash).
- The unified backend drafts exactly once by construction;
  the cap cannot raise it, and at ≥1 does not lower it
  (pinned: one backend call with and without the cap).
- `None` = no cap beyond construction. A value below 1
  raises `ValueError` naming the parameter — zero drafts is
  a caller error, not a knowledge-only turn.
- The docstring's old "default from hardware profile" claim
  is gone; the frozen docstring states the cap semantics
  above. (The turboquant engine's own `generate_drafts`
  num_drafts is a separate, already-live parameter and is
  untouched.)

F1 is therefore CLOSED by wiring, with the evidence above.
CORE_LOOP's remaining findings keep their owners (F2 →
TF-041; F4 → TF-039; the trace half of F3 → TF-038).

## 6. What is deliberately NOT frozen

- Return *values'* inner fields beyond the record types
  named: FinalResponseRecord / RuntimeTurnState are the
  shared record home (`thoughtforge.knowledge.models`); their
  field-level evolution belongs to the parts that own those
  records, with TF-061/TF-084 consuming them for inspection
  and traces.
- Strategy *execution*: `selected_strategy` remains
  selection-only (TF-022/TF-032 rulings stand); the freeze
  promises the selection surface, not execution.
- The service shape: TF-153/TF-154 build the callable
  service on top of this library freeze and version their
  own contract against it.

## 7. Post-freeze minor additions — recorded at TF-039

TF-035–TF-038 grew the surface by **minor additions only**
(§4's rules: adding a name is minor). They were recorded in
their own slices as they landed, each deferring the freeze
record to the next reconciliation; TF-039 makes that record
here and as data (`POST_FREEZE_ADDITIONS` in
`src/thoughtforge/cognition/core_docs.py`, live-checked by
`additions_problems()`):

| Addition | Kind | Added by |
|---|---|---|
| `determinism_mode` | core property | TF-036 |
| `last_turn_trace` | core method | TF-038 |
| `thoughtforge.cognition.determinism` | module (`DeterminismMode`, `turn_fingerprint`, `run_seeded_script`) | TF-036 |
| `thoughtforge.cognition.concurrency` | module (`CONCURRENCY_SURFACES`, `isolation_problems`, `run_concurrent_turns`) | TF-035 |
| `thoughtforge.cognition.small_model_floor` | module (`FloorBudget`, `check_floor_turn`, `run_floor_conversation`) | TF-037 |
| `thoughtforge.cognition.traces` | module (`TRACE_FORMAT`, `export_turn_trace`, `trace_problems`, `write_turn_trace`) | TF-038 |
| `ConversationConfig.max_refinement_passes` | record field (the D-PROF-4 fix) | TF-037 |
| `prune_memory` | core method (the D-ORPHAN-1 wiring; recorded by TF-045 as it landed) | TF-045 |
| `apply_writeback` | core method (the conversation → memory leg; recorded by TF-047 as it landed) | TF-047 |
| `last_writeback_outcome` | core property | TF-047 |

The frozen tables and `API_VERSION` (1.0.0) are **not** changed
by the recording: folding these additions into the tables with
a 1.1.0 bump is a freeze reconciliation act of its own, and
TF-192 verifies at campaign end that none stands unrecorded.
Until that fold-in, a consumer relying on an addition above
should name it as post-freeze, exactly as this table does.
