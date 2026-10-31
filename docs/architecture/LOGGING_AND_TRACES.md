# ThoughtForge Architecture — Logging & Traces (TF-038)

Logging & traces: a turn's full cognition trace exportable.
Built 2026-10-31 on branch `yrsa/thoughtforge-road-build`.

The exporter lives in `src/thoughtforge/cognition/traces.py`
(`export_turn_trace`, `trace_to_json`, `write_turn_trace`,
`trace_problems`); the core surface is
`ThoughtForgeCore.last_turn_trace()`; the tests are
`tests/test_tf038_traces.py` (25 tests). This slice closes the
export half of TF-019's **F3** (TF-025 retained the turn state;
nothing could get it *out* of the core as a record) and narrows
TF-016's **D-COV-3** to its `paths` share (§5).

## 1. What was true before

- The retained turn state (TF-025) was readable only as a live
  Python object inside the process that ran the turn. Nothing
  serialised it: no export, no file shape, no format a consumer
  (an inspector UI, Part IX's cognition API, a bug report) could
  carry away.
- What a turn *did* was otherwise visible only scattered: log
  lines from a dozen modules, the perf tracker's aggregates, and
  the final record alone. TF-031 had already ruled that a failure
  caught by the core must not exist *only in the log* — its
  records ride the turn state, but until this slice they still
  could not leave the process as one document.
- `dataclasses.asdict` on the state happens to be JSON-shaped for
  simple turns, but it is not a trace format: it drops the score
  composites and quality tier (properties, not fields), carries
  no format identity or section contract, and breaks the first
  time a store record's `raw` payload holds a non-JSON value.

## 2. The trace format

One named format — `TRACE_FORMAT = "thoughtforge-turn-trace/1"` —
with sections in pipeline order (`TRACE_SECTIONS`):

| Section | Carries |
|---|---|
| `turn` | turn id's state: phase, `is_complete` / `is_interrupted`, interrupt reason + phase, resume count, repair flag, started / completed timestamps |
| `conversation` | the core-level context a bare turn state does not carry — selected profile / strategy / depth / candidate modes, determinism seed, persona id — when exported through the core; `None` on a bare turn-state export |
| `route` | the input sketch (raw text, intent, topic, tone, response mode, memory triggers, urgency, retrieval path) |
| `retrieval` | the memory bundle: personality id, totals, confidence, and every activated record (id, type, score, cue, raw payload) |
| `scaffold` | goal / tone / focus / avoid / depth / candidate modes / fact_block, exactly as built |
| `candidates` | every candidate: id, mode, text, token estimate, full scores **with composite** |
| `fragments` | every fragment: id, source candidate, text, position, keep verdict, scores with composite |
| `generation_failures` | TF-031's failure records (stage / mode / detail) |
| `final` | the final record whole: text, mode, salvage path, source candidate / fragment ids, citations, enforcement verdict + notes, token count, scores with composite **and quality tier** |
| `writeback` | the writeback record when one exists (usually `None` today — TF-047 owns the write-back leg) |
| `timings` | retrieval / generation / salvage ms and the token total |

Rules of the format:

- **Detached.** The export is built field by field into fresh
  containers; mutating an exported trace never touches the live
  turn state (pinned by tampering with a trace and re-exporting).
- **JSON-safe by construction.** Every value passes a conversion
  whose only lossy rule is stated: a value with no JSON shape
  becomes its `str()` — visible in the trace, never silently
  dropped. `json.loads(trace_to_json(trace)) == trace` is pinned.
- **Honest at any phase.** A turn exports finished, failed, or
  interrupted. An unfinished turn exports with `final: None` and
  its phase stated — never an invented completion. The trace of
  an interrupted turn (measured: token fired mid-generation)
  records the cut reason and phase and an empty `completed_at`.
- **Files by one writer.** `write_turn_trace(trace, path)`
  creates parent directories and writes the one JSON document;
  consumers read it back with `json.load` and judge it with the
  checker below.

## 3. Trace ≠ fingerprint

TF-036's `turn_fingerprint` and this trace look alike and are
deliberately opposite on two rulings:

| | Fingerprint (TF-036) | Trace (TF-038) |
|---|---|---|
| Question | would this run reproduce? | what happened in this turn? |
| Timestamps / timings | **excluded** by ruling (wall clock is not content) | **included** (they are part of what happened) |
| Unfinished turn | raises `ValueError` | exports, `final: None` |
| Scores | final's only, as pinned there | every candidate's, fragment's, and the final's, composites + tier included |

Neither replaces the other; both are built from the same
retained state. Pinned side by side in the tests.

## 4. The checker

`trace_problems(trace)` judges an exported trace as a
self-contained record and returns problems as data (`[]` =
sound): wrong format, a missing section, an unknown phase, no
turn id, a turn marked complete with no final, a final whose
turn id disagrees with the trace's, duplicated candidate /
fragment ids, a fragment naming a source candidate absent from
the trace, the final naming a source candidate / fragment absent
from the trace, a negative timing. Each is tamper-proven in the
tests. Content quality — whether the answer was *good* — is the
scorers' business, not this checker's.

## 5. Logging's half, and D-COV-3 narrowed

The tree's logging surface is
`thoughtforge.utils.logging_setup.setup_logging`: called by both
entry points (`forge_memory.py` at import; `run_thoughtforge.py`
when its own debug setup has not run) and by nobody else. The
cognition modules log through the standard `logging` module
under their own names and never configure. Traces do **not** go
through logging: a trace is a returned value, not a log line —
TF-061's inspector data, TF-084's joined inference traces, and
Part IX's cognition API consume it as data.

Before this slice `logging_setup` had no direct test import
(D-COV-3's logging_setup share). The slice's test file pins it
directly: one-shot configuration (a second call changes neither
level nor handlers), the level honoured, an unknown level name
falling back to INFO, the console switch adding exactly one
plain `StreamHandler`, the file switch creating
`thoughtforge.log` under the configured dir with a
`RotatingFileHandler` of 10 MB × 5 backups. (The tests measure
handler *deltas*: pytest's own logging plugin keeps capture
handlers on the root logger, and counting absolute handlers was
the first draft's false failure.) D-COV-3 is narrowed in TF-016,
not closed: its `utils/paths.py` share stands for TF-069.

## 6. Measured

Through real turns on this VM (fixture generation sources):

- **knowledge-only turn** — problems `[]`; route / retrieval /
  scaffold / final sections whole; candidates and fragments
  empty; `salvage_ms` exactly 0 (did-not-run, TF-025's ruling);
  conversation section names the in-force persona `pers_001`
  (TF-030's shipped-config fallback) and no profile.
- **stub-backend turn** — 1 candidate, fragments produced with at
  least one kept, the final's source ids resolving inside the
  trace (checker-clean), the candidate composite in the trace
  equal to the hand-recomputed composite of its own scores.
- **stated-failure turn** (backend error channel) — the trace's
  `generation_failures` is exactly TF-031's record
  (`backend` / `practical` / the backend's error string) and the
  final's mode is `failed`.
- **interrupted turn** — §2's honest-at-any-phase case.
- **conversation selections** — a core selecting phone_low +
  determinism seed 7 exports both in its conversation section.

Proof grade: FIXTURE generation sources on this VM. No live
model was traced; a live model's turn differs only in who wrote
the candidates, not in the state's shape.

## 7. What this slice did not do

- **Inference-side trace content** (backend calls, tokeniser,
  quant detail) joins at TF-084, which consumes this format.
- **Why a retrieval fired**, beyond what the bundle itself
  carries, is TF-061's inspector data — also a consumer of this
  export, not a change to it. *(TF-061, landed: the inspector
  lives in `knowledge/inspector.py` + `docs/architecture/
  MEMORY_INSPECTOR.md`, and this trace's retrieval entries now
  also carry each record's TF-048 citation payload — the one
  change TF-061 made here, on TF-048's assignment.)*
- **Prompt text per candidate** is not in the trace: prompts are
  rebuilt by the prompt builders from the scaffold + bundle the
  trace *does* carry (TF-023's contract), and TF-019's scaffold
  note stands — a turn builds many prompts; the trace records
  the turn, not a transcript of every string sent.
- The core surface addition (`last_turn_trace`) is a **MINOR**
  addition under TF-034's versioning rules; the frozen v1 tables
  are untouched and TF-039/TF-192 record it at the next freeze
  reconciliation, beside TF-036's `determinism_mode`.
