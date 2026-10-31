# TF-035 — Concurrency: Two Conversations at Once

*Part II, slice TF-035. Code:
`src/thoughtforge/cognition/concurrency.py`
(`CONCURRENCY_SURFACES`, `surface_rule()`,
`isolation_problems()`, `run_concurrent_turns()`) and the
turn lock in `src/thoughtforge/cognition/core.py`. Tests:
`tests/test_tf035_concurrency.py` (24 tests).*

## 1. The question

TF-032 ruled the shape — one core = one conversation — but
nothing had ever run two conversations at the same time and
proven they stay separate. This slice ran them, with
barrier-synchronised fixture backends so both generations
were provably in flight at the same instant, and measured
every surface the two conversations could share. One real
bleed was found — not between two cores, but inside one.

## 2. The bleed: two threads, one core (found, fixed)

Before this slice, two threads calling `think()` on ONE
core raced on the retained turn state. The pipeline body is
per-call (locals, a fresh `RuntimeTurnState` per turn), but
`think()` read `self._last_turn_state` back after its turn
finished — and a second thread's turn could assign that
field in between. Probe, pre-fix (slow fixture backend,
two different questions, 3 trials): **both callers received
the same final text in 3 of 3 trials** — one of them the
other conversation's answer, whole and unmarked.

The fix, in `core.py`:

- `ThoughtForgeCore._turn_lock` — a per-instance lock the
  turn drivers hold across the whole turn: `think()` across
  the turn and its read-back, `think_stream()` from the
  first event until the stream is exhausted or closed
  (closing releases it, proven by test), and `resume_turn`
  too, since a resume is a turn on the same core. The
  pipeline body (`_run_turn`) never takes it itself, so
  there is no re-entry to deadlock.
- `think()` reads back **its own thread's** turn (a
  thread-local the pipeline sets), not the shared retained
  state — so a queued turn that starts the instant the lock
  frees cannot cross the read-back either.

Post-fix probe: each caller receives its own answer, 3 of
3, and the fixture backend's generate intervals on one core
never overlap. The honest shape of the guarantee: sharing
one core is now **safe** (turns queue; answers stay correct)
but never **parallel**, and the core still holds only one
conversation's will, persona, and last-turn state. Parallel
conversations use parallel cores.

## 3. Two cores in parallel: no bleed measured

Two cores, barrier-synchronised so their generations
overlap exactly, in the synthetic Lantern Hollow world:

- each caller received its own answer; the finals differ
  and each names its own request;
- turn ids are distinct; each core's retained turn state,
  scaffold, and final are its own objects;
- histories (caller-owned, one per conversation) stayed
  separate in contents and in message objects;
- conversation A's planning turn set its standing goal;
  A's next turn carried `Standing goal: …` in its scaffold
  and B's — asked the identical ferry question at the same
  time — carried no trace of it; B's ledger stayed empty;
- per-conversation config held under concurrency: A
  (depth `light`, modes `[practical]`) scaffolded exactly
  so, while B, asked a planning question with no
  selections, kept the builder's own derivation (`expert`,
  `[strict_spec, practical]`);
- persona held per conversation: A, built on the synthetic
  store, carried Wrenna's steering block (`Persona traits:
  … plain-spoken …`) in its scaffold; B, persona-free,
  carried none.

## 4. The surface table (the contract)

`CONCURRENCY_SURFACES` in the code states each surface, its
owner, and its rule as data. Summary:

| Surface | Owner | Rule in one line |
|---|---|---|
| conversation config | conversation | Frozen at construction on its own core (TF-032) |
| will ledger | conversation | One per core; steers only its own fact_block (TF-029) |
| persona expectation | conversation | Construction-time snapshot per core (TF-030) |
| last turn state | conversation | Per core, under its own turn_id (TF-025) |
| generation source | conversation | The core's own engine/backend instance |
| chat history | caller | Core never owns it; one per conversation |
| knowledge database | shared-read | Fresh SQLite connection per retrieval; the turn loop writes nothing (TF-019) |
| memory dir | shared-read | Turns read persona + thread state, write nothing; write-side persona separation is TF-055 |
| perf tracker | shared-aggregate | One thread-safe singleton — accounting, never per-conversation evidence |
| turns on one core | serialised | Turn lock: queued and correct, never parallel |

The pipeline helpers a core holds (router, scaffold
builder, prompt builder, salvage, enforcement) needed no
rule of their own: they are per-core instances whose
methods keep all state in locals — the census found no
per-turn field on any of them to bleed through.

Two shared-read notes, stated plainly:

- Two cores on **one knowledge DB path** ran concurrent
  turns cleanly, and two `KnowledgeForge` instances
  retrieved from one DB at the same time without error.
  Both bundles were empty — the store was an empty schema,
  and the core loop's retrieval gate still reads the
  default DB path rather than the configured one
  (CORE_LOOP §8-F2, owned by TF-041). This slice pins
  *sharing safety*, not retrieval content.
- Two cores may share a **memory_dir for reads** (same
  persona, same stored thread state). Nothing in a turn
  writes there today (TF-019's verdict stands: the
  write-back leg is invoked by nothing). The moment a
  write leg lands (TF-047), this rule must be re-read —
  and per-persona store separation is TF-055's slice
  either way.

## 5. The checker and the runner

`isolation_problems(core_a, core_b, history_a, history_b)`
inspects retained state and names every bleed it finds:
same core on both sides, a missing completed turn, shared
or colliding turn state / turn ids / scaffolds / finals, a
shared will-ledger or persona object, one conversation's
active goal steering the other's scaffold, a shared history
object or shared message objects. Tests prove it has
teeth: a tampered core whose retained state is replaced
with the other core's is caught by name, as is one history
object handed to both conversations.

`run_concurrent_turns(...)` is the fixture runner: one
turn per core on two threads, thread errors captured as
data (never swallowed, never raised into the caller), the
finished pair judged by the checker. TF-166 inherits both
at service scale: two bindings, one store, these rules.

## 6. Proof grade and limits

FIXTURE proof, on this VM: stub unified backends (plain
`generate`, re-chunked streams), knowledge-only cores, the
TF-017 synthetic world. What is **NOT PROVEN** here:

- two *live models* generating concurrently — engine-level
  thread safety of a real TurboQuant/llama run, and what
  concurrency does to the TF-033 latency budgets, are
  TF-081's measurements;
- streaming consumers that pull one stream from several
  threads at once (a generator is single-consumer by
  nature; the lock is held by the consuming thread);
- model lifecycle (`load_model` / `unload_model`) called
  from another thread *during* a turn — lifecycle remains
  a caller-synchronised act, as before this slice; the
  lazy load *inside* a turn is under the turn lock.
