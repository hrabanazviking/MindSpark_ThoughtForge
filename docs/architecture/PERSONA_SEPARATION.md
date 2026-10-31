# TF-055 — Persona Memory Separation: Per-Persona Stores Never Bleed

*Part III, slice TF-055. Code:
`src/thoughtforge/knowledge/persona_separation.py` (the
surfaces as data, the two-persona fixtures, the separation
report, the adversarial prompts, and
`persona_separation_problems()` re-deriving every claim below
from the live tree), the D-PER-1 fix in
`src/thoughtforge/utils/self_heal.py` +
`src/thoughtforge/cognition/core.py`. Tests:
`tests/test_tf055_persona_separation.py`.*

## 1. The question and the measured starting state

The roadmap asks: do per-persona stores ever bleed — with
fixtures of two personae and adversarial prompts? TF-035 had
ruled the concurrency half (one core = one conversation) and
expressly left write-side store separation to this slice.
Measured before this slice, by direct probe on synthetic
stores:

- **Distinct directories are fully separate — by
  construction, unproven until now.** A `MemoryStore` derives
  all seven of its paths from its one directory, so two stores
  in two directories share nothing: loads, tag index and
  recent queue are each store's own (probe P1).
- **The same directory is the same persona (probe P2).** Two
  stores pointed at one directory see everything the other
  wrote. A directory is not a namespace inside a store — it
  IS the persona's home. "Sharing a store" and "being the
  same persona" are one statement in this tree.
- **D-PER-1 — one persona's core rewrote ANOTHER persona's
  store at construction (found here, FIXED).** The core's
  startup `SelfHealer().heal_all()` took no target: it healed
  the DEFAULT deployment's memory dir and knowledge DB
  whatever store the core was built on. Probe P3 planted a
  corrupt episodic line in a fake default home, then built a
  core on a different persona's store: the default home's
  file was rewritten, the bad line quarantined to a
  `corrupt_backup_*` file inside the default home. The same
  pass could reset that store's personality core or thread
  state and rename its knowledge DB — while the core's OWN
  store was never healed at all.
- **The config personality fallback is an assignment, not a
  store bleed (D-PER-2, CLOSED-BY-VERDICT).** A core whose
  store holds no personality core loads the YAML at the
  shared config's `cognition.personality_core_path` (TF-030
  measured it — default config → shipped `pers_001`; TF-038
  pins it in the trace). Probe P4: an empty store plus a
  config naming persona A's core file yields a core running
  as A; a store that DOES hold a core wins over the config,
  always (B's store + A's file in config → B).

## 2. The rule

**One directory = one persona.** Identity precedence for a
core is: the store's own personality core, then the config's
assignment, then none. Serving two personae takes two stores
AND two configs (or a config carrying no personality path) —
one shared directory or one shared config is one shared
assignment, by the same rule. Every surface is stated once
in `SEPARATION_SURFACES`:

- **Per-persona:** the seven store files (personality core,
  profile records, episodes, patterns, thread state, tag
  index, recent queue), the core-bound persona expectation
  and will ledger, and the writeback / lifecycle / self-heal
  targets — a core writes, prunes, forgets and heals ONLY in
  its own store.
- **Shared by rule, holding no personal memory:** the world
  knowledge DB (TF-046's split — a read-only reference estate
  in a turn), the config assignment, the perf aggregate and
  logging (TF-035 accounting, never state), the embedding
  model cache (models, not memory), and the caller-owned chat
  history.
- **Explicit transfer — the only crossing:** TF-052's bundle
  import. An operator's deliberate act, verified whole before
  anything lands, never silently overwriting in merge mode,
  every landed record stamped `source='import'`. Nothing
  else moves records between stores, and this slice's tests
  pin that import is exactly as loud as it claims.

## 3. The fix (D-PER-1)

`SelfHealer` now takes an explicit heal target —
`SelfHealer(memory_dir=…, knowledge_db_path=…)`; the defaults
preserve the standalone behaviour exactly (the doctor and the
phase-8 tests construct it bare). `ThoughtForgeCore` passes
its OWN store dir and its own knowledge DB path. Post-fix
probe P3: the planted default home is byte-identical after
another persona's core is constructed, and the owning core's
own corrupt line is healed in its own home, backup included.
Config healing stays global — config is a shared assignment
surface, not persona memory (stated in the surfaces table).

## 4. The proof

Fixtures: two invented Lantern Hollow personae (TF-017's
world) — Wrenna the Lantern-Keeper and Tammas the
Lock-Keeper, both synthetic — each with a unique secret
marker, trait, thread topic and preference value, a shared
`ferry` tag, and one deliberately colliding record id
(`fct_shared_tf055`) whose content differs per store: ids are
per-store namespaces, pinned harmless. The separation report
runs 19 checks: per-store loads of every record home, the
colliding id's content, tag-index and recent-queue contents,
and a raw scan of each directory for the other persona's
markers. A negative control plants a secret in the wrong
home and proves the scan catches it.

Adversarial prompts (five, carrying no secret themselves)
demand the other persona's secret, identity and thread topic
through real cores — knowledge-only turns and a capturing
stub backend. Everything a turn produces (response, scaffold
fact block, the backend's generation requests) is scanned:
the other persona's markers never appear, and the persona in
force steers with its own trait.

## 5. Boundaries, stated

- Separation is proven between STORES and CORES on this
  machine, fixture grade — two live personae served in
  production is TF-166's service-scale question, not claimed
  here.
- The knowledge DB is shared world reference by TF-046's
  ruling; a deployment that wants per-persona WORLD stores
  passes per-persona `db_path` values — the core already
  takes it as a parameter, and separation then follows the
  same directory rule.
- A process is trusted with the directories it is given:
  separation here is an architecture property (no code path
  crosses), not an OS sandbox between mutually hostile
  processes on one host.
