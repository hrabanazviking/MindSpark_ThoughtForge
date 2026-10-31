# TF-045 — Lifecycle: Aging, Decay, and Retirement by Rule

*Part III, slice TF-045. Code: `src/thoughtforge/knowledge/lifecycle.py`
(the engine), `src/thoughtforge/knowledge/lifecycle_rules.py` (the
rules as verifiable data — `lifecycle_problems()` re-derives every
claim below from the live tree). Tests:
`tests/test_tf045_lifecycle.py` (32 tests).*

## 1. What the lifecycle is

`MemoryLifecycle` (over a file `MemoryStore`) is how memory ages:
a prune score is built per record from age and quality signals,
each prune mode's threshold tables decide archive/delete, hard
store caps are enforced by lowest survival score, profile records
deduplicate by category+tags, records can be reinforced, and the
active thread state cools per turn until it goes stale.

Until this slice it was **test-only** — TF-011's D-ORPHAN-1: no
fact ever aged at runtime. §5 records the wiring that fixed it.

## 2. The score's floor and ceiling (measured)

Every prune path builds components with the builder defaults —
redundancy is never measured at prune time (0.0) and future
relevance is the neutral 0.5 — so from the shipped weights
(`PRUNE_SCORE_WEIGHTS`, TF-044's census):

- **Floor 0.05**: even a fresh, perfect record scores the neutral
  future-relevance share (0.10 × 0.5).
- **Ceiling 0.75**: even an infinitely old, zero-confidence,
  zero-importance, zero-quality, never-reinforced record
  asymptotes to 0.25 + 0.15 + 0.15 + 0.10 + 0.05 + 0.05.

Both extremes are reproduced by `lifecycle_problems()` through
the real component builder, not just the arithmetic.

## 3. Reachability — what each mode can actually do by score

Derived from the ceiling against the live threshold tables
(`score_reachability()`), and proven with engineered records:

| Mode | Archive by score | Delete by score |
|---|---|---|
| light | **never** (lowest archive 0.85 > 0.75) | never |
| routine | yes, all kinds | **never** (lowest delete 0.82 > 0.75) |
| heavy | yes, all kinds | fact 0.72 / episode 0.70 / pattern 0.72 at ordinary ages; preference 0.75 sits exactly at the ceiling — fires only once decay underflows to zero in floating point (≈265 years at a preference's 90-day half-life) |
| emergency | yes, all kinds | yes, all kinds |

So under **light**, only deduplication and hard caps ever act;
under **routine**, the score path archives but cannot delete.
Deletion by score is a heavy/emergency act. This is the shipped
tables' arithmetic, stated — not retuned by this slice.

Boundaries are inclusive (`decision()`): a score exactly on the
delete threshold deletes; exactly on archive archives; just
below archive is kept. The boundary test engineers zero-signal
facts to score 0.73 / 0.71 against routine's 0.72 archive line
(via `age_for_prune_score`, the score's age inverse) and watches
one archive and one stay.

## 4. The other edges

- **Retirement is physical.** Archived/deleted records are
  absent from their file after the pass's rewrite (TF-041 K13) —
  the counts report what left the live store.
- **Protected outranks everything.** Score and dedup already
  skipped protected records; the cap path did not (§6, fixed).
  A protected surplus may leave a cap exceeded — protection
  outranks the cap, stated as the rule.
- **Caps** delete lowest-survival first, per store, only above
  the hard cap (preferences/facts 256, episodes 512, patterns
  256; recent queue 128 and open loops 8 are trimmed by their
  own paths — open loops trim to the *soft* cap 6 during any
  prune, thread state included in every mode).
- **Age edges.** An unparseable timestamp ages a record by
  exactly `AGE_PARSE_DEFAULT_DAYS` = 30 days (named by this
  slice; neither immortal nor instantly ancient); a future
  timestamp ages 0.
- **Reinforcement** adds the named deltas × signal strength
  (confidence +0.05, importance +0.03, quality +0.02), clamped
  at 1.0, bumps counters, sets status `reinforced`; a missing
  id returns False. Episodes take all three deltas; patterns
  take quality and bump both `times_successful` and
  `times_reinforced` (§6).
- **Thread cooling** is cumulative by design: cooling *n*
  subtracts 0.08·(turns_since_touch) from priority and the same
  count from `expires_after_turns` — measured on a 0.9/10
  thread: (0.82, 1, 9) after one cool, (0.66, 2, 7) after two.
  Stale (expiry ≤ 0 or priority < 0.20) clears the file.
- **Mode discipline.** An unknown prune mode is refused up
  front with a ValueError naming the valid modes (§6); the tag
  index is rebuilt after routine/heavy/emergency passes, not
  after light.

## 5. The wiring (D-ORPHAN-1 FIXED)

- **Per-turn aging**: `ThoughtForgeCore._finish_turn` calls
  `MemoryLifecycle(store).cool_thread()` once per completed
  turn — the ephemeral thread state ages at runtime and is
  cleared when it goes stale. A lifecycle failure is logged at
  warning and never kills a completed turn. Interrupted turns
  that never reach `_finish_turn` do not cool — the turn did
  not complete.
- **Explicit prune**: `ThoughtForgeCore.prune_memory(mode)`
  runs a full pass over the core's own store and returns its
  counts. Fact pruning is **never automatic per turn** —
  retirement of facts is an operator's act, not a side effect
  of chatting. Recorded as a post-freeze minor addition
  (TF-034 rules) in `POST_FREEZE_ADDITIONS` + the freeze doc
  §7; the frozen v1 tables are untouched.
- TF-041's K10 checker was flipped: it now pins the wiring's
  presence and names its loss as a regression.
- Named, not built here: a `forge_memory.py` prune command is
  TF-057's (the memory-store toolset slice); budget-driven
  eviction is TF-056's.
- **TF-056 note (landed since):** budget-driven eviction now
  stands — `MemoryLifecycle` accepts a `profile_id` and enforces
  that profile's per-kind caps (resolved live from its shipped
  JSON by `knowledge/memory_budgets.py`) in place of the flat
  table; the threshold tables above stay flat across profiles
  by design (budgets cap counts, modes rule retirement), and a
  lifecycle built with no profile enforces the flat table —
  proven to be desktop_cpu/desktop_gpu's own budget — exactly
  as this doc describes. See `MEMORY_BUDGETS.md`.

## 6. Defects found by this slice's probes — all fixed in scope

1. **Hard caps deleted PROTECTED records.** `_enforce_cap`
   ignored `retention_class`: probed with 257 preferences, the
   protected one (ancient, zero-scored, lowest survival) came
   back deleted by an ordinary light prune — contradicting
   TF-041's K15 ("pruning never touches a protected record, in
   any mode"). Fixed: cap candidates exclude protected records.
   Regression test pins the probe exactly.
2. **Reinforcing a response pattern crashed.**
   `MemoryLifecycle.reinforce` does `p.times_reinforced += 1`,
   but `ResponsePatternRecord` was the one record type with no
   `times_reinforced` field — AttributeError on any pattern
   reinforce. Fixed at the model: field added with default 0
   (TF-042's counter vocabulary already named it; store loads
   of older pattern files are unaffected — the field defaults).
3. **Unknown prune modes died mid-pass.** `prune("bogus")`
   raised a bare KeyError from the threshold tables after
   earlier stores had already been rewritten. Fixed: validated
   up front, ValueError naming the valid modes, before any
   store is touched.

## 7. Proof grade

FIXTURE (synthetic stores and records, engineered ages, this
VM) + live-tree proofs (the checkers read the shipped tables,
weights, and core source). A live store aging over real weeks
of use is **NOT PROVEN** — no real store exists on this machine
(TF-041); per-profile lifecycle tuning is **NOT PROVEN**
(the threshold tables are flat across profiles — TF-056's
budget slice owns any per-profile ruling).
