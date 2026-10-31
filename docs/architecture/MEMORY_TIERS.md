# TF-046 — Memory Tiers: Working / Episodic / Semantic

*Part III, slice TF-046. Code:
`src/thoughtforge/knowledge/memory_tiers.py` (the tiers as
verifiable data — `tier_problems()` re-derives every claim below
from the live tree). Tests: `tests/test_tf046_memory_tiers.py`
(22 tests). Builds on TF-041's store map, TF-042's ingestion
contract, and TF-045's lifecycle.*

## 1. The question and the method

The roadmap asks for working / episodic / semantic separation
"where the code hints it; unified where it doesn't
(verdict-driven)". This slice changes no storage and no
behaviour: it reads the hints the shipped code actually gives —
file layout, record docstrings, retention classes, stale
horizons, the half-life ladder, and who consumes what — and
states the tier verdict for each, as data a checker re-proves.

## 2. The verdicts

| Tier | Verdict | Records | Storage |
|---|---|---|---|
| working | **separate** | `active_thread_state` | `active_thread_state.json`, one single record |
| episodic | **separate** | `episode` | `episodic_store.jsonl`, episodes alone |
| semantic (personal) | **unified** | `user_fact` + `user_preference` | `user_profile_store.jsonl`, both types interleaved in ONE file |
| semantic (world) | **split across stores** | knowledge-DB tables | SQLite DB (TF-041), read by `KnowledgeForge` |
| identity | outside the tiers | `PersonalityCoreRecord` | `personality_core.yaml` |
| procedural | outside the tiers | `response_pattern` | `response_patterns.jsonl` |

**Working** is hinted twice, though never named. The thread
state is a single ephemeral record with the aggressive expiry
its docstring promises (`expires_after_turns=10`, cooled once
per completed turn since TF-045), and it is the one memory
record the core reads on *every* turn (routing, Step 1). The
tier's other half is transient: the turn state, input sketch,
scaffold, activation bundle, and the caller-owned `ChatHistory`
window (TF-024: 20 turns / 1,000 word-tokens / 2,000 chars).

**Episodic** is the one tier the code names outright — its own
record type, its own file, the shortest stale horizon (30
turns), a 7-day half-life, and the only cross-tier links in the
estate: an episode's `linked_preferences` / `linked_facts`
point into the semantic tier (proven on a synthetic store: the
links resolve to real semantic-tier ids).

**Semantic** is where the code refuses to separate. Facts and
preferences share one file and one loader
(`load_profile_records` returns both, in append order) — two
kinds of knowing-about-the-user, one home. The world half
(entities, statements, labels, ConceptNet edges, GeoNames
places, DBpedia entries, reference chunks) lives in a different
store entirely, behind the ETL wing and `KnowledgeForge`.

**Outside the three**, by ruling: *identity* (the personality
core — protected, never auto-pruned, half-life 9,999 days,
loaded at core construction, steering every turn; it is who is
remembering, not a tier of memory) and *procedural* (response
patterns — own file, 30-day half-life, 100-turn stale horizon,
pattern-learning scores: a genuine fourth-tier hint, recorded
as such; no fourth tier is built on a hint alone).

## 3. The ladder is the tier ordering

One table, `HALF_LIFE_DAYS`, orders the whole estate, and its
order *is* the tier order (TF-041 K3):

working 0.07 d < episodic 7 d < procedural 30 d <
preference 90 d < fact 180 d < identity 9,999 d

Measured consequence: after one day, a working record's
recency has decayed below 0.001 while a fact's is still above
0.99 (`compute_recency_score`, pinned in tests). The stale
horizons order the same way: episode 30 < pattern 100 <
preference 200 < fact 500 turns.

## 4. What is unified

One `MemoryStore` and one `load_all` (5 sections) over all
tiers; the personal-semantic file (§2); one tag index rebuilt
over profile + episodes + patterns (a single tag can return a
fact, an episode, and a pattern together — proven in tests);
one tier-blind recent queue; one lifecycle prune pass loading
and rewriting all three JSONL stores (TF-045); one ingestion
contract gating every tier's appends (TF-042); and the one
half-life ladder (§3). Tier separation in this codebase is a
matter of *homes and horizons*, never of separate machinery.

## 5. The honest half: consumption

Separation is storage-deep, not yet retrieval-deep:

- The core consumes the **working** tier every turn and the
  **identity** row at construction.
- Retrieval (`core._retrieve`) reads the **world semantic**
  store only, via `KnowledgeForge`.
- **No code outside store/lifecycle loads the personal tiers
  at all** — `load_episodes`, `load_patterns`, and
  `load_profile_records` have no caller in the core. Episodes,
  preferences, facts, and patterns are written, aged, and
  pruned, but no turn ever reads them back.

That gap is pinned in `tier_problems()` (a core that starts
calling those loaders fails the checker by name), so the slice
that wires personal-tier retrieval must flip the pin with
evidence. The inbound leg — what a chat writes back — is
TF-047's slice, next in roadmap order.

## 6. Proof grade

FIXTURE (synthetic stores and records on this VM) + live-tree
proofs (file map, ladder, horizons, and consumption re-derived
from the shipped code by `tier_problems()`). Live-store tier
behaviour over real use is NOT PROVEN — no real store exists
on this machine (TF-041). No counts were retuned and no
storage moved: this slice names the tiers the code already
keeps.
