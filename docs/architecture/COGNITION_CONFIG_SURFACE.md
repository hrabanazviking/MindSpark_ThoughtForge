# TF-032 — Cognition Config Surface: Scaffolds/Profiles per Conversation

*Part II, slice TF-032. Code:
`src/thoughtforge/cognition/config_surface.py`
(`ConversationConfig`, `resolve_conversation_config`,
`apply_conversation_config`), wired minimally into
`ThoughtForgeCore` (construction, the scaffold step, and the two
unified-backend generation paths). Tests:
`tests/test_tf032_config_surface.py` (42 tests).*

## 1. The gap, as found

Before this slice a conversation could not be configured at all:

- **The core read one config key.** `ThoughtForgeCore.__init__`
  stored the config dict and read only
  `cognition.personality_core_path` from it (TF-009's census: the
  sole live cognition key in `default.yaml`). Everything else in
  the file was shadowed by hard-coded constants or dead (27
  SHADOWED + 2 DEAD of 34 leaves — D-CFG-1).
- **The core was profile-blind (TF-012, D-PROF-3).** No profile
  parameter existed (its benchmark child, D-BENCH-1: a bogus
  `profile_id` in `benchmark_profiles` exits 0 and changes
  nothing). The unified backend's token cap was the literal `512`
  in two places in `core.py`, on every profile.
- **TF-022's strategies were unselectable.** The strategy library
  ships three strategies on one interface, and its doc recorded:
  "TF-032 owns making strategies selectable per conversation
  through config." Nothing could select one.
- **No merged view (D-CFG-5).** Three loaders never met; no code
  path saw one conversation's cognition selections together.

## 2. The live-keys-only rule (TF-009's brief, honoured as law)

TF-009 §6 named TF-032's exact allowance: the genuinely live keys
— `logging.*` (4), `cognition.personality_core_path` (1), and the
10 `user_config.yaml` backend keys. This surface's rulings:

| Key family | Ruling |
|---|---|
| `cognition.personality_core_path` | **Carried through** in the resolved view; the core's existing construction-time load from it is unchanged. |
| The 10 live backend keys | **Not touched.** The backend factory already reads them live; TF-065/TF-153 own their shape. This surface is cognition only. |
| `logging.*` | **Not touched.** `setup_logging` already reads them live. |
| `hardware.profile`, `hardware.profiles_dir` (SHADOWED) | **Never read here** — pinned by test. Profiles resolve through the live loaders (`list_hardware_profiles` / `load_hardware_profile`), not through a file string no code ever read. |
| `cognition.scaffold_depth` (DEAD file key) | **Never read here** — pinned by test. The per-conversation key is the new `conversation.depth`; reviving the file key would repeat D-CFG-1's mirage. |
| `profile` / `knowledge_db_built` / `chat_*` (DEAD user_config keys) | **Never read here** — pinned by test. |

Everything selectable is a **new** key in a `conversation` section
of the config dict handed to one core instance — a shape that did
not exist at TF-009's census, so no shadowed or dead key is being
quietly revived. One core instance is one conversation (the
campaign's standing shape: one will ledger per instance, TF-029;
one persona expectation per instance, TF-030), so a per-instance
config dict *is* per-conversation config. The shipped
`default.yaml` has no `conversation` section, and
`load_config()`'s local-override merge can supply one without any
file in the tree changing.

## 3. The surface

`resolve_conversation_config(config)` returns a frozen
`ConversationConfig` (dict round-trip exact):

| Selection (`conversation.*`) | Validated against | Effect |
|---|---|---|
| `profile_id` | `list_hardware_profiles()` — the six shipped profiles | Profile loaded at construction and held on the core (`core.conversation_config`); its `generation.max_response_tokens` becomes the unified backend's cap (plain + streamed). |
| `strategy` | TF-022's strategy registry (`chain_of_thought`, `plan_then_answer`, `self_check`) | Resolved and exposed as `core.selected_strategy`. **Selection only** — see §5. |
| `depth` | `light` / `medium` / `expert` (the prompt builder's length-hint table) | Replaces the scaffold's derived depth every turn. |
| `candidate_modes` | TF-021's candidate-mode registry; 1–2 entries, no duplicates | Replaces the scaffold's intent-derived candidate modes every turn. |

Invalid selections — unknown profile / strategy / depth / mode,
a duplicated mode, an empty or 3+-long mode list, a bare string
where a list belongs, a non-mapping section — raise `ValueError`
at resolution, naming the valid options. A bogus profile can no
longer silently pass *through this surface* (the benchmark's own
validation stays TF-121/TF-128's).

## 4. What it does, measured

All six profiles resolve to their JSON's own cap: desktop_cpu
400, desktop_gpu 600, phone_low 180, pi_5 250, pi_zero 120,
server_gpu 2000. Through a real core with a capturing stub
backend: a `phone_low` conversation's backend request carries
`max_tokens=180` on the plain path and on the streamed path; a
conversation with no profile carries the standing 512 — the
pre-TF-032 behaviour, byte-identical. Through a real
knowledge-only core, a selected `depth="light"` +
`candidate_modes=["practical"]` land on the turn's retained
scaffold exactly, whatever the intent derived. Application happens
once, in `_run_turn`'s scaffold step; a resumed turn reuses its
retained scaffold and is not re-applied (the TF-027/TF-029
pattern). A default `ConversationConfig` leaves a built scaffold
untouched (pinned against the builder's own output).

## 5. Deliberate non-goals, stated plainly

- **Strategy execution is not wired.** TF-022 recorded that the
  core loop does not run strategies; pass budgets, streaming and
  traces for strategy execution belong to TF-033/TF-038's
  surfaces. This slice makes strategies *selectable and resolved*
  — claiming the loop runs them would be false.
- **The local-engine path gains no token cap.** It takes no cap
  from anywhere today (TF-021 recorded it); the profile cap lands
  where a cap already existed — the unified backend's 512.
- **The other flat constants stay flat.** Retrieval top_k,
  salvage passes, history budgets and mode temperatures are
  unchanged on every profile: TF-033 (latency budgets), TF-037
  (small-model floor) and TF-098 (refinement passes) own them.
  D-PROF-3 is **narrowed, not closed**: the core is no longer
  profile-blind for the unified-backend token cap when a profile
  is explicitly selected.
- **No file changes.** No shipped config file gains or loses a
  key by this slice; the mirage keys stay exactly as TF-009 found
  them until their owning slices wire or strike them.

## 6. Docket verdicts (TF-016 closure rule)

- **D-CFG-1 — narrowed, still OPEN.** The live-keys-only half is
  discharged: this surface reads the one live cognition key and
  no shadowed/dead key (pinned). The scoring-weights half stays
  TF-044's, and the mirage keys themselves stay unwired/unstruck
  until their owners land.
- **D-CFG-5 — narrowed, still OPEN.** For cognition, one merged
  view now exists: `ConversationConfig` is the single resolved
  object a conversation's selections live in. The system-wide
  merged view (service shape) stays TF-153's.
- **D-PROF-3 — narrowed, still OPEN** (see §5): profile-selected
  token cap only; TF-033/TF-037 keep the rest.
- **D-DEAD-1 — TF-032's share ruled, entry stays OPEN.**
  `validators.validate_config` requires the user_config `backend`
  key: it validates a *backend* dict, the wrong shape to guard a
  cognition config, so it is **not wired here** (wiring it would
  refuse every cognition config that carries no backend — i.e.
  all of them). Its wire-or-retire verdict belongs to TF-094 /
  TF-065, where the backend and policy surfaces live. This
  surface validates its own selections itself (§3).

Suite at banking: see CAMPAIGN_STATE.md (TF-014 zero line + this
slice's 42 tests exactly, the one known env failure D-ENV-1
unchanged).
