# ThoughtForge Architecture — The Core Loop (TF-019)

The architecture doc for the cognition core: how one user turn travels
through ThoughtForge, drawn from the code as it stands, not from any
plan. Mapped 2026-10-30 on branch `yrsa/thoughtforge-road-build`.
Every claim below names the module it was read from; every place where
the code disagrees with its own docstrings or with the roadmap's
shorthand is listed as a numbered finding in §8, with a proposed
owning slice. TF-039 reconciles this doc with shipped behaviour at
the end of Part II; TF-182 reconciles it again for developers.

**TF-039 reconciliation (2026-10-31):** this doc's §8 findings
now carry a status table (5 closed, 2 ruled, F2 still open under
TF-041), and TF-039 notes are appended where TF-020–TF-038
changed the behaviour mapped below (drivers, config caps,
retained state, traces). The original TF-019 text stands
unrewritten as the map it was; the reconciled-as-data record is
`src/thoughtforge/cognition/core_docs.py` with the write-up in
`docs/architecture/CORE_DOCS_RECONCILED.md`.

Principal source: `src/thoughtforge/cognition/core.py`
(`ThoughtForgeCore.think`, 735 lines in the module). Supporting
modules: `cognition/router.py`, `cognition/scaffold.py`,
`cognition/prompt_builder.py`, `cognition/chat_history.py`,
`knowledge/forge.py`, `knowledge/models.py`, `knowledge/store.py`,
`refinement/salvage.py`, `refinement/enforcement.py`.

## 1. The loop at a glance

```
user text
   │  sanitise_query (utils/validators)
   ▼
[1] ROUTE      InputRouter.route ──────────────► InputSketch
   │             reads: MemoryStore.load_thread_state (tags only)
   ▼
[2] RETRIEVE   ThoughtForgeCore._retrieve ─────► MemoryActivationBundle
   │             KnowledgeForge.retrieve(query, path, top_k=10)
   ▼
[3] SCAFFOLD   ScaffoldBuilder.build ──────────► CognitionScaffold
   │             + optional ChatHistory text folded into fact_block
   ▼
[4] GENERATE   _generate_candidates ───────────► list[CandidateRecord]
   │             local engine │ unified backend │ none
   ▼
[5] FRAGMENT   _score_and_extract_fragments ───► list[FragmentRecord]
   │             sentence-split, per-sentence scores, keep ≥ 0.25
   ▼
[6] COMPOSE    _compose_final ─────────────────► FinalResponseRecord
   │             FragmentSalvage.forge (≤2 refine passes)
   │             then EnforcementGate.check (cite / length / generic)
   │        — or, with no candidates —
[6′] KNOWLEDGE-ONLY  _knowledge_only_response ─► FinalResponseRecord
   ▼
[7] QUALITY GATE  composite < 0.40 and candidates exist
   │             → _repair (one tighter, cooler generation)
   │             → kept only if its composite beats the original
   ▼
FinalResponseRecord  (the only object a caller ever receives)
```

There is **no step 8**. A turn writes nothing back to persistent
memory: see §7.

## 2. Construction — what a core is made of

`ThoughtForgeCore.__init__(config, model_path, memory_dir, db_path,
backend)` wires, in order:

| Component | Type | Role in the loop |
|---|---|---|
| `_knowledge` | `KnowledgeForge(db_path or default)` | retrieval (§4, step 2) |
| `_store` | `MemoryStore(memory_dir or default)` | personality + thread state reads |
| `_router` | `InputRouter()` | step 1 |
| `_scaffold_builder` | `ScaffoldBuilder()` | step 3 |
| `_prompt_builder` | `PromptBuilder()` | steps 4, 6, 7 prompts |
| `_salvage` | `FragmentSalvage()` | step 6 |
| `_enforcement` | `EnforcementGate()` | step 6 |
| `_perf` | shared perf tracker (`utils/perf`) | `think.total/retrieval/generation` timings |
| `_engine` | `None` until lazily loaded | `TurboQuantEngine`, step 4 branch A |
| `_unified_backend` | caller-supplied or `None` | step 4 branch B |

Personality is loaded once at construction: first
`MemoryStore.load_personality_core()`, and if that returns nothing,
`_load_personality_from_config` tries the YAML at
`config["cognition"]["personality_core_path"]` (relative paths resolve
against the project root). A best-effort `SelfHealer().heal_all()`
runs at startup; its failure is logged at debug level and never
blocks construction.

Callers in the tree: `run_thoughtforge.py` (single-shot, REPL, and
chat modes), `setup_thoughtforge.py` (one smoke `think`), and
`locustfile.py` (concurrent knowledge-only load). `think()` is the
only public turn entry point; `load_model` / `unload_model` manage
the local engine explicitly.

*TF-039 reconciliation:* `think()` is no longer the only driver —
`think_stream()` (TF-026) and `resume_turn()` (TF-027) drive the
same `_run_turn` pipeline, and the inspection surface
(`last_turn_state`, `last_turn_trace`, `last_turn_latency`, the
conversation properties) is frozen at TF-034 with its post-freeze
minor additions recorded at TF-039 (CORE_API_FREEZE_V1 §7).

## 3. Step 1 — Routing (`cognition/router.py`)

`InputRouter` is heuristic-only: regex signal tables, no model call,
deterministic. `route(raw_text, thread_state)` returns an
`InputSketch` (`knowledge/models.py`):

- **Intent** — one of 9: `technical_debugging`,
  `technical_spec_request`, `editing_refinement`, `brainstorming`,
  `naming_or_branding`, `planning_request`, `emotional_support`,
  `factual_query`, or the fallback `light_conversation` when no
  signal matches. Scored by counting matching patterns per intent;
  highest count wins.
- **Tone in** — one of `frustrated | tired | urgent | curious |
  playful | focused`, else `neutral`, same counting method.
- **Response mode** — fixed table from intent (`_INTENT_TO_MODE`):
  e.g. both technical intents → `structured_technical`,
  `emotional_support` → `calm_supportive`, `factual_query` →
  `factual_direct`.
- **Topic** — the raw text with a leading question/command prefix
  stripped, truncated to 6 words; falls back to the intent name.
- **Memory triggers** — stopword-filtered terms (length > 2), deduped,
  max 8 from the message, plus up to 3 tags from the active thread
  state, capped at 10. (Retrieval itself re-derives its own query
  from `sketch.raw_text`; the triggers ride the sketch — see §8-F5.)
- **Urgency** — float: base 0.3, +0.4 urgent tone / +0.2 frustrated /
  +0.1 tired, +0.15 debugging / +0.1 emotional support, capped 1.0.
- **Retrieval path** — `sql` for {factual, spec, planning}, `vector`
  for {emotional, brainstorming, light conversation}, else `hybrid`.

A caller-supplied `retrieval_path` argument to `think()` overwrites
`sketch.retrieval_path` after routing.

The router's full decision table — scoring, tie-breaking, every
intent/tone/mode/retrieval/urgency row, and the TF-020 dead-pattern
fix — is `docs/architecture/ROUTER_DECISION_TABLE.md`.

Measured live this slice (router alone): “fix this traceback error,
it crashes” → `technical_debugging / neutral / structured_technical /
hybrid / 0.45`; “what is a ferry?” → `factual_query / sql / 0.30`;
“I feel exhausted and overwhelmed” → `emotional_support / tired /
calm_supportive / vector / 0.50`.

## 4. Steps 2–3 — Retrieval and scaffold

**Retrieve.** `core._retrieve(sketch)` calls
`KnowledgeForge.retrieve(query=sketch.raw_text,
path=sketch.retrieval_path, top_k=10)` — SQL and/or vector passes,
score fusion, minimum-score filter, ranked
`MemoryActivationBundle(activated_records, total_retrieved,
retrieval_confidence)`. Two failure softenings: if the knowledge DB
file does not exist, or retrieval raises, the turn continues with an
empty bundle. (The existence check has a defect — §8-F2.)

**Scaffold.** `ScaffoldBuilder.build(sketch, bundle, personality)`
is table-driven, no model call. It derives, all from the sketch plus
the personality core:

- `goal` — one fixed sentence per intent (`_INTENT_TO_GOAL`).
- `tone` — response-mode defaults merged with a user-tone overlay,
  max 4 entries.
- `focus` — 2–3 fixed entries per intent.
- `avoid` — 3 universal entries + intent extras + the personality
  core's first 3 `avoid` entries (the only expressive personality
  field the live loop consumes beyond this — cf. TF-009 D-CFG-2).
- `depth` — `expert` (spec/debugging/planning, unless urgency ≥ 0.8
  → `medium`), `light` (light conversation, naming, emotional),
  else `medium`.
- `candidate_modes` — an ordered pair per intent
  (`_INTENT_TO_CANDIDATE_MODES`), e.g. factual →
  `[implementation_friendly, strict_spec]`.
- `fact_block` — top 8 activated records by score, one compact line
  each: `[QID] label — description≤120 chars`, or `[source] label`,
  or the record cue.

**History fold-in.** If the caller passes a `ChatHistory`,
`core.think` renders `history.to_prompt_context(max_chars=1500)`
and prepends it into `scaffold.fact_block` (labelled
`Conversation history:`; the knowledge part is re-labelled
`Knowledge:` when both exist). This happens *after* scaffold
construction, inside `think` — the scaffold builder never sees
history. `ChatHistory` itself (20-turn default window, OpenAI-format
export, JSON save/load) is owned and persisted by the caller;
`run_thoughtforge.py` chat mode adds the user and assistant messages
around the `think()` call. The core never touches it otherwise.

## 5. Step 4 — Generation: three branches

`_generate_candidates` picks exactly one branch:

**A. Local engine (`TurboQuantEngine`).** Loaded lazily on the first
turn when `model_path` was given (load failure → warning, turn
continues knowledge-only). For each of `scaffold.candidate_modes[:2]`
— at most two passes — it builds a candidate prompt
(`PromptBuilder.build_candidate_prompt`: mode opener, user request,
intent, topic, tone, goal, focus, avoid, up to 3 memory cues, the
fact block as `Knowledge context:`, and an output-shape + length
hint) and generates with a per-mode temperature
(`_mode_temperature`: 0.50 strict_spec … 0.80 brainstorm_distinct,
default 0.60). Each result becomes a `CandidateRecord` scored by
`_score_candidate` (keyword-overlap relevance, length clarity, a
0.6/0.4 sentence-count coherence constant, genericness-penalised
personality fit, regex specificity, goal fit).

**B. Unified backend** (Ollama / LM Studio / HF etc., when no local
engine exists but a backend was supplied): exactly **one** candidate,
first candidate mode only, `GenerationRequest(max_tokens=512)` —
the 512 is hard-coded here (profile-blindness, TF-012 D-PROF-3).
A backend-reported error yields zero candidates.

*TF-039 reconciliation:* since TF-032 the 512 is the *default*,
not a hard code — a conversation selecting a profile gets that
profile's `generation.max_response_tokens` as its cap on the
plain and streamed paths alike (phone_low → 180, measured in
TF-032/TF-037). Backend errors now also land on the turn as
TF-031 failure records, not only as zero candidates.

**C. No engine, no backend:** zero candidates → step 6′.

A generation exception inside a mode is logged and that mode is
skipped; it does not fail the turn.

## 6. Steps 5–7 — Fragments, compose, quality gate

**Fragment scoring (step 5).** Every candidate is split into
sentences (`re.split` on `.!?`); sentences under 4 words are dropped;
each remaining sentence is scored (`_score_fragment`) and kept when
its composite ≥ `_MIN_FRAGMENT_SCORE = 0.25` (core's own threshold —
`FragmentSalvage` uses a different one, 0.54; §8-F6). Fragments are
sorted best-first onto the turn state. Note: the final record's
`source_fragment_ids` come from this list, while the salvage engine
re-extracts its own fragments internally — two fragment systems run
per turn (TF-022/TF-039 reconciliation material).

**Compose (step 6).** `_compose_final` delegates text production to
`FragmentSalvage.forge(candidates, bundle, engine, prompt_builder,
sketch, scaffold)`:

1. Drafts scored `composite = 0.45·length + 0.55·citation`
   (length = chars/550 capped; citation = fraction of retrieved QIDs
   present, neutral 0.5 when the bundle holds no QIDs).
2. Fragments extracted from the top 2 drafts (threshold 0.54,
   max 5 kept).
3. If a **local engine** is present, up to 2 refine passes
   (`_MAX_REFINE_PASSES`) at temperatures 0.50, 0.45: each pass
   generates from the refine prompt and replaces the best text only
   if its score improves. Unified-backend turns never refine —
   salvage falls through to the best draft.

   *TF-039 reconciliation:* since TF-037 the pass count is the
   conversation's resolved refinement cap — the selected
   profile's `max_refinement_passes` where one is selected
   (phone_low 1, pi_zero 0), the standing 2 otherwise. TF-037
   found and fixed the pre-fix state, in which the profile's
   promise was read by no cognition code.
4. Fallback: best-scoring draft verbatim (`salvage_path="best_draft"`).

Then `EnforcementGate.check(text, citations, retrieved_qids)` runs
three checks — length ≥ 5 words, at least one retrieved QID cited
(only when QIDs were retrieved), no hard generic phrase (“as an
ai”, “i cannot provide”, …). Failure is **soft**: a `[Forge: …]`
note is appended to the text and `enforcement_passed=False` with
notes; the text is never blocked or replaced. The
`FinalResponseRecord` is then scored by `_score_final` (composite:
relevance .24, clarity .18, coherence .18, personality .14,
usefulness .16, goal fit .10, − genericness .10; tiers at
.85/.70/.55/.40) and returned with citations, kept-fragment ids,
`salvage_path`, `retrieval_confidence`, and `mode="full"`.

**Knowledge-only (step 6′).** With zero candidates: if the bundle is
empty, a fixed honest refusal — “The forge has no knowledge relevant
to this query.” (9 tokens, composite 0.408, `enforcement_passed=
False`, notes `no knowledge available`; measured live this slice on
the TF-017 synthetic store with no DB). If records exist, the top 5
record cues are listed as bullet lines under
“Retrieved knowledge relevant to '<topic>':”, with fixed hand-set
scores (relevance 0.6 …), `mode="knowledge_only"`.

**Quality gate / repair (step 7).** If the composed final's composite
< `_QUALITY_REPAIR_THRESHOLD = 0.40` **and** candidates existed, one
repair generation runs: `build_repair_prompt` (top 2 memory cues,
more constrained) at temperature 0.45, local engine only. The repair
replaces the final only when its `_score_final` composite is strictly
higher. Repair does **not** pass through salvage or the
`EnforcementGate`; it uses core's local `_check_citation_gate`
helper and stamps `salvage_path="repair"`, `mode="repair"`. With a
unified-backend-only core, `_repair` returns the previous final
unchanged (its engine is `None`).

**Turn bookkeeping.** `think()` accumulates a `RuntimeTurnState`
(sketch, bundle, scaffold, candidates, fragments, final, retrieval /
generation milliseconds, token count = final's word count) and
records three perf metrics. The turn state is then discarded —
callers receive only the `FinalResponseRecord` (§8-F3).

*TF-039 reconciliation:* the discard is history. Since TF-025
the state is phased, retained on `core.last_turn_state`, and
carries a measured `salvage_ms`; since TF-038 the whole turn
exports as one trace via `core.last_turn_trace()`. §8-F3 is
CLOSED (§8 status table).

## 7. Memory: the loop reads, it does not write

The roadmap shorthand for the lifecycle ends “… → memory”. As built,
that leg does not exist in the core:

- Reads per turn: thread state (router tags), knowledge DB
  (retrieval), personality (construction time).
- Writes per turn: **none** to `MemoryStore` or the knowledge DB.
  Only perf-tracker metrics and log lines leave the process side.

The write-back machinery exists as unconnected parts:
`WritebackRecord` (`models.py` §13), write-back scoring formulas
(`knowledge/scoring.py`: preference / episode / pattern),
`PromptBuilder.build_episode_summary_prompt` and
`build_preference_inference_prompt`, and `MemoryStore` append
methods — none of it is invoked by `think()` or any caller in the
tree. Conversation → memory is TF-047's slice; this doc records the
gap so Part II work does not assume it.

## 8. Findings — where code and its own story disagree

Numbered for the TF-016 docket's closure rule (owning slice flips
status with evidence; TF-192 verifies none left open without a
reason). All verified by reading, F1/F3 also by the live trace.

**TF-039 reconciliation — status of every finding:**

| Finding | Status | Settled by |
|---|---|---|
| F1 `num_drafts` dead parameter | CLOSED | TF-034 (wired as a draft cap) |
| F2 retrieval gate checks the wrong path | CLOSED | TF-041 (gate now reads the configured `self._knowledge.db_path`; proven both ways — was OPEN at TF-039) |
| F3 turn state built and discarded | CLOSED | TF-025 (retained, `salvage_ms` measured) + TF-038 (trace export) |
| F4 module docstring's 9 steps ≠ code's steps | CLOSED | TF-039 (docstring rewritten to the reconciled 7-step loop) |
| F5 memory triggers computed, not consumed | RULED | TF-020 (extraction is the router's contract) |
| F6 two fragment thresholds / systems | RULED not-a-defect | TF-022 (different instruments, cross-documented) |
| F7 `prompt_text` never filled | CLOSED | TF-023 (field removed) |
| F8 brainstorming duplicates a mode | CLOSED | TF-021 (`[brainstorm_distinct, practical]`) |

The same record as data, with live evidence checks, is
`RECONCILED_FINDINGS` in `src/thoughtforge/cognition/core_docs.py`.

- **F1 — `num_drafts` is a dead parameter.** `think(user_text,
  retrieval_path, num_drafts)` documents it as “Override draft count
  (default from hardware profile)”; `_generate_candidates` accepts
  and never uses it. Draft count is fixed by construction: ≤2 local
  modes, exactly 1 unified. Proposed owner: TF-034 (core API freeze
  must either wire or remove it).
- **F2 — Retrieval gate checks the wrong path.**
  `core._retrieve` tests `get_knowledge_db_path().exists()` — the
  *default* DB path — not `self._knowledge.db_path`. A core built
  with a custom `db_path` is gated by the default file's existence:
  a valid custom DB is skipped when the default is absent, and a
  missing custom DB is not what the check detects. Proposed owner:
  TF-041 (knowledge store mapped). **TF-041 update: FIXED** —
  `_retrieve` now gates on `self._knowledge.db_path` (the
  configured path) and never calls the default-path helper; the
  behaviour is proven both ways in
  `tests/test_tf041_knowledge_store_map.py` (a populated custom
  DB is retrieved while the default is absent; a missing custom
  DB is not even created while the default exists), and the
  `RECONCILED_FINDINGS` record in `core_docs.py` was flipped to
  closed with this evidence, as TF-039's checker instructed.
- **F3 — Turn state is built, timed, and thrown away.**
  `RuntimeTurnState.salvage_ms` is never assigned (compose/salvage
  time is unmeasured), and the state object itself is local to
  `think()` — no caller, log sink, or trace export receives it.
  Proposed owners: TF-025 (turn state), TF-038 (traces).
- **F4 — The module docstring's 9 steps are not the code's steps.**
  `core.py`'s header lists prompt-building and enforcement as
  top-level steps and omits fragment scoring; in code, enforcement
  lives inside `_compose_final`, fragment scoring is its own step
  between generation and compose, and generation timing
  (`generation_ms`) spans candidate generation *and* fragment
  scoring. This doc (§1) is the corrected map; the docstring is
  TF-039 reconciliation material.
- **F5 — Memory triggers are computed but not consumed.**
  `InputSketch.memory_triggers` is populated every turn, but
  retrieval queries with `sketch.raw_text` and prompt cues come from
  the bundle; nothing in the loop reads the triggers. Proposed
  owner: TF-020 (router hardening — decision table must rule on
  them) or TF-041.
- **F6 — Two fragment thresholds, two fragment systems.** Core keeps
  fragments at composite ≥ 0.25 for the record; salvage keeps its
  own at ≥ 0.54 for refinement, re-derived from the same candidates.
  Neither references the other. Proposed owner: TF-022 (scaffold /
  salvage interface) with TF-039 reconciliation.
- **F7 — `CognitionScaffold.prompt_text` is never filled.**
  `ScaffoldBuilder` sets it to `""` with the comment “filled in by
  PromptBuilder”; nothing in the tree ever assigns it — the prompt
  builder returns prompt strings directly to its callers. Proposed
  owner: TF-023 (prompt builder contract must rule on the field).
- **F8 — Brainstorming generates the same mode twice.**
  `_INTENT_TO_CANDIDATE_MODES["brainstorming"]` is
  `["brainstorm_distinct", "brainstorm_distinct"]` — two local
  passes with identical mode and temperature, differing only by
  sampling. Every other intent pairs two distinct modes. Proposed
  owner: TF-021 (scaffold catalogue — rule on it there).

## 9. Verification for this slice

- Mapped from a full read of `core.py`, `router.py`, `scaffold.py`,
  `prompt_builder.py`, `chat_history.py`, `salvage.py`,
  `enforcement.py`, plus the relevant sections of `forge.py`,
  `models.py`, `store.py`, and the `think()` callers.
- Live trace on the TF-017 synthetic store (Wrenna / Lantern Hollow,
  no DB): personality resolved from the store, `think()` returned
  the knowledge-only empty-bundle response exactly as §6′ describes
  (mode `knowledge_only`, 9 tokens, composite 0.408, tier `weak`,
  enforcement notes `no knowledge available`); router outputs in §3
  measured on the same run.
- Full suite re-run at banking; counts recorded in
  `CAMPAIGN_STATE.md` (TF-014 zero line held or the delta named).
