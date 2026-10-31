# TF-009 — Config Census (Part I Audit)

Measured 2026-10-30. The three shipped files under `configs/`
inventoried leaf by leaf (34 + 15 + 15 = 64 leaf keys) and traced
to the code that reads them: an AST/text scan of `src/`, the root
scripts, and `benchmarks/`, plus a manual read of every loader
(`utils/config.py`, `inference/unified_backend.py`,
`cognition/core.py`, `utils/logging_setup.py`,
`knowledge/store.py`, `utils/validators.py`, `utils/health.py`,
`utils/self_heal.py`, `setup_thoughtforge.py`).

Classification used throughout:

- **LIVE** — a code path reads that key from that file.
- **SHADOWED** — no code reads the config key, but a hard-coded
  constant/argument default with the *same value* does the job,
  so editing the config silently changes nothing.
- **DEAD** — no reader and no equivalent; the value is inert.
- **DORMANT** (personality file only) — parsed into a record and
  carried, but no downstream code ever reads the field.

## 1. `configs/default.yaml` — the runtime config that mostly isn't

Loader: `utils/config.py:load_config()` — reads `default.yaml`,
deep-merges `configs/local_config.yaml` if present (the file does
not exist in the tree; the header comment in `default.yaml` calls
it gitignored), caches the result in a module-global. Only two
callers exist in the whole tree: `cognition/core.py:86`
(`ThoughtForgeCore.__init__`) and `forge_memory.py:31`
(`setup_logging(load_config())`).

| Section / key | Class | Reader or hard-coded twin |
|---|---|---|
| `hardware.profile` (`auto`) | SHADOWED | No config reader. `run_thoughtforge.py --profile` default `auto` + `inference/profiles.py:ProfileSelector` auto-detection |
| `hardware.profiles_dir` | SHADOWED | `utils/paths.py:get_hardware_profiles_dir()` hard-codes `hardware_profiles/` |
| `knowledge.db_path` (`data/knowledge.db`) | SHADOWED | `utils/paths.py:get_knowledge_db_path()` hard-codes the same path |
| `knowledge.embeddings_dir` | SHADOWED | `utils/paths.py:get_embeddings_dir()` hard-codes `data/embeddings` |
| `knowledge.embedding_model` (`all-MiniLM-L6-v2`) | SHADOWED | `etl/embeddings.py:30` `DEFAULT_MODEL = "all-MiniLM-L6-v2"` |
| `knowledge.reference_data_dir` | SHADOWED | `utils/paths.py:get_knowledge_reference_dir()` hard-codes it |
| `retrieval.scoring_weights.*` (5 keys) | SHADOWED | `knowledge/scoring.py:RETRIEVAL_WEIGHTS_BASE` — same values (0.35/0.20/0.20/0.10/0.15), **different key names** (see Finding 2) |
| `memory.default_pruning_mode` (`routine`) | SHADOWED | `knowledge/lifecycle.py:131` `prune(mode="routine")` default argument |
| `memory.pruning_score_weights.*` (7 keys) | SHADOWED | `knowledge/scoring.py:PRUNE_SCORE_WEIGHTS` — values identical, key for key |
| `memory.survival_score_weights.*` (6 keys) | SHADOWED | `knowledge/scoring.py:SURVIVAL_SCORE_WEIGHTS` — values identical, key for key |
| `cognition.personality_core_path` | **LIVE** | `cognition/core.py:717` `_load_personality_from_config()` — the *only* non-logging key `ThoughtForgeCore` ever reads from its own config |
| `cognition.scaffold_depth` (`medium`) | **DEAD** | No reader. Depth is derived per-turn by `cognition/scaffold.py:_derive_depth(sketch)` |
| `refinement.fragment_scoring_weights.*` (2 keys) | SHADOWED | `refinement/salvage.py:38-39` `_LENGTH_WEIGHT = 0.45`, `_CITATION_WEIGHT = 0.55` |
| `refinement.citation_confidence_threshold` (0.70) | **DEAD** | No reader. `refinement/enforcement.py` has no threshold at all — its citation check is binary QID presence |
| `logging.level / log_dir / log_to_file / log_to_console` (4 keys) | **LIVE** | `utils/logging_setup.py:19-40` `setup_logging()` reads all four |

**Totals for `default.yaml`: 34 leaf keys — 5 LIVE, 27 SHADOWED,
2 DEAD.** The file is, in effect, documentation of constants that
live in code. A user editing 27 of its 34 values changes nothing
and gets no warning.

## 2. `configs/personality_core.yaml` — parsed fully, used barely

Two different files share this name, and the shipped one is the
*fallback*:

1. Primary: `knowledge/store.py:45` — `MemoryStore` loads
   `personality_core.yaml` from the **memory directory**
   (`data/memory/`), not from `configs/`.
2. Fallback: only if the store has none does
   `cognition/core.py:106` load the `configs/` copy, via the one
   live `default.yaml` key above.

All 15 leaf keys map 1-to-1 onto `knowledge/models.py:
PersonalityCoreRecord` fields and survive the round trip
(`from_dict` filters to dataclass fields; every shipped key is a
field — no orphan keys in the file). But downstream consumption
is a different story:

| Field(s) | Class | Consumer |
|---|---|---|
| `record_id` | LIVE | Logged at startup (`core.py:118`) |
| `avoid` | LIVE | `cognition/scaffold.py:190-193` — first 3 entries merged into the scaffold's avoid list; the only personality field that steers a turn |
| `traits`, `speech_style`, `behavior_rules`, `weights.{directness,warmth,depth,wit,formality}`, `version`, `status`, `retention_class`, `created_at`, `updated_at` (13 leaves) | **DORMANT** | Loaded and carried on the record; a full-tree grep finds no reader outside `models.py` (definition + `to_dict`) |

So the persona's voice — traits, speech style, behaviour rules,
and the five tuning weights — currently has **no effect on any
prompt**. `PromptBuilder` never sees the personality record; the
scaffold takes only its `avoid` list. (This is TF-030/TF-141
evidence, recorded here, not fixed here.)

## 3. `configs/user_config.yaml` — two configs in one file

Written by `setup_thoughtforge.py:_write_config()` (known keys in
a fixed order, then any extras), read at runtime by exactly one
function: `inference/unified_backend.py:load_backend_from_config()`.
`utils/validators.py:validate_config()`, `utils/health.py` and
`utils/self_heal.py` also parse it, but only for the `backend`
key and backend-specific fields.

| Key | Class | Reader |
|---|---|---|
| `backend` | LIVE | `unified_backend.py:68`; also the sole required key in validators/health/self-heal |
| `ollama_url`, `ollama_model` | LIVE | `unified_backend.py:77-78` |
| `lmstudio_url`, `lmstudio_model` | LIVE | `unified_backend.py:83-84` |
| `openai_api_key`, `openai_base_url` | LIVE | `unified_backend.py:83-85` (LM-Studio/OpenAI-compatible branch) |
| `hf_token`, `hf_model` | LIVE | `unified_backend.py:91-92` |
| `gguf_model_path` | LIVE | `unified_backend.py:96` |
| `profile` (`auto`) | **DEAD** | No reader of the *user_config* value. Profile comes from the CLI flag / `ProfileSelector` |
| `knowledge_db_built` | **DEAD** | No reader anywhere in `src/`, root scripts, or `benchmarks/` |
| `chat_max_turns` (20) | **DEAD** | No reader. `cognition/chat_history.py:ChatHistory(max_turns=20)` hard-codes the same number; `run_thoughtforge.py` constructs it with only a system prompt |
| `chat_system_prompt` | **DEAD** | No reader. The chat system prompt comes from the `--system` CLI flag |
| `chat_save_history` | **DEAD** | No reader. History saving in `run_thoughtforge.py` is driven by `--history`/the chat loop, not this flag |

**Totals for `user_config.yaml`: 15 leaf keys — 10 LIVE, 5 DEAD.**

Inverse gap: `unified_backend.py:84` reads `openai_model` (as the
fallback model for the LM-Studio branch) and
`setup_thoughtforge.py` writes it, but the **shipped
`user_config.yaml` does not contain the key at all** — a fresh
checkout's OpenAI-compatible branch silently falls back to `""`.

Credential-shaped keys (`openai_api_key`, `hf_token`) live in
this plain YAML file — both are empty in the shipped copy, so no
secret is in the tree, but the *shape* is the finding: TF-160
(Kista custody) owns the fix.

## Findings

1. **There is no single config system — there are three loaders
   that never meet.** `default.yaml` goes through the cached
   `load_config()`; `user_config.yaml` is re-parsed by hand in
   `unified_backend.py` (and again in validators/health/self-heal);
   `personality_core.yaml` is loaded by the memory store from a
   different directory with the `configs/` copy as fallback. No
   code path ever sees a merged view.
2. **`default.yaml`'s retrieval weights have drifted in name from
   the code they pretend to control.** Config says `tone_match`
   and `user_preference`; `scoring.py` calls the same slots
   `tone_similarity` and `preference_relevance`. Even wiring the
   config through naively would drop two of five weights.
   → TF-016 docket D-CFG-1; the wiring belongs to TF-032
   (cognition config surface, live keys only) and TF-044
   (scoring weights made tunable).
3. **Editing configs is a silent no-op for 32 of 64 leaves**
   (27 shadowed + 2 dead in `default.yaml`, 5 dead in
   `user_config.yaml`), plus 13 dormant personality leaves that
   parse but steer nothing. This is the audit's headline: the
   config estate *looks* tunable and mostly is not.
   → TF-016 docket D-CFG-1 / D-CFG-2 / D-CFG-3.
4. **The personality core's expressive fields are DORMANT**
   (§2). The forge's persona currently influences a turn only
   through 3 avoid phrases and a log line. → TF-016 docket
   D-CFG-2; consumers are TF-030 (persona consistency hooks)
   and TF-141 (persona compiler verdict).
5. **`user_config.yaml` mixes live backend settings with dead
   chat/profile settings in one flat file**, and the one key the
   backend factory falls back to (`openai_model`) is absent from
   the shipped file. → TF-016 docket D-CFG-3; TF-065 (backend
   contract) and TF-153 own the shape fix.
6. **What is genuinely live is small and nameable** — and
   TF-032's brief ("TF-009's live keys only") now has its exact
   list: `logging.*` (4), `cognition.personality_core_path` (1),
   and the 10 `user_config.yaml` backend keys. Everything else
   must be wired deliberately or struck from the files, slice
   by slice, in Parts II–VI — never silently left pretending.
