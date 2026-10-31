# MindSpark ThoughtForge — The Up-to-Date Massive Roadmap

*Commissioned by Volmarr Wyrd, 2026-10-30: "make a massive up to date
roadmap for this project and then proceed with working on it."*

Repo: `hrabanazviking/MindSpark_ThoughtForge` (default branch
`development`). This roadmap is the law of the build that follows it.

## 1. What ThoughtForge is (the vision, in Volmarr's own terms)

From BUILD_PLAN_v1, confirmed by Volmarr 2026-03-31 and still the
soul of the project: **"a universal cognitive enhancement layer for
AI models of any size… It is not a model. It is the forge that makes
any model sharper."** The repo's own description says it plainly: a
rune-forged conversation engine built for tiny GPT-Nothing-class
minds — through guided memory, lean cognition, and relentless
refinement, it gives small local models **depth, presence, and will**
on edge and low-power hardware.

The five pillars (named in the plan, visible in the tree):

1. **Sovereign local RAG** — knowledge lives on the user's machine,
   retrieved locally, cited honestly.
2. **TurboQuant inference** — aggressive quantisation discipline so
   small hardware carries real models.
3. **Cognition scaffolds** — structured thinking wrapped around a
   small model's raw generation.
4. **Fragment salvage** — broken, partial, rambling model output is
   refined into coherence instead of discarded.
5. **Memory-enforced cognition** — *cite-or-explain*: claims drawn
   from memory must cite it; the rest must own themselves as the
   model's own reasoning.

Hardware truth (the plan's profiles, the road's budgets): `phone_low`
(2 GB) up through `server_gpu` (24 GB+). Every slice respects the
profile it targets; a feature that only works on the big profile is
labelled so, never smuggled in as universal.

## 2. The true state (recon 2026-10-30, GitHub API — to be re-proven in-tree by Part I)

The plan is **stale against the tree**: BUILD_PLAN_v1 marks Phase 0
"not yet started" while real code far beyond it exists. What stands
today:

- `src/thoughtforge/cognition/` — core.py (29 KB), router, scaffold,
  prompt_builder, chat_history: the thinking layer is real code.
- `src/thoughtforge/inference/` — backends, turboquant.py (17.8 KB),
  ollama / lmstudio / hf backends, hardware profiles, onnx_export.
- `src/thoughtforge/knowledge/` — forge, lifecycle, models, scoring,
  store: the memory/knowledge engine.
- `src/thoughtforge/etl/` — wikidata, sources, schema, embeddings,
  subset: the ingestion pipeline.
- `src/thoughtforge/refinement/` — salvage (14.8 KB), enforcement:
  pillars 4 and 5 as code.
- Root instruments: `run_thoughtforge.py`, `forge_memory.py`,
  `forge_doctor.py`; `benchmarks/` incl. **persona_consistency**;
  configs (default / personality_core / user).
- `research_data/src/wyrdforge/` — a research prototype *inside* the
  repo: bond graph, micro-RAG, persona compiler, truth calibrator,
  with its own tests. A second, younger mind growing in the same
  house — Part VIII graduates or retires it deliberately.
- `data/` — large Norse datasets + a knowledge_reference library.
  **This is Volmarr's canon: private, never published, never pushed
  anywhere new.** The clone for this campaign is sparse/blob-filtered
  to keep data weight out of the working tree where possible.
- TASK_PHASE0–8 docs exist beside the stale plan — the plan-vs-tree
  reconciliation is Part I's whole job.

## 3. Laws of this campaign

- **Volmarr's laws, all of them:** keeper discipline (the 10-minute
  timer CONTINUES stopped work), never leave a discovered bug
  unfixed, no Docker (the repo's Dockerfile is never used), never
  touch any README.md, plain language first in reports.
- **Counts come from the runner only.** Baselines are reproduced,
  not quoted from docs.
- **Canon data is private.** Datasets, persona material, and the
  knowledge_reference library never leave the machine or enter any
  public artefact; test fixtures use synthetic stand-ins.
- **Small-first honesty:** every capability is labelled with the
  smallest hardware profile it is *proven* on (proven / partial /
  NOT PROVEN per profile).
- **The forge serves the ecosystem:** Aesir is a first-class backend
  target (Part IX); Rúnmál and Samtal are the named consumers; WYRD
  is the world-model neighbour, not a rival (Part VIII's boundary).
- Banking is local per slice; pushes await Volmarr's word (the sole
  exception: this roadmap's own filing).

## 4. The roadmap — 196 slices in 10 parts

Numbering: **TF-###**. Citation discipline: BUILD_PLAN phases are
cited as BP-Ph#; elder task docs as TASK_PHASE#; never bare numbers.

---

## Part I — True State: the Audit the "Up-to-Date" Demands (TF-001 – TF-018)

*Goal: replace the stale plan's claims with the tree's proven truth —
every module, test, and phase claim re-measured.*

- **TF-001** Campaign opening: sparse clone, branch
  `yrsa/thoughtforge-road-build`, roadmap filed under `docs/` +
  pushed (filing only), state file created.
- **TF-002** Module census: every package under `src/thoughtforge/`
  inventoried — purpose, size, public surface, test coverage presence.
- **TF-003** The plan-vs-tree matrix: BUILD_PLAN_v1 phases 0–8 and
  TASK_PHASE0–8 claims × tree reality (DONE-in-fact / PARTIAL /
  ABSENT / SUPERSEDED), each cell with file evidence.
- **TF-004** Fresh baseline I: the test suite run whole on this tree;
  counts recorded from the runner; failures classified
  (real / environment / stale-test).
- **TF-005** Fresh baseline II: the entry points actually run —
  `run_thoughtforge.py` boots, `forge_doctor.py` reports,
  `forge_memory.py` operates on a synthetic store.
- **TF-006** The research wing census: `research_data/src/wyrdforge/`
  inventoried against the shipping tree (overlap, uniqueness,
  test state) — evidence for Part VIII's verdicts.
- **TF-007** Data census (metadata only): dataset inventory — names,
  sizes, provenance notes — without copying canon anywhere.
- **TF-008** Dependency & licence audit: every dependency's licence
  recorded; contamination flagged with the fix named.
- **TF-009** Config census: default / personality_core / user configs
  mapped to the code paths that read them (dead keys named).
- **TF-010** The benchmark estate: `benchmarks/` inventoried —
  which run today, which are scripts vs suites, which are stale.
- **TF-011** Dead & duplicate code census (kin to TF-003): modules
  nothing imports, functions defined twice, TODO archaeology.
- **TF-012** Hardware-profile truth table v1: each profile's claimed
  budget vs the code's actual assumptions.
- **TF-013** The doctor's report: forge_doctor extended (or its gaps
  listed) to emit the audit's headline numbers by itself.
- **TF-014** Baseline freeze: the suite + entry-point results pinned
  as the campaign's zero line; the method recorded so any keeper
  re-measures identically.
- **TF-015** Audit verdicts on the elder docs: BUILD_PLAN_v1 marked
  (in the audit doc, never by editing the plan) as SUPERSEDED by this
  roadmap, with the concordance that preserves its vision (§1).
- **TF-016** The bug & gap docket: every defect found in Part I,
  numbered, each assigned to the part that fixes it (nothing found,
  nothing unfixed — the docket is the proof).
- **TF-017** Fixture policy: synthetic canon stand-ins created for
  tests so no test ever needs real data.
- **TF-018** Part I gate: audit doc published in-repo; Volmarr-facing
  summary in plain language; the road below re-cut if the facts
  demanded it (changes recorded here, with reasons).

## Part II — The Cognition Core (TF-019 – TF-040)

*Goal: core.py and its kin become a documented, tested thinking
engine — the scaffold a small model thinks inside.*

- **TF-019** Core loop mapped: the turn lifecycle (input → routing →
  scaffold → generation → refinement → memory) drawn from the code,
  published as the architecture doc.
- **TF-020** Router hardening: the cognition router's decision table
  made explicit + tests for every row.
- **TF-021** Scaffold library I: the existing scaffolds catalogued
  (shape, purpose, cost in tokens) in one registry.
- **TF-022** Scaffold library II: chain-of-thought, plan-then-answer,
  and self-check scaffolds brought to one interface.
- **TF-023** Prompt builder contract: prompt_builder's assembly order
  specified and golden-tested (persona, memory, scaffold, context).
- **TF-024** Chat history discipline: windowing, summarisation hooks,
  and the token budget enforced + tested.
- **TF-025** Turn state: a typed turn-state object replaces ad-hoc
  dicts where the census found them (TF-011's list).
- **TF-026** Streaming turns: cognition works on streamed generation
  (scaffold applied pre/post stream honestly).
- **TF-027** Interruption & resume: a turn can be cut and the state
  remains coherent (the Rúnmál voice lesson, applied to text).
- **TF-028** Multi-turn coherence fixtures: 20-turn conversations on
  a small model fixture, scored for drift (with Part VII's harness).
- **TF-029** The will layer: goal/intent persistence across turns —
  the "presence and will" promise — specified from the code's hints
  + the plan, implemented minimally, tested.
- **TF-030** Persona consistency hooks: the cognition layer's half of
  the persona_consistency benchmark (its benchmark half is TF-128).
- **TF-031** Error honesty: when the model fails mid-turn, the user
  sees a truthful degraded answer or a stated failure — never a
  silently fabricated completion.
- **TF-032** Cognition config surface: scaffolds/profiles selectable
  per conversation via config (TF-009's live keys only).
- **TF-033** Latency budgets per profile for the core loop, measured
  + enforced in tests on the VM profile.
- **TF-034** Core API freeze v1: the cognition layer's public
  interface versioned + documented (consumers in Part IX bind here).
- **TF-035** Concurrency: two conversations at once without state
  bleed (fixtures prove isolation).
- **TF-036** Determinism mode: seeded runs reproducible for tests
  and benchmarks.
- **TF-037** The small-model floor: the core proven against the
  smallest fixture model on phone_low budgets (simulated profile).
- **TF-038** Logging & traces: a turn's full cognition trace
  exportable (the Inspector instinct; Rúnmál P-207's cousin).
- **TF-039** Core docs: the architecture doc (TF-019) reconciled
  with shipped behaviour; examples runnable.
- **TF-040** Part II gate: the drift fixtures (TF-028) + floor test
  (TF-037) + API freeze all green; plain-language summary filed.

## Part III — Memory & Knowledge: the Sovereign Mind (TF-041 – TF-064)

*Goal: pillars 1 and 5 as one engine — knowledge stored locally,
retrieved precisely, and enforced in cognition: cite-or-explain.*

- **TF-041** Knowledge store mapped: models/lifecycle/scoring/store
  documented from code; the store's invariants listed + tested.
- **TF-042** Ingestion contract: what a "fact" is in ThoughtForge
  (shape, provenance, confidence) — one schema, validated.
- **TF-043** Retrieval precision fixtures: a synthetic corpus with
  known right answers; recall/precision measured, baseline pinned.
- **TF-044** Scoring honesty: the scoring module's weights made
  explicit + tunable per profile; no magic constants unlisted.
- **TF-045** Lifecycle: facts age, decay, and are retired by rule —
  the lifecycle module's rules tested at their edges.
- **TF-046** Memory tiers: working / episodic / semantic separation
  where the code hints it; unified where it doesn't (verdict-driven).
- **TF-047** Conversation → memory: what a chat writes back to the
  store, gated by the enforcement rules (Part V).
- **TF-048** Cite-or-explain I (retrieval side): every retrieved item
  carries its citation payload into the prompt by construction.
- **TF-049** Cite-or-explain II (enforcement side): post-generation
  check — claims matching stored facts must cite or the answer is
  flagged/regenerated per policy (with TF-093).
- **TF-050** Contradiction handling: new fact vs stored fact — the
  conflict is surfaced, scored, and resolved by the stated rule,
  never silently overwritten.
- **TF-051** Forgetting: user-ordered deletion is total and provable
  (store, indices, caches) — the right to be forgotten, locally.
- **TF-052** Store portability: export/import of a whole store as a
  documented bundle (Hermes-portability law applied to memory).
- **TF-053** Embedding discipline: the embeddings module's models
  per profile, with offline availability proven per profile.
- **TF-054** Vector store shape: the store's index survives restarts
  + corruption fixtures (atomic writes, the Samtal lesson).
- **TF-055** Persona memory separation: per-persona stores never
  bleed (fixtures with two personae, adversarial prompts).
- **TF-056** Memory budgets: store size + retrieval cost capped per
  hardware profile; eviction by the lifecycle rules.
- **TF-057** Knowledge forge UI-shape: forge_memory's operations
  (add/inspect/prune/export) completed to a coherent toolset.
- **TF-058** Reference-library binding: the knowledge_reference
  library readable as a store source (read-only, canon-respecting).
- **TF-059** Freshness: time-aware retrieval (recent facts weighted
  per the stated rule) for time-sensitive domains.
- **TF-060** Multilingual memory: non-English facts stored/retrieved
  without translation loss (fixtures in two languages).
- **TF-061** Memory inspector data: what the store holds + why a
  retrieval fired, exportable per turn (consumes TF-038).
- **TF-062** Adversarial memory fixtures: poisoned facts, prompt-
  injection-shaped memories — refused/quarantined by rule.
- **TF-063** The long-conversation proof: 100-turn synthetic saga;
  memory stays coherent, citations stay true (Part VII harness).
- **TF-064** Part III gate: precision fixtures improved-or-equal vs
  TF-043 baseline, cite-or-explain enforced end-to-end, forgetting
  proven; summary filed.

## Part IV — TurboQuant & the Inference Layer (TF-065 – TF-086)

*Goal: pillar 2 made rigorous — quantisation and backends measured
per profile, with Aesir joining as a first-class citizen.*

- **TF-065** Backend contract v1: what a backend must provide
  (generate, stream, tokenise, capability truth) — the existing
  ollama/lmstudio/hf backends conformed + tested against it.
- **TF-066** TurboQuant mapped: turboquant.py's method documented
  from code (what it quantises, where, with what loss claims).
- **TF-067** Quant quality fixtures: the same prompts across quant
  levels; quality scored by Part VII's harness; the loss curve
  published from measurements.
- **TF-068** Quant selection policy: per profile, the default quant
  chosen *from TF-067's curve*, recorded as an ADR-shaped note.
- **TF-069** Profile runner: each hardware profile boots its stack
  in simulation (budgets enforced: RAM ceilings as hard test limits).
- **TF-070** ONNX export path: onnx_export's truth — what exports,
  what doesn't, fixtures for the exportable set.
- **TF-071** Tokeniser discipline: per-backend token counting agreed
  within tolerance (the Aesir tokenizer lessons applied).
- **TF-072** Streaming backpressure: slow consumers don't corrupt
  generation; cancellation clean on every backend.
- **TF-073** Backend health & fallback: a dead backend degrades to
  the next per the stated policy, announced honestly.
- **TF-074** **The Aesir backend**: Project Aesir bound as a backend
  under TF-065's contract (its capability ledger respected — what
  Aesir proves, ThoughtForge may claim; no more).
- **TF-075** Aesir + TurboQuant interplay: quantised models served
  by Aesir measured through ThoughtForge's fixtures.
- **TF-076** OpenAI-compatible backend: any server speaking the
  common shape binds under the same contract.
- **TF-077** Local-first routing: profile-aware backend choice
  (privacy + cost + capability), the router's table published.
- **TF-078** Model cards in-repo: per supported model, the measured
  truth (size, quant, profiles proven, scaffold behaviour).
- **TF-079** Cold-start budgets: time-to-first-token per profile
  measured + capped in tests where the VM allows.
- **TF-080** Memory-pressure behaviour: at the profile's ceiling the
  engine sheds load by rule (context first), never crashes mute.
- **TF-081** Batch & concurrency per profile, measured honestly.
- **TF-082** The Pi profile: the whole stack on Mimir-class hardware
  (phone_low/pí budgets) — proven on the real Pi or NOT PROVEN,
  stated plainly.
- **TF-083** GPU-optional proof: no profile secretly requires a GPU;
  CPU paths measured on the laptop line.
- **TF-084** Inference traces join cognition traces (TF-038): one
  exportable story per turn.
- **TF-085** Backend docs + a new-backend guide (the contract as a
  recipe).
- **TF-086** Part IV gate: quant curve published, Aesir backend
  green under contract tests, per-profile truth table v2 issued.

## Part V — Refinement: Salvage & Enforcement (TF-087 – TF-104)

*Goal: pillars 4 and 5 finished — broken output rescued by rule, and
memory's law enforced on every answer.*

- **TF-087** Salvage mapped: salvage.py's strategies catalogued
  (truncation repair, rambling compression, format rescue, …) each
  with fixtures of broken input.
- **TF-088** Salvage safety: rescue never *invents* content — the
  line between repair and fabrication specified + tested with
  adversarial broken outputs.
- **TF-089** Salvage metrics: rescue rate + corruption rate measured
  on the fixture corpus; policy thresholds from measurements.
- **TF-090** Format enforcement: structured outputs (JSON-shaped
  asks) repaired or refused honestly, per policy.
- **TF-091** Repetition & loop rescue: degenerate loops detected
  and cut by rule (small models' classic failure).
- **TF-092** Language drift rescue: answers wandering off-language
  pulled back or flagged (fixtures bilingual).
- **TF-093** Enforcement mapped + completed: enforcement module's
  checks unified with TF-049's cite-or-explain into one policy
  surface.
- **TF-094** Policy configuration: enforcement strictness per
  conversation (strict / standard / relaxed) with the defaults
  stated + tested.
- **TF-095** Refusal honesty: when enforcement blocks an answer, the
  user learns *that and why*, with the salvage path offered.
- **TF-096** Persona-voice preservation under salvage: refinement
  must not sand a character's voice flat (fixtures from TF-030).
- **TF-097** Provenance of rescue: a salvaged answer carries its
  rescue record into the trace (TF-084).
- **TF-098** Cost discipline: refinement passes capped per profile
  (a second small-model call is a budget, not a habit).
- **TF-099** Streaming salvage: what can be rescued mid-stream vs
  post-hoc, specified honestly per strategy.
- **TF-100** The chaos corpus: a published fixture set of real-shaped
  small-model failures (truncations, loops, hallucinations,
  format collapses) — the part's proving ground.
- **TF-101** Enforcement + chaos: the full policy stack run over
  TF-100; outcomes scored (rescued / refused / passed-clean).
- **TF-102** User-visible quality marks: where the product surfaces
  answers, a rescued/cited answer can *show* its marks (data ready
  for consumers; UI is the consumer's).
- **TF-103** Regression armour: every historical bad output named in
  the repo's docs becomes a fixture before its fix is trusted.
- **TF-104** Part V gate: chaos-corpus score published; zero
  fabrication rescues in the adversarial set; summary filed.

## Part VI — The Ingestion Forge (ETL) (TF-105 – TF-120)

*Goal: how knowledge gets in — wikidata, sources, subsets — made
safe, provenanced, and reproducible.*

- **TF-105** ETL mapped: sources → schema → embeddings → subset →
  store, drawn from code; the pipeline's contracts per stage.
- **TF-106** Source registry: every source carries provenance +
  licence (TF-008's audit consumed); unlicensed-in-fact sources
  quarantined by rule.
- **TF-107** Wikidata pipeline hardened: rate-respecting, resumable,
  checkpointed (long ingestions survive interruption).
- **TF-108** Schema validation at the gate: malformed records refused
  with reasons, counted, never silently dropped.
- **TF-109** Dedup & merge rules for overlapping sources (consumes
  TF-050's contradiction rule).
- **TF-110** Subset craft: profile-sized subsets (phone_low's
  knowledge must fit phone_low) built by rule + measured.
- **TF-111** Embedding at ingestion: model per profile (TF-053),
  resumable, checksummed.
- **TF-112** Ingestion dry-run: a source can be test-ingested into a
  scratch store with a full report before touching a real store.
- **TF-113** Canon ingestion path: Volmarr's own materials ingested
  by the same pipeline (private stores only; fixtures synthetic).
- **TF-114** Update ingestion: re-running a source updates facts by
  the lifecycle rules instead of duplicating them.
- **TF-115** Ingestion metrics: records/sec, refusal rate, store
  growth per source — reported per run.
- **TF-116** Poisoned-source fixtures: adversarial source data
  refused/quarantined (kin to TF-062).
- **TF-117** Offline ingestion: everything except the fetch itself
  works with the network cut (sovereignty, proven).
- **TF-118** The reference library as a source (TF-058's write side):
  its ingestion reproducible + checksummed.
- **TF-119** ETL docs + a new-source recipe.
- **TF-120** Part VI gate: a full source → store → retrieval round
  trip on fixtures, with provenance intact at every hop.

## Part VII — Benchmarks & the Proof Harness (TF-121 – TF-138)

*Goal: ThoughtForge's claims become numbers — a harness that scores
depth, presence, and will instead of asserting them.*

- **TF-121** Harness architecture: benchmarks as code with fixtures,
  scorers, and pinned baselines (the `benchmarks/` estate, TF-010,
  unified).
- **TF-122** Persona consistency suite: the existing
  persona_consistency benchmark completed into a scored suite
  (identity facts, voice, values under pressure).
- **TF-123** Memory fidelity suite: cite-or-explain scored
  mechanically (citations present, citations *true*).
- **TF-124** Depth suite: multi-step reasoning with scaffolds on/off
  — the scaffold value measured, not assumed.
- **TF-125** Presence suite: long-conversation coherence + the will
  layer's follow-through (TF-029) scored over scripted sagas.
- **TF-126** Small-model honesty panel: the same suites across model
  sizes — where the forge helps most, published from results.
- **TF-127** Quant-vs-quality panel (consumes TF-067 mechanically).
- **TF-128** Profile panels: suites per hardware profile with the
  profile's budgets enforced during the run.
- **TF-129** Regression gates: benchmark deltas beyond tolerance
  fail the campaign's slice gate (thresholds from baseline noise,
  measured).
- **TF-130** Human-shaped spot checks: a small set of transcripts
  Volmarr can read and judge himself (the numbers' sanity anchor).
- **TF-131** Benchmark data hygiene: all fixtures synthetic or
  licensed; canon never a benchmark input.
- **TF-132** The WYRD-shaped panel: world-state consistency questions
  (feeds Part VIII's boundary with evidence).
- **TF-133** Cost panel: tokens + time per suite per profile —
  sharpness per watt, the edge-hardware point of the project.
- **TF-134** Baseline book: every suite's pinned baseline + method,
  in-repo, regenerated by command.
- **TF-135** Flake discipline: nondeterministic suites quarantined
  or made deterministic (TF-036) — no silent re-rolls.
- **TF-136** Public-claims audit: every performance claim in the
  repo's docs matched to a suite result or struck in the audit doc.
- **TF-137** The harness as a tool: one command runs the panel a
  slice needs; keepers cite its output verbatim.
- **TF-138** Part VII gate: the full panel green on the campaign
  tree; the baseline book published; summary filed.

## Part VIII — The Wyrdforge Wing: Graduate or Retire (TF-139 – TF-152)

*Goal: the research prototype inside the repo gets its verdicts —
each of its four inventions adopted properly, or honourably retired
with its lessons harvested.*

- **TF-139** Bond graph verdict: wyrdforge's bond graph vs the
  shipping knowledge store — adopt (as a layer), merge, or retire,
  decided on TF-006's census + a working spike.
- **TF-140** Micro-RAG verdict: its retrieval vs Part III's engine
  on TF-043's fixtures — the better mechanism wins by measurement.
- **TF-141** Persona compiler verdict: its persona compilation vs
  the shipping config/persona path (TF-009/TF-030) — adopt the
  compiler's shape where it proves better.
- **TF-142** Truth calibrator verdict: its calibration vs Part V's
  enforcement — measured on TF-100's chaos corpus.
- **TF-143** Adoptions implemented: each ADOPT verdict lands as
  production code under the shipping tree's discipline + tests.
- **TF-144** Retirements executed: each RETIRE verdict's code
  removed or archived-in-place (per Volmarr's repo habits), lessons
  written into the wing's closing doc.
- **TF-145** WYRD boundary doc: ThoughtForge vs the WYRD Protocol —
  what each owns (cognition vs world-model), the interface between
  them, written for the Second Life/Hermes consumers.
- **TF-146** The bond-graph bridge (if adopted at TF-139): bonds as
  WYRD-consumable data.
- **TF-147** Persona compiler → Rúnmál cards (if adopted): compiled
  personae exportable in Rúnmál's card shape.
- **TF-148** Research hygiene: `research_data/` either graduates,
  archives, or is clearly fenced as research with its own test
  discipline — no ambiguous middle state remains.
- **TF-149** The wing's tests: adopted code carries shipping-grade
  tests; retired code's tests archived with it.
- **TF-150** Provenance preserved: the wing's authorship/history
  recorded in the closing doc (ORIGINS-shaped honesty).
- **TF-151** Panel re-run: TF-132 + persona suites re-measured with
  the verdicts landed; deltas published.
- **TF-152** Part VIII gate: no research code left in limbo; every
  verdict executed; the boundary doc shipped.

## Part IX — Bindings: the Ecosystem Drinks from the Forge (TF-153 – TF-170)

*Goal: ThoughtForge as a service to Volmarr's other works — Rúnmál,
Samtal, Hermes, Ember — through small, documented, sovereign APIs.*

- **TF-153** Service shape: ThoughtForge callable as a library AND
  as a local service (loopback only by default) — one contract.
- **TF-154** The cognition API: turn-in/turn-out with traces
  (TF-084), versioned (TF-034 consumed).
- **TF-155** The memory API: store operations (TF-057's toolset) as
  service calls with the same enforcement.
- **TF-156** **Rúnmál binding**: Rúnmál's cognition/memory calls can
  route through ThoughtForge (behind its backend table; off by
  default; the parity campaign's P-slices consume the doc).
- **TF-157** **Samtal binding**: a Samtal agent seat's thinking
  layer via ThoughtForge, under Samtal's own rules (equal seats,
  Astrid Law untouched — the binding is cognition only).
- **TF-158** **Hermes binding**: the volmarr-core shaped contract —
  ThoughtForge as a cognition provider for the Hermes hack
  (consumes the 2026-10-30 refresh's plugin surfaces).
- **TF-159** **Ember binding**: the tiny profile as Ember's cognition
  layer candidate — budgets proven on phone_low first.
- **TF-160** Kista custody: any credential a binding needs lives in
  Kista-shaped custody, never in configs.
- **TF-161** Binding fixtures: each consumer binding has a contract
  test with a fake consumer (consumers needn't be present to prove
  the shape).
- **TF-162** Versioning & refusal: a consumer on an old contract
  gets a truthful refusal, never silent corruption.
- **TF-163** Local-only proof: bindings bind loopback by default;
  any wider exposure is an explicit, logged config act.
- **TF-164** Streaming across bindings (TF-026/TF-072 composed).
- **TF-165** Persona portability: a persona defined once serves
  every binding (compiler output if TF-141 adopted).
- **TF-166** Store sharing rules: two bindings, one store — the
  concurrency + separation rules (TF-035/TF-055) at service scale.
- **TF-167** The doctor, service edition: forge_doctor reports
  binding health per consumer.
- **TF-168** Docs: a binding recipe per consumer, with the honesty
  labels (proven with fake consumer / proven live) per binding.
- **TF-169** Load truth: the service under two consumers at once on
  the laptop profile, measured.
- **TF-170** Part IX gate: four bindings contract-green; at least
  one proven live end-to-end (the first available consumer); the
  rest labelled with their proof grade, honestly.

## Part X — Hardening, Docs & the Release Truth (TF-171 – TF-196)

*Goal: the forge, finished honestly — secured, documented, measured
on real small hardware, and released with its truth table attached.*

- **TF-171** Security audit of the whole surface (service, ETL,
  store, bindings) — findings fixed or roaded with owners.
- **TF-172** Privacy audit: what leaves the machine in every mode —
  nothing by default, proven at socket level where runnable.
- **TF-173** The real-hardware lab I: phone_low proven on an actual
  small device available to the house, or NOT PROVEN stated.
- **TF-174** The real-hardware lab II: the Pi profile re-proven on
  Mimir (TF-082's verdict revisited with the finished engine).
- **TF-175** The laptop line: full panel (Part VII) on
  purplelongship-class hardware, published.
- **TF-176** Endurance: a 24-hour soak of the service shape on the
  laptop profile (memory growth, store integrity) — or the longest
  honest soak the campaign can run, labelled.
- **TF-177** Upgrade path: stores/configs from the pre-campaign tree
  migrate to the campaign's shapes, fixture-proven.
- **TF-178** The plan's ghost laid: BUILD_PLAN_v1's successor doc —
  this roadmap's completion state — written into `docs/` (the plan
  file itself untouched).
- **TF-179** User docs I: getting sharp on your own machine — install
  (native, no Docker), first model, first persona, first store.
- **TF-180** User docs II: profiles & budgets — choosing your
  hardware truth.
- **TF-181** User docs III: memory law for users — cite-or-explain,
  forgetting, export — in plain language.
- **TF-182** Developer docs: architecture (TF-019/TF-145) reconciled
  with the shipped tree; the backend + binding recipes.
- **TF-183** README care: extended **by hand** with the campaign's
  truth; existing text preserved (standing law).
- **TF-184** The claims audit, final: every public claim matched to
  evidence (TF-136 completed for the release).
- **TF-185** Release shaping: version, changelog truthful by part
  (proven / partial / NOT PROVEN per part + profile).
- **TF-186** The small-model showcase: three tiny models, before/
  after the forge, transcripts published (TF-130's set, final).
- **TF-187** Regression armour review: the full fixture estate
  (chaos corpus, drift, precision, adversarial) runs as one gate.
- **TF-188** Store corruption drill: kill -9 mid-write, power-loss
  shape — the store opens clean or says why.
- **TF-189** Accessibility of outputs: traces/docs readable by
  screen reader (alt text law where images appear).
- **TF-190** Licence & attribution final pass (TF-008 closed out).
- **TF-191** The retrospective: lessons to the workshop ledger;
  patterns named as skills where they earned it.
- **TF-192** Roadmap audit: every TF-slice DONE with evidence,
  PARTIAL with its road, or NOT PROVEN with its reason.
- **TF-193** Handover pack for Hermes: campaign state + contracts in
  the portable shapes (the standing Hermes-portability rule).
- **TF-194** Final baseline book (TF-134 regenerated at the release
  tree).
- **TF-195** Push review: the campaign's banked slices presented for
  Volmarr's push word as one verified stack (heads, counts, truth
  table attached).
- **TF-196** Final report to Volmarr: plain language first — what
  the forge now provably does for small models, per profile, and
  what remains NOT PROVEN — evidence index attached.

---

## 5. Order and dependencies

Part I before everything (it may re-cut the road; changes are
recorded in TF-018). Part II next (everything thinks through the
core). III and V grow together (enforcement needs the store; the
store needs enforcement). IV can run beside III. VI after III's
contracts. VII's harness starts early (TF-121–123 feed II–V) and
completes when they do. VIII after III/V measurements exist. IX
after II/III/IV contracts freeze. X closes. The Pi and device legs
share machines with other campaigns — load etiquette per house law;
heavy panels belong on the laptop line.

## 6. What this roadmap deliberately does not do

- No training or fine-tuning of models — the forge sharpens minds
  at inference time; that is the project's whole idea.
- No cloud requirement; remote backends are guests, never landlords.
- No canon data in fixtures, benchmarks, or any artefact that could
  be published.
- No Docker, no README overwrites, no silent claims: every number
  traces to a runner, a fixture, or a named device.

## 7. First steps when the word is given

1. File this roadmap into the repo under `docs/`, extending — never
   overwriting — what stands there (filing push only).
2. Open the campaign: state file + the 10-minute keeper (house law).
3. Part I's audit runs before any building: the plan is stale, the
   tree is the truth, and the audit is how we learn it.

*Carved 2026-10-30, at Volmarr's word, by Yrsa.*
