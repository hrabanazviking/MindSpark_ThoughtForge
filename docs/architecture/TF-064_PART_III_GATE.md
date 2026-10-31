# TF-064 — Part III Gate: Memory & Knowledge Stand Proven as One Engine (Part III Gate)

Closed 2026-10-31 on branch `yrsa/thoughtforge-road-build`.
Roadmap wording: "Part III gate: precision fixtures
improved-or-equal vs TF-043 baseline, cite-or-explain
enforced end-to-end, forgetting proven; summary filed."

## The gate verdict

**Part III is COMPLETE. All 23 build slices (TF-041–TF-063)
are banked with evidence in `docs/architecture/`, and the
three gate proofs were re-measured fresh at this gate — not
quoted from their slices:**

1. **Precision fixtures (TF-043) — GREEN, equal to baseline
   at both settings.** Fresh `evaluate_retrieval()` runs on
   a newly built fixture DB at this gate measure, at the
   defaults (sql, top_k 10, min_score 0.45): precision
   **0.8889**, recall **0.9630**, F1 **0.9111**, MRR
   **1.000**; at the open setting (min_score 0): precision
   **0.8148**, recall **1.0000**, F1 **0.8815**, MRR
   **1.000**. `baseline_problems()` returns `[]` for both
   settings — every aggregate is exactly its pinned
   TF-043 value, so the gate's "improved-or-equal" test
   passes at *equal*: no Part III slice (TF-044's named
   weights, TF-059's timestamp recency, TF-060's folded
   term extraction) moved the pinned baseline in either
   direction. The one default-setting recall loss stands
   where TF-043 found it: Q9 (GeoNames) at 0.667 — see
   gate ruling 2 below.
2. **Cite-or-explain enforced end-to-end — GREEN.**
   `citation_problems()` (TF-048, retrieval side: every
   retrieved item carries its citation payload by
   construction) and `citation_enforcement_problems()`
   (TF-049, enforcement side: per-claim check, uncited
   claims flagged by name, fabricated markers refused)
   both return `[]` live at this gate, and TF-063's saga
   re-proves the composed leg over 100 turns:
   `long_conversation_problems()` returns `[]` — every
   answer cited, every citation resolving to the record
   it claims, zero fabricated or untrue citations, with
   the fabricated/untrue/contradiction controls caught.
3. **Forgetting proven — GREEN.** `forgetting_problems()`
   (TF-051) returns `[]` live at this gate: a user-ordered
   deletion is total across home stores, tag index,
   recent queue, and episode links, and the proof re-reads
   every surface plus a raw byte scan of the whole memory
   directory before the word "forgotten" is claimed.

The targeted gate tests were also re-run as one set at
this gate: `test_tf043` + `test_tf048` + `test_tf049` +
`test_tf051` + `test_tf063` = **127 tests, 127 passed** —
exactly the five slices' own counts (23 + 24 + 33 + 25 +
22).

## Two rulings this gate was assigned, decided on the evidence

TF-044 explicitly left two scoring decisions to this gate.
Both are decided here, on measurements, not taste:

1. **The per-profile weight shifts are REJECTED.** TF-044's
   recorded experiment (semantic→importance shift of 0.05
   on `phone_low`, 0.10 on `pi_zero`) measures precision
   **0.8519** against the shipped **0.8889**, with recall
   and MRR unchanged — a cost with no purchase. No nonzero
   shift earns its place; the shipped zero-shift tables
   stand as the defaults. The shift data remains in
   `scoring_honesty.py` as a labelled failed experiment and
   a working tuning surface (`weights_for_profile` is
   opt-in at forge construction; nothing selects it by
   default), so a future corpus with more known answers
   can re-run the experiment instead of re-arguing it.
2. **GeoNames scoring is NOT retuned at this gate; its
   limitation stands as a named PARTIAL.** The arithmetic
   is exact and named (TF-044): a GeoNames record's fixed
   fts score (0.5) passes through the semantic proxy at
   0.025, giving a composite of exactly **0.285**, under
   the 0.45 default threshold by construction — so place
   records are retrievable at the open setting (Q9 recall
   1.000) and threshold-dead at the defaults (Q9 recall
   0.667). The gate does not retune a source-wide scoring
   rule on the strength of the fixture estate's *single*
   known-answer place query: that is tuning to the
   fixture, and ruling 1's experiment shows what
   speculative retunes cost on this corpus. What would
   change the verdict is a place-query corpus with many
   known answers; building one is assigned to **TF-121**
   (Part VII's harness, where benchmarks become code with
   fixtures), any retune is judged there against both
   corpora, and **TF-192** verifies at the end that this
   remainder carries its verdict. Until then the release
   truth table (TF-185) must carry GeoNames-at-defaults
   as PARTIAL — proven at open threshold, absent at
   defaults — never as silently whole. The ConceptNet
   single-term firing stands likewise, as the measured
   trade TF-043 pinned: it is Q7's only answer and
   exactly the 2/3-precision cost on Q1/Q5/Q8.

## Plain-language summary (the Volmarr-facing half of this gate)

Here is what Part III did, in plain words:

- **The memory now has a law, and the law is enforced.**
  At the start of Part III, ThoughtForge's knowledge
  store was real code with unwritten rules: facts went
  in loosely, retrieval scored by constants nobody had
  listed, answers could repeat stored facts without
  citing them, and "deleting" a memory left its traces
  behind in the indexes. All of that is now written,
  tested, and checked by machines that re-derive the
  claims from the live code.
- **Cite-or-explain is real at both ends.** Every piece
  of knowledge the engine retrieves carries its source
  marker into the prompt by construction, and after the
  model answers, each claim that matches a stored fact
  is checked: cite that fact, in that sentence, or the
  answer is flagged with the missing citations named.
  An answer citing something never retrieved — a made-up
  source — fails by name. Over a 100-turn conversation
  the citations stayed true on every single turn.
- **Forgetting is total, and it proves itself.** When
  the user orders a memory deleted, it leaves the store,
  the tag index, the recent-items list, and every link
  pointing at it — and a checker re-reads all of those
  places, plus the raw files, before anyone is told it
  is gone. A protected memory still obeys its owner's
  deletion order; only the persona's own identity record
  is refused, with the reason stated.
- **Memory survives contact with the world.** Poisoned
  or instruction-shaped "memories" ("ignore your
  instructions…") are refused at the door or quarantined
  by rule, in both directions — they cannot be planted
  through the files and retrieved later. Two people's
  memories in two stores never bleed into each other,
  even under prompts demanding the other person's
  secrets. Old facts about changeable things (schedules,
  prices) fade by a stated half-life instead of competing
  forever with fresh ones. Spanish and German facts are
  stored word-for-word and found again — even by a query
  typed without the accents.
- **The honest limits, named plainly:** precision at the
  defaults is *equal* to where Part III started, not
  improved — the part bought law, safety, and proof, not
  a higher score, and one known weakness stands open by
  this gate's own ruling: place-name (GeoNames) results
  cannot pass the default strictness threshold, so they
  only appear when the threshold is opened; fixing that
  honestly needs a bigger test corpus of place questions,
  which is now Part VII's assigned work. Everything is
  proven on synthetic fixture stores on this machine —
  there is no real lived-in memory store here to prove
  against, and the vector/embedding retrieval path still
  cannot be measured on this machine at all (the
  embedding package is absent, the campaign's one known
  environment failure). Live-model memory behaviour
  remains Part VII's road, NOT PROVEN here.

## The evidence, slice by slice

| Slice | Artefact | Headline |
|---|---|---|
| TF-041 | `KNOWLEDGE_STORE_MAP.md` | Store mapped from code; invariants listed + tested; fusion arithmetic pinned (0.49 composite) |
| TF-042 | `INGESTION_CONTRACT.md` | One validated fact schema (shape, provenance, confidence); the gate every write passes |
| TF-043 | `RETRIEVAL_PRECISION.md` | Known-answer fixture corpus; baseline pinned (gate proof 1, re-measured here) |
| TF-044 | `SCORING_HONESTY.md` | Every scoring constant named + per-profile tunable; shift experiment measured and rejected at this gate |
| TF-045 | `LIFECYCLE.md` | Facts age, decay, retire by stated rule, edges tested; D-ORPHAN-1 fixed |
| TF-046 | `MEMORY_TIERS.md` | Working / episodic / semantic tiers ruled from the code's own shape |
| TF-047 | `CONVERSATION_WRITEBACK.md` | What a chat writes back, gated by the contract + enforcement (the leg TF-019 found unbuilt) |
| TF-048 | `CITATIONS.md` | Every retrieved item carries its citation payload by construction (gate proof 2, retrieval side) |
| TF-049 | `CITATION_ENFORCEMENT.md` | Per-claim cite-or-explain check wired into the gate + core; fabricated markers fail (gate proof 2, enforcement side) |
| TF-050 | `CONTRADICTIONS.md` | New fact vs stored fact surfaced, scored, resolved by the stated rule — never silently overwritten |
| TF-051 | `FORGETTING.md` | User-ordered deletion total + provable across stores, indices, caches (gate proof 3) |
| TF-052 | `STORE_PORTABILITY.md` | A whole store exports/imports as one verified bundle (Hermes-portability applied to memory) |
| TF-053 | `EMBEDDING_DISCIPLINE.md` | Embedding models assigned per profile with offline availability proven per profile; forge degrades honestly (D-EMB-1, D-DEAD-2, D-DEAD-3 fixed) |
| TF-054 | `VECTOR_STORE_SHAPE.md` | The vector index's true shape documented; one corrupt row no longer poisons search (D-VEC-1, D-VEC-2 fixed) |
| TF-055 | `PERSONA_SEPARATION.md` | Per-persona stores never bleed; the core healed the *default* store whatever it was built on — found + fixed (D-PER-1; D-PER-2 closed by verdict) |
| TF-056 | `MEMORY_BUDGETS.md` | Store size + retrieval cost capped per profile from the shipped profile files; eviction by the lifecycle rules |
| TF-057 | `FORGE_TOOLSET.md` | forge_memory's add/inspect/prune/export completed into one coherent toolset over the store |
| TF-058 | `REFERENCE_LIBRARY_BINDING.md` | The reference library readable directly as a read-only store source; its FTS index was never rebuilt after a plain build — found + fixed (D-REF-1) |
| TF-059 | `FRESHNESS.md` | Time-aware retrieval: perishable domains decay by stated half-lives; the forge's recency now reads a record's own timestamps (D-FRESH-1 fixed for timestamped records) |
| TF-060 | `MULTILINGUAL_MEMORY.md` | Non-English facts stored verbatim and retrieved without translation loss; term extraction was English-only — found + fixed (D-ML-1) |
| TF-061 | `MEMORY_INSPECTOR.md` | What the store holds + why a retrieval fired, exportable per turn; the why was discarded at scoring time — found + fixed (D-INSP-1) |
| TF-062 | `ADVERSARIAL_MEMORY.md` | Poisoned / injection-shaped memories refused or quarantined at both boundaries (D-ADV-1 fixed) |
| TF-063 | `LONG_CONVERSATION_PROOF.md` | 100-turn saga over the store: retrieval 100/100, citations true on every turn, drift 0.0, store byte-unchanged |

## What Part III leaves open — carried forward, each with its owner

Nothing found in Part III is left unfixed *in scope* or
unowned. The gate records the honest remainders exactly:

- **GeoNames at the default threshold (PARTIAL, gate
  ruling 2):** a place record's composite is 0.285 by
  named arithmetic and can never clear 0.45. The
  multi-query place corpus that could justify a retune
  is owned by **TF-121**; the final audit of this
  remainder is **TF-192**'s; the release truth table
  (**TF-185**) must carry it as PARTIAL.
- **The knowledge DB's own recency (TF-059's stated
  remainder):** the DB tables carry no timestamp column,
  so DB-estate recency stays a per-source proxy until
  ingestion carries a date — owned by **TF-106/TF-111**
  (Part VI). The same slices own the DB estate's
  per-language labels (TF-060's remainder) and its
  adversarial screening boundary (TF-062's remainder).
- **Docket (TF-016) at the gate:** 58 entries — 40 OPEN /
  1 OPEN-ENVIRONMENT / 15 FIXED / 2 CLOSED-BY-VERDICT.
  Part III fixed thirteen entries outright (D-ORPHAN-1,
  D-EMB-1, D-EMB-2, D-DEAD-2, D-DEAD-3, D-VEC-1, D-VEC-2,
  D-PER-1, D-REF-1, D-FRESH-1, D-ML-1, D-INSP-1,
  D-ADV-1) and closed D-PER-2 by verdict; every
  remaining OPEN entry already has its later owner
  (Part IV's backend contract owns D-DEP-1, Part VI the
  source licences, Part VIII the research wing, TF-190
  the licence choice — Volmarr's word — and TF-194 the
  environment entry D-ENV-1). TF-192 verifies at the end
  that no entry is left open without a reason.
- **Proof grades stay as labelled:** fixture stores and
  fixture corpora on this VM throughout. A real lived-in
  store, vector/hybrid retrieval quality (D-ENV-1), and
  live-model citation behaviour (TF-123/TF-125) are NOT
  PROVEN and were never claimed by any Part III slice.

## Road re-cut assessment

Part III changed no Part IV–X slice's wording or order,
with one assignment *added* by this gate's ruling 2:
TF-121's harness work now also owns building the
multi-query place corpus that a GeoNames retune would
need as evidence (recorded here, per TF-018's rule that
road changes are written down with reasons). Part IV
inherits a store whose contracts (TF-042), scoring
(TF-044), budgets (TF-056), and embedding discipline
(TF-053) are frozen as data — the surface its backend
contract (TF-065) and Aesir binding (TF-074) assumed.

## Suite at the gate (measured this slice, TF-014 method)

`PYTHONPATH=src python3 -m pytest tests/ -q` on this VM:
**1884 collected / 1882 passed / 1 skipped / 1 failed
(D-ENV-1, unchanged — `test_yaml_present`, missing
sentence-transformers on this VM, an environment entry
not a code defect)** — exactly TF-063's banking line.
The gate's targeted re-run (TF-043 + TF-048 + TF-049 +
TF-051 + TF-063) is 127 / 127 passed, and the three gate
proofs above were re-measured by direct runs of
`evaluate_retrieval()` + `baseline_problems()` on a fresh
fixture DB and of `citation_problems()`,
`citation_enforcement_problems()`, `forgetting_problems()`,
and `long_conversation_problems()` at this gate, all
returning the banked values exactly. This slice adds one
gate doc and no code or tests, so the count does not move
at banking.

## What opens next

Part IV — TurboQuant & the Inference Layer (TF-065–
TF-086), starting with TF-065: the backend contract v1 —
what a backend must provide — with the existing ollama /
lmstudio / hf backends conformed and tested against it,
and D-DEP-1 (the undeclared `requests` dependency) owned
in its scope.
