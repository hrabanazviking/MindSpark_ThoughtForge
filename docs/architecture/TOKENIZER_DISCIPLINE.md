# TF-071 — Tokenizer Discipline

**The plain truth first: before this slice, no two
token counters in the tree had ever been checked
against each other, and "token" named three different
things.** TF-065 gave every backend a `count_tokens`
with a declared basis, but agreement was explicitly
NOT PROVEN and no tolerance existed. Underneath, the
tree counted in contract estimates (chars / 4), in
cognition word counts (`len(text.split())`), and in
server-reported model counts — and one backend (HF)
reported a word count as its `tokens_generated`. This
slice files the discipline as data
(`src/thoughtforge/inference/tokenizer_discipline.py`,
format `thoughtforge-tokenizer-discipline/1`), measures
every surface against a stated corpus, fixes the three
defects the measurements exposed (D-TOK-1/2/3), and
states the tolerance future backends (TF-074's Aesir
backend first) will be judged by.

## The Aesir lessons, applied

Project Aesir's Part II retrospective ruled that most
apparent divergences are **comparator artifacts** —
template rendering, BOS rendering, backend mismatch,
prompt-boundary step-0 effects — in neither engine's
maths. Token counting has the same traps, and the
discipline is built from them:

- **Special tokens are a basis, not an error.** A
  llama.cpp `tokenize` call includes BOS by default: a
  loaded engine counts `""` as 1 and every text as +1
  against a special-free counter. Neither side is
  wrong; the count's basis must be *stated*, and the
  tolerance's absolute floor absorbs the +1.
- **Short texts are all boundary.** `"Yes."` counts 1
  (estimate), 2 (real BPE), 3 (BOS-carrying stub) — all
  honest. Purely relative agreement is meaningless at
  small counts, so the rule carries an absolute floor.
- **Equality across tokenisers is never the demand.**
  Different vocabularies split differently. The demand
  is a stated tolerance on a stated corpus,
  re-derivable by the checker.
- **Script is part of the basis.** A chars-per-token
  density measured on Latin prose does not transfer to
  scripts that tokenise per character (see D-TOK-2).

## The agreement rule

Two counters **agree** on a text iff
`|a − b| ≤ max(2, 0.35 × max(a, b))`
(`ABS_TOKENS = 2`, `REL_TOLERANCE = 0.35` in
`tokenizer_discipline`). The constants are derived,
not chosen: against a real BPE tokeniser (tiktoken
cl100k_base, measured on this VM — an environment
tool, not a shipped dependency; the counts are filed
as `MEASURED_BPE` and re-derived live by the checker
when tiktoken is importable), the contract estimate's
Latin relative error peaked at **0.25** (long prose)
and post-fix CJK at **0.31** — a 0.35 band covers every
measured honest pair, and still catches the pre-fix
CJK breach at **0.88**. The floor of 2 absorbs BOS
(+1) plus a boundary token.

## The corpus, measured

Nine synthetic rows (Lantern Hollow / neutral — never
canon), every counter measured live by the checker:

| row | estimate | word | BPE (filed) | stub model (+BOS) |
|---|---|---|---|---|
| prose | 18 | 14 | 15 | 21 |
| long_prose | 147 | 102 | 118 | 166 |
| numbers | 21 | 15 | 27 | 28 |
| code | 33 | 11 | 38 | 46 |
| punct | 17 | 11 | 20 | 25 |
| cjk | 11 | 1 | 16 | 12 |
| mixed | 15 | 11 | 18 | 18 |
| short | 1 | 1 | 2 | 3 |
| empty | 0 | 0 | 0 | 1 |

- The **estimate agrees with BPE on all nine rows**
  (post-fix) and with the BOS-carrying stub model on
  all nine.
- All **four shipped backends count identically** on
  every row today (all estimate-based; TurboQuant
  unloaded) — the per-backend agreement the roadmap
  asks for, with the tolerance as the standing rule
  for when a model tokeniser joins.
- The **word count breaches** the BPE tolerance on
  five rows (numbers, code, punct, cjk, mixed) — the
  pinned proof that the cognition word convention is a
  different discipline (below).

## The counting surfaces (filed in `COUNTING_SURFACES`)

| surface | basis |
|---|---|
| `backend_contract.estimate_tokens` | estimate — the one home; ~4 chars/token, script-aware since TF-071 |
| `UnifiedBackend.count_tokens` (default) | estimate — inherited by ollama / lmstudio / hf, which therefore count identically |
| `TurboQuantBackend.count_tokens` | model while the engine is loaded, estimate until then; `""` guarded to 0 at the surface |
| `TurboQuantEngine.estimate_tokens` | model while loaded (llama tokeniser, BOS included — a loaded engine counts `""` as 1, stated on the method), else the contract estimate **by delegation** |
| ollama / lmstudio / turboquant `tokens_generated` | server-model — the server's / engine's own completion count |
| hf `tokens_generated` | estimate — the backend's own `count_tokens` basis since TF-071 (D-TOK-3) |
| cognition `history_discipline.estimate_tokens` | **word — a budget count, NOT a token count** (TF-021/TF-024 convention: history budgets, prompt costs, final `token_count`), kept deliberately and never substituted for a backend count |

The two functions named `estimate_tokens` (inference
contract vs cognition history) are filed as a standing
hazard: same name, different disciplines, guarded by
the `word_counts_are_not_tokens` rule and a checker
pin that they remain distinct objects with distinct
behaviour.

## The defects, fixed

- **D-TOK-1** — `TurboQuantEngine.estimate_tokens`
  kept a private copy of the heuristic that counted
  `""` as **1** with no model loaded, breaching the
  TF-065 contract's empty rule (the backend wrapper
  guarded it; the engine itself did not). Fixed: the
  fallback delegates to `backend_contract.estimate_tokens`
  — one estimate, one home — and the model path's BOS
  basis is documented on the method.
- **D-TOK-2** — the contract estimate was
  **script-blind**: flat `len // 4` counted the
  11-character CJK corpus sentence as **2** against a
  real tokeniser's **16** (relative error 0.88). Fixed:
  script-dense characters (CJK ideographs, kana,
  Hangul — `SCRIPT_TOKEN_RANGES`) count one token
  each; the sentence now estimates 11, within
  tolerance of 16. Latin-only counts are byte-identical
  to TF-065's (every existing Latin pin stands).
- **D-TOK-3** — HF's `tokens_generated` was
  `len(text.split())` — a **word count under a token
  name** — on both the client and raw-request legs
  (TF-065 had it declared in the capability note; the
  discipline removes the third convention instead of
  merely labelling it). Fixed: both legs report
  `self.count_tokens(text)`, the backend's declared
  basis; the capability note says so, and TF-065's
  note pin was updated in scope.

## Proof grade

FIXTURE — the shipped backends and the real engine on
this VM, a deterministic stub llama model (filed
reference subword rule + BOS) standing in for a loaded
model's tokeniser, and BPE counts measured from
tiktoken's cl100k_base (re-derived live when the tool
is importable; filed as data when not). Agreement with
a **loaded GGUF model's** tokeniser is **NOT PROVEN**:
no llama_cpp and no GGUF exist on this VM. TF-074's
Aesir backend joins under this same tolerance, and
the live-model slices own the loaded-tokeniser road.
Never claimed.
