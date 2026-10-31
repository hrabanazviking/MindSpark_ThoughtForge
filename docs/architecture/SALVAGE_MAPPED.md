# TF-087 — Salvage Mapped: What the Rescue Engine Actually Does

*Salvage's strategies catalogued, each with fixtures
of broken input. The estate lives in
`src/thoughtforge/refinement/salvage_map.py`
(`STRATEGIES` / `ABSENT_STRATEGIES` /
`BROKEN_FIXTURES` / `SALVAGE_RULES` as data, the
checkers `salvage_map_problems` /
`salvage_map_live_problems`, format
`thoughtforge-salvage-map/1`). Part V opens here.*

## The plain truth first

Read plainly, `FragmentSalvage` is a **selector and
re-composer, not a repair shop**. It scores the
drafts a small model produced, picks the best one,
pulls good sentences out of the top two, and — when
an engine is available — asks that engine to write a
cleaner answer from those sentences, keeping the new
answer only if it scores better. When no engine is
available, the best draft goes out **exactly as the
model wrote it**, broken edges and all.

The roadmap named this part with words like
"truncation repair, rambling compression, format
rescue". Probed live this tick, **none of those
rescues exist**: a draft cut off mid-sentence, a
broken JSON-shaped draft, a sentence looped twenty
times, and a lone rambling draft were each returned
byte-identical. They are filed below as one gap,
D-SALV-1, each with the Part V slice that owns
building it. Nothing in this map claims otherwise.

## The five strategies that exist

| Strategy | How it works | Output |
|---|---|---|
| Draft scoring & selection | composite = 0.45·length (chars/550, capped) + 0.55·citation (share of retrieved QIDs in the text; neutral 0.5 when none were retrieved) | drafts ranked; the best proceeds |
| Fragment extraction | sentences from the top 2 drafts only; ≥4 words, exact-deduped, kept at fragment score ≥0.54 (length chars/200 ·0.45 + citation 0.8/0.4/0.5 ·0.55); at most 5 | fragment texts — inputs to refine only |
| Refine passes | only with engine + prompt builder + sketch + scaffold all present; up to `max_passes` (default 2, profile-resolved since TF-037) at temperature 0.50 −0.05 per pass; a pass is adopted only when its score strictly beats the current best | the best text seen; path `refine_pass_<passes attempted>` |
| Best-draft fallback | no engine surface, or refine completed zero passes | the best draft, byte-identical |
| Empty fallback | no candidates at all | one fixed sentence at confidence 0, path `empty` |

The filed constants are judged live against
salvage.py's own values by the checker — a silent
constant change fails this slice's tests by name.

Three behaviours worth naming plainly:

- **Citation outweighs length.** A short cited draft
  (94 chars, composite 0.627) beat a 527-char
  rambling draft (0.431) — selection can rescue a
  turn from rambling, but only when a better rival
  draft exists. Alone, the rambling draft went out
  verbatim.
- **Fragments do not clean an answer.** The mixed
  fixture (junk short sentences plus one long cited
  sentence) kept exactly 1 fragment, yet the
  returned text was the whole draft, junk included.
  `fragments_kept` is a refine input count, never a
  cleanliness claim.
- **A refine path names attempts, not adoption.**
  When a pass ran but scored no better, the original
  text stood — under the path `refine_pass_1` with
  `passes_used=1`. TF-097's rescue record must carry
  that distinction, or the trace will overclaim.

## The fixtures

Nine fixtures of broken input are filed in
`BROKEN_FIXTURES` (synthetic Lantern Hollow
material, never canon) and run through the real
`FragmentSalvage`: empty candidates, a truncated
draft, rambling beaten by a cited rival, rambling
alone, the fragment filter, broken JSON, a 20×
loop, a refine pass adopting a better output, and
a refine pass whose engine fails and falls back.
The no-engine fixtures are re-run by the live
checker on every suite run; the refine fixtures
run in the tests with a stub engine and the real
`PromptBuilder`.

## Filed OPEN — D-SALV-1, the absent rescues

| Absent strategy | Evidence | Owner |
|---|---|---|
| Truncation repair | mid-sentence draft returned byte-identical | TF-090 |
| Rambling compression | lone rambling draft returned byte-identical | TF-089 |
| Format rescue | broken JSON returned byte-identical | TF-090 |
| Repetition / loop cut | 20× loop returned byte-identical (dedup touches fragments only) | TF-091 |
| Language-drift rescue | no language check exists in salvage.py at all | TF-092 |

## Not proven — the honest remainder

- **A live model's broken outputs: NOT PROVEN.**
  Every fixture here is synthetic and deterministic;
  no model or engine ships in this tree. TF-100's
  chaos corpus owns real-shaped failures, and
  TF-101 runs the full policy stack over them.
- **Salvage safety is not yet judged.** Whether the
  refine path can *invent* content is TF-088's
  question, next in this part — this slice maps
  what exists and files what does not; it does not
  certify the refine path safe.

## Docket

- **D-SALV-1 added OPEN by TF-087**: the five absent
  rescue strategies above, owners TF-089 / TF-090 /
  TF-091 / TF-092.
- TF-016 totals move to 87 entries — 45 OPEN / 1
  OPEN-ENVIRONMENT / 38 FIXED / 3
  CLOSED-BY-VERDICT (Part V 2 → 3).

## Proof grade

FIXTURE — every catalogued behaviour and every
absence was run through the real `FragmentSalvage`
on this VM this tick (scoring, extraction, stub-
engine refine legs with the real PromptBuilder,
fallbacks). A live model's broken outputs are
**NOT PROVEN**, never claimed.
