# TF-089 — Salvage Metrics: Rescue and Corruption, Measured

*The fixture corpus scored: how often salvage
actually rescues a broken answer, and how often
its output is corrupted — with policy thresholds
taken from the measurements, not wished for. The
estate lives in
`src/thoughtforge/refinement/salvage_metrics.py`
(`METRIC_RULES` / `BROKEN_OUTCOMES` /
`ADVERSARIAL_OUTCOMES` / `FILED_MEASUREMENTS` /
`THRESHOLDS` as data, the live measurement
`measure_metrics`, the checkers
`salvage_metrics_problems` /
`salvage_metrics_live_problems`, format
`thoughtforge-salvage-metrics/1`).*

## The plain truth first

Salvage rescues less than its name suggests, and
corrupts less than it could. Measured live over
the whole filed fixture corpus — TF-087's nine
broken inputs plus TF-088's nine adversarial
outputs, 18 entries, every one run through the
real `FragmentSalvage` this tick:

- **Rescue rate: 2 of 7 eligible (0.286).** The
  two rescues are selection (a cited concise rival
  chosen over a rambling draft) and one refine
  adoption. The other five eligible broken inputs
  come back **byte-identical** — the number this
  project has to be honest about.
- **Marker corruption: 0.** All six fabricated
  refine outputs TF-088 filed are rejected; no
  invented number, citation, name, polarity flip
  or displaced time is ever adopted.
- **Broad corruption is not zero: 1 of the 2
  adopted adversarial refine outputs** is TF-088's
  filed residual — facts added in common words
  with no marker for any guard to catch.

## The corpus

No new fixtures were invented for this slice —
grading your own new homework flatters the rates.
The corpus is exactly the estate TF-087 and
TF-088 filed, synthetic Lantern Hollow material
throughout:

| Corpus | Source | Entries | What it grades |
|---|---|---|---|
| Rescue | TF-087 `BROKEN_FIXTURES` | 9 | broken input in, judged output out |
| Corruption | TF-088 `ADVERSARIAL_FIXTURES` | 9 | invented content offered, adopted or refused |

## Rescue, fixture by fixture

Two of the nine are handled safely but are not
rescue-eligible, and leave the denominator: the
empty input (nothing exists to rescue; the fixed
fallback sentence answers) and the engine-failure
fixture (its input is already a clean cited
draft; the fallback returns it untouched, which
is safety, not rescue).

| Fixture | Outcome |
|---|---|
| truncated_mid_sentence | **unrescued** — returned byte-identical, still ending mid-sentence |
| rambling_beaten_by_cited_rival | **rescued by selection** — the cited rival is returned, the rambling discarded |
| rambling_alone_verbatim | **unrescued** — returned byte-identical |
| fragment_filter_keeps_only_scored_sentences | **unrescued** — one fragment kept, the whole draft (junk included) returned |
| broken_json_verbatim | **unrescued** — returned byte-identical |
| loop_verbatim | **unrescued** — all 20 repetitions returned |
| refine_adopts_better_output | **rescued by refine** — the better output adopted as refine_pass_1 |

**Unrescued verbatim: 5.** That count is D-SALV-1
in numbers: truncation repair, format rescue,
loop cutting and output cleaning do not exist
yet (TF-090 / TF-091 own them), and this slice's
live checker fails by name if a filed outcome
and the live output ever disagree.

### The rambling share (D-SALV-1, TF-089's part)

The rambling problem appears in two fixtures and
is rescued in one — **rambling rescue 1/2**, and
the one rescue is selection, not compression.
A lone rambling draft is compressed **0/1**:
there is still no compression mechanism in
salvage, and this slice claims none. TF-089
discharges the rambling share of D-SALV-1 as a
*measurement*; the entry itself stays OPEN for
its remaining owners (TF-090 truncation+format,
TF-091 loops, TF-092 drift), and TF-101 will
score whatever mechanisms exist on TF-100's
chaos corpus.

## Corruption, in two tiers

Merging these into one number would lie in both
directions, so they are filed separately:

| Tier | Measured | Reading |
|---|---|---|
| Marker corruption (TF-088's guard kinds) | **0 of 6** fabrication attempts adopted; prevention 6/6 | the guard holds on its whole filed set |
| Broad corruption (adopted outputs) | **1 of 2** adopted adversarial refine outputs corrupted | the one is the common-word residual, human-judged at TF-088's filing, marker-clean by definition |
| Outputs carrying invented facts | 2 of 9 adversarial outputs | the residual adoption (salvage's) plus the source-invention draft passed through verbatim (the draft's own inventions — generation/enforcement judge drafts; salvage is pool-relative, TF-088) |

## Policy thresholds — from the measurements

| Threshold | Value | Basis |
|---|---|---|
| Marker adoption ceiling | 0.0 | absolute — TF-088's gate, inherited by TF-104 |
| Fabrication prevention floor | 1.0 | absolute — every filed fabrication refused |
| Rescue rate floor | 2/7 (0.286) | the measured rate; any drop is a regression |
| Unrescued verbatim ceiling | 5 | the measured count; TF-090/TF-091 should *lower* it, slice by slice, in the open |
| Adopted corruption ceiling | 0.5 | the measured rate; its cause is named (the residual), never averaged away |
| Rambling rescue floor | 0.5 | the measured rate |

A future slice that improves a rate moves the
filed measurement and its threshold together in
its own commit — the live checker recomputes
every measurement on every suite run, so a stale
filed number fails by name instead of drifting.

## Docket

- **D-SALV-1's rambling share measured by
  TF-089**: rambling rescue 1/2 (by selection),
  lone-rambling compression 0/1 (no mechanism —
  still absent, not fixed). The entry stays OPEN;
  its other shares keep their owners (TF-090,
  TF-091, TF-092). TF-016 totals unchanged:
  88 entries — 45 OPEN / 1 OPEN-ENVIRONMENT /
  39 FIXED / 3 CLOSED-BY-VERDICT.

## Proof grade

FIXTURE — both corpora run live through the real
`FragmentSalvage` (stub engines, the real
`PromptBuilder`) on this VM, and every filed
measurement is the live recomputation. A live
model's rescue and corruption rates are **NOT
PROVEN** (no model in tree); TF-100's chaos
corpus owns real-shaped failures and TF-101
runs the full policy stack over them.
