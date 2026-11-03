# The Wing's Provenance — Authorship and History Preserved (TF-150)

GENERATED from the filed data in `src/thoughtforge/wing_provenance.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.wing_provenance` (the command also completes `docs/architecture/WING_CLOSING.md` with this account as its section 9). The checker (`wing_provenance_problems()`) re-derives every claim live from git history and fails on any drift.

Format: `thoughtforge-wing-provenance/1`
History rows filed: 14; authorship rows filed: 7 (known 3 / known-none 1 / unknown 3)
Commits touching research_data/: 3 (the import + TF-144 + TF-148)
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built: `git log -- research_data` shows the wing entered the repo in ONE commit, `6ba2df7` (2026-03-31, Volmarr Wyrd, message '1'), 96 files and 23,701 insertions, and was later touched only by TF-144 (retirement banners) and TF-148 (RESEARCH_FENCE.md). Repo-wide git records exactly three authors — Yrsa Freydisdottir (the campaign), Volmarr Wyrd, and google-labs-jules[bot] (2 commits, neither touching research_data/). TF-006's census is the wing's first internal account. Who authored the wing's files before the import, when they were written, and why wyrdforge.models was never committed are recorded nowhere in this repo and are filed UNKNOWN below — ORIGINS-shaped honesty: the evidence is recorded exactly, and the gaps are filed as gaps, never filled by invention.

## 2. The wing's recorded history

| When | Event | Wing files touched? | Evidence |
|---|---|---|---|
| 2026-03-31 (`6ba2df7`) | the wing enters this repo in a single commit — 96 files, 23,701 insertions, every file under research_data/; author Volmarr Wyrd; the commit message is the single character '1' with no body — filed exactly as git records it, its meaning not interpreted | yes | git log/show for the import commit, re-derived live by the checker |
| 2026-10-30 | the wing's first internal account: TF-006's census — the census counts 96 files, files the 14 schemas as the wing's most complete artefact, and records the headline finding that wyrdforge.models was never committed — filed as D-WING-1; the census is an audit doc and touches no wing file | no | docs/audit/TF-006_RESEARCH_WING_CENSUS.md |
| 2026-11-02 (`7bfedd8`) | bond graph verdict: ADOPT_AS_LAYER — the bond graph adopted as a layer beside the shipping knowledge store, decided on TF-006's census + a working spike; no wing file changed | no | docs/architecture/BOND_GRAPH_VERDICT.md |
| 2026-11-02 (`15c0b6b`) | micro-RAG verdict: RETIRE — the micro-RAG retired as a retrieval mechanism by measurement on TF-043's fixtures; no wing file changed | no | docs/architecture/MICRO_RAG_VERDICT.md |
| 2026-11-02 (`6140c4d`) | persona compiler verdict: MERGE (adopt-in-part) — the persona compiler's shape adopted in part against the shipping config/persona path; no wing file changed | no | docs/architecture/PERSONA_COMPILER_VERDICT.md |
| 2026-11-02 (`330e7f1`) | truth calibrator verdict: RETIRE — the truth calibrator retired as a calibration / enforcement mechanism by measurement on TF-100's chaos corpus; no wing file changed | no | docs/architecture/TRUTH_CALIBRATOR_VERDICT.md |
| 2026-11-02 (`55541a2`) | adoptions implemented in the shipping tree — TF-139's bond graph and TF-141's persona compiler landed as production code under src/thoughtforge/; the wing package itself was not resurrected and no wing file changed | no | docs/architecture/BOND_GRAPH.md + docs/architecture/PERSONA_COMPILER.md |
| 2026-11-02 (`c41aade`) | retirements executed: archived-in-place — retirement banners prepended verbatim to the micro-RAG service, the truth calibrator service and their shared wing test file — bodies byte-for-byte below, insertions only; one of only two campaign commits that touch the wing | yes | git log/show for the TF-144 slice commit + docs/architecture/WING_CLOSING.md |
| 2026-11-02 (`ca2d41c`) | WYRD boundary doc filed — ThoughtForge vs the WYRD Protocol ownership and interface filed for the Second Life/Hermes consumers; no wing file changed | no | docs/architecture/WYRD_BOUNDARY.md |
| 2026-11-02 (`47ca8ea`) | the bond-graph bridge built from production code — bonds as WYRD-consumable data, built FROM TF-143's production BondGraph, not the wing; no wing file changed | no | docs/architecture/BOND_BRIDGE.md |
| 2026-11-02 (`919d98d`) | persona compiler to Runmal cards — compiled personae exportable in Runmal's card shape, built FROM TF-143's production compiler; no wing file changed | no | docs/architecture/RUNMAL_CARDS.md |
| 2026-11-02 (`158bae7`) | research hygiene: the wing fenced as research — research_data/RESEARCH_FENCE.md added (the tracked census becomes 96 baseline + 1 fence = 97) and every wing area/file given exactly one disposition; the second of only two campaign commits that touch the wing | yes | git log/show for the TF-148 slice commit + docs/architecture/RESEARCH_HYGIENE.md |
| 2026-11-02 (`da42954`) | the wing's tests disposed; D-WING-1 closed — every wing test function given its final disposition and D-WING-1 CLOSED-BY-VERDICT; TF-149's closure files TF-150/TF-152 as the owners of provenance/gate, not that entry; no wing file changed | no | docs/architecture/WING_TESTS.md |
| 2026-11-02 | provenance preserved (this slice) — the wing's authorship/history recorded here and in the closing doc's provenance section; this slice files data and docs only — it changes no wing file and adds no docket entry | no | docs/architecture/WING_PROVENANCE.md + the provenance section of docs/architecture/WING_CLOSING.md |

## 3. Authorship — known and unknown

| Question | Status | What is filed |
|---|---|---|
| who committed the wing into this repo? | known | the import commit 6ba2df7 (2026-03-31) is authored by Volmarr Wyrd, under the noreply address git records for that commit |
| who authored the wing's files before the import commit? | unknown | UNKNOWN — not recorded: git history for these paths begins at the import commit, and no earlier history, authorship note or attribution for the wing's files stands in this repo — no author is invented |
| when were the wing's files written? | unknown | UNKNOWN — not recorded: the only date git records for the wing's arrival is the import commit's 2026-03-31; the files' writing dates before that commit are recorded nowhere in this repo |
| who authored the campaign-era touches to the wing? | known | the only two post-import commits touching research_data/ — TF-144 (retirement banners) and TF-148 (RESEARCH_FENCE.md) — are campaign commits authored by Yrsa Freydisdottir, as are all Part VIII slice commits TF-139..TF-149 |
| did the repo's bot author touch the wing? | known_none | none: the bot authored exactly 2 commits in this repo (2026-03-30: a resources download under data_resources/ and a research document at the repo root) and neither touches research_data/ — filed exactly as git shows |
| why was wyrdforge.models never committed? | unknown | UNKNOWN — not recorded beyond the fact itself: TF-006 and D-WING-1 file THAT the models subpackage was never committed; no reason, motive or circumstance for the omission is recorded anywhere in this repo — none is invented |
| which authors does git record in this repo at all? | known | exactly three distinct authors: Volmarr Wyrd, Yrsa Freydisdottir and google-labs-jules[bot] — the set is filed, not counts, so a later campaign commit cannot silently falsify it |

## 4. The closing doc, completed

TF-144 filed that TF-150 completes the wing's provenance in the closing doc. It now stands there as section 9 of `docs/architecture/WING_CLOSING.md`, appended below TF-144's render, which stands byte-for-byte above it — the completion is checked live as exactly TF-144's render plus this slice's appendix. TF-144's checker and doc test were updated minimally in the open to accept exactly that completed shape, the estate's established in-the-open neighbour update.

## 5. Docket

None. D-WING-1 is already CLOSED-BY-VERDICT by TF-149, whose closure files TF-150/TF-152 as the owners of provenance/gate, not that entry. This slice adds no docket entry, reopens nothing, and TF-016 totals do not move: 137 entries — 31 OPEN / 1 OPEN-ENVIRONMENT / 101 FIXED / 4 CLOSED-BY-VERDICT.

## 6. The probes

- `clean_estate_passes`: the filed history rows + authorship rows pass their validators
- `import_row_wrong_hash_is_named`: an import row filing a different commit hash is named
- `unknown_row_claiming_an_answer_is_named`: an unknown authorship row that files a claim (an invented answer) is named
- `invented_author_is_named`: an authorship row naming an author outside the repo's recorded author set is named
- `extra_wing_touch_is_named`: a history row claiming a wing touch beyond the import / TF-144 / TF-148 is named
- `slice_row_without_a_commit_is_named`: a TF-139..TF-149 history row with its commit emptied is named
- `doc_drift_is_named`: a doc text that drifts from the filed render (or a closing doc that is not TF-144's render plus the appendix) is named

## 7. Filed limits

- Git-history grade only: every known claim above is re-derived live from this repo's git history; the history before the import commit is not in this repo and is filed UNKNOWN, never filled in.
- Pre-import authorship UNKNOWN: who authored the wing's files before 2026-03-31, when they were written, and why `wyrdforge.models` was never committed are recorded nowhere in this repo — this slice files the questions and the absence, and invents no answers, dates or motives.
- The import message '1' is filed verbatim and not interpreted: git records the character, not a meaning.
- Provenance here covers the wing's code/docs history only; no character/persona content is republished or filed by this slice.
- No README.md anywhere was touched, no wing file was changed, and the docket is untouched: D-WING-1 stands CLOSED-BY-VERDICT by TF-149 with TF-016 totals unchanged (137 — 31 OPEN / 1 OPEN-ENVIRONMENT / 101 FIXED / 4 CLOSED-BY-VERDICT); TF-151 owns the panel re-run and TF-152 the Part VIII gate.
- No live model was read; live behaviour is NOT PROVEN (TF-173 class), never claimed.
