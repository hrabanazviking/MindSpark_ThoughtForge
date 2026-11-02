"""
TF-150 — the wyrdforge wing's provenance, preserved.

TF-150's roadmap text: *Provenance preserved: the
wing's authorship/history recorded in the closing
doc (ORIGINS-shaped honesty).*

ORIGINS-shaped honesty, as this slice practises
it: record exactly what the evidence shows, and
file as UNKNOWN what is not recorded. Never invent
authorship, dates, or motives.

The measured start, probed live on this tree
before anything was built:

  * ``git log -- research_data`` shows the wing
    entered this repo in ONE commit:
    ``6ba2df79f2c09a72eeb8a83191e1cddc463d2e34``
    (2026-03-31, author Volmarr Wyrd, message
    ``1`` — the single character, no body), 96
    files, 23,701 insertions, every one of them
    under ``research_data/``.
  * Since the import, exactly two commits have
    touched ``research_data/``: TF-144
    (``c41aade``, retirement banners on the two
    retired service files + the shared wing test
    file) and TF-148 (``158bae7``, the new
    ``research_data/RESEARCH_FENCE.md`` fence
    file). Both are campaign commits by Yrsa
    Freydisdottir. No other commit in the repo's
    history touches the wing.
  * Repo-wide, git records exactly three distinct
    authors: Yrsa Freydisdottir (the campaign),
    Volmarr Wyrd, and ``google-labs-jules[bot]``
    (2 commits, 2026-03-30 — a resources download
    under ``data_resources/`` and a research
    document at the repo root). Neither bot
    commit touches ``research_data/`` — checked
    live, exactly what git shows.
  * TF-006's census
    (``docs/audit/TF-006_RESEARCH_WING_CENSUS.md``,
    measured 2026-10-30) is the wing's first
    internal account: 96 files, the 14 schemas as
    its most complete artefact, and the headline
    finding that ``wyrdforge.models`` was never
    committed (filed as D-WING-1).
  * What is NOT recorded anywhere in this repo:
    who authored the wing's files before the
    import commit, when they were written, and
    why ``wyrdforge.models`` was never committed
    — beyond what TF-006 and the docket already
    file (that it was not committed). Those three
    questions are filed UNKNOWN below. This slice
    invents no answer for them.

**What this slice files** (literal data below):

  * ``HISTORY_ROWS`` — every recorded event in
    the wing's history in this repo (14): the
    import commit, TF-006's census, and each
    Part VIII slice TF-139..TF-150 with its
    executed disposition, its slice commit where
    git records one, and whether it touched the
    wing's files.
  * ``AUTHORSHIP_ROWS`` — the authorship
    questions, one row each (7): what is KNOWN
    files its claim and its evidence source;
    what is UNKNOWN files no claim and names why
    it is not recorded. A row naming an author
    outside the repo's recorded author set is
    named by the validators.
  * 7 filed probes through the pure validators.

**The closing doc**: TF-144 promised TF-150
completes the wing's provenance in
``docs/architecture/WING_CLOSING.md``. This
slice appends the provenance account there as
section 9, BELOW TF-144's render, which stands
byte-for-byte above it. Mechanics, in the open:
TF-144's own checker and TF-144's doc test were
updated minimally in this slice to accept
exactly ``TF-144 render + this slice's
appendix`` — the same in-the-open neighbour
update TF-137/TF-149 practised. This module's
``__main__`` regenerates BOTH docs: the
standalone ``docs/architecture/WING_PROVENANCE.md``
and the completed closing doc.

**Docket**: none. D-WING-1 is already
CLOSED-BY-VERDICT by TF-149, whose closure files
"TF-150/TF-152 own provenance/gate, not this
entry". TF-150 adds no docket entry, reopens
nothing, and TF-016 totals do not move (137 —
31 OPEN / 1 OPEN-ENVIRONMENT / 101 FIXED / 4
CLOSED-BY-VERDICT).

Provenance here is about the wing's code/docs
history only — no character/persona content is
republished or filed.

Proof grade: GIT-HISTORY / FIXTURE-STATIC —
filings re-derived live from this repo's git
history and tree on this VM; pre-import
authorship is UNKNOWN, never claimed; no live
model, never claimed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

WING_PROVENANCE_FORMAT = "thoughtforge-wing-provenance/1"
WING_PROVENANCE_DOC = "docs/architecture/WING_PROVENANCE.md"
WING_CLOSING_DOC = "docs/architecture/WING_CLOSING.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

IMPORT_COMMIT = "6ba2df79f2c09a72eeb8a83191e1cddc463d2e34"
IMPORT_AUTHOR = "Volmarr Wyrd"
IMPORT_DATE = "2026-03-31"
IMPORT_MESSAGE = "1"
IMPORT_FILES = 96
IMPORT_INSERTIONS = 23701

TF144_COMMIT = "c41aade2520344612e36084ae4bf966aa9d493fd"
TF148_COMMIT = "158bae76de3fa89155ea8374a0ba197c76f12047"
BOT_AUTHOR = "google-labs-jules[bot]"

#: Every distinct author git records in this repo.
REPO_AUTHORS: tuple[str, ...] = (
    "Volmarr Wyrd",
    "Yrsa Freydisdottir",
    "google-labs-jules[bot]",
)

#: The statuses an authorship row may file.
AUTHORSHIP_STATUSES: tuple[str, ...] = ("known", "unknown", "known_none")


# ── The wing's recorded history, as literal data ─────────────────────────────

HISTORY_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "import_commit",
     "slice": "",
     "date": "2026-03-31",
     "event": "the wing enters this repo in a single commit",
     "detail": "96 files, 23,701 insertions, every file under "
               "research_data/; author Volmarr Wyrd; the commit "
               "message is the single character '1' with no body "
               "— filed exactly as git records it, its meaning "
               "not interpreted",
     "commit": IMPORT_COMMIT,
     "touches_wing": True,
     "evidence": "git log/show for the import commit, re-derived "
                 "live by the checker"},
    {"id": "tf006_census",
     "slice": "TF-006",
     "date": "2026-10-30",
     "event": "the wing's first internal account: TF-006's census",
     "detail": "the census counts 96 files, files the 14 schemas "
               "as the wing's most complete artefact, and records "
               "the headline finding that wyrdforge.models was "
               "never committed — filed as D-WING-1; the census "
               "is an audit doc and touches no wing file",
     "commit": "",
     "touches_wing": False,
     "evidence": "docs/audit/TF-006_RESEARCH_WING_CENSUS.md"},
    {"id": "tf139_bond_verdict",
     "slice": "TF-139",
     "date": "2026-11-02",
     "event": "bond graph verdict: ADOPT_AS_LAYER",
     "detail": "the bond graph adopted as a layer beside the "
               "shipping knowledge store, decided on TF-006's "
               "census + a working spike; no wing file changed",
     "commit": "7bfedd8",
     "touches_wing": False,
     "evidence": "docs/architecture/BOND_GRAPH_VERDICT.md"},
    {"id": "tf140_micro_verdict",
     "slice": "TF-140",
     "date": "2026-11-02",
     "event": "micro-RAG verdict: RETIRE",
     "detail": "the micro-RAG retired as a retrieval mechanism "
               "by measurement on TF-043's fixtures; no wing "
               "file changed",
     "commit": "15c0b6b",
     "touches_wing": False,
     "evidence": "docs/architecture/MICRO_RAG_VERDICT.md"},
    {"id": "tf141_persona_verdict",
     "slice": "TF-141",
     "date": "2026-11-02",
     "event": "persona compiler verdict: MERGE (adopt-in-part)",
     "detail": "the persona compiler's shape adopted in part "
               "against the shipping config/persona path; no "
               "wing file changed",
     "commit": "6140c4d",
     "touches_wing": False,
     "evidence": "docs/architecture/PERSONA_COMPILER_VERDICT.md"},
    {"id": "tf142_truth_verdict",
     "slice": "TF-142",
     "date": "2026-11-02",
     "event": "truth calibrator verdict: RETIRE",
     "detail": "the truth calibrator retired as a calibration / "
               "enforcement mechanism by measurement on TF-100's "
               "chaos corpus; no wing file changed",
     "commit": "330e7f1",
     "touches_wing": False,
     "evidence": "docs/architecture/TRUTH_CALIBRATOR_VERDICT.md"},
    {"id": "tf143_adoptions",
     "slice": "TF-143",
     "date": "2026-11-02",
     "event": "adoptions implemented in the shipping tree",
     "detail": "TF-139's bond graph and TF-141's persona "
               "compiler landed as production code under "
               "src/thoughtforge/; the wing package itself was "
               "not resurrected and no wing file changed",
     "commit": "55541a2",
     "touches_wing": False,
     "evidence": "docs/architecture/BOND_GRAPH.md + "
                 "docs/architecture/PERSONA_COMPILER.md"},
    {"id": "tf144_retirements",
     "slice": "TF-144",
     "date": "2026-11-02",
     "event": "retirements executed: archived-in-place",
     "detail": "retirement banners prepended verbatim to the "
               "micro-RAG service, the truth calibrator service "
               "and their shared wing test file — bodies "
               "byte-for-byte below, insertions only; one of "
               "only two campaign commits that touch the wing",
     "commit": TF144_COMMIT,
     "touches_wing": True,
     "evidence": "git log/show for the TF-144 slice commit + "
                 "docs/architecture/WING_CLOSING.md"},
    {"id": "tf145_boundary",
     "slice": "TF-145",
     "date": "2026-11-02",
     "event": "WYRD boundary doc filed",
     "detail": "ThoughtForge vs the WYRD Protocol ownership "
               "and interface filed for the Second Life/Hermes "
               "consumers; no wing file changed",
     "commit": "ca2d41c",
     "touches_wing": False,
     "evidence": "docs/architecture/WYRD_BOUNDARY.md"},
    {"id": "tf146_bridge",
     "slice": "TF-146",
     "date": "2026-11-02",
     "event": "the bond-graph bridge built from production code",
     "detail": "bonds as WYRD-consumable data, built FROM "
               "TF-143's production BondGraph, not the wing; "
               "no wing file changed",
     "commit": "47ca8ea",
     "touches_wing": False,
     "evidence": "docs/architecture/BOND_BRIDGE.md"},
    {"id": "tf147_runmal",
     "slice": "TF-147",
     "date": "2026-11-02",
     "event": "persona compiler to Runmal cards",
     "detail": "compiled personae exportable in Runmal's card "
               "shape, built FROM TF-143's production compiler; "
               "no wing file changed",
     "commit": "919d98d",
     "touches_wing": False,
     "evidence": "docs/architecture/RUNMAL_CARDS.md"},
    {"id": "tf148_hygiene",
     "slice": "TF-148",
     "date": "2026-11-02",
     "event": "research hygiene: the wing fenced as research",
     "detail": "research_data/RESEARCH_FENCE.md added (the "
               "tracked census becomes 96 baseline + 1 fence = "
               "97) and every wing area/file given exactly one "
               "disposition; the second of only two campaign "
               "commits that touch the wing",
     "commit": TF148_COMMIT,
     "touches_wing": True,
     "evidence": "git log/show for the TF-148 slice commit + "
                 "docs/architecture/RESEARCH_HYGIENE.md"},
    {"id": "tf149_wing_tests",
     "slice": "TF-149",
     "date": "2026-11-02",
     "event": "the wing's tests disposed; D-WING-1 closed",
     "detail": "every wing test function given its final "
               "disposition and D-WING-1 CLOSED-BY-VERDICT; "
               "TF-149's closure files TF-150/TF-152 as the "
               "owners of provenance/gate, not that entry; no "
               "wing file changed",
     "commit": "da42954",
     "touches_wing": False,
     "evidence": "docs/architecture/WING_TESTS.md"},
    {"id": "tf150_provenance",
     "slice": "TF-150",
     "date": "2026-11-02",
     "event": "provenance preserved (this slice)",
     "detail": "the wing's authorship/history recorded here "
               "and in the closing doc's provenance section; "
               "this slice files data and docs only — it "
               "changes no wing file and adds no docket entry",
     "commit": "",
     "touches_wing": False,
     "evidence": "docs/architecture/WING_PROVENANCE.md + the "
                 "provenance section of "
                 "docs/architecture/WING_CLOSING.md"},
)


# ── The authorship questions, as literal data ────────────────────────────────

AUTHORSHIP_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "who_committed_the_wing",
     "question": "who committed the wing into this repo?",
     "status": "known",
     "author": "Volmarr Wyrd",
     "claim": "the import commit 6ba2df7 (2026-03-31) is "
              "authored by Volmarr Wyrd, under the noreply "
              "address git records for that commit",
     "evidence": "git show for the import commit, re-derived "
                 "live by the checker",
     "unknown_reason": ""},
    {"id": "who_authored_the_files_before_import",
     "question": "who authored the wing's files before the "
                 "import commit?",
     "status": "unknown",
     "author": "",
     "claim": "",
     "evidence": "",
     "unknown_reason": "not recorded: git history for these "
                       "paths begins at the import commit, and "
                       "no earlier history, authorship note or "
                       "attribution for the wing's files stands "
                       "in this repo — no author is invented"},
    {"id": "when_were_the_files_written",
     "question": "when were the wing's files written?",
     "status": "unknown",
     "author": "",
     "claim": "",
     "evidence": "",
     "unknown_reason": "not recorded: the only date git "
                       "records for the wing's arrival is the "
                       "import commit's 2026-03-31; the files' "
                       "writing dates before that commit are "
                       "recorded nowhere in this repo"},
    {"id": "campaign_era_wing_touches",
     "question": "who authored the campaign-era touches to "
                 "the wing?",
     "status": "known",
     "author": "Yrsa Freydisdottir",
     "claim": "the only two post-import commits touching "
              "research_data/ — TF-144 (retirement banners) "
              "and TF-148 (RESEARCH_FENCE.md) — are campaign "
              "commits authored by Yrsa Freydisdottir, as are "
              "all Part VIII slice commits TF-139..TF-149",
     "evidence": "git log for research_data/ and for the "
                 "Part VIII slices, re-derived live by the "
                 "checker",
     "unknown_reason": ""},
    {"id": "bot_authorship_in_the_wing",
     "question": "did the repo's bot author touch the wing?",
     "status": "known_none",
     "author": "google-labs-jules[bot]",
     "claim": "none: the bot authored exactly 2 commits in "
              "this repo (2026-03-30: a resources download "
              "under data_resources/ and a research document "
              "at the repo root) and neither touches "
              "research_data/ — filed exactly as git shows",
     "evidence": "git log by author, with and without the "
                 "research_data/ pathspec, re-derived live "
                 "by the checker",
     "unknown_reason": ""},
    {"id": "why_models_never_committed",
     "question": "why was wyrdforge.models never committed?",
     "status": "unknown",
     "author": "",
     "claim": "",
     "evidence": "",
     "unknown_reason": "not recorded beyond the fact itself: "
                       "TF-006 and D-WING-1 file THAT the "
                       "models subpackage was never committed; "
                       "no reason, motive or circumstance for "
                       "the omission is recorded anywhere in "
                       "this repo — none is invented"},
    {"id": "repo_wide_authors",
     "question": "which authors does git record in this "
                 "repo at all?",
     "status": "known",
     "author": "",
     "claim": "exactly three distinct authors: Volmarr Wyrd, "
              "Yrsa Freydisdottir and google-labs-jules[bot] "
              "— the set is filed, not counts, so a later "
              "campaign commit cannot silently falsify it",
     "evidence": "git log author set, re-derived live by the "
                 "checker",
     "unknown_reason": ""},
)


# ── Pure validators (the probes run against these) ───────────────────────────

def history_row_problems(row: Any) -> list[str]:
    """Judge one history filing, purely: an id, a
    date, an event, a detail, an evidence source,
    a boolean wing-touch flag, and the commit
    discipline of its kind — the import row must
    file the full import hash, a slice row for
    TF-139..TF-149 must file a commit, and only
    the import / TF-144 / TF-148 rows may file
    that they touched the wing. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["history: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("history: the row files no id")
    if not row.get("date"):
        problems.append(f"history {row.get('id')}: no date filed")
    if not row.get("event"):
        problems.append(f"history {row.get('id')}: no event filed")
    if not row.get("detail"):
        problems.append(f"history {row.get('id')}: no detail filed")
    if not row.get("evidence"):
        problems.append(
            f"history {row.get('id')}: no evidence filed")
    if not isinstance(row.get("touches_wing"), bool):
        problems.append(
            f"history {row.get('id')}: wing-touch flag is not "
            "filed as a boolean")
    if row.get("id") == "import_commit" \
            and row.get("commit") != IMPORT_COMMIT:
        problems.append(
            "history import_commit: the commit is not the "
            "filed full import hash — the import is the one "
            "event whose hash must stand in full")
    slice_id = str(row.get("slice") or "")
    if slice_id in ("TF-139", "TF-140", "TF-141", "TF-142",
                    "TF-143", "TF-144", "TF-145", "TF-146",
                    "TF-147", "TF-148", "TF-149") \
            and not row.get("commit"):
        problems.append(
            f"history {row.get('id')}: slice {slice_id} files "
            "no commit — git records one for every Part VIII "
            "slice")
    if row.get("touches_wing") is True and row.get("id") not in (
            "import_commit", "tf144_retirements", "tf148_hygiene"):
        problems.append(
            f"history {row.get('id')}: files that it touched "
            "the wing, but git records exactly three wing "
            "commits — the import, TF-144 and TF-148")
    return problems


def authorship_row_problems(row: Any) -> list[str]:
    """Judge one authorship filing, purely: a
    question, a status in the filed vocabulary,
    and the honesty discipline of that status —
    a known row files a claim + evidence, an
    unknown row files NO claim and names why it
    is not recorded, a known_none row files the
    none-claim + evidence, and no row may name
    an author outside the repo's recorded author
    set. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["authorship: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("authorship: the row files no id")
    if not row.get("question"):
        problems.append(
            f"authorship {row.get('id')}: no question filed")
    status = row.get("status")
    if status not in AUTHORSHIP_STATUSES:
        problems.append(
            f"authorship {row.get('id')}: status is "
            f"{status!r}, not one of {AUTHORSHIP_STATUSES}")
    author = row.get("author")
    if author and author not in REPO_AUTHORS:
        problems.append(
            f"authorship {row.get('id')}: names author "
            f"{author!r}, who is not in the repo's recorded "
            "author set — no authorship may be invented")
    if status == "known":
        if not row.get("claim"):
            problems.append(
                f"authorship {row.get('id')}: a known row "
                "files no claim")
        if not row.get("evidence"):
            problems.append(
                f"authorship {row.get('id')}: a known row "
                "files no evidence source")
    if status == "known_none":
        if not row.get("claim") or "none" not in str(
                row.get("claim")).lower():
            problems.append(
                f"authorship {row.get('id')}: a known_none "
                "row must file its none-claim")
        if not row.get("evidence"):
            problems.append(
                f"authorship {row.get('id')}: a known_none "
                "row files no evidence source")
    if status == "unknown":
        if row.get("claim"):
            problems.append(
                f"authorship {row.get('id')}: an unknown row "
                "files a claim — what is not recorded must "
                "be filed unknown, never answered by "
                "invention")
        if row.get("author"):
            problems.append(
                f"authorship {row.get('id')}: an unknown row "
                "names an author — no authorship may be "
                "invented for an unrecorded question")
        if not row.get("unknown_reason"):
            problems.append(
                f"authorship {row.get('id')}: an unknown row "
                "files no reason why it is not recorded")
    if row.get("id") == "bot_authorship_in_the_wing" \
            and status != "known_none":
        problems.append(
            "authorship bot_authorship_in_the_wing: the bot's "
            "wing share must file known_none — git shows it "
            "touched no wing file")
    return problems


def provenance_state_problems(state: Any, expected: Any) -> list[str]:
    """Judge a measured provenance state against
    its expected filing, purely. Empty-safe."""
    if not isinstance(state, dict) or not state:
        return ["provenance state: the state is empty"]
    if not isinstance(expected, dict) or not expected:
        return ["provenance state: the expected filing is empty"]
    problems: list[str] = []
    for key, value in expected.items():
        if state.get(key) != value:
            problems.append(
                f"provenance state: {key} measures "
                f"{state.get(key)!r}, not the filed {value!r}")
    return problems


def doc_text_problems(text: Any, render: Any, label: Any) -> list[str]:
    """Judge a filed doc against its render,
    purely: any drift is named. Empty-safe."""
    if not isinstance(render, str) or not render:
        return ["doc: the render is empty"]
    if not isinstance(text, str) or not text:
        return [f"doc: the {label} text is empty"]
    if text != render:
        return [f"doc: the {label} has drifted from its render"]
    return []


def closing_doc_problems(
        text: Any, tf144_render: Any, appendix: Any) -> list[str]:
    """Judge the completed closing doc, purely:
    it must be exactly TF-144's render (standing
    byte-for-byte as its prefix) followed by the
    filed provenance appendix, nothing else.
    Empty-safe."""
    if not isinstance(tf144_render, str) or not tf144_render:
        return ["closing doc: the TF-144 render is empty"]
    if not isinstance(appendix, str) or not appendix:
        return ["closing doc: the provenance appendix is empty"]
    if not isinstance(text, str) or not text:
        return ["closing doc: the closing doc text is empty"]
    if text != tf144_render + appendix:
        if not text.startswith(tf144_render):
            return ["closing doc: TF-144's render no longer "
                    "stands byte-for-byte at the head of the "
                    "closing doc"]
        return ["closing doc: the closing doc is not TF-144's "
                "render plus the filed provenance appendix"]
    return []


# ── Live measurement helpers (git history, re-derived) ───────────────────────

def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=_REPO_ROOT, capture_output=True,
        text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr[:200]}")
    return result.stdout


def _wing_commit_hashes() -> list[str]:
    out = _git("log", "--format=%H", "--", "research_data")
    return [line for line in out.splitlines() if line]


def _import_stats() -> tuple[int, int]:
    out = _git("show", "--numstat", "--format=", IMPORT_COMMIT)
    files = 0
    insertions = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].startswith("research_data/"):
            files += 1
            insertions += int(parts[0]) if parts[0].isdigit() else 0
    return files, insertions


def _repo_authors() -> set[str]:
    out = _git("log", "--format=%an")
    return {line for line in out.splitlines() if line}


def _bot_wing_commits() -> int:
    # --fixed-strings: the bot's name carries [bot],
    # which git would otherwise read as a regex
    # character class and match nothing.
    out = _git("log", "--fixed-strings",
               f"--author={BOT_AUTHOR}", "--format=%H",
               "--", "research_data")
    return len([line for line in out.splitlines() if line])


def _bot_commits() -> int:
    out = _git("log", "--fixed-strings",
               f"--author={BOT_AUTHOR}", "--format=%H")
    return len([line for line in out.splitlines() if line])


def _tracked_wing_files() -> int:
    out = _git("ls-files", "research_data")
    return len([line for line in out.splitlines() if line])


# ── The live provenance state ────────────────────────────────────────────────

PROVENANCE_EXPECTED: dict[str, Any] = {
    "history_rows": 14,
    "authorship_rows": 7,
    "known_rows": 3,
    "known_none_rows": 1,
    "unknown_rows": 3,
    "wing_commits_total": 3,
    "wing_touch_rows": 3,
    "import_files": 96,
    "import_insertions": 23701,
    "census_baseline_files": 96,
    "tracked_files_now": 97,
    "repo_authors": 3,
    "bot_commits": 2,
    "bot_wing_commits": 0,
    "models_package_present": False,
}


def run_provenance_state() -> dict[str, Any]:
    """Measure the provenance on the live repo:
    git's wing history, the import's size, the
    repo author set, the bot's shares, and the
    census counts. Deterministic: counts, sets
    counted, and booleans only — no commit-count
    for any author whose count a later campaign
    commit would move."""
    try:
        files, insertions = _import_stats()
    except Exception:
        files, insertions = 0, 0
    try:
        wing_commits = len(_wing_commit_hashes())
    except Exception:
        wing_commits = 0
    try:
        authors = len(_repo_authors())
    except Exception:
        authors = 0
    try:
        bot_total, bot_wing = _bot_commits(), _bot_wing_commits()
    except Exception:
        bot_total, bot_wing = 0, -1
    try:
        tracked = _tracked_wing_files()
    except Exception:
        tracked = 0
    return {
        "history_rows": len(HISTORY_ROWS),
        "authorship_rows": len(AUTHORSHIP_ROWS),
        "known_rows": sum(
            1 for r in AUTHORSHIP_ROWS if r["status"] == "known"),
        "known_none_rows": sum(
            1 for r in AUTHORSHIP_ROWS
            if r["status"] == "known_none"),
        "unknown_rows": sum(
            1 for r in AUTHORSHIP_ROWS if r["status"] == "unknown"),
        "wing_commits_total": wing_commits,
        "wing_touch_rows": sum(
            1 for r in HISTORY_ROWS if r["touches_wing"]),
        "import_files": files,
        "import_insertions": insertions,
        "census_baseline_files": 96,
        "tracked_files_now": tracked,
        "repo_authors": authors,
        "bot_commits": bot_total,
        "bot_wing_commits": bot_wing,
        "models_package_present": (
            _REPO_ROOT / "research_data/src/wyrdforge/models"
        ).exists(),
    }


# ── The filed probes ─────────────────────────────────────────────────────────

PROVENANCE_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_estate_passes",
     "kind": "clean",
     "detail": "the filed history rows + authorship rows pass "
               "their validators"},
    {"id": "import_row_wrong_hash_is_named",
     "kind": "import_wrong_hash",
     "detail": "an import row filing a different commit hash "
               "is named"},
    {"id": "unknown_row_claiming_an_answer_is_named",
     "kind": "unknown_with_claim",
     "detail": "an unknown authorship row that files a claim "
               "(an invented answer) is named"},
    {"id": "invented_author_is_named",
     "kind": "invented_author",
     "detail": "an authorship row naming an author outside "
               "the repo's recorded author set is named"},
    {"id": "extra_wing_touch_is_named",
     "kind": "extra_wing_touch",
     "detail": "a history row claiming a wing touch beyond "
               "the import / TF-144 / TF-148 is named"},
    {"id": "slice_row_without_a_commit_is_named",
     "kind": "slice_no_commit",
     "detail": "a TF-139..TF-149 history row with its commit "
               "emptied is named"},
    {"id": "doc_drift_is_named",
     "kind": "doc_drift",
     "detail": "a doc text that drifts from the filed render "
               "(or a closing doc that is not TF-144's render "
               "plus the appendix) is named"},
)


def run_provenance_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in PROVENANCE_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            problems: list[str] = []
            for row in HISTORY_ROWS:
                problems += history_row_problems(row)
            for row in AUTHORSHIP_ROWS:
                problems += authorship_row_problems(row)
            passed = problems == []
        elif kind == "import_wrong_hash":
            row = dict(HISTORY_ROWS[0], commit="0" * 40)
            passed = bool(history_row_problems(row))
        elif kind == "unknown_with_claim":
            row = dict(AUTHORSHIP_ROWS[1],
                       claim="invented: an unnamed helper wrote it")
            passed = bool(authorship_row_problems(row))
        elif kind == "invented_author":
            row = dict(AUTHORSHIP_ROWS[0], author="A. N. Other")
            passed = bool(authorship_row_problems(row))
        elif kind == "extra_wing_touch":
            row = dict(HISTORY_ROWS[2], touches_wing=True)
            passed = bool(history_row_problems(row))
        elif kind == "slice_no_commit":
            row = dict(HISTORY_ROWS[2], commit="")
            passed = bool(history_row_problems(row))
        elif kind == "doc_drift":
            render = render_wing_provenance_markdown()
            named_drift = bool(doc_text_problems(
                render + "drift\n", render, "provenance doc"))
            named_closing = bool(closing_doc_problems(
                "not the closing doc", render, "appendix\n"))
            passed = named_drift and named_closing
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable renders (both docs, verbatim) ───────────────────────────────

def _history_table_lines() -> list[str]:
    lines = [
        "| When | Event | Wing files touched? | Evidence |",
        "|---|---|---|---|",
    ]
    for row in HISTORY_ROWS:
        commit = f" (`{row['commit'][:7]}`)" if row["commit"] else ""
        lines.append(
            f"| {row['date']}{commit} | {row['event']} — "
            f"{row['detail']} | "
            f"{'yes' if row['touches_wing'] else 'no'} | "
            f"{row['evidence']} |")
    return lines


def _authorship_table_lines() -> list[str]:
    lines = [
        "| Question | Status | What is filed |",
        "|---|---|---|",
    ]
    for row in AUTHORSHIP_ROWS:
        if row["status"] == "unknown":
            filed = f"UNKNOWN — {row['unknown_reason']}"
        else:
            filed = row["claim"]
        lines.append(
            f"| {row['question']} | {row['status']} | {filed} |")
    return lines


def _limits_lines() -> list[str]:
    return [
        "- Git-history grade only: every known claim above is "
        "re-derived live from this repo's git history; the "
        "history before the import commit is not in this repo "
        "and is filed UNKNOWN, never filled in.",
        "- Pre-import authorship UNKNOWN: who authored the "
        "wing's files before 2026-03-31, when they were "
        "written, and why `wyrdforge.models` was never "
        "committed are recorded nowhere in this repo — this "
        "slice files the questions and the absence, and "
        "invents no answers, dates or motives.",
        "- The import message '1' is filed verbatim and not "
        "interpreted: git records the character, not a "
        "meaning.",
        "- Provenance here covers the wing's code/docs "
        "history only; no character/persona content is "
        "republished or filed by this slice.",
        "- No README.md anywhere was touched, no wing file "
        "was changed, and the docket is untouched: D-WING-1 "
        "stands CLOSED-BY-VERDICT by TF-149 with TF-016 "
        "totals unchanged (137 — 31 OPEN / 1 "
        "OPEN-ENVIRONMENT / 101 FIXED / 4 CLOSED-BY-VERDICT); "
        "TF-151 owns the panel re-run and TF-152 the Part "
        "VIII gate.",
        "- No live model was read; live behaviour is NOT "
        "PROVEN (TF-173 class), never claimed.",
    ]


def render_wing_closing_appendix_markdown() -> str:
    """Render the provenance section appended to
    the closing doc deterministically — the exact
    text filed below TF-144's render in
    docs/architecture/WING_CLOSING.md."""
    state = run_provenance_state()
    lines: list[str] = []
    lines.append("")
    lines.append("## 9. Provenance preserved (TF-150)")
    lines.append("")
    lines.append(
        "Filed by TF-150 as the completion TF-144 promised "
        "for this doc. GENERATED from the filed data in "
        "`src/thoughtforge/wing_provenance.py` — regenerate "
        "both this doc and "
        "`docs/architecture/WING_PROVENANCE.md` with "
        "`python3 -m thoughtforge.wing_provenance`. Sections "
        "1–8 above stand byte-for-byte as TF-144 filed them; "
        "this section is appended below them. The checker "
        "(`wing_provenance_problems()`) re-derives every "
        "claim live from git history and fails on any drift.")
    lines.append("")
    lines.append(
        f"The wing entered this repo in ONE commit: "
        f"`{IMPORT_COMMIT}` ({IMPORT_DATE}, {IMPORT_AUTHOR}, "
        f"message '1'), {state['import_files']} files / "
        f"{state['import_insertions']} insertions. Since "
        f"then exactly {state['wing_commits_total']} commits "
        "in total touch `research_data/` — the import plus "
        "TF-144's retirement banners and TF-148's fence file. "
        f"Repo-wide git records {state['repo_authors']} "
        f"distinct authors; the bot's wing commits number "
        f"{state['bot_wing_commits']}. Authorship rows: "
        f"{state['known_rows']} known / "
        f"{state['known_none_rows']} known-none / "
        f"{state['unknown_rows']} unknown.")
    lines.append("")
    lines.append("### History")
    lines.append("")
    lines.extend(_history_table_lines())
    lines.append("")
    lines.append("### Authorship")
    lines.append("")
    lines.extend(_authorship_table_lines())
    lines.append("")
    lines.append("### Filed limits")
    lines.append("")
    for line in _limits_lines():
        lines.append(line)
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_wing_provenance_markdown() -> str:
    """Render the standalone provenance doc
    deterministically — the exact text filed as
    docs/architecture/WING_PROVENANCE.md."""
    state = run_provenance_state()
    lines: list[str] = []
    lines.append("# The Wing's Provenance — Authorship and "
                 "History Preserved (TF-150)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/wing_provenance.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.wing_provenance` (the "
        "command also completes "
        "`docs/architecture/WING_CLOSING.md` with this "
        "account as its section 9). The checker "
        "(`wing_provenance_problems()`) re-derives every "
        "claim live from git history and fails on any "
        "drift.")
    lines.append("")
    lines.append(f"Format: `{WING_PROVENANCE_FORMAT}`")
    lines.append(
        f"History rows filed: {state['history_rows']}; "
        f"authorship rows filed: {state['authorship_rows']} "
        f"(known {state['known_rows']} / known-none "
        f"{state['known_none_rows']} / unknown "
        f"{state['unknown_rows']})")
    lines.append(
        f"Commits touching research_data/: "
        f"{state['wing_commits_total']} (the import + TF-144 "
        f"+ TF-148)")
    lines.append(f"Probes: {len(PROVENANCE_PROBES)}/"
                 f"{len(PROVENANCE_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was "
        "built: `git log -- research_data` shows the wing "
        "entered the repo in ONE commit, `6ba2df7` "
        "(2026-03-31, Volmarr Wyrd, message '1'), 96 files "
        "and 23,701 insertions, and was later touched only "
        "by TF-144 (retirement banners) and TF-148 "
        "(RESEARCH_FENCE.md). Repo-wide git records "
        "exactly three authors — Yrsa Freydisdottir (the "
        "campaign), Volmarr Wyrd, and "
        "google-labs-jules[bot] (2 commits, neither "
        "touching research_data/). TF-006's census is the "
        "wing's first internal account. Who authored the "
        "wing's files before the import, when they were "
        "written, and why wyrdforge.models was never "
        "committed are recorded nowhere in this repo and "
        "are filed UNKNOWN below — ORIGINS-shaped honesty: "
        "the evidence is recorded exactly, and the gaps "
        "are filed as gaps, never filled by invention.")
    lines.append("")
    lines.append("## 2. The wing's recorded history")
    lines.append("")
    lines.extend(_history_table_lines())
    lines.append("")
    lines.append("## 3. Authorship — known and unknown")
    lines.append("")
    lines.extend(_authorship_table_lines())
    lines.append("")
    lines.append("## 4. The closing doc, completed")
    lines.append("")
    lines.append(
        "TF-144 filed that TF-150 completes the wing's "
        "provenance in the closing doc. It now stands "
        "there as section 9 of "
        "`docs/architecture/WING_CLOSING.md`, appended "
        "below TF-144's render, which stands byte-for-byte "
        "above it — the completion is checked live as "
        "exactly TF-144's render plus this slice's "
        "appendix. TF-144's checker and doc test were "
        "updated minimally in the open to accept exactly "
        "that completed shape, the estate's established "
        "in-the-open neighbour update.")
    lines.append("")
    lines.append("## 5. Docket")
    lines.append("")
    lines.append(
        "None. D-WING-1 is already CLOSED-BY-VERDICT by "
        "TF-149, whose closure files TF-150/TF-152 as the "
        "owners of provenance/gate, not that entry. This "
        "slice adds no docket entry, reopens nothing, and "
        "TF-016 totals do not move: 137 entries — 31 OPEN "
        "/ 1 OPEN-ENVIRONMENT / 101 FIXED / 4 "
        "CLOSED-BY-VERDICT.")
    lines.append("")
    lines.append("## 6. The probes")
    lines.append("")
    for probe in PROVENANCE_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 7. Filed limits")
    lines.append("")
    for line in _limits_lines():
        lines.append(line)
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_provenance_docs() -> tuple[Path, Path]:
    """Write both renders to their filed homes:
    the standalone provenance doc, and the
    closing doc completed as TF-144's render
    plus the provenance appendix."""
    from thoughtforge.wing_retirements import (
        render_wing_closing_markdown,
    )

    provenance = _REPO_ROOT / WING_PROVENANCE_DOC
    provenance.write_text(
        render_wing_provenance_markdown(), encoding="utf-8")
    closing = _REPO_ROOT / WING_CLOSING_DOC
    closing.write_text(
        render_wing_closing_markdown()
        + render_wing_closing_appendix_markdown(),
        encoding="utf-8")
    return provenance, closing


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.wing_provenance`."""
    provenance, closing = write_provenance_docs()
    print(f"wing provenance doc regenerated: -> "
          f"{WING_PROVENANCE_DOC} ({provenance})")
    print(f"wing closing doc completed: -> "
          f"{WING_CLOSING_DOC} ({closing})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def _git_provenance_problems() -> list[str]:
    """Re-derive the git-history claims live:
    the wing's commit list, the import's author /
    date / message / size, the slice commits, the
    author set and the bot's shares."""
    problems: list[str] = []
    try:
        wing = _wing_commit_hashes()
    except Exception as exc:
        return [f"git: the wing history cannot be read: {exc}"]
    expected_wing = [TF148_COMMIT, TF144_COMMIT, IMPORT_COMMIT]
    if wing != expected_wing:
        problems.append(
            f"git: commits touching research_data/ are "
            f"{[h[:7] for h in wing]}, not the filed "
            "[TF-148, TF-144, import] — the wing's recorded "
            "history has changed and this filing must be "
            "updated")
    try:
        head = _git(
            "show", "-s", "--format=%an|%ad|%s",
            "--date=short", IMPORT_COMMIT).strip()
    except Exception as exc:
        return problems + [f"git: the import commit cannot "
                           f"be read: {exc}"]
    if head != f"{IMPORT_AUTHOR}|{IMPORT_DATE}|{IMPORT_MESSAGE}":
        problems.append(
            f"git: the import commit reads {head!r}, not the "
            f"filed {IMPORT_AUTHOR}|{IMPORT_DATE}|"
            f"{IMPORT_MESSAGE!r}")
    try:
        authors = _repo_authors()
    except Exception as exc:
        authors = set()
        problems.append(f"git: the author set cannot be "
                        f"read: {exc}")
    if authors != set(REPO_AUTHORS):
        problems.append(
            f"git: the repo author set is {sorted(authors)}, "
            f"not the filed {sorted(REPO_AUTHORS)}")
    # Every Part VIII slice row's commit resolves to
    # a commit whose subject names that slice.
    for row in HISTORY_ROWS:
        commit = row["commit"]
        if not commit or row["id"] in (
                "import_commit", "tf144_retirements",
                "tf148_hygiene"):
            continue
        try:
            subject = _git(
                "show", "-s", "--format=%s", commit).strip()
        except Exception:
            problems.append(
                f"git: history {row['id']} files commit "
                f"{commit}, which does not resolve")
            continue
        if not subject.startswith(f"{row['slice']}:"):
            problems.append(
                f"git: history {row['id']} files commit "
                f"{commit} whose subject does not name "
                f"{row['slice']}")
    return problems


def wing_provenance_problems() -> list[str]:
    """Re-derive TF-150's claims from the live
    repo; name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    history_ids = [r["id"] for r in HISTORY_ROWS]
    if len(history_ids) != len(set(history_ids)) \
            or len(HISTORY_ROWS) != 14:
        problems.append("the HISTORY_ROWS filing is not "
                        "fourteen unique events")
    for row in HISTORY_ROWS:
        problems.extend(history_row_problems(row))
    authorship_ids = [r["id"] for r in AUTHORSHIP_ROWS]
    if len(authorship_ids) != len(set(authorship_ids)) \
            or len(AUTHORSHIP_ROWS) != 7:
        problems.append("the AUTHORSHIP_ROWS filing is not "
                        "seven unique questions")
    for row in AUTHORSHIP_ROWS:
        problems.extend(authorship_row_problems(row))

    # Git history, re-derived live.
    problems.extend(_git_provenance_problems())

    # The census counts TF-148 files still stand.
    from thoughtforge.research_hygiene import (
        research_hygiene_problems,
        run_research_hygiene_state,
    )

    hygiene_state = run_research_hygiene_state()
    if hygiene_state.get("baseline_tracked_files") != 96 \
            or hygiene_state.get("tracked_files_with_fence") != 97:
        problems.append(
            "census: TF-148's census no longer files 96 "
            "baseline + fence = 97 tracked wing files")

    # The measured state matches its filing.
    problems.extend(provenance_state_problems(
        run_provenance_state(), PROVENANCE_EXPECTED))

    # Neighbouring checkers clean live.
    from thoughtforge.research_hygiene import (
        research_hygiene_problems as _hygiene,
    )
    from thoughtforge.wing_retirements import (
        wing_retirements_problems,
    )
    from thoughtforge.wing_tests import wing_tests_problems

    for name, checker in (
            ("research_hygiene", _hygiene),
            ("wing_retirements", wing_retirements_problems),
            ("wing_tests", wing_tests_problems)):
        neighbour = checker()
        if neighbour:
            problems.append(
                f"neighbour: {name}_problems() is not clean: "
                f"{neighbour[:3]}")

    # The docket stands exactly as TF-149 left it:
    # totals unchanged, D-WING-1 closed, no TF-150
    # entry added by this slice.
    import re

    docket_text = _DOCKET.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", docket_text)
    if "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, 101 FIXED" \
            not in normalized:
        problems.append(
            "docket: the header totals no longer file 137 "
            "entries — 31 OPEN / 1 OPEN-ENVIRONMENT / 101 "
            "FIXED; TF-150 must not move them")
    wing_line = next(
        (line for line in docket_text.splitlines()
         if line.startswith("| D-WING-1 |")), "")
    if "**CLOSED-BY-VERDICT**" not in wing_line:
        problems.append(
            "docket: D-WING-1 is not CLOSED-BY-VERDICT as "
            "TF-149 left it")

    # The probes all behave.
    probes = run_provenance_probes()
    if len(probes) != len(PROVENANCE_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the provenance probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # Both filed docs match their renders verbatim,
    # the closing doc as TF-144's render + appendix.
    from thoughtforge.wing_retirements import (
        render_wing_closing_markdown,
    )

    doc = _REPO_ROOT / WING_PROVENANCE_DOC
    if not doc.exists():
        problems.append(
            f"the provenance doc {WING_PROVENANCE_DOC} is missing")
    else:
        problems.extend(
            f"doc: {p}" for p in doc_text_problems(
                doc.read_text(encoding="utf-8"),
                render_wing_provenance_markdown(),
                "provenance doc"))
    closing = _REPO_ROOT / WING_CLOSING_DOC
    if not closing.exists():
        problems.append(
            f"the closing doc {WING_CLOSING_DOC} is missing")
    else:
        problems.extend(
            f"closing doc: {p}" for p in closing_doc_problems(
                closing.read_text(encoding="utf-8"),
                render_wing_closing_markdown(),
                render_wing_closing_appendix_markdown()))
    return problems


def wing_provenance_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_provenance_probes()
    return {
        "format": WING_PROVENANCE_FORMAT,
        "doc": WING_PROVENANCE_DOC,
        "closing_doc": WING_CLOSING_DOC,
        "history": [dict(r) for r in HISTORY_ROWS],
        "authorship": [dict(r) for r in AUTHORSHIP_ROWS],
        "state": run_provenance_state(),
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "AUTHORSHIP_ROWS",
    "AUTHORSHIP_STATUSES",
    "BOT_AUTHOR",
    "HISTORY_ROWS",
    "IMPORT_AUTHOR",
    "IMPORT_COMMIT",
    "IMPORT_DATE",
    "IMPORT_FILES",
    "IMPORT_INSERTIONS",
    "IMPORT_MESSAGE",
    "PROVENANCE_EXPECTED",
    "PROVENANCE_PROBES",
    "REPO_AUTHORS",
    "TF144_COMMIT",
    "TF148_COMMIT",
    "WING_CLOSING_DOC",
    "WING_PROVENANCE_DOC",
    "WING_PROVENANCE_FORMAT",
    "authorship_row_problems",
    "closing_doc_problems",
    "doc_text_problems",
    "history_row_problems",
    "main",
    "provenance_state_problems",
    "render_wing_closing_appendix_markdown",
    "render_wing_provenance_markdown",
    "run_provenance_probes",
    "run_provenance_state",
    "wing_provenance_as_data",
    "wing_provenance_problems",
    "write_provenance_docs",
]

if __name__ == "__main__":
    raise SystemExit(main())
