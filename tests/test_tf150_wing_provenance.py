"""
TF-150 — Provenance preserved: the wing's
authorship/history recorded in the closing doc
(ORIGINS-shaped honesty).

The proof lives as verifiable data in
``src/thoughtforge/wing_provenance.py``
(``HISTORY_ROWS``, ``AUTHORSHIP_ROWS``,
``PROVENANCE_PROBES``, checked by
``wing_provenance_problems()``); the doc
``docs/architecture/WING_PROVENANCE.md`` is
filed verbatim as the module's render, and the
provenance account completes
``docs/architecture/WING_CLOSING.md`` as its
section 9, below TF-144's render standing
byte-for-byte above it.

Pinned here:

  * the wing entered the repo in ONE commit
    (6ba2df7, 2026-03-31, Volmarr Wyrd, message
    '1', 96 files / 23,701 insertions), re-derived
    live from git,
  * exactly 3 commits touch research_data/ —
    the import, TF-144 and TF-148,
  * repo-wide exactly 3 authors; the bot's 2
    commits touch no wing file (known_none),
  * 14 history rows (import + TF-006 census +
    TF-139..TF-150) and 7 authorship rows
    (3 known / 1 known-none / 3 unknown — the
    pre-import authorship, the writing dates
    and the models omission's reason filed
    UNKNOWN, with no claim and no author named),
  * the closing doc is exactly TF-144's render
    plus TF-150's appendix (TF-144's checker +
    doc test updated in the open for exactly
    that shape),
  * no docket touch: D-WING-1 stands
    CLOSED-BY-VERDICT, TF-016 totals unchanged
    137 — 32 OPEN / 1 OPEN-ENVIRONMENT /
    100 FIXED / 4 CLOSED-BY-VERDICT,
  * the probes: 7/7 against the pure validators,
  * the provenance doc matches its render
    verbatim.

All fixtures synthetic (TF-017); no canon —
provenance here is the wing's code/docs history
only, no persona content filed.
Proof grade: GIT-HISTORY / FIXTURE-STATIC — no
live model was read, never claimed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from thoughtforge.wing_provenance import (
    AUTHORSHIP_ROWS,
    AUTHORSHIP_STATUSES,
    BOT_AUTHOR,
    HISTORY_ROWS,
    IMPORT_AUTHOR,
    IMPORT_COMMIT,
    IMPORT_DATE,
    IMPORT_MESSAGE,
    PROVENANCE_EXPECTED,
    PROVENANCE_PROBES,
    REPO_AUTHORS,
    TF144_COMMIT,
    TF148_COMMIT,
    WING_CLOSING_DOC,
    WING_PROVENANCE_DOC,
    WING_PROVENANCE_FORMAT,
    authorship_row_problems,
    closing_doc_problems,
    doc_text_problems,
    history_row_problems,
    provenance_state_problems,
    render_wing_closing_appendix_markdown,
    render_wing_provenance_markdown,
    run_provenance_probes,
    run_provenance_state,
    wing_provenance_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True,
        text=True, timeout=30, check=True).stdout


# ── The filing ───────────────────────────────────────────────────────────────

class TestFiling:
    def test_problems_empty(self) -> None:
        assert wing_provenance_problems() == []

    def test_format_is_versioned(self) -> None:
        assert WING_PROVENANCE_FORMAT == \
            "thoughtforge-wing-provenance/1"

    def test_history_rows_are_fourteen_unique(self) -> None:
        assert len(HISTORY_ROWS) == 14
        assert len({r["id"] for r in HISTORY_ROWS}) == 14
        for row in HISTORY_ROWS:
            assert history_row_problems(row) == []
        slices = [r["slice"] for r in HISTORY_ROWS if r["slice"]]
        assert slices == ["TF-006"] + [
            f"TF-{n}" for n in range(139, 151)]

    def test_exactly_three_rows_touch_the_wing(self) -> None:
        touched = [r["id"] for r in HISTORY_ROWS
                   if r["touches_wing"]]
        assert touched == [
            "import_commit", "tf144_retirements", "tf148_hygiene"]

    def test_authorship_rows_are_seven_with_honest_split(
            self) -> None:
        assert len(AUTHORSHIP_ROWS) == 7
        for row in AUTHORSHIP_ROWS:
            assert authorship_row_problems(row) == []
            assert row["status"] in AUTHORSHIP_STATUSES
        by_status: dict[str, int] = {}
        for row in AUTHORSHIP_ROWS:
            by_status[row["status"]] = \
                by_status.get(row["status"], 0) + 1
        assert by_status == {
            "known": 3, "known_none": 1, "unknown": 3}

    def test_unknown_rows_file_no_claim_and_no_author(self) -> None:
        unknown = [r for r in AUTHORSHIP_ROWS
                   if r["status"] == "unknown"]
        assert {r["id"] for r in unknown} == {
            "who_authored_the_files_before_import",
            "when_were_the_files_written",
            "why_models_never_committed"}
        for row in unknown:
            assert row["claim"] == "" and row["author"] == ""
            assert row["unknown_reason"]

    def test_no_row_names_an_author_outside_the_repo_set(
            self) -> None:
        for row in AUTHORSHIP_ROWS:
            assert not row["author"] or \
                row["author"] in REPO_AUTHORS
        assert set(REPO_AUTHORS) == {
            "Volmarr Wyrd", "Yrsa Freydisdottir", BOT_AUTHOR}


# ── Git history, live ────────────────────────────────────────────────────────

class TestGitHistoryLive:
    def test_wing_history_is_exactly_three_commits(self) -> None:
        hashes = _git(
            "log", "--format=%H", "--", "research_data").split()
        assert hashes == [TF148_COMMIT, TF144_COMMIT,
                          IMPORT_COMMIT]

    def test_import_commit_is_as_filed(self) -> None:
        head = _git("show", "-s", "--format=%an|%ad|%s",
                    "--date=short", IMPORT_COMMIT).strip()
        assert head == \
            f"{IMPORT_AUTHOR}|{IMPORT_DATE}|{IMPORT_MESSAGE}"
        numstat = _git("show", "--numstat", "--format=",
                       IMPORT_COMMIT)
        files = [line for line in numstat.splitlines()
                 if line.split("\t")[-1].startswith(
                     "research_data/")]
        assert len(files) == 96
        assert sum(int(line.split("\t")[0])
                   for line in files) == 23701

    def test_repo_author_set_is_as_filed(self) -> None:
        authors = {line for line in
                   _git("log", "--format=%an").splitlines()
                   if line}
        assert authors == set(REPO_AUTHORS)

    def test_bot_touched_no_wing_file(self) -> None:
        # --fixed-strings: [bot] is a regex class to git.
        wing = _git("log", "--fixed-strings",
                    f"--author={BOT_AUTHOR}",
                    "--format=%H", "--", "research_data")
        assert wing.strip() == ""
        total = _git("log", "--fixed-strings",
                     f"--author={BOT_AUTHOR}",
                     "--format=%H").split()
        assert len(total) == 2
        row = next(r for r in AUTHORSHIP_ROWS
                   if r["id"] == "bot_authorship_in_the_wing")
        assert row["status"] == "known_none"

    def test_slice_commits_resolve_and_name_their_slice(
            self) -> None:
        for row in HISTORY_ROWS:
            if not row["commit"] or row["id"] in (
                    "import_commit", "tf144_retirements",
                    "tf148_hygiene"):
                continue
            subject = _git("show", "-s", "--format=%s",
                           row["commit"]).strip()
            assert subject.startswith(f"{row['slice']}:")


# ── State, probes, validators ────────────────────────────────────────────────

class TestStateAndProbes:
    def test_state_matches_its_filing(self) -> None:
        state = run_provenance_state()
        assert state == PROVENANCE_EXPECTED
        assert provenance_state_problems(
            state, PROVENANCE_EXPECTED) == []

    def test_probes_all_pass(self) -> None:
        probes = run_provenance_probes()
        assert len(probes) == len(PROVENANCE_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert history_row_problems(None)
        assert authorship_row_problems(None)
        assert provenance_state_problems(None, None)
        assert doc_text_problems(None, None, "x")
        assert closing_doc_problems(None, None, None)

    def test_each_probe_shape_is_named_by_the_validator(
            self) -> None:
        row = dict(HISTORY_ROWS[0], commit="0" * 40)
        assert history_row_problems(row)
        row = dict(AUTHORSHIP_ROWS[1], claim="invented")
        assert any("unknown" in p
                   for p in authorship_row_problems(row))
        row = dict(AUTHORSHIP_ROWS[0], author="A. N. Other")
        assert any("invented" in p or "not in" in p
                   for p in authorship_row_problems(row))
        row = dict(HISTORY_ROWS[2], touches_wing=True)
        assert history_row_problems(row)
        row = dict(HISTORY_ROWS[2], commit="")
        assert history_row_problems(row)
        assert doc_text_problems(
            render_wing_provenance_markdown() + "x",
            render_wing_provenance_markdown(), "provenance doc")
        assert closing_doc_problems(
            "not the closing doc", "render\n", "appendix\n")

    def test_neighbouring_checkers_still_clean(self) -> None:
        from thoughtforge.research_hygiene import (
            research_hygiene_problems,
        )
        from thoughtforge.wing_retirements import (
            wing_retirements_problems,
        )
        from thoughtforge.wing_tests import wing_tests_problems

        assert research_hygiene_problems() == []
        assert wing_retirements_problems() == []
        assert wing_tests_problems() == []


# ── Docs + docket ────────────────────────────────────────────────────────────

class TestDocsAndDocket:
    def test_provenance_doc_matches_render(self) -> None:
        assert _read(WING_PROVENANCE_DOC) == \
            render_wing_provenance_markdown()

    def test_closing_doc_is_tf144_render_plus_appendix(
            self) -> None:
        from thoughtforge.wing_retirements import (
            render_wing_closing_markdown,
        )

        closing = _read(WING_CLOSING_DOC)
        tf144 = render_wing_closing_markdown()
        assert closing.startswith(tf144)
        assert closing == tf144 + \
            render_wing_closing_appendix_markdown()
        assert closing_doc_problems(
            closing, tf144,
            render_wing_closing_appendix_markdown()) == []

    def test_closing_doc_sections_1_to_8_byte_stable(
            self) -> None:
        from thoughtforge.wing_retirements import (
            render_wing_closing_markdown,
        )

        closing = _read(WING_CLOSING_DOC)
        prefix = closing[
            :len(render_wing_closing_markdown())]
        assert prefix == render_wing_closing_markdown()
        assert "## 9. Provenance preserved (TF-150)" in closing

    def test_docs_file_the_honesty_shape(self) -> None:
        for doc in (_read(WING_PROVENANCE_DOC),
                    _read(WING_CLOSING_DOC)):
            assert "6ba2df7" in doc
            assert "UNKNOWN" in doc
            assert "NOT PROVEN" in doc
            assert "TF-006" in doc

    def test_docket_untouched_totals_unchanged(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 32 OPEN, 1 OPEN-ENVIRONMENT, " \
            "100 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT (D-CARD-2 by TF-086's " \
            "Part IV gate verdict; D-WING-1 by TF-149)" in \
            normalized
        wing_line = next(
            line for line in docket.splitlines()
            if line.startswith("| D-WING-1 |"))
        assert "**CLOSED-BY-VERDICT**" in wing_line
        assert "TF-150/TF-152 own provenance/gate" in wing_line

    def test_fixtures_are_synthetic_only(self) -> None:
        # This slice files git history and questions only:
        # no canon / persona content is republished.
        for row in HISTORY_ROWS + AUTHORSHIP_ROWS:
            assert "veyrunn" not in str(row).lower()
