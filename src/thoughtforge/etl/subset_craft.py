"""
Subset craft, as data (TF-110).

Roadmap: "Subset craft: profile-sized subsets
(phone_low's knowledge must fit phone_low) built by rule
+ measured."

TF-105 mapped the subset stage and fixed D-SUBSET-1.
What it mapped was a builder whose only sizing rule was
an entity COUNT: ``_PROFILE_ENTITY_LIMITS`` (pi_zero
50,000 / phone_low 80,000 / pi_5 200,000 ...), with a
profile-JSON ``subset_entity_limit`` override no shipped
profile carries. TF-110's probes measured what that
means in bytes — the thing a device actually feels:

  * **No byte budget existed at all (D-SUBSET-2).** Only
    pi_zero's profile declares any knowledge size at all
    (``deployment.knowledge_subset_max_gb`` = 0.5 — its
    entire RAM, filed as declared, not endorsed); the
    other five profiles declare nothing, the builder
    never compared a build's size to anything, and its
    docstring claimed output sizes (~50MB / ~80MB /
    ~200MB) measured nowhere. Worse, the reference-chunk
    cap floored at 1,000 rows regardless of the entity
    limit, so a phone_low build limited to 5 entities
    still copied all 50 synthetic chunks (1.24 MB of
    chunk text against 5 small entities) and reported
    success. An entity count is not a size: one entity
    with a 20 KB description outweighs a thousand stubs.
  * **A re-build was not the subset it claimed
    (D-SUBSET-3).** The builder connected to an existing
    output file and INSERT OR IGNOREd into it: re-building
    at limit 2 onto a limit-5 output reported 2 entities
    while the file still held 5 entities and all 50
    chunks. Fixed in ``subset.py``: every build starts
    from no file, and building over the source itself is
    refused by name.
  * **Selection order was undefined on ties
    (D-SUBSET-4).** ``ORDER BY popularity_score DESC``
    alone leaves tied rows to SQLite's unspecified scan
    order, so the same source could yield different
    subsets. Fixed in ``subset.py``: ties break by qid.

The craft, filed here as data:

  * ``budget_for_profile`` — one profile's byte budget,
    resolved LIVE from its shipped JSON: a declared
    ``deployment.knowledge_subset_max_gb`` wins where
    present (pi_zero only, today); otherwise the budget
    is DERIVED by the campaign rule stated in
    ``SUBSET_RULES`` — at most ``RAM_BUDGET_FRACTION``
    (1/4) of the profile's declared RAM, in bytes. The
    derivation is a campaign rule, never presented as a
    profile fact; the budget's ``budget_source`` says
    which road produced it, every time.
  * ``measure_subset`` — a built subset measured, not
    estimated: the file's bytes (db + -wal + -shm,
    named apart), per-table row counts, and per-table
    bytes from SQLite's own ``dbstat`` (stated ``None``
    where dbstat is unavailable — never estimated).
  * ``build_profile_subset`` — the craft loop: build at
    the profile's entity limit (builder's live table),
    measure, and while the measured size exceeds the
    budget, trim the entity limit by the measured
    overshoot (``limit * budget / size``, floored, at
    least one step down) and rebuild fresh. Every
    attempt is recorded. The loop ends in exactly one
    of two honest states: a subset whose MEASURED size
    fits the budget, or ``success=False`` with the
    smallest attempt's measured size named (a store
    whose schema + FTS alone exceeds a tiny budget
    cannot be crafted by trimming entities, and the
    result says so). An unlimited profile (server_gpu,
    entity limit 0) is built once, measured, and its
    fit reported — never trimmed. One trap the loop
    itself must honour: the builder reads an explicit
    limit of 0 as UNLIMITED, so the zero-entity attempt
    is built by the real builder from an empty schema
    store, never by passing it 0 (the craft's first
    version looped on exactly that collision; TF-110's
    impossible-budget test caught it).
  * Selection stays the builder's rule, now
    deterministic: popularity descending, qid ascending
    on ties; statements ride their kept subjects;
    reference chunks ride at the filed proportion
    (entity_limit // 10, no floor after D-SUBSET-2).

TF-105's living map is updated in the open
(``SUBSET_FACTS`` gains the byte-budget facts; the
chunk-cap line loses its floor). D-PROF-1's last filed
share — the deployment section's subset claim — is
discharged here: pi_zero's ``knowledge_subset`` flag
and ``knowledge_subset_max_gb`` are now read live by
this module (the checker below pins exactly that).

Proof grade: FIXTURE — synthetic stores through the
real builder on this VM, budgets in bytes measured off
the files built. A real dump's subset size, and a real
phone's storage behaviour, are NOT PROVEN, never
claimed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thoughtforge.utils.config import (
    list_hardware_profiles,
    load_hardware_profile,
)

SUBSET_CRAFT_FORMAT = "thoughtforge-subset-craft/1"

#: The campaign rule for profiles that declare no
#: knowledge byte budget: a subset may occupy at most
#: this fraction of the profile's declared RAM, in
#: bytes. A rule this campaign sets — never a profile
#: fact, and ``budget_source`` says so wherever used.
RAM_BUDGET_FRACTION: float = 0.25

SUBSET_RULES: tuple[dict[str, Any], ...] = (
    {"id": "budget_is_bytes_not_entities",
     "rule": "a subset fits its profile by MEASURED file bytes against a byte budget; an entity count is a selection limit, never a size claim"},
    {"id": "declared_budget_wins",
     "rule": "a profile's declared deployment.knowledge_subset_max_gb is its budget where present (today: pi_zero only); the value is filed as declared, not endorsed"},
    {"id": "derived_budget_is_a_campaign_rule",
     "rule": "with no declared budget, budget_bytes = RAM_BUDGET_FRACTION (0.25) of the profile's declared RAM; budget_source names the derivation every time"},
    {"id": "selection_is_deterministic",
     "rule": "entities are selected by popularity_score DESC, qid ASC — ties never decide membership by scan order (D-SUBSET-4)"},
    {"id": "chunks_ride_the_proportion",
     "rule": "reference chunks copy at entity_limit // 10 with NO floor (D-SUBSET-2); statements ride kept subjects only"},
    {"id": "every_build_is_fresh",
     "rule": "a build starts from no file and refuses the source itself as its output (D-SUBSET-3); a re-build's file holds exactly that build"},
    {"id": "trim_by_measured_overshoot",
     "rule": "while a measured build exceeds its budget, the entity limit is cut to floor(copied * budget / measured) (at least one below the count the last build copied) and the build is redone fresh; every attempt is recorded with its measured bytes"},
    {"id": "unfittable_is_a_failure_not_a_claim",
     "rule": "if the smallest build (0 entities) still measures over budget, the result is success=False naming the measured size — a subset is never reported as fitting when it does not"},
    {"id": "unlimited_profiles_are_measured_not_trimmed",
     "rule": "an entity limit of 0 (server_gpu) builds once at full size; its measured fit against the derived budget is reported, never enforced by trimming"},
    {"id": "measurement_is_the_files_own",
     "rule": "sizes are the built file's stat bytes (db + -wal + -shm named apart) after the builder's VACUUM; per-table bytes come from SQLite dbstat or are stated None — never estimated"},
)


# ── The budget, resolved live ─────────────────────────────────────────────


@dataclass(frozen=True)
class SubsetBudget:
    """One profile's subset budget: entity limit + byte
    budget, with the byte budget's source named."""

    profile_id: str
    ram_gb: float
    entity_limit: int
    declared_budget_bytes: int | None
    budget_bytes: int
    budget_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "ram_gb": self.ram_gb,
            "entity_limit": self.entity_limit,
            "declared_budget_bytes": self.declared_budget_bytes,
            "budget_bytes": self.budget_bytes,
            "budget_source": self.budget_source,
        }


def budget_for_profile(profile_id: str) -> SubsetBudget:
    """Resolve ``profile_id``'s subset budget live from
    its shipped JSON + the builder's live limit table.

    Unknown profiles raise ValueError naming the valid
    ids — a budget is never silently another profile's.
    """
    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile for subset budget: {profile_id!r} "
            f"(valid: {valid})"
        )
    from thoughtforge.etl.subset import _PROFILE_ENTITY_LIMITS

    profile = load_hardware_profile(profile_id)
    ram_gb = float(profile["hardware"]["ram_gb"])
    deployment = profile.get("deployment", {}) or {}
    declared = deployment.get("knowledge_subset_max_gb")
    declared_bytes = (
        int(float(declared) * 1024 ** 3) if declared is not None else None
    )
    if declared_bytes is not None:
        budget_bytes = declared_bytes
        source = "declared: deployment.knowledge_subset_max_gb"
    else:
        budget_bytes = int(ram_gb * 1024 ** 3 * RAM_BUDGET_FRACTION)
        source = (
            f"derived: {RAM_BUDGET_FRACTION} x declared RAM "
            "(campaign rule — no budget declared by the profile)"
        )
    return SubsetBudget(
        profile_id=profile_id,
        ram_gb=ram_gb,
        entity_limit=int(_PROFILE_ENTITY_LIMITS[profile_id]),
        declared_budget_bytes=declared_bytes,
        budget_bytes=budget_bytes,
        budget_source=source,
    )


# ── Measurement ────────────────────────────────────────────────────────────

#: Tables whose rows + bytes a measurement reports, in
#: schema order (TF-105's subset copies three of them;
#: the rest are measured so an empty claim is checked,
#: not assumed).
MEASURED_TABLES: tuple[str, ...] = (
    "entities", "statements", "labels", "embeddings",
    "conceptnet_edges", "geonames_places", "dbpedia_entities",
    "reference_chunks", "reference_embeddings", "ingestion_log",
)


def measure_subset(db_path: Path | str) -> dict[str, Any]:
    """Measure a built subset file: total bytes (db +
    sidecars, named apart), per-table row counts, and
    per-table bytes from dbstat (None where dbstat is
    unavailable on a table — stated, never estimated).
    """
    path = Path(db_path)
    sidecars = {}
    total = 0
    for label, candidate in (
        ("db", path),
        ("wal", Path(str(path) + "-wal")),
        ("shm", Path(str(path) + "-shm")),
    ):
        size = candidate.stat().st_size if candidate.exists() else 0
        sidecars[label] = size
        total += size
    counts: dict[str, int] = {}
    table_bytes: dict[str, int | None] = {}
    conn = sqlite3.connect(str(path))
    try:
        try:
            dbstat = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name"
                )
            }
        except sqlite3.OperationalError:
            dbstat = {}
        for table in MEASURED_TABLES:
            try:
                counts[table] = conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = -1
            table_bytes[table] = dbstat.get(table)
    finally:
        conn.close()
    return {
        "path": str(path),
        "bytes": total,
        "file_bytes": sidecars,
        "counts": counts,
        "table_bytes": table_bytes,
    }


# ── The craft loop ─────────────────────────────────────────────────────────


@dataclass
class SubsetCraftResult:
    """The outcome of one profile-subset craft: the
    budget it was judged against, every attempt's
    measured bytes, and the final build's own numbers."""

    profile_id: str
    budget: dict[str, Any]
    success: bool
    fits_budget: bool
    attempts: list[dict[str, Any]] = field(default_factory=list)
    entities_copied: int = 0
    statements_copied: int = 0
    reference_chunks_copied: int = 0
    measured_bytes: int | None = None
    measurement: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SUBSET_CRAFT_FORMAT,
            "profile_id": self.profile_id,
            "budget": dict(self.budget),
            "success": self.success,
            "fits_budget": self.fits_budget,
            "attempts": [dict(a) for a in self.attempts],
            "entities_copied": self.entities_copied,
            "statements_copied": self.statements_copied,
            "reference_chunks_copied": self.reference_chunks_copied,
            "measured_bytes": self.measured_bytes,
            "measurement": self.measurement,
            "notes": list(self.notes),
        }


def build_profile_subset(
    source_db: Path | str,
    output_path: Path | str,
    profile_id: str,
    *,
    max_bytes: int | None = None,
    max_entities: int | None = None,
) -> SubsetCraftResult:
    """Build ``profile_id``'s subset of ``source_db`` by
    the filed rules and measure it against the budget.

    ``max_bytes`` / ``max_entities`` override the
    resolved budget / limit (fixtures + operator use);
    an override rides the result's budget dict, named
    as an override, never as the profile's own figure.
    """
    from thoughtforge.etl.subset import EdgeSubsetBuilder

    budget = budget_for_profile(profile_id)
    budget_dict = budget.to_dict()
    byte_budget = budget.budget_bytes
    if max_bytes is not None:
        byte_budget = int(max_bytes)
        budget_dict["budget_bytes"] = byte_budget
        budget_dict["budget_source"] = (
            f"override: max_bytes={byte_budget} "
            f"(profile's own: {budget.budget_source})"
        )
    limit = budget.entity_limit
    if max_entities is not None:
        limit = int(max_entities)
        budget_dict["entity_limit"] = limit
        budget_dict["entity_limit_source"] = "override: max_entities"
    unlimited = limit == 0

    builder = EdgeSubsetBuilder()
    attempts: list[dict[str, Any]] = []
    notes: list[str] = []
    current = limit
    while True:
        if current == 0 and not unlimited:
            # The builder's own convention reads an
            # explicit limit of 0 as UNLIMITED, so a
            # zero-entity attempt cannot be asked of it
            # directly (that collision looped the first
            # version of this craft forever, caught by
            # TF-110's own impossible-budget test). The
            # smallest build is the real builder run on
            # an empty schema store: the same schema
            # clone + FTS rebuild, zero rows.
            from thoughtforge.etl.schema import initialize_schema

            empty_source = Path(output_path).parent / ".subset_craft_empty.db"
            initialize_schema(empty_source)
            try:
                built = builder.build(
                    source_db=empty_source,
                    output_path=Path(output_path),
                    profile_id=profile_id,
                    max_entities=1,
                )
            finally:
                empty_source.unlink(missing_ok=True)
        else:
            built = builder.build(
                source_db=Path(source_db),
                output_path=Path(output_path),
                profile_id=profile_id,
                max_entities=current,
            )
        measurement = measure_subset(output_path)
        attempts.append({
            "entity_limit": current,
            "entities_copied": built.entities_copied,
            "measured_bytes": measurement["bytes"],
            "builder_success": built.success,
        })
        fits = measurement["bytes"] <= byte_budget
        if fits or unlimited:
            return SubsetCraftResult(
                profile_id=profile_id,
                budget=budget_dict,
                success=bool(built.success),
                fits_budget=fits,
                attempts=attempts,
                entities_copied=built.entities_copied,
                statements_copied=built.statements_copied,
                reference_chunks_copied=built.reference_chunks_copied,
                measured_bytes=measurement["bytes"],
                measurement=measurement,
                notes=list(built.notes),
            )
        if built.entities_copied == 0:
            notes.append(
                f"smallest build (0 entities) measures "
                f"{measurement['bytes']} bytes > budget {byte_budget} — "
                "no entity trim can craft this subset"
            )
            return SubsetCraftResult(
                profile_id=profile_id,
                budget=budget_dict,
                success=False,
                fits_budget=False,
                attempts=attempts,
                entities_copied=built.entities_copied,
                statements_copied=built.statements_copied,
                reference_chunks_copied=built.reference_chunks_copied,
                measured_bytes=measurement["bytes"],
                measurement=measurement,
                notes=list(built.notes) + notes,
            )
        # Trim by the measured overshoot, from the
        # count the last build actually COPIED (a cap
        # above the source's size trims nothing), at
        # least one step below that count, never below
        # zero. The chunk share rides the limit, so it
        # shrinks with the same cut.
        copied = built.entities_copied
        overshoot_cut = int(copied * byte_budget / measurement["bytes"])
        current = max(0, min(copied - 1, overshoot_cut))


# ── Checkers ───────────────────────────────────────────────────────────────


def subset_craft_problems() -> list[str]:
    """Static checker: the filed rules against the live
    tree (budgets resolve for every shipped profile,
    pi_zero's declared budget wins, the builder carries
    the D-SUBSET-2/3/4 fixes)."""
    problems: list[str] = []
    rule_ids = [r["id"] for r in SUBSET_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("subset rule ids are not unique")
    for profile_id in list_hardware_profiles():
        try:
            budget = budget_for_profile(profile_id)
        except Exception as exc:  # noqa: BLE001 — named below
            problems.append(f"budget does not resolve for {profile_id}: {exc}")
            continue
        if budget.budget_bytes <= 0:
            problems.append(f"{profile_id}: budget_bytes is not positive")
        if budget.profile_id == "pi_zero":
            if budget.declared_budget_bytes is None:
                problems.append(
                    "pi_zero's declared knowledge_subset_max_gb is no "
                    "longer read (D-PROF-1 subset share reopened)"
                )
            elif not budget.budget_source.startswith("declared:"):
                problems.append("pi_zero's budget does not ride the declared road")
        elif budget.declared_budget_bytes is not None:
            problems.append(
                f"{profile_id}: a declared budget appeared — update the "
                "filed 'pi_zero only' claim in the open"
            )
    from thoughtforge.etl import subset as subset_module

    src = Path(subset_module.__file__).read_text(encoding="utf-8")
    if "max(1000, entity_limit // 10)" in src:
        problems.append("D-SUBSET-2 reopened: the chunk cap floors at 1000 again")
    if "ORDER BY popularity_score DESC, qid ASC" not in src:
        problems.append("D-SUBSET-4 reopened: entity selection lost its qid tie-break")
    if "subset output must not be the source DB itself" not in src:
        problems.append("D-SUBSET-3 reopened: the builder no longer refuses its source as output")
    return problems


def subset_craft_live_problems(root: Path | str) -> list[str]:
    """Live checker: a synthetic store with oversized
    rows, crafted for phone_low under a small explicit
    byte budget — the result must fit by MEASUREMENT,
    its file must hold exactly the reported rows, and
    an impossible budget must fail honestly."""
    from thoughtforge.etl.schema import initialize_schema

    problems: list[str] = []
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "craft_source.db"
    initialize_schema(source)
    conn = sqlite3.connect(str(source))
    try:
        big = "x" * 20000
        for i in range(30):
            conn.execute(
                "INSERT INTO entities VALUES (?,?,?,?,?,?,?)",
                (f"Q{i:02d}", f"entity_{i}", big, "", "item",
                 float(30 - i), "wikidata"),
            )
            conn.execute(
                "INSERT INTO statements (subject_qid, property_pid,"
                " object_value, object_type) VALUES (?,?,?,?)",
                (f"Q{i:02d}", "P1", "v", "t"),
            )
        for j in range(50):
            conn.execute(
                "INSERT INTO reference_chunks (source_file, chunk_index,"
                " title, content, tags, word_count) VALUES (?,?,?,?,?,?)",
                (f"f{j}.md", 0, "t", big, "", 1),
            )
        conn.commit()
    finally:
        conn.close()

    out = root / "craft_subset.db"
    result = build_profile_subset(
        source, out, "phone_low", max_bytes=200_000,
    )
    if not result.success or not result.fits_budget:
        problems.append(f"crafted subset does not fit its budget: {result.to_dict()}")
        return problems
    if result.measured_bytes is None or result.measured_bytes > 200_000:
        problems.append(f"measured size over budget: {result.measured_bytes}")
    if len(result.attempts) < 2:
        problems.append(
            f"overshoot was not trimmed through attempts: {result.attempts}"
        )
    check = sqlite3.connect(str(out))
    try:
        held = check.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    finally:
        check.close()
    if held != result.entities_copied:
        problems.append(
            f"file holds {held} entities, result reports "
            f"{result.entities_copied} (D-SUBSET-3 shape)"
        )
    impossible = build_profile_subset(
        source, root / "craft_impossible.db", "phone_low", max_bytes=1024,
    )
    if impossible.success or impossible.fits_budget:
        problems.append(
            f"an impossible budget was reported as success: {impossible.to_dict()}"
        )
    return problems
