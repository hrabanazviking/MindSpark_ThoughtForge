"""
Freshness — time-aware retrieval (TF-059).

The roadmap's rule: *time-aware retrieval (recent facts
weighted per the stated rule) for time-sensitive domains.*

The measured starting state, before this slice (probed
this tick, keeper 08:42, on this VM):

  * **Retrieval recency was time-blind.** The knowledge
    forge's ``_score_records`` sets the recency dimension
    from ``SOURCE_RECENCY_PROXIES`` — a per-source
    constant (reference 0.9, wikidata 0.7, conceptnet
    0.8, geonames 0.75, dbpedia 0.65). Probed: two records
    identical except a ``created_at`` of 2020 vs 2026
    scored *exactly* the same composite (0.57) — a
    record's actual age never entered its score. The
    knowledge DB's tables carry no timestamp column at
    all, so for DB records the proxy was the only recency
    available; nothing stated that, and nothing used a
    timestamp when a record *did* carry one
    (**D-FRESH-1**, found + FIXED in this slice for
    timestamp-carrying records — see below).
  * **Personal records carried time nobody read.** Every
    persisted memory record carries ``created_at`` /
    ``updated_at`` / ``last_used_at`` and a
    ``recency_weight`` field — and ``recency_weight`` is
    read by *no code anywhere* (its only appearances are
    the model definitions and TF-042's field census).
    ``compute_recency_score`` (the half-life decay,
    TF-041 K3) was consumed only by the lifecycle's
    prune/survival legs: time decided what got *retired*,
    never what got *retrieved first*.
  * **No domain was time-sensitive.** Nothing in the
    tree distinguished a ferry departure time (wrong
    within days) from a river's name (true for
    centuries): both would age at the record type's one
    half-life (user_fact 180 days) if they aged at all,
    and a stale schedule fact carried no mark saying so.

What this slice builds — ``knowledge/freshness.py``:

  * ``TIME_SENSITIVE_DOMAINS`` — the domains *as data*:
    schedule, location, status, price. Each row names
    its markers (matched, case/hyphen/underscore-folded,
    against a record's tags, a preference's category,
    and a fact's ``fact_value`` keys), its own half-life
    in days, and its stale horizon in days. A record
    matching several domains is governed by the *most
    perishable* (shortest half-life) — stated in
    ``FRESHNESS_RULES``. A record matching none is
    **timeless** for freshness purposes: its age is
    still measured (at its record type's TF-041
    half-life), but age is not a validity question for
    it and it can never be verdicted ``stale``.
  * **The freshness clock** is the record's
    ``updated_at`` (when the stored claim last changed
    or was re-stamped by a write), falling back to
    ``created_at``, then ``last_used_at``. A missing or
    unparseable clock reads as
    ``FRESHNESS_AGE_DEFAULT_DAYS`` old (30 days — the
    lifecycle's own parse-default, pinned equal by the
    checker): an unreadable timestamp is neither brand
    new (immortal) nor ancient (instantly stale for a
    timeless record — and for a sensitive record 30
    days is honestly past every domain's fresh window).
  * ``freshness_score`` — decayed recency, never binary
    (the Retrieval spec §10.4's own rule): ``0.5 **
    (age / half_life)`` with the *domain's* half-life
    for sensitive records and the record type's
    ``HALF_LIFE_DAYS`` (through scoring's own
    ``compute_recency_score``) for timeless ones.
  * ``freshness_verdict`` — ``fresh`` (score >= 0.5,
    i.e. within one half-life), ``ageing`` (0.2 <=
    score < 0.5), ``stale`` (sensitive only: score <
    0.2 *or* age beyond the domain's stale horizon),
    ``timeless`` (no domain — the verdict states that
    freshness does not judge this record).
  * **The weighting rule** (``rank_records`` /
    ``search_store``): a record's retrieval composite
    is scoring's own ``score_retrieval`` over stated
    dimensions — semantic = the caller's relevance,
    importance = the record's own importance (episodes
    and patterns: their quality rides there at the
    neutral share stated in ``FRESHNESS_RULES``),
    recency = the freshness score. Sensitive records
    are scored with ``TIME_SENSITIVE_WEIGHTS``, a row
    in which recency carries 0.40 — so two otherwise
    identical schedule facts a month apart separate by
    ~0.40 of composite, not by the fact row's 0.10.
    Timeless records keep their record type's shipped
    row, byte-identical weights. Ties break by
    record_id, so ranking is reproducible.
  * ``search_store`` — term-overlap retrieval over a
    MemoryStore's personal records (profile facts +
    preferences, episodes, patterns) returning a
    ``MemoryActivationBundle`` whose raw dicts carry
    the full freshness assessment (domain, age, score,
    verdict), so a consumer can *see* why a record
    ranked where it did and how fresh it is. **Stale
    sensitive records are excluded by default** and
    named in the returned report — a stale fact is
    never silently presented as current; a caller that
    wants them asks (``include_stale=True``) and gets
    them flagged ``stale`` in their raw. Read-only:
    the store is loaded, never written (byte-pinned).
  * ``freshness_report`` — one JSON-safe report over a
    record set or a store: counts by verdict and by
    domain, and every sensitive record named with its
    age/score/verdict.

Wiring (this slice): ``KnowledgeForge._score_records``
now takes a record's own timestamp when it carries one
(``updated_at`` / ``created_at`` / ``last_used_at`` /
``timestamp``): its recency dimension is the decayed
recency of that age at the default half-life, *instead
of* the source proxy. Records carrying no timestamp —
every record the shipped DB tables can produce today,
and TF-043's whole fixture corpus — score exactly as
before, proxy and all (pinned: the TF-043 baseline
arithmetic is unchanged, and a timestamped pair
separates old from new). **D-FRESH-1 is therefore
FIXED for timestamp-carrying records and stands
stated, not hidden, for the timestamp-less DB estate:**
the schema carries no date column, so DB recency
remains a proxy until a slice gives the ETL a date to
carry (TF-106/TF-111's territory, named in the doc).

Stated boundaries:

  * Freshness judges *when a claim was last stamped*,
    not whether it is true: a fresh falsehood scores
    fresh, an old truth about a timeless subject is
    timeless. Contradiction (TF-050) judges claims
    against each other; freshness judges claims
    against the clock.
  * ``search_store``'s relevance is the stated
    term-overlap share (TF-058's rule), not the forge's
    bm25/vector ranking; its fixtures prove the
    freshness weighting and the stale rule, not ranking
    quality — Part VII owns ranking research.
  * No record is deleted, archived, or rewritten by
    anything here: retirement is the lifecycle's
    (TF-045), forgetting is TF-051's. A stale verdict
    is a retrieval/report fact, not a store mutation.
  * Proof grade is FIXTURE (synthetic stores and
    records with explicit clocks, this VM) — freshness
    behaviour over a real lived-in store is NOT PROVEN
    (no real store on this machine, TF-041), and
    live-model use of the flagged bundle is NOT PROVEN,
    never claimed.

The human-readable write-up is
``docs/architecture/FRESHNESS.md``.
"""

from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thoughtforge.knowledge.scoring import (
    DEFAULT_HALF_LIFE_DAYS,
    HALF_LIFE_DAYS,
    RETRIEVAL_WEIGHTS_BY_TYPE,
    compute_recency_score,
)

# ── Thresholds, as data ──────────────────────────────────────────────────────

#: score >= this is "fresh" (within one half-life).
FRESH_SCORE: float = 0.5
#: a sensitive record under this score is "stale".
STALE_SCORE: float = 0.2
#: the assumed age (days) of a record whose freshness
#: clock cannot be parsed — the lifecycle's own default
#: (knowledge/lifecycle.AGE_PARSE_DEFAULT_DAYS), pinned
#: equal by freshness_problems().
FRESHNESS_AGE_DEFAULT_DAYS: float = 30.0

#: The freshness clock's fields, in the order tried.
CLOCK_FIELDS: tuple[str, ...] = ("updated_at", "created_at", "last_used_at")

#: Extra clock field names a forge (DB-shaped) record may
#: carry — the forge wiring accepts these too.
FORGE_CLOCK_FIELDS: tuple[str, ...] = CLOCK_FIELDS + ("timestamp",)

# ── The time-sensitive domains, as data ──────────────────────────────────────
#
# Markers are matched folded (lowercased, '-' and '_' and
# spaces removed) against a record's tags, a preference's
# category, and a fact's fact_value keys. Half-lives are
# deliberately short next to the record types' own: a
# schedule is perishable in days, a price in weeks.

TIME_SENSITIVE_DOMAINS: tuple[dict[str, Any], ...] = (
    {"name": "schedule",
     "markers": ("schedule", "timetable", "departure", "departures",
                 "appointment", "deadline", "openinghours", "ferrytimes"),
     "half_life_days": 2.0,
     "stale_after_days": 7.0,
     "rule": "times of day / dates of events: a week-old schedule "
             "is a claim about a world that has moved on"},
    {"name": "location",
     "markers": ("location", "whereabouts", "address", "currentaddress",
                 "travel", "journey"),
     "half_life_days": 7.0,
     "stale_after_days": 30.0,
     "rule": "where someone or something is now: moves in days, "
             "settled within a month or it is history"},
    {"name": "status",
     "markers": ("status", "currentproject", "currentrole",
                 "availability", "available", "inprogress",
                 "currentwork"),
     "half_life_days": 7.0,
     "stale_after_days": 30.0,
     "rule": "what is currently true of a project, role, or "
             "person's availability: same clock as location"},
    {"name": "price",
     "markers": ("price", "cost", "fee", "fees", "rate", "rates",
                 "fare", "fares"),
     "half_life_days": 30.0,
     "stale_after_days": 90.0,
     "rule": "what something costs: drifts in weeks, unusable "
             "as current after a season"},
)

#: The retrieval weights for a time-sensitive record: the
#: user_fact row's shape with recency carrying 0.40 (the
#: stated freshness rule) — semantic and importance keep
#: real shares, so a relevant stale fact still outranks
#: an irrelevant fresh one on relevance alone only when
#: the relevance gap exceeds the freshness gap.
TIME_SENSITIVE_WEIGHTS: dict[str, float] = {
    "semantic_similarity": 0.35,
    "tone_similarity": 0.0,
    "preference_relevance": 0.10,
    "recency": 0.40,
    "importance": 0.15,
    "thread_boost": 0.0,
    "pattern_quality": 0.0,
    "priority": 0.0,
}

#: The freshness rules, as data.
FRESHNESS_RULES: tuple[dict[str, Any], ...] = (
    {"name": "domain_governs",
     "rule": "a record matching a time-sensitive domain ages at "
             "that domain's half-life, not its record type's; "
             "several domains -> the shortest half-life governs"},
    {"name": "timeless_never_stale",
     "rule": "a record matching no domain is timeless: its age "
             "is measured at its type's half-life, it is weighted "
             "by its type's shipped retrieval row, and it can "
             "never be verdicted stale"},
    {"name": "decayed_not_binary",
     "rule": "freshness is 0.5 ** (age / half_life) — a decay, "
             "never a fresh/expired switch (Retrieval spec 10.4)"},
    {"name": "stale_never_silent",
     "rule": "a stale sensitive record is excluded from "
             "search_store's bundle by default and named in its "
             "report; include_stale=True returns it flagged "
             "stale in its raw dict"},
    {"name": "clock_is_the_write",
     "rule": "age runs from updated_at, falling back to "
             "created_at, then last_used_at; an unparseable "
             "clock reads as FRESHNESS_AGE_DEFAULT_DAYS old"},
    {"name": "read_only",
     "rule": "freshness assessment, ranking, search, and "
             "reports never write to a store or a record"},
)


def _fold(text: Any) -> str:
    return re.sub(r"[\s\-_]+", "", str(text).lower())


def _get(record: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _record_markers(record: Any) -> set[str]:
    markers: set[str] = set()
    for tag in _get(record, "tags", []) or []:
        markers.add(_fold(tag))
    category = _get(record, "category", "")
    if category:
        markers.add(_fold(category))
    fact_value = _get(record, "fact_value", {}) or {}
    if isinstance(fact_value, dict):
        for key in fact_value:
            markers.add(_fold(key))
    return markers


def domain_for(record: Any) -> dict[str, Any] | None:
    """The governing time-sensitive domain row for a
    record, or None when the record is timeless. The
    most perishable matching domain governs."""
    markers = _record_markers(record)
    matched = [
        row for row in TIME_SENSITIVE_DOMAINS
        if markers & set(row["markers"])
    ]
    if not matched:
        return None
    return min(matched, key=lambda row: row["half_life_days"])


def classify_domain(record: Any) -> str:
    """The governing domain's name, or "" (timeless)."""
    row = domain_for(record)
    return str(row["name"]) if row else ""


# ── The clock ────────────────────────────────────────────────────────────────


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def freshness_timestamp(record: Any) -> str:
    """The record's freshness-clock value (first present
    of CLOCK_FIELDS), or "" when it carries none."""
    for field_name in CLOCK_FIELDS:
        value = _get(record, field_name, "")
        if value:
            return str(value)
    return ""


def record_age_days(
    record: Any, now: datetime | None = None
) -> float:
    """Age in days from the freshness clock to ``now``
    (default: the real now). Future clocks age 0; a
    missing/unparseable clock ages the stated default."""
    moment = now or datetime.now(timezone.utc)
    parsed = _parse_ts(freshness_timestamp(record))
    if parsed is None:
        return FRESHNESS_AGE_DEFAULT_DAYS
    return max(0.0, (moment - parsed).total_seconds() / 86400.0)


def recency_from_timestamps(
    record: dict[str, Any], now: datetime | None = None
) -> float | None:
    """Decayed recency for a dict-shaped (forge) record
    carrying a timestamp in any FORGE_CLOCK_FIELDS field,
    at the default half-life; None when it carries no
    parseable timestamp (the caller keeps its proxy)."""
    moment = now or datetime.now(timezone.utc)
    for field_name in FORGE_CLOCK_FIELDS:
        parsed = _parse_ts(record.get(field_name))
        if parsed is not None:
            age = max(0.0, (moment - parsed).total_seconds() / 86400.0)
            return math.pow(0.5, age / DEFAULT_HALF_LIFE_DAYS)
    return None


# ── Assessment ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FreshnessAssessment:
    """One record judged against the clock."""

    record_id: str
    record_type: str
    domain: str                 # "" = timeless
    time_sensitive: bool
    age_days: float
    half_life_days: float
    score: float
    verdict: str                # fresh | ageing | stale | timeless

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "domain": self.domain,
            "time_sensitive": self.time_sensitive,
            "age_days": round(self.age_days, 4),
            "half_life_days": self.half_life_days,
            "score": round(self.score, 4),
            "verdict": self.verdict,
        }


def assess_freshness(
    record: Any, now: datetime | None = None
) -> FreshnessAssessment:
    """Judge one record: its domain, age, decayed
    freshness score, and verdict, by the stated rules."""
    record_type = str(_get(record, "record_type", "") or "")
    domain = domain_for(record)
    age = record_age_days(record, now=now)
    if domain is not None:
        half_life = float(domain["half_life_days"])
        score = math.pow(0.5, age / half_life) if half_life > 0 else 0.0
        if score < STALE_SCORE or age > float(domain["stale_after_days"]):
            verdict = "stale"
        elif score >= FRESH_SCORE:
            verdict = "fresh"
        else:
            verdict = "ageing"
        return FreshnessAssessment(
            record_id=str(_get(record, "record_id", "") or ""),
            record_type=record_type,
            domain=str(domain["name"]),
            time_sensitive=True,
            age_days=age,
            half_life_days=half_life,
            score=score,
            verdict=verdict,
        )
    score = compute_recency_score(age, record_type)
    return FreshnessAssessment(
        record_id=str(_get(record, "record_id", "") or ""),
        record_type=record_type,
        domain="",
        time_sensitive=False,
        age_days=age,
        half_life_days=float(HALF_LIFE_DAYS.get(
            record_type, DEFAULT_HALF_LIFE_DAYS)),
        score=score,
        verdict="timeless",
    )


def freshness_score(record: Any, now: datetime | None = None) -> float:
    return assess_freshness(record, now=now).score


def freshness_verdict(record: Any, now: datetime | None = None) -> str:
    return assess_freshness(record, now=now).verdict


# ── Ranking: recent facts weighted per the stated rule ───────────────────────


def weights_for(record: Any) -> dict[str, float]:
    """The retrieval weight row a record is ranked with:
    TIME_SENSITIVE_WEIGHTS for a sensitive record, else
    its record type's shipped row (the base row for an
    unknown type)."""
    from thoughtforge.knowledge.scoring import RETRIEVAL_WEIGHTS_BASE

    if domain_for(record) is not None:
        return dict(TIME_SENSITIVE_WEIGHTS)
    record_type = str(_get(record, "record_type", "") or "")
    row = RETRIEVAL_WEIGHTS_BY_TYPE.get(record_type)
    return dict(row) if row is not None else dict(RETRIEVAL_WEIGHTS_BASE)


def rank_records(
    records: list[Any],
    relevance: dict[str, float],
    now: datetime | None = None,
) -> list[tuple[FreshnessAssessment, float]]:
    """Rank records by the stated rule: composite =
    score_retrieval(semantic=relevance, recency=
    freshness score, importance=the record's own
    importance — episodes/patterns carry quality there
    at the same field via getattr fallback) under the
    record's weight row. Sorted composite desc, ties by
    record_id (reproducible). Returns (assessment,
    composite) pairs."""
    from thoughtforge.knowledge.scoring import RetrievalDimensions, score_retrieval

    ranked: list[tuple[FreshnessAssessment, float]] = []
    for record in records:
        assessment = assess_freshness(record, now=now)
        importance = _get(record, "importance", None)
        if importance is None:
            importance = _get(record, "quality", 0.5)
        dims = RetrievalDimensions(
            semantic_similarity=float(relevance.get(assessment.record_id, 0.0)),
            recency=assessment.score,
            importance=float(importance),
        )
        composite = score_retrieval(
            dims, assessment.record_type, weights=weights_for(record))
        ranked.append((assessment, round(composite, 4)))
    ranked.sort(key=lambda item: (-item[1], item[0].record_id))
    return ranked


def _store_records(store: Any) -> list[Any]:
    records: list[Any] = list(store.load_profile_records())
    records.extend(store.load_episodes())
    records.extend(store.load_patterns())
    return records


def search_store(
    store: Any,
    query: str,
    now: datetime | None = None,
    top_k: int = 10,
    include_stale: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Freshness-weighted retrieval over a MemoryStore's
    personal records, by the stated rule.

    Relevance is the term-overlap share (the forge's key
    terms present whole-word in summary + tags — TF-058's
    rule). Stale sensitive records are excluded unless
    ``include_stale`` and are always named in the
    report. Returns (MemoryActivationBundle, report);
    each activated record's raw carries its freshness
    assessment, so its ranking is explainable. The store
    is only read."""
    from thoughtforge.knowledge.forge import KnowledgeForge
    from thoughtforge.knowledge.models import (
        ActivatedRecord,
        MemoryActivationBundle,
    )

    terms = KnowledgeForge._extract_key_terms(query, max_terms=8)
    report: dict[str, Any] = {
        "query": query,
        "terms": terms,
        "considered": 0,
        "matched": 0,
        "stale_excluded": [],
        "include_stale": include_stale,
    }
    if not terms or top_k <= 0:
        return MemoryActivationBundle(
            activated_records=[], total_retrieved=0,
            retrieval_confidence=0.0), report
    # TF-060: matching is multilingual — folded tokens
    # (casefold, spaceless-script bigrams) with the
    # diacritic-stripped form as the auxiliary match, so
    # an accent-stripped query still finds the accented
    # record here. English behaviour is unchanged (the
    # folded form of an English token is its lowercase).
    from thoughtforge.knowledge.multilingual import (
        match_tokens,
        terms_matching,
    )

    relevance: dict[str, float] = {}
    candidates: list[Any] = []
    for record in _store_records(store):
        report["considered"] += 1
        haystack = match_tokens(
            f"{_get(record, 'summary', '')} "
            f"{' '.join(_get(record, 'tags', []) or [])}")
        matched = terms_matching(terms, haystack)
        if matched:
            relevance[str(_get(record, "record_id", ""))] = \
                len(matched) / len(terms)
            candidates.append(record)
    report["matched"] = len(candidates)
    ranked = rank_records(candidates, relevance, now=now)
    assessments = {a.record_id: a for a, _ in ranked}
    activated: list[ActivatedRecord] = []
    for assessment, composite in ranked:
        if assessment.verdict == "stale" and not include_stale:
            report["stale_excluded"].append(assessment.record_id)
            continue
        record = next(
            r for r in candidates
            if str(_get(r, "record_id", "")) == assessment.record_id)
        raw = {
            "record_id": assessment.record_id,
            "record_type": assessment.record_type,
            "summary": str(_get(record, "summary", "")),
            "tags": list(_get(record, "tags", []) or []),
            "source": str(_get(record, "source", "") or "memory_store"),
            "freshness": assessment.to_dict(),
        }
        cue = KnowledgeForge._build_cue({
            "label_en": assessment.record_id,
            "content": raw["summary"],
            "source": raw["source"],
        })
        activated.append(ActivatedRecord(
            record_id=assessment.record_id,
            record_type=assessment.record_type,
            score=composite,
            cue=cue,
            raw=raw,
        ))
        if len(activated) >= top_k:
            break
    confidence = round(
        sum(r.score for r in activated) / len(activated), 4
    ) if activated else 0.0
    bundle = MemoryActivationBundle(
        activated_records=activated,
        total_retrieved=len(activated),
        retrieval_confidence=confidence,
    )
    report["returned"] = len(activated)
    return bundle, report


def freshness_report(
    records_or_store: Any, now: datetime | None = None
) -> dict[str, Any]:
    """One JSON-safe freshness report over a record list
    or a MemoryStore: counts by verdict and domain, and
    every time-sensitive record named with its age,
    score, and verdict."""
    if hasattr(records_or_store, "load_profile_records"):
        records = _store_records(records_or_store)
    else:
        records = list(records_or_store)
    by_verdict: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    sensitive: list[dict[str, Any]] = []
    for record in records:
        assessment = assess_freshness(record, now=now)
        by_verdict[assessment.verdict] = \
            by_verdict.get(assessment.verdict, 0) + 1
        if assessment.time_sensitive:
            by_domain[assessment.domain] = \
                by_domain.get(assessment.domain, 0) + 1
            sensitive.append(assessment.to_dict())
    sensitive.sort(key=lambda row: (row["score"], row["record_id"]))
    return {
        "total": len(records),
        "by_verdict": by_verdict,
        "by_domain": by_domain,
        "time_sensitive": sensitive,
        "stale": [row["record_id"] for row in sensitive
                  if row["verdict"] == "stale"],
    }


# ── The live checker ─────────────────────────────────────────────────────────


def freshness_problems() -> list[str]:
    """Re-derive TF-059's claims from the live tree; [] = holds.

    Pins: the domains as data (names, ordered half-lives,
    stale horizons after half-lives); the sensitive
    weights sum to 1.0 with recency at 0.40; the parse
    default equals the lifecycle's; the clock order;
    classification (schedule by tag, price by fact key,
    timeless by absence, most-perishable governs);
    decay at exactly one half-life == 0.5 for a domain
    and for a type; verdict bands incl. timeless never
    stale at any age; two identical schedule facts a
    stated age apart separate by the recency share;
    search_store over a synthetic store returns the
    fresh fact first, excludes the stale one by default
    (named), and returns it flagged on request; the
    forge wiring (a timestamped pair separates, a
    timestamp-less pair keeps the proxy arithmetic);
    every leg read-only over the store."""
    problems: list[str] = []

    import inspect

    from thoughtforge.knowledge import forge as forge_module
    from thoughtforge.knowledge.lifecycle import AGE_PARSE_DEFAULT_DAYS
    from thoughtforge.knowledge.models import (
        EpisodicMemoryRecord,
        UserFactRecord,
        UserPreferenceRecord,
    )
    from thoughtforge.knowledge.store import MemoryStore

    if [row["name"] for row in TIME_SENSITIVE_DOMAINS] != [
            "schedule", "location", "status", "price"]:
        problems.append("TIME_SENSITIVE_DOMAINS drifted")
    for row in TIME_SENSITIVE_DOMAINS:
        if not row["half_life_days"] < row["stale_after_days"]:
            problems.append(f"domain {row['name']}: stale horizon <= half-life")
        if not row["markers"]:
            problems.append(f"domain {row['name']}: no markers")
    if [row["name"] for row in FRESHNESS_RULES] != [
            "domain_governs", "timeless_never_stale", "decayed_not_binary",
            "stale_never_silent", "clock_is_the_write", "read_only"]:
        problems.append("FRESHNESS_RULES drifted")
    if not math.isclose(sum(TIME_SENSITIVE_WEIGHTS.values()), 1.0, abs_tol=1e-9):
        problems.append("TIME_SENSITIVE_WEIGHTS do not sum to 1.0")
    if TIME_SENSITIVE_WEIGHTS["recency"] != 0.40:
        problems.append("TIME_SENSITIVE_WEIGHTS recency != 0.40")
    if FRESHNESS_AGE_DEFAULT_DAYS != AGE_PARSE_DEFAULT_DAYS:
        problems.append("freshness parse default != lifecycle's")
    if CLOCK_FIELDS != ("updated_at", "created_at", "last_used_at"):
        problems.append("CLOCK_FIELDS drifted")

    now = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)

    def _fact(rid: str, age_days: float, **kw: Any) -> UserFactRecord:
        from datetime import timedelta

        stamp = (now - timedelta(days=age_days)).isoformat()
        return UserFactRecord(
            record_id=rid, created_at=stamp, updated_at=stamp,
            last_used_at=stamp, **kw)

    sched = _fact("f_sched", 0.0, tags=["schedule"],
                  summary="the ferry departs at dawn")
    if classify_domain(sched) != "schedule":
        problems.append("a schedule-tagged fact is not schedule domain")
    priced = _fact("f_price", 0.0, tags=["ferry"],
                   summary="the crossing price",
                   fact_value={"fare": 3})
    if classify_domain(priced) != "price":
        problems.append("a fare-keyed fact is not price domain")
    plain = _fact("f_plain", 0.0, tags=["ferry"],
                  summary="the ferry crosses the river")
    if classify_domain(plain) != "":
        problems.append("an unmarked fact is not timeless")
    both = _fact("f_both", 0.0, tags=["schedule"],
                 summary="x", fact_value={"fare": 3})
    if classify_domain(both) != "schedule":
        problems.append("most-perishable domain does not govern (schedule+price)")
    pref = UserPreferenceRecord(
        record_id="p_loc", category="location",
        summary="currently based at the mill")
    if classify_domain(pref) != "location":
        problems.append("a location-category preference is not location domain")

    two_day = _fact("f_2d", 2.0, tags=["schedule"], summary="x")
    if not math.isclose(freshness_score(two_day, now=now), 0.5, abs_tol=1e-9):
        problems.append("schedule freshness at one half-life != 0.5")
    if freshness_verdict(_fact("f_old_sched", 30.0, tags=["schedule"],
                                summary="x"), now=now) != "stale":
        problems.append("a 30-day schedule fact is not stale")
    ancient_plain = _fact("f_ancient", 3650.0, tags=["ferry"], summary="x")
    if freshness_verdict(ancient_plain, now=now) != "timeless":
        problems.append("an ancient timeless fact was verdicted stale")
    if not math.isclose(
            freshness_score(ancient_plain, now=now),
            compute_recency_score(3650.0, "user_fact"), abs_tol=1e-9):
        problems.append("timeless freshness != the type half-life decay")

    fresh = _fact("f_fresh", 0.0, tags=["schedule"],
                  summary="the ferry departs at dawn")
    month_old = _fact("f_month", 30.0, tags=["schedule"],
                      summary="the ferry departs at dawn")
    ranked = rank_records(
        [month_old, fresh],
        {"f_fresh": 1.0, "f_month": 1.0}, now=now)
    if [a.record_id for a, _ in ranked] != ["f_fresh", "f_month"]:
        problems.append("fresh schedule fact did not outrank the month-old one")
    else:
        gap = ranked[0][1] - ranked[1][1]
        if not gap > 0.3:
            problems.append(f"sensitive freshness gap {gap} <= 0.3")

    # search_store over a synthetic store, read-only.
    with tempfile.TemporaryDirectory(prefix="tf059-") as tmp:
        store = MemoryStore(memory_dir=Path(tmp))
        for record in (fresh, month_old, plain):
            store.append_profile_record(record)
        store.append_episode(EpisodicMemoryRecord(
            record_id="ep_ferry", tags=["ferry"],
            summary="the ferry lantern was lit",
            created_at=now.isoformat(), updated_at=now.isoformat(),
            last_used_at=now.isoformat()))
        before = {p.name: p.read_bytes()
                  for p in sorted(Path(tmp).iterdir()) if p.is_file()}
        bundle, report = search_store(store, "ferry", now=now)
        ids = [r.record_id for r in bundle.activated_records]
        if "f_month" not in report["stale_excluded"]:
            problems.append(f"stale fact not named excluded: {report}")
        if "f_month" in ids:
            problems.append("stale fact returned by default")
        if not ids or ids[0] != "f_fresh":
            problems.append(f"fresh fact not first: {ids}")
        flagged, _ = search_store(store, "ferry", now=now, include_stale=True)
        stale_rows = [r for r in flagged.activated_records
                      if r.record_id == "f_month"]
        if not stale_rows or \
                stale_rows[0].raw["freshness"]["verdict"] != "stale":
            problems.append("include_stale did not return the fact flagged stale")
        after = {p.name: p.read_bytes()
                 for p in sorted(Path(tmp).iterdir()) if p.is_file()}
        if after != before:
            problems.append("search_store wrote to the store (read-only broken)")

    # The forge wiring: timestamped records age for real,
    # timestamp-less records keep the proxy.
    forge = forge_module.KnowledgeForge.__new__(forge_module.KnowledgeForge)
    forge.retrieval_weights = None
    base = {"label_en": "ferry", "description_en": "x",
            "popularity_score": 0.5, "fts_score": 10.0,
            "source": "wikidata", "record_type": "entity"}
    old = dict(base, qid="Q_old", updated_at="2020-01-01T00:00:00+00:00")
    new = dict(base, qid="Q_new", updated_at=now.isoformat())
    scored = forge._score_records([old, new], "ferry")
    by_id = {r["qid"]: r["_composite_score"] for r in scored}
    if not by_id["Q_new"] > by_id["Q_old"]:
        problems.append("forge: timestamped new record did not outscore old")
    plain_pair = [dict(base, qid="Q_a"), dict(base, qid="Q_b")]
    scored_plain = forge._score_records(plain_pair, "ferry")
    if scored_plain[0]["_composite_score"] != scored_plain[1]["_composite_score"]:
        problems.append("forge: timestamp-less records lost proxy equality")
    if "recency_from_timestamps" not in inspect.getsource(
            forge_module.KnowledgeForge._score_records):
        problems.append("forge wiring: _score_records does not consult timestamps")
    return problems


__all__ = [
    "CLOCK_FIELDS",
    "FORGE_CLOCK_FIELDS",
    "FRESHNESS_AGE_DEFAULT_DAYS",
    "FRESHNESS_RULES",
    "FRESH_SCORE",
    "STALE_SCORE",
    "TIME_SENSITIVE_DOMAINS",
    "TIME_SENSITIVE_WEIGHTS",
    "FreshnessAssessment",
    "assess_freshness",
    "classify_domain",
    "domain_for",
    "freshness_problems",
    "freshness_report",
    "freshness_score",
    "freshness_timestamp",
    "freshness_verdict",
    "rank_records",
    "recency_from_timestamps",
    "record_age_days",
    "search_store",
    "weights_for",
]
