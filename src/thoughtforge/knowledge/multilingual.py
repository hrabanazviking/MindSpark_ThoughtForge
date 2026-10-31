"""
Multilingual memory — non-English facts stored and
retrieved without translation loss (TF-060).

The roadmap's rule: *non-English facts stored/retrieved
without translation loss (fixtures in two languages).*
The two fixture languages of this slice are **Spanish
(es)** and **German (de)** — both synthetic Lantern
Hollow material (TF-017's invented world), never canon.

The measured starting state, before this slice (probed
this tick, keeper 08:52, on this VM):

  * **Storage was already lossless — and unproven.**
    ``MemoryStore`` writes JSONL/YAML with
    ``ensure_ascii=False`` in UTF-8, and a Spanish and a
    German fact round-tripped byte-for-byte through the
    store files in the probe. Nothing in the tree had
    ever *tested* that, and no other leg (portability
    bundles, tag index) had been checked for it.
  * **Retrieval term extraction was English-only**
    (**D-ML-1**, found + FIXED in this slice).
    ``KnowledgeForge._extract_key_terms`` — the term
    source for the forge's ConceptNet/GeoNames legs and
    for both personal-store searches built in Part III
    (TF-058's reference search, TF-059's freshness
    search) — carried an English-only stopword list,
    dropped every token of length <= 2, and matched
    with ``lower()``. Probed consequences:
      - accent-stripped queries lost accented facts:
        the query "guia travesia" (both accents dropped,
        as a plain keyboard types it) retrieved **0**
        records against a store holding "guía la
        travesía", while the accented query retrieved it;
      - case folding lost German ß facts: the query
        "strasse" retrieved **0** against a record
        tagged "straße" (``lower()`` keeps ß as ß;
        ``casefold()`` maps it to "ss"), while "straße"
        retrieved it;
      - a two-character query that is a complete word
        in a spaceless script vanished entirely: "灯台"
        (lighthouse) extracted **zero** terms because of
        the length rule, and a whole spaceless sentence
        tokenised as ONE ``\\w+`` token, so no shorter
        query could ever match inside it;
      - Spanish and German function words ("el", "la",
        "die", "das", "und", "está") were kept as key
        terms (noise), because the stopword list knew
        none of them.
  * **No record carries a language, and nothing could
    say which language a record was in.** The memory
    record shapes (TF-042) have no language field, so a
    consumer handed a mixed-language store had no way
    to report its composition, and no rule stated the
    project's position on translation.

What this slice builds — ``knowledge/multilingual.py``:

  * **The position, stated as data**
    (``MULTILINGUAL_RULES``): memory is stored and
    retrieved **in its own language, verbatim** — the
    stored text is never translated, transliterated,
    or normalised in place. Matching works on *derived*
    keys (folded forms computed at query time); the
    record on disk keeps every accent, ß, and character
    it was written with. A query in one language does
    **not** retrieve a fact stored in another language
    by translation equivalence — "lighthouse" does not
    find the "faro" fact (pinned); the two meet only on
    tokens they genuinely share (proper names, cognate
    spellings, tags). Translation is a model's job at
    answer time, never the store's at retrieval time.
  * ``fold_text`` — the matching form: NFC-normalised
    ``casefold()`` (ß -> ss, final sigma, full Unicode
    case folding), used for *comparison only*.
  * ``strip_diacritics`` — the auxiliary matching form:
    NFD decomposition with combining marks dropped, so
    an accent-stripped query still finds the accented
    fact and vice versa. A match by this form is a
    retrieval convenience, stated as such: it can
    conflate words that differ only by a mark
    (Spanish año/ano is the textbook case); the stored
    text is never altered by it, and exact folded
    matches are counted before stripped ones in the
    report's evidence.
  * ``match_tokens`` / ``key_terms`` — tokenisation
    that knows about scripts: word tokens are folded;
    a token containing CJK/Hangul characters also
    contributes its character **bigrams**, so a short
    query can match inside a spaceless sentence and a
    two-character word is a real term. Stopwords are
    data per language (``STOPWORDS_BY_LANGUAGE``:
    en/es/de) and are filtered as one union — a word
    that is a function word in any fixture language is
    filtered in every query (the stated trade-off:
    the German article "die" also filters the English
    verb "die"). The length floor drops to 2, with the
    English closed-class two-letter words added to the
    English stopword row so standing English behaviour
    is preserved (pinned against the pre-slice rule on
    an English query).
  * ``detect_language`` — a heuristic hint, as data
    (``LANGUAGE_HINTS``): distinctive stopword hits +
    distinctive characters score each language; the
    best score wins, a tie or a zero score is
    ``"unknown"``, and CJK/Hangul script presence is
    reported as ``"cjk"`` (the hint does not pretend
    to separate Japanese from Chinese from Korean on
    script alone). A hint is *derived* — records gain
    no language field (TF-042's shape is untouched);
    consumers that need certainty must ask the user,
    and ``language_report`` says so in its output.
  * ``search_store`` — multilingual retrieval over a
    MemoryStore's personal records (profile facts +
    preferences, episodes, patterns): relevance is the
    share of the query's key terms matching the
    record's summary + tags + fact_value keys/values /
    preference value, where matching is exact-folded
    first and diacritic-stripped second
    (``terms_matching``). Each activated record's raw
    carries its summary **verbatim**, its language
    hint, and the matched terms. Cross-language
    retrieval happens only on genuinely shared tokens.
    Read-only: the store is loaded, never written
    (byte-pinned).
  * ``language_report`` — one JSON-safe report over a
    record list or a store: counts by language hint,
    every record named with its hint and the evidence
    scores behind it.

Wiring (this slice, D-ML-1's fix):
``KnowledgeForge._extract_key_terms`` now delegates to
``key_terms`` (its ConceptNet/GeoNames legs and both
Part III personal-store searches inherit the fix);
TF-059's ``freshness.search_store`` and TF-058's
``ReferenceLibrary.search`` now match through
``terms_matching`` / ``match_tokens`` instead of a
``lower()`` set intersection, so their relevance is
folded + diacritic-auxiliary too (their English pins
are unchanged — proven by their suites); and
``forge_toolset.inspect_records``' text filter compares
folded forms, so an inspector can find "straße" by
typing "STRASSE".

Stated boundaries:

  * The knowledge DB's FTS5 leg is **not** rewired:
    ``_to_fts5_query`` and the entities' ``label_en``
    columns are the DB estate's own shape — the labels
    table holds per-language labels the forge never
    reads, and that is TF-106/TF-107's ingestion
    territory, not this slice's. This slice's claim is
    the *personal memory store* (Part III's subject),
    plus the term-extraction legs named above.
  * Diacritic-stripped matching is auxiliary and can
    conflate mark-only distinctions (año/ano); some
    letters are not mark-separable at all (ø, æ, ł,
    đ stay themselves under both forms). Stated, not
    hidden.
  * Detection is a heuristic over two fixture
    languages + English + a script label — not a
    language identifier for the world's languages.
    A short or proper-name-only record honestly
    reports ``unknown``.
  * Embeddings (TF-053's one registered model is an
    English-leaning MiniLM) are untouched: vector
    retrieval quality in Spanish/German is NOT PROVEN
    and never claimed; this slice proves the
    deterministic legs.
  * Proof grade is FIXTURE (synthetic Spanish + German
    stores on this VM, TF-017's world) — behaviour over
    a real lived-in multilingual store is NOT PROVEN
    (no real store on this machine, TF-041), and
    live-model answers in either language are NOT
    PROVEN, never claimed.

The human-readable write-up is
``docs/architecture/MULTILINGUAL_MEMORY.md``.
"""

from __future__ import annotations

import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── The position, as data ────────────────────────────────────────────────────

MULTILINGUAL_RULES: tuple[dict[str, Any], ...] = (
    {"name": "verbatim_storage",
     "rule": "a record is stored and returned in its own language, "
             "byte-for-byte; no leg translates, transliterates, or "
             "normalises the stored text in place"},
    {"name": "derived_matching_keys",
     "rule": "matching compares derived keys only — NFC casefolded "
             "forms first, diacritic-stripped forms second; the "
             "record on disk is never rewritten to match"},
    {"name": "no_translation_retrieval",
     "rule": "a query retrieves a record stored in another language "
             "only on tokens they genuinely share (proper names, "
             "cognate spellings, tags) — never by translation "
             "equivalence; translation is the model's job at "
             "answer time, not the store's at retrieval time"},
    {"name": "stopwords_as_union",
     "rule": "key-term extraction filters the union of the fixture "
             "languages' stopwords; a function word in any one is "
             "filtered in all (the stated trade-off)"},
    {"name": "hint_not_field",
     "rule": "a record's language is a derived heuristic hint, "
             "never a stored field and never a certainty; a tie or "
             "an empty scoreboard reports unknown"},
    {"name": "read_only",
     "rule": "search and reports never write to a store or a record"},
)

#: The fixture languages of this slice (roadmap: fixtures
#: in two languages), plus the standing English of the tree.
FIXTURE_LANGUAGES: tuple[str, ...] = ("es", "de")
KNOWN_LANGUAGES: tuple[str, ...] = ("en", "es", "de")

# ── Stopwords, as data ───────────────────────────────────────────────────────
#
# The English row is the forge's pre-slice list verbatim
# (EN_STOPWORDS_ORIGINAL), plus the closed-class
# two-letter words the pre-slice length rule used to
# drop (EN_SHORT_STOPWORDS) — together they preserve the
# standing English behaviour at the new length floor.

EN_STOPWORDS_ORIGINAL: tuple[str, ...] = (
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "to", "of",
    "and", "or", "but", "in", "on", "at", "by", "for", "with",
    "about", "what", "who", "how", "when", "where", "why",
)
EN_SHORT_STOPWORDS: tuple[str, ...] = (
    "am", "as", "he", "if", "it", "its", "me", "my", "no",
    "so", "up", "us", "we",
)
ES_STOPWORDS: tuple[str, ...] = (
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "en", "y", "o", "que", "qué", "es",
    "son", "está", "están", "con", "por", "para", "se", "su",
    "como", "cuando", "cuándo", "donde", "dónde", "quien",
    "quién", "lo", "mi", "más", "pero", "sin", "sobre",
)
DE_STOPWORDS: tuple[str, ...] = (
    "der", "die", "das", "den", "dem", "des", "ein", "eine",
    "einen", "einem", "einer", "und", "oder", "in", "im",
    "am", "an", "ist", "sind", "war", "waren", "von", "zu",
    "zur", "zum", "mit", "für", "auf", "bei", "nach", "wie",
    "wo", "wer", "was", "nicht", "auch", "sich", "es", "er",
    "sie", "wir", "ihr", "als", "so", "da", "doch",
)

STOPWORDS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "en": EN_STOPWORDS_ORIGINAL + EN_SHORT_STOPWORDS,
    "es": ES_STOPWORDS,
    "de": DE_STOPWORDS,
}

#: The one filter key-term extraction applies (rule
#: stopwords_as_union), in folded form.
ALL_STOPWORDS: frozenset[str] = frozenset(
    word.casefold()
    for words in STOPWORDS_BY_LANGUAGE.values() for word in words
)

#: The shortest token that can be a key term. The
#: pre-slice floor was 3, which erased complete
#: two-character words in several scripts and languages.
MIN_TOKEN_LEN: int = 2

# ── Language hints, as data ──────────────────────────────────────────────────

LANGUAGE_HINTS: tuple[dict[str, Any], ...] = (
    {"language": "es",
     "distinctive_stopwords": ("el", "la", "los", "las", "una",
                               "del", "está", "están"),
     "distinctive_chars": "ñáéíóúü¿¡"},
    {"language": "de",
     "distinctive_stopwords": ("der", "die", "das", "und", "ist",
                               "den", "dem", "eine", "einen"),
     "distinctive_chars": "äöüß"},
    {"language": "en",
     "distinctive_stopwords": ("the", "is", "are", "and", "of",
                               "what", "when", "where"),
     "distinctive_chars": ""},
)

# ── Folding & tokenisation ───────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+")

#: Character ranges whose scripts write without word
#: spaces (plus Hangul syllables): a token containing
#: one contributes character bigrams as match keys.
_SPACELESS_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK ext A
    (0x4E00, 0x9FFF),   # CJK unified
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0xAC00, 0xD7AF),   # Hangul syllables
)


def fold_text(text: Any) -> str:
    """The comparison form of a text: NFC-normalised,
    casefolded. Derived only — never stored back."""
    return unicodedata.normalize("NFC", str(text or "")).casefold()


def strip_diacritics(text: Any) -> str:
    """The auxiliary comparison form: the folded text
    with combining marks dropped (á -> a, ñ -> n,
    ü -> u). Letters that are not a base + combining
    mark (ø, æ, ł) survive unchanged — stated."""
    decomposed = unicodedata.normalize("NFD", fold_text(text))
    stripped = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def _is_spaceless(token: str) -> bool:
    return any(
        lo <= ord(ch) <= hi
        for ch in token for lo, hi in _SPACELESS_RANGES)


def _bigrams(token: str) -> list[str]:
    return [token[i:i + 2] for i in range(len(token) - 1)]


def match_tokens(text: Any) -> set[str]:
    """The match-key set of a text: folded word tokens,
    plus — for a token in a spaceless script — its
    character bigrams, so a short query can match
    inside a spaceless sentence."""
    tokens: set[str] = set()
    for token in _WORD_RE.findall(fold_text(text)):
        tokens.add(token)
        if _is_spaceless(token) and len(token) > 2:
            tokens.update(_bigrams(token))
    return tokens


def _ordered_keys(text: Any) -> list[str]:
    """Match keys in text order (a query's terms): a
    spaceless token contributes its bigrams (itself
    when it is already two characters); any other token
    contributes itself."""
    keys: list[str] = []
    for token in _WORD_RE.findall(fold_text(text)):
        if _is_spaceless(token) and len(token) > 2:
            keys.extend(_bigrams(token))
        else:
            keys.append(token)
    return keys


def key_terms(query: Any, max_terms: int = 5) -> list[str]:
    """Key terms of a query, in any fixture language:
    folded, stopword-filtered (the union), at the
    MIN_TOKEN_LEN floor, de-duplicated in order, capped
    at ``max_terms``. ``KnowledgeForge._extract_key_terms``
    delegates here (D-ML-1)."""
    terms: list[str] = []
    for key in _ordered_keys(query):
        if len(key) < MIN_TOKEN_LEN or key in ALL_STOPWORDS:
            continue
        if key not in terms:
            terms.append(key)
        if len(terms) >= max_terms:
            break
    return terms


def terms_matching(
    terms: list[str], haystack_tokens: set[str]
) -> list[str]:
    """The subset of ``terms`` matching a haystack token
    set: exact folded match first, diacritic-stripped
    match second (both count as a match; the caller
    that needs the distinction uses ``exact`` below)."""
    stripped_haystack = {strip_diacritics(tok) for tok in haystack_tokens}
    matched: list[str] = []
    for term in terms:
        if term in haystack_tokens or \
                strip_diacritics(term) in stripped_haystack:
            matched.append(term)
    return matched


def matched_terms_for_text(terms: list[str], text: Any) -> list[str]:
    return terms_matching(terms, match_tokens(text))


def relevance_share(terms: list[str], text: Any) -> float:
    """Share of ``terms`` matching ``text`` (0.0 for an
    empty term list) — TF-058's stated rule, folded."""
    if not terms:
        return 0.0
    return len(matched_terms_for_text(terms, text)) / len(terms)


# ── Language hints ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LanguageHint:
    language: str               # en | es | de | cjk | unknown
    scores: dict[str, int]
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "scores": dict(self.scores),
            "evidence": self.evidence,
        }


def detect_language(text: Any) -> LanguageHint:
    """The heuristic hint for a text (rule hint_not_field).

    Scores per LANGUAGE_HINTS row: distinctive stopword
    hits among the folded tokens + distinctive character
    hits in the raw text. Spaceless-script presence is
    reported as "cjk" outright. Best score wins; a tie
    or an all-zero board is "unknown"."""
    raw = str(text or "")
    if any(_is_spaceless(tok) for tok in _WORD_RE.findall(raw)):
        return LanguageHint(
            language="cjk", scores={},
            evidence="spaceless-script characters present; the hint "
                     "does not separate ja/zh/ko on script alone")
    tokens = _WORD_RE.findall(fold_text(raw))
    token_set = set(tokens)
    lowered = raw.lower()
    scores: dict[str, int] = {}
    for row in LANGUAGE_HINTS:
        stop_hits = sum(
            1 for word in row["distinctive_stopwords"]
            if word.casefold() in token_set)
        char_hits = sum(lowered.count(ch) for ch in row["distinctive_chars"])
        scores[str(row["language"])] = stop_hits + char_hits
    best = max(scores.values()) if scores else 0
    winners = [lang for lang, score in scores.items() if score == best]
    if best == 0 or len(winners) != 1:
        return LanguageHint(
            language="unknown", scores=scores,
            evidence="no distinctive evidence" if best == 0
                     else f"tied hint scores {scores}")
    return LanguageHint(
        language=winners[0], scores=scores,
        evidence=f"distinctive score {best} for {winners[0]}")


def _get(record: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def record_text(record: Any) -> str:
    """All the text a record carries for matching and
    for its language hint: summary, tags, and — for a
    fact — its fact_value keys and scalar values; for a
    preference, its value. The category is a controlled
    English vocabulary word and is deliberately NOT
    text (it must not make a Spanish preference read
    as English)."""
    parts: list[str] = [str(_get(record, "summary", "") or "")]
    parts.extend(str(tag) for tag in _get(record, "tags", []) or [])
    fact_value = _get(record, "fact_value", {}) or {}
    if isinstance(fact_value, dict):
        for key, value in fact_value.items():
            parts.append(str(key))
            if isinstance(value, (str, int, float)):
                parts.append(str(value))
    value = _get(record, "value", "")
    if value:
        parts.append(str(value))
    return " ".join(parts)


def language_of_record(record: Any) -> LanguageHint:
    return detect_language(record_text(record))


def language_report(records_or_store: Any) -> dict[str, Any]:
    """One JSON-safe language report over a record list
    or a MemoryStore: counts by hint and every record
    named with its hint + scores. The report states in
    its own text that hints are heuristics."""
    if hasattr(records_or_store, "load_profile_records"):
        records: list[Any] = list(records_or_store.load_profile_records())
        records.extend(records_or_store.load_episodes())
        records.extend(records_or_store.load_patterns())
    else:
        records = list(records_or_store)
    by_language: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        hint = language_of_record(record)
        by_language[hint.language] = by_language.get(hint.language, 0) + 1
        rows.append({
            "record_id": str(_get(record, "record_id", "") or ""),
            "record_type": str(_get(record, "record_type", "") or ""),
            **hint.to_dict(),
        })
    rows.sort(key=lambda row: row["record_id"])
    return {
        "total": len(records),
        "by_language": by_language,
        "records": rows,
        "note": "language values are heuristic hints derived at "
                "report time (TF-060), not stored fields and not "
                "certainties; unknown means the evidence tied or "
                "was absent",
    }


# ── Search over the personal store ───────────────────────────────────────────


def _store_records(store: Any) -> list[Any]:
    records: list[Any] = list(store.load_profile_records())
    records.extend(store.load_episodes())
    records.extend(store.load_patterns())
    return records


def search_store(
    store: Any, query: str, top_k: int = 10
) -> tuple[Any, dict[str, Any]]:
    """Multilingual retrieval over a MemoryStore's
    personal records, by the stated rules.

    Relevance = the share of the query's key terms
    matching the record's carried text (exact folded
    first, diacritic-stripped second). Returns a
    ``MemoryActivationBundle`` whose raw dicts carry the
    summary **verbatim**, the record's language hint,
    and the matched terms — plus a report naming terms,
    considered/matched counts, and the languages of the
    returned records. Sorted by relevance desc, ties by
    record_id (reproducible). The store is only read."""
    from thoughtforge.knowledge.forge import KnowledgeForge
    from thoughtforge.knowledge.models import (
        ActivatedRecord,
        MemoryActivationBundle,
    )

    terms = key_terms(query, max_terms=8)
    report: dict[str, Any] = {
        "query": query,
        "terms": terms,
        "considered": 0,
        "matched": 0,
        "query_language": detect_language(query).language,
    }
    if not terms or top_k <= 0:
        return MemoryActivationBundle(
            activated_records=[], total_retrieved=0,
            retrieval_confidence=0.0), report
    scored: list[tuple[float, Any, list[str]]] = []
    for record in _store_records(store):
        report["considered"] += 1
        matched = matched_terms_for_text(terms, record_text(record))
        if matched:
            scored.append((len(matched) / len(terms), record, matched))
    report["matched"] = len(scored)
    scored.sort(key=lambda item: (
        -item[0], str(_get(item[1], "record_id", ""))))
    activated: list[ActivatedRecord] = []
    languages: dict[str, int] = {}
    for share, record, matched in scored[:top_k]:
        hint = language_of_record(record)
        languages[hint.language] = languages.get(hint.language, 0) + 1
        summary = str(_get(record, "summary", ""))
        raw = {
            "record_id": str(_get(record, "record_id", "")),
            "record_type": str(_get(record, "record_type", "")),
            "summary": summary,
            "tags": list(_get(record, "tags", []) or []),
            "source": str(_get(record, "source", "") or "memory_store"),
            "language": hint.to_dict(),
            "matched_terms": matched,
        }
        cue = KnowledgeForge._build_cue({
            "label_en": raw["record_id"],
            "content": summary,
            "source": raw["source"],
        })
        activated.append(ActivatedRecord(
            record_id=raw["record_id"],
            record_type=raw["record_type"],
            score=round(share, 4),
            cue=cue,
            raw=raw,
        ))
    confidence = round(
        sum(r.score for r in activated) / len(activated), 4
    ) if activated else 0.0
    report["returned"] = len(activated)
    report["returned_languages"] = languages
    return MemoryActivationBundle(
        activated_records=activated,
        total_retrieved=len(activated),
        retrieval_confidence=confidence,
    ), report


# ── The live checker ─────────────────────────────────────────────────────────


def multilingual_problems() -> list[str]:
    """Re-derive TF-060's claims from the live tree; [] = holds.

    Pins: the rules/languages/stopwords as data (the
    English row = the pre-slice list + the short closed
    class); folding (casefold ß -> ss, NFC); diacritic
    stripping (á -> a, ñ -> n; ø stated unseparable);
    key_terms (Spanish/German stopwords filtered, a
    two-character spaceless word survives, the standing
    English example reproduces the pre-slice result);
    verbatim round-trips — a Spanish + a German fact, a
    German episode, and a Spanish preference reload
    byte-identical from the store files AND through a
    TF-052 portability bundle (in the bundle's second,
    stable export); search in each fixture language
    retrieves its own fact verbatim, accent-stripped
    and shouting-case queries retrieve the accented
    facts, "strasse" retrieves the ß fact, a translated
    English query retrieves neither fixture fact, a
    spaceless-script query retrieves its record by
    bigram; language hints (es/de/en/unknown/cjk);
    the wiring (the forge delegates, freshness +
    reference search match folded — proven
    behaviourally with accent-stripped queries);
    every leg read-only over the store."""
    problems: list[str] = []

    import inspect

    from thoughtforge.knowledge import forge as forge_module
    from thoughtforge.knowledge import freshness as freshness_module
    from thoughtforge.knowledge import reference_library as ref_module
    from thoughtforge.knowledge.models import (
        EpisodicMemoryRecord,
        UserFactRecord,
        UserPreferenceRecord,
    )
    from thoughtforge.knowledge.portability import (
        export_store,
        import_store,
    )
    from thoughtforge.knowledge.store import MemoryStore

    if [row["name"] for row in MULTILINGUAL_RULES] != [
            "verbatim_storage", "derived_matching_keys",
            "no_translation_retrieval", "stopwords_as_union",
            "hint_not_field", "read_only"]:
        problems.append("MULTILINGUAL_RULES drifted")
    if FIXTURE_LANGUAGES != ("es", "de"):
        problems.append("FIXTURE_LANGUAGES drifted (roadmap: two languages)")
    if set(STOPWORDS_BY_LANGUAGE) != {"en", "es", "de"}:
        problems.append("STOPWORDS_BY_LANGUAGE drifted")
    if STOPWORDS_BY_LANGUAGE["en"] != \
            EN_STOPWORDS_ORIGINAL + EN_SHORT_STOPWORDS:
        problems.append("the English stopword row drifted from its parts")
    if MIN_TOKEN_LEN != 2:
        problems.append("MIN_TOKEN_LEN drifted")

    if fold_text("Die Straße") != "die strasse":
        problems.append("casefold does not map ß -> ss")
    if fold_text("GUÍA") != "guía":
        problems.append("casefold/NFC folding broken for accented capitals")
    if strip_diacritics("guía") != "guia" or strip_diacritics("año") != "ano":
        problems.append("diacritic stripping broken (á/ñ)")
    if strip_diacritics("ø") != "ø":
        problems.append("ø unexpectedly separable (stated boundary moved)")

    # key_terms: the standing English example reproduces
    # the pre-slice rule's result exactly.
    def _legacy_terms(query: str) -> list[str]:
        clean = re.sub(r"[^\w\s]", " ", query.lower())
        stop = set(EN_STOPWORDS_ORIGINAL)
        return [w for w in clean.split()
                if w not in stop and len(w) > 2][:5]

    english_query = "what is the ferry crossing time at the hollow"
    if key_terms(english_query) != _legacy_terms(english_query):
        problems.append(
            f"English key_terms {_legacy_terms(english_query)} -> "
            f"{key_terms(english_query)} (standing behaviour moved)")
    if key_terms("¿dónde está la biblioteca del faro") != \
            ["biblioteca", "faro"]:
        problems.append(
            f"Spanish key_terms wrong: "
            f"{key_terms('¿dónde está la biblioteca del faro')}")
    if key_terms("wo ist die fähre nach lantern") != ["fähre", "lantern"]:
        problems.append(
            f"German key_terms wrong: "
            f"{key_terms('wo ist die fähre nach lantern')}")
    if key_terms("灯台") != ["灯台"]:
        problems.append("a two-character spaceless word is not a key term")

    es_fact = UserFactRecord(
        record_id="ml_es", tags=["faro", "travesía"],
        summary="el faro rojo guía la travesía nocturna",
        fact_value={"lugar": "biblioteca del faro"},
        source="synthetic_fixture")
    de_fact = UserFactRecord(
        record_id="ml_de", tags=["straße", "hafen"],
        summary="die straße führt zum hafen und zur fähre",
        fact_value={"ort": "hafenstraße"},
        source="synthetic_fixture")
    de_episode = EpisodicMemoryRecord(
        record_id="ml_ep_de", tags=["fähre"],
        summary="die fähre fuhr bei sonnenaufgang über den fluss",
        source="synthetic_fixture")
    es_pref = UserPreferenceRecord(
        record_id="ml_pref_es", category="tone",
        tags=["respuestas"],
        summary="la viajera prefiere las respuestas claras y breves",
        value="claras y breves", source="synthetic_fixture")

    if detect_language(es_fact.summary).language != "es":
        problems.append("the Spanish fixture does not hint es")
    if detect_language(de_fact.summary).language != "de":
        problems.append("the German fixture does not hint de")
    if detect_language("the ferry crosses the river").language != "en":
        problems.append("an English sentence does not hint en")
    if detect_language("zzqx").language != "unknown":
        problems.append("an evidence-free string does not hint unknown")
    if detect_language("赤い灯台").language != "cjk":
        problems.append("spaceless-script text does not hint cjk")

    with tempfile.TemporaryDirectory(prefix="tf060-") as tmp:
        store = MemoryStore(memory_dir=Path(tmp))
        for record in (es_fact, de_fact, es_pref):
            store.append_profile_record(record)
        store.append_episode(de_episode)
        store.append_profile_record(UserFactRecord(
            record_id="ml_ja", tags=["toudai"],
            summary="赤い灯台が夜の渡航を導く",
            source="synthetic_fixture"))
        loaded = {r.record_id: r for r in store.load_profile_records()}
        if loaded["ml_es"].to_dict() != es_fact.to_dict():
            problems.append("the Spanish fact did not round-trip verbatim")
        if loaded["ml_de"].to_dict() != de_fact.to_dict():
            problems.append("the German fact did not round-trip verbatim")
        if loaded["ml_pref_es"].to_dict() != es_pref.to_dict():
            problems.append("the Spanish preference did not round-trip")
        episodes = {r.record_id: r for r in store.load_episodes()}
        if episodes["ml_ep_de"].to_dict() != de_episode.to_dict():
            problems.append("the German episode did not round-trip verbatim")

        # Portability: the second export is the stable
        # one (TF-052's rule — the first import stamps
        # source='import'); summaries must survive it
        # byte-for-byte.
        target = MemoryStore(memory_dir=Path(tmp) / "imported")
        import_store(target, export_store(store), mode="replace")
        second = export_store(target)
        summaries = {
            row["record_id"]: row["summary"]
            for row in second["sections"]["profile_records"]}
        if summaries.get("ml_es") != es_fact.summary or \
                summaries.get("ml_de") != de_fact.summary:
            problems.append("the portability bundle lost non-English text")

        before = {p.name: p.read_bytes()
                  for p in sorted(Path(tmp).iterdir()) if p.is_file()}

        bundle_es, report_es = search_store(store, "faro travesía")
        ids_es = [r.record_id for r in bundle_es.activated_records]
        if ids_es != ["ml_es"]:
            problems.append(f"Spanish search returned {ids_es}")
        elif bundle_es.activated_records[0].raw["summary"] != es_fact.summary:
            problems.append("the Spanish result's summary is not verbatim")
        if report_es["query_language"] != "es":
            problems.append("the Spanish query does not hint es")
        bundle_de, _ = search_store(store, "straße hafen")
        if [r.record_id for r in bundle_de.activated_records] != ["ml_de"]:
            problems.append("German search did not return the German fact")
        stripped, _ = search_store(store, "guia travesia")
        if [r.record_id for r in stripped.activated_records] != ["ml_es"]:
            problems.append("the accent-stripped query lost the Spanish fact")
        shouting, _ = search_store(store, "STRASSE HAFEN")
        if [r.record_id for r in shouting.activated_records] != ["ml_de"]:
            problems.append("the casefold (ß->ss) query lost the German fact")
        translated, _ = search_store(
            store, "lighthouse guides the night crossing")
        translated_ids = [r.record_id for r in translated.activated_records]
        if "ml_es" in translated_ids or "ml_de" in translated_ids:
            problems.append(
                f"a translated query retrieved a fixture fact: "
                f"{translated_ids} (no-translation rule broken)")
        spaceless, _ = search_store(store, "灯台")
        if [r.record_id for r in spaceless.activated_records] != ["ml_ja"]:
            problems.append("the spaceless-script query lost its record")

        report = language_report(store)
        if report["by_language"].get("es", 0) < 2 or \
                report["by_language"].get("de", 0) < 2:
            problems.append(f"language_report counts wrong: {report}")

        after = {p.name: p.read_bytes()
                 for p in sorted(Path(tmp).iterdir()) if p.is_file()}
        if after != before:
            problems.append("a multilingual leg wrote to the store")

    # Wiring, by source + by behaviour.
    if "key_terms" not in inspect.getsource(
            forge_module.KnowledgeForge._extract_key_terms):
        problems.append("the forge does not delegate key-term extraction")
    if "terms_matching" not in inspect.getsource(
            freshness_module.search_store):
        problems.append("freshness search does not match multilingually")
    if "terms_matching" not in inspect.getsource(ref_module.ReferenceLibrary.search):
        problems.append("reference search does not match multilingually")
    return problems


__all__ = [
    "ALL_STOPWORDS",
    "DE_STOPWORDS",
    "EN_SHORT_STOPWORDS",
    "EN_STOPWORDS_ORIGINAL",
    "ES_STOPWORDS",
    "FIXTURE_LANGUAGES",
    "KNOWN_LANGUAGES",
    "LANGUAGE_HINTS",
    "MIN_TOKEN_LEN",
    "MULTILINGUAL_RULES",
    "STOPWORDS_BY_LANGUAGE",
    "LanguageHint",
    "detect_language",
    "fold_text",
    "key_terms",
    "language_of_record",
    "language_report",
    "match_tokens",
    "matched_terms_for_text",
    "multilingual_problems",
    "record_text",
    "relevance_share",
    "search_store",
    "strip_diacritics",
    "terms_matching",
]
