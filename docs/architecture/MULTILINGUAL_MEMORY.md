# TF-060 — Multilingual Memory: No Translation Loss

*Part III, slice TF-060. Code:
`src/thoughtforge/knowledge/multilingual.py` (the rules,
stopwords, hints, folding, search, report, and
`multilingual_problems()` re-deriving every claim below
from the live tree), plus wiring in
`knowledge/forge.py` (D-ML-1), `knowledge/freshness.py`,
`knowledge/reference_library.py`, and
`knowledge/forge_toolset.py`. Tests:
`tests/test_tf060_multilingual.py`. Fixture languages:
**Spanish** and **German** — synthetic Lantern Hollow
material (TF-017), never canon.*

## 1. The question and the measured starting state

The roadmap asks: can non-English facts be stored and
retrieved without translation loss? Measured this tick
by direct probe:

- **Storage was already lossless — and unproven.** The
  store writes UTF-8 with `ensure_ascii=False`, and a
  Spanish and a German fact round-tripped byte-for-byte
  through the store files in the probe. No test, and no
  other leg (portability, tag index), had ever checked.
- **Retrieval term extraction was English-only
  (D-ML-1).** `KnowledgeForge._extract_key_terms` — the
  term source for the forge's ConceptNet/GeoNames legs
  and for both personal-store searches built in Part III
  (TF-058, TF-059) — had an English-only stopword list,
  dropped every token of length ≤ 2, and matched with
  `lower()`:
  - the accent-stripped query `guia travesia` retrieved
    **0** records against a store holding *guía la
    travesía*, while the accented query retrieved it;
  - the query `strasse` retrieved **0** against a record
    tagged *straße* (`lower()` keeps ß as ß; `casefold()`
    maps it to `ss`), while `straße` retrieved it;
  - the two-character query `灯台` (a complete word,
    lighthouse) extracted **zero terms** under the length
    rule, and a whole spaceless sentence tokenised as
    one `\w+` token, unmatchable from any shorter query;
  - Spanish/German function words (*el, la, die, das,
    und, está*) were kept as key terms — noise in every
    non-English query.
- **Nothing could say which language a record was in.**
  The record shapes (TF-042) carry no language field.

## 2. The position, as data

`MULTILINGUAL_RULES`:

| Rule | Statement |
|---|---|
| verbatim_storage | A record is stored and returned in its own language, byte-for-byte; no leg translates, transliterates, or normalises the stored text in place. |
| derived_matching_keys | Matching compares derived keys only — NFC-casefolded forms first, diacritic-stripped forms second; the record on disk is never rewritten to match. |
| no_translation_retrieval | A query retrieves a record stored in another language only on tokens they genuinely share (proper names, cognates, tags) — never by translation equivalence. Translation is the model's job at answer time, not the store's at retrieval time. |
| stopwords_as_union | Key-term extraction filters the union of the fixture languages' stopwords; a function word in any one is filtered in all (the stated trade-off: the German article *die* also filters the English verb *die*). |
| hint_not_field | A record's language is a derived heuristic hint, never a stored field and never a certainty; a tie or an empty scoreboard reports `unknown`. |
| read_only | Search and reports never write. |

## 3. Folding, stripping, tokenising

- `fold_text` — the comparison form: NFC-normalised
  `casefold()` (ß → ss, accented capitals fold, a
  decomposed `a` + combining acute equals precomposed
  `á`). Derived at query time only.
- `strip_diacritics` — the auxiliary form: NFD, drop
  combining marks (`á → a`, `ñ → n`, `ü → u`). This is
  what lets a plain-keyboard query find the accented
  fact. Stated cost: it can conflate words that differ
  only by a mark (Spanish *año/ano* is the textbook
  case) — a retrieval convenience over unaltered text,
  never a rewrite. Letters that are not base + mark
  (ø, æ, ł) survive both forms unchanged.
- Tokenisation — folded `\w+` word tokens; a token in
  a spaceless script (CJK, kana, Hangul ranges, as
  data) also contributes its character **bigrams**, so
  `灯台` matches inside `赤い灯台が夜の渡航を導く`.
- Key terms — stopwords as data per language
  (`STOPWORDS_BY_LANGUAGE`: en/es/de; the English row
  is the pre-slice list verbatim plus the closed-class
  two-letter words the old length rule used to drop),
  filtered as one union, length floor **2** (down from
  3). Pinned: the standing English example
  ("what is the ferry crossing time at the hollow")
  reproduces the pre-slice rule's terms exactly.

## 4. Language hints

`detect_language` scores each row of `LANGUAGE_HINTS`
(distinctive stopwords among the folded tokens +
distinctive characters in the raw text: ñ/accents/¿¡
for Spanish, ä/ö/ü/ß for German, function words for
English). Spaceless-script text is reported `cjk` —
the hint does not pretend to separate Japanese from
Chinese from Korean on script alone. Best score wins;
tie or zero → `unknown`. Records gain **no** language
field (TF-042's shape is untouched): the hint is
derived wherever it is shown, and `language_report`
says so in its own `note`.

`language_report(records_or_store)` gives counts by
hint and every record named with its hint and scores.

## 5. Search over the personal store

`search_store(store, query)` ranks profile records,
episodes, and patterns by the share of the query's key
terms matching the record's carried text (summary +
tags + fact_value keys/values / preference value — the
preference *category* is a controlled English
vocabulary word and is deliberately not text, lest it
make a Spanish preference read as English). Matching
is exact-folded first, diacritic-stripped second. The
returned bundle's raw carries the summary **verbatim**,
the language hint, and the matched terms; the report
names the query's terms, its own language hint, and
the languages of what came back. Read-only, byte-pinned.

Proven on the fixtures (synthetic, this VM):

- `faro travesía` → the Spanish fact, score 1.0,
  summary byte-identical to what was stored;
- `straße hafen` → the German fact; `STRASSE HAFEN`
  (casefold) and `guia travesia` (accent-stripped)
  retrieve the same facts at the same scores;
- `lighthouse guides the night crossing` → **neither**
  fixture fact: no translation happens at retrieval;
- `灯台` → the spaceless-script record, by bigram.

## 6. D-ML-1 — the wiring

- `KnowledgeForge._extract_key_terms` delegates to
  `multilingual.key_terms` — its ConceptNet/GeoNames
  legs, and every caller, inherit the fix.
- TF-059's `freshness.search_store` and TF-058's
  `ReferenceLibrary.search` match through
  `match_tokens` / `terms_matching` instead of a
  `lower()` set intersection (their English pins —
  including the reference search's 1-of-2 = 0.5 — are
  unchanged, proven by their suites).
- TF-057's `forge_toolset.inspect_records` text filter
  compares folded forms: `STRASSE` finds *straße*.

## 7. Docket

- **D-ML-1 added + FIXED by this slice** (see §1/§6).
  Totals: 56 entries — 40 OPEN / 1 OPEN-ENVIRONMENT /
  13 FIXED / 2 CLOSED-BY-VERDICT.

## 8. Boundaries, stated

- The knowledge DB's FTS5 leg is not rewired:
  `_to_fts5_query` and the entities' `label_en` columns
  are the DB estate's shape — its labels table holds
  per-language labels the forge never reads, and that
  is TF-106/TF-107's ingestion territory. This slice's
  claim is the personal memory store (Part III's
  subject) plus the term legs named in §6.
- Detection is a heuristic over two fixture languages +
  English + a script label, not a language identifier
  for the world's languages; short or proper-name-only
  records honestly report `unknown`.
- Vector retrieval quality in Spanish/German is NOT
  PROVEN: TF-053's one registered embedding model is
  English-leaning, and this slice proves the
  deterministic legs only.
- Proof grade is FIXTURE (synthetic Spanish + German
  stores, this VM). Behaviour over a real lived-in
  multilingual store is **NOT PROVEN** (no real store
  on this machine, TF-041), and live-model answers in
  either language are NOT PROVEN — never claimed.
