# TF-007 — Data Census, Metadata Only (Part I Audit)

Measured 2026-10-30 from the GitHub git-tree of `development`
(d267148 — the campaign base; `data/` is unchanged at campaign HEAD)
via the API tree listing, cross-checked against `git ls-tree` in the
working tree. **Metadata only**: names, sizes, blob hashes, and
provenance notes from code/config references. No canon file was
opened, copied, or quoted. Individual picture filenames are not
listed — imagery is counted in aggregate only.

## Headline numbers

| Holding | Files | Bytes | Share of repo |
|---|---|---|---|
| `data/` (all) | 484 | 242,173,121 | 70.1% of 345,351,298 B total |
| — `data/` root datasets | 27 | 25,769,159 | |
| — `data/knowledge_reference/` root files | 215 | 62,006,186 | |
| — `data/knowledge_reference/pictures_of_sigrid/` | 225 | 152,862,826 | 63.1% of all `data/` |
| — `data/knowledge_reference/dnd-5e-srd-markdown/` | 17 | 1,534,950 | |

By extension under `data/`: .jpg 225 files / 152,862,826 B ·
.md 193 / 39,664,449 B · .jsonl 25 / 36,647,009 B ·
.json 27 / 12,459,287 B · .csv 1 / 415,475 B · .yaml 8 / 112,890 B ·
.py 1 / 5,023 B · .txt 3 / 4,978 B · .rst 1 / 1,184 B.

Adjacent data-like holdings outside `data/` (for scale, not this
slice's subject): `docs/research/` holds 15 research PDFs /
70,336,160 B; `assets/` holds 7 media files / 15,897,377 B.

## The root datasets (data/, 27 files)

Volmarr-authored/generated Norse & Viking training-style sets —
JSONL conversation/fact sets and JSON/YAML reference sets, a
"volume1" series throughout. Largest first:

| Bytes | File |
|---|---|
| 10,272,549 | Viking_combined_dataset_volume1.jsonl |
| 1,841,597 | Norse_Gods_and_Goddesses_Personality_Traits_Volume1.jsonl |
| 1,829,564 | viking_everyday_conversations_complete_volume1.jsonl |
| 1,805,730 | norse_paganism_1000_training_pairsv1.jsonl |
| 1,805,730 | norse_paganism_1000_training_pairsv2.jsonl |
| 1,696,149 | Norse_Magick_Spells_and_Rituals.json |
| 1,110,641 | viking_social_and_political_ideas_volume1.jsonl |
| 1,054,539 | trolldom_and_magick_practices_in_norse_paganism_volume1.jsonl |
| 1,006,454 | viking_life_everyday_grounding_questions_dataset_volume1.jsonl |
| 826,522 | viking_sailing_travel_trade_raiding_volume1.jsonl |
| 735,984 | famous_legendary_and_heroic_vikings_volume1.jsonl |
| 719,008 | viking_history_and_important_events_volume1.jsonl |
| 589,157 | Authentic_Norse_Religious_Practices.json |
| 171,199 | Poetic_Edda_Translation.json |
| 66,258 | Viking_Era_Cities.json |
| 57,559 | about_norse_paganism.json |
| 37,028 | Speak_Like_a_Viking.json |
| 26,368 | viking_values.yaml |
| 22,216 | voluspa.json |
| 17,958 | Voluspa_the_Seeresss_Vision_the_Ultimate_Poetic_Rendering.jsonl |
| 16,757 | viking_and_norse_pagan_social_protocols.json |
| 16,757 | viking_social_protocols.json |
| 16,466 | Viking_bondmaids.json |
| 8,147 | viking_trolldom_the_ancient_northern_ways.yaml |
| 7,324 | Viking_Frith.json |
| 6,533 | Viking_Honor.json |
| 4,965 | viking_cultural_practices.yaml |

## knowledge_reference/ root (215 files, grouped)

The ingestion library that `ReferenceDataETL` reads
(`configs/default.yaml`: `reference_data_dir: data/knowledge_reference/`;
`src/thoughtforge/etl/sources.py`). Grouped by filename family:

| Family | Files | Bytes | Provenance note |
|---|---|---|---|
| sigrid_* (incl. sigrid_data_*) | 74 | 5,649,504 | Sigrid knowledge-build outputs + progress/final reports (SIGRID_KNOWLEDGE_BUILD_PROGRESS.md, SIGRID_KNOWLEDGE_FINAL_REPORT.md, DOMAIN_PROGRESS.md ride along) |
| guide_* | 58 | 1,823,603 | Topical guides (incl. tech-domain guides: AI, cybersecurity, data science, networking, software engineering, system administration) |
| 9th-century research reports | 4 | 23,970,702 | The four largest texts in the library (Finnish/Slavic/Celtic witch reports + Viking bondmaids report, 4.8–6.9 MB each) |
| book_* | 14 | 52,204 | Public-domain-classic texts (Bellows Poetic Edda, Brodeur Prose Edda, Dracula, Frankenstein, Meditations, The Prince, Tractatus, etc.) — licence audit is TF-008 |
| viking_* / norse_* / other topical .md/.json/.jsonl/.yaml | 65 | ~30.5 MB | Mixed authored reference sets; largest single file norse_gods.json (6,941,960 B) |
| dnd-5e-srd-markdown/ | 17 | 1,534,950 | D&D 5e SRD chapters 00legal–16npcs — third-party SRD text, licence → TF-008/TF-106 |

Control files present in the library root:
INSTRUCTIONS_FOR_KNOWLEDGE_GENERATION.md (1,997 B),
KNOWLEDGE_DOMAINS.md (3,617 B), and
PLEASE_INSTALL_DATA_FILES_HERE.txt (188 B) — the directory was
designed to also receive externally installed data files.

## Findings

1. **Most of `data/` is not ingestible knowledge text.**
   `pictures_of_sigrid/` (225 .jpg, 152.9 MB — 31 root + 194 in one
   subfolder, aggregated here deliberately) is persona imagery.
   `ReferenceDataETL`'s supported set is {.md, .json, .jsonl, .yaml,
   .yml, .txt} (sources.py:409) — images, the lone .csv, the .py and
   the .rst are never ingested by it.
2. **Stale count in code**: sources.py's docstring (line 8) and
   ReferenceDataETL's docstring both say "40 built-in files"; the
   library actually holds 215 root files + 17 SRD chapters. → TF-016
   docket (doc-drift, Part VI owns).
3. **Duplication: 37 duplicate blob hashes, 44 extra copies,
   28,109,539 B redundant.** Nearly every root dataset is copied
   byte-identically into knowledge_reference/ (11+ files). Sharper:
   `norse_paganism_1000_training_pairsv1.jsonl` and `...v2.jsonl`
   are the *same blob* (b5e13ae1) in both locations — v2 is not a
   second version in fact. Two differently-named protocol files
   (viking_and_norse_pagan_social_protocols.json /
   viking_social_protocols.json) are also one blob. → TF-016
   docket; dedup/merge rules are TF-109, canon handling stays
   Volmarr's word — nothing was moved or deleted by this census.
4. **No provenance/licence file inside `data/`**: the only
   readme-like file is guide_old_norse_dictionary_readme.md. Where
   each set came from is recorded nowhere in-tree except the Sigrid
   build reports — provenance for the external-source sets
   (ConceptNet/GeoNames/DBpedia/Wikidata per sources.py headers)
   lives only in ETL code comments. → TF-008 (licences) and TF-106
   (source registry) own the fix.
5. **No runtime stores or databases are committed** — clean.
   .gitignore excludes data/knowledge.db*, data/embeddings/, and
   the ETL caches (wikidata/dbpedia/yago/conceptnet/geonames);
   no .db/.sqlite/.onnx/.gguf blob exists anywhere in the tree.
   The runtime JSONL stores named in knowledge/store.py
   (user_profile_store, episodic_store, response_patterns) are
   generated at runtime, never committed.

## Verdict input for later parts
- TF-058/TF-118 (reference-library binding/ingestion) must reckon
  with findings 1–3: ingest the text library by rule, skip imagery
  by rule, and dedup by blob hash before chunking.
- TF-110 (profile-sized subsets): the text corpus is ~89 MB
  without imagery — still far above phone_low; subsetting is a
  real cut, not a copy.
