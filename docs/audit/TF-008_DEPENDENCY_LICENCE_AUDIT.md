# TF-008 — Dependency & Licence Audit (Part I Audit)

Measured 2026-10-30. Declared dependencies from `pyproject.toml` +
`requirements.txt` (they agree) and `research_data/pyproject.toml`;
actual use from an AST import scan of `src/` + the root scripts;
licences from PyPI package metadata (license / license_expression /
OSI classifiers) retrieved this run. Versions below are the latest
PyPI releases at audit time, not pins — the project declares only
lower bounds everywhere (no lockfile exists in the tree).

## Declared core dependencies — licence + actual use

| Package | Licence (PyPI) | Imported? |
|---|---|---|
| sqlalchemy >=2.0 | MIT | **NO — nowhere in src/ or root scripts** (the knowledge DB is raw sqlite3, etl/schema.py) |
| sentence-transformers >=2.7 | Apache-2.0 | yes (etl/embeddings.py) — also the D-ENV-1 missing package on this VM (TF-004) |
| ijson >=3.2 | BSD-3-Clause AND ISC | yes (etl) |
| numpy >=1.26 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (bundled components) | yes |
| tqdm >=4.66 | MPL-2.0 AND MIT (dual) | yes |
| pyyaml >=6.0 | MIT | yes |
| click >=8.1 | BSD-3-Clause | only in root script forge_memory.py — **not** by the package (see finding 3) |
| rich >=13.0 | MIT | **NO — nowhere** |
| platformdirs >=4.0 | MIT | **NO — nowhere** (utils/paths.py hand-rolls path logic) |

## Declared optional extras

| Extra | Packages (licence) | Use |
|---|---|---|
| inference | llama-cpp-python (MIT) | yes, lazily in turboquant.py:388 |
| quantization | bitsandbytes (MIT), auto-gptq (MIT) | no imports found in src/ — declared, unused so far |
| gpu | torch (BSD-3-Clause family + Apache-2.0/LLVM-exception components per its SPDX expression), accelerate (Apache-2.0) | torch lazily (backends/profiles/onnx_export, all try-guarded); accelerate not imported |
| dev | pytest (MIT), pytest-cov (MIT), ruff (MIT), mypy (MIT), locust (MIT) | tooling |
| docs | mkdocs (BSD), mkdocs-material (MIT), mkdocstrings (ISC) | tooling |
| build | setuptools (MIT), wheel (MIT) | build only |
| research wing | pydantic (MIT), PyYAML (MIT) | research_data/pyproject.toml |

## Undeclared imports actually in the code

| Package | Licence | Where | Guarded? |
|---|---|---|---|
| **requests (Apache-2.0)** | — | **top-level import in 4 inference modules** (hf_backend.py:12, lmstudio_backend.py:12, model_browser.py:13, ollama_backend.py:10) | **NO — hard, unconditional** |
| transformers (Apache-2.0) | | onnx_export.py (3 lazy sites) | lazy, in-function |
| huggingface_hub (Apache-2.0) | | hf_backend.py, model_browser.py, setup_thoughtforge.py — lazy | lazy |
| onnxruntime (MIT) | | onnx_export.py:270 | try-guarded |
| optimum (Apache-2.0) | | onnx_export.py:121/160 | lazy |
| psutil (BSD-3-Clause) | | wikidata.py:339, profiles.py:122 | try-guarded |

requests works today only because it arrives transitively (e.g.
via sentence-transformers). In a minimal install of the declared
core set, importing any backend module breaks. transformers and
huggingface_hub likewise ride transitively; onnxruntime, optimum
and psutil are declared **nowhere at all**.

## Findings

1. **No copyleft contamination in the Python dependency set.**
   Every declared or imported package is permissive
   (MIT / BSD / Apache-2.0 / ISC); the only non-pure-permissive
   entry is tqdm's MPL-2.0+MIT dual licence (file-level copyleft,
   no effect on this project's own licensing). Nothing to purge.
2. **The project's own licence declaration contradicts itself —
   the real contamination is internal.** pyproject declares
   `license = { text = "CC BY 4.0" }` (also MODEL_CARD.md:170),
   while its own classifier says "License :: Free for
   non-commercial use" — CC BY 4.0 *permits* commercial use, so
   the two statements cannot both be true. There is **no LICENSE
   file anywhere in the repo root**, and CC BY is a content
   licence being applied to code. Fix (named, for TF-190 with
   Volmarr's word — his licence choice is his): pick one licence
   statement, ship the LICENSE file, drop the contradictory
   classifier. → TF-016 docket D-LIC-1.
3. **Broken console scripts**: `[project.scripts]` points at
   `thoughtforge.cli:main` and `thoughtforge.etl.cli:main` — no
   `cli.py` exists in either package (verified: find src -name
   'cli*' is empty). `pip install` succeeds; both commands crash
   on first run. → TF-016 docket; the fix belongs with the
   entry-point work (TF-005 proved the root scripts run; the
   packaged entry points were never exercised).
4. **requests is an undeclared hard dependency** (table above).
   Fix: add requests to core dependencies, or move the backend
   imports behind the same lazy pattern the rest use — the
   contract slice TF-065 owns the choice. → TF-016 docket D-DEP-1.
5. **Declared-but-unused core weight**: sqlalchemy, rich,
   platformdirs are promised to every installer and used by no
   code; bitsandbytes/auto-gptq/accelerate are declared extras
   with no imports. On phone_low budgets, install weight is a
   feature cost — TF-069/TF-110 should weigh removal. → TF-016.
6. **No pins, no lockfile**: every requirement is a bare lower
   bound, so "the dependency set" is a moving target (this audit's
   PyPI versions already exceed several floors). A reproducible
   baseline needs at least a tested-version record — TF-014 owns.
7. **Data-source licences are a separate, unverified ledger.**
   The ETL ingests ConceptNet, GeoNames, DBpedia, Wikidata
   (sources.py / wikidata.py headers) and the library ships the
   D&D 5e SRD (TF-007 finding 6) — share-alike and attribution
   terms upstream were **not verified in this slice** and are
   recorded nowhere in-tree. Flagged, not asserted: TF-106's
   source registry must record each source's licence from its
   upstream terms before any redistribution claim; TF-190 closes.
   The default embedding model all-MiniLM-L6-v2
   (configs/default.yaml:13) is a model artefact, not a package —
   its model-card licence is likewise for TF-053 to record.
