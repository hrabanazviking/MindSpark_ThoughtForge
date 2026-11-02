# RESEARCH FENCE — this tree is fenced research (TF-148)

GENERATED from the filed data in `src/thoughtforge/research_hygiene.py` — regenerate with `python3 -m thoughtforge.research_hygiene`.

Everything under `research_data/` is **fenced research**: provenance, design, spec and verdict evidence for ThoughtForge. No file in this tree is shipping code, and no shipping file imports this tree's package.

## What graduated

- The bond graph service graduated in TF-143 — the shipping code is `src/thoughtforge/knowledge/bond_graph.py`.
- The persona compiler service graduated in TF-143 — the shipping code is `src/thoughtforge/cognition/persona_compiler.py`.

## What is archived-retired

- The micro-RAG pipeline service, the truth calibrator service and their shared test file are archived-in-place retired by TF-144, each under its filed retirement banner (TF-140 / TF-142 verdicts). They stand as verdict evidence.

## What remains fenced research or evidence

- The design and spec markdown and the wyrd_runtime packets: fenced reference.
- The 14 schemas: fenced evidence, kept unmarked.
- Config and examples: fenced examples. The scripts generator: fenced tooling.
- The memory store, the runtime demo seed and the memory/persona test file: fenced research — they import the missing models package, cannot run, and are not shipping.
- The two security guards: fenced standalone research — importable on their own, still not adopted.
- The package init: the filed broken-import state — it re-exports ten names from the missing models package. The wing cannot import, and that state is filed here, not hidden.

## Its own test discipline

This tree is its own package (see its `pyproject.toml`) with its own testpaths. The campaign suite at the repo root never collects its tests. Running its own suite errors at collection while the models package is missing — a filed state (TF-006). TF-149 owns the wing tests' final disposition.

## Revival

Reviving, importing or building on any fenced or archived file in this tree needs a new roadmap word. The full filing is `docs/architecture/RESEARCH_HYGIENE.md`.
