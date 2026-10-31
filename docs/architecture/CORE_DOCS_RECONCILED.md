# ThoughtForge Architecture — Core Docs Reconciled (TF-039)

Core docs: the architecture doc (TF-019) reconciled with shipped
behaviour; examples runnable. Built 2026-10-31 on branch
`yrsa/thoughtforge-road-build`.

The reconciliation lives as verifiable data in
`src/thoughtforge/cognition/core_docs.py`
(`RECONCILED_FINDINGS` + `findings_problems()`,
`POST_FREEZE_ADDITIONS` + `additions_problems()`, `EXAMPLES` +
`run_example()`); the tests are `tests/test_tf039_core_docs.py`.
TF-019's architecture doc (`CORE_LOOP.md`) carries a TF-039
status table in its §8; this doc is the slice's own write-up.

## 1. What was true before

TF-019 mapped the loop honestly and filed eight findings (F1–F8)
where code and story disagreed. TF-020–TF-038 then shipped the
fixes and rulings those findings proposed owners for — but three
surfaces were never brought back into line:

- `core.py`'s module docstring still told the old 9-step story
  (prompt-building and enforcement as top-level steps, fragment
  scoring omitted) — F4, TF-019's own proposed owner for it was
  this slice.
- `InputSketch`'s field comments in `knowledge/models.py` still
  described intents and modes the router never emits ("question",
  "command", "conversational | technical | supportive |
  creative") — drift TF-020 flagged for TF-034/TF-039.
- The TF-034 freeze gained minor additions from TF-035–TF-038
  that were recorded only in those slices' own docstrings, each
  deferring the record to "TF-039/TF-192".

And no worked example of using the core existed anywhere that a
reader could run.

## 2. The findings, reconciled (CORE_LOOP §8)

| Finding | Status at TF-039 | Owner / evidence |
|---|---|---|
| F1 `num_drafts` dead | **CLOSED** | TF-034 wired it as a real draft cap in both drivers |
| F2 retrieval gate checks the default DB path | **OPEN** | TF-041 owns the fix; still true today — `_retrieve` tests the default path's existence, not the configured one. Recorded open, not hidden |
| F3 turn state built and discarded | **CLOSED** | TF-025 retained it + measured `salvage_ms`; TF-038 made it exportable as a trace |
| F4 module docstring's 9 steps ≠ the code's steps | **CLOSED** | this slice (§3) |
| F5 memory triggers computed, not consumed | **RULED** | TF-020: trigger extraction is the router's whole contract; retrieval consumes `raw_text` by design |
| F6 two fragment thresholds | **RULED not-a-defect** | TF-022: 0.25 (record provenance) vs 0.54 (refine-worthiness) are different instruments; both constants cross-documented |
| F7 `CognitionScaffold.prompt_text` never filled | **CLOSED** | TF-023 removed the field |
| F8 brainstorming duplicates one mode | **CLOSED** | TF-021: now `[brainstorm_distinct, practical]`, registry-wide no-duplicate rule |

`findings_problems()` re-checks the machine-checkable evidence
against the live tree — signatures, thresholds, the registry,
the scaffold's fields, the docstring's steps, and F2's still-open
gate (if TF-041 lands its fix without flipping this record, the
checker names that too). TF-192 verifies at campaign end that no
finding stands open without a reason; F2's reason is named here
and in CORE_LOOP.

## 3. F4 closed: the docstring and the record comments

`core.py`'s module docstring now states the reconciled loop —
ROUTE → RETRIEVE → SCAFFOLD → GENERATE → FRAGMENT → COMPOSE →
QUALITY GATE — with prompt-building placed inside generation
(and the repair leg of the gate), enforcement inside compose,
the history/will/persona fold-in named at the scaffold step, and
`think()` / `think_stream()` / `resume_turn()` named as three
drivers of the one `_run_turn` pipeline. No behaviour changed.

`InputSketch`'s comments now name the router's true taxonomies:
the 9 intents, the 7 tones, and the 6 response modes, with the
`response_mode` default called out as a construction fallback
the router never emits. Defaults themselves are unchanged —
comments were reconciled, construction behaviour was not.

## 4. Post-freeze additions, recorded

Since the v1.0.0 freeze (TF-034), the core's surface grew by
minor additions only, each recorded in
`POST_FREEZE_ADDITIONS` and in CORE_API_FREEZE_V1.md §7:

- `determinism_mode` property and the `determinism` module
  (TF-036), `last_turn_trace` method and the `traces` module
  (TF-038), the `concurrency` module (TF-035), the
  `small_model_floor` module (TF-037), and
  `ConversationConfig.max_refinement_passes` (TF-037's D-PROF-4
  fix).

The frozen tables and `API_VERSION` are deliberately **not**
changed by this slice: under the freeze's own rules additions
are minor, and folding them into the tables with a 1.1.0 bump is
a freeze reconciliation act — TF-192 is named as its verifier
so the fold-in cannot happen silently or be forgotten.
`additions_problems()` proves every recorded addition exists
live exactly as named, and that the v1 freeze still verifies
clean (`verify_api_surface() == []`).

## 5. The examples (runnable)

Each example below is printed **verbatim** from
`core_docs.EXAMPLES` — the tests execute every one against the
real core and assert the measured results in §6, and assert
that this doc's copy matches the module's source character for
character. All five are synthetic-only (the invented River Sarn
ferry) and temp-dir-only; none needs a model or a knowledge DB.

### Example: knowledge_only_turn

```python
import tempfile
from pathlib import Path

from thoughtforge.cognition.core import ThoughtForgeCore

FERRY_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)


def make_core(**kwargs):
    """A persona-free core on a temp memory dir and no DB file."""
    tmp = Path(tempfile.mkdtemp())
    return ThoughtForgeCore(
        config={"cognition": {}},
        memory_dir=tmp / "memory",
        db_path=tmp / "none.db",
        **kwargs,
    )


def run():
    core = make_core()
    final = core.think("What time does the ferry leave?")
    return {
        "mode": final.mode,
        "text": final.text,
        "tokens": final.token_count,
    }
```

### Example: stub_backend_turn

```python
import tempfile
from pathlib import Path

from thoughtforge.cognition.core import ThoughtForgeCore

FERRY_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)


def make_core(**kwargs):
    """A persona-free core on a temp memory dir and no DB file."""
    tmp = Path(tempfile.mkdtemp())
    return ThoughtForgeCore(
        config={"cognition": {}},
        memory_dir=tmp / "memory",
        db_path=tmp / "none.db",
        **kwargs,
    )

from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)


class StubBackend(UnifiedBackend):
    """One completed-text candidate per turn (the unified path)."""

    def generate(self, request):
        return GenerationResponse(
            text=FERRY_TEXT, tokens_generated=len(FERRY_TEXT.split())
        )

    def health_check(self):
        return True

    def backend_name(self):
        return "example-stub"


def run():
    core = make_core(backend=StubBackend())
    final = core.think("What time does the ferry leave?")
    state = core.last_turn_state
    return {
        "mode": final.mode,
        "candidates": len(state.candidates),
        "salvage_path": final.salvage_path,
        "text": final.text,
    }
```

### Example: streaming_turn

```python
import tempfile
from pathlib import Path

from thoughtforge.cognition.core import ThoughtForgeCore

FERRY_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)


def make_core(**kwargs):
    """A persona-free core on a temp memory dir and no DB file."""
    tmp = Path(tempfile.mkdtemp())
    return ThoughtForgeCore(
        config={"cognition": {}},
        memory_dir=tmp / "memory",
        db_path=tmp / "none.db",
        **kwargs,
    )

from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)


class StubBackend(UnifiedBackend):
    """One completed-text candidate per turn (the unified path)."""

    def generate(self, request):
        return GenerationResponse(
            text=FERRY_TEXT, tokens_generated=len(FERRY_TEXT.split())
        )

    def health_check(self):
        return True

    def backend_name(self):
        return "example-stub"


def run():
    core = make_core(backend=StubBackend())
    events = list(core.think_stream("What time does the ferry leave?"))
    return {
        "kinds": [event.kind for event in events],
        "final_mode": events[-1].final.mode,
    }
```

### Example: trace_export

```python
import tempfile
from pathlib import Path

from thoughtforge.cognition.core import ThoughtForgeCore

FERRY_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)


def make_core(**kwargs):
    """A persona-free core on a temp memory dir and no DB file."""
    tmp = Path(tempfile.mkdtemp())
    return ThoughtForgeCore(
        config={"cognition": {}},
        memory_dir=tmp / "memory",
        db_path=tmp / "none.db",
        **kwargs,
    )

from thoughtforge.cognition.traces import trace_problems


def run():
    core = make_core()
    core.think("What time does the ferry leave?")
    trace = core.last_turn_trace()
    return {
        "format": trace["format"],
        "sections": list(trace),
        "problems": trace_problems(trace),
        "final_mode": trace["final"]["mode"],
    }
```

### Example: seeded_replay

```python
import tempfile
from pathlib import Path

from thoughtforge.cognition.core import ThoughtForgeCore

FERRY_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)


def make_core(**kwargs):
    """A persona-free core on a temp memory dir and no DB file."""
    tmp = Path(tempfile.mkdtemp())
    return ThoughtForgeCore(
        config={"cognition": {}},
        memory_dir=tmp / "memory",
        db_path=tmp / "none.db",
        **kwargs,
    )


def run():
    config = {"cognition": {}, "determinism": {"seed": 42}}
    turn_ids = []
    for _ in range(2):
        tmp = Path(tempfile.mkdtemp())
        core = ThoughtForgeCore(
            config=config,
            memory_dir=tmp / "memory",
            db_path=tmp / "none.db",
        )
        core.think("What time does the ferry leave?")
        turn_ids.append(core.last_turn_state.turn_id)
    return {"same_turn_id": turn_ids[0] == turn_ids[1],
            "turn_id": turn_ids[0]}
```

## 6. Measured (this VM, through the real core)

- **knowledge_only_turn** — `mode "knowledge_only"`, the exact
  honest refusal "The forge has no knowledge relevant to this
  query.", 9 tokens (TF-019's §6′ measurement, still exact).
- **stub_backend_turn** — `mode "full"`, exactly 1 candidate,
  `salvage_path "best_draft"`, the final text is the stub's
  ferry text verbatim.
- **streaming_turn** — event kinds `[turn_started,
  candidate_start, chunk ×4, candidate_end, final]`: exactly
  one terminal `final`, its mode `"full"` (TF-026's contract).
- **trace_export** — format `thoughtforge-turn-trace/1`, every
  TRACE_SECTIONS section present, `trace_problems` `[]`, the
  final's mode `knowledge_only` (TF-038's format).
- **seeded_replay** — two seed-42 cores open the *same* first
  turn id, `turn_3881d6d8` — byte-identical to TF-036's own
  recorded measurement, cross-pinned in this slice's tests.

Proof grade: FIXTURE generation sources on this VM, as with
TF-019–TF-038. No live model appears in any example; a live
model's turn differs only in who wrote the candidates.

## 7. What this slice did not do

- **F2 is not fixed here.** The retrieval gate's wrong-path
  check is Part III's first slice (TF-041, knowledge store
  mapped) — this slice records it open with its owner, in the
  findings data, in CORE_LOOP §8's status table, and here.
  **TF-041 addendum (2026-10-31):** TF-041 has since landed
  the fix — `_retrieve` gates on the configured
  `self._knowledge.db_path` — and flipped F2's record in
  `RECONCILED_FINDINGS` to **CLOSED** with its evidence, as
  §2's checker instructed; the table in §2 stands as the
  TF-039-time record.
- **The freeze is not bumped.** §4 records the additions;
  folding them into the frozen tables as v1.1.0 is left as a
  named, verifiable act (TF-192), not done silently inside a
  docs slice.
- **TF-182's developer reconciliation still stands.** Part II's
  docs are now true to shipped behaviour; the second,
  developer-facing reconciliation at TF-182 remains on the road
  as roaded.
