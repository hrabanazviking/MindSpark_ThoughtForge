# TF-090 — Format Enforcement: Repaired or Refused, Honestly

*The gate for structured answers: when the ask was
JSON-shaped, the output is made usable by repairs
that invent nothing — or it is refused, with the
reason named. The estate lives in
`src/thoughtforge/refinement/format_enforcement.py`
(`FORMAT_RULES` / `FORMAT_FIXTURES` /
`TRUNCATION_FIXTURES` as data, `enforce_format` /
`repair_truncation`, the checkers
`format_enforcement_problems` /
`format_enforcement_live_problems`, format
`thoughtforge-format-enforcement/1`).*

## The plain truth first

Until this slice, nothing in the refinement
package could judge a structured answer at all.
TF-087 probed the broken JSON draft
`{"ferry": "dawn", "tide": ` coming back
byte-identical from `FragmentSalvage`, and no
other surface even looked at an answer's shape.

That does not change inside salvage — deliberately.
Salvage is handed candidate drafts with no signal
that the ask was structured, and a component with
no signal must not guess. The enforcement is a
**caller-invoked gate**, the same shape TF-049's
citation enforcement took beside `EnforcementGate`:
the caller who asked for JSON calls
`enforce_format`; the caller with a prose draft
calls `repair_truncation`. `FragmentSalvage` is
untouched, so TF-087's catalogue and TF-089's
filed metrics (rescue 2/7, unrescued verbatim 5)
stand exactly as banked. TF-093 owns unifying
this gate with the enforcement surface; TF-095
owns telling the user a refusal happened.

## The line: repair removes, never invents

TF-088 drew the line for refine passes — rescue
never invents content. This gate holds the same
line for structure. A repair may only remove or
re-delimit what is already there:

| Repair | What it does | Example |
|---|---|---|
| `extracted_from_fence` | unwraps a ```` ```json ```` fence | the span inside parses as-is |
| `extracted_from_prose` | takes the one outermost JSON span out of surrounding prose | `Here is…: {"ferry": "dawn"} — hope…` → the object |
| `dropped_trailing_comma` | removes commas directly before `}` / `]` | `{"a": 1,}` → `{"a": 1}` |
| `converted_python_literal` | re-encodes a Python literal (single quotes, `True`/`None`) as JSON, content-exact | `{'bell': True}` → `{"bell": true}` |
| `dropped_incomplete_tail` | drops the trailing element a truncation cut — a cut key, a key with no value, a cut string or cut literal; the dropped text is recorded verbatim | `{"ferry": "dawn", "tide": ` → ferry only |
| `closed_open_containers` | appends the closers a truncation left off | `{"ferry": "dawn"` → `{"ferry": "dawn"}` |

The truncation repair is the heart of the slice,
and its rule is strict: the scanner walks the
text once, tracking containers and expectations,
and remembers every point where a **complete
value** ended. A truncated output is cut at the
*latest* such point and its open containers are
closed. Every complete key/value survives; the
incomplete tail — and only the incomplete tail —
is dropped, recorded verbatim in the result's
`dropped_tail`. A cut value is never completed:
`"tide": "ris` does not become `"low"`, it
disappears, and the record says so.

## Refusal is a verdict, not a shrug

When no honest repair exists, the verdict is
`refused`, `text` is `None`, and every problem is
named:

- `no_json_span` — prose arrived when JSON was asked;
- `multiple_json_spans` — two objects; picking one would be a guess;
- `unparseable_json` — balanced but invalid (a bare-word value like `{"ferry": dawn}` is refused, never "helpfully" quoted into a string);
- `missing_required_key: <key>` — the schema is a gate, not a repair target: the gate never invents the missing field. This holds *after* repair too: `{"tide": "ris` honestly repairs to `{}`, and is then refused for the missing `ferry` rather than shipped;
- `wrong_top_level_shape` — an array when an object was asked;
- `empty_output`.

A repaired verdict carries the ordered names of
its repairs, so a consumer can see exactly what
the gate changed (TF-097 owns carrying that
record into the turn trace).

## Prose truncation: cut, or refuse

`repair_truncation` applies the same honesty to a
plain draft's ending:

- ends on a terminal mark (`. ! ? …`, allowing a
  closing quote or bracket) → `intact`, untouched;
- ends mid-sentence with an earlier complete
  sentence → `repaired`: cut to the last complete
  sentence, dropped tail recorded verbatim;
- no complete sentence at all → `refused`
  (`no_complete_sentence`). TF-087's truncated
  fixture — *"The ferry leaves at dawn and the
  tide bell rings across the"* — lands here: its
  one sentence never completes, so the honest
  verdict is refusal. Ending it would mean
  inventing its ending.

## The fixtures

25 filed cases, all synthetic Lantern Hollow
material, every one run live through the real
gate by the slice's checker: 20 for
`enforce_format` (1 valid, 12 repaired, 7
refused — including TF-087's own broken-JSON
draft, which now repairs to exactly its one
complete pair) and 5 for `repair_truncation`
(2 intact, 1 cut, 2 refused).

## Docket

- **D-SALV-1's truncation + format shares are
  FIXED at gate grade by TF-090**: format rescue
  and structured-output truncation repair exist
  in `refinement/format_enforcement.py`, and
  prose truncation is cut-or-refused by the same
  honesty rule. The entry stays **OPEN** — its
  remaining shares keep their owners: rambling
  compression (measured by TF-089, still no
  mechanism), repetition/loop cut (TF-091),
  language drift (TF-092). TF-016 totals
  unchanged: 88 entries — 45 OPEN /
  1 OPEN-ENVIRONMENT / 39 FIXED /
  3 CLOSED-BY-VERDICT.
- Filed honestly as limits, not defects: a
  repair that drops *everything* after the last
  complete value can yield an empty structure
  (`{"a": tru` → `{}`) — the dropped tail is
  recorded, and the schema gate is the caller's
  guard against shipping it; TF-101 scores this
  behaviour in the full stack. A second JSON
  span anywhere in the output refuses the whole
  answer as ambiguous, even when the first span
  is clean.

## Proof grade

FIXTURE — every filed case recomputed live
through the real gate functions on this VM, and
the gate's repairs verified content-preserving
by re-parsing to the filed structures. A live
model's structured outputs are **NOT PROVEN**
(no model in tree); TF-100's chaos corpus owns
real-shaped failures and TF-101 runs the full
policy stack — this gate included — over them.
