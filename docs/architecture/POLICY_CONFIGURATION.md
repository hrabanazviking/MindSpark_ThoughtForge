# TF-094 — Policy Configuration: Strictness Per Conversation

*Enforcement now carries a strictness level per
conversation — `strict` / `standard` / `relaxed` —
with the default stated plainly: **standard**,
which is TF-093's shipped behaviour exactly. The
estate lives in
`src/thoughtforge/refinement/policy_config.py`
(`CONFIG_RULES` / `DISPOSITIONS` /
`CONFIG_FIXTURES` as data,
`resolve_policy_config` / `enforce_configured`,
the checkers `policy_config_problems` /
`policy_config_live_problems`, format
`thoughtforge-policy-configuration/1`), a thin
disposition layer over TF-093's real surface —
no gate is re-implemented and none is retuned.*

## The plain truth first

Until this slice, TF-093's surface had exactly
one disposition. `enforce_policy` composes the
seven checks, combines them
refused > flagged > rescued > clean, and that
was the whole policy: a flagged answer always
left (annotated where the gate annotates), a
refusal always refused. Nothing in the
refinement estate named a strictness level, a
default, or a way for one conversation to be
held tighter or looser than another.

This slice adds the level — and only the level.
Strictness governs **disposition** (what a
verdict *does*), never **detection** (what a
check *finds*). Length's five-word floor, the
generic-phrase list, and TF-090/091/092's repair
rules live inside their gates; a level that
silently retuned them would rewrite filed gates
mid-Part, so none does.

## The levels

| Level | Clean | Rescued | Flagged | Refused |
|---|---|---|---|---|
| strict | clean | rescued | **refused** — blocked | refused |
| standard *(default)* | clean | rescued | flagged | refused |
| relaxed | clean | rescued | **warnings** — leaves as clean/rescued | refused |

- **Standard is TF-093 exactly.** Same verdict,
  same text, same flags, same problems, same
  notes, same check outcomes — proven leg-by-leg
  against `enforce_policy` itself in the live
  checker, for every filed fixture.
- **Strict blocks every flag.** A base
  `flagged` verdict becomes `refused`: no usable
  text leaves, the flags are preserved verbatim
  and carried as the refusal's named problems,
  and every check outcome stands exactly as
  judged — a blocked flag is still *recorded*
  flagged, never rewritten to refused at check
  level. TF-095 owns telling the user that and
  why an answer was blocked.
- **Relaxed downgrades flags to warnings, never
  hides them.** A base `flagged` verdict takes
  the verdict the run earned underneath the
  flags (`rescued` if any check rescued, else
  `clean`), the text leaves exactly as TF-093
  returned it — gate annotations included — and
  the flags move verbatim to the result's
  `warnings`. The check outcomes still say
  flagged, by name. Relaxed is a disposition,
  not an eraser.
- **A refusal is never relaxed.** Empty output,
  an unrepairable structured answer, a
  truncation with no complete sentence, an
  unsupported expected language: there is no
  honest text to pass, at any level, and no
  level invents one. Refused is refused under
  all three levels, with the same problems.
- **A rescue is allowed at every level.** A
  base `rescued` or `clean` verdict is identical
  under strict, standard and relaxed —
  strictness governs flagged and refused
  dispositions, not whether an honest repair
  may stand.

## The default, and how a level is chosen

**The default is `standard`.** A caller who
names no level gets the shipped TF-093
behaviour — stated here and in
`DEFAULT_STRICTNESS`, never inferred and never
environment-dependent.

Strictness is **per conversation and
caller-supplied, never guessed** — TF-093's
signal rule carried to configuration. One level
applies to a whole conversation's run through
the surface; the layer never infers a level
from the draft, the ask, the user, or a
hardware profile. Level names resolve
case-insensitively (surrounding whitespace
ignored) to their canonical form. An **unknown
name raises `ValueError`** naming the value
and the three real levels (the TF-015 precedent
for a bad configuration value): a caller bug is
never silently mapped to the default. Passing
both `strictness=` and a `config=` that
disagree raises likewise.

**Filed limit:** there are no per-check
overrides. A caller cannot relax citation while
tightening length in this slice — a
conversation's policy must always be nameable
in one word. TF-098 owns cost caps per profile;
TF-095 owns the user-facing telling of a block.

## D-DEAD-1: TF-094's share, ruled

The docket assigned `utils/validators.py`'s
dead functions a wire-or-retire verdict per
function, with TF-094 named for the policy
configuration share. **Ruling:
`validate_config` is NOT wired into this
configuration surface.** It validates a backend
dict — it requires the `backend` key and judges
backend-specific fields — the wrong shape for a
strictness level. This surface validates its
own selections itself, exactly as TF-032 ruled
for the cognition config surface. The
validators module is **unchanged** (a test here
pins that it still judges a backend dict as
before), the remaining functions' verdicts
stand with their filed owners (TF-065 and the
ingestion line), and **D-DEAD-1 stays OPEN**.
TF-016 totals are unchanged: 88 entries —
45 OPEN / 1 OPEN-ENVIRONMENT / 39 FIXED /
3 CLOSED-BY-VERDICT.

## The fixtures

12 filed cases — TF-093's synthetic cases
reused verbatim, never canon — every one run
live through the real TF-093 surface at **all
three levels**: a clean prose answer and a loop
rescue (identical everywhere); five flagged
cases (whole-answer drift, an uncited stored-fact
claim, the legacy floor without records, a
too-short answer, a hard generic phrase), each
blocked under strict, flagged under standard,
and downgraded to a verbatim warning under
relaxed; a structured repair (rescued
everywhere); and four refusals (unrepairable
structured output, a truncation with no
complete sentence, empty output, an unsupported
expected language) refused identically at every
level.

## Filed limits, not hidden

- **The core does not yet compose through the
  configured layer**, as it does not yet compose
  through TF-093's surface (`_compose_final`
  still calls `EnforcementGate` directly, its
  TF-049 wiring pinned). Routing the core
  through either surface is a core slice's
  call, not silently taken here — the API
  freeze (TF-034) is respected.
- **A strict block is silent here by design:**
  this slice returns the blocked verdict with
  its named problems; turning that into the
  user-facing refusal message is TF-095's
  work, and this slice does not pre-write it.
- **Detection limits stand exactly as filed**
  by TF-090–TF-093: the levels inherit every
  underlying gate's limits unchanged.

## Proof grade

FIXTURE — every filed case recomputed live
through the real TF-093 surface at all three
levels on this VM, standard proven identical
to the surface itself, strict and relaxed
proven against the filed disposition table.
A live model's behaviour under any level is
**NOT PROVEN** (no model in tree); TF-100's
chaos corpus owns real-shaped failures and
TF-101 runs the full policy stack over them.
