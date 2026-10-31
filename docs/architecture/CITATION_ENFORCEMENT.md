# TF-049 — Cite-or-Explain II: the Enforcement Side

*Part III, slice TF-049. Code:
`src/thoughtforge/refinement/citation_enforcement.py` (the
matching rules, the policy, `check_citations()`, and
`citation_enforcement_problems()` re-deriving every claim
below from the live tree), the `records` leg of
`EnforcementGate.check` in
`src/thoughtforge/refinement/enforcement.py`, and the core
wiring in `src/thoughtforge/cognition/core.py`. Tests:
`tests/test_tf049_citation_enforcement.py`. Builds on
TF-048's citation payloads; TF-093 owns unifying this
check with the rest of the enforcement module into one
policy surface.*

## 1. The question and the measured starting state

The roadmap asks: after generation, are claims that match
stored facts actually checked for citations — and is the
answer flagged or regenerated per policy when they are
not? Measured before this slice, the honest answer was
**only in the weakest possible form**:

- The one citation check was `EnforcementGate`'s
  answer-level QID rule: if any Wikidata QIDs were
  retrieved, **at least one** had to appear **somewhere**
  in the answer. It was claim-blind (a second stored fact
  could be stated uncited and the answer passed on an
  unrelated citation), kind-blind (reference chunks,
  GeoNames places, ConceptNet edges — all carrying TF-048
  markers — had no check at all; a bundle with no QIDs
  skipped citation enforcement entirely), and
  identity-blind (citing a QID that was never retrieved —
  a fabricated citation — was ignored).
- The core carried a second copy of the same minimal
  logic (`_check_citation_gate`, repair path), and
  `_extract_citations` could only extract QIDs.

## 2. The per-claim check

`check_citations(text, records)` splits the answer into
claims (sentences, via the tree's own `split_sentences`)
and checks each claim against each retrieved record.

**Matching** — a claim matches a record under exactly the
rules in `MATCH_RULES`, over distinctive terms (lowercase
tokens of length ≥ 3, a stated singular fold, stopwords
removed), against the record's claim text (label +
description/content + cue):

| rule | fires when |
|---|---|
| `label_phrase` | the record's full label phrase appears in the claim |
| `label_plus_term` | a distinctive label term appears in the claim and ≥ 1 further distinctive term is shared |
| `term_overlap` | ≥ `MIN_SHARED_TERMS` (3) distinctive terms are shared |

**Citing** — a matched claim is cited only when that
record's own TF-048 marker appears **inside that claim**
(a bare QID, or the bracketed `[marker]` the prompt
taught the model). A citation in a different sentence
does not cover the claim: that is the difference between
cite-or-explain and cite-somewhere.

**Foreign markers** — a bracketed/QID citation in the
answer that no retrieved record owns is a fabricated
citation and fails the check by name. (The gate's own
`[Forge: …]` notes are not citations and are excluded.)

The verdict is data: `CitationCheck` carries every
claim, its matches (with the rule and shared terms that
fired), its cited markers, its uncited record ids, the
foreign markers, and the policy action taken.
`passed` is true exactly when no matched claim is
uncited and no foreign marker appears.

## 3. The policy

`CITATION_POLICY`, as data:

- **On an uncited claim: `flag`** (the shipped default).
  The EnforcementGate fails the answer's citation check
  and appends its forge note; the notes name the uncited
  markers. Through TF-047, a flagged turn's writeback
  gate closes — an uncited answer teaches the store
  nothing.
- **On a foreign marker: `flag`**, same path.
- **On no matching claims: pass** — there is nothing to
  cite. The legacy any-QID floor is a separate, older
  rule; this slice does **not** relax it (see §5).
- **Regeneration is a named option, not the default and
  not wired.** The estate's only regeneration path (the
  quality-gate repair pass) builds its prompt with no
  knowledge context (TF-023's stated narrowing), so a
  regenerated answer cannot cite what it was never
  shown. Wiring regeneration under the unified policy
  is TF-093's work; the policy table is shaped for it,
  and `CitationCheck.regeneration_required` is always
  False under the shipped policy — pinned, not implied.

## 4. Wiring

- `EnforcementGate.check(..., records=...)`: with the
  bundle's records, the per-claim verdict is ANDed into
  the citation check and rides the result as
  `claim_check`. Without records, the gate's behaviour
  is byte-identical to before.
- The core passes `records=bundle.activated_records` in
  `_compose_final` and retains the check on
  `core.last_citation_check` (a post-freeze minor
  addition, recorded in `POST_FREEZE_ADDITIONS` and in
  the freeze doc's §7 table, exactly as TF-047 recorded
  its surface). It is None until a turn with candidates
  composes: the knowledge-only path presents retrieved
  records as retrieved knowledge and cites record ids by
  construction (TF-048) — there is no generated claim to
  police, and this slice does not pretend to police it.
- `_check_citation_gate` (repair path) now delegates to
  the same checker plus the same any-QID floor — the
  duplicate logic is one logic.
- `_extract_citations` extracts every kind's marker.
  For entities the marker is the QID, so its pinned
  behaviour there is unchanged.

## 5. Stated boundaries

- **The legacy any-QID floor stands**, in both the gate
  and the repair path: an answer over a QID-bearing
  bundle that cites no QID at all still fails, even when
  no claim matched. Reconciling the floor with the
  per-claim rule is TF-093's unification, recorded in
  the policy — not taken silently here.
- **Matching is a heuristic**, stated as such:
  deterministic term overlap, not semantic entailment.
  A paraphrase below the thresholds is not caught; a
  coincidental overlap can be flagged. Flagging is the
  safe direction (the user sees the note); TF-095 owns
  refusal wording, TF-123 will score this properly.
- Salvage's internal citation *scoring* (its 0.55 draft
  weight) still counts QIDs only — it ranks drafts, it
  does not enforce; TF-093's unification owns whether
  it moves to markers.

## 6. Proof

Fixture grade: synthetic records of every citation kind
through the checker and the gate, the TF-043 corpus with
one real-shaped-QID entity added (the corpus's synthetic
QF-ids do not match the QID shape the legacy floor
regexes — TF-048's fixture note) through real core turns
with a stub backend (a cited claim passes end to end; an
uncited one is flagged, named, and retained on
`core.last_citation_check`), and the legacy gate/gate-
helper pins re-proven unchanged.
`citation_enforcement_problems() == []` re-derives the
whole leg from the live tree. A live model's citation
behaviour is NOT PROVEN — no live model ran.
