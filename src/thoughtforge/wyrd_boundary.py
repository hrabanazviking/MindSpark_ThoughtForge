"""
TF-145 — the WYRD boundary, filed.

TF-145's roadmap text: *WYRD boundary doc:
ThoughtForge vs the WYRD Protocol — what each owns
(cognition vs world-model), the interface between
them, written for the Second Life/Hermes
consumers.* The roadmap's own framing (its laws
section) files the relationship in one line: *WYRD
is the world-model neighbour, not a rival (Part
VIII's boundary).* This module files that boundary
as verifiable data — ownership rows, interface
rows and consumer rows, each grounded in evidence
that stands in THIS repo — and renders it verbatim
as ``docs/architecture/WYRD_BOUNDARY.md``.

The measured start, probed live on this tree
before anything was built:

  * **No WYRD interface code exists in the
    shipping tree.** A grep for WYRD across
    ``src/thoughtforge/`` finds the word only in
    filings ABOUT the boundary: TF-139's verdict
    module (TF-146 owns the WYRD-consumable bond
    bridge), TF-143's production bond graph (the
    same TF-146 line in its render) — and nothing
    else. There is no WYRD client, no WYRD
    protocol implementation, no live feed, and no
    file under ``src/`` named for WYRD. WYRD
    itself lives outside this repo. This doc
    therefore files the boundary and the interface
    SHAPES the estate's evidence supports; it does
    not claim, anywhere, that a live WYRD
    integration exists — live behaviour is NOT
    PROVEN (TF-173 class), never claimed.
  * **The evidence the boundary rests on stands
    clean live.** TF-132's WYRD-shaped panel
    (``benchmarks/wyrd_panel.py``) measures
    world-state consistency against a filed
    SYNTHETIC world — 18 cases / 9 consistent /
    8 contradicted / 1 honest-unanswerable,
    verdict PASS, ``wyrd_panel_problems()`` [] —
    and files in its own rules that it *feeds
    TF-145, it does not decide it*. TF-139's bond
    verdict (ADOPT_AS_LAYER, checker clean) files
    the bond layer's evidence link: bond evidence
    names shipping-store record ids. TF-143's
    production bond graph (checker clean) lands
    that layer: ``BondGraph.excerpt()`` files 9
    lines on its parity fixtures, and its filing
    names TF-146 as the owner of the
    WYRD-consumable bridge — the bridge is NOT
    built in this tree.
  * **The neighbouring checkers stood clean
    live**: ``wyrd_panel_problems()`` [],
    ``bond_graph_verdict_problems()`` [],
    ``bond_graph_problems()`` [],
    ``wing_retirements_problems()`` [] (TF-144,
    the wing's closing state).

**What the boundary files** (literal data below):

  * ``OWNERSHIP_ROWS`` — twelve rows: six owned by
    ThoughtForge (the cognition turn, the
    knowledge store, the bond layer, persona
    compilation, runtime truth enforcement, and
    the consistency MEASUREMENT instrument), three
    owned by WYRD (the canonical lived world
    state, its change log, and entity identity in
    the world), and three owned by NEITHER
    (embodiment / transport on the consumer side,
    Volmarr's canon data, and proof of live
    behaviour — unclaimed and NOT PROVEN). Every
    row files its evidence in this repo; a WYRD
    row's evidence is the roadmap's filing plus
    the shape TF-132's panel stands in for, never
    code — there is no WYRD code here to cite.
  * ``INTERFACE_ROWS`` — five interface shapes
    between the two sides, each with its honest
    status: the world-state replay shape and the
    consistency-judgement shape exist only as
    TF-132's panel fixtures over a synthetic
    world; the bond evidence link and the
    cognition turn exist in production (TF-143 /
    the Part II core); the bond excerpt bridge to
    WYRD-consumable data is NOT built — TF-146
    owns it, exactly as TF-139/TF-143 filed.
  * ``CONSUMER_ROWS`` — the two consumers the
    roadmap names for this doc: Second Life /
    Heimdall and Hermes. Neither is integrated in
    this tree: no Heimdall or Second Life code
    stands under ``src/`` (the word Heimdall
    appears once in the estate, inside a
    persona-suite fixture sentence), and Hermes
    appears only in portability / binding filings
    (Part IX owns the Hermes binding, TF-158).

**Docket**: TF-145 has NO share of TF-016's
D-WING-1 — the entry assigns verdicts,
adoptions, retirements, research hygiene
(TF-148) and wing tests (TF-149); a boundary doc
over the estate those slices built is none of
those shares. Per the TF-144 pattern's rule (a
slice touches the docket only for its own filed
share), this slice does NOT edit the docket and
adds no entry: TF-016 totals stand unchanged
(137 entries — 33 OPEN / 1 OPEN-ENVIRONMENT /
100 FIXED / 3 CLOSED-BY-VERDICT). That is filed
in the open here and in the render, not silently.

The human-readable boundary doc is
``docs/architecture/WYRD_BOUNDARY.md``, filed
verbatim as this module's render (regenerate
with ``python3 -m thoughtforge.wyrd_boundary``).
Proof grade: FIXTURE/STATIC — filings + the live
tree re-measured on this VM; no live model, no
live WYRD, never claimed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

WYRD_BOUNDARY_FORMAT = "thoughtforge-wyrd-boundary/1"
WYRD_BOUNDARY_DOC = "docs/architecture/WYRD_BOUNDARY.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "thoughtforge"
_ROADMAP = _REPO_ROOT / "docs" / "MINDSPARK_THOUGHTFORGE_ROADMAP_2026-10-30.md"

#: The owners the boundary may file.
OWNERS: tuple[str, ...] = ("thoughtforge", "wyrd", "neither")

#: The interface statuses the boundary may file.
#: There is deliberately no "live" status in this
#: vocabulary: no live WYRD integration exists in
#: this tree, and the validator names any row that
#: claims a status outside this set.
INTERFACE_STATUSES: tuple[str, ...] = (
    "fixture-stand-in-only",
    "exists-as-panel-fixture",
    "exists-in-production",
    "not-built-tf146",
)

#: Tokens that would name a live WYRD client in
#: the shipping tree. None exists (measured live);
#: the checker pins that.
_LIVE_CLIENT_TOKENS = ("WyrdClient", "WYRDClient", "wyrd_client")

#: Shipping modules whose WYRD mentions are
#: filings ABOUT this boundary (the verdict, the
#: production bond graph's TF-146 line, and this
#: module itself) — not interface code.
_BOUNDARY_FILING_FILES = {
    "bond_graph_verdict.py", "bond_graph.py", "wyrd_boundary.py",
}


# ── Ownership, as literal data ───────────────────────────────────────────────
#
# Every row files: id, subject, owner, what the
# ownership means, and the evidence IN THIS REPO
# the row stands on. ThoughtForge rows cite code
# or filed docs; WYRD rows cite the roadmap's
# filing and the panel shape that stands in for
# the lived world — never code, because no WYRD
# code stands in this repo.

OWNERSHIP_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "cognition_turn",
     "subject": "the cognition turn — persona, memory "
                "activation, scaffolding, candidate "
                "scoring and the citation gate that turn "
                "a question into an answer",
     "owner": "thoughtforge",
     "meaning": "ThoughtForgeCore.think() / "
                "think_stream() (src/thoughtforge/"
                "cognition/core.py) is the estate's "
                "cognition surface; Part II files its "
                "loop (docs/architecture/CORE_LOOP.md)",
     "evidence": "src/thoughtforge/cognition/core.py "
                 "(ThoughtForgeCore) + docs/architecture/"
                 "CORE_LOOP.md (Part II)"},
    {"id": "knowledge_store",
     "subject": "the knowledge store — what is known: "
                "the filed record types and their "
                "retrieval",
     "owner": "thoughtforge",
     "meaning": "MemoryStore / KnowledgeForge "
                "(Part III) answer 'what is known'; "
                "TF-140's verdict leaves KnowledgeForge "
                "the estate's sole retriever (TF-144)",
     "evidence": "src/thoughtforge/knowledge/store.py "
                 "+ models.py + forge.py (Part III) + "
                 "TF-139's comparison rows + TF-144's "
                 "successor filing"},
    {"id": "bond_layer",
     "subject": "the bond layer — how entities stand "
                "with each other: edges, vows, hurts, "
                "repair state and the relational excerpt",
     "owner": "thoughtforge",
     "meaning": "TF-139 adopted wyrdforge's bond "
                "graph AS A LAYER beside the store and "
                "TF-143 landed it as production "
                "BondGraph; the layer references store "
                "record ids as evidence and duplicates "
                "no facts",
     "evidence": "src/thoughtforge/knowledge/"
                 "bond_graph.py (TF-143) + bond_graph_"
                 "verdict.py (TF-139 ADOPT_AS_LAYER)"},
    {"id": "persona_compilation",
     "subject": "persona compilation — canonical "
                "records compiled into the expectation "
                "a turn is steered by",
     "owner": "thoughtforge",
     "meaning": "TF-141's MERGE landed in TF-143 "
                "beside TF-030: CompiledPersona"
                "Expectation extends PersonaExpectation, "
                "approval checked first, defects not "
                "imported",
     "evidence": "src/thoughtforge/cognition/"
                 "persona_compiler.py (TF-143) + "
                 "persona_compiler_verdict.py (TF-141)"},
    {"id": "truth_enforcement",
     "subject": "runtime truth enforcement — the "
                "per-claim law an answer is judged "
                "under as it is produced",
     "owner": "thoughtforge",
     "meaning": "Part V's shipping enforcement is "
                "the estate's sole runtime truth law; "
                "TF-142 retired the wing's calibrator "
                "against it and TF-144 executed that "
                "retirement",
     "evidence": "TF-142's verdict filing + TF-144's "
                 "successor filing in src/thoughtforge/"
                 "wing_retirements.py"},
    {"id": "consistency_measurement",
     "subject": "the consistency MEASUREMENT — the "
                "instrument that scores answers against "
                "a stated world state",
     "owner": "thoughtforge",
     "meaning": "TF-132's WYRD-shaped panel is a "
                "ThoughtForge-side measuring instrument "
                "over a filed synthetic world; measuring "
                "consistency is cognition-side work — "
                "the panel is not a world model and "
                "files that it feeds this boundary, it "
                "does not decide it",
     "evidence": "benchmarks/wyrd_panel.py (TF-132: "
                 "18 cases, verdict PASS, its rule "
                 "the_panel_feeds_tf145_it_does_not_"
                 "decide_it)"},
    {"id": "canonical_world_state",
     "subject": "the canonical lived world state — "
                "the entities, attributes, locations "
                "and relations of the world as it "
                "actually stands",
     "owner": "wyrd",
     "meaning": "the world-model side owns what the "
                "world IS; ThoughtForge owns only what "
                "it knows and infers — a cognition "
                "answer that contradicts the canonical "
                "state is wrong however fluent it is "
                "(TF-132's contradicted cases are "
                "exactly that shape, in miniature)",
     "evidence": "the roadmap's law (docs/MINDSPARK_"
                 "THOUGHTFORGE_ROADMAP_2026-10-30.md: "
                 "WYRD is the world-model neighbour) + "
                 "TF-132's filed world standing in for "
                 "it as a synthetic fixture"},
    {"id": "world_change_log",
     "subject": "the world's change log — the "
                "timeline of changes by which the "
                "canonical state moves",
     "owner": "wyrd",
     "meaning": "state is a replay of changes, not a "
                "snapshot a cognition layer may freeze: "
                "TF-132's stale-answer cases (the wheel "
                "still turning after the belt breaks) "
                "are failures to honour the change log",
     "evidence": "TF-132's WORLD_EVENTS + world_state_"
                 "at() replay shape in benchmarks/"
                 "wyrd_panel.py, standing in for the "
                 "lived log this repo does not hold"},
    {"id": "entity_identity_in_world",
     "subject": "entity identity in the world — "
                "which entity in the lived world a "
                "name or id denotes",
     "owner": "wyrd",
     "meaning": "ThoughtForge-side ids (store record "
                "ids, bond ids, panel entity ids) are "
                "references; the world's own identity "
                "for an entity is the world side's to "
                "file — the bond layer's evidence link "
                "already works this way, referencing "
                "ids it does not own",
     "evidence": "TF-139/TF-143's evidence-link shape "
                 "(BondEvidence names record ids) + "
                 "TF-132's panel-local entity ids, "
                 "filed as fixture ids only"},
    {"id": "embodiment_and_transport",
     "subject": "embodiment and transport — logging "
                "in, moving, perceiving and messaging "
                "inside a lived world",
     "owner": "neither",
     "meaning": "neither the cognition layer nor the "
                "world-model protocol is a body: the "
                "consumer side (the Second Life / "
                "Heimdall body) owns embodiment, as "
                "Part IX files bindings as cognition "
                "only (TF-157's Samtal shape)",
     "evidence": "the roadmap's Part IX filing "
                 "(bindings are cognition only) + no "
                 "embodiment code anywhere under src/"},
    {"id": "canon_data",
     "subject": "Volmarr's canon data — the datasets "
                "and persona material under data/",
     "owner": "neither",
     "meaning": "canon is Volmarr's, private, never "
                "published and never either system's "
                "property; both sides consume only "
                "what he files for them, and every "
                "fixture in this estate is synthetic "
                "(TF-017)",
     "evidence": "the roadmap's canon law + TF-017's "
                 "synthetic-fixtures rule, filed across "
                 "the estate"},
    {"id": "live_behaviour_proof",
     "subject": "proof of live behaviour — whether a "
                "live system actually stays true to "
                "the world in use",
     "owner": "neither",
     "meaning": "unclaimed because unproven: every "
                "instrument in this repo is fixture "
                "grade, so neither side may file live "
                "truthfulness as owned or done — it is "
                "NOT PROVEN (TF-173 class) until a live "
                "run proves it",
     "evidence": "TF-132's filed limits (no live "
                 "model was read; live world-state "
                 "behaviour NOT PROVEN, TF-173)"},
)


# ── The interface, as filed shapes ───────────────────────────────────────────

INTERFACE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "world_state_replay",
     "name": "world state, replayed to a point",
     "direction": "wyrd -> thoughtforge",
     "shape": "a filed world (entities with "
              "attributes / locations / relations) "
              "+ an ordered change log; the state at "
              "a point is the dawn state plus the "
              "first N changes, recomputed purely — "
              "never a copied snapshot",
     "status": "fixture-stand-in-only",
     "evidence": "benchmarks/wyrd_panel.py: WORLD_"
                 "ENTITIES (8) + WORLD_EVENTS (6) + "
                 "world_state_at() over the synthetic "
                 "Emberwick; no live WYRD feed or "
                 "client stands in this tree"},
    {"id": "consistency_judgement",
     "name": "an answer judged against the world",
     "direction": "thoughtforge measurement over "
                  "wyrd-shaped state",
     "shape": "the answer's structured claim (bound "
              "to its text by its claim phrase) vs "
              "the replayed world value -> consistent "
              "/ contradicted / honest_unanswerable, "
              "with inventing an unfiled value scored "
              "contradicted",
     "status": "exists-as-panel-fixture",
     "evidence": "benchmarks/wyrd_panel.py: score_"
                 "case() over 18 filed cases (9 "
                 "consistent / 8 contradicted / 1 "
                 "honest-unanswerable, PASS) — scripted "
                 "answers only, no live model"},
    {"id": "bond_evidence_link",
     "name": "bond evidence naming store records",
     "direction": "thoughtforge layer -> thoughtforge "
                  "store (the layer link the bridge "
                  "builds on)",
     "shape": "BondEvidence.supporting_record_ids / "
              "contradiction_record_ids reference "
              "shipping-store record ids, and "
              "apply_event(source_record_id=...) "
              "appends the source record id — the "
              "layer holds relational state only and "
              "duplicates no fact",
     "status": "exists-in-production",
     "evidence": "src/thoughtforge/knowledge/bond_"
                 "graph.py (TF-143, parity exact with "
                 "TF-139's spike)"},
    {"id": "bond_excerpt_bridge",
     "name": "bonds as WYRD-consumable data",
     "direction": "thoughtforge -> wyrd",
     "shape": "the source shape exists: BondGraph."
              "excerpt(bond_id) files deterministic "
              "lines (domain / status / the three "
              "indices / relational mode / emotional "
              "weather / unresolved hurts / vows — 9 "
              "lines on the parity fixtures); the "
              "bridge that files those lines as "
              "WYRD-consumable data is TF-146's work "
              "and is NOT built here",
     "status": "not-built-tf146",
     "evidence": "TF-139's conditions + TF-143's "
                 "filing, both naming TF-146 as the "
                 "bridge's owner; no bridge code "
                 "stands in this tree"},
    {"id": "cognition_turn",
     "name": "the cognition turn a consumer calls",
     "direction": "consumer -> thoughtforge",
     "shape": "turn-in / turn-out through "
              "ThoughtForgeCore.think() / "
              "think_stream(), with the turn's traces "
              "and citation check filed on the core; "
              "service-shaped bindings over this "
              "surface are Part IX's work (TF-153.."
              "TF-158), not this boundary's",
     "status": "exists-in-production",
     "evidence": "src/thoughtforge/cognition/core.py "
                 "(ThoughtForgeCore.think) + the "
                 "roadmap's Part IX filing"},
)


# ── The consumers this doc is written for ────────────────────────────────────

CONSUMER_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "second_life_heimdall",
     "consumer": "Second Life / Heimdall",
     "consumes": "the cognition turn and, once "
                 "TF-146 builds it, the bond bridge's "
                 "WYRD-consumable bond data — a body "
                 "in a lived world thinking with "
                 "ThoughtForge while the world's state "
                 "stays WYRD-side",
     "status": "named-consumer-not-integrated",
     "evidence": "TF-145's roadmap text names the "
                 "Second Life consumers; no Heimdall "
                 "or Second Life code stands under "
                 "src/ in this tree"},
    {"id": "hermes",
     "consumer": "Hermes",
     "consumes": "ThoughtForge as a cognition "
                 "provider under the volmarr-core "
                 "shaped contract (TF-158), with the "
                 "campaign's contracts handed over in "
                 "the portable shapes (TF-193); memory "
                 "portability is already filed on that "
                 "law",
     "status": "named-consumer-not-integrated",
     "evidence": "the roadmap's TF-158 / TF-193 "
                 "filings + src/thoughtforge/knowledge/"
                 "portability.py (Hermes-portability "
                 "law applied to memory); no Hermes "
                 "binding stands in this tree"},
)


# ── Pure validators (the probes run against these) ───────────────────────────

def ownership_row_problems(row: Any) -> list[str]:
    """Judge one ownership filing, purely: an id, a
    subject, an owner in the filed vocabulary, a
    meaning and evidence in this repo. A WYRD row
    must not cite shipping code as its evidence —
    no WYRD code stands in this repo. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["ownership: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("ownership: the row files no id")
    if not row.get("subject"):
        problems.append(
            f"ownership {row.get('id')}: no subject filed")
    if row.get("owner") not in OWNERS:
        problems.append(
            f"ownership {row.get('id')}: owner is "
            f"{row.get('owner')!r}, not one of {OWNERS}")
    if not row.get("meaning"):
        problems.append(
            f"ownership {row.get('id')}: no meaning filed")
    if not row.get("evidence"):
        problems.append(
            f"ownership {row.get('id')}: no evidence "
            "filed — a boundary claim must stand on "
            "evidence in this repo")
    if row.get("owner") == "wyrd" \
            and "src/thoughtforge" in str(row.get("evidence", "")) \
            and "bond" not in str(row.get("evidence", "")):
        problems.append(
            f"ownership {row.get('id')}: a WYRD row "
            "cites shipping code as its evidence — no "
            "WYRD code stands in this repo")
    return problems


def interface_row_problems(row: Any) -> list[str]:
    """Judge one interface filing, purely: an id, a
    name, a direction, a shape, evidence, and a
    status in the filed vocabulary — which carries
    no live status, so a row claiming live
    integration is named here. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["interface: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("interface: the row files no id")
    for key in ("name", "direction", "shape", "evidence"):
        if not row.get(key):
            problems.append(
                f"interface {row.get('id')}: no {key} filed")
    if row.get("status") not in INTERFACE_STATUSES:
        problems.append(
            f"interface {row.get('id')}: status is "
            f"{row.get('status')!r}, not one of "
            f"{INTERFACE_STATUSES} — no live WYRD "
            "integration exists in this tree to file")
    return problems


def consumer_row_problems(row: Any) -> list[str]:
    """Judge one consumer filing, purely: an id, the
    consumer, what it consumes, its evidence, and
    the not-integrated status this tree proves.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["consumer: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("consumer: the row files no id")
    for key in ("consumer", "consumes", "evidence"):
        if not row.get(key):
            problems.append(
                f"consumer {row.get('id')}: no {key} filed")
    if row.get("status") != "named-consumer-not-integrated":
        problems.append(
            f"consumer {row.get('id')}: status is "
            f"{row.get('status')!r} — neither consumer "
            "is integrated in this tree")
    return problems


def boundary_state_problems(state: Any, expected: Any) -> list[str]:
    """Judge a measured boundary state against its
    expected filing, purely. Empty-safe."""
    if not isinstance(state, dict) or not state:
        return ["boundary state: the state is empty"]
    if not isinstance(expected, dict) or not expected:
        return ["boundary state: the expected filing is empty"]
    problems: list[str] = []
    for key, value in expected.items():
        if state.get(key) != value:
            problems.append(
                f"boundary state: {key} measures "
                f"{state.get(key)!r}, not the filed {value!r}")
    return problems


# ── The live boundary state ──────────────────────────────────────────────────

#: The filed expectation for the boundary —
#: deterministic counts and booleans only.
BOUNDARY_EXPECTED: dict[str, Any] = {
    "ownership_rows": 12,
    "thoughtforge_owns": 6,
    "wyrd_owns": 3,
    "neither_owns": 3,
    "interface_rows": 5,
    "interfaces_in_production": 2,
    "interfaces_fixture_grade": 2,
    "interfaces_not_built": 1,
    "consumers": 2,
    "wyrd_panel_cases": 18,
    "wyrd_panel_consistent": 9,
    "wyrd_panel_contradicted": 8,
    "wyrd_panel_honest": 1,
    "wyrd_panel_verdict": "PASS",
    "bond_excerpt_lines": 9,
    "live_wyrd_interface_files": 0,
    "live_client_tokens": 0,
    "heimdall_mentions_in_src": 0,
}


def run_boundary_state() -> dict[str, Any]:
    """Measure the boundary on the live tree: the
    filings' counts, TF-132's panel re-run, TF-143's
    bond parity excerpt, and the absence any live
    WYRD interface would contradict — no src file
    named for WYRD outside the boundary filings, no
    live-client token, no Heimdall mention under
    src/. Deterministic: counts only."""
    from benchmarks.wyrd_panel import run_wyrd_panel
    from thoughtforge.knowledge.bond_graph import run_bond_parity

    panel = run_wyrd_panel()
    parity = run_bond_parity()
    interface_files = 0
    client_tokens = 0
    heimdall_mentions = 0
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "wyrd" in path.name and \
                path.name not in _BOUNDARY_FILING_FILES:
            interface_files += 1
        if path.name == "wyrd_boundary.py":
            continue  # this filing names the tokens it scans for
        if any(token in text for token in _LIVE_CLIENT_TOKENS):
            client_tokens += 1
        if "Heimdall" in text or "heimdall" in text:
            heimdall_mentions += 1
    statuses = [r["status"] for r in INTERFACE_ROWS]
    return {
        "ownership_rows": len(OWNERSHIP_ROWS),
        "thoughtforge_owns": sum(
            1 for r in OWNERSHIP_ROWS
            if r["owner"] == "thoughtforge"),
        "wyrd_owns": sum(
            1 for r in OWNERSHIP_ROWS if r["owner"] == "wyrd"),
        "neither_owns": sum(
            1 for r in OWNERSHIP_ROWS if r["owner"] == "neither"),
        "interface_rows": len(INTERFACE_ROWS),
        "interfaces_in_production": statuses.count(
            "exists-in-production"),
        "interfaces_fixture_grade": statuses.count(
            "fixture-stand-in-only") + statuses.count(
            "exists-as-panel-fixture"),
        "interfaces_not_built": statuses.count("not-built-tf146"),
        "consumers": len(CONSUMER_ROWS),
        "wyrd_panel_cases": panel.total_cases,
        "wyrd_panel_consistent": panel.consistent_cases,
        "wyrd_panel_contradicted": panel.contradicted_cases,
        "wyrd_panel_honest": panel.honest_unanswerable_cases,
        "wyrd_panel_verdict": panel.verdict,
        "bond_excerpt_lines": len(parity["excerpt"]),
        "live_wyrd_interface_files": interface_files,
        "live_client_tokens": client_tokens,
        "heimdall_mentions_in_src": heimdall_mentions,
    }


# ── The filed probes ─────────────────────────────────────────────────────────

BOUNDARY_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_boundary_filing_passes",
     "kind": "clean",
     "detail": "the filed ownership + interface + "
               "consumer rows pass their validators"},
    {"id": "ownership_without_evidence_is_named",
     "kind": "ownership_no_evidence",
     "detail": "an ownership row with its evidence "
               "emptied is named"},
    {"id": "unknown_owner_is_named",
     "kind": "unknown_owner",
     "detail": "an ownership row filing an owner "
               "outside the filed vocabulary is named"},
    {"id": "wyrd_row_citing_shipping_code_is_named",
     "kind": "wyrd_cites_code",
     "detail": "a WYRD row citing a shipping code "
               "path as its evidence is named — no "
               "WYRD code stands in this repo"},
    {"id": "live_interface_claim_is_named",
     "kind": "live_status",
     "detail": "an interface row claiming a live "
               "status outside the filed vocabulary "
               "is named"},
    {"id": "interface_without_a_shape_is_named",
     "kind": "interface_no_shape",
     "detail": "an interface row with its shape "
               "emptied is named"},
    {"id": "consumer_claiming_integration_is_named",
     "kind": "consumer_integrated",
     "detail": "a consumer row filing any status "
               "but not-integrated is named"},
)


def run_boundary_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in BOUNDARY_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            problems: list[str] = []
            for row in OWNERSHIP_ROWS:
                problems += ownership_row_problems(row)
            for row in INTERFACE_ROWS:
                problems += interface_row_problems(row)
            for row in CONSUMER_ROWS:
                problems += consumer_row_problems(row)
            passed = problems == []
        elif kind == "ownership_no_evidence":
            row = dict(OWNERSHIP_ROWS[0], evidence="")
            passed = bool(ownership_row_problems(row))
        elif kind == "unknown_owner":
            row = dict(OWNERSHIP_ROWS[0], owner="the-model")
            passed = bool(ownership_row_problems(row))
        elif kind == "wyrd_cites_code":
            row = dict(
                [r for r in OWNERSHIP_ROWS
                 if r["owner"] == "wyrd"][0],
                evidence="src/thoughtforge/cognition/core.py")
            passed = bool(ownership_row_problems(row))
        elif kind == "live_status":
            row = dict(INTERFACE_ROWS[0], status="live-integrated")
            passed = bool(interface_row_problems(row))
        elif kind == "interface_no_shape":
            row = dict(INTERFACE_ROWS[0], shape="")
            passed = bool(interface_row_problems(row))
        elif kind == "consumer_integrated":
            row = dict(CONSUMER_ROWS[0], status="integrated")
            passed = bool(consumer_row_problems(row))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the boundary doc, verbatim) ─────────────────────────

def render_wyrd_boundary_markdown() -> str:
    """Render the boundary doc deterministically —
    the exact text filed as
    docs/architecture/WYRD_BOUNDARY.md."""
    state = run_boundary_state()
    lines: list[str] = []
    lines.append("# The WYRD Boundary — ThoughtForge vs the WYRD Protocol (TF-145)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/wyrd_boundary.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.wyrd_boundary`. The "
        "checker (`wyrd_boundary_problems()`) re-derives "
        "every claim live from the tree and fails on any "
        "drift.")
    lines.append("")
    lines.append(f"Format: `{WYRD_BOUNDARY_FORMAT}`")
    lines.append(
        f"Ownership rows: {state['ownership_rows']} "
        f"(ThoughtForge {state['thoughtforge_owns']} / "
        f"WYRD {state['wyrd_owns']} / neither "
        f"{state['neither_owns']})")
    lines.append(
        f"Interface rows: {state['interface_rows']} "
        f"(in production {state['interfaces_in_production']} "
        f"/ fixture grade {state['interfaces_fixture_grade']} "
        f"/ not built {state['interfaces_not_built']})")
    lines.append(f"Consumers named: {state['consumers']}")
    lines.append(f"Probes: {len(BOUNDARY_PROBES)}/"
                 f"{len(BOUNDARY_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was "
        "built: **no WYRD interface code exists in the "
        "shipping tree** — a grep for WYRD across "
        "`src/thoughtforge/` finds the word only in "
        "filings about this boundary (TF-139's verdict "
        "module and TF-143's production bond graph, both "
        "naming TF-146 as the bridge's owner). There is "
        "no WYRD client, no protocol implementation, no "
        "live feed, and no file under `src/` named for "
        "WYRD outside those filings (measured live by "
        "this doc's checker: "
        f"{state['live_wyrd_interface_files']} interface "
        f"files, {state['live_client_tokens']} live-client "
        "tokens). **WYRD itself lives outside this "
        "repo**, and this doc claims no live WYRD "
        "integration anywhere — live behaviour is NOT "
        "PROVEN (TF-173 class), never claimed. The "
        "evidence the boundary rests on stood clean "
        "live: TF-132's panel re-runs here at "
        f"{state['wyrd_panel_cases']} cases / "
        f"{state['wyrd_panel_consistent']} consistent / "
        f"{state['wyrd_panel_contradicted']} contradicted / "
        f"{state['wyrd_panel_honest']} honest-unanswerable, "
        f"verdict {state['wyrd_panel_verdict']}; TF-143's "
        "bond parity excerpt files "
        f"{state['bond_excerpt_lines']} lines.")
    lines.append("")
    lines.append("## 2. What each side owns")
    lines.append("")
    lines.append("| Subject | Owner | Evidence in this repo |")
    lines.append("|---|---|---|")
    for row in OWNERSHIP_ROWS:
        lines.append(
            f"| {row['subject']} | **{row['owner']}** | "
            f"{row['evidence']} |")
    lines.append("")
    for row in OWNERSHIP_ROWS:
        lines.append(f"### {row['id']} — {row['owner']}")
        lines.append("")
        lines.append(row["meaning"])
        lines.append("")
    lines.append("## 3. The interface between them")
    lines.append("")
    lines.append(
        "Five filed shapes, each with its honest status. "
        "There is deliberately no live status in the "
        "vocabulary: a shape is in production, fixture "
        "grade, or not built — and the one bridge that "
        "would carry ThoughtForge data to WYRD "
        "consumers is the not-built one, TF-146's.")
    lines.append("")
    for row in INTERFACE_ROWS:
        lines.append(
            f"### {row['id']} — {row['name']} "
            f"[{row['status']}]")
        lines.append("")
        lines.append(f"Direction: {row['direction']}")
        lines.append("")
        lines.append(f"Shape: {row['shape']}")
        lines.append("")
        lines.append(f"Evidence: {row['evidence']}")
        lines.append("")
    lines.append("## 4. Written for the consumers")
    lines.append("")
    for row in CONSUMER_ROWS:
        lines.append(f"### {row['consumer']} [{row['status']}]")
        lines.append("")
        lines.append(f"Consumes: {row['consumes']}")
        lines.append("")
        lines.append(f"Evidence: {row['evidence']}")
        lines.append("")
    lines.append(
        "For a Second Life / Heimdall reader: your body "
        "owns embodiment and transport (neither row "
        "`embodiment_and_transport`); the world you stand "
        "in is WYRD-side; the thinking you do with it is "
        "ThoughtForge-side, through the cognition turn — "
        "and the bond data becomes WYRD-consumable only "
        "when TF-146 builds its bridge. For a Hermes "
        "reader: ThoughtForge is a cognition provider "
        "under TF-158's contract; nothing in this "
        "boundary makes WYRD a Hermes component or "
        "Hermes a world model.")
    lines.append("")
    lines.append("## 5. Docket")
    lines.append("")
    lines.append(
        "TF-145 has **no share** of TF-016's D-WING-1: "
        "the entry assigns the verdicts (TF-139.."
        "TF-142), the adoptions (TF-143), the "
        "retirements (TF-144), research hygiene "
        "(TF-148) and the wing's tests (TF-149) — a "
        "boundary doc over the estate those slices "
        "built is none of those shares. This slice "
        "therefore does not edit the docket and adds "
        "no entry; TF-016 totals stand unchanged (137 "
        "entries — 33 OPEN / 1 OPEN-ENVIRONMENT / 100 "
        "FIXED / 3 CLOSED-BY-VERDICT), filed in the "
        "open here, per the TF-144 pattern's rule that "
        "a slice touches the docket only for its own "
        "filed share.")
    lines.append("")
    lines.append("## 6. The probes")
    lines.append("")
    for probe in BOUNDARY_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 7. Filed limits")
    lines.append("")
    lines.append(
        "- Boundary grade only: this doc files who owns "
        "what and the shapes between them, grounded in "
        "this repo's evidence. It builds no bridge "
        "(TF-146), no binding (Part IX) and no live "
        "integration of any kind.")
    lines.append(
        "- The WYRD side is filed from the roadmap's "
        "own words plus the shapes TF-132's panel "
        "stands in for: WYRD's implementation lives "
        "outside this repo and was not read here, so "
        "no WYRD internal is described — only the "
        "boundary surface this estate can evidence.")
    lines.append(
        "- TF-132's world is synthetic and small on "
        "purpose (8 entities / 6 changes / 18 cases): "
        "it evidences the consistency shape, not scale, "
        "and its answers are scripted — a live system "
        "staying true to a lived world is NOT PROVEN "
        "(TF-173).")
    lines.append(
        "- No live model was read; no canon data was "
        "touched (TF-017); no README.md anywhere was "
        "touched.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_wyrd_boundary_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / WYRD_BOUNDARY_DOC
    doc.write_text(render_wyrd_boundary_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.wyrd_boundary`."""
    doc = write_wyrd_boundary_doc()
    print(f"wyrd boundary doc regenerated: -> {WYRD_BOUNDARY_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def wyrd_boundary_problems() -> list[str]:
    """Re-derive TF-145's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    row_ids = [r["id"] for r in OWNERSHIP_ROWS]
    if len(row_ids) != len(set(row_ids)) or len(OWNERSHIP_ROWS) != 12:
        problems.append("the OWNERSHIP_ROWS filing is not twelve "
                        "unique rows")
    for row in OWNERSHIP_ROWS:
        problems.extend(ownership_row_problems(row))
    iface_ids = [r["id"] for r in INTERFACE_ROWS]
    if len(iface_ids) != len(set(iface_ids)) or len(INTERFACE_ROWS) != 5:
        problems.append("the INTERFACE_ROWS filing is not five "
                        "unique rows")
    for row in INTERFACE_ROWS:
        problems.extend(interface_row_problems(row))
    consumer_ids = [r["id"] for r in CONSUMER_ROWS]
    if len(consumer_ids) != len(set(consumer_ids)) \
            or len(CONSUMER_ROWS) != 2:
        problems.append("the CONSUMER_ROWS filing is not two "
                        "unique rows")
    for row in CONSUMER_ROWS:
        problems.extend(consumer_row_problems(row))

    # The roadmap files this slice and TF-146's bridge.
    roadmap = _ROADMAP.read_text(encoding="utf-8")
    if "TF-145" not in roadmap or "WYRD boundary doc" not in roadmap:
        problems.append("roadmap: TF-145's boundary filing is missing")
    if "TF-146" not in roadmap or "bond-graph bridge" not in roadmap:
        problems.append("roadmap: TF-146's bridge filing is missing")
    if "world-model neighbour" not in roadmap:
        problems.append("roadmap: the WYRD neighbour filing is missing")

    # The evidence neighbours still stand clean (cheap
    # verdict values; their own checkers carry the
    # deep proof, re-run in this slice's tests).
    from thoughtforge.knowledge.bond_graph_verdict import (
        VERDICT as BOND_VERDICT,
    )

    if BOND_VERDICT.get("verdict") != "ADOPT_AS_LAYER":
        problems.append("evidence: TF-139 no longer files "
                        "ADOPT_AS_LAYER")

    # The measured state matches its filing (the panel
    # PASS, the bond excerpt at 9 lines, and NO live
    # WYRD interface anywhere in the shipping tree).
    problems.extend(boundary_state_problems(
        run_boundary_state(), BOUNDARY_EXPECTED))

    # The probes all behave.
    probes = run_boundary_probes()
    if len(probes) != len(BOUNDARY_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the boundary probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The boundary doc files the render verbatim.
    doc = _REPO_ROOT / WYRD_BOUNDARY_DOC
    if not doc.exists():
        problems.append(f"the boundary doc {WYRD_BOUNDARY_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != render_wyrd_boundary_markdown():
        problems.append(
            f"the filed doc {WYRD_BOUNDARY_DOC} is not the render "
            "verbatim (regenerate with python3 -m "
            "thoughtforge.wyrd_boundary)")
    return problems


def wyrd_boundary_as_data() -> dict[str, Any]:
    """The whole boundary as plain data, for consumers."""
    probes = run_boundary_probes()
    return {
        "format": WYRD_BOUNDARY_FORMAT,
        "doc": WYRD_BOUNDARY_DOC,
        "ownership": [dict(r) for r in OWNERSHIP_ROWS],
        "interfaces": [dict(r) for r in INTERFACE_ROWS],
        "consumers": [dict(r) for r in CONSUMER_ROWS],
        "state": run_boundary_state(),
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "BOUNDARY_EXPECTED",
    "BOUNDARY_PROBES",
    "CONSUMER_ROWS",
    "INTERFACE_ROWS",
    "INTERFACE_STATUSES",
    "OWNERS",
    "OWNERSHIP_ROWS",
    "WYRD_BOUNDARY_DOC",
    "WYRD_BOUNDARY_FORMAT",
    "boundary_state_problems",
    "consumer_row_problems",
    "interface_row_problems",
    "main",
    "ownership_row_problems",
    "render_wyrd_boundary_markdown",
    "run_boundary_probes",
    "run_boundary_state",
    "write_wyrd_boundary_doc",
    "wyrd_boundary_as_data",
    "wyrd_boundary_problems",
]

if __name__ == "__main__":
    raise SystemExit(main())
