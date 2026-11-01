"""
FragmentSalvage — multi-pass draft scoring and intelligent reassembly.

Implements the BUILD_PLAN_v1 Phase 4 salvage pipeline:
  1. Score each candidate draft: length_score (45%) + citation_score (55%)
  2. Extract best sentence-level fragments from top-scoring drafts
  3. If an inference engine is available: attempt a refine pass (max 2 passes)
  4. Return the best result as a SalvageResult

Design: works with or without an inference engine. Without one, returns the
best-scoring draft directly. With one, tries to synthesize fragments into a
stronger response.

Fragment keep threshold follows the Retrieval_and_Scoring_Spec: 0.54 default.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# TF-025 (D-DUP-1): _split_sentences' one implementation lives in
# utils/common.py; this name is its alias (existing tests import it
# from here).
from thoughtforge.utils.common import split_sentences as _split_sentences

# TF-088 (D-SALV-2): the refine adoption rule's safety guard —
# a pass is adopted only when it introduces no new checkable
# content (numbers, citations, named entities, polarity,
# displaced time markers) against the source pool. The line
# is specified in refinement/salvage_safety.py.
from thoughtforge.refinement.salvage_safety import fabrication_problems

# TF-096 (D-SALV-3): the persona-voice guard — when the
# caller supplies the persona in force, selection must
# not choose a flattened draft over an in-voice one at
# hand, and a refine pass is adopted only when it
# introduces no new persona avoid violation (TF-030's
# checkable subset) against the current best text. The
# judgement is specified in refinement/persona_voice.py.
from thoughtforge.refinement.persona_voice import (
    persona_violations,
    voice_problems,
)

# TF-097 (D-TRACE-1): every forge() carries its rescue
# record — candidates scored, selection, fragments, and
# each refine pass individually (attempted vs adopted,
# rejection reasons, failure) — specified in
# refinement/rescue_provenance.py.
from thoughtforge.refinement.rescue_provenance import (
    RESCUE_RECORD_FORMAT,
)

if TYPE_CHECKING:
    from thoughtforge.cognition.prompt_builder import PromptBuilder
    from thoughtforge.knowledge.models import (
        CandidateRecord,
        CognitionScaffold,
        InputSketch,
        MemoryActivationBundle,
    )

logger = logging.getLogger(__name__)

# ── Scoring constants (BUILD_PLAN_v1 Phase 4) ─────────────────────────────────

_LENGTH_WEIGHT = 0.45
_CITATION_WEIGHT = 0.55
_IDEAL_LENGTH_CHARS = 550          # chars at which length_score reaches 1.0
_MAX_REFINE_PASSES = 2
_MIN_FRAGMENT_SCORE = 0.54         # from Retrieval_and_Scoring_Spec §30
# TF-022 ruling on CORE_LOOP §8-F6: this threshold gates salvage's own
# length+citation score (see _score_fragment below) and decides which
# fragments are *worth a refine pass*. It is NOT the core's threshold:
# core._MIN_FRAGMENT_SCORE (0.25) gates the unrelated FragmentScores
# composite and decides only turn-record provenance. Different
# metrics, different jobs — the numbers are not a drift to reconcile.
_MIN_FRAGMENT_WORDS = 4
_MAX_FRAGMENTS_IN_REFINE = 5

_QID_RE = re.compile(r'\bQ\d{3,}\b')


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class ScoredDraft:
    """A candidate draft with its salvage score."""
    candidate_id: str
    text: str
    length_score: float
    citation_score: float
    composite: float
    citations_found: list[str] = field(default_factory=list)


@dataclass
class SalvageResult:
    """Output of FragmentSalvage.forge()."""
    text: str
    citations: list[str]
    confidence: float
    passes_used: int
    best_draft_score: float
    fragments_kept: int
    salvage_path: str        # "direct" | "refine_pass_1" | "refine_pass_2" | "best_draft"
    # TF-097: the rescue record (format
    # thoughtforge-rescue-record/1) — detached data,
    # never the live objects; None only on a result
    # built outside forge().
    rescue_record: dict[str, Any] | None = None


# ── FragmentSalvage ────────────────────────────────────────────────────────────

class FragmentSalvage:
    """
    Multi-pass draft scoring and fragment reassembly engine.

    Can operate with or without an inference engine:
      - With engine: attempts up to MAX_REFINE_PASSES refinement passes
      - Without engine: returns best-scoring draft directly
    """

    def forge(
        self,
        candidates: list["CandidateRecord"],
        bundle: "MemoryActivationBundle",
        engine: Any | None = None,
        prompt_builder: "PromptBuilder | None" = None,
        sketch: "InputSketch | None" = None,
        scaffold: "CognitionScaffold | None" = None,
        max_passes: int = _MAX_REFINE_PASSES,
        generation_seed: int | None = None,
        persona_expectation: Any | None = None,
    ) -> SalvageResult:
        """
        Score candidates and attempt fragment-based refinement.

        Args:
            candidates:     List of CandidateRecord from generation.
            bundle:         MemoryActivationBundle for citation validation.
            engine:         Optional TurboQuantEngine for refine passes.
            prompt_builder: Optional PromptBuilder for refine prompts.
            sketch:         Optional InputSketch for prompt context.
            scaffold:       Optional CognitionScaffold for prompt context.
            max_passes:     Max refinement passes (default 2).
            generation_seed: TF-036 — per-pass sampling seed base
                for the refine generations (pass i uses base + i);
                None = unseeded, the standing behaviour.
            persona_expectation: TF-096 — the persona in force
                (a PersonaExpectation), when the caller has one.
                Given, selection ranks only persona-consistent
                drafts when at least one is at hand and a refine
                pass introducing a new avoid violation is never
                adopted. None (the default) = every leg
                byte-identical to before TF-096.

        Returns:
            SalvageResult with best text, citations, confidence.
        """
        if not candidates:
            logger.debug("FragmentSalvage: no candidates — returning empty result")
            return SalvageResult(
                text="The forge finds no candidates to salvage.",
                citations=[],
                confidence=0.0,
                passes_used=0,
                best_draft_score=0.0,
                fragments_kept=0,
                salvage_path="empty",
                rescue_record={
                    "format": RESCUE_RECORD_FORMAT,
                    "candidates": [],
                    "selection": {
                        "selected_candidate_id": None,
                        "persona_filter_applied": False,
                        "persona_consistent_count": None,
                    },
                    "fragments_kept": 0,
                    "refine_attempted": False,
                    "refine_passes": [],
                    "final": {
                        "salvage_path": "empty",
                        "final_source": "empty",
                        "final_pass": None,
                        "passes_used": 0,
                        "confidence": 0.0,
                        "best_draft_score": 0.0,
                    },
                },
            )

        retrieved_qids = self._extract_qids(bundle)

        # Step 1: Score all drafts
        scored = [self._score_draft(c, retrieved_qids) for c in candidates]
        scored.sort(key=lambda s: s.composite, reverse=True)
        # TF-097: the record's candidate rows are the
        # scored drafts as scored (before any persona
        # filtering) — provenance is what was at hand.
        record_candidates = [
            {
                "candidate_id": s.candidate_id,
                "text_chars": len(s.text),
                "length_score": round(s.length_score, 6),
                "citation_score": round(s.citation_score, 6),
                "composite": round(s.composite, 6),
                "citations_found": list(s.citations_found),
            }
            for s in scored
        ]
        persona_consistent_count: int | None = None
        # TF-096 (D-SALV-3): with a persona in force, a
        # persona-consistent draft at hand outranks any
        # flattened rival — the composite rewards length and
        # cannot see voice, so the consistent drafts (score
        # order kept within them) become the ranked list.
        # When none is consistent the full ranking stands:
        # the guard restricts choice, it invents no voice.
        if persona_expectation is not None:
            consistent = [
                s for s in scored
                if not persona_violations(persona_expectation, s.text)
            ]
            persona_consistent_count = len(consistent)
            if consistent:
                scored = consistent
        best = scored[0]

        logger.debug(
            "FragmentSalvage: %d candidates scored. Best: id=%s composite=%.3f",
            len(scored), best.candidate_id, best.composite,
        )

        # Step 2: Extract best fragments
        fragments = self._extract_fragments(scored, retrieved_qids)
        logger.debug("FragmentSalvage: %d fragments kept (threshold=%.2f)", len(fragments), _MIN_FRAGMENT_SCORE)

        # Step 3: Attempt refine passes if engine available
        pass_records: list[dict[str, Any]] = []
        refine_attempted = (
            engine is not None and prompt_builder is not None
            and sketch is not None and scaffold is not None)
        if refine_attempted:
            result, pass_records = self._refine_with_engine(
                best_draft=best,
                fragments=fragments,
                bundle=bundle,
                retrieved_qids=retrieved_qids,
                engine=engine,
                prompt_builder=prompt_builder,
                sketch=sketch,
                scaffold=scaffold,
                max_passes=max_passes,
                generation_seed=generation_seed,
                persona_expectation=persona_expectation,
            )
            if result is not None:
                result.rescue_record = self._rescue_record(
                    candidates=record_candidates,
                    selected_id=best.candidate_id,
                    persona_filter_applied=(
                        persona_expectation is not None),
                    persona_consistent_count=persona_consistent_count,
                    fragments_kept=len(fragments),
                    refine_attempted=True,
                    pass_records=pass_records,
                    result=result,
                )
                return result

        # Step 4: Fall back to best-scoring draft directly
        fallback = SalvageResult(
            text=best.text,
            citations=best.citations_found,
            confidence=best.composite,
            passes_used=0,
            best_draft_score=best.composite,
            fragments_kept=len(fragments),
            salvage_path="best_draft",
        )
        fallback.rescue_record = self._rescue_record(
            candidates=record_candidates,
            selected_id=best.candidate_id,
            persona_filter_applied=(persona_expectation is not None),
            persona_consistent_count=persona_consistent_count,
            fragments_kept=len(fragments),
            refine_attempted=refine_attempted,
            pass_records=pass_records,
            result=fallback,
        )
        return fallback

    # ── TF-097: the rescue record ──────────────────────────────────────────

    @staticmethod
    def _rescue_record(
        *,
        candidates: list[dict[str, Any]],
        selected_id: str,
        persona_filter_applied: bool,
        persona_consistent_count: int | None,
        fragments_kept: int,
        refine_attempted: bool,
        pass_records: list[dict[str, Any]],
        result: "SalvageResult",
    ) -> dict[str, Any]:
        """The rescue record for one forge() (TF-097).

        Derived only from what forge() itself did and
        measured — scores it computed, passes it ran,
        latencies it timed. ``final_source`` answers
        TF-087's naming debt: ``salvage_path``
        refine_pass_N names passes *attempted*; the
        record states whether the final text is the
        selected draft or an adopted pass, and which.
        """
        adopted = [p for p in pass_records
                   if p.get("outcome") == "adopted"]
        if adopted:
            final_source = "refine_pass"
            final_pass: int | None = adopted[-1]["pass"]
        else:
            final_source = "selected_draft"
            final_pass = None
        return {
            "format": RESCUE_RECORD_FORMAT,
            "candidates": candidates,
            "selection": {
                "selected_candidate_id": selected_id,
                "persona_filter_applied": persona_filter_applied,
                "persona_consistent_count": persona_consistent_count,
            },
            "fragments_kept": fragments_kept,
            "refine_attempted": refine_attempted,
            "refine_passes": pass_records,
            "final": {
                "salvage_path": result.salvage_path,
                "final_source": final_source,
                "final_pass": final_pass,
                "passes_used": result.passes_used,
                "confidence": round(result.confidence, 6),
                "best_draft_score": round(result.best_draft_score, 6),
            },
        }

    # ── Draft scoring ──────────────────────────────────────────────────────────

    def _score_draft(self, candidate: "CandidateRecord", retrieved_qids: set[str]) -> ScoredDraft:
        """
        Score a candidate draft using the BUILD_PLAN_v1 formula:
          composite = length_score * 0.45 + citation_score * 0.55
        """
        text = candidate.text or ""

        # Length score: 0.0 → 1.0 as char count approaches ideal
        length_score = min(1.0, len(text) / _IDEAL_LENGTH_CHARS)

        # Citation score: fraction of retrieved QIDs that appear in the text
        cited = set(_QID_RE.findall(text))
        if retrieved_qids:
            citation_score = len(cited & retrieved_qids) / len(retrieved_qids)
        else:
            # No QIDs in knowledge base — treat as neutral
            citation_score = 0.5

        composite = length_score * _LENGTH_WEIGHT + citation_score * _CITATION_WEIGHT

        return ScoredDraft(
            candidate_id=candidate.candidate_id,
            text=text,
            length_score=length_score,
            citation_score=citation_score,
            composite=composite,
            citations_found=list(cited & retrieved_qids),
        )

    # ── Fragment extraction ────────────────────────────────────────────────────

    def _extract_fragments(
        self,
        scored_drafts: list[ScoredDraft],
        retrieved_qids: set[str],
    ) -> list[str]:
        """
        Extract and score sentence-level fragments from top-scoring drafts.
        Returns kept fragment texts sorted by their individual scores.
        """
        fragment_scores: list[tuple[str, float]] = []
        seen: set[str] = set()

        for draft in scored_drafts[:2]:     # only top 2 drafts contribute fragments
            sentences = _split_sentences(draft.text)
            for sentence in sentences:
                s = sentence.strip()
                if not s or len(s.split()) < _MIN_FRAGMENT_WORDS:
                    continue
                if s in seen:
                    continue
                seen.add(s)

                score = self._score_fragment(s, retrieved_qids)
                if score >= _MIN_FRAGMENT_SCORE:
                    fragment_scores.append((s, score))

        fragment_scores.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in fragment_scores[:_MAX_FRAGMENTS_IN_REFINE]]

    def _score_fragment(self, sentence: str, retrieved_qids: set[str]) -> float:
        """
        Score a single fragment sentence:
          composite = length_score * 0.45 + citation_score * 0.55
        (same formula as draft scoring, applied at sentence level)
        """
        length_score = min(1.0, len(sentence) / 200.0)
        if retrieved_qids:
            cited = set(_QID_RE.findall(sentence))
            citation_score = 0.8 if (cited & retrieved_qids) else 0.4
        else:
            citation_score = 0.5
        return length_score * _LENGTH_WEIGHT + citation_score * _CITATION_WEIGHT

    # ── Refine passes ──────────────────────────────────────────────────────────

    def _refine_with_engine(
        self,
        best_draft: ScoredDraft,
        fragments: list[str],
        bundle: "MemoryActivationBundle",
        retrieved_qids: set[str],
        engine: Any,
        prompt_builder: "PromptBuilder",
        sketch: "InputSketch",
        scaffold: "CognitionScaffold",
        max_passes: int,
        generation_seed: int | None = None,
        persona_expectation: Any | None = None,
    ) -> tuple[SalvageResult | None, list[dict[str, Any]]]:
        """
        Attempt up to max_passes refine generations using salvaged fragments.
        Returns (the best result, the TF-097 per-pass records), or
        (None, records) if all passes fail — the records ride
        either way, so a failed attempt is provenance too.
        """
        from thoughtforge.knowledge.models import FragmentRecord, FragmentScores

        current_best_text = best_draft.text
        current_best_score = best_draft.composite
        passes_used = 0
        pass_records: list[dict[str, Any]] = []

        # Build FragmentRecord stubs for PromptBuilder
        frag_records = [
            FragmentRecord(
                fragment_id=f"frag_{uuid.uuid4().hex[:6]}",
                source_candidate_id=best_draft.candidate_id,
                text=frag_text,
                keep=True,
                scores=FragmentScores(),
            )
            for frag_text in fragments
        ]

        for pass_num in range(1, max_passes + 1):
            if not frag_records:
                break

            # TF-097: the pass record is built alongside
            # the pass itself — outcome, score, rejection
            # reasons, and the wall latency this method
            # measures around the engine call. A record
            # is appended for every attempt, adopted or
            # not, failed or not.
            record: dict[str, Any] = {
                "pass": pass_num,
                "outcome": "failed",
                "counted_in_passes_used": False,
                "score": None,
                "text_chars": None,
                "fabrication_problems": [],
                "voice_problems": [],
                "error": None,
                "latency_ms": None,
                "reported_latency_ms": None,
            }
            pass_records.append(record)
            try:
                from thoughtforge.inference.turboquant import GenerationParams
                refine_prompt = prompt_builder.build_refine_prompt(
                    fragments=frag_records,
                    sketch=sketch,
                    scaffold=scaffold,
                )
                t_pass = time.perf_counter()
                result = engine.generate(
                    refine_prompt,
                    params=GenerationParams(
                        temperature=0.50 - (pass_num - 1) * 0.05,
                        # TF-036: pass i of a seeded compose carries
                        # generation_seed + i (the core draws one
                        # seed per compose; None = unseeded, as
                        # before this slice).
                        seed=(
                            generation_seed + (pass_num - 1)
                            if generation_seed is not None
                            else None
                        ),
                    ),
                )
                record["latency_ms"] = round(
                    (time.perf_counter() - t_pass) * 1000, 3)
                reported = getattr(result, "duration_ms", None)
                record["reported_latency_ms"] = (
                    round(float(reported), 3)
                    if reported is not None else None)
                passes_used = pass_num
                record["counted_in_passes_used"] = True

                # Score the refine output
                refined_text = result.text
                scored_refined = self._score_raw_text(refined_text, retrieved_qids)
                record["score"] = round(scored_refined, 6)
                record["text_chars"] = len(refined_text)

                logger.debug(
                    "FragmentSalvage refine pass %d: score=%.3f (prev=%.3f)",
                    pass_num, scored_refined, current_best_score,
                )

                # TF-088 (D-SALV-2): safety gate BEFORE the
                # score gate — a fabricated pass is never
                # adopted, however well it scores; the attempt
                # still counts in passes_used (TF-087 naming).
                safety_problems = fabrication_problems(
                    [current_best_text, *fragments,
                     *(rec.cue for rec in
                       bundle.activated_records)],
                    refined_text, retrieved_qids)
                # TF-096 (D-SALV-3): the voice gate, beside
                # TF-088's — with a persona in force, a pass
                # introducing a new avoid violation against
                # the current best text is never adopted,
                # however well it scores; the attempt still
                # counts in passes_used (TF-087 naming).
                voice_new = (
                    voice_problems(
                        persona_expectation, current_best_text,
                        refined_text)
                    if persona_expectation is not None else []
                )
                record["fabrication_problems"] = list(safety_problems)
                record["voice_problems"] = list(voice_new)
                if safety_problems:
                    record["outcome"] = "rejected_fabrication"
                    logger.warning(
                        "FragmentSalvage refine pass %d rejected "
                        "as fabrication: %s",
                        pass_num, safety_problems)
                elif voice_new:
                    record["outcome"] = "rejected_voice"
                    logger.warning(
                        "FragmentSalvage refine pass %d rejected "
                        "as voice-breaking: %s",
                        pass_num, voice_new)
                elif scored_refined > current_best_score:
                    record["outcome"] = "adopted"
                    current_best_text = refined_text
                    current_best_score = scored_refined
                    # Update fragments for next pass from the refined text
                    next_fragments = self._extract_fragments(
                        [ScoredDraft(
                            candidate_id=f"refine_{pass_num}",
                            text=refined_text,
                            length_score=min(1.0, len(refined_text) / _IDEAL_LENGTH_CHARS),
                            citation_score=0.5,
                            composite=scored_refined,
                        )],
                        retrieved_qids,
                    )
                    if next_fragments:
                        frag_records = [
                            FragmentRecord(
                                fragment_id=f"frag_{uuid.uuid4().hex[:6]}",
                                source_candidate_id=f"refine_{pass_num}",
                                text=t,
                                keep=True,
                                scores=FragmentScores(),
                            )
                            for t in next_fragments
                        ]
                else:
                    record["outcome"] = "not_improved"

            except Exception as e:
                record["outcome"] = "failed"
                record["error"] = str(e)
                logger.warning("FragmentSalvage refine pass %d failed: %s", pass_num, e)
                break

        if passes_used == 0:
            return None, pass_records

        cited = set(_QID_RE.findall(current_best_text))
        return SalvageResult(
            text=current_best_text,
            citations=list(cited & retrieved_qids),
            confidence=current_best_score,
            passes_used=passes_used,
            best_draft_score=best_draft.composite,
            fragments_kept=len(fragments),
            salvage_path=f"refine_pass_{passes_used}",
        ), pass_records

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _score_raw_text(self, text: str, retrieved_qids: set[str]) -> float:
        """Score raw text string using the build-plan formula."""
        length_score = min(1.0, len(text) / _IDEAL_LENGTH_CHARS)
        if retrieved_qids:
            cited = set(_QID_RE.findall(text))
            citation_score = len(cited & retrieved_qids) / len(retrieved_qids)
        else:
            citation_score = 0.5
        return length_score * _LENGTH_WEIGHT + citation_score * _CITATION_WEIGHT

    @staticmethod
    def _extract_qids(bundle: "MemoryActivationBundle") -> set[str]:
        qids: set[str] = set()
        for rec in bundle.activated_records:
            raw = getattr(rec, "raw", {}) or {}
            qid = raw.get("qid", "")
            if qid:
                qids.add(qid)
        return qids

