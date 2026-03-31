"""
ThoughtForgeCore — the main memory-enforced cognition pipeline.

Orchestrates the full think() loop:
  1. InputRouter     → InputSketch
  2. KnowledgeForge  → MemoryActivationBundle
  3. ScaffoldBuilder → CognitionScaffold
  4. PromptBuilder   → candidate prompts
  5. TurboQuantEngine → draft generation
  6. Fragment scoring → FragmentRecord list
  7. Refine pass     → final text
  8. Quality gate    → repair pass if needed
  9. FinalResponseRecord assembly

Works without a model (knowledge-only mode) and without a knowledge DB
(model-only mode). Degrades gracefully in both cases.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from thoughtforge.cognition.prompt_builder import PromptBuilder
from thoughtforge.cognition.router import InputRouter
from thoughtforge.cognition.scaffold import ScaffoldBuilder
from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.models import (
    CandidateRecord,
    CandidateScores,
    CognitionScaffold,
    FinalResponseRecord,
    FinalResponseScores,
    FragmentRecord,
    FragmentScores,
    InputSketch,
    MemoryActivationBundle,
    RuntimeTurnState,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.utils.config import load_config
from thoughtforge.utils.paths import get_knowledge_db_path, get_memory_dir

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_QUALITY_REPAIR_THRESHOLD = 0.40     # below this composite score → trigger repair
_MIN_FRAGMENT_SCORE = 0.25
_MIN_RESPONSE_TOKENS = 8             # fewer tokens than this → probably garbage


# ── ThoughtForgeCore ───────────────────────────────────────────────────────────

class ThoughtForgeCore:
    """
    Central orchestrator for ThoughtForge's memory-enforced cognition pipeline.

    Usage (with a GGUF model):
        core = ThoughtForgeCore(model_path="/models/phi-3-mini-q4.gguf")
        result = core.think("What is the Yggdrasil?")
        print(result.text)

    Usage (knowledge-only, no model):
        core = ThoughtForgeCore()
        result = core.think("What is ConceptNet?")
        # result.text will be assembled from retrieved knowledge context
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
        memory_dir: Path | None = None,
        db_path: Path | None = None,
    ) -> None:
        self._config = config or load_config()
        self._model_path = Path(model_path) if model_path else None

        self._knowledge = KnowledgeForge(
            db_path=db_path or get_knowledge_db_path(),
        )
        self._store = MemoryStore(memory_dir=memory_dir or get_memory_dir())
        self._router = InputRouter()
        self._scaffold_builder = ScaffoldBuilder()
        self._prompt_builder = PromptBuilder()

        self._engine: Any = None      # TurboQuantEngine — loaded lazily
        self._personality = self._store.load_personality_core()

        # Try to load personality from config path if not in memory dir
        if self._personality is None:
            self._personality = _load_personality_from_config(self._config)

        logger.info(
            "ThoughtForgeCore ready | model=%s | personality=%s | db=%s",
            self._model_path or "none (knowledge-only mode)",
            self._personality.record_id if self._personality else "none",
            db_path or get_knowledge_db_path(),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def think(
        self,
        user_text: str,
        retrieval_path: str | None = None,
        num_drafts: int | None = None,
    ) -> FinalResponseRecord:
        """
        Full memory-enforced cognition pipeline for one user turn.

        Args:
            user_text:      Raw user message.
            retrieval_path: Override retrieval path ("sql" | "vector" | "hybrid").
            num_drafts:     Override draft count (default from hardware profile).

        Returns:
            FinalResponseRecord — always returns, never raises on content failure.
        """
        turn_id = _new_turn_id()
        turn = RuntimeTurnState(turn_id=turn_id)
        t_start = time.perf_counter()

        logger.info("think() turn=%s text=%r", turn_id, user_text[:80])

        # ── Step 1: Route input ────────────────────────────────────────────────
        thread_state = self._store.load_thread_state()
        sketch = self._router.route(user_text, thread_state=thread_state)
        if retrieval_path:
            sketch.retrieval_path = retrieval_path
        turn.input_sketch = sketch

        # ── Step 2: Retrieve knowledge ─────────────────────────────────────────
        t_ret = time.perf_counter()
        bundle = self._retrieve(sketch)
        turn.memory_bundle = bundle
        turn.retrieval_ms = int((time.perf_counter() - t_ret) * 1000)
        logger.debug("Retrieved %d records in %dms", bundle.total_retrieved, turn.retrieval_ms)

        # ── Step 3: Build scaffold ─────────────────────────────────────────────
        scaffold = self._scaffold_builder.build(sketch, bundle, self._personality)
        turn.scaffold = scaffold

        # ── Step 4–6: Generate, score, salvage ────────────────────────────────
        t_gen = time.perf_counter()
        candidates = self._generate_candidates(sketch, scaffold, bundle, num_drafts)
        turn.candidates = candidates

        fragments = self._score_and_extract_fragments(candidates, sketch, scaffold)
        turn.fragments = fragments
        turn.generation_ms = int((time.perf_counter() - t_gen) * 1000)

        # ── Step 7: Refine or fallback to knowledge-only summary ───────────────
        if candidates:
            final = self._compose_final(sketch, scaffold, bundle, candidates, fragments)
        else:
            final = self._knowledge_only_response(turn_id, sketch, bundle)

        # ── Step 8: Quality gate → repair if needed ────────────────────────────
        if final.scores.composite < _QUALITY_REPAIR_THRESHOLD and candidates:
            logger.debug(
                "Quality below threshold (%.3f < %.3f) — attempting repair",
                final.scores.composite, _QUALITY_REPAIR_THRESHOLD,
            )
            repaired = self._repair(turn_id, sketch, scaffold, bundle, final)
            if repaired.scores.composite > final.scores.composite:
                final = repaired

        turn.final_response = final
        turn.total_tokens_used = final.token_count
        turn.completed_at = _now_iso()

        logger.info(
            "think() done turn=%s | score=%.3f tier=%s | tokens=%d | %dms total",
            turn_id,
            final.scores.composite,
            final.scores.quality_tier,
            final.token_count,
            int((time.perf_counter() - t_start) * 1000),
        )

        return final

    def load_model(self, model_path: str | Path | None = None) -> None:
        """Explicitly load the inference model. Called automatically on first think() if not done."""
        from thoughtforge.inference.turboquant import TurboQuantEngine
        path = model_path or self._model_path
        self._engine = TurboQuantEngine(model_path=path)
        self._engine.load()
        logger.info("Inference model loaded: %s", path)

    def unload_model(self) -> None:
        """Release the model from memory."""
        if self._engine is not None:
            self._engine.unload()
            self._engine = None

    # ── Internal pipeline steps ────────────────────────────────────────────────

    def _retrieve(self, sketch: InputSketch) -> MemoryActivationBundle:
        """Retrieve knowledge records. Returns empty bundle if DB unavailable."""
        db = get_knowledge_db_path()
        if not db.exists():
            logger.debug("Knowledge DB not found at %s — skipping retrieval", db)
            return MemoryActivationBundle()

        try:
            return self._knowledge.retrieve(
                query=sketch.raw_text,
                path=sketch.retrieval_path,
                top_k=10,
            )
        except Exception as e:
            logger.warning("Knowledge retrieval failed: %s — using empty bundle", e)
            return MemoryActivationBundle()

    def _generate_candidates(
        self,
        sketch: InputSketch,
        scaffold: CognitionScaffold,
        bundle: MemoryActivationBundle,
        num_drafts: int | None,
    ) -> list[CandidateRecord]:
        """Generate candidate responses via TurboQuantEngine. Returns [] if no model."""
        if self._engine is None and self._model_path is not None:
            try:
                self.load_model()
            except Exception as e:
                logger.warning("Failed to load model: %s — running knowledge-only", e)
                return []

        if self._engine is None:
            return []

        memory_cues = self._prompt_builder.extract_memory_cues(bundle)
        candidates: list[CandidateRecord] = []
        modes = scaffold.candidate_modes[:2]    # max 2 candidate passes

        for mode in modes:
            prompt = self._prompt_builder.build_candidate_prompt(
                sketch=sketch,
                scaffold=scaffold,
                mode=mode,
                memory_cues=memory_cues,
            )
            try:
                from thoughtforge.inference.turboquant import GenerationParams
                result = self._engine.generate(
                    prompt,
                    params=GenerationParams(
                        temperature=_mode_temperature(mode),
                    ),
                )
                cid = f"cand_{uuid.uuid4().hex[:6]}"
                candidates.append(CandidateRecord(
                    candidate_id=cid,
                    mode=mode,
                    text=result.text,
                    token_estimate=result.tokens_generated,
                    scores=_score_candidate(result.text, sketch, scaffold),
                ))
            except Exception as e:
                logger.warning("Candidate generation failed (mode=%s): %s", mode, e)

        return candidates

    def _score_and_extract_fragments(
        self,
        candidates: list[CandidateRecord],
        sketch: InputSketch,
        scaffold: CognitionScaffold,
    ) -> list[FragmentRecord]:
        """
        Score each candidate and extract the best sentence-level fragments.
        Each candidate is split into sentences; each sentence scored independently.
        """
        all_fragments: list[FragmentRecord] = []
        for cand in candidates:
            sentences = _split_sentences(cand.text)
            for i, sentence in enumerate(sentences):
                if len(sentence.split()) < 4:
                    continue
                fscores = _score_fragment(sentence, sketch, scaffold)
                fid = f"frag_{uuid.uuid4().hex[:6]}"
                keep = fscores.composite >= _MIN_FRAGMENT_SCORE
                all_fragments.append(FragmentRecord(
                    fragment_id=fid,
                    source_candidate_id=cand.candidate_id,
                    text=sentence,
                    position=i,
                    scores=fscores,
                    keep=keep,
                ))

        all_fragments.sort(key=lambda f: f.scores.composite, reverse=True)
        return all_fragments

    def _compose_final(
        self,
        sketch: InputSketch,
        scaffold: CognitionScaffold,
        bundle: MemoryActivationBundle,
        candidates: list[CandidateRecord],
        fragments: list[FragmentRecord],
    ) -> FinalResponseRecord:
        """
        Compose the final response:
        - If good fragments exist: build refine prompt → generate refined response.
        - Otherwise: use the best candidate directly.
        """
        rid = f"resp_{uuid.uuid4().hex[:8]}"
        kept = [f for f in fragments if f.keep]

        # Try refine pass if we have fragments and a model
        if kept and self._engine is not None:
            memory_cues = self._prompt_builder.extract_memory_cues(bundle)
            refine_prompt = self._prompt_builder.build_refine_prompt(
                fragments=fragments,
                sketch=sketch,
                scaffold=scaffold,
            )
            try:
                from thoughtforge.inference.turboquant import GenerationParams
                result = self._engine.generate(
                    refine_prompt,
                    params=GenerationParams(temperature=0.55),
                )
                scores = _score_final(result.text, sketch, scaffold)
                return FinalResponseRecord(
                    response_id=rid,
                    text=result.text,
                    source_candidate_ids=[c.candidate_id for c in candidates],
                    source_fragment_ids=[f.fragment_id for f in kept],
                    citations=_extract_citations(result.text, bundle),
                    scores=scores,
                    enforcement_passed=_check_citation_gate(result.text, bundle),
                    token_count=result.tokens_generated,
                )
            except Exception as e:
                logger.warning("Refine pass failed: %s — falling back to best candidate", e)

        # Fall back to best candidate
        best = max(candidates, key=lambda c: c.scores.composite)
        scores = _score_final(best.text, sketch, scaffold)
        return FinalResponseRecord(
            response_id=rid,
            text=best.text,
            source_candidate_ids=[best.candidate_id],
            source_fragment_ids=[f.fragment_id for f in kept],
            citations=_extract_citations(best.text, bundle),
            scores=scores,
            enforcement_passed=_check_citation_gate(best.text, bundle),
            token_count=best.token_estimate,
        )

    def _knowledge_only_response(
        self,
        turn_id: str,
        sketch: InputSketch,
        bundle: MemoryActivationBundle,
    ) -> FinalResponseRecord:
        """
        Build a response from retrieved knowledge records alone (no model).
        Used when no inference model is available.
        """
        rid = f"resp_{uuid.uuid4().hex[:8]}"

        if not bundle.activated_records:
            text = "The forge has no knowledge relevant to this query."
            scores = FinalResponseScores(
                relevance=0.1, clarity=0.8, coherence=0.8,
                personality_fit=0.5, usefulness=0.1, goal_fit=0.1,
            )
            return FinalResponseRecord(
                response_id=rid,
                text=text,
                scores=scores,
                enforcement_passed=False,
                enforcement_notes="no knowledge available",
                token_count=len(text.split()),
            )

        # Assemble knowledge summary from top records
        lines: list[str] = []
        for rec in bundle.activated_records[:5]:
            if rec.cue:
                lines.append(f"• {rec.cue}")

        text = (
            f"Retrieved knowledge relevant to '{sketch.topic}':\n"
            + "\n".join(lines)
        )
        scores = FinalResponseScores(
            relevance=0.6,
            clarity=0.7,
            coherence=0.7,
            personality_fit=0.4,
            usefulness=0.5,
            goal_fit=0.4,
        )
        return FinalResponseRecord(
            response_id=rid,
            text=text,
            citations=[r.record_id for r in bundle.activated_records[:5]],
            scores=scores,
            enforcement_passed=True,
            enforcement_notes="knowledge-only mode",
            token_count=len(text.split()),
        )

    def _repair(
        self,
        turn_id: str,
        sketch: InputSketch,
        scaffold: CognitionScaffold,
        bundle: MemoryActivationBundle,
        previous: FinalResponseRecord,
    ) -> FinalResponseRecord:
        """Repair pass — tighter prompt, lower temperature."""
        if self._engine is None:
            return previous

        memory_cues = self._prompt_builder.extract_memory_cues(bundle, max_cues=2)
        repair_prompt = self._prompt_builder.build_repair_prompt(
            sketch=sketch,
            scaffold=scaffold,
            memory_cues=memory_cues,
        )
        try:
            from thoughtforge.inference.turboquant import GenerationParams
            result = self._engine.generate(
                repair_prompt,
                params=GenerationParams(temperature=0.45),
            )
            scores = _score_final(result.text, sketch, scaffold)
            rid = f"resp_{uuid.uuid4().hex[:8]}"
            return FinalResponseRecord(
                response_id=rid,
                text=result.text,
                citations=_extract_citations(result.text, bundle),
                scores=scores,
                enforcement_passed=_check_citation_gate(result.text, bundle),
                enforcement_notes="repair pass",
                token_count=result.tokens_generated,
            )
        except Exception as e:
            logger.warning("Repair pass failed: %s — returning previous result", e)
            return previous


# ── Scoring helpers ────────────────────────────────────────────────────────────

_GENERIC_PHRASES = [
    "of course", "certainly", "absolutely", "sure thing", "great question",
    "as an ai", "i'm an ai", "i cannot", "i don't have", "i am unable",
    "i'd be happy to", "i hope this helps",
]

_SPECIFICITY_MARKERS = re.compile(
    r'\b(\d+[\d.,]*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}|'
    r'https?://|Q\d{3,}|v\d+\.\d+)\b'
)


def _genericness_penalty(text: str) -> float:
    lower = text.lower()
    hits = sum(1 for p in _GENERIC_PHRASES if p in lower)
    return min(hits * 0.2, 1.0)


def _specificity_score(text: str) -> float:
    hits = len(_SPECIFICITY_MARKERS.findall(text))
    words = max(len(text.split()), 1)
    return min(hits / (words * 0.1), 1.0)


def _keyword_overlap(text: str, query: str) -> float:
    stopwords = {"the", "a", "an", "is", "are", "was", "to", "of", "and", "in", "i"}
    q_words = {w.lower() for w in re.findall(r'\w+', query) if w.lower() not in stopwords}
    t_words = {w.lower() for w in re.findall(r'\w+', text)}
    if not q_words:
        return 0.5
    overlap = len(q_words & t_words) / len(q_words)
    return min(overlap, 1.0)


def _length_score(text: str, min_words: int = 8, ideal_words: int = 60) -> float:
    words = len(text.split())
    if words < min_words:
        return 0.1
    if words <= ideal_words:
        return 0.5 + 0.5 * (words / ideal_words)
    return max(0.5, 1.0 - (words - ideal_words) / (ideal_words * 3))


def _score_candidate(text: str, sketch: InputSketch, scaffold: CognitionScaffold) -> CandidateScores:
    relevance = _keyword_overlap(text, sketch.raw_text)
    clarity = _length_score(text)
    coherence = 0.6 if len(text.split(".")) > 1 else 0.4
    personality_fit = max(0.0, 0.7 - _genericness_penalty(text))
    specificity = _specificity_score(text)
    genericness = _genericness_penalty(text)
    goal_fit = relevance * 0.6 + specificity * 0.4

    return CandidateScores(
        relevance=relevance,
        clarity=clarity,
        coherence=coherence,
        personality_fit=personality_fit,
        specificity=specificity,
        genericness_penalty=genericness,
        goal_fit=goal_fit,
    )


def _score_fragment(text: str, sketch: InputSketch, scaffold: CognitionScaffold) -> FragmentScores:
    relevance = _keyword_overlap(text, sketch.raw_text)
    words = len(text.split())
    clarity = min(1.0, words / 12.0) if words <= 25 else max(0.4, 1.0 - (words - 25) / 50)
    specificity = _specificity_score(text)
    usefulness = relevance * 0.5 + specificity * 0.5
    personality_fit = max(0.0, 0.75 - _genericness_penalty(text))
    genericness = _genericness_penalty(text)

    return FragmentScores(
        relevance=relevance,
        clarity=clarity,
        specificity=specificity,
        usefulness=usefulness,
        personality_fit=personality_fit,
        genericness_penalty=genericness,
    )


def _score_final(text: str, sketch: InputSketch, scaffold: CognitionScaffold) -> FinalResponseScores:
    relevance = _keyword_overlap(text, sketch.raw_text)
    words = len(text.split())
    clarity = _length_score(text, min_words=_MIN_RESPONSE_TOKENS)
    coherence = min(1.0, len(re.split(r'[.!?]', text)) / 3.0)
    personality_fit = max(0.0, 0.8 - _genericness_penalty(text))
    usefulness = relevance * 0.4 + _specificity_score(text) * 0.6
    goal_fit = _keyword_overlap(text, scaffold.goal)
    genericness = _genericness_penalty(text)

    return FinalResponseScores(
        relevance=relevance,
        clarity=clarity,
        coherence=coherence,
        personality_fit=personality_fit,
        usefulness=usefulness,
        goal_fit=goal_fit,
        genericness_penalty=genericness,
    )


# ── Citation helpers ───────────────────────────────────────────────────────────

_QID_RE = re.compile(r'\bQ\d{3,}\b')


def _extract_citations(text: str, bundle: MemoryActivationBundle) -> list[str]:
    """Extract QID citations mentioned in the text."""
    qids_in_text = set(_QID_RE.findall(text))
    valid_qids = {
        r.raw.get("qid", "")
        for r in bundle.activated_records
        if r.raw.get("qid")
    }
    cited = list(qids_in_text & valid_qids)
    return cited


def _check_citation_gate(text: str, bundle: MemoryActivationBundle) -> bool:
    """
    Minimal citation enforcement: pass if KB was empty (nothing to cite)
    or if at least one QID from the retrieved bundle appears in the text.
    """
    if not bundle.activated_records:
        return True
    valid_qids = {
        r.raw.get("qid", "")
        for r in bundle.activated_records
        if r.raw.get("qid")
    }
    if not valid_qids:
        return True     # No Wikidata records — citation gate passes
    return bool(set(_QID_RE.findall(text)) & valid_qids)


# ── Text helpers ───────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for fragment extraction."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _mode_temperature(mode: str) -> float:
    return {
        "strict_spec": 0.50,
        "implementation_friendly": 0.55,
        "editorial_sharp": 0.50,
        "brainstorm_distinct": 0.80,
        "empathic": 0.65,
        "reflective": 0.70,
        "practical": 0.55,
        "playful": 0.75,
    }.get(mode, 0.60)


# ── Utility ────────────────────────────────────────────────────────────────────

def _new_turn_id() -> str:
    return f"turn_{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _load_personality_from_config(config: dict[str, Any]) -> Any:
    """Try to load personality_core.yaml from the path in config."""
    try:
        import yaml
        core_path_str = (
            config.get("cognition", {}).get("personality_core_path", "")
        )
        if not core_path_str:
            return None
        core_path = Path(core_path_str)
        if not core_path.is_absolute():
            # Resolve relative to project root (parent of src/)
            from thoughtforge.utils.paths import get_project_root
            core_path = get_project_root() / core_path
        if not core_path.exists():
            return None
        from thoughtforge.knowledge.models import PersonalityCoreRecord
        with core_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data:
            return PersonalityCoreRecord.from_dict(data)
    except Exception as e:
        logger.debug("Could not load personality_core from config path: %s", e)
    return None
