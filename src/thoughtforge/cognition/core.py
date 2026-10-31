"""
ThoughtForgeCore — the main memory-enforced cognition pipeline.

Orchestrates the one turn loop (the reconciled map is
docs/architecture/CORE_LOOP.md, TF-019 as reconciled by TF-039):

  1. ROUTE      InputRouter → InputSketch
  2. RETRIEVE   KnowledgeForge → MemoryActivationBundle
  3. SCAFFOLD   ScaffoldBuilder → CognitionScaffold (the caller's
                history, the will layer's standing goal, and the
                persona block are folded into its fact_block)
  4. GENERATE   local engine | unified backend | none →
                CandidateRecords (PromptBuilder builds every
                prompt inside this step and step 7; generation
                failures are recorded on the turn, never silently
                swallowed — TF-031)
  5. FRAGMENT   per-sentence scoring; fragments kept at ≥ 0.25 on
                the turn record (the salvage engine's own 0.54
                threshold inside step 6 is a different instrument)
  6. COMPOSE    FragmentSalvage → best text, then EnforcementGate
                inside the same step → FinalResponseRecord — or,
                with no candidates, the knowledge-only / stated-
                failure record
  7. QUALITY GATE  composite < 0.40 and candidates exist → one
                repair generation, kept only if it scores higher

think(), think_stream(), and resume_turn() are three drivers of
this one pipeline (_run_turn); the finished turn's state is
retained on last_turn_state and exportable via last_turn_trace().

Works without a model (knowledge-only mode) and without a knowledge DB
(model-only mode). Degrades gracefully in both cases.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from thoughtforge.cognition.error_honesty import (
    GenerationFailure,
    apply_partial_disclosure,
    degraded_knowledge_text,
    failure_text,
    generation_failures,
    turn_failures,
)
from thoughtforge.cognition.config_surface import (
    ConversationConfig,
    apply_conversation_config,
    resolve_conversation_config,
)
from thoughtforge.cognition.determinism import (
    DeterminismMode,
    resolve_determinism_mode,
)
from thoughtforge.cognition.history_discipline import CORE_HISTORY_MAX_CHARS
from thoughtforge.cognition.interruption import (
    InterruptToken,
    TurnInterrupted,
    check_interrupt,
    mark_interrupted,
)
from thoughtforge.cognition.persona import PersonaExpectation, inject_persona
from thoughtforge.cognition.prompt_builder import PromptBuilder
from thoughtforge.cognition.streaming import StreamEvent, chunk_words
from thoughtforge.cognition.turn_state import advance_phase, phase_index
from thoughtforge.cognition.will import GoalLedger, inject_will
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
from thoughtforge.knowledge.lifecycle import MemoryLifecycle
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.refinement.enforcement import EnforcementGate
from thoughtforge.refinement.salvage import FragmentSalvage
from thoughtforge.utils.config import load_config
from thoughtforge.utils.paths import get_knowledge_db_path, get_memory_dir
from thoughtforge.utils.perf import get_perf_tracker
from thoughtforge.utils.validators import sanitise_query

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_QUALITY_REPAIR_THRESHOLD = 0.40     # below this composite score → trigger repair
# TF-022 ruling on CORE_LOOP §8-F6: this threshold gates the core's own
# FragmentScores composite (relevance/clarity/specificity/usefulness/
# personality, knowledge/models.py) and decides only which fragments
# are *kept on the turn record* for provenance. It is NOT the salvage
# engine's threshold: FragmentSalvage._MIN_FRAGMENT_SCORE (0.54) gates
# a different formula (length+citation, from the scoring spec) and
# decides which fragments are *worth a refine pass*. The numbers
# differ because the metrics differ — do not "unify" them.
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
        backend: Any = None,
    ) -> None:
        self._config = config or load_config()
        # TF-032: the cognition config surface — this conversation's
        # selections resolved from the config dict (live keys only;
        # cognition/config_surface.py). An invalid selection raises
        # ValueError here, at construction, never a silent default.
        self._conversation_config = resolve_conversation_config(self._config)
        # TF-036: determinism mode — this conversation's seed, if
        # it selected one (cognition/determinism.py). An invalid
        # seed raises ValueError here, at construction. The two
        # seeded streams (record ids, generation seeds) are
        # dedicated random.Random instances, never the global
        # module; both stay None when the mode is off, and every
        # consumer below then takes the pre-TF-036 path exactly.
        self._determinism: DeterminismMode = resolve_determinism_mode(
            self._config
        )
        self._det_ids = (
            self._determinism.new_id_stream()
            if self._determinism.enabled
            else None
        )
        self._det_gen_rng = (
            self._determinism.new_generation_rng()
            if self._determinism.enabled
            else None
        )
        self._model_path = Path(model_path) if model_path else None
        self._unified_backend = backend   # UnifiedBackend instance or None

        self._knowledge = KnowledgeForge(
            db_path=db_path or get_knowledge_db_path(),
        )
        self._store = MemoryStore(memory_dir=memory_dir or get_memory_dir())
        # TF-045: the memory lifecycle, wired into the live engine
        # over this core's own store (was test-only — TF-011's
        # D-ORPHAN-1: no fact ever aged at runtime). Two surfaces:
        # the active thread state is cooled once per completed turn
        # in _finish_turn, and prune_memory() below is the explicit
        # operator surface for a full prune pass. Fact pruning is
        # NEVER automatic per turn — retirement of facts is an
        # operator's act, not a side effect of chatting.
        self._lifecycle = MemoryLifecycle(store=self._store)
        self._router = InputRouter()
        self._scaffold_builder = ScaffoldBuilder()
        self._prompt_builder = PromptBuilder()
        self._salvage = FragmentSalvage()
        self._enforcement = EnforcementGate()

        self._engine: Any = None      # TurboQuantEngine — loaded lazily
        # TF-035: one core serves ONE conversation at a time. This
        # lock serialises whole turns on this instance (think /
        # think_stream / resume_turn): before it, two threads
        # calling think() on one core raced on _last_turn_state
        # and BOTH callers received whichever turn assigned it
        # last — the other conversation's answer (measured in the
        # TF-035 probe, 3/3 trials). Two conversations in parallel
        # use two cores (cognition/concurrency.py); turns on one
        # core queue instead of bleeding. The thread-local below
        # is the lock's other half: think() reads back the turn
        # ITS thread ran, so the read-back after the turn cannot
        # be crossed by a queued turn starting on another thread.
        self._turn_lock = threading.Lock()
        self._turn_local = threading.local()
        # TF-025: the most recent turn's full typed state, retained
        # after think() returns (was built and discarded — CORE_LOOP
        # §8-F3). Trace *export* of it is TF-038's slice.
        self._last_turn_state: RuntimeTurnState | None = None
        # TF-029: the will layer — one goal ledger per core instance
        # (a conversation's standing goal, persisted across turns;
        # cognition/will.py). Observed after each turn's scaffold
        # is built; injects into that scaffold from the next turn on.
        self._will = GoalLedger()
        self._perf = get_perf_tracker()
        self._personality = self._store.load_personality_core()

        # Try to load personality from config path if not in memory dir
        if self._personality is None:
            self._personality = _load_personality_from_config(self._config)

        # TF-030: the persona expectation — a construction-time
        # snapshot of the personality in force (cognition/persona.py).
        # Its steering block is injected into every turn's scaffold;
        # it is also the object persona-consistency consumers score
        # against. None when no personality is loaded.
        self._persona_expectation = PersonaExpectation.from_record(
            self._personality
        )

        # Self-heal any detectable issues on startup
        try:
            from thoughtforge.utils.self_heal import SelfHealer
            SelfHealer().heal_all()
        except Exception as _heal_exc:
            logger.debug("SelfHealer startup pass failed (non-critical): %s", _heal_exc)

        logger.info(
            "ThoughtForgeCore ready | model=%s | personality=%s | db=%s",
            self._model_path or "none (knowledge-only mode)",
            self._personality.record_id if self._personality else "none",
            db_path or get_knowledge_db_path(),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def last_turn_state(self) -> RuntimeTurnState | None:
        """TF-025: the last completed turn's typed state (None before
        the first think()). The same object think() built — retained,
        not copied; callers must treat it as read-only."""
        return self._last_turn_state

    @property
    def conversation_config(self) -> ConversationConfig:
        """TF-032: this conversation's resolved cognition selections
        (profile / strategy / depth / candidate modes). Frozen at
        construction from the config dict."""
        return self._conversation_config

    @property
    def selected_strategy(self) -> Any:
        """TF-032: the TF-022 strategy this conversation selected,
        or None. Selection only — the core loop does not execute
        strategies (TF-022's ruling stands)."""
        if self._conversation_config.strategy is None:
            return None
        from thoughtforge.cognition.scaffold_strategies import (
            strategy_registry,
        )

        return strategy_registry()[self._conversation_config.strategy]

    @property
    def latency_budget(self) -> Any:
        """TF-033: this conversation's core-loop latency budget —
        the selected profile's (TF-032), or the VM profile's
        (desktop_cpu) when no profile was selected, so a verdict
        always has a stated budget to answer to. Resolution is
        through cognition/latency_budgets.py; an invalid selected
        profile already raised at construction."""
        from thoughtforge.cognition.latency_budgets import (
            VM_PROFILE_ID,
            budget_for_profile,
        )

        return budget_for_profile(
            self._conversation_config.profile_id or VM_PROFILE_ID
        )

    def last_turn_latency(self, total_ms: float | None = None) -> Any:
        """TF-033: the last finished turn's latency verdict against
        this conversation's budget (None before the first turn).
        ``total_ms`` is the caller's wall-clock measurement around
        the turn; when given, the verdict also judges the total."""
        if self._last_turn_state is None:
            return None
        from thoughtforge.cognition.latency_budgets import (
            check_turn_latency,
        )

        return check_turn_latency(
            self._last_turn_state, self.latency_budget, total_ms=total_ms
        )

    def last_turn_trace(self) -> dict[str, Any] | None:
        """TF-038: the last turn's full cognition trace, exported
        as detached JSON-safe data (cognition/traces.py) — route,
        retrieval, scaffold, candidates, fragments, failures,
        final, timings — with this conversation's context (selected
        profile / strategy / depth / modes, determinism seed,
        persona id) in its conversation section. None before the
        first turn. A MINOR addition under TF-034's rules — the
        frozen v1 tables are untouched; TF-039/TF-192 record the
        addition when the freeze is next reconciled."""
        if self._last_turn_state is None:
            return None
        from thoughtforge.cognition.traces import export_turn_trace

        conversation = {
            "profile_id": self._conversation_config.profile_id,
            "strategy": self._conversation_config.strategy,
            "depth": self._conversation_config.depth,
            "candidate_modes": (
                list(self._conversation_config.candidate_modes)
                if self._conversation_config.candidate_modes is not None
                else None
            ),
            "determinism_seed": self._determinism.seed,
            "persona_id": (
                self._personality.record_id if self._personality else None
            ),
        }
        return export_turn_trace(
            self._last_turn_state, conversation=conversation
        )

    def prune_memory(self, mode: str = "routine") -> dict[str, int]:
        """TF-045: run one memory-lifecycle prune pass over this
        core's store and return its counts ({archived, deleted,
        deduplicated}) — the live engine's surface for the lifecycle
        that was test-only until this slice (D-ORPHAN-1). ``mode``
        is one of light / routine / heavy / emergency (the
        lifecycle's own threshold tables rule what each retires;
        knowledge/lifecycle_rules.py states their edges). Pruning
        is explicit: the core never prunes facts on its own — the
        only automatic lifecycle act is per-turn thread cooling in
        _finish_turn, and the thread state is ephemeral by design.
        A MINOR addition under TF-034's rules — the frozen v1
        tables are untouched; TF-039's additions record (extended
        by TF-045) / TF-192 carry it into the next reconciliation.
        """
        return self._lifecycle.prune(mode)  # type: ignore[arg-type]

    @property
    def determinism_mode(self) -> DeterminismMode:
        """TF-036: this conversation's determinism selection
        (seed, or off). A MINOR addition under TF-034's rules —
        the frozen v1 tables are untouched; TF-039/TF-192 record
        the addition when the freeze is next reconciled."""
        return self._determinism

    def _mint_id(self, prefix: str, hex_len: int) -> str:
        """TF-036: mint one record id. In determinism mode the id
        comes from this core's seeded stream (same seed + same
        call order = same ids, across processes); otherwise the
        standing uuid4 shape, unchanged."""
        if self._det_ids is not None:
            return self._det_ids.new_id(prefix, hex_len)
        return f"{prefix}_{uuid.uuid4().hex[:hex_len]}"

    def _next_generation_seed(self) -> int | None:
        """TF-036: the next per-call generation seed from this
        core's seeded generation stream, or None when determinism
        mode is off (generation then carries no seed at all, as
        before this slice). One draw per generation call, in
        call order, on every generation path."""
        if self._det_gen_rng is None:
            return None
        return self._det_gen_rng.getrandbits(31)

    @property
    def persona_expectation(self) -> PersonaExpectation | None:
        """TF-030: the persona in force for this core (None when no
        personality is loaded). The live snapshot — callers must
        treat it as read-only."""
        return self._persona_expectation

    @property
    def will_ledger(self) -> GoalLedger:
        """TF-029: this conversation's goal ledger (the will layer).
        The live object — callers must treat it as read-only."""
        return self._will

    def think(
        self,
        user_text: str,
        retrieval_path: str | None = None,
        num_drafts: int | None = None,
        history: "Any | None" = None,
        interrupt: InterruptToken | None = None,
    ) -> FinalResponseRecord:
        """
        Full memory-enforced cognition pipeline for one user turn.

        Args:
            user_text:      Raw user message.
            retrieval_path: Override retrieval path ("sql" | "vector" | "hybrid").
            num_drafts:     TF-034 (freezing CORE_LOOP §8-F1): cap on
                            the candidate drafts generated this turn.
                            The local engine drafts once per candidate
                            mode (at most 2 by construction), so the
                            cap selects the first num_drafts modes;
                            the unified backend drafts exactly once
                            by construction, so the cap cannot raise
                            it. None = no cap beyond construction.
                            Values below 1 raise ValueError.
            interrupt:      TF-027: an InterruptToken that can cut
                            the turn at a cut point (see
                            cognition/interruption.py).

        Returns:
            FinalResponseRecord — always returns, never raises on content failure.

        Raises:
            TurnInterrupted: the interrupt token fired; the turn
                stands cut and coherent on last_turn_state and can
                be finished with resume_turn(). A cut is NOT a
                content failure, so it is not swallowed into a
                fabricated answer.
        """
        # TF-035: the turn lock is held across the whole turn
        # AND the read-back below — a second turn attempted on
        # this core waits here instead of racing (see __init__).
        with self._turn_lock:
            for _ in self._run_turn(
                user_text, retrieval_path, num_drafts, history, stream=False, interrupt=interrupt,
            ):
                pass
            # TF-035: read back THIS thread's turn (set in
            # _run_turn), not only the shared retained state —
            # belt and braces with the lock above.
            turn = getattr(self._turn_local, "turn", None) or self._last_turn_state
            assert turn is not None and turn.final_response is not None
            return turn.final_response

    def think_stream(
        self,
        user_text: str,
        retrieval_path: str | None = None,
        num_drafts: int | None = None,
        history: "Any | None" = None,
        interrupt: InterruptToken | None = None,
    ) -> Iterator[StreamEvent]:
        """
        TF-026: one full turn, delivered as a stream of events.

        Same pipeline as think() — routing, retrieval and scaffold
        all run pre-stream, on the same prompts; only candidate
        generation streams (chunk events, ``native`` marking whether
        the source itself streamed); salvage + enforcement run on
        completed text and their answer arrives as the single
        terminal ``final`` event (see cognition/streaming.py for
        the honesty contract). The finished turn is retained on
        last_turn_state exactly as think() retains it.

        TF-027: pass ``interrupt`` to allow the turn to be cut —
        the stream then ends with a terminal ``interrupted`` event
        instead of ``final``, and the turn can be finished with
        resume_turn(). Closing this stream mid-turn also cuts the
        turn (recorded on the retained state).

        TF-035: the turn lock is held from the first event until
        this stream is exhausted or closed (closing releases it);
        a second turn on this same core waits for the stream to
        finish rather than racing it.
        """
        with self._turn_lock:
            yield from self._run_turn(
                user_text, retrieval_path, num_drafts, history, stream=True, interrupt=interrupt,
            )

    def resume_turn(
        self,
        interrupt: InterruptToken | None = None,
    ) -> FinalResponseRecord:
        """TF-035: serialised entry point — a resume is a turn on
        this core, so it takes the same turn lock as think() (see
        __init__). The work itself is _resume_turn_inner (TF-027)."""
        with self._turn_lock:
            return self._resume_turn_inner(interrupt=interrupt)

    def _resume_turn_inner(
        self,
        interrupt: InterruptToken | None = None,
    ) -> FinalResponseRecord:
        """
        TF-027: finish the retained cut turn — the SAME turn.

        The interrupted turn on last_turn_state is taken up from
        its phase: retained sketch / bundle / scaffold are reused
        (no re-routing, no re-retrieval), generation runs only for
        the scaffold modes that produced no candidate before the
        cut, fragment scoring and the composed final are built
        from the full candidate set, and the turn completes
        through the normal quality gate with its original turn_id.

        Raises:
            ValueError: there is no interrupted turn to resume, or
                the turn was cut before its scaffold existed (only
                possible when a stream consumer closed before
                generation started) — start a new turn instead.
            TurnInterrupted: the new interrupt token fired during
                the resume; the turn stands cut again, coherently.
        """
        turn = self._last_turn_state
        if turn is None or not turn.interrupted:
            raise ValueError("no interrupted turn to resume")
        if phase_index(turn.phase) < phase_index("scaffolded"):
            raise ValueError(
                "turn was cut before its scaffold existed — nothing to "
                "resume from; start a new turn"
            )
        assert turn.input_sketch is not None and turn.scaffold is not None
        assert turn.memory_bundle is not None
        turn.resume_count += 1
        turn.interrupted = False
        turn.interrupt_reason = ""
        turn.interrupted_phase = ""

        t_start = time.perf_counter()
        try:
            check_interrupt(turn, interrupt)
            if phase_index(turn.phase) < phase_index("generated"):
                t_gen = time.perf_counter()
                try:
                    # Only the modes that produced no candidate before
                    # the cut are generated; modes already on the turn
                    # are kept, never regenerated (_generate_candidates
                    # routes engine / unified backend as in think()).
                    done_modes = {c.mode for c in turn.candidates}
                    missing = [
                        m for m in turn.scaffold.candidate_modes[:2]
                        if m not in done_modes
                    ]
                    self._generate_candidates(
                        turn.input_sketch, turn.scaffold, turn.memory_bundle,
                        None, interrupt=interrupt, sink=turn.candidates,
                        only_modes=missing, turn=turn,
                    )
                finally:
                    turn.generation_ms = int((time.perf_counter() - t_gen) * 1000)
                turn.fragments = self._score_and_extract_fragments(
                    turn.candidates, turn.input_sketch, turn.scaffold,
                )
                advance_phase(turn, "generated")
            check_interrupt(turn, interrupt)
            final = self._finish_turn(turn)
        except TurnInterrupted as exc:
            mark_interrupted(turn, exc.reason)
            raise
        self._perf.record(
            "think.resume", (time.perf_counter() - t_start) * 1000,
        )
        return final

    def _run_turn(
        self,
        user_text: str,
        retrieval_path: str | None,
        num_drafts: int | None,
        history: "Any | None",
        *,
        stream: bool,
        interrupt: InterruptToken | None = None,
    ) -> Iterator[StreamEvent]:
        """The one turn pipeline; think() exhausts it silently,
        think_stream() yields its events (stream=True only).

        TF-027: ``interrupt`` is checked at the cut points named in
        cognition/interruption.py. A cut raises TurnInterrupted out
        of think(); a streamed turn instead yields one terminal
        ``interrupted`` event. A consumer closing the stream
        mid-turn (GeneratorExit) records the same cut and the
        exception propagates untouched.

        TF-035: callers (think / think_stream / resume_turn) hold
        this core's turn lock for the whole turn; this body never
        takes it itself, so there is no re-entry to deadlock.
        """
        # TF-034: num_drafts is a real cap now (it was documented
        # but dead — CORE_LOOP §8-F1). Below 1 is a caller error,
        # stated loudly, never a silent knowledge-only turn.
        if num_drafts is not None and num_drafts < 1:
            raise ValueError(
                f"num_drafts must be at least 1 or None (no cap); "
                f"got {num_drafts!r}"
            )
        user_text = sanitise_query(user_text)
        turn_id = self._mint_id("turn", 8)
        turn = RuntimeTurnState(turn_id=turn_id)
        self._last_turn_state = turn
        # TF-035: this thread's own turn, for think()'s read-back.
        self._turn_local.turn = turn
        t_start = time.perf_counter()
        try:
            if stream:
                yield StreamEvent(kind="turn_started", turn_id=turn_id)

            logger.info("think() turn=%s text=%r", turn_id, user_text[:80])

            # ── Step 1: Route input ────────────────────────────────────────────
            thread_state = self._store.load_thread_state()
            sketch = self._router.route(user_text, thread_state=thread_state)
            if retrieval_path:
                sketch.retrieval_path = retrieval_path
            turn.input_sketch = sketch
            advance_phase(turn, "routed")

            # ── Step 2: Retrieve knowledge ─────────────────────────────────────
            t_ret = time.perf_counter()
            bundle = self._retrieve(sketch)
            turn.memory_bundle = bundle
            turn.retrieval_ms = int((time.perf_counter() - t_ret) * 1000)
            advance_phase(turn, "retrieved")
            logger.debug("Retrieved %d records in %dms", bundle.total_retrieved, turn.retrieval_ms)

            # ── Step 3: Build scaffold ─────────────────────────────────────────
            scaffold = self._scaffold_builder.build(sketch, bundle, self._personality)
            # TF-032: this conversation's scaffold selections (depth /
            # candidate modes, when configured) replace the builder's
            # derived values. Unselected = the builder's scaffold
            # stands byte-identical. Applied once here; a resumed turn
            # reuses its retained scaffold and is not re-applied.
            apply_conversation_config(scaffold, self._conversation_config)
            # Inject chat history context into the fact_block when provided
            if history is not None:
                history_context = history.to_prompt_context(max_chars=CORE_HISTORY_MAX_CHARS)
                if history_context:
                    if scaffold.fact_block:
                        scaffold.fact_block = (
                            f"Conversation history:\n{history_context}\n\n"
                            f"Knowledge:\n{scaffold.fact_block}"
                        )
                    else:
                        scaffold.fact_block = f"Conversation history:\n{history_context}"
            # ── TF-030: the persona steering block (traits / speech
            # style / behavior rules from the loaded personality)
            # prepends to the fact_block — the same injection
            # channel as history above. Persona steers every turn,
            # including the first: identity is not persistence.
            inject_persona(scaffold, self._persona_expectation)
            # ── TF-029: the will layer observes the turn, then the
            # standing goal (if any, and not set this very turn)
            # prepends its one line to the fact_block — the same
            # injection channel history uses above. Observation is
            # before the interrupt cut points on purpose: a cut
            # turn still stated its goal.
            self._will.observe_turn(sketch, scaffold.goal)
            inject_will(scaffold, self._will)
            turn.scaffold = scaffold
            advance_phase(turn, "scaffolded")

            # ── TF-027 cut point: after scaffold, before generation ──────────
            # A token fired during routing / retrieval / scaffold is
            # honoured here — stages are atomic between cut points.
            check_interrupt(turn, interrupt)

            # ── Step 4–6: Generate + score fragments ───────────────────────────
            # The sink IS turn.candidates from the start: a cut inside
            # generation leaves exactly the completed candidates on
            # the turn; a candidate in flight is never appended.
            t_gen = time.perf_counter()
            try:
                if stream:
                    yield from self._generate_candidates_streaming(
                        turn, sketch, scaffold, bundle, num_drafts,
                        interrupt=interrupt, sink=turn.candidates,
                    )
                else:
                    self._generate_candidates(
                        sketch, scaffold, bundle, num_drafts,
                        interrupt=interrupt, sink=turn.candidates, turn=turn,
                    )
            finally:
                # TF-027: the partial leg's time is measured too —
                # a cut turn's generation_ms is real, not a default.
                turn.generation_ms = int((time.perf_counter() - t_gen) * 1000)
            candidates = turn.candidates

            fragments = self._score_and_extract_fragments(candidates, sketch, scaffold)
            turn.fragments = fragments
            advance_phase(turn, "generated")

            # ── TF-027 cut point: after generation, before compose ───────────
            check_interrupt(turn, interrupt)

            # ── Steps 7–8: Compose + quality gate + completion ────────────────
            final = self._finish_turn(turn)

            total_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "think() done turn=%s | score=%.3f tier=%s | tokens=%d | %dms total",
                turn_id,
                final.scores.composite,
                final.scores.quality_tier,
                final.token_count,
                int(total_ms),
            )

            # Record performance metrics
            self._perf.record("think.total", total_ms)
            self._perf.record("think.retrieval", turn.retrieval_ms)
            self._perf.record("think.generation", turn.generation_ms)
            self._perf.record("think.salvage", turn.salvage_ms)

            if stream:
                yield StreamEvent(kind="final", turn_id=turn_id, text=final.text, final=final)
        except TurnInterrupted as exc:
            mark_interrupted(turn, exc.reason)
            logger.info(
                "think() turn=%s interrupted at phase=%s: %s",
                turn_id, turn.phase, exc.reason or "no reason given",
            )
            if stream:
                yield StreamEvent(
                    kind="interrupted", turn_id=turn_id, text=exc.reason,
                )
                return
            raise
        except GeneratorExit:
            # TF-027: the stream consumer cut the turn by closing
            # it. Record the cut on the retained state, then let
            # GeneratorExit propagate untouched — a closing
            # generator must never yield again. A consumer closing
            # AFTER the final event changes nothing: that turn
            # already completed, and completion is not rewritten.
            if not turn.is_complete:
                mark_interrupted(turn, "consumer closed the stream")
            raise

    def _finish_turn(self, turn: RuntimeTurnState) -> FinalResponseRecord:
        """Steps 7–8 of the turn pipeline, shared by _run_turn and
        TF-027's resume_turn: compose (or the knowledge-only
        fallback), the quality-gate repair branch, and completion
        bookkeeping. Expects turn.phase == "generated" (candidate
        and fragment stages finished) and fills final_response.
        """
        sketch = turn.input_sketch
        bundle = turn.memory_bundle
        candidates = turn.candidates
        assert sketch is not None and bundle is not None

        # TF-025: salvage_ms is MEASURED around the compose step
        # (salvage + enforcement). On the knowledge-only path no
        # salvage runs, so it stays exactly 0 — "did not run", never
        # "unmeasured" (closes CORE_LOOP §8-F3's timing half).
        if candidates:
            t_salvage = time.perf_counter()
            final = self._compose_final(turn)
            turn.salvage_ms = int((time.perf_counter() - t_salvage) * 1000)
        else:
            final = self._knowledge_only_response(turn.turn_id, sketch, bundle)
        advance_phase(turn, "composed")

        # Quality gate → repair if needed.
        if final.scores.composite < _QUALITY_REPAIR_THRESHOLD and candidates:
            logger.debug(
                "Quality below threshold (%.3f < %.3f) — attempting repair",
                final.scores.composite, _QUALITY_REPAIR_THRESHOLD,
            )
            turn.repair_attempted = True
            repaired = self._repair(turn, final)
            if repaired.scores.composite > final.scores.composite:
                final = repaired

        # ── TF-031: error honesty — the final text must say what
        # actually happened when generation failed. Applied after
        # the repair decision, to whichever final stands, exactly
        # once. A deliberate no-model turn records no failures and
        # is untouched (byte-identical knowledge-only answers).
        visible_failures = generation_failures(turn_failures(turn))
        if visible_failures:
            clauses = "; ".join(f.describe() for f in visible_failures)
            if candidates:
                # Partial failure: the composed answer is real, and
                # carries its disclosure.
                apply_partial_disclosure(final, visible_failures)
            elif bundle.activated_records:
                # Total failure with knowledge to show: the records
                # are presented as retrieved knowledge under a
                # stated failure — never as a model answer.
                final.text = degraded_knowledge_text(visible_failures, final.text)
                final.token_count = len(final.text.split())
                final.enforcement_notes = (
                    f"{final.enforcement_notes}; model failure: {clauses}"
                    if final.enforcement_notes
                    else f"model failure: {clauses}"
                )
            else:
                # Total failure, nothing to show: a stated failure
                # replaces the empty-store text (which would blame
                # the knowledge for the model's failure).
                final.text = failure_text(visible_failures)
                final.token_count = len(final.text.split())
                final.enforcement_passed = False
                final.enforcement_notes = f"model failure: {clauses}"
                final.salvage_path = "model_failed"
                final.mode = "failed"

        turn.final_response = final
        turn.total_tokens_used = final.token_count
        turn.completed_at = _now_iso()
        advance_phase(turn, "completed")
        # TF-045: one turn of lifecycle cooling for the active
        # thread state — the runtime aging the lifecycle module was
        # built for (D-ORPHAN-1). cool_thread() is a no-op when no
        # thread state exists; when the cooled state goes stale it
        # is cleared by the lifecycle itself. A lifecycle failure
        # must never kill a completed turn, so it is logged at
        # warning (visible, never silent) and the final stands.
        try:
            self._lifecycle.cool_thread()
        except Exception as exc:  # noqa: BLE001 — turn must survive
            logger.warning("TF-045 thread cooling failed: %s", exc)
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

    def _record_failure(
        self,
        turn: RuntimeTurnState | None,
        stage: str,
        detail: str = "",
        *,
        exc: BaseException | None = None,
        mode: str = "",
    ) -> None:
        """TF-031: record one generation failure on the turn state.

        Every swallow point in candidate generation calls this, so
        a caught failure is data on the turn — never only a log
        line. ``turn`` may be None on legacy call paths that carry
        no turn state; recording is then skipped (the log line at
        the call site remains).
        """
        if turn is None:
            return
        if exc is not None:
            failure = GenerationFailure.from_exception(stage, exc, mode=mode)
        else:
            failure = GenerationFailure(stage=stage, mode=mode, detail=detail)
        turn.generation_failures.append(failure)

    def _retrieve(self, sketch: InputSketch) -> MemoryActivationBundle:
        """Retrieve knowledge records. Returns empty bundle if DB unavailable.

        TF-041 (CORE_LOOP §8-F2, closed): the existence gate reads
        the CONFIGURED db path — ``self._knowledge.db_path``, the
        path this core was built with — never the default path.
        Before this fix the gate tested ``get_knowledge_db_path()``,
        so a core built with a custom ``db_path`` was gated by a
        different file's existence entirely.
        """
        db = self._knowledge.db_path
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
        *,
        interrupt: InterruptToken | None = None,
        sink: list[CandidateRecord] | None = None,
        only_modes: list[str] | None = None,
        turn: RuntimeTurnState | None = None,
    ) -> list[CandidateRecord]:
        """Generate candidate responses via TurboQuantEngine or UnifiedBackend. Returns [] if no model.

        TF-027: completed candidates are appended to ``sink`` (the
        list returned either way), the interrupt token is checked
        before each mode (raising TurnInterrupted, which needs the
        turn via ``turn``), and ``only_modes`` restricts generation
        to the named modes — resume_turn's way of generating only
        what a cut turn is missing.
        """
        candidates: list[CandidateRecord] = sink if sink is not None else []

        def _check() -> None:
            if interrupt is not None:
                assert turn is not None
                check_interrupt(turn, interrupt)

        if only_modes is not None and not only_modes and candidates:
            return candidates   # resume: nothing missing

        if self._engine is None and self._model_path is not None:
            try:
                self.load_model()
            except Exception as e:
                logger.warning("Failed to load model: %s — running knowledge-only", e)
                self._record_failure(turn, "model_load", exc=e)
                return candidates

        # Route through UnifiedBackend when no local engine is loaded
        if self._engine is None and self._unified_backend is not None:
            _check()
            produced = self._generate_via_unified_backend(
                sketch, scaffold, bundle, only_modes=only_modes, turn=turn,
            )
            candidates.extend(produced)
            return candidates

        if self._engine is None:
            return candidates

        memory_cues = self._prompt_builder.extract_memory_cues(bundle)
        modes = scaffold.candidate_modes[:2]    # max 2 candidate passes
        if num_drafts is not None:
            # TF-034: the caller's draft cap selects the first
            # num_drafts modes (validated >= 1 in _run_turn).
            modes = modes[:num_drafts]
        if only_modes is not None:
            modes = [m for m in modes if m in only_modes]

        for mode in modes:
            # TF-027 cut point: before each candidate.
            _check()
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
                        # TF-036: one seeded draw per generation
                        # call in determinism mode; None otherwise.
                        seed=self._next_generation_seed(),
                    ),
                )
                if not result.text or not result.text.strip():
                    # TF-031: an empty completion is a failure, not
                    # a candidate — nothing was answered.
                    logger.warning("Candidate generation returned no text (mode=%s)", mode)
                    self._record_failure(
                        turn, "candidate",
                        "engine returned no text (empty generation)", mode=mode,
                    )
                    continue
                cid = self._mint_id("cand", 6)
                candidates.append(CandidateRecord(
                    candidate_id=cid,
                    mode=mode,
                    text=result.text,
                    token_estimate=result.tokens_generated,
                    scores=_score_candidate(result.text, sketch, scaffold),
                ))
            except TurnInterrupted:
                raise
            except Exception as e:
                logger.warning("Candidate generation failed (mode=%s): %s", mode, e)
                self._record_failure(turn, "candidate", exc=e, mode=mode)

        return candidates

    def _generate_via_unified_backend(
        self,
        sketch: InputSketch,
        scaffold: CognitionScaffold,
        bundle: MemoryActivationBundle,
        *,
        only_modes: list[str] | None = None,
        turn: RuntimeTurnState | None = None,
    ) -> list[CandidateRecord]:
        """Generate a single candidate via the UnifiedBackend (Ollama, LM Studio, HF, etc.).

        TF-031: every non-production is recorded on the turn — the
        backend's error channel, an exception, and an empty text
        alike. Before TF-031 all three silently became []."""
        from thoughtforge.inference.unified_backend import GenerationRequest

        memory_cues = self._prompt_builder.extract_memory_cues(bundle)
        mode = scaffold.candidate_modes[0] if scaffold.candidate_modes else "practical"
        if only_modes is not None and mode not in only_modes:
            return []   # TF-027 resume: this mode's candidate already exists
        prompt = self._prompt_builder.build_candidate_prompt(
            sketch=sketch,
            scaffold=scaffold,
            mode=mode,
            memory_cues=memory_cues,
        )
        try:
            req = GenerationRequest(
                prompt=prompt,
                temperature=_mode_temperature(mode),
                # TF-032: the selected profile's cap, or the standing
                # 512 when no profile was selected.
                max_tokens=self._conversation_config.generation_max_tokens(),
                # TF-036: the per-call seed (None when unseeded).
                seed=self._next_generation_seed(),
            )
            resp = self._unified_backend.generate(req)
            if resp.error:
                logger.warning("UnifiedBackend error: %s", resp.error)
                self._record_failure(turn, "backend", resp.error, mode=mode)
                return []
            if not resp.text or not resp.text.strip():
                logger.warning("UnifiedBackend returned no text (mode=%s)", mode)
                self._record_failure(
                    turn, "backend",
                    "backend returned no text (empty generation)", mode=mode,
                )
                return []
            cid = self._mint_id("cand", 6)
            return [CandidateRecord(
                candidate_id=cid,
                mode=mode,
                text=resp.text,
                token_estimate=resp.tokens_generated,
                scores=_score_candidate(resp.text, sketch, scaffold),
            )]
        except Exception as e:
            logger.warning("UnifiedBackend generate failed: %s", e)
            self._record_failure(turn, "backend", exc=e, mode=mode)
            return []

    # ── TF-026: streamed candidate generation ─────────────────────────────

    def _generate_candidates_streaming(
        self,
        turn: RuntimeTurnState,
        sketch: InputSketch,
        scaffold: CognitionScaffold,
        bundle: MemoryActivationBundle,
        num_drafts: int | None,
        *,
        interrupt: InterruptToken | None = None,
        sink: list[CandidateRecord] | None = None,
    ) -> Iterator[StreamEvent]:
        """Streamed twin of _generate_candidates: same prompts, same
        modes, same per-mode failure handling — plus events.

        Yields StreamEvents and returns the candidate list (as the
        yield-from value). Mirrors _generate_candidates' routing:
        local engine when loaded/loadable, else the unified backend,
        else no candidates (the turn degrades to knowledge-only).

        TF-027: completed candidates are appended to ``sink`` as
        they finish, and the interrupt token is checked before each
        candidate and before each chunk is delivered (a fired token
        raises TurnInterrupted — the in-flight candidate's parts
        are discarded whole, never appended).
        """
        turn_id = turn.turn_id
        candidates: list[CandidateRecord] = sink if sink is not None else []
        if self._engine is None and self._model_path is not None:
            try:
                self.load_model()
            except Exception as e:
                logger.warning("Failed to load model: %s — running knowledge-only", e)
                self._record_failure(turn, "model_load", exc=e)
                return candidates

        if self._engine is None and self._unified_backend is not None:
            yield from self._stream_via_unified_backend(
                turn, sketch, scaffold, bundle, interrupt=interrupt, sink=candidates,
            )
            return candidates

        if self._engine is None:
            return candidates

        memory_cues = self._prompt_builder.extract_memory_cues(bundle)
        modes = scaffold.candidate_modes[:2]    # max 2 candidate passes
        if num_drafts is not None:
            # TF-034: the same draft cap as the plain path.
            modes = modes[:num_drafts]

        for mode in modes:
            # TF-027 cut point: before each candidate.
            check_interrupt(turn, interrupt)
            prompt = self._prompt_builder.build_candidate_prompt(
                sketch=sketch,
                scaffold=scaffold,
                mode=mode,
                memory_cues=memory_cues,
            )
            cid = self._mint_id("cand", 6)
            yield StreamEvent(kind="candidate_start", turn_id=turn_id, mode=mode, candidate_id=cid)
            try:
                from thoughtforge.inference.turboquant import GenerationParams
                params = GenerationParams(
                    temperature=_mode_temperature(mode),
                    seed=self._next_generation_seed(),
                )
                native_stream = getattr(self._engine, "generate_stream", None)
                if callable(native_stream):
                    parts: list[str] = []
                    for delta in native_stream(prompt, params=params):
                        # TF-027 cut point: before each chunk is
                        # delivered. A cut here discards `parts`
                        # whole — the partial candidate never lands.
                        check_interrupt(turn, interrupt)
                        parts.append(delta)
                        yield StreamEvent(
                            kind="chunk", turn_id=turn_id, mode=mode,
                            candidate_id=cid, text=delta, native=True,
                        )
                    # generate() strips its completed text; the
                    # streamed candidate is the same text, stripped.
                    text = "".join(parts).strip()
                    token_estimate = len(text.split())
                else:
                    result = self._engine.generate(prompt, params=params)
                    text = result.text
                    token_estimate = result.tokens_generated
                    for chunk in chunk_words(text):
                        check_interrupt(turn, interrupt)
                        yield StreamEvent(
                            kind="chunk", turn_id=turn_id, mode=mode,
                            candidate_id=cid, text=chunk, native=False,
                        )
                if not text or not text.strip():
                    # TF-031: a stream that completes empty produced
                    # no candidate — a failure, stated as such.
                    logger.warning("Streamed candidate returned no text (mode=%s)", mode)
                    self._record_failure(
                        turn, "candidate",
                        "engine returned no text (empty generation)", mode=mode,
                    )
                    yield StreamEvent(
                        kind="candidate_error", turn_id=turn_id, mode=mode,
                        candidate_id=cid,
                        text="engine returned no text (empty generation)",
                    )
                    continue
                candidates.append(CandidateRecord(
                    candidate_id=cid,
                    mode=mode,
                    text=text,
                    token_estimate=token_estimate,
                    scores=_score_candidate(text, sketch, scaffold),
                ))
                yield StreamEvent(
                    kind="candidate_end", turn_id=turn_id, mode=mode,
                    candidate_id=cid, text=text,
                )
            except TurnInterrupted:
                raise
            except Exception as e:
                # A partial stream is discarded whole: a candidate
                # is its completed text or it is not a candidate.
                logger.warning("Streamed candidate generation failed (mode=%s): %s", mode, e)
                self._record_failure(turn, "candidate", exc=e, mode=mode)
                yield StreamEvent(
                    kind="candidate_error", turn_id=turn_id, mode=mode,
                    candidate_id=cid, text=str(e),
                )

        return candidates

    def _stream_via_unified_backend(
        self,
        turn: RuntimeTurnState,
        sketch: InputSketch,
        scaffold: CognitionScaffold,
        bundle: MemoryActivationBundle,
        *,
        interrupt: InterruptToken | None = None,
        sink: list[CandidateRecord] | None = None,
    ) -> Iterator[StreamEvent]:
        """Streamed twin of _generate_via_unified_backend (one
        candidate, first scaffold mode, same max_tokens as there).

        TF-027: the token is checked before each chunk is delivered
        and the completed candidate lands in ``sink``.
        """
        from thoughtforge.inference.unified_backend import GenerationRequest, UnifiedBackend

        turn_id = turn.turn_id
        candidates: list[CandidateRecord] = sink if sink is not None else []
        memory_cues = self._prompt_builder.extract_memory_cues(bundle)
        mode = scaffold.candidate_modes[0] if scaffold.candidate_modes else "practical"
        prompt = self._prompt_builder.build_candidate_prompt(
            sketch=sketch,
            scaffold=scaffold,
            mode=mode,
            memory_cues=memory_cues,
        )
        cid = self._mint_id("cand", 6)
        yield StreamEvent(kind="candidate_start", turn_id=turn_id, mode=mode, candidate_id=cid)
        try:
            req = GenerationRequest(
                prompt=prompt,
                temperature=_mode_temperature(mode),
                # TF-032: as the plain path — the selected profile's
                # cap, or the standing 512.
                max_tokens=self._conversation_config.generation_max_tokens(),
                # TF-036: the per-call seed (None when unseeded).
                seed=self._next_generation_seed(),
            )
            backend = self._unified_backend
            # Native only when the backend class overrides the
            # default generate_stream; the default yields completed
            # text, which is re-chunked and labelled simulated.
            native = type(backend).generate_stream is not UnifiedBackend.generate_stream
            parts: list[str] = []
            for chunk in backend.generate_stream(req):
                # TF-027 cut point: before each chunk is delivered.
                check_interrupt(turn, interrupt)
                parts.append(chunk)
                if native:
                    yield StreamEvent(
                        kind="chunk", turn_id=turn_id, mode=mode,
                        candidate_id=cid, text=chunk, native=True,
                    )
                else:
                    for piece in chunk_words(chunk):
                        check_interrupt(turn, interrupt)
                        yield StreamEvent(
                            kind="chunk", turn_id=turn_id, mode=mode,
                            candidate_id=cid, text=piece, native=False,
                        )
            text = "".join(parts) if native else (parts[0] if parts else "")
            if not text:
                # The default stream yields nothing on a backend
                # error (GenerationResponse.error) — no candidate,
                # exactly as _generate_via_unified_backend returns [].
                # TF-031: recorded on the turn as a backend failure.
                self._record_failure(
                    turn, "backend",
                    "backend returned no text (error or empty)", mode=mode,
                )
                yield StreamEvent(
                    kind="candidate_error", turn_id=turn_id, mode=mode,
                    candidate_id=cid, text="backend returned no text (error or empty)",
                )
                return candidates
            candidate = CandidateRecord(
                candidate_id=cid,
                mode=mode,
                text=text,
                token_estimate=len(text.split()),
                scores=_score_candidate(text, sketch, scaffold),
            )
            candidates.append(candidate)
            yield StreamEvent(
                kind="candidate_end", turn_id=turn_id, mode=mode,
                candidate_id=cid, text=text,
            )
            return candidates
        except TurnInterrupted:
            raise
        except Exception as e:
            logger.warning("UnifiedBackend streamed generate failed: %s", e)
            self._record_failure(turn, "backend", exc=e, mode=mode)
            yield StreamEvent(
                kind="candidate_error", turn_id=turn_id, mode=mode,
                candidate_id=cid, text=str(e),
            )
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
                fid = self._mint_id("frag", 6)
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
        turn: RuntimeTurnState,
    ) -> FinalResponseRecord:
        """
        Compose the final response via FragmentSalvage, then run EnforcementGate.

        TF-025: takes the typed turn state — sketch, scaffold, bundle,
        candidates and fragments are read from it, not passed as a
        loose argument bundle.
        """
        sketch = turn.input_sketch
        scaffold = turn.scaffold
        bundle = turn.memory_bundle
        candidates = turn.candidates
        fragments = turn.fragments
        turn_id = turn.turn_id
        assert sketch is not None and scaffold is not None and bundle is not None
        rid = self._mint_id("resp", 8)
        kept = [f for f in fragments if f.keep]

        # Delegate to FragmentSalvage for multi-pass scored reassembly
        salvage_result = self._salvage.forge(
            candidates=candidates,
            bundle=bundle,
            engine=self._engine,
            prompt_builder=self._prompt_builder,
            sketch=sketch,
            scaffold=scaffold,
            # TF-037: the selected profile's refine-pass cap
            # (phone_low 1, pi_zero 0); no profile → the standing
            # salvage default of 2, byte-identical to before.
            max_passes=self._conversation_config.refinement_max_passes(),
            # TF-036: one seeded draw per compose for the salvage
            # refine passes (None when determinism mode is off).
            generation_seed=self._next_generation_seed(),
        )

        # Run EnforcementGate
        retrieved_qids = EnforcementGate.extract_qids_from_bundle(bundle.activated_records)
        enforcement = self._enforcement.check(
            text=salvage_result.text,
            citations=salvage_result.citations,
            retrieved_qids=retrieved_qids,
        )

        final_text = enforcement.text     # may have forge note appended on fail
        scores = _score_final(final_text, sketch, scaffold)
        token_count = len(final_text.split())

        return FinalResponseRecord(
            response_id=rid,
            text=final_text,
            source_candidate_ids=[c.candidate_id for c in candidates],
            source_fragment_ids=[f.fragment_id for f in kept],
            citations=salvage_result.citations,
            scores=scores,
            enforcement_passed=enforcement.passed,
            enforcement_notes=enforcement.notes,
            token_count=token_count,
            turn_id=turn_id,
            salvage_path=salvage_result.salvage_path,
            retrieval_confidence=bundle.retrieval_confidence,
            mode="full",
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
        rid = self._mint_id("resp", 8)

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
                turn_id=turn_id,
                salvage_path="knowledge_only",
                retrieval_confidence=0.0,
                mode="knowledge_only",
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
            turn_id=turn_id,
            salvage_path="knowledge_only",
            retrieval_confidence=bundle.retrieval_confidence,
            mode="knowledge_only",
        )

    def _repair(
        self,
        turn: RuntimeTurnState,
        previous: FinalResponseRecord,
    ) -> FinalResponseRecord:
        """Repair pass — tighter prompt, lower temperature.

        TF-025: takes the typed turn state, like _compose_final.
        """
        if self._engine is None:
            return previous
        turn_id = turn.turn_id
        sketch = turn.input_sketch
        scaffold = turn.scaffold
        bundle = turn.memory_bundle
        assert sketch is not None and scaffold is not None and bundle is not None

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
                params=GenerationParams(
                    temperature=0.45,
                    # TF-036: the repair pass is a generation call
                    # too — it draws from the same seeded stream.
                    seed=self._next_generation_seed(),
                ),
            )
            scores = _score_final(result.text, sketch, scaffold)
            rid = self._mint_id("resp", 8)
            return FinalResponseRecord(
                response_id=rid,
                text=result.text,
                citations=_extract_citations(result.text, bundle),
                scores=scores,
                enforcement_passed=_check_citation_gate(result.text, bundle),
                enforcement_notes="repair pass",
                token_count=result.tokens_generated,
                turn_id=turn_id,
                salvage_path="repair",
                retrieval_confidence=bundle.retrieval_confidence,
                mode="repair",
            )
        except Exception as e:
            logger.warning("Repair pass failed: %s — returning previous result", e)
            # TF-031: recorded on the turn, not announced on the
            # answer — the composed answer the user holds is real
            # and complete; nothing was fabricated (error_honesty).
            self._record_failure(turn, "repair", exc=e)
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

# TF-025 (D-DUP-1): _split_sentences' one implementation lives in
# utils/common.py; this name is its alias (existing tests import it
# from here).
from thoughtforge.utils.common import split_sentences as _split_sentences  # noqa: E402


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

# TF-036: turn ids are minted by ThoughtForgeCore._mint_id (seeded
# in determinism mode, uuid4 otherwise) — the old module-level
# _new_turn_id helper is gone; _mint_id is its one home now.

# TF-025 (D-DUP-1): _now_iso's one implementation lives in
# utils/common.py; this name is its alias.
from thoughtforge.utils.common import now_iso as _now_iso  # noqa: E402


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
