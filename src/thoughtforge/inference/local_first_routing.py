"""TF-077 — Local-first routing: profile-aware backend choice.

Roadmap: "Local-first routing: profile-aware backend choice
(privacy + cost + capability), the router's table published."

Measured start (probed this tick, VM):

- **No router existed.** ``src/thoughtforge/inference/``
  held no routing module at all; the only router in the
  tree is ``cognition/router.py``, which routes *intent*,
  never backends. The config factory
  (``load_backend_from_config``) loads exactly ONE backend
  chosen by the config's ``backend`` key alone — its source
  contains no profile, privacy, cost, or routing notion.
- **Profiles could not choose a backend**: the hardware
  profiles' ``inference_backend`` field names an *engine*
  ("cpu" / "cuda", fallbacks "vulkan" / "directml" /
  "cpu"), not a ThoughtForge backend family — no profile
  data anywhere maps to ollama / lmstudio / aesir /
  turboquant / huggingface.
- **Placement truth existed but was never consumed**:
  TF-065 capability records state it per backend
  (huggingface ``remote`` — the one shipped backend whose
  traffic leaves the machine; turboquant + aesir
  ``local``; ollama / lmstudio / openai-compatible
  ``local_or_remote`` with loopback defaults), and
  TF-073's FallbackBackend will serve any ordered chain —
  but nothing *produced* an order, so TF-073's own doc
  left "which chain a deployment should run" to this
  slice.
- **D-HEALTH-2** (filed by TF-073, owned here): the
  server-shaped backends' ``health_check()`` cannot see
  the configured model — ollama healthy with its model
  absent from ``/api/tags``, lmstudio the same over
  ``/v1/models``. The listing surfaces existed but were
  scattered and lossy: ollama's ``list_local_models()``
  and lmstudio's ``list_models()`` both return ``[]``
  alike for "server down" and "model absent" — unknown
  dressed as absent — and only TF-076's generic binding
  had a tri-state ``model_listed()``.

THE ROUTER (``ROUTING_RULES`` as data), implemented by
:func:`route` over caller-constructed backend instances:

- ``local_first_is_absolute`` — a candidate whose
  resolved placement is local always ranks ahead of one
  whose placement is remote, whatever either's family:
  the sort key is (remote tier, table rank).
- ``remote_is_opt_in`` — remote candidates are excluded
  by default (reason ``remote_not_allowed``); only
  ``allow_remote=True`` admits them, and then still last.
  "Remote" is *resolved per instance*: a
  ``local_or_remote`` backend is local exactly when its
  base URL's host is loopback (localhost / 127.0.0.1 /
  ::1); the hosted huggingface backend is remote by
  construction; an unresolvable placement counts as
  remote (privacy-conservative).
- ``the_table_is_the_policy`` — within a tier, the order
  is ``ROUTING_TABLE[profile]``, published below and in
  ``docs/architecture/LOCAL_FIRST_ROUTING.md``: one row
  per shipped hardware profile, every backend family
  exactly once, the hosted family last in every row.
  Constrained profiles (RAM <= 4 GB in their own JSONs)
  put the in-process turboquant first — no server
  process competes for the profile's RAM; GPU profiles
  (VRAM > 0 in their JSONs) put aesir first — the
  accelerator-first engine of TF-074. TF-073's chain
  order remains the caller's once built; this slice
  *authors* the order TF-073 consumes.
- ``privacy_and_cost_are_resolved_per_instance`` — each
  decision entry carries the resolved privacy class
  (``data_stays_on_machine`` / ``data_leaves_machine``)
  and cost class (``local_compute`` /
  ``hosted_metered`` / ``remote_declared_unknown``: a
  remote self-hosted server's metering is the operator's
  own business and is declared unknown, never guessed).
- ``capability_is_a_gate_not_a_rank`` — a candidate
  whose TF-065 record says ``generate`` is not True is
  excluded (reason ``generate_not_offered``); beyond the
  gate, capabilities weighed in the *table's* bases
  (native streaming, in-process counting), never as a
  hidden runtime score that would reorder the published
  table behind the operator's back.
- ``per_model_health_is_consumed`` (D-HEALTH-2, FIXED
  here at routing grade) — :func:`model_available`
  answers True / False / None per candidate from its
  per-model surface: the tri-state ``model_listed()``
  (ollama / lmstudio / aesir gained one this slice,
  mirroring TF-076's; aesir's is *catalog* membership —
  TF-074: the catalog is not readiness, and its native
  readiness route stays the stronger separate check),
  turboquant's model-file existence, huggingface's
  per-model health itself. With
  ``require_model_listed=True`` a False excludes
  (reason ``model_not_listed``); a None never excludes —
  unknown is kept and *annotated* in the decision, never
  dressed as listed. ``health_check()`` semantics are
  unchanged: server-level health stays TF-073's gate
  inside the chain; per-model truth is this slice's.
- ``unknown_family_is_excluded`` — a backend whose name
  matches no roster family is never silently routed; it
  is excluded with its name and reason recorded.
- ``the_decision_is_the_announcement`` — :class:`Routing
  Decision` records the ordered entries, every exclusion
  with its reason, the profile, and whether remote was
  allowed; :func:`build_chain` turns a decision into
  TF-073's ``FallbackBackend`` over the ordered
  instances, so the served chain is exactly the published
  order and TF-073's report announces any degraded serve.
- ``live_proof_not_claimed`` — no live server faced this
  router on this VM; proof is fixture grade (below).

Proof grade FIXTURE (stub backends + the real classes'
records and listings over mocked transports, VM) — a
live deployment's routing against real servers is NOT
PROVEN, and per-server *speed* ranking is not claimed
anywhere: the table weighs placement, deployment weight,
and contract capabilities, never benchmark numbers this
tree has not measured.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from thoughtforge.inference.unified_backend import UnifiedBackend

ROUTING_FORMAT = "thoughtforge-local-first-routing/1"

PRIVACY_CLASSES = ("data_stays_on_machine", "data_leaves_machine")
COST_CLASSES = (
    "local_compute",
    "hosted_metered",
    "remote_declared_unknown",
)
EXCLUSION_REASONS = (
    "remote_not_allowed",
    "model_not_listed",
    "generate_not_offered",
    "unknown_family",
)

#: Loopback hosts: a server bound to one of these is local.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

# ── The roster ─────────────────────────────────────────────────
# One row per backend family the router will route. The
# capability fields are the family's contract shape, judged
# live against real instances by routing_problems(); the
# model_surface names the per-model surface
# model_available() consumes (D-HEALTH-2).

BACKEND_ROSTER: tuple[dict[str, Any], ...] = (
    {
        "family": "turboquant",
        "name_prefix": "turboquant",
        "kind": "local_process",
        "privacy_class": "local",
        "cost_class": "local_compute",
        "model_surface": "model_file_exists",
        "basis": "in-process llama.cpp engine (TF-065/066): "
                 "no server process, native streaming; the "
                 "model is a file on this machine",
    },
    {
        "family": "aesir",
        "name_prefix": "aesir",
        "kind": "local_server",
        "privacy_class": "local",
        "cost_class": "local_compute",
        "model_surface": "model_listed_catalog",
        "basis": "Project Aesir's serve (TF-074): loopback "
                 "by its own design, native streaming; its "
                 "/v1/models listing is a catalog, not "
                 "readiness — model_listed() says listing "
                 "only, readiness_check() stays separate",
    },
    {
        "family": "ollama",
        "name_prefix": "ollama",
        "kind": "configurable_server",
        "privacy_class": "configurable",
        "cost_class": "follows_placement",
        "model_surface": "model_listed",
        "basis": "the factory's first server family; "
                 "loopback by default, remote when the "
                 "operator points it elsewhere",
    },
    {
        "family": "lmstudio",
        "name_prefix": "lmstudio",
        "kind": "configurable_server",
        "privacy_class": "configurable",
        "cost_class": "follows_placement",
        "model_surface": "model_listed",
        "basis": "LM Studio's local server (TF-076's "
                 "preset leg); loopback by default",
    },
    {
        "family": "openai_compatible",
        "name_prefix": "openai-compatible",
        "kind": "configurable_server",
        "privacy_class": "configurable",
        "cost_class": "follows_placement",
        "model_surface": "model_listed",
        "basis": "any OpenAI-shaped server (TF-076); the "
                 "hosted API is this family pointed remote, "
                 "a local server is this family at loopback — "
                 "placement is resolved per instance",
    },
    {
        "family": "huggingface",
        "name_prefix": "huggingface",
        "kind": "remote_hosted",
        "privacy_class": "remote",
        "cost_class": "hosted_metered",
        "model_surface": "per_model_health",
        "basis": "the hosted inference API (TF-065): the "
                 "one family whose traffic leaves the "
                 "machine by construction; its health_check "
                 "is already per-model (D-HEALTH-1)",
    },
)

ROSTER_BY_PREFIX: dict[str, dict[str, Any]] = {
    row["name_prefix"]: row for row in BACKEND_ROSTER
}
ROSTER_BY_FAMILY: dict[str, dict[str, Any]] = {
    row["family"]: row for row in BACKEND_ROSTER
}

# ── The published table ────────────────────────────────────────
# One row per shipped hardware profile: every family
# exactly once, hosted last. Bases are policy rationale
# (placement / deployment weight / contract capability),
# never unmeasured speed claims.

ROUTING_TABLE: tuple[dict[str, Any], ...] = (
    {
        "profile": "pi_zero",
        "order": ("turboquant", "ollama", "aesir",
                  "lmstudio", "openai_compatible", "huggingface"),
        "basis": "constrained class (0.5 GB in its JSON): "
                 "in-process first — no server process "
                 "competes for the profile's RAM; the "
                 "lightest local server next; hosted last "
                 "and remote-gated",
    },
    {
        "profile": "phone_low",
        "order": ("turboquant", "ollama", "aesir",
                  "lmstudio", "openai_compatible", "huggingface"),
        "basis": "constrained class (2 GB in its JSON): "
                 "in-process first, as pi_zero; hosted last "
                 "and remote-gated",
    },
    {
        "profile": "pi_5",
        "order": ("turboquant", "ollama", "aesir",
                  "lmstudio", "openai_compatible", "huggingface"),
        "basis": "constrained class (4 GB in its JSON): "
                 "in-process first, as pi_zero; hosted last "
                 "and remote-gated",
    },
    {
        "profile": "desktop_cpu",
        "order": ("turboquant", "aesir", "ollama",
                  "lmstudio", "openai_compatible", "huggingface"),
        "basis": "in-process first (native streaming, no "
                 "server to run); aesir's local serve "
                 "second (native streaming under contract); "
                 "then the configurable local servers in "
                 "factory precedence; hosted last",
    },
    {
        "profile": "desktop_gpu",
        "order": ("aesir", "turboquant", "ollama",
                  "lmstudio", "openai_compatible", "huggingface"),
        "basis": "GPU class (VRAM in its JSON): aesir "
                 "first — the accelerator-first engine "
                 "(TF-074); in-process second; hosted last "
                 "and remote-gated",
    },
    {
        "profile": "server_gpu",
        "order": ("aesir", "turboquant", "ollama",
                  "lmstudio", "openai_compatible", "huggingface"),
        "basis": "GPU class (VRAM in its JSON): aesir "
                 "first, as desktop_gpu; hosted last and "
                 "remote-gated",
    },
)

TABLE_BY_PROFILE: dict[str, dict[str, Any]] = {
    row["profile"]: row for row in ROUTING_TABLE
}

ROUTING_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "local_first_is_absolute",
        "rule": "a locally-resolved candidate always ranks "
                "ahead of a remotely-resolved one, whatever "
                "either's family — the sort key is (remote "
                "tier, table rank)",
        "measured_by": "a remote-configured ollama stub "
                       "ranks behind every local stub "
                       "despite its family's table rank",
    },
    {
        "id": "remote_is_opt_in",
        "rule": "remote candidates are excluded by default "
                "(remote_not_allowed); allow_remote=True "
                "admits them, still in the remote tier; "
                "placement resolves per instance from the "
                "capability record + base URL host, and an "
                "unresolvable placement counts as remote",
        "measured_by": "the hosted stub is excluded by "
                       "default with the reason recorded, "
                       "admitted last when allowed; a "
                       "loopback ollama resolves local and "
                       "an off-box one remote",
    },
    {
        "id": "the_table_is_the_policy",
        "rule": "within a tier the order is the profile's "
                "ROUTING_TABLE row — every family exactly "
                "once, the hosted family last in every row; "
                "constrained profiles (RAM <= 4 GB in their "
                "JSONs) lead with the in-process family, GPU "
                "profiles (VRAM > 0) with aesir",
        "measured_by": "the checker re-derives the "
                       "invariants from the live profile "
                       "JSONs; stub routes per profile "
                       "reproduce each row's local order",
    },
    {
        "id": "privacy_and_cost_are_resolved_per_instance",
        "rule": "every decision entry carries its resolved "
                "privacy class (data_stays_on_machine / "
                "data_leaves_machine) and cost class "
                "(local_compute / hosted_metered / "
                "remote_declared_unknown — a remote "
                "self-hosted server's metering is declared "
                "unknown, never guessed)",
        "measured_by": "decision entries over stubs + "
                       "real instances carry classes in "
                       "the filed vocabularies, resolved "
                       "from placement",
    },
    {
        "id": "capability_is_a_gate_not_a_rank",
        "rule": "a candidate whose TF-065 record offers "
                "no generate is excluded "
                "(generate_not_offered); no hidden runtime "
                "score reorders the published table",
        "measured_by": "a generate=False stub is excluded "
                       "with the reason; equal-tier order "
                       "equals table order exactly",
    },
    {
        "id": "per_model_health_is_consumed",
        "rule": "D-HEALTH-2's fix at routing grade: "
                "model_available() answers True/False/None "
                "from each family's per-model surface; "
                "with require_model_listed=True a False "
                "excludes (model_not_listed) and a None is "
                "kept, annotated — unknown is never dressed "
                "as listed or as absent",
        "measured_by": "stubs with listed/absent/unknown "
                       "models route, exclude, and annotate "
                       "respectively; the real ollama / "
                       "lmstudio / aesir model_listed() "
                       "legs are pinned over mocked "
                       "transports",
    },
    {
        "id": "unknown_family_is_excluded",
        "rule": "a backend matching no roster family is "
                "excluded with its name + reason recorded — "
                "never silently routed",
        "measured_by": "an unrostered stub appears only "
                       "in the decision's exclusions",
    },
    {
        "id": "the_decision_is_the_announcement",
        "rule": "the RoutingDecision records ordered "
                "entries, exclusions with reasons, profile "
                "and the remote flag; build_chain() makes "
                "TF-073's FallbackBackend over exactly the "
                "ordered instances",
        "measured_by": "a built chain's chain_names() "
                       "equals the decision's ordered "
                       "names, and the chain serves through "
                       "its first healthy member",
    },
    {
        "id": "live_proof_not_claimed",
        "rule": "no live server has faced this router; "
                "the table weighs placement, deployment "
                "weight and contract capabilities — never "
                "unmeasured speed",
        "measured_by": "filed here and in the doc; no "
                       "test asserts a live route",
    },
)


# ── Classification ─────────────────────────────────────────────


def family_of(backend: Any) -> str | None:
    """The roster family a backend instance belongs to,
    from its contract name prefix; None when unrostered."""
    name = backend.backend_name()
    prefix = name.split(":", 1)[0]
    row = ROSTER_BY_PREFIX.get(prefix)
    return row["family"] if row else None


def is_loopback_url(url: str | None) -> bool:
    if not url:
        return False
    host = urlparse(url).hostname
    return host is not None and host.lower() in LOOPBACK_HOSTS


def resolved_placement(backend: Any) -> str:
    """'local' or 'remote', resolved per instance: the
    TF-065 placement record first (local / remote are
    conclusive), a local_or_remote record resolved by the
    base URL's host, anything unresolvable counted remote
    (privacy-conservative)."""
    placement = backend.capabilities().placement
    if placement == "local":
        return "local"
    if placement == "remote":
        return "remote"
    if placement == "local_or_remote":
        return "local" if is_loopback_url(
            getattr(backend, "base_url", None)) else "remote"
    return "remote"


def resolved_cost(row: dict[str, Any], placement: str) -> str:
    if row["cost_class"] == "follows_placement":
        return "local_compute" if placement == "local" \
            else "remote_declared_unknown"
    return row["cost_class"]


# ── Per-model availability (D-HEALTH-2's consumption) ─────────


def model_available(backend: Any) -> bool | None:
    """True / False / None from the family's per-model
    surface. None is unknown — never dressed as either."""
    family = family_of(backend)
    if family is None:
        return None
    surface = ROSTER_BY_FAMILY[family]["model_surface"]
    if surface in ("model_listed", "model_listed_catalog"):
        listed = getattr(backend, "model_listed", None)
        return listed() if callable(listed) else None
    if surface == "model_file_exists":
        path = getattr(backend, "model_path", None)
        return Path(path).exists() if path is not None else None
    if surface == "per_model_health":
        try:
            return bool(backend.health_check())
        except Exception:  # noqa: BLE001 — unknown, stated
            return None
    return None


# ── The decision ───────────────────────────────────────────────


@dataclass
class RoutingEntry:
    backend: Any = field(repr=False)
    name: str = ""
    family: str = ""
    placement: str = ""            # local | remote (resolved)
    privacy: str = ""              # PRIVACY_CLASSES
    cost: str = ""                 # COST_CLASSES
    model_listed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "placement": self.placement,
            "privacy": self.privacy,
            "cost": self.cost,
            "model_listed": self.model_listed,
        }


@dataclass
class RoutingExclusion:
    name: str = ""
    reason: str = ""               # EXCLUSION_REASONS
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "reason": self.reason,
                "detail": self.detail}


@dataclass
class RoutingDecision:
    profile: str = ""
    allow_remote: bool = False
    ordered: tuple[RoutingEntry, ...] = ()
    excluded: tuple[RoutingExclusion, ...] = ()

    @property
    def chosen(self) -> RoutingEntry | None:
        return self.ordered[0] if self.ordered else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": ROUTING_FORMAT,
            "profile": self.profile,
            "allow_remote": self.allow_remote,
            "ordered": [e.to_dict() for e in self.ordered],
            "excluded": [e.to_dict() for e in self.excluded],
            "chosen": self.chosen.name if self.chosen else None,
        }


def route(
    profile_id: str,
    backends: list[Any],
    *,
    allow_remote: bool = False,
    require_model_listed: bool = False,
) -> RoutingDecision:
    """Route candidate instances for a hardware profile.

    The candidates are caller-constructed (the factory /
    deployment owns construction and credentials); the
    router classifies, gates, and orders them — it never
    constructs a backend and never touches the network
    except the per-model listing calls the model gate
    makes when ``require_model_listed`` asks for them
    (model availability is also *recorded*, unasked-for,
    as None in that case — not probed, stated)."""
    if profile_id not in TABLE_BY_PROFILE:
        raise ValueError(
            f"no routing table row for profile {profile_id!r}")
    rank = {family: idx for idx, family in enumerate(
        TABLE_BY_PROFILE[profile_id]["order"])}
    entries: list[RoutingEntry] = []
    excluded: list[RoutingExclusion] = []
    for backend in backends:
        name = backend.backend_name()
        family = family_of(backend)
        if family is None:
            excluded.append(RoutingExclusion(
                name, "unknown_family",
                "no roster family matches its name prefix"))
            continue
        row = ROSTER_BY_FAMILY[family]
        if not backend.capabilities().generate:
            excluded.append(RoutingExclusion(
                name, "generate_not_offered",
                "its TF-065 record offers no generate"))
            continue
        placement = resolved_placement(backend)
        if placement == "remote" and not allow_remote:
            excluded.append(RoutingExclusion(
                name, "remote_not_allowed",
                "resolved placement is remote and remote "
                "was not allowed"))
            continue
        listed: bool | None = None
        if require_model_listed:
            listed = model_available(backend)
            if listed is False:
                excluded.append(RoutingExclusion(
                    name, "model_not_listed",
                    "its per-model surface answered False"))
                continue
        entries.append(RoutingEntry(
            backend=backend,
            name=name,
            family=family,
            placement=placement,
            privacy=("data_stays_on_machine"
                     if placement == "local"
                     else "data_leaves_machine"),
            cost=resolved_cost(row, placement),
            model_listed=listed,
        ))
    entries.sort(key=lambda e: (
        0 if e.placement == "local" else 1, rank[e.family]))
    return RoutingDecision(
        profile=profile_id,
        allow_remote=allow_remote,
        ordered=tuple(entries),
        excluded=tuple(excluded),
    )


def build_chain(decision: RoutingDecision):
    """TF-073's FallbackBackend over the decision's
    ordered instances — the served chain is exactly the
    published order."""
    from thoughtforge.inference.backend_fallback import FallbackBackend

    if not decision.ordered:
        raise ValueError(
            "routing decision holds no backend to chain")
    return FallbackBackend([e.backend for e in decision.ordered])


def routing_as_data() -> dict[str, Any]:
    return {
        "format": ROUTING_FORMAT,
        "rules": [dict(r) for r in ROUTING_RULES],
        "roster": [dict(r) for r in BACKEND_ROSTER],
        "table": [
            {**row, "order": list(row["order"])}
            for row in ROUTING_TABLE
        ],
        "privacy_classes": list(PRIVACY_CLASSES),
        "cost_classes": list(COST_CLASSES),
        "exclusion_reasons": list(EXCLUSION_REASONS),
    }


# ── Stubs for the live probes ──────────────────────────────────


class _StubBackend(UnifiedBackend):
    """A contract-shaped stub in a roster family's name."""

    def __init__(self, name, *, placement="local",
                 base_url=None, generate=True, listed=None):
        self._name = name
        self._placement = placement
        self.base_url = base_url
        self._generate = generate
        self._listed = listed

    def backend_name(self):
        return self._name

    def generate(self, request):
        from thoughtforge.inference.unified_backend import (
            GenerationResponse,
        )

        return GenerationResponse(
            text=f"text from {self._name}",
            backend_used=self._name)

    def health_check(self):
        return True

    def model_listed(self):
        return self._listed

    def capabilities(self):
        from thoughtforge.inference.backend_contract import (
            BackendCapabilities,
        )

        return BackendCapabilities(
            name=self._name, generate=self._generate,
            placement=self._placement,
            model=getattr(self, "model", ""))


def _stub_set() -> list[_StubBackend]:
    return [
        _StubBackend("huggingface:stub-remote",
                     placement="remote"),
        _StubBackend("openai-compatible:stub-local",
                     placement="local_or_remote",
                     base_url="http://127.0.0.1:8000"),
        _StubBackend("turboquant:stub.gguf", placement="local"),
        _StubBackend("ollama:stub-model",
                     placement="local_or_remote",
                     base_url="http://localhost:11434"),
        _StubBackend("aesir:stub-resident", placement="local"),
    ]


# ── Checkers ───────────────────────────────────────────────────


def _live_profiles() -> dict[str, dict[str, Any]]:
    from thoughtforge.utils.config import (
        list_hardware_profiles,
        load_hardware_profile,
    )

    return {pid: load_hardware_profile(pid)
            for pid in list_hardware_profiles()}


def _cheap_instances() -> dict[str, Any]:
    from pathlib import Path as _P

    from thoughtforge.inference.aesir_backend import AesirBackend
    from thoughtforge.inference.hf_backend import HuggingFaceBackend
    from thoughtforge.inference.lmstudio_backend import LMStudioBackend
    from thoughtforge.inference.ollama_backend import OllamaBackend
    from thoughtforge.inference.openai_compatible import (
        OpenAICompatibleBackend,
    )
    from thoughtforge.inference.turboquant_backend import (
        TurboQuantBackend,
    )

    return {
        "turboquant": TurboQuantBackend(
            model_path=_P("/nonexistent/tf077.gguf")),
        "aesir": AesirBackend(),
        "ollama": OllamaBackend(),
        "lmstudio": LMStudioBackend(),
        "openai_compatible": OpenAICompatibleBackend(),
        "huggingface": HuggingFaceBackend(),
    }


def routing_problems() -> list[str]:
    """The filed estate judged against the live tree."""
    problems: list[str] = []

    families = [r["family"] for r in BACKEND_ROSTER]
    if len(families) != len(set(families)):
        problems.append("BACKEND_ROSTER families are not unique")
    prefixes = [r["name_prefix"] for r in BACKEND_ROSTER]
    if len(prefixes) != len(set(prefixes)):
        problems.append("BACKEND_ROSTER name prefixes are not unique")
    rule_ids = [r["id"] for r in ROUTING_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("ROUTING_RULES ids are not unique")

    # The roster judged against the real classes.
    instances = _cheap_instances()
    if set(instances) != set(families):
        problems.append("cheap instance set != roster families")
    for family, instance in instances.items():
        row = ROSTER_BY_FAMILY[family]
        if family_of(instance) != family:
            problems.append(
                f"{family}: real instance classifies as "
                f"{family_of(instance)}")
        caps = instance.capabilities()
        expected_privacy = {
            "local": "local", "remote": "remote",
            "local_or_remote": "configurable",
        }.get(caps.placement)
        if row["privacy_class"] != expected_privacy:
            problems.append(
                f"{family}: roster privacy {row['privacy_class']} "
                f"!= live placement {caps.placement}")
        surface = row["model_surface"]
        if surface in ("model_listed", "model_listed_catalog"):
            if not callable(getattr(instance, "model_listed", None)):
                problems.append(
                    f"{family}: filed surface model_listed() "
                    f"is not implemented (D-HEALTH-2)")
        elif surface == "model_file_exists":
            if not hasattr(instance, "model_path"):
                problems.append(
                    f"{family}: filed surface needs model_path")
        elif surface == "per_model_health":
            src = inspect.getsource(type(instance).health_check)
            if "status_code == 200" not in src:
                problems.append(
                    f"{family}: per-model health no longer "
                    f"requires exactly 200 (D-HEALTH-1)")

    # The table judged against the live profile JSONs.
    profiles = _live_profiles()
    if {r["profile"] for r in ROUTING_TABLE} != set(profiles):
        problems.append(
            f"table profiles "
            f"{sorted(r['profile'] for r in ROUTING_TABLE)} != "
            f"shipped profiles {sorted(profiles)}")
    for row in ROUTING_TABLE:
        pid = row["profile"]
        order = row["order"]
        if sorted(order) != sorted(families):
            problems.append(
                f"{pid}: table row is not every family "
                f"exactly once")
            continue
        if order[-1] != "huggingface":
            problems.append(f"{pid}: hosted family is not last")
        profile = profiles.get(pid, {})
        ram = profile.get("hardware", {}).get("ram_gb", 0)
        vram = profile.get("hardware", {}).get("vram_gb", 0)
        if ram and ram <= 4 and order[0] != "turboquant":
            problems.append(
                f"{pid}: constrained profile (ram {ram}) "
                f"does not lead with the in-process family")
        if vram and vram > 0 and order[0] != "aesir":
            problems.append(
                f"{pid}: GPU profile (vram {vram}) does not "
                f"lead with aesir")
    return problems


def routing_live_problems() -> list[str]:
    """The router re-derived live over stub candidates."""
    problems: list[str] = []

    # Default route on desktop_cpu: local only, table order.
    decision = route("desktop_cpu", _stub_set())
    names = [e.name for e in decision.ordered]
    expected = ["turboquant:stub.gguf", "aesir:stub-resident",
                "ollama:stub-model", "openai-compatible:stub-local"]
    if names != expected:
        problems.append(f"desktop_cpu local order {names}")
    if [e.reason for e in decision.excluded] != ["remote_not_allowed"]:
        problems.append("hosted stub not excluded as remote by default")
    if decision.chosen is None \
            or decision.chosen.name != "turboquant:stub.gguf":
        problems.append("chosen is not the table's first local family")
    for entry in decision.ordered:
        if entry.privacy != "data_stays_on_machine" \
                or entry.cost != "local_compute":
            problems.append(
                f"{entry.name}: local entry classes wrong "
                f"({entry.privacy}/{entry.cost})")

    # Remote admitted: still last, classes resolved.
    allowed = route("desktop_cpu", _stub_set(), allow_remote=True)
    if not allowed.ordered \
            or allowed.ordered[-1].family != "huggingface":
        problems.append("allowed remote is not last")
    elif allowed.ordered[-1].privacy != "data_leaves_machine" \
            or allowed.ordered[-1].cost != "hosted_metered":
        problems.append("allowed remote classes not resolved")

    # Local-first is absolute: a remote-configured family
    # member falls behind every local despite table rank.
    mixed = route("desktop_gpu", [
        _StubBackend("ollama:off-box", placement="local_or_remote",
                     base_url="https://models.example.org:11434"),
        _StubBackend("lmstudio:near", placement="local_or_remote",
                     base_url="http://localhost:1234"),
    ], allow_remote=True)
    if [e.family for e in mixed.ordered] != ["lmstudio", "ollama"]:
        problems.append(
            "remote-configured ollama did not fall behind "
            "the local lmstudio (local-first not absolute)")
    if mixed.ordered[-1].cost != "remote_declared_unknown":
        problems.append("remote self-hosted cost not declared unknown")

    # The model gate: False excludes, None annotates.
    gated = route("pi_5", [
        _StubBackend("ollama:absent", placement="local_or_remote",
                     base_url="http://localhost:11434",
                     listed=False),
        _StubBackend("turboquant:unknown.gguf", placement="local",
                     listed=None),
    ], require_model_listed=True)
    if [e.family for e in gated.ordered] != ["turboquant"]:
        problems.append("model gate did not exclude the False stub")
    if gated.ordered and gated.ordered[0].model_listed is not None:
        problems.append("unknown model state dressed as known")
    if [e.reason for e in gated.excluded] != ["model_not_listed"]:
        problems.append("model exclusion reason not recorded")

    # Unknown family + no-generate gate.
    odd = route("pi_5", [
        _StubBackend("mystery:1", placement="local"),
        _StubBackend("aesir:nogen", placement="local",
                     generate=False),
        _StubBackend("turboquant:ok.gguf", placement="local"),
    ])
    reasons = {e.name: e.reason for e in odd.excluded}
    if reasons.get("mystery:1") != "unknown_family":
        problems.append("unknown family not excluded as such")
    if reasons.get("aesir:nogen") != "generate_not_offered":
        problems.append("generate=False not excluded as such")

    # The chain: TF-073 serves the decision's exact order.
    chain = build_chain(decision)
    if chain.chain_names() != names:
        problems.append("built chain order != decision order")
    from thoughtforge.inference.unified_backend import GenerationRequest

    resp = chain.generate(GenerationRequest(prompt="prompt"))
    if resp.text != "text from turboquant:stub.gguf":
        problems.append(f"chain served {resp.text!r}")
    return problems


__all__ = [
    "BACKEND_ROSTER",
    "COST_CLASSES",
    "EXCLUSION_REASONS",
    "LOOPBACK_HOSTS",
    "PRIVACY_CLASSES",
    "ROSTER_BY_FAMILY",
    "ROSTER_BY_PREFIX",
    "ROUTING_FORMAT",
    "ROUTING_RULES",
    "ROUTING_TABLE",
    "TABLE_BY_PROFILE",
    "RoutingDecision",
    "RoutingEntry",
    "RoutingExclusion",
    "build_chain",
    "family_of",
    "is_loopback_url",
    "model_available",
    "resolved_placement",
    "route",
    "routing_as_data",
    "routing_live_problems",
    "routing_problems",
]
