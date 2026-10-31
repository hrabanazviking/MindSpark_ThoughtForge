# TF-077 — Local-First Routing

*Profile-aware backend choice (privacy + cost +
capability), the router's table published. The estate
lives in `src/thoughtforge/inference/local_first_routing.py`
(`BACKEND_ROSTER` / `ROUTING_TABLE` / `ROUTING_RULES` as
data, `route()` / `build_chain()` / `model_available()`,
format `thoughtforge-local-first-routing/1`).*

## Measured start

No router existed: inference held no routing module
(the tree's only router, `cognition/router.py`, routes
intent, never backends). The config factory loads
exactly one backend, chosen by the config's `backend`
key alone — no profile, privacy, cost, or routing
notion in its source. The hardware profiles could not
choose a backend at all: their `inference_backend`
field names an engine ("cpu" / "cuda", fallbacks
"vulkan" / "directml" / "cpu"), not a ThoughtForge
backend family. TF-065's placement truth (huggingface
remote; turboquant + aesir local; ollama / lmstudio /
openai-compatible local-or-remote with loopback
defaults) existed unconsumed, and TF-073's
FallbackBackend would serve any ordered chain but
nothing authored an order — its doc left "which chain
a deployment should run" to this slice.

**D-HEALTH-2** (TF-073's filing, owned here): the
server-shaped backends' health checks cannot see the
configured model, and their listing surfaces were
lossy — ollama's `list_local_models()` and lmstudio's
`list_models()` return `[]` alike for "server down"
and "model absent", unknown dressed as absent. Only
TF-076's generic binding had a tri-state
`model_listed()`.

## The roster

Six families, each filed with its kind, privacy class,
cost class, and per-model surface, judged live against
the real classes' TF-065 records: **turboquant**
(in-process, local), **aesir** (local server, loopback
by its own design), **ollama**, **lmstudio**,
**openai_compatible** (configurable servers — loopback
by default, remote when pointed elsewhere), and
**huggingface** (remote hosted — the one family whose
traffic leaves the machine by construction).

## The table

One row per shipped hardware profile, every family
exactly once, the hosted family last in every row:

| profile | order (first → last) |
|---|---|
| pi_zero | turboquant, ollama, aesir, lmstudio, openai_compatible, huggingface |
| phone_low | turboquant, ollama, aesir, lmstudio, openai_compatible, huggingface |
| pi_5 | turboquant, ollama, aesir, lmstudio, openai_compatible, huggingface |
| desktop_cpu | turboquant, aesir, ollama, lmstudio, openai_compatible, huggingface |
| desktop_gpu | aesir, turboquant, ollama, lmstudio, openai_compatible, huggingface |
| server_gpu | aesir, turboquant, ollama, lmstudio, openai_compatible, huggingface |

Bases (filed per row, policy rationale — never
unmeasured speed): constrained profiles (RAM ≤ 4 GB
in their own JSONs) lead with the in-process family,
because no server process competes for the profile's
RAM; GPU profiles (VRAM > 0) lead with aesir, the
accelerator-first engine of TF-074; desktop_cpu leads
in-process, then aesir's local serve (both native
streaming under the contract), then the configurable
servers in factory precedence.

## The rules

Filed as `ROUTING_RULES`, each measured in
`tests/test_tf077_local_first_routing.py`:

- `local_first_is_absolute` — the sort key is
  (remote tier, table rank): a locally-resolved
  candidate always ranks ahead of a remotely-resolved
  one, whatever either's family.
- `remote_is_opt_in` — remote candidates are excluded
  by default (`remote_not_allowed`); `allow_remote=True`
  admits them, still last. Placement resolves per
  instance: the TF-065 record first, a
  `local_or_remote` record by its base URL's host
  (loopback = local); anything unresolvable counts as
  remote, privacy-conservative.
- `the_table_is_the_policy` — within a tier, the
  profile's table row is the order; the checker
  re-derives the row invariants from the live profile
  JSONs.
- `privacy_and_cost_are_resolved_per_instance` — every
  decision entry carries `data_stays_on_machine` /
  `data_leaves_machine` and `local_compute` /
  `hosted_metered` / `remote_declared_unknown` (a
  remote self-hosted server's metering is the
  operator's own business — declared unknown, never
  guessed).
- `capability_is_a_gate_not_a_rank` — a candidate
  whose TF-065 record offers no `generate` is excluded
  (`generate_not_offered`); no hidden runtime score
  reorders the published table.
- `per_model_health_is_consumed` — D-HEALTH-2's fix
  (below).
- `unknown_family_is_excluded` — an unrostered backend
  is excluded with its name and reason, never silently
  routed.
- `the_decision_is_the_announcement` — the
  `RoutingDecision` records ordered entries plus every
  exclusion with its reason; `build_chain()` turns it
  into TF-073's FallbackBackend over exactly those
  instances, and TF-073's report announces any
  degraded serve from there.
- `live_proof_not_claimed` — below.

## D-HEALTH-2 — fixed at routing grade

Ollama, LMStudio, and Aesir backends gained tri-state
`model_listed()` mirroring TF-076's surface (None =
unconfigured or unaskable — unknown, never dressed as
absent; ollama's configured name without a tag matches
the server's implied `:latest`). Aesir's is *catalog*
membership only: TF-074's ruling stands — the catalog
is not readiness, and its native `readiness_check()`
remains the stronger separate check. The router's
`model_available()` consumes every family's surface —
`model_listed()` for the server families, model-file
existence for turboquant, the per-model health itself
for huggingface — and with `require_model_listed=True`
a False excludes (`model_not_listed`) while a None is
kept and annotated in the decision. `health_check()`
semantics are unchanged: server-level health stays
TF-073's in-chain gate; per-model truth is routing's.

## Placement

The router authors chains; it does not join the
factory and does not construct backends — construction
(config, credentials, model files) stays the factory's
and the deployment's, TF-153's road. Candidates arrive
as instances; the decision is data plus the instances
it ordered.

## Proof grade

**Fixture.** Stub candidates under the real router,
the real classes' records and listings over mocked
transports, the table judged against the live profile
JSONs, and a built chain serving through TF-073's real
wrapper. No live server has faced this router on this
VM, and no per-server speed ranking is claimed
anywhere — the table weighs placement, deployment
weight, and contract capabilities only. Live routing
against real servers is **NOT PROVEN**; TF-086's gate
truth table should carry local-first routing as
fixture-proven until a deployment proves it live.
