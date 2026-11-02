"""
TF-160 — Kista custody, filed: any credential
a binding needs lives in Kista-shaped custody,
never in configs.

TF-160's roadmap text: *Kista custody: any
credential a binding needs lives in
Kista-shaped custody, never in configs.* It
is Part IX slice 8, following TF-159's Ember
binding, TF-158's Hermes binding, TF-157's
Samtal binding, TF-156's Rúnmál binding,
TF-155's memory API, TF-154's cognition API
and TF-153's service shape — every one of
which filed credential custody NOT routed and
named this slice as its owner.

The measured start, probed live before
anything was built:

  * **No custody existed on the ThoughtForge
    side.** No ``thoughtforge.kista_custody``
    module existed on this tree, and the
    credential-shaped config keys TF-009
    censused (``openai_api_key``, ``hf_token``
    in ``configs/user_config.yaml``) still sat
    in a plain YAML file as raw keys — empty
    in the shipped copy (verified at the byte
    level: both values are ``""``, so no
    secret was in the tree), but the shape
    was the defect (TF-016's D-CFG-4, OPEN,
    assigned to this slice). The live readers
    of those raw keys are
    ``inference/unified_backend.py``'s
    ``load_backend_from_config()`` (the
    LM-Studio/OpenAI-compatible branch and the
    HuggingFace branch) and the setup wizard
    ``setup_thoughtforge.py``, which writes
    them.
  * **Every binding filed credential custody
    NOT routed, verified live before the
    build**: TF-158's Hermes binding and
    TF-159's Ember binding both file a
    ``credential_custody`` call NOT routed
    naming TF-160 as owner; TF-157's Samtal
    binding files the seat token no-home
    (``seat.token``, disposition
    ``no_home_not_passed`` — a Samtal seat
    never carries its own credentials);
    TF-156's Rúnmál binding files no
    credential surface at all (no credential
    call or field row exists in its filings).
    The honest start this slice files:
    **no current binding routes a credential,
    and no current binding needs one** —
    custody is the shape any future need must
    use, plus the governance of the one place
    credentials are consumed today (the
    backend factory's config keys).
  * **The real Kista shape, read READ-ONLY
    from ``~/workspace/hermes-hack-build``
    (never edited, never imported)** — the
    fork's ``kista-secret-source`` plugin
    (``plugins/kista-secret-source/``:
    ``plugin.yaml`` manifest_version 2, name
    ``kista-secret-source``, version
    ``"0.2.0"``, "Profile-local Kista
    credentials through Hermes' Secret Source
    API", homepage
    ``https://github.com/hrabanazviking/kista``;
    ``__init__.py`` registers exactly one
    thing, ``ctx.register_secret_source(
    KistaSource())`` — no tools, hooks or
    prompt-visible surfaces). Its
    ``source.py`` files the shape this slice
    mirrors:

      - **the reference form**: a binding is
        ``ENV_VAR -> kista://service/field``,
        judged by the plugin's own pattern
        ``^kista://(?P<service>[a-z0-9]
        [a-z0-9_.-]{0,127})/(?P<field>[a-z_]
        [a-z0-9_]{0,63})$`` — a reference is a
        pointer filed in config, never a
        value.
      - **the source**: ``KistaSource``
        (``agent.secret_sources.base``'s
        ``SecretSource``): name ``kista``,
        label ``Kista``, shape ``mapped``,
        scheme ``kista``,
        ``override_existing_default`` False.
        Its config schema: ``enabled``
        (default False — custody is off until
        enabled), ``env`` (the bindings map),
        ``binary_path`` (a pinned Kista
        executable), ``vault_dir`` (default
        ``credentials``, relative to the
        active profile and refused if it
        escapes it), ``cli_timeout_seconds``
        (default 30), ``override_existing``
        (default False).
      - **the vault**: a profile-local
        directory holding ``.vault_key`` and
        ``vault.json.enc``; before any read
        the plugin proves it is a real
        directory (no links/reparse points),
        owned by the current user, with no
        group/world permission bits — an
        unsafe vault is a refusal, never a
        read.
      - **the resolution**: the official
        Kista CLI is invoked as ``kista get
        -- <service>`` with ``KISTA_DIR``
        pointing at the vault; the entry
        comes back as JSON and the filed
        ``field`` is read from it. Failures
        are classified (REF_INVALID /
        NOT_CONFIGURED / AUTH_FAILED /
        TIMEOUT / BINARY_MISSING / INTERNAL)
        and — the discipline this slice
        mirrors — **a failure carries no CLI
        output and no reference echo**, and
        reference validation warns by env-var
        name only, "without echoing refs".
  * **Hermes-side custody facts TF-158
    already filed**: volmarr-core reads its
    endpoint key from a profile credential
    file (``cognition_api_key_file``, default
    ``secrets/aesir.key``) and the fork's
    kista-secret-source plugin resolves
    ``kista://`` bindings — on that side,
    too, the key never rides a request.
  * **The neighbours stood clean live**:
    ``service_shape_problems()``,
    ``cognition_api_problems()``,
    ``memory_api_problems()``,
    ``runmal_binding_problems()``,
    ``samtal_binding_problems()``,
    ``hermes_binding_problems()`` and
    ``ember_binding_problems()`` all returned
    [] on this tree before the build.

**What this slice files**

  * ``BINDING_NEED_ROWS`` — every consumer
    binding's credential need, as literal
    data, re-derived live by the checker
    from TF-156..TF-159's own filings: the
    four bindings need NO credential today
    (filed NOT routed / no-home there), the
    TF-153 service shape needs none
    (loopback, no credential surface), and
    the one consumer that does consume
    credentials today — the backend factory
    behind ``configs/user_config.yaml`` —
    is filed as custody-referenced, its two
    raw keys migrated by this slice (below).
  * ``CUSTODY_ROWS`` — the Kista shape
    itself, as literal data filed from the
    plugin source read above: the plugin,
    the reference form, the vault, the CLI
    resolution, the source config schema,
    and the Hermes profile key-file fact —
    each row stating what was verified and
    what is shape-only here.
  * ``CONFIG_KEY_ROWS`` — the two
    credential-shaped config keys TF-009
    censused, as literal data: each raw key,
    the custody reference key that replaces
    it in the shipped file, and the filed
    ``kista://`` reference. **This slice
    migrates the shipped file**:
    ``configs/user_config.yaml`` no longer
    carries ``openai_api_key`` / ``hf_token``
    as keys at all; it carries
    ``openai_api_key_ref: "kista://
    thoughtforge/openai_api_key"`` and
    ``hf_token_ref: "kista://thoughtforge/
    hf_token"``. The shipped values were
    empty, so the factory's behaviour on
    the shipped file is unchanged (an empty
    key and an absent key both fall back to
    its defaults, verified in the tests).
  * ``FIELD_ROWS`` — the custody resolution
    request/response mapping, as literal
    data: the reference is the only secret-
    shaped input; the resolved value has NO
    response field — it is delivered only
    to a caller-supplied consumer callable,
    in-process, and never appears in any
    envelope, filing, trace or log this
    module produces (proven with a sentinel
    value by the probes).
  * ``RULE_ROWS`` — the custody law, as
    literal data: never in configs;
    reference, never value; the resolved
    value is never serialised; custody off
    by default; resolution only through an
    injected resolver; failures name a
    kind, never a value or an echo; called,
    never reimplemented (no Kista code is
    imported or copied — the shape is filed
    as data and the pattern as a pattern);
    every field disposed; responses
    detached, JSON-safe, TF-034-versioned.
  * ``KistaCustody`` — the custody handle.
    It wraps an injectable resolver callable
    ``resolver(service, field) -> str``
    (directly or by factory; built lazily —
    constructing the custody builds no
    resolver). **DEFAULT DISABLED**:
    ``DEFAULT_ENABLED = False`` is filed as
    data, mirroring the Kista source's own
    ``enabled`` default False; while
    disabled every resolution returns a
    truthful refusal as data and touches no
    resolver. Enabling is an explicit act:
    the constructor flag ``enabled=True``
    or ``enable()`` (``disable()``
    reverses). An enabled custody with no
    resolver configured returns the filed
    not-configured refusal as data — this
    module ships NO live Kista client and
    claims none. ``validate_config()``
    judges a config mapping by the pure
    validators and returns the problems as
    data.
  * ``kista_custody_problems()`` — a checker
    that re-derives every claim live: the
    filings, the neighbour checkers and the
    bindings' own credential filings, the
    shipped configs scanned live (no raw
    credential key or value anywhere under
    ``configs/``; the migrated references
    present and valid), a disabled custody
    refusing with nothing built, a fake-
    resolver resolution whose sentinel value
    provably never serialises, the filed
    probes, the docket header and row, and
    the filed doc.

**Docket**: TF-160 FIXES TF-016's D-CFG-4
(credential-shaped keys live in a plain YAML
file), its assigned share: the shipped
``configs/user_config.yaml`` no longer
carries the raw keys — they are custody
references now — and the custody module
governs every config in the tree (its
checker scans ``configs/`` live and refuses
the shape). TF-016 totals move in the open:
137 entries — 31 OPEN / 1 OPEN-ENVIRONMENT /
101 FIXED / 4 CLOSED-BY-VERDICT (D-CFG-4
OPEN -> FIXED), and every live pin of the
old totals in the tree's checkers, renders
and tests is followed to the new totals in
this same slice. Filed remainders, NOT
claimed fixed: the backend factory still
*reads* raw ``openai_api_key`` / ``hf_token``
keys when a user hand-writes them into a
local config (the validator refuses such a
config; rewiring the factory to resolve
through a live Kista is NOT PROVEN — no
live Kista serves this tree), and the setup
wizard still *writes* raw keys when a user
enters one at setup time (TF-171's security
audit and TF-177's upgrade path own those
two roads).

The human-readable doc is
``docs/architecture/KISTA_CUSTODY.md``,
filed verbatim as this module's render
(regenerate with
``python3 -m thoughtforge.kista_custody``).
Proof grade: FIXTURE — a fake resolver
shaped on the filed Kista data; NO live
Kista vault or CLI was called and NO Kista
or Hermes code was imported or copied. Live
end-to-end custody resolution is NOT PROVEN
— TF-170's Part IX gate owns the
at-least-one-live proof, never claimed
here. TF-161 owns binding fixtures in depth;
TF-162 owns versioning enforcement in depth;
TF-171 owns the security audit that takes
the wizard/factory remainders.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from thoughtforge.cognition.api_surface import (
    API_VERSION,
    verify_api_surface,
)
from thoughtforge.cognition_api import cognition_api_problems
from thoughtforge.ember_binding import (
    NOT_ROUTED_CALL_IDS as EMBER_NOT_ROUTED,
)
from thoughtforge.ember_binding import (
    ember_binding_problems,
)
from thoughtforge.hermes_binding import (
    NOT_ROUTED_CALL_IDS as HERMES_NOT_ROUTED,
)
from thoughtforge.hermes_binding import (
    hermes_binding_problems,
)
from thoughtforge.memory_api import memory_api_problems
from thoughtforge.runmal_binding import (
    CALL_ROWS as RUNMAL_CALL_ROWS,
)
from thoughtforge.runmal_binding import (
    FIELD_ROWS as RUNMAL_FIELD_ROWS,
)
from thoughtforge.runmal_binding import (
    runmal_binding_problems,
)
from thoughtforge.samtal_binding import (
    FIELD_ROWS as SAMTAL_FIELD_ROWS,
)
from thoughtforge.samtal_binding import (
    samtal_binding_problems,
)
from thoughtforge.service_shape import service_shape_problems

KISTA_CUSTODY_FORMAT = "thoughtforge-kista-custody/1"
KISTA_CUSTODY_DOC = "docs/architecture/KISTA_CUSTODY.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"
_CONFIGS_DIR = _REPO_ROOT / "configs"

#: The plugin that files the shape, as its
#: plugin.yaml names it (read READ-ONLY).
PLUGIN_NAME = "kista-secret-source"
PLUGIN_VERSION = "0.2.0"
PLUGIN_MANIFEST_VERSION = 2

#: The secret source's own filed identity
#: (source.py's KistaSource class fields).
SOURCE_NAME = "kista"
SOURCE_LABEL = "Kista"
SOURCE_SHAPE = "mapped"
SOURCE_SCHEME = "kista"

#: The reference scheme / prefix every
#: custody reference carries.
REFERENCE_SCHEME = "kista"
REFERENCE_PREFIX = "kista://"

#: The reference pattern, copied exactly
#: from the plugin's source.py
#: (_REFERENCE_RE) — service 1..128 chars,
#: field 1..64 chars, both lowercase
#: machine shapes.
REFERENCE_PATTERN = (
    r"^kista://(?P<service>[a-z0-9][a-z0-9_.-]{0,127})"
    r"/(?P<field>[a-z_][a-z0-9_]{0,63})$")

_REFERENCE_RE = re.compile(REFERENCE_PATTERN)

#: The vault's filed shape (source.py):
#: the default directory name under the
#: active profile and the two files an
#: initialized vault must hold.
VAULT_DIR_DEFAULT = "credentials"
VAULT_FILES: tuple[str, ...] = (".vault_key", "vault.json.enc")

#: The custody is OFF unless someone enables
#: it explicitly — filed as data, mirroring
#: the Kista source's own config default
#: (enabled: False).
DEFAULT_ENABLED = False

#: The refusal every resolution returns
#: while custody is disabled — as data,
#: naming the custody as disabled.
DISABLED_ERROR = (
    "kista custody is disabled: TF-160 "
    "custody is off by default — enable it "
    "explicitly (KistaCustody(enabled=True) "
    "or custody.enable()) before any "
    "reference resolves")

#: The refusal an enabled custody with no
#: resolver configured returns — as data,
#: naming that no live Kista is configured
#: here (never a fabricated value).
NOT_CONFIGURED_ERROR = (
    "kista custody is not configured: no "
    "resolver is injected into this custody "
    "— TF-160 files the custody shape and "
    "ships no live Kista client, so no "
    "reference can resolve here (live "
    "resolution is NOT PROVEN, TF-170 owns "
    "the live proof)")

#: The credential-shaped config keys TF-009
#: censused — the keys that must never carry
#: a value (or appear at all) in a config.
CREDENTIAL_CONFIG_KEYS: tuple[str, ...] = (
    "openai_api_key", "hf_token")

#: The custody-reference keys that replace
#: them in the shipped config, in the same
#: order.
CREDENTIAL_REF_KEYS: tuple[str, ...] = (
    "openai_api_key_ref", "hf_token_ref")

#: The shipped config file the credential
#: keys lived in (TF-009 §3).
USER_CONFIG_FILE = "configs/user_config.yaml"

#: The fields a resolution request may
#: carry — a reference, and optionally the
#: binding asking (identity only).
REQUEST_FIELDS: tuple[str, ...] = ("reference", "binding")


def parse_reference(reference: Any) -> tuple[str, str] | None:
    """Parse one custody reference into its
    (service, field) pair by the plugin's own
    pattern, or None when it is not a
    Kista-shaped reference. Pure."""
    if not isinstance(reference, str):
        return None
    match = _REFERENCE_RE.fullmatch(reference.strip())
    if match is None:
        return None
    return match.group("service"), match.group("field")


def reference_for_service_field(service: str, field: str) -> str:
    """Build the reference string for one
    (service, field) pair."""
    return f"{REFERENCE_PREFIX}{service}/{field}"


# ── The binding needs, as literal data ───────────────────────────────────────
#
# One row per consumer surface TF-153..
# TF-159 filed. Each row's claim is
# re-derived live by the checker from that
# surface's own filings — a binding that
# grew a credential route would fail here
# by name. The honest headline: no binding
# needs a credential today; the backend
# factory behind the shipped user config
# is the one credential consumer, and it
# is custody-referenced by this slice.

BINDING_NEED_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "need_runmal_binding",
     "consumer": "TF-156 Rúnmál binding",
     "needs_credential": False,
     "custody_reference": None,
     "filed_evidence": "TF-156's filings contain no "
                       "credential call or field row "
                       "at all — Rúnmál's backend "
                       "table names no key, and the "
                       "binding accepts none",
     "summary": "no credential needed: the Rúnmál "
                "binding routes cognition and memory "
                "only, and files no credential "
                "surface for custody to hold"},
    {"id": "need_samtal_binding",
     "consumer": "TF-157 Samtal binding",
     "needs_credential": False,
     "custody_reference": None,
     "filed_evidence": "TF-157 files the seat token "
                       "no-home (FIELD_ROWS seat.token, "
                       "disposition no_home_not_passed): "
                       "a Samtal seat token is Samtal's "
                       "own door key, never carried by "
                       "the binding, and the Astrid Law "
                       "forbids a seat carrying its own "
                       "credentials",
     "summary": "no credential needed: the Samtal "
                "binding holds no seat token and "
                "authenticates as no one"},
    {"id": "need_hermes_binding",
     "consumer": "TF-158 Hermes binding",
     "needs_credential": False,
     "custody_reference": None,
     "filed_evidence": "TF-158 files credential_custody "
                       "NOT routed: credentials live "
                       "in Hermes/Kista custody — the "
                       "profile credential file "
                       "(cognition_api_key_file, "
                       "default secrets/aesir.key) and "
                       "the kista-secret-source "
                       "plugin's kista:// bindings — "
                       "and the binding's request "
                       "validator refuses any key",
     "summary": "no credential needed on the "
                "ThoughtForge side: Hermes-side "
                "custody is Kista's own shape, the "
                "one this slice mirrors"},
    {"id": "need_ember_binding",
     "consumer": "TF-159 Ember binding",
     "needs_credential": False,
     "custody_reference": None,
     "filed_evidence": "TF-159 files credential_custody "
                       "NOT routed with E-024's own "
                       "truth: no credential is "
                       "involved in that binding at "
                       "all, and its request validator "
                       "refuses a credential field by "
                       "name",
     "summary": "no credential needed: the Ember "
                "binding involves no credential at "
                "all"},
    {"id": "need_service_shape",
     "consumer": "TF-153 service shape",
     "needs_credential": False,
     "custody_reference": None,
     "filed_evidence": "TF-153's contract files three "
                       "ops (health / contract / turn) "
                       "and no credential field "
                       "anywhere — the service binds "
                       "loopback by default and "
                       "authenticates no one",
     "summary": "no credential needed: the local "
                "service is loopback-only and "
                "credential-free by shape"},
    {"id": "need_backend_factory",
     "consumer": "backend factory (inference/"
                 "unified_backend.py) behind "
                 "configs/user_config.yaml",
     "needs_credential": True,
     "custody_reference": "kista://thoughtforge/"
                          "openai_api_key + "
                          "kista://thoughtforge/hf_token",
     "filed_evidence": "TF-009 §3 + TF-016 D-CFG-4: "
                       "the factory's "
                       "load_backend_from_config() "
                       "reads openai_api_key and "
                       "hf_token from the plain YAML "
                       "user config — the tree's one "
                       "live credential consumption; "
                       "this slice migrates the "
                       "shipped file to custody "
                       "references (CONFIG_KEY_ROWS) "
                       "and the validator refuses "
                       "the raw shape",
     "summary": "the one credential consumer: its "
                "two keys are custody-referenced in "
                "the shipped config, never values "
                "in a config again"},
)

BINDING_NEED_IDS: tuple[str, ...] = tuple(
    r["id"] for r in BINDING_NEED_ROWS)


def binding_need_row_for(
    consumer_needle: str,
) -> dict[str, Any] | None:
    """The filed need row whose consumer names
    the needle, or None."""
    for row in BINDING_NEED_ROWS:
        if consumer_needle in row["consumer"]:
            return row
    return None


# ── The Kista shape, as literal data ─────────────────────────────────────────
#
# Filed from the kista-secret-source plugin
# source read READ-ONLY in the Hermes tree
# (plugin.yaml / __init__.py / source.py).
# No Kista or Hermes code is imported or
# copied: the pattern is filed as a pattern,
# the shapes as data.

CUSTODY_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "custody_plugin",
     "aspect": "the plugin",
     "shape": "plugins/kista-secret-source — "
              "plugin.yaml (manifest_version 2, "
              "name kista-secret-source, version "
              "0.2.0, 'Profile-local Kista "
              "credentials through Hermes' Secret "
              "Source API', homepage "
              "github.com/hrabanazviking/kista); "
              "__init__.py's register(ctx) calls "
              "exactly ctx.register_secret_source"
              "(KistaSource()) — no tools, hooks "
              "or prompt-visible surfaces",
     "verified": "read READ-ONLY from the Hermes "
                 "tree by this slice (plugin.yaml "
                 "+ __init__.py + source.py)",
     "summary": "custody is a secret-source "
                "plugin behind Hermes' Secret "
                "Source API — a registration, not "
                "a tool a model can call"},
    {"id": "custody_reference_form",
     "aspect": "the reference form",
     "shape": "a binding is ENV_VAR -> "
              "kista://service/field, judged by "
              "the plugin's own pattern "
              "^kista://(?P<service>[a-z0-9]"
              "[a-z0-9_.-]{0,127})/(?P<field>"
              "[a-z_][a-z0-9_]{0,63})$ — the "
              "reference is filed in config; the "
              "value never is; validation warns "
              "by env-var name only, 'without "
              "echoing refs'",
     "verified": "the pattern is copied exactly "
                 "from source.py's _REFERENCE_RE "
                 "and _validate_references; "
                 "parse_reference() in this module "
                 "judges by the same pattern",
     "summary": "the only credential-shaped "
                "thing a config may hold is a "
                "kista:// service/field reference"},
    {"id": "custody_vault",
     "aspect": "the vault",
     "shape": "a profile-local directory "
              "(config vault_dir, default "
              "'credentials', relative to the "
              "active profile and refused if it "
              "escapes it) holding .vault_key "
              "and vault.json.enc; before any "
              "read the plugin proves the "
              "directory and both files are "
              "real (no links / reparse points), "
              "owned by the current user, and "
              "carry no group/world permission "
              "bits (POSIX) or a private ACL "
              "(Windows) — an unsafe vault is a "
              "refusal, never a read",
     "verified": "read from source.py's "
                 "_resolve_vault_dir / "
                 "_validate_vault_security / "
                 "_REQUIRED_VAULT_FILES",
     "summary": "the secret at rest is an "
                "encrypted vault private to its "
                "owner — never a config file"},
    {"id": "custody_cli_resolution",
     "aspect": "the resolution",
     "shape": "the official Kista CLI is "
              "invoked as 'kista get -- "
              "<service>' with KISTA_DIR set to "
              "the vault directory; the entry "
              "returns as JSON and the binding's "
              "field is read from it; failures "
              "are classified REF_INVALID / "
              "NOT_CONFIGURED / AUTH_FAILED / "
              "TIMEOUT / BINARY_MISSING / "
              "INTERNAL and a failure carries "
              "no CLI output and no reference "
              "echo; CLI output is capped at "
              "256 KiB",
     "verified": "read from source.py's "
                 "_read_entry / _failure_kind / "
                 "KistaSource.fetch",
     "summary": "resolution is a local CLI "
                "read of the vault — the value "
                "moves from vault to caller in "
                "process, never into a file, a "
                "log or an error"},
    {"id": "custody_source_config",
     "aspect": "the source config",
     "shape": "KistaSource (SecretSource ABC): "
              "name 'kista', label 'Kista', "
              "shape 'mapped', scheme 'kista', "
              "override_existing_default False; "
              "config schema — enabled (default "
              "False), env (ENV_VAR -> reference "
              "map, default {}), binary_path "
              "(pinned Kista executable, "
              "default ''), vault_dir (default "
              "'credentials'), "
              "cli_timeout_seconds (default "
              "30), override_existing (default "
              "False)",
     "verified": "read from source.py's "
                 "KistaSource class fields and "
                 "config_schema()",
     "summary": "custody is off by default and "
                "explicitly mapped — a value "
                "resolves only for a binding "
                "someone filed, never by "
                "convention or discovery"},
    {"id": "custody_hermes_profile_key_file",
     "aspect": "the Hermes profile key file",
     "shape": "volmarr-core reads its cognition "
              "endpoint key from a profile "
              "credential file "
              "(cognition_api_key_file, default "
              "secrets/aesir.key) — a file on "
              "Hermes' side, under the profile, "
              "never a value in a request and "
              "never read by TF-158's binding",
     "verified": "filed by TF-158 from the "
                 "Hermes tree (its "
                 "credential_custody row) and "
                 "consumed here as TF-158's "
                 "filing, not re-read",
     "summary": "even Hermes' file-shaped "
                "custody keeps the key out of "
                "configs that travel and out of "
                "requests — Kista is its "
                "reference-shaped successor"},
)

CUSTODY_ROW_IDS: tuple[str, ...] = tuple(
    r["id"] for r in CUSTODY_ROWS)


# ── The config keys, as literal data ─────────────────────────────────────────
#
# The two credential-shaped keys TF-009
# censused in configs/user_config.yaml.
# This slice migrates the shipped file: the
# raw keys are gone; the reference keys
# below stand in their place. The checker
# re-derives the migration live from the
# file on disk.

CONFIG_KEY_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "config_openai_api_key",
     "raw_key": "openai_api_key",
     "ref_key": "openai_api_key_ref",
     "reference": "kista://thoughtforge/openai_api_key",
     "config_file": USER_CONFIG_FILE,
     "consumer": "the LM-Studio / "
                 "OpenAI-compatible branch of "
                 "load_backend_from_config() "
                 "(inference/unified_backend.py)",
     "summary": "the OpenAI-compatible "
                "endpoint key: a custody "
                "reference in the shipped "
                "config, never a value"},
    {"id": "config_hf_token",
     "raw_key": "hf_token",
     "ref_key": "hf_token_ref",
     "reference": "kista://thoughtforge/hf_token",
     "config_file": USER_CONFIG_FILE,
     "consumer": "the HuggingFace branch of "
                 "load_backend_from_config() "
                 "(inference/unified_backend.py)",
     "summary": "the HuggingFace token: a "
                "custody reference in the "
                "shipped config, never a value"},
)

CONFIG_KEY_IDS: tuple[str, ...] = tuple(
    r["id"] for r in CONFIG_KEY_ROWS)


def config_key_row_for(raw_key: str) -> dict[str, Any] | None:
    """The filed config-key row for one raw
    credential key, or None."""
    for row in CONFIG_KEY_ROWS:
        if row["raw_key"] == raw_key:
            return row
    return None


# ── The field mapping, as literal data ───────────────────────────────────────

FIELD_ROWS: tuple[dict[str, Any], ...] = (
    # ── In: resolution request -> the resolver ──
    {"direction": "in",
     "id": "request.reference",
     "source": "the request's reference — a "
               "kista://service/field string, "
               "the only credential-shaped "
               "input custody accepts",
     "target": "the resolver's (service, field) "
               "pair, parsed by the plugin's own "
               "pattern",
     "disposition": "mapped",
     "summary": "a reference is a pointer, not "
                "a secret: it may be filed in "
                "configs and named in responses; "
                "an invalid reference is an "
                "invalid request, named by "
                "reference_problems()"},
    {"direction": "in",
     "id": "request.binding",
     "source": "the request's optional binding "
               "name — which binding asks",
     "target": "the response envelope's "
               "binding; no resolver input",
     "disposition": "identity_only_not_passed",
     "summary": "the asking binding is "
                "provenance for the envelope "
                "only — the Kista shape resolves "
                "by (service, field), never by "
                "who asks"},
    {"direction": "in",
     "id": "request.secret_value",
     "source": "nothing: a resolution request "
               "never carries a value — custody "
               "exists so callers hold "
               "references, not values",
     "target": "no target — never filed, "
               "never passed, never stored",
     "disposition": "no_home_not_passed",
     "summary": "a request carrying a value-"
                "shaped field (value / secret / "
                "api_key / token or any unfiled "
                "field) is refused by the "
                "validator, named"},
    {"direction": "in",
     "id": "config.raw_credential_key",
     "source": "a config mapping's raw "
               "credential-shaped key "
               "(openai_api_key / hf_token) — "
               "TF-009's censused shape",
     "target": "no target — refused by "
               "config_mapping_problems()",
     "disposition": "no_home_not_passed",
     "summary": "present at all, the raw key "
                "is the defect shape (D-CFG-4): "
                "empty it is inert but wrongly "
                "shaped, non-empty it is a "
                "secret in a plain file — both "
                "are named; the config must "
                "carry the *_ref custody "
                "reference instead"},
    # ── Out: the resolution -> the response envelope ──
    {"direction": "out",
     "id": "response.reference",
     "source": "the request's reference",
     "target": "the response's reference",
     "disposition": "mapped",
     "summary": "the pointer rides the "
                "envelope so a caller can see "
                "WHAT resolved — a pointer is "
                "not the secret"},
    {"direction": "out",
     "id": "response.service_field",
     "source": "the parsed reference",
     "target": "the response's service and "
               "field",
     "disposition": "mapped",
     "summary": "the parsed pair rides the "
                "envelope (names, never values)"},
    {"direction": "out",
     "id": "response.value",
     "source": "the resolver's returned value",
     "target": "NO response field — the value "
               "is delivered only to the "
               "caller-supplied consumer "
               "callable, in process",
     "disposition": "no_home_never_filed",
     "summary": "the resolved value has no "
                "home in any envelope, filing, "
                "trace or log this module "
                "produces — proven with a "
                "sentinel by the probes; a "
                "resolution without a consumer "
                "discards the value deliberately"},
    {"direction": "out",
     "id": "response.consumer_result",
     "source": "the consumer callable's return",
     "target": "the response's consumer_result",
     "disposition": "binding_filed",
     "summary": "what the consumer returns "
                "rides the envelope as detached "
                "JSON-safe data — the consumer "
                "decides what to report (a "
                "length, a backend object id, a "
                "verdict), and custody never "
                "inspects it for the value"},
)

FIELD_IDS: tuple[str, ...] = tuple(
    f["id"] for f in FIELD_ROWS)


def field_rows_for(direction: str) -> tuple[dict[str, Any], ...]:
    """One direction's filed field rows, in
    filing order."""
    return tuple(f for f in FIELD_ROWS
                 if f["direction"] == direction)


# ── The rules, as literal data ───────────────────────────────────────────────

RULE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "never_in_configs",
     "rule": "a credential value never lives in "
             "a config: the only credential-"
             "shaped thing a config may carry is "
             "a kista:// service/field reference "
             "under a *_ref key; a raw "
             "credential-shaped key in a config "
             "mapping is refused by "
             "config_mapping_problems() — empty "
             "(inert but wrongly shaped) or "
             "non-empty (a secret in a plain "
             "file) alike, D-CFG-4's fix"},
    {"id": "reference_never_value",
     "rule": "custody trades in references: "
             "parse_reference() judges a "
             "reference by the Kista plugin's own "
             "pattern exactly (service 1..128 "
             "lowercase machine chars, field "
             "1..64), and nothing in this module "
             "files, logs or returns a credential "
             "value — values exist only as a "
             "resolver's return, in process, on "
             "their way to a consumer"},
    {"id": "resolved_value_never_serialised",
     "rule": "a resolved value is never "
             "serialised: the resolution envelope "
             "files no value field, the custody "
             "object stores no value, and the "
             "contract / as-data outputs contain "
             "none — the value reaches only the "
             "caller-supplied consumer callable; "
             "proven by the probes with a "
             "sentinel secret that appears in no "
             "serialised output"},
    {"id": "off_by_default",
     "rule": "custody is off by default: "
             "DEFAULT_ENABLED is False, filed as "
             "data (mirroring the Kista source's "
             "own enabled default False); "
             "enabling is an explicit act — the "
             "constructor flag enabled=True or "
             "the enable() method — and "
             "disable() returns custody to "
             "refusal; while disabled, every "
             "resolution returns a truthful "
             "refusal as data and touches no "
             "resolver — no resolver is built "
             "and no reference is parsed for "
             "resolution, proven by the probes "
             "with a counting factory"},
    {"id": "resolution_only_through_injected_resolver",
     "rule": "resolution happens only through "
             "an injected resolver callable "
             "(service, field) -> value: this "
             "module ships no Kista client, "
             "invokes no CLI, opens no vault "
             "and reads no key file — an enabled "
             "custody with no resolver returns "
             "the filed not-configured refusal; "
             "a live resolver is TF-170's live "
             "proof to wire, never a silent "
             "default here"},
    {"id": "failures_name_a_kind_never_a_value",
     "rule": "a failed resolution is data "
             "naming its kind (disabled / not "
             "configured / resolver refused / "
             "resolver returned nothing) — "
             "never the resolver's message, "
             "never a value, never CLI output: "
             "the Kista plugin's own discipline "
             "(failures carry no output and no "
             "reference echo beyond the "
             "caller's own reference)"},
    {"id": "no_binding_needs_a_credential_today",
     "rule": "the filed truth of this slice: "
             "no TF-153..TF-159 binding routes "
             "or needs a credential — each "
             "binding's credential custody is "
             "NOT routed / no-home in its own "
             "filings, re-derived live by the "
             "checker — so custody here is "
             "governance and shape, not a route: "
             "any future binding that needs a "
             "credential must file its need here "
             "and resolve it through this "
             "custody, never through a config"},
    {"id": "called_never_reimplemented",
     "rule": "no Kista or Hermes code is "
             "imported or copied: the plugin's "
             "shapes are filed as literal data "
             "read READ-ONLY from the Hermes "
             "tree (plugins/kista-secret-source's "
             "plugin.yaml / __init__.py / "
             "source.py) and nothing under that "
             "tree is edited by this slice; the "
             "pattern is filed as a pattern and "
             "judged by Python's re, not by "
             "Kista's code"},
    {"id": "every_field_has_a_disposition",
     "rule": "every field on either side has a "
             "FIELD_ROWS row with a disposition "
             "— none is silently dropped and "
             "none is invented; the resolved "
             "value is filed no-home in the open"},
    {"id": "responses_detached_json_safe_versioned",
     "rule": "every response is detached "
             "JSON-safe data (a JSON round-trip "
             "copy) whose contract_version is "
             "TF-034's API_VERSION, consumed "
             "from thoughtforge.cognition."
             "api_surface — no competing "
             "version is invented here"},
)

#: TF-159-style name for the custody rules.
KISTA_CUSTODY_RULES = RULE_ROWS

RULE_IDS: tuple[str, ...] = tuple(
    r["id"] for r in RULE_ROWS)


# ── The contract as plain data ───────────────────────────────────────────────

def kista_custody_contract() -> dict[str, Any]:
    """The Kista custody contract as plain
    JSON-safe data, built from the filed
    literals above and TF-034's version. A
    fresh copy each call. Contains no
    credential value — by construction it
    cannot: no filing holds one."""
    return {
        "format": KISTA_CUSTODY_FORMAT,
        "contract_version": API_VERSION,
        "plugin_name": PLUGIN_NAME,
        "plugin_version": PLUGIN_VERSION,
        "source_name": SOURCE_NAME,
        "reference_scheme": REFERENCE_SCHEME,
        "reference_pattern": REFERENCE_PATTERN,
        "default_enabled": DEFAULT_ENABLED,
        "disabled_error": DISABLED_ERROR,
        "not_configured_error": NOT_CONFIGURED_ERROR,
        "credential_config_keys": list(CREDENTIAL_CONFIG_KEYS),
        "credential_ref_keys": list(CREDENTIAL_REF_KEYS),
        "binding_needs": [dict(r) for r in BINDING_NEED_ROWS],
        "custody_rows": [dict(r) for r in CUSTODY_ROWS],
        "config_keys": [dict(r) for r in CONFIG_KEY_ROWS],
        "fields": [dict(f) for f in FIELD_ROWS],
        "rules": [dict(r) for r in RULE_ROWS],
    }


# ── Pure validators ──────────────────────────────────────────────────────────

def reference_problems(reference: Any) -> list[str]:
    """Judge one custody reference, purely.
    Empty-safe."""
    if not isinstance(reference, str) or not reference:
        return ["reference: the reference is empty "
                "or not a string — a custody "
                "reference is a kista://service/field "
                "string"]
    if parse_reference(reference) is None:
        return [f"reference: {reference!r} is not a "
                "Kista-shaped reference — the filed "
                "form is kista://<service>/<field> "
                "with service [a-z0-9][a-z0-9_.-]"
                "{0,127} and field [a-z_][a-z0-9_]"
                "{0,63} (the kista-secret-source "
                "plugin's own pattern)"]
    return []


def binding_need_row_problems(row: Any) -> list[str]:
    """Judge one binding-need filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["binding need row: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("binding need row: the row files no id")
    for key in ("consumer", "filed_evidence", "summary"):
        if not row.get(key):
            problems.append(
                f"binding need row {row.get('id')}: no {key} filed")
    if not isinstance(row.get("needs_credential"), bool):
        problems.append(
            f"binding need row {row.get('id')}: "
            "needs_credential is not a boolean")
    elif row["needs_credential"] and not row.get("custody_reference"):
        problems.append(
            f"binding need row {row.get('id')}: needs a "
            "credential but files no custody reference")
    elif not row["needs_credential"] \
            and row.get("custody_reference") is not None:
        problems.append(
            f"binding need row {row.get('id')}: needs no "
            "credential but files a custody reference")
    return problems


def custody_row_problems(row: Any) -> list[str]:
    """Judge one Kista-shape filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["custody row: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("custody row: the row files no id")
    for key in ("aspect", "shape", "verified", "summary"):
        if not row.get(key):
            problems.append(
                f"custody row {row.get('id')}: no {key} filed")
    return problems


def config_key_row_problems(row: Any) -> list[str]:
    """Judge one config-key filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["config key row: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("config key row: the row files no id")
    for key in ("raw_key", "ref_key", "reference",
                "config_file", "consumer", "summary"):
        if not row.get(key):
            problems.append(
                f"config key row {row.get('id')}: no {key} filed")
    if row.get("raw_key") not in CREDENTIAL_CONFIG_KEYS:
        problems.append(
            f"config key row {row.get('id')}: raw_key is "
            f"{row.get('raw_key')!r}, not one of TF-009's "
            f"credential-shaped keys {list(CREDENTIAL_CONFIG_KEYS)}")
    if row.get("ref_key") != f"{row.get('raw_key')}_ref":
        problems.append(
            f"config key row {row.get('id')}: ref_key is "
            f"{row.get('ref_key')!r}, not the raw key + _ref")
    problems.extend(
        f"config key row {row.get('id')}: {p}"
        for p in reference_problems(row.get("reference")))
    parsed = parse_reference(row.get("reference"))
    if parsed is not None and parsed[1] != row.get("raw_key"):
        problems.append(
            f"config key row {row.get('id')}: the "
            f"reference's field is {parsed[1]!r}, not the "
            f"raw key {row.get('raw_key')!r} it custodies")
    return problems


def field_row_problems(row: Any) -> list[str]:
    """Judge one field filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["field: the row is empty"]
    problems: list[str] = []
    if row.get("direction") not in ("in", "out"):
        problems.append(
            f"field {row.get('id')}: direction is "
            f"{row.get('direction')!r}, not in / out")
    for key in ("id", "source", "target", "disposition",
                "summary"):
        if not row.get(key):
            problems.append(
                f"field {row.get('id')}: no {key} filed")
    return problems


def rule_row_problems(row: Any) -> list[str]:
    """Judge one rule filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["rule: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("rule: the row files no id")
    if not row.get("rule"):
        problems.append(f"rule {row.get('id')}: no rule filed")
    return problems


def _flatten_config(mapping: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a parsed config mapping to leaf
    keys (section.key for nested sections).
    Pure."""
    flat: dict[str, Any] = {}
    if not isinstance(mapping, dict):
        return flat
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_config(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def config_mapping_problems(mapping: Any) -> list[str]:
    """Judge one config mapping against the
    custody law, purely. Empty-safe.

    The law: a raw credential-shaped key
    (openai_api_key / hf_token, at any nesting
    depth) must not appear at all — its *_ref
    custody reference stands in its place, and
    that reference must be Kista-shaped. Any
    other value that *looks* like a filed
    kista:// reference must parse."""
    if not isinstance(mapping, dict):
        return ["config: the config is not a mapping — "
                "a config is judged as a parsed mapping"]
    problems: list[str] = []
    flat = _flatten_config(mapping)
    for name, value in flat.items():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in CREDENTIAL_CONFIG_KEYS:
            row = config_key_row_for(leaf)
            ref_key = row["ref_key"] if row else f"{leaf}_ref"
            if value:
                problems.append(
                    f"config: {name} carries a raw "
                    "credential value in a plain config "
                    "— credentials live in Kista "
                    "custody, never in configs "
                    f"(D-CFG-4); file {ref_key} with a "
                    "kista:// reference instead")
            else:
                problems.append(
                    f"config: {name} is a raw "
                    "credential-shaped key in a plain "
                    "config — empty, so no secret is "
                    "carried, but the shape is the "
                    "defect (D-CFG-4); file "
                    f"{ref_key} with a kista:// "
                    "reference instead")
        elif leaf in CREDENTIAL_REF_KEYS:
            problems.extend(
                f"config: {name}: {p}"
                for p in reference_problems(value))
        elif isinstance(value, str) \
                and value.startswith(REFERENCE_PREFIX):
            problems.extend(
                f"config: {name}: {p}"
                for p in reference_problems(value))
    return problems


def resolve_request_problems(request: Any) -> list[str]:
    """Judge one resolution request, purely.
    Empty-safe.

    Shape only: exactly reference / binding;
    the reference Kista-shaped; above all, no
    value-shaped field is ever a request
    field."""
    problems: list[str] = []
    if not isinstance(request, dict) or not request:
        return ["resolve request: request must be a "
                "mapping shaped like {reference, "
                "binding} — a custody reference and "
                "optionally the asking binding"]
    for key in request:
        if key not in REQUEST_FIELDS:
            problems.append(
                f"resolve request: request files "
                f"{key!r}, not a filed custody request "
                f"field {list(REQUEST_FIELDS)} — above "
                "all, no credential value is ever a "
                "request field here")
    problems.extend(
        f"resolve request: {p}"
        for p in reference_problems(request.get("reference")))
    if "binding" in request \
            and not isinstance(request["binding"], str):
        problems.append(
            "resolve request: binding must be a string "
            "when present (the asking binding's name)")
    return problems


def custody_response_problems(row: Any) -> list[str]:
    """Judge one custody response dict against
    the envelope, purely. Empty-safe. The
    envelope files no value field, ever."""
    if not isinstance(row, dict) or not row:
        return ["response: the row is empty"]
    problems: list[str] = []
    for field in ("call", "ok", "status",
                  "contract_version", "error"):
        if field not in row:
            problems.append(f"response: no {field} filed")
    if problems:
        return problems
    if row.get("call") not in ("resolve", "validate_config"):
        problems.append(
            f"response: call is {row.get('call')!r}, not "
            "one of the custody calls (resolve / "
            "validate_config)")
    if "value" in row or "secret" in row:
        problems.append(
            "response: a value/secret field is filed — "
            "a resolved value has NO response field, "
            "ever (it reaches only the consumer)")
    if not isinstance(row.get("ok"), bool):
        problems.append("response: ok is not a boolean")
    if not isinstance(row.get("status"), str) \
            or not row.get("status"):
        problems.append("response: status is not a non-empty string")
    if row.get("contract_version") != API_VERSION:
        problems.append(
            f"response: contract_version is "
            f"{row.get('contract_version')!r}, not "
            f"TF-034's {API_VERSION!r}")
    if row.get("ok") is True and row.get("error") is not None:
        problems.append(
            "response: ok is True but an error is filed — "
            "a success files error None")
    if row.get("ok") is False:
        if row.get("status") != "refused":
            problems.append(
                "response: ok is False but status is "
                f"{row.get('status')!r}, not \"refused\"")
        if not isinstance(row.get("error"), str) \
                or not row.get("error"):
            problems.append(
                "response: ok is False but no error names "
                "the refusal")
    if row.get("call") == "resolve" and row.get("ok") is True:
        problems.extend(
            f"response: {p}"
            for p in reference_problems(row.get("reference")))
    try:
        json.dumps(row)
    except (TypeError, ValueError) as exc:
        problems.append(f"response: not JSON-safe: {exc}")
    return problems


# ── The shipped-config scan (live) ───────────────────────────────────────────

def _load_yaml_file(path: Path) -> Any:
    """Parse one YAML file with the tree's own
    YAML (pyyaml is a shipped dependency —
    utils/config.py loads these same files)."""
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def shipped_config_problems() -> list[str]:
    """Judge the shipped configs on disk against
    the custody law, live: every YAML file
    under configs/ parsed and judged by
    config_mapping_problems(), and the user
    config's migration checked key by key."""
    problems: list[str] = []
    if not _CONFIGS_DIR.is_dir():
        return [f"configs: {_CONFIGS_DIR} is missing"]
    files = sorted(_CONFIGS_DIR.glob("*.yaml")) \
        + sorted(_CONFIGS_DIR.glob("*.yml"))
    if not files:
        return ["configs: no YAML config files found"]
    for path in files:
        try:
            parsed = _load_yaml_file(path)
        except Exception as exc:
            problems.append(
                f"configs: {path.name} does not parse: {exc}")
            continue
        problems.extend(
            f"configs/{path.name}: {p}"
            for p in config_mapping_problems(parsed or {}))
    user_config = _CONFIGS_DIR / "user_config.yaml"
    try:
        parsed = _load_yaml_file(user_config) or {}
    except Exception as exc:
        problems.append(f"configs: user_config.yaml: {exc}")
        parsed = {}
    for row in CONFIG_KEY_ROWS:
        if row["raw_key"] in parsed:
            problems.append(
                f"configs: user_config.yaml still carries "
                f"the raw key {row['raw_key']} — D-CFG-4 "
                "is not migrated")
        if parsed.get(row["ref_key"]) != row["reference"]:
            problems.append(
                f"configs: user_config.yaml files "
                f"{row['ref_key']} = "
                f"{parsed.get(row['ref_key'])!r}, not the "
                f"filed custody reference "
                f"{row['reference']!r}")
    return problems


# ── The custody handle ───────────────────────────────────────────────────────

def _detached(data: Any) -> Any:
    """A detached JSON-safe copy (a JSON round-trip)."""
    return json.loads(json.dumps(data))


class KistaCustody:
    """Kista-shaped credential custody for the
    bindings, off by default, values never
    serialised.

    Wraps an injectable resolver callable
    ``resolver(service, field) -> str``: pass
    ``resolver`` to inject one, or
    ``resolver_factory`` to build one lazily.
    Constructing the custody builds no
    resolver. This module ships NO live Kista
    client: a live resolver (the Kista CLI
    against a real vault, TF-170's proof) is
    injected by its owner, never defaulted.

    The custody is DISABLED unless ``enabled``
    is True at construction or ``enable()``
    is called (mirroring the Kista source's
    own enabled default False): while
    disabled every resolution returns the
    filed refusal as data and touches no
    resolver. An enabled custody with no
    resolver returns the filed not-configured
    refusal. A resolved value is delivered
    ONLY to the caller-supplied ``consumer``
    callable, in process; it is never
    stored on this object, never filed in
    the response envelope, and never logged.
    """

    def __init__(
        self,
        resolver: Callable[[str, str], Any] | None = None,
        *,
        enabled: bool = DEFAULT_ENABLED,
        resolver_factory: Callable[
            [], Callable[[str, str], Any]] | None = None,
    ) -> None:
        self._resolver = resolver
        self._resolver_factory = resolver_factory
        self._enabled = bool(enabled)

    # ── The explicit enable act ──

    @property
    def enabled(self) -> bool:
        """Whether custody currently resolves."""
        return self._enabled

    def enable(self) -> None:
        """Enable resolution — the explicit act
        the off-by-default law requires."""
        self._enabled = True

    def disable(self) -> None:
        """Return custody to refusal."""
        self._enabled = False

    # ── Lazy resolver construction ──

    def _get_resolver(self) -> Callable[[str, str], Any] | None:
        if self._resolver is None \
                and self._resolver_factory is not None:
            self._resolver = self._resolver_factory()
        return self._resolver

    # ── Envelopes ──

    def _envelope(
        self, call: str, ok: bool, status: str, **fields: Any,
    ) -> dict[str, Any]:
        return _detached({
            "call": call, "ok": ok, "status": status,
            "contract_version": API_VERSION,
            "error": None if ok else fields.pop(
                "error", ""),
            **fields,
        })

    def _refused(self, call: str, error: str) -> dict[str, Any]:
        return self._envelope(call, False, "refused", error=error)

    # ── The custody calls ──

    def resolve(
        self,
        reference: str,
        consumer: Callable[[Any], Any] | None = None,
        binding: str | None = None,
    ) -> dict[str, Any]:
        """Resolve one custody reference: the
        value is delivered only to ``consumer``
        (in process) and the envelope files
        the reference, the parsed (service,
        field) and the consumer's own result —
        never the value.

        While disabled, the filed refusal
        returns as data and nothing is judged,
        built or resolved. When enabled, an
        invalid request raises ValueError
        naming the problem(s), judged by
        resolve_request_problems() before any
        resolver is built or called. A
        resolver that raises or returns
        nothing surfaces as a refusal as data
        naming the kind only — never the
        resolver's message, never a value.
        A resolution with no consumer discards
        the value deliberately (it is resolved
        only to prove the binding holds)."""
        if not self._enabled:
            return self._refused("resolve", DISABLED_ERROR)
        request: dict[str, Any] = {"reference": reference}
        if binding is not None:
            request["binding"] = binding
        problems = resolve_request_problems(request)
        if problems:
            raise ValueError("; ".join(problems))
        parsed = parse_reference(reference)
        assert parsed is not None  # judged clean above
        service, field = parsed
        resolver = self._get_resolver()
        if resolver is None:
            return self._refused(
                "resolve", NOT_CONFIGURED_ERROR)
        try:
            value = resolver(service, field)
        except Exception:
            return self._envelope(
                "resolve", False, "refused",
                error="kista custody refused: the "
                      "resolver failed for this "
                      "reference (kind: resolver "
                      "error — the failure carries "
                      "no value and no resolver "
                      "message, by the filed "
                      "discipline)",
                reference=reference, service=service,
                field=field, binding=binding)
        if value is None or value == "":
            return self._envelope(
                "resolve", False, "refused",
                error="kista custody refused: the "
                      "resolver returned no value "
                      "for this reference (kind: "
                      "empty — the binding does "
                      "not hold in this custody)",
                reference=reference, service=service,
                field=field, binding=binding)
        consumer_result: Any = None
        if consumer is not None:
            consumer_result = consumer(value)
        # The value goes no further: it is not
        # stored, not filed, not returned.
        return self._envelope(
            "resolve", True, "resolved",
            reference=reference, service=service,
            field=field, binding=binding,
            consumer_result=consumer_result)

    def validate_config(
        self,
        mapping: Any,
    ) -> dict[str, Any]:
        """Judge one config mapping against the
        custody law and return the verdict as
        data: ok True exactly when
        config_mapping_problems() is empty.
        Available while disabled too — judging
        a config resolves nothing."""
        problems = config_mapping_problems(mapping)
        return self._envelope(
            "validate_config", not problems,
            "valid" if not problems else "refused",
            error=None if not problems else problems[0],
            problems=problems)

    def reference_for(self, raw_key: str) -> str | None:
        """The filed custody reference for one
        raw credential config key, or None
        when the key is not a filed
        credential-shaped key."""
        row = config_key_row_for(raw_key)
        return row["reference"] if row else None


# ── Fixtures (checker + probes) ──────────────────────────────────────────────

#: The sentinel a fake resolver returns — the
#: probes prove it appears in NO serialised
#: output (envelope, contract, as-data, or
#: the custody object's own state).
SENTINEL_SECRET = "tf160-sentinel-secret-value-7f3a"


class _FakeResolver:
    """A recording fake resolver shaped on the
    filed Kista data: answers the sentinel for
    the thoughtforge service, nothing for any
    other. Fixture only — no vault, no CLI."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, service: str, field: str) -> Any:
        self.calls.append((service, field))
        if service == "thoughtforge":
            return SENTINEL_SECRET
        return None


def _serialised_outputs(custody: KistaCustody,
                        envelope: dict[str, Any]) -> str:
    """Every serialised surface the opacity
    probe judges, as one string."""
    return json.dumps({
        "envelope": envelope,
        "contract": kista_custody_contract(),
        "custody_state": {
            k: str(v) for k, v in vars(custody).items()},
    })


# ── The filed probes ─────────────────────────────────────────────────────────

KISTA_CUSTODY_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "disabled_refusal_touches_nothing",
     "kind": "disabled_refusal",
     "detail": "a default custody refuses every "
               "resolution as data (ok=False, the "
               "error naming custody disabled), and "
               "a counting resolver factory proves "
               "no resolver is ever built; "
               "validate_config still judges while "
               "disabled (it resolves nothing)"},
    {"id": "reference_shape_is_kistas_own",
     "kind": "reference_shape",
     "detail": "references parse by the "
               "kista-secret-source plugin's own "
               "pattern exactly: the two filed "
               "config references parse to "
               "(thoughtforge, <key>); an http URL, "
               "an uppercase service, a missing "
               "field, an empty field and a bare "
               "value are each refused by name"},
    {"id": "resolved_value_never_serialised",
     "kind": "opacity",
     "detail": "an enabled custody over a fake "
               "resolver resolves the filed "
               "reference: the consumer receives "
               "the sentinel value in process, "
               "and the sentinel appears in NO "
               "serialised output — not the "
               "envelope, not the contract, not "
               "the custody object's state; the "
               "envelope files no value field"},
    {"id": "configs_on_disk_carry_references_not_values",
     "kind": "shipped_configs",
     "detail": "the shipped configs are scanned "
               "live: no file under configs/ "
               "carries a raw credential key or "
               "value, and user_config.yaml "
               "carries exactly the two filed "
               "*_ref custody references"},
    {"id": "raw_config_values_refused_by_validator",
     "kind": "config_validator",
     "detail": "config_mapping_problems() names a "
               "non-empty raw credential value, "
               "names an empty raw key as the "
               "defect shape, accepts the migrated "
               "reference config, and names a "
               "malformed custody reference — "
               "including at nesting depth"},
    {"id": "bindings_need_no_credential_today",
     "kind": "binding_needs",
     "detail": "the filed needs are re-derived "
               "live from TF-156..TF-159's own "
               "filings: hermes and ember file "
               "credential_custody NOT routed, "
               "samtal files seat.token no-home, "
               "runmal files no credential surface "
               "— only the backend factory row "
               "needs a credential, custody-"
               "referenced"},
    {"id": "doc_matches_render_and_version_is_tf034",
     "kind": "doc_and_version",
     "detail": "the filed doc is the render "
               "verbatim, the contract version is "
               "TF-034's API_VERSION with its "
               "surface clean, and DEFAULT_ENABLED "
               "is False"},
)


def run_kista_custody_probes() -> list[dict[str, Any]]:
    """Run the filed probes; each returns its
    verdict."""
    results: list[dict[str, Any]] = []
    for probe in KISTA_CUSTODY_PROBES:
        kind = probe["kind"]
        try:
            if kind == "disabled_refusal":
                built: list[str] = []

                def factory() -> Any:
                    built.append("resolver")
                    return _FakeResolver()

                custody = KistaCustody(
                    resolver_factory=factory)
                out = custody.resolve(
                    "kista://thoughtforge/openai_api_key",
                    consumer=lambda v: len(v))
                judged = custody.validate_config(
                    {"backend": "none"})
                passed = custody.enabled is False \
                    and built == [] \
                    and custody._resolver is None \
                    and out.get("ok") is False \
                    and out.get("status") == "refused" \
                    and out.get("error") == DISABLED_ERROR \
                    and judged.get("ok") is True \
                    and custody_response_problems(out) == [] \
                    and custody_response_problems(judged) == []
            elif kind == "reference_shape":
                valid = all(
                    parse_reference(r["reference"])
                    == ("thoughtforge", r["raw_key"])
                    for r in CONFIG_KEY_ROWS)
                invalid = [
                    "http://thoughtforge/openai_api_key",
                    "kista://ThoughtForge/openai_api_key",
                    "kista://thoughtforge",
                    "kista://thoughtforge/",
                    "kista://thoughtforge/OpenAI_API_KEY",
                    "sk-live-value-not-a-reference",
                    "",
                ]
                passed = valid \
                    and all(parse_reference(v) is None
                            for v in invalid) \
                    and all(reference_problems(v)
                            for v in invalid) \
                    and reference_problems(
                        "kista://thoughtforge/hf_token") == []
            elif kind == "opacity":
                fake = _FakeResolver()
                custody = KistaCustody(
                    resolver=fake, enabled=True)
                received: list[Any] = []
                out = custody.resolve(
                    "kista://thoughtforge/openai_api_key",
                    consumer=lambda v: received.append(v)
                    or len(v),
                    binding="backend_factory")
                passed = out.get("ok") is True \
                    and out.get("status") == "resolved" \
                    and received == [SENTINEL_SECRET] \
                    and out.get("consumer_result") \
                    == len(SENTINEL_SECRET) \
                    and "value" not in out \
                    and SENTINEL_SECRET not in \
                    _serialised_outputs(custody, out) \
                    and fake.calls == [
                        ("thoughtforge", "openai_api_key")] \
                    and custody_response_problems(out) == []
            elif kind == "shipped_configs":
                passed = shipped_config_problems() == []
            elif kind == "config_validator":
                raw = config_mapping_problems(
                    {"openai_api_key": "sk-live"})
                empty = config_mapping_problems(
                    {"hf_token": ""})
                nested = config_mapping_problems(
                    {"backend_cfg": {"hf_token": "x"}})
                migrated = config_mapping_problems(
                    {"openai_api_key_ref":
                     "kista://thoughtforge/openai_api_key",
                     "hf_token_ref":
                     "kista://thoughtforge/hf_token",
                     "backend": "none"})
                malformed = config_mapping_problems(
                    {"hf_token_ref": "hf_token_value"})
                passed = any("openai_api_key" in p
                             and "raw credential value" in p
                             for p in raw) \
                    and any("hf_token" in p
                            and "shape is the defect" in p
                            for p in empty) \
                    and any("hf_token" in p for p in nested) \
                    and migrated == [] \
                    and any("hf_token_ref" in p
                            for p in malformed)
            elif kind == "binding_needs":
                samtal_token = next(
                    (f for f in SAMTAL_FIELD_ROWS
                     if f["id"] == "seat.token"), None)
                runmal_clean = not any(
                    "credential" in r["id"]
                    or "api_key" in r["id"]
                    for r in RUNMAL_CALL_ROWS) \
                    and not any(
                        "credential" in f["id"]
                        or "api_key" in f["id"]
                        for f in RUNMAL_FIELD_ROWS)
                needing = [r for r in BINDING_NEED_ROWS
                           if r["needs_credential"]]
                passed = "credential_custody" \
                    in HERMES_NOT_ROUTED \
                    and "credential_custody" in EMBER_NOT_ROUTED \
                    and samtal_token is not None \
                    and samtal_token["disposition"] \
                    == "no_home_not_passed" \
                    and runmal_clean \
                    and len(needing) == 1 \
                    and needing[0]["id"] \
                    == "need_backend_factory" \
                    and all(
                        binding_need_row_problems(r) == []
                        for r in BINDING_NEED_ROWS)
            elif kind == "doc_and_version":
                doc = _REPO_ROOT / KISTA_CUSTODY_DOC
                passed = doc.exists() \
                    and doc.read_text(encoding="utf-8") \
                    == render_kista_custody_markdown() \
                    and kista_custody_contract()[
                        "contract_version"] == API_VERSION \
                    and verify_api_surface() == [] \
                    and DEFAULT_ENABLED is False
            else:  # pragma: no cover — the checker names unknown kinds
                passed = False
        except Exception:
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the kista-custody doc, verbatim) ────────────────────

def render_kista_custody_markdown() -> str:
    """Render the kista-custody doc
    deterministically — the exact text filed as
    docs/architecture/KISTA_CUSTODY.md."""
    lines: list[str] = []
    lines.append("# Kista Custody — Credentials Live in "
                 "Custody, Never in Configs (TF-160)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/kista_custody.py` — the "
        "filed doc IS the render: regenerate with "
        "`python3 -m thoughtforge.kista_custody`. "
        "The checker (`kista_custody_problems()`) "
        "re-derives every claim live from the tree "
        "and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{KISTA_CUSTODY_FORMAT}`")
    lines.append(
        f"Contract version: `{API_VERSION}` (TF-034's "
        "`API_VERSION` — consumed, not reinvented)")
    lines.append(
        f"Kista side: `{PLUGIN_NAME}` "
        f"{PLUGIN_VERSION} (manifest_version "
        f"{PLUGIN_MANIFEST_VERSION}) — the Hermes "
        "fork's secret-source plugin, read READ-ONLY "
        "from the Hermes tree; no Kista or Hermes "
        "code imported or copied")
    lines.append(
        f"Reference form: `{REFERENCE_PREFIX}"
        "<service>/<field>` judged by the plugin's "
        "own pattern")
    lines.append(
        f"Default enabled: {DEFAULT_ENABLED} — "
        "custody is OFF by default; enabling is an "
        "explicit act (`KistaCustody(enabled=True)` "
        "or `custody.enable()`), mirroring the "
        "Kista source's own `enabled` default")
    needing = sum(1 for r in BINDING_NEED_ROWS
                  if r["needs_credential"])
    lines.append(
        f"Binding needs: {len(BINDING_NEED_ROWS)} "
        f"({needing} needing a credential, custody-"
        f"referenced); custody rows: "
        f"{len(CUSTODY_ROWS)}; config keys: "
        f"{len(CONFIG_KEY_ROWS)}; field rows: "
        f"{len(FIELD_ROWS)}; rules: {len(RULE_ROWS)}")
    lines.append(f"Probes: {len(KISTA_CUSTODY_PROBES)}/"
                 f"{len(KISTA_CUSTODY_PROBES)}")
    lines.append("")
    lines.append("## 1. One custody")
    lines.append("")
    lines.append(
        "`KistaCustody` wraps an injectable resolver "
        "callable `(service, field) -> value` "
        "(directly or by factory; built lazily — "
        "constructing the custody builds no "
        "resolver). While disabled, every resolution "
        "returns a truthful refusal as data "
        "(`ok=False`, `status=\"refused\"`, the "
        "`error` naming custody as disabled) and "
        "touches no resolver. When enabled, a "
        "resolution parses the request's "
        "`kista://service/field` reference by the "
        "plugin's own pattern, calls the resolver "
        "with the parsed pair, and delivers the "
        "value ONLY to the caller-supplied consumer "
        "callable, in process: the response envelope "
        "files the reference, the parsed pair, the "
        "asking binding and the consumer's own "
        "result — never the value, which is stored "
        "nowhere and serialised nowhere. An enabled "
        "custody with no resolver returns the filed "
        "not-configured refusal: this module ships "
        "no live Kista client and claims none. An "
        "invalid request raises `ValueError` naming "
        "the problem(s), judged by the pure "
        "validators before any resolver is built or "
        "called. `validate_config()` judges a config "
        "mapping by the same pure validators and "
        "returns the verdict as data. Every response "
        "is a plain JSON-safe detached dict: `call`, "
        "`ok`, `status`, `contract_version` "
        "(TF-034's `API_VERSION`), `error`, plus the "
        "call's payload. No binding needs a "
        "credential today — TF-156..TF-159 each "
        "filed credential custody NOT routed / "
        "no-home — so custody here is governance "
        "plus shape: the shipped user config is "
        "migrated to references, every config in "
        "the tree is governed by the validator and "
        "the live scan, and any future need must "
        "resolve through this custody.")
    lines.append("")
    lines.append("## 2. The binding needs")
    lines.append("")
    lines.append("| Row | Consumer | Needs a "
                 "credential | Custody reference | "
                 "Filed evidence | What it is |")
    lines.append("|---|---|---|---|---|---|")
    for row in BINDING_NEED_ROWS:
        needs = "yes" if row["needs_credential"] else "no"
        ref = row["custody_reference"] or "—"
        lines.append(
            f"| {row['id']} | {row['consumer']} | "
            f"{needs} | {ref} | "
            f"{row['filed_evidence']} | "
            f"{row['summary']} |")
    lines.append("")
    lines.append(
        "The headline is the honest one: the four "
        "bindings and the service shape need no "
        "credential at all — their own filings say "
        "so, and the checker re-derives it live from "
        "those filings. The tree's one credential "
        "consumer is the backend factory behind the "
        "shipped user config, and its two keys are "
        "the subject of section 4.")
    lines.append("")
    lines.append("## 3. The Kista shape — filed from the plugin")
    lines.append("")
    lines.append("| Row | Aspect | Shape (as the "
                 "plugin files it) | Verified | "
                 "What it is |")
    lines.append("|---|---|---|---|---|")
    for row in CUSTODY_ROWS:
        lines.append(
            f"| {row['id']} | {row['aspect']} | "
            f"{row['shape']} | {row['verified']} | "
            f"{row['summary']} |")
    lines.append("")
    lines.append(
        "Nothing here is invented: the plugin, the "
        "pattern, the vault and the CLI resolution "
        "were read from "
        "`plugins/kista-secret-source/` in the "
        "Hermes tree (READ-ONLY) by this slice, and "
        "the Hermes profile key-file row is consumed "
        "from TF-158's own filing. What this slice "
        "does NOT have is a live Kista on the "
        "ThoughtForge side — see the filed limits.")
    lines.append("")
    lines.append("## 4. The config keys — migrated")
    lines.append("")
    lines.append("| Row | Raw key (gone) | "
                 "Reference key | Reference | "
                 "Config file | Consumer | What it "
                 "is |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in CONFIG_KEY_ROWS:
        lines.append(
            f"| {row['id']} | {row['raw_key']} | "
            f"{row['ref_key']} | {row['reference']} | "
            f"{row['config_file']} | "
            f"{row['consumer']} | {row['summary']} |")
    lines.append("")
    lines.append(
        "TF-009 censused `openai_api_key` and "
        "`hf_token` as raw keys in "
        "`configs/user_config.yaml` — empty in the "
        "shipped copy (verified at the byte level "
        "by this slice: both values were `\"\"`, "
        "so no secret was in the tree), with the "
        "*shape* as the finding. This slice "
        "migrates the shipped file: the raw keys "
        "are removed and the `*_ref` custody "
        "references above stand in their place. "
        "Because the shipped values were empty, "
        "the factory's behaviour on the shipped "
        "file is unchanged — an absent key and an "
        "empty key both fall back to its defaults. "
        "The checker scans every YAML file under "
        "`configs/` live: any raw credential key, "
        "at any depth, fails it by name.")
    lines.append("")
    lines.append("## 5. Field mapping — in (request -> resolver)")
    lines.append("")
    lines.append("| Field | Source | Target | "
                 "Disposition | What it is |")
    lines.append("|---|---|---|---|---|")
    for field in field_rows_for("in"):
        lines.append(
            f"| {field['id']} | {field['source']} | "
            f"{field['target']} | "
            f"{field['disposition']} | "
            f"{field['summary']} |")
    lines.append("")
    lines.append("## 6. Field mapping — out (resolution -> envelope)")
    lines.append("")
    lines.append("| Field | Source | Target | "
                 "Disposition | What it is |")
    lines.append("|---|---|---|---|---|")
    for field in field_rows_for("out"):
        lines.append(
            f"| {field['id']} | {field['source']} | "
            f"{field['target']} | "
            f"{field['disposition']} | "
            f"{field['summary']} |")
    lines.append("")
    lines.append("## 7. The rules")
    lines.append("")
    for rule in RULE_ROWS:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## 8. Docket")
    lines.append("")
    lines.append(
        "TF-160 **fixes** TF-016's D-CFG-4 — its "
        "assigned share: credential-shaped keys "
        "(`openai_api_key`, `hf_token`) lived in a "
        "plain YAML file (empty in the shipped "
        "copy, so no secret was in the tree, but "
        "the shape was the defect). The shipped "
        "`configs/user_config.yaml` no longer "
        "carries the raw keys — custody references "
        "stand in their place — and this module's "
        "validator + live scan govern every config "
        "in the tree. TF-016 totals move in the "
        "open: 137 entries — 31 OPEN / 1 "
        "OPEN-ENVIRONMENT / 101 FIXED / 4 "
        "CLOSED-BY-VERDICT (D-CFG-4 OPEN -> "
        "FIXED), and every live pin of the old "
        "totals in the tree's checkers, renders "
        "and tests is followed to the new totals "
        "in this same slice. Filed remainders, "
        "NOT claimed fixed: the backend factory "
        "still *reads* raw keys when a user "
        "hand-writes them into a local config "
        "(the validator refuses such a config; "
        "rewiring the factory to resolve through "
        "a live Kista is NOT PROVEN — no live "
        "Kista serves this tree), and the setup "
        "wizard still *writes* raw keys when a "
        "user enters one at setup time — TF-171's "
        "security audit and TF-177's upgrade path "
        "own those two roads.")
    lines.append("")
    lines.append("## 9. The probes")
    lines.append("")
    for probe in KISTA_CUSTODY_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 10. Filed limits")
    lines.append("")
    lines.append(
        "- Proof grade FIXTURE: the checker and "
        "probes run a fake resolver shaped on the "
        "filed Kista data. NO live Kista vault or "
        "CLI was called and NO Kista or Hermes "
        "code was imported or copied; live "
        "end-to-end custody resolution is NOT "
        "PROVEN — TF-170's Part IX gate owns the "
        "at-least-one-live proof, never claimed "
        "here.")
    lines.append(
        "- The Kista shape is verified from the "
        "Hermes fork's plugin source, read "
        "READ-ONLY; the standalone Kista project "
        "(the plugin's homepage, "
        "github.com/hrabanazviking/kista) was not "
        "itself read by this slice — the shape "
        "filed here is the plugin's, exactly as "
        "Hermes consumes it.")
    lines.append(
        "- The migration covers the shipped "
        "configs in this tree. A user's local "
        "config (configs/local_config.yaml, "
        "gitignored, absent here) is governed by "
        "the same validator when judged, but no "
        "such file exists on this machine to "
        "migrate or scan.")
    lines.append(
        "- The backend factory's raw-key read "
        "path and the setup wizard's raw-key "
        "write path stand as filed remainders "
        "(see Docket): the validator refuses the "
        "shape, but the factory is not rewired "
        "and the wizard is not rewritten by this "
        "slice — TF-171 / TF-177 own those roads.")
    lines.append(
        "- TF-161 owns binding fixtures in depth; "
        "TF-162 owns versioning enforcement in "
        "depth; TF-163 owns the local-only proof "
        "for bindings. TF-160 files the custody "
        "shape, the config migration and the "
        "governance the bindings' NOT-routed "
        "credential rows named as this slice's "
        "burden.")
    lines.append(
        "- No canon data was touched (TF-017); no "
        "README.md anywhere was touched; no "
        "Docker was used; nothing under the "
        "Hermes tree was edited.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_kista_custody_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / KISTA_CUSTODY_DOC
    doc.write_text(render_kista_custody_markdown(),
                   encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.kista_custody`."""
    doc = write_kista_custody_doc()
    print(f"kista custody doc regenerated: -> "
          f"{KISTA_CUSTODY_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def kista_custody_problems() -> list[str]:
    """Re-derive TF-160's claims from the live
    tree; name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    if KISTA_CUSTODY_FORMAT != "thoughtforge-kista-custody/1":
        problems.append(
            f"the format is {KISTA_CUSTODY_FORMAT!r}, not "
            "thoughtforge-kista-custody/1")
    if DEFAULT_ENABLED is not False:
        problems.append("DEFAULT_ENABLED is not False")
    if (PLUGIN_NAME, PLUGIN_VERSION, SOURCE_NAME,
            SOURCE_SCHEME) != (
            "kista-secret-source", "0.2.0", "kista", "kista"):
        problems.append(
            f"the plugin / source filing is "
            f"{PLUGIN_NAME!r} {PLUGIN_VERSION!r} / "
            f"{SOURCE_NAME!r} / {SOURCE_SCHEME!r}, not "
            "kista-secret-source 0.2.0 / kista / kista")
    if REFERENCE_PATTERN != (
            r"^kista://(?P<service>[a-z0-9][a-z0-9_.-]{0,127})"
            r"/(?P<field>[a-z_][a-z0-9_]{0,63})$"):
        problems.append(
            "REFERENCE_PATTERN is not the plugin's own "
            "pattern verbatim")
    if not DISABLED_ERROR or "disabled" not in DISABLED_ERROR:
        problems.append(
            "DISABLED_ERROR does not name custody as disabled")
    if not NOT_CONFIGURED_ERROR \
            or "not configured" not in NOT_CONFIGURED_ERROR:
        problems.append(
            "NOT_CONFIGURED_ERROR does not name custody "
            "as not configured")

    need_ids = [r["id"] for r in BINDING_NEED_ROWS]
    expected_needs = [
        "need_runmal_binding", "need_samtal_binding",
        "need_hermes_binding", "need_ember_binding",
        "need_service_shape", "need_backend_factory"]
    if need_ids != expected_needs \
            or len(need_ids) != len(set(need_ids)):
        problems.append(
            f"the BINDING_NEED_ROWS filing is not as "
            f"filed: {need_ids}")
    for row in BINDING_NEED_ROWS:
        problems.extend(binding_need_row_problems(row))
    needing = [r["id"] for r in BINDING_NEED_ROWS
               if r["needs_credential"]]
    if needing != ["need_backend_factory"]:
        problems.append(
            f"the credential-needing rows are {needing}, "
            "not exactly the backend factory — no "
            "binding may need a credential in this "
            "slice's filings")

    custody_ids = [r["id"] for r in CUSTODY_ROWS]
    expected_custody = [
        "custody_plugin", "custody_reference_form",
        "custody_vault", "custody_cli_resolution",
        "custody_source_config",
        "custody_hermes_profile_key_file"]
    if custody_ids != expected_custody \
            or len(custody_ids) != len(set(custody_ids)):
        problems.append(
            f"the CUSTODY_ROWS filing is not as filed: "
            f"{custody_ids}")
    for row in CUSTODY_ROWS:
        problems.extend(custody_row_problems(row))

    key_ids = [r["id"] for r in CONFIG_KEY_ROWS]
    if key_ids != ["config_openai_api_key", "config_hf_token"] \
            or len(key_ids) != len(set(key_ids)):
        problems.append(
            f"the CONFIG_KEY_ROWS filing is not as "
            f"filed: {key_ids}")
    for row in CONFIG_KEY_ROWS:
        problems.extend(config_key_row_problems(row))
    if tuple(r["raw_key"] for r in CONFIG_KEY_ROWS) \
            != CREDENTIAL_CONFIG_KEYS \
            or tuple(r["ref_key"] for r in CONFIG_KEY_ROWS) \
            != CREDENTIAL_REF_KEYS:
        problems.append(
            "the config-key rows do not file TF-009's "
            "two credential-shaped keys in order")

    field_ids = [f["id"] for f in FIELD_ROWS]
    if len(field_ids) != len(set(field_ids)):
        problems.append("the FIELD_ROWS filing has duplicate ids")
    for field in FIELD_ROWS:
        problems.extend(field_row_problems(field))
    value_row = next(
        (f for f in FIELD_ROWS if f["id"] == "response.value"),
        None)
    if value_row is None \
            or value_row.get("disposition") \
            != "no_home_never_filed":
        problems.append(
            "the resolved value is not filed no-home in "
            "FIELD_ROWS — it must have no response field")
    raw_row = next(
        (f for f in FIELD_ROWS
         if f["id"] == "config.raw_credential_key"), None)
    if raw_row is None \
            or raw_row.get("disposition") \
            != "no_home_not_passed":
        problems.append(
            "the raw config credential key is not filed "
            "no-home in FIELD_ROWS")

    rule_ids = [r["id"] for r in RULE_ROWS]
    if len(rule_ids) != len(set(rule_ids)) \
            or len(RULE_ROWS) != 10:
        problems.append(
            "the RULE_ROWS filing is not ten unique rules")
    for rule in RULE_ROWS:
        problems.extend(rule_row_problems(rule))
    probe_ids = [p["id"] for p in KISTA_CUSTODY_PROBES]
    if len(probe_ids) != len(set(probe_ids)) \
            or len(KISTA_CUSTODY_PROBES) != 7:
        problems.append(
            "the KISTA_CUSTODY_PROBES filing is not "
            "seven unique probes")

    # The contract is TF-034-versioned and its
    # surface is clean.
    contract = kista_custody_contract()
    if contract["contract_version"] != API_VERSION \
            or contract["default_enabled"] is not False \
            or contract["reference_pattern"] \
            != REFERENCE_PATTERN:
        problems.append(
            "the filed contract is not TF-034-versioned, "
            "off by default and plugin-pattern-filed")
    surface = verify_api_surface()
    if surface:
        problems.append(
            f"TF-034: verify_api_surface() is not clean: "
            f"{surface[:3]}")
    try:
        json.dumps(contract)
    except (TypeError, ValueError) as exc:
        problems.append(f"the contract is not JSON-safe: {exc}")
    if SENTINEL_SECRET in json.dumps(contract):
        problems.append(
            "the contract serialises the probe sentinel "
            "— a filing must never carry a value")

    # The neighbours stay clean live underneath,
    # and the bindings' own credential filings
    # still say what BINDING_NEED_ROWS claims.
    for name, checker in (
            ("TF-153 service_shape", service_shape_problems),
            ("TF-154 cognition_api", cognition_api_problems),
            ("TF-155 memory_api", memory_api_problems),
            ("TF-156 runmal_binding", runmal_binding_problems),
            ("TF-157 samtal_binding", samtal_binding_problems),
            ("TF-158 hermes_binding", hermes_binding_problems),
            ("TF-159 ember_binding", ember_binding_problems)):
        found = checker()
        if found:
            problems.append(
                f"{name}: its checker is not clean live: "
                f"{found[:3]}")
    if "credential_custody" not in HERMES_NOT_ROUTED:
        problems.append(
            "TF-158 no longer files credential_custody "
            "NOT routed — the hermes need row's evidence "
            "drifted")
    if "credential_custody" not in EMBER_NOT_ROUTED:
        problems.append(
            "TF-159 no longer files credential_custody "
            "NOT routed — the ember need row's evidence "
            "drifted")
    samtal_token = next(
        (f for f in SAMTAL_FIELD_ROWS
         if f["id"] == "seat.token"), None)
    if samtal_token is None \
            or samtal_token.get("disposition") \
            != "no_home_not_passed":
        problems.append(
            "TF-157 no longer files seat.token no-home — "
            "the samtal need row's evidence drifted")
    if any("credential" in r["id"] or "api_key" in r["id"]
           for r in RUNMAL_CALL_ROWS) \
            or any("credential" in f["id"]
                   or "api_key" in f["id"]
                   for f in RUNMAL_FIELD_ROWS):
        problems.append(
            "TF-156 has grown a credential surface — "
            "the runmal need row's evidence drifted")

    # The shipped configs carry references, not
    # values — scanned live.
    problems.extend(shipped_config_problems())

    # A disabled custody refuses, building
    # nothing; an enabled one over a fake
    # resolver delivers the sentinel only to
    # the consumer.
    try:
        built: list[str] = []

        def factory() -> Any:
            built.append("resolver")
            return _FakeResolver()

        custody = KistaCustody(resolver_factory=factory)
        refused = custody.resolve(
            "kista://thoughtforge/openai_api_key")
        problems.extend(
            f"disabled refusal: {p}"
            for p in custody_response_problems(refused))
        if refused.get("error") != DISABLED_ERROR or built:
            problems.append(
                f"the disabled custody did not refuse "
                f"untouched: built={built}, out={refused}")
        fake = _FakeResolver()
        enabled = KistaCustody(resolver=fake, enabled=True)
        received: list[Any] = []
        out = enabled.resolve(
            "kista://thoughtforge/hf_token",
            consumer=lambda v: received.append(v) or len(v))
        problems.extend(
            f"resolution: {p}"
            for p in custody_response_problems(out))
        if out.get("ok") is not True \
                or received != [SENTINEL_SECRET] \
                or SENTINEL_SECRET in \
                _serialised_outputs(enabled, out):
            problems.append(
                f"the fake-resolver resolution is not an "
                f"opaque delivery to the consumer only: "
                f"{out}")
        unconfigured = KistaCustody(enabled=True).resolve(
            "kista://thoughtforge/hf_token")
        if unconfigured.get("error") != NOT_CONFIGURED_ERROR:
            problems.append(
                f"the resolver-less custody did not "
                f"refuse as not configured: {unconfigured}")
    except Exception as exc:
        problems.append(f"custody round-trip failed: {exc}")

    # The probes all behave.
    probes = run_kista_custody_probes()
    if len(probes) != len(KISTA_CUSTODY_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the kista custody probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The docket records the fix.
    import re

    docket_text = _DOCKET.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", docket_text)
    if "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, 101 FIXED" \
            not in normalized:
        problems.append(
            "docket: the header totals no longer file 137 "
            "entries — 31 OPEN / 1 OPEN-ENVIRONMENT / 101 "
            "FIXED; TF-160 must move them exactly once")
    cfg4 = next(
        (line for line in docket_text.splitlines()
         if line.startswith("| D-CFG-4")), "")
    if "FIXED by TF-160" not in cfg4:
        problems.append(
            "docket: the D-CFG-4 row does not record "
            "FIXED by TF-160")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / KISTA_CUSTODY_DOC
    if not doc.exists():
        problems.append(
            f"the kista custody doc {KISTA_CUSTODY_DOC} "
            "is missing")
    elif doc.read_text(encoding="utf-8") \
            != render_kista_custody_markdown():
        problems.append(
            f"the filed doc {KISTA_CUSTODY_DOC} is not "
            "the render verbatim (regenerate with "
            "python3 -m thoughtforge.kista_custody)")
    return problems


def kista_custody_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for
    consumers."""
    probes = run_kista_custody_probes()
    return {
        "format": KISTA_CUSTODY_FORMAT,
        "doc": KISTA_CUSTODY_DOC,
        "contract": kista_custody_contract(),
        "measured": {
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "BINDING_NEED_IDS",
    "BINDING_NEED_ROWS",
    "CONFIG_KEY_IDS",
    "CONFIG_KEY_ROWS",
    "CREDENTIAL_CONFIG_KEYS",
    "CREDENTIAL_REF_KEYS",
    "CUSTODY_ROW_IDS",
    "CUSTODY_ROWS",
    "DEFAULT_ENABLED",
    "DISABLED_ERROR",
    "FIELD_IDS",
    "FIELD_ROWS",
    "KISTA_CUSTODY_DOC",
    "KISTA_CUSTODY_FORMAT",
    "KISTA_CUSTODY_PROBES",
    "KISTA_CUSTODY_RULES",
    "NOT_CONFIGURED_ERROR",
    "PLUGIN_MANIFEST_VERSION",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "REFERENCE_PATTERN",
    "REFERENCE_PREFIX",
    "REFERENCE_SCHEME",
    "REQUEST_FIELDS",
    "RULE_IDS",
    "RULE_ROWS",
    "SENTINEL_SECRET",
    "SOURCE_LABEL",
    "SOURCE_NAME",
    "SOURCE_SCHEME",
    "SOURCE_SHAPE",
    "USER_CONFIG_FILE",
    "VAULT_DIR_DEFAULT",
    "VAULT_FILES",
    "KistaCustody",
    "binding_need_row_for",
    "binding_need_row_problems",
    "config_key_row_for",
    "config_key_row_problems",
    "config_mapping_problems",
    "custody_response_problems",
    "custody_row_problems",
    "field_row_problems",
    "field_rows_for",
    "kista_custody_as_data",
    "kista_custody_contract",
    "kista_custody_problems",
    "main",
    "parse_reference",
    "reference_for_service_field",
    "reference_problems",
    "render_kista_custody_markdown",
    "resolve_request_problems",
    "rule_row_problems",
    "run_kista_custody_probes",
    "shipped_config_problems",
    "write_kista_custody_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
