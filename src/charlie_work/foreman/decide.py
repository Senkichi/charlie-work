"""The foreman Decide launch: one blocking, read-only, schema-validated model call.

Design doc §8.2 (issue #3120): Decide is a Turn's second step -- a single
synchronous ``claude -p`` call that answers the briefing with a JSON
document constrained by ``--json-schema``. Unlike the worker and reviewer
lanes it *blocks* the caller (the host ``decide_launch`` port wraps
``subprocess_runner.run_captured`` with a timeout and a process-tree kill),
so a Turn stalls at most ``foreman.decide_timeout_seconds`` on a hung model.

The launch:

* resolves its ``(harness, model)`` through the ``foreman`` role chain with
  the same quota-ledger bookkeeping as a reviewer launch -- a provider
  throttle, quota, or credential death is classified off the transcript,
  recorded against the selected entry, and comes back ``deferred``: a
  back-off the Turn driver retries next slice, never a Turn failure;
* runs with the foreman's pinned tool allowlist and isolated MCP
  (``tool_scope._apply_tool_scope``), ``--permission-mode plan``, no
  settings files, and a sanitized environment -- it cannot write, shell
  out, or reach the network;
* runs with ``cwd`` at the foreman's ``tree/`` worktree -- never the repo
  the foreman is registered at;
* writes ``turn_dir/decide.log`` on every launch attempt -- the transcript
  the failure classifier reads and the record an operator replays.

Only the ``claude-code`` harness is supported for now; any other configured
harness is a config error, not a launch failure.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import env_sanitize, host, role_quota_ledger, role_selection, tool_scope
from ..adapter_fate_profile import classify_for
from ..atomic_write import write_text_atomic
from ..claude_code import (
    _REVIEW_COMMAND_TEMPLATE,
    _apply_max_turns_pin,
    _apply_model_pin,
    _sanitize_review_command_template,
)
from ..config import OrchestratorConfig
from ..foreman_config import FOREMAN_DEFAULT_MODEL
from ..harnesses import HARNESS_REGISTRY
from ..role_chain import RoleEntry
from ..role_selection import RoleSelection
from ..subprocess_runner import RunResult, command_failure_message, resolve_cli_binary
from ..throttle_signatures import is_provider_throttle_failure
from .paths import tree_path

_DECIDE_LOG_NAME = "decide.log"

# The only harness Decide launches through (issue #3120's "Support the
# claude-code harness"). The config field is validated against
# REVIEWER_HARNESSES, which is wider -- so this is a launch-time check, not
# a config-validation guarantee.
_DECIDE_HARNESS = "claude-code"

# ``WorkerView.adapter_kind`` spelling ``classify_for`` wants for that
# harness -- derived from the registry so the two names cannot drift.
_DECIDE_ADAPTER_KIND = HARNESS_REGISTRY[_DECIDE_HARNESS].adapter_kind

# Quota-ledger stamp role: what ``session_stamp`` records and what
# ``record_classified_death`` keys a ``(harness, model)`` restriction on.
_DECIDE_ROLE = "foreman"

# Flags Decide pins itself -- the structured-output contract and the
# settings-source isolation. A configured ``claude_code.command`` template
# gets every occurrence of these stripped before the authoritative values
# are appended: the same last-flag-wins defense the pin helpers apply to
# ``--permission-mode``/``--model``/``--tools``. A template must not be able
# to widen the Decide output contract.
_DECIDE_PINNED_FLAGS = ("--output-format", "--json-schema", "--setting-sources")


@dataclass(frozen=True)
class DecideForeman:
    """The foreman slice a Decide launch acts on: its store dir and the fleet config.

    ``root`` is the resolved ``fleet_dir()/foremen/<name>/`` directory (from
    ``foreman.paths.foreman_dir``); ``tree/`` underneath it is the launch's
    working directory.
    """

    root: Path
    config: OrchestratorConfig


@dataclass(frozen=True)
class DecideResult:
    """The outcome of one Decide launch, as a value (issue #3120).

    Exactly one of three shapes: ``output`` set (the parsed, schema-validated
    action document), ``deferred`` set (a provider throttle/quota/auth death
    or an exhausted role chain -- a retryable back-off, not a Turn failure),
    or ``error`` set (every other failure -- a Turn failure the driver
    counts). ``selection`` records which role-chain entry ran and which were
    skipped, for the Turn journal.
    """

    output: dict[str, Any] | None = None
    error: str | None = None
    deferred: bool = False
    failure_kind: str | None = None
    selection: RoleSelection | None = None
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.output is not None and not self.deferred and self.error is None

    @property
    def entry(self) -> RoleEntry | None:
        """The ``(harness, model)`` chain entry that ran (``None`` when none did)."""
        return self.selection.entry if self.selection is not None else None


def _strip_pinned_flags(command: tuple[str, ...]) -> tuple[str, ...]:
    """Remove every Decide-pinned flag (``--flag value`` and ``--flag=value`` forms)."""
    filtered: list[str] = []
    skip_next = False
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token in _DECIDE_PINNED_FLAGS:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in _DECIDE_PINNED_FLAGS):
            continue
        filtered.append(token)
    return tuple(filtered)


def _decide_argv(
    config: OrchestratorConfig, entry: RoleEntry, schema_json: str
) -> tuple[str, ...]:
    """Build the fully-pinned ``claude -p`` argv for one Decide launch.

    The configured ``claude_code.command`` template is the base (the seam a
    test double for the ``claude`` binary uses; decide feeds the briefing via
    stdin and does not render ``{prompt_path}`` placeholders), then every
    flag the Decide contract depends on is stripped and re-pinned: read-only
    permission mode, the selected role-chain model, the foreman turn cap, the
    foreman tool allowlist with MCP isolation, and the structured-output
    contract.
    """
    template = config.claude_code.command or _REVIEW_COMMAND_TEMPLATE
    if isinstance(template, str):  # defensive: a hand-built config
        template = tuple(template.split())
    command = _sanitize_review_command_template(tuple(template))
    command = _apply_model_pin(command, entry.model or FOREMAN_DEFAULT_MODEL)
    command = _apply_max_turns_pin(command, config.foreman.max_model_turns)
    command = tool_scope._apply_tool_scope(
        command, config.foreman.tools, config.claude_code.isolate_mcp
    )
    command = _strip_pinned_flags(command)
    command = (
        *command,
        "--output-format",
        "json",
        "--json-schema",
        schema_json,
        # Load no settings files (same reasoning as run_quota_probe): nothing
        # the model needs lives in them, and an operator's SessionStart hook
        # must not fire inside the blocking launch.
        "--setting-sources",
        "",
    )
    return (resolve_cli_binary(command[0]), *command[1:])


def _write_decide_log(log_path: Path, argv: tuple[str, ...], result: RunResult) -> None:
    """Write the launch transcript: argv header, then stderr, then stdout last.

    stdout goes last so the terminal result event sits at the tail -- the
    region the failure classifier reads.
    """
    status = (
        f"timed out after the deadline (error: {result.error})"
        if result.timed_out
        else f"returncode={result.returncode} error={result.error!r}"
    )
    write_text_atomic(
        log_path,
        f"$ {' '.join(argv)}\n[{status}]\n=== stderr ===\n{result.stderr}\n"
        f"=== stdout ===\n{result.stdout}",
    )


def _envelope_failure(envelope: Any) -> str | None:
    """Extract the Decide error from a parsed ``--output-format json`` envelope.

    ``None`` means the envelope carried a valid ``structured_output`` object.
    """
    if not isinstance(envelope, dict):
        return "decide output is not a JSON result envelope"
    if envelope.get("is_error"):
        detail = str(envelope.get("result") or "").strip()
        return f"decide reported is_error (subtype={envelope.get('subtype')!r})" + (
            f": {detail}" if detail else ""
        )
    if not isinstance(envelope.get("structured_output"), dict):
        return f"decide produced no schema-valid output (subtype={envelope.get('subtype')!r})"
    return None


def decide(
    foreman: DecideForeman,
    briefing_text: str,
    schema: Mapping[str, Any],
    turn_dir: Path,
) -> DecideResult:
    """Run one blocking Decide launch for ``foreman``; never raises.

    ``schema`` is the Turn's action schema, passed to the CLI as
    ``--json-schema`` (the CLI enforces it server-side per call and retries
    internally; ``structured_output`` is only populated when the response
    validates). ``turn_dir`` is this Turn's directory -- the transcript
    lands at ``turn_dir/decide.log``.
    """
    ports = host.current()
    config = foreman.config
    selection = role_selection.select_for_config(
        config.foreman.chain, config, now=ports.clock.now()
    )

    if selection.entry is None:
        return DecideResult(
            error=f"foreman role chain exhausted: {selection.exhausted_reason}",
            deferred=True,
            selection=selection,
        )
    entry = selection.entry
    if entry.harness != _DECIDE_HARNESS:
        return DecideResult(
            error=(
                f"unsupported foreman harness {entry.harness!r}: "
                "decide supports claude-code only -- a config error, not a launch failure"
            ),
            selection=selection,
        )
    if not isinstance(schema, Mapping):
        return DecideResult(
            error=f"action schema must be a JSON object, got {type(schema).__name__}",
            selection=selection,
        )
    try:
        schema_json = json.dumps(dict(schema))
    except (TypeError, ValueError) as exc:
        return DecideResult(
            error=f"action schema is not JSON-serializable: {exc}", selection=selection
        )
    try:
        tree = tree_path(foreman.root)
    except ValueError as exc:
        return DecideResult(error=str(exc), selection=selection)
    if not tree.is_dir():
        return DecideResult(
            error=f"foreman tree missing at {tree} -- Collect creates it before Decide",
            selection=selection,
        )

    argv = _decide_argv(config, entry, schema_json)
    try:
        env = env_sanitize.sanitize_env(tree, tmp_dir=turn_dir / "tmp", config=config)
    except OSError as exc:
        return DecideResult(
            error=f"failed to prepare decide environment: {exc}", selection=selection
        )
    # Harness defaults (background tasks off) over the sanitized base; the
    # operator ``worker_env`` merge is worker-lane config decide does not
    # inherit.
    env = env_sanitize.build_worker_env(env, None)

    launched = ports.clock.monotonic()
    try:
        result = ports.decide_launch.launch(
            argv,
            cwd=tree,
            env=env,
            timeout_seconds=config.foreman.decide_timeout_seconds,
            stdin=briefing_text,
        )
    except Exception as exc:  # errors-as-values boundary -- the port never should raise
        result = RunResult(
            returncode=None, stdout="", stderr="", error=f"{type(exc).__name__}: {exc}"
        )
    duration = ports.clock.monotonic() - launched
    log_path = turn_dir / _DECIDE_LOG_NAME
    try:
        _write_decide_log(log_path, argv, result)
    except OSError:
        pass  # the RunResult still carries the failure; the log is best-effort

    error_text: str | None = None
    output: dict[str, Any] | None = None
    num_turns = 0
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    if result.ok:
        try:
            envelope: Any = json.loads(result.stdout)
        except json.JSONDecodeError:
            envelope = None
        error_text = _envelope_failure(envelope)
        if error_text is None:
            output = envelope["structured_output"]
            usage = envelope.get("usage")
            if isinstance(usage, Mapping):
                input_tokens = int(usage.get("input_tokens") or 0)
                output_tokens = int(usage.get("output_tokens") or 0)
            num_turns = int(envelope.get("num_turns") or 0)
            cost_usd = float(envelope.get("total_cost_usd") or 0.0)
    else:
        error_text = command_failure_message(list(argv), result, "no output")

    if error_text is None:
        return DecideResult(
            output=output,
            selection=selection,
            num_turns=num_turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            duration_seconds=duration,
        )

    # A failed launch may still be a provider throttle/quota/auth death --
    # classify it off the transcript tail exactly like a reviewer death, and
    # record the restriction so the next selection skips the entry.
    failure_kind, until = classify_for(
        _DECIDE_ADAPTER_KIND,
        log_path,
        config.runtime.throttle_error_markers,
        quota_error_markers=config.runtime.quota_error_markers,
        now=ports.clock.now(),
        harness_auth=role_selection.harness_auth_enabled(config),
    )
    if failure_kind is not None:
        stamp = role_quota_ledger.session_stamp(
            _DECIDE_ROLE, entry.harness, entry.model, selection.index or 0
        )
        role_quota_ledger.record_classified_death(
            {role_quota_ledger.SESSION_ROLE_KEY: stamp},
            failure_kind,
            until,
            source="foreman_decide",
            harness=entry.harness,
            evidence=error_text[:500],
        )
    return DecideResult(
        error=error_text,
        deferred=is_provider_throttle_failure(failure_kind),
        failure_kind=failure_kind,
        selection=selection,
        duration_seconds=duration,
    )
