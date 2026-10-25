"""Harness-auth fault response (issue #2810).

A harness CLI that cannot authenticate (``devin`` with a dead or missing
credential) fails *every* launch on that harness the same way, whatever the
issue, repo or model. Before this module the fleet never noticed: each death was
charged to its own issue and the next pass launched again (31 dispatches in 22
hours on the operator host). This module is the fleet-level response, built from
parts that already exist rather than a second mechanism:

* **Classify** -- ``throttle_signatures.devin_harness_auth_evidence`` (declared on
  ``AdapterFateProfile.harness_auth_evidence``) turns the dead session's log into
  the failure kind ``harness_auth_failed``; the kind is exempt from every cap
  through ``is_provider_throttle_failure``.
* **Block** -- ``role_quota_ledger.record_classified_death`` records a
  harness-wide block in the existing quota ledger; ``role_selection`` routes
  every chain around it and ``worker_launch_gate`` defers an exhausted chain as
  ``harness_auth_blocked``. Live sessions are untouched.
* **Alert once** -- :func:`run_fleet_pass` claims the episode's alert in the
  ledger (so racing passes cannot both send it), emits ``harness_auth_blocked``
  and one ``notify`` digest.
* **Resume** -- in two stages, driven by the same pass. Stage A
  (:func:`run_stage_a`) runs the harness's own ``auth status`` in a sanitized
  environment once the block's window lapses; red re-arms the window with no
  launch. Green only *permits* Stage B: :func:`decide_for_harness` admits exactly
  one real launch as the probe, through ``fleet_provider_throttle``'s admit-one /
  survival gate under the key ``harness_auth:<harness>``. The harness reopens when
  that probe outlives the survival period (:func:`run_fleet_pass` then ends the
  episode and emits ``harness_auth_resumed``); a probe that dies of the signature
  re-arms the block with the ledger's escalating backoff and sends no second
  alert for the same episode.

Stage A deliberately does not trust ``devin auth status``'s exit code: it exits 0
and prints "Logged in" while its team-info section reports the invalid-key error
(owner comment on #2810). It is green only for exit 0 with no auth-error text, and
even then it is a cheap pre-filter, not proof: the Stage B launch is the proof that
the credential works (the team-settings fetch succeeds and the worker stays up).

``fleet.harness_auth_block_enabled: false`` is the kill switch: nothing here
classifies, blocks, probes or alerts, and an already-recorded block is ignored by
every reader (``role_quota_ledger.load_restrictions(harness_auth=False)``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import role_quota_ledger
from .env_sanitize import sanitize_env
from .instrumentation import log_event
from .notify import emit_digest, harness_auth_alert_digest
from .role_selection import harness_auth_enabled
from .subprocess_runner import RunResult, run_captured

logger = logging.getLogger(__name__)

#: Bound on the Stage A ``auth status`` subprocess (timeout counts as red).
STAGE_A_TIMEOUT_SECONDS = 20

#: Probe-marker key prefix in ``provider_resume_probe.json``.
PROBE_KEY_PREFIX = "harness_auth:"

#: Scratch directory (under the fleet dir) the Stage A subprocess runs in, so
#: ``sanitize_env``'s per-target tmp/gh-config directories never touch a worktree.
PROBE_DIRNAME = "harness_auth_probe"

#: ``<harness> -> args`` of its credential-status command, appended to the
#: harness's own executable. A harness is listed only once its real output has
#: been captured -- it is the Stage A counterpart of
#: ``AdapterFateProfile.harness_auth_evidence``. A harness with a block but no
#: entry here skips Stage A (green) and lets the one real probe launch decide.
_STATUS_ARGS: Mapping[str, tuple[str, ...]] = {"devin-shell": ("auth", "status")}

# Output of the status command that means the credential is NOT working,
# whatever the exit code: the CLI's own "not logged in", the server-side
# rejection, and a failed team-settings fetch (the call the invalid key fails).
_STAGE_A_RED = re.compile(
    r"Not logged in|Authentication required:\s*invalid api key|"
    r"Failed to (?:load|fetch) team settings",
    re.IGNORECASE,
)


def probe_key(harness: str) -> str:
    """The ``provider_resume_probe.json`` key of ``harness``'s resume probe."""
    return f"{PROBE_KEY_PREFIX}{harness}"


def session_repo_label(payload: Mapping[str, Any], sessions_dir: Path) -> str | None:
    """Best-effort repo name of a dead session, for the alert's "affected" list.

    The sidecar carries no repo field, so it is read off the layout: the
    component before ``worktrees`` in the session's ``worktree_path``, else the
    directory that owns the ``.var`` state root ``sessions_dir`` lives under.
    ``None`` when neither shape fits (the issue number is still recorded by the
    caller's ledger write only when a repo is known).
    """
    for raw, marker in ((payload.get("worktree_path"), "worktrees"), (str(sessions_dir), ".var")):
        if not isinstance(raw, str) or not raw:
            continue
        parts = Path(raw).parts
        for index in range(len(parts) - 1, 0, -1):
            if parts[index] == marker:
                return parts[index - 1] or None
    return None


def death_hook_fields(
    adapter_kind: str, log_path: Path, payload: Mapping[str, Any], sessions_dir: Path
) -> dict[str, str | None]:
    """Extra ``record_classified_death`` fields for a ``harness_auth_failed`` death.

    ``harness`` (so an unstamped, pre-#2086 sidecar still blocks), ``evidence``
    (the line the classification fired on) and ``repo``. Never raises.
    """
    # Lazy: adapter_fate_profile imports the adapter modules' dependencies.
    from . import adapter_fate_profile
    from .failure_classifier import harness_auth_evidence_from_log

    profile = adapter_fate_profile.profile_for(adapter_kind)
    if profile is None:
        return {}
    evidence = (
        harness_auth_evidence_from_log(
            log_path, profile.harness_auth_evidence, log_digest=profile.log_digest
        )
        if profile.harness_auth_evidence is not None
        else None
    )
    return {
        "harness": profile.harness,
        "evidence": evidence,
        "repo": session_repo_label(payload, sessions_dir),
    }


# ---------------------------------------------------------------------------
# Stage A -- the cheap credential check.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageAResult:
    """Outcome of the credential check: ``green`` plus a short operator-readable reason."""

    green: bool
    detail: str


def _status_command(harness: str, config: Any) -> list[str] | None:
    args = _STATUS_ARGS.get(harness)
    if args is None:
        return None
    from .devin_shell import DEFAULT_COMMAND_TEMPLATE

    configured = tuple(getattr(getattr(config, "devin", None), "shell_command", None) or ())
    return [(configured or DEFAULT_COMMAND_TEMPLATE)[0], *args]


def run_stage_a(
    harness: str,
    config: Any,
    *,
    fleet_dir_override: str | None = None,
    runner: Callable[..., RunResult] = run_captured,
) -> StageAResult:
    """Run the harness's credential-status command in a sanitized environment.

    Red: a non-zero exit, a timeout, a missing binary, or any output matching
    :data:`_STAGE_A_RED` (checked even on exit 0 -- the exit code is not trusted).
    Green otherwise; a harness with no declared status command is green by
    default (the one real probe launch decides). Never raises: the subprocess
    comes back as a value (``run_captured``).

    The command runs under :func:`env_sanitize.sanitize_env` passed as the full
    ``base_env`` -- layering it onto ``os.environ`` could not drop an ambient
    ``ACP_BACKEND`` (issue #2784), which would make a daemon started from an IDE
    terminal read "Not logged in" forever and never resume.
    """
    command = _status_command(harness, config)
    if command is None:
        return StageAResult(True, "no credential-status command declared; one real launch decides")
    from .fleet_paths import fleet_dir

    probe_root = fleet_dir(override=fleet_dir_override) / PROBE_DIRNAME
    tmp_dir = probe_root / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env = sanitize_env(probe_root, tmp_dir=tmp_dir, config=config)
    except OSError as exc:
        return StageAResult(False, f"could not prepare the probe environment: {exc}")
    result = runner(command, cwd=probe_root, timeout_seconds=STAGE_A_TIMEOUT_SECONDS, base_env=env)
    command_text = " ".join(command)
    if result.timed_out:
        return StageAResult(False, f"`{command_text}` timed out after {STAGE_A_TIMEOUT_SECONDS}s")
    if result.returncode is None:
        return StageAResult(False, f"`{command_text}` did not run: {result.error}")
    output = f"{result.stdout}\n{result.stderr}"
    match = _STAGE_A_RED.search(output)
    if match is not None:
        return StageAResult(False, f"`{command_text}` reports: {match.group(0)}")
    if result.returncode != 0:
        return StageAResult(False, f"`{command_text}` exited {result.returncode}")
    return StageAResult(True, f"`{command_text}` shows no auth error")


# ---------------------------------------------------------------------------
# Stage B -- the launch gate (one real launch as the probe).
# ---------------------------------------------------------------------------


def decide_for_harness(
    harness: str, *, fleet_dir_override: str | None = None, now: datetime | None = None
) -> Any:
    """The harness-auth half of the fleet launch gate, or ``None`` for "not blocked".

    ``None`` when ``harness`` has no open block episode. Otherwise a
    ``ResumeDecision`` flagged ``harness_auth``:

    * ``defer_throttled`` while the block's window is live, or has lapsed but
      Stage A has not yet passed for it (a launch must never be the credential
      check);
    * then ``fleet_provider_throttle.decide_probe_gate`` under
      :func:`probe_key`: ``admit_one`` (carrying the probe key so the launch
      stamps it), ``defer_probe`` while the probe is inside the survival period,
      and ``open`` once it provably outlived it.

    A survival the gate observes is persisted here so a probe that later exits
    normally cannot put the fleet back into single-launch mode.
    """
    from . import fleet_provider_throttle

    block = role_quota_ledger.read_harness_block(harness)
    if block is None or not block.episode_open:
        return None
    resolved_now = now if now is not None else datetime.now(UTC)
    if block.until > resolved_now or block.stage_a_green_at is None:
        return fleet_provider_throttle.ResumeDecision(
            "defer_throttled", throttled_until=block.until, harness_auth=True
        )
    key = probe_key(harness)
    gate = fleet_provider_throttle.decide_probe_gate(
        block.until, key, fleet_dir_override=fleet_dir_override, now=resolved_now
    )
    if gate.probe_survived:
        fleet_provider_throttle.note_probe_survived(
            key, fleet_dir_override=fleet_dir_override, now=resolved_now
        )
    return replace(
        gate,
        harness_auth=True,
        probe_survived=False,
        probe_keys=(key,) if gate.action == "admit_one" else (),
    )


# ---------------------------------------------------------------------------
# The fleet pass: alert once, run Stage A, end the episode on a surviving probe.
# ---------------------------------------------------------------------------


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _affected(block: role_quota_ledger.HarnessBlock) -> dict[str, list[int]]:
    by_repo: dict[str, list[int]] = {}
    for repo, issue in block.affected:
        by_repo.setdefault(repo, []).append(issue)
    return {repo: sorted(issues) for repo, issues in sorted(by_repo.items())}


def _emit_blocked(
    block: role_quota_ledger.HarnessBlock, *, fleet_state_path: Path, notify_config: Any
) -> None:
    affected = _affected(block)
    payload = {
        "harness": block.harness,
        "evidence": block.evidence,
        "affected": affected,
        "affected_repos": sorted(affected),
        "affected_issues": sorted({n for issues in affected.values() for n in issues}),
        "next_probe_at": _iso(block.until),
        "consecutive_hits": block.consecutive_hits,
        "episode_started_at": _iso(block.episode_started_at) if block.episode_started_at else None,
    }
    try:
        # write-gate-exempt(issue=2810): fleet-level event on the fleet events.db, no repo write_gate receiver.
        log_event(  # event-consumer: audit-only -- the operator-facing signal is the notify digest sent beside it; the block itself is enforced from the quota ledger (role_selection / worker_launch_gate), pinned by tests/test_harness_auth_failure.py.
            fleet_state_path, "harness_auth_blocked", payload, level="error"
        )
    except Exception:  # noqa: BLE001 - an alert must never break the pass
        logger.warning("harness_auth_blocked event failed for %s", block.harness, exc_info=True)
    if notify_config is None:
        return
    try:
        emit_digest(
            notify_config,
            harness_auth_alert_digest(
                _iso(datetime.now(UTC)),
                block.harness,
                evidence=block.evidence,
                affected=affected,
                next_probe_at=_iso(block.until),
            ),
        )
    except Exception:  # noqa: BLE001 - sinks report failures as values, but never trust that
        logger.warning("harness_auth_blocked digest failed for %s", block.harness, exc_info=True)


def _emit_resumed(
    block: role_quota_ledger.HarnessBlock, *, fleet_state_path: Path, now: datetime
) -> None:
    started = block.episode_started_at
    payload = {
        "harness": block.harness,
        "affected": _affected(block),
        "episode_started_at": _iso(started) if started else None,
        "resumed_at": _iso(now),
        "blocked_seconds": int((now - started).total_seconds()) if started else None,
        "consecutive_hits": block.consecutive_hits,
    }
    try:
        # write-gate-exempt(issue=2810): fleet-level event on the fleet events.db, no repo write_gate receiver.
        log_event(  # event-consumer: audit-only -- closes the episode opened by harness_auth_blocked for the operator; the reopening itself is the ledger stamp (role_quota_ledger.mark_harness_resumed), pinned by tests/test_harness_auth_failure.py.
            fleet_state_path, "harness_auth_resumed", payload, level="info"
        )
    except Exception:  # noqa: BLE001 - an alert must never break the pass
        logger.warning("harness_auth_resumed event failed for %s", block.harness, exc_info=True)


def run_fleet_pass(
    config: Any,
    *,
    fleet_dir_override: str | None,
    fleet_state_path: Path,
    notify_config: Any,
    dry_run: bool = False,
    now: datetime | None = None,
    stage_a: Callable[..., StageAResult] = run_stage_a,
) -> list[str]:
    """One fleet-pass step over every open harness-auth block; returns what it did.

    For each harness block with an open episode:

    1. take the episode's alert claim and, if this call won it, emit
       ``harness_auth_blocked`` and the notify digest (exactly once per episode);
    2. once the window has lapsed, run Stage A: green stamps the block so one
       probe launch may be admitted, red re-arms the window *without* escalating
       (no launch happened);
    3. once Stage A passed and the probe outlived the survival period, end the
       episode and emit ``harness_auth_resumed``.

    A no-op under ``dry_run`` or with the kill switch off. Never raises: a failure
    on one harness is logged and the pass moves on.
    """
    if dry_run or config is None or not harness_auth_enabled(config):
        return []
    from . import fleet_provider_throttle

    resolved_now = (now if now is not None else datetime.now(UTC)).replace(microsecond=0)
    actions: list[str] = []
    for block in role_quota_ledger.list_harness_blocks():
        if not block.episode_open:
            continue
        harness = block.harness
        try:
            claimed = role_quota_ledger.claim_harness_alert(harness, now=resolved_now)
            if claimed is not None:
                _emit_blocked(
                    claimed, fleet_state_path=fleet_state_path, notify_config=notify_config
                )
                actions.append(f"{harness}:alerted")
                block = claimed
            if block.until > resolved_now:
                continue
            if block.stage_a_green_at is None:
                result = stage_a(harness, config, fleet_dir_override=fleet_dir_override)
                if result.green:
                    role_quota_ledger.mark_stage_a_green(harness, now=resolved_now)
                    actions.append(f"{harness}:stage_a_green")
                else:
                    role_quota_ledger.arm_harness_block(
                        harness,
                        base_window=role_quota_ledger.HARNESS_AUTH_BASE_WINDOW,
                        evidence=None,
                        source="harness_auth_stage_a",
                        now=resolved_now,
                        escalate=False,
                    )
                    logger.info("harness auth stage A red for %s: %s", harness, result.detail)
                    actions.append(f"{harness}:stage_a_red")
                continue
            key = probe_key(harness)
            gate = fleet_provider_throttle.decide_probe_gate(
                block.until, key, fleet_dir_override=fleet_dir_override, now=resolved_now
            )
            if gate.action != "open":
                continue
            if gate.probe_survived:
                fleet_provider_throttle.note_probe_survived(
                    key, fleet_dir_override=fleet_dir_override, now=resolved_now
                )
            resumed = role_quota_ledger.mark_harness_resumed(harness, now=resolved_now)
            if resumed is not None:
                _emit_resumed(resumed, fleet_state_path=fleet_state_path, now=resolved_now)
                actions.append(f"{harness}:resumed")
        except Exception:  # noqa: BLE001 - one harness's failure must not stop the pass
            logger.warning("harness auth pass failed for %s", harness, exc_info=True)
    return actions
