"""Test impact selection for the worker loop and the local merge gate.

charlie-work never selects tests itself. It wraps the repository's own runner in
``ci-fleet test``, which picks the tests a change can affect, runs the full suite
whenever it cannot be sure, and exits with the runner's exit code. This module is
the one place that derives what the wrapper needs: its executable, the repo slug
and the base ref.

The executable is the ``ci-fleet`` console script beside this interpreter:
``ci-fleet`` is a charlie-work dependency, so the two share a venv
(``Scripts/ci-fleet.exe`` on Windows, ``bin/ci-fleet`` elsewhere). Nothing here
imports ``ci_fleet.selection``; the wrapper is only ever a subprocess.

Everything fails soft. When the wrapper cannot be used -- no executable,
``enabled = false`` in the repo's ``ci-fleet.toml``, no slug or no base --
``resolve_selection`` says why, the callers keep today's commands, and the
reason is recorded as an event.

A usable wrapper can still be useless: when the repo's selection map is stale
(or missing) ``ci-fleet test`` runs the whole suite, which for a worker is a serial
run that outlasts its foreground limit. ``probe_selection`` asks the selector
itself, once per rendered worker or rework prompt, whether it can narrow anything;
when it can only answer ``full`` the prompt keeps the hand-picked targeted
command (reason ``selection-full``) and returns to the wrapper by itself once the
map is fresh again. The merge gate is not probed: a full run there is the correct
fail-open.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .subprocess_runner import run_captured

CI_FLEET_EXE_NAME = "ci-fleet.exe" if os.name == "nt" else "ci-fleet"
CI_FLEET_CONFIG = "ci-fleet.toml"
GIT_TIMEOUT_SECONDS = 30
# ``shadow-status`` is one ledger read; the bound only stops a wedged process
# from holding the merge gate. A timeout reads as "window not met" (shadow).
SHADOW_STATUS_TIMEOUT_SECONDS = 60
# ``select`` reads the selection map and diffs two refs (about 2 seconds live); the
# bound only stops a wedged process from holding up a prompt render.
SELECT_PROBE_TIMEOUT_SECONDS = 60

# Why the wrapper is not used. ``disabled`` is the repo's own choice; the rest are faults.
# ``selection-full`` is a usable wrapper that can only run everything (see ``probe_selection``).
NO_EXECUTABLE = "no-executable"
DISABLED = "disabled"
NO_REPO = "no-repo"
NO_BASE = "no-base"
SELECTION_FULL = "selection-full"

# The gate's two wrapped modes. ``shadow`` records the selection and runs everything.
SHADOW = "shadow"
ENFORCED = "enforced"

_GITHUB_REMOTE = re.compile(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/*$")


@dataclass(frozen=True)
class SelectionTarget:
    """What ``ci-fleet test`` needs: its executable, the repo slug and the base ref."""

    exe: Path
    repo: str
    base: str


@dataclass(frozen=True)
class SelectionUnavailable:
    """Why the wrapper cannot be used; the caller keeps today's command."""

    reason: str
    detail: str = ""


@dataclass(frozen=True)
class GateSelection:
    """The merge gate's suite argv, and whether (and how) it was wrapped."""

    argv: tuple[str, ...]
    mode: str
    detail: str = ""


def ci_fleet_executable() -> Path | None:
    """The ``ci-fleet`` console script beside this interpreter, else None."""
    exe = Path(sys.executable).parent / CI_FLEET_EXE_NAME
    return exe if exe.is_file() else None


def selection_disabled(repo_root: Path) -> bool:
    """True only for an explicit ``[test_selection] enabled = false``.

    A missing or malformed file is not a reason to skip the wrapper: ``ci-fleet
    test`` reads the same file, and on a config error it runs the full suite and
    records why.
    """
    try:
        data = tomllib.loads((repo_root / CI_FLEET_CONFIG).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    section = data.get("test_selection")
    return isinstance(section, dict) and section.get("enabled") is False


def repo_slug(repo_root: Path) -> str | None:
    """``owner/name`` from a GitHub ``origin``; ``local/<directory>`` with no ``origin``.

    ``local/<directory>`` is the identity the local backend reports for a repo with
    no remote (``LocalFileGitHub.name_with_owner``), so selections are recorded
    under the name the nightly uses. An ``origin`` that is not GitHub yields None.
    """
    remotes = run_captured(["git", "remote"], cwd=repo_root, timeout_seconds=GIT_TIMEOUT_SECONDS)
    if not remotes.ok:
        return None
    if "origin" not in remotes.stdout.split():
        return f"local/{repo_root.name}"
    url = run_captured(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        timeout_seconds=GIT_TIMEOUT_SECONDS,
    )
    match = _GITHUB_REMOTE.search(url.stdout.strip()) if url.ok else None
    return f"{match.group(1)}/{match.group(2)}" if match else None


def selection_base(repo_root: Path, configured_base_ref: str) -> str | None:
    """The ref worker branches start from: ``dispatch.base_ref``, else ``origin/<default>``.

    A repo with no ``origin`` bases workers on local ``HEAD``. Inside a worker's
    worktree that would name the worker's own commit, so the main worktree's branch
    is used instead. None when neither resolves.
    """
    from .local_lane import local_base_branch
    from .worktree import _resolve_default_branch_ref

    base = configured_base_ref.strip()
    if not base:
        try:
            base = _resolve_default_branch_ref(repo_root)
        except RuntimeError:
            return None
    if base == "HEAD":
        return local_base_branch(repo_root)
    return base


def resolve_selection(
    repo_root: Path | None, *, configured_base_ref: str = "", base: str | None = None
) -> SelectionTarget | SelectionUnavailable:
    """Everything ``ci-fleet test`` needs, or why it cannot be used.

    ``base`` (the gate's resolved base sha) replaces the base derivation.
    """
    if repo_root is None:
        return SelectionUnavailable(NO_REPO, "no repository root")
    exe = ci_fleet_executable()
    if exe is None:
        return SelectionUnavailable(
            NO_EXECUTABLE, f"no {CI_FLEET_EXE_NAME} beside {sys.executable}"
        )
    if selection_disabled(repo_root):
        return SelectionUnavailable(DISABLED, f"{CI_FLEET_CONFIG} sets enabled = false")
    repo = repo_slug(repo_root)
    if repo is None:
        return SelectionUnavailable(NO_REPO, "origin is not a GitHub remote")
    resolved = base or selection_base(repo_root, configured_base_ref)
    if not resolved:
        return SelectionUnavailable(NO_BASE, "no base ref resolvable")
    return SelectionTarget(exe=exe, repo=repo, base=resolved)


def probe_selection(target: SelectionTarget, repo_root: Path) -> SelectionUnavailable | None:
    """Ask ``ci-fleet select`` whether it can narrow anything; unavailable when it can only say full.

    The probe diffs the base against itself. An empty diff cannot trigger any
    diff-dependent reason, so a ``full`` verdict there (``map-stale``, ``no-map``,
    ``map-project-dir``) holds for every change the worker will make. Every other
    outcome -- ``selected``, a non-zero exit, a timeout, output that is not one JSON
    object, a missing ``mode`` -- returns None, so the prompt keeps ``ci-fleet test``
    exactly as it did before the probe existed.
    """
    result = run_captured(
        [
            str(target.exe),
            "select",
            "--repo",
            target.repo,
            "--base",
            target.base,
            "--head",
            target.base,
            "--format",
            "json",
        ],
        cwd=repo_root,
        timeout_seconds=SELECT_PROBE_TIMEOUT_SECONDS,
    )
    if not result.ok or result.timed_out:
        return None
    try:
        verdict = json.loads(result.stdout)
    except (ValueError, RecursionError):
        return None
    if not isinstance(verdict, dict) or verdict.get("mode") != "full":
        return None
    reason = verdict.get("reason")
    return SelectionUnavailable(
        SELECTION_FULL,
        f"ci-fleet select answers full for an empty diff against {target.base} "
        f"(reason: {reason if isinstance(reason, str) and reason else 'none given'}), "
        "so that verdict holds for every change",
    )


def worker_selection(
    repo_root: Path | None,
    configured_base_ref: str,
    *,
    state_file: Path,
    payload: Mapping[str, object],
    probe: bool = True,
) -> SelectionTarget | None:
    """``resolve_selection`` for a worker or rework prompt, recording why it fell back.

    With ``probe`` (``dispatch.worker_selection_probe``) a usable target is also
    checked by ``probe_selection``; a selector that can only answer ``full`` falls
    back like any other unavailable wrapper, through the same event.

    ``log_event`` (not ``_record_event``): prompts are rendered outside the state
    lock, like ``_build_module_map_value``. With no repo root (test callers only)
    there is nothing to report.
    """
    from .instrumentation import log_event

    if repo_root is None:
        return None
    selection = resolve_selection(repo_root, configured_base_ref=configured_base_ref)
    if isinstance(selection, SelectionTarget):
        unavailable = probe_selection(selection, repo_root) if probe else None
        if unavailable is None:
            return selection
        selection = unavailable
    fields = {**payload, "reason": selection.reason, "detail": selection.detail}
    if selection.reason == DISABLED:
        log_event(
            state_file,
            "worker_test_selection_disabled",
            fields,
            repo=repo_root.name,
            level="info",
        )
    else:
        log_event(
            state_file,
            "worker_test_selection_unavailable",
            fields,
            repo=repo_root.name,
            level="warning",
        )
    return None


def worker_command(target: SelectionTarget, runner: str, flags: str) -> str:
    """The worker's one test command: ``ci-fleet test`` around the repo's runner.

    Forward slashes and shell quoting keep the line valid in bash and PowerShell.
    """
    return " ".join(
        [
            shlex.quote(target.exe.as_posix()),
            "test",
            "--repo",
            shlex.quote(target.repo),
            "--base",
            shlex.quote(target.base),
            "--context",
            "worker",
            "--",
            runner,
            flags,
        ]
    )


def shadow_window_met(exe: Path, repo: str, *, cwd: Path) -> bool:
    """``ci-fleet nightly shadow-status --repo`` exited 0: the shadow window is met.

    Any other outcome (exit 1, a timeout, a missing subcommand) keeps the gate in
    shadow, which runs the full suite.
    """
    result = run_captured(
        [str(exe), "nightly", "shadow-status", "--repo", repo],
        cwd=cwd,
        timeout_seconds=SHADOW_STATUS_TIMEOUT_SECONDS,
    )
    return result.ok and not result.timed_out


def gate_argv(
    target: SelectionTarget, suite_argv: Sequence[str], *, head: str, shadow: bool
) -> list[str]:
    """The gate's suite argv inside ``ci-fleet test --context gate``.

    The wrapper runs the suite itself; its selection is never spliced into argv.
    """
    return [
        str(target.exe),
        "test",
        "--repo",
        target.repo,
        "--base",
        target.base,
        "--head",
        head,
        "--context",
        "gate",
        *(["--shadow"] if shadow else []),
        "--",
        *suite_argv,
    ]


def gate_selection(
    repo_root: Path, suite_argv: Sequence[str], *, head: str | None, base: str | None
) -> GateSelection:
    """Wrap the merge gate's suite, in shadow until the repo's shadow window is met.

    The switch to enforcement is read from the ledger on every launch, so it
    needs no follow-up change. Fails open: the argv is returned unchanged with
    the reason as ``mode``.
    """
    if not head or not base:
        return GateSelection(tuple(suite_argv), NO_BASE, "gate head or base unresolved")
    selection = resolve_selection(repo_root, base=base)
    if isinstance(selection, SelectionUnavailable):
        return GateSelection(tuple(suite_argv), selection.reason, selection.detail)
    shadow = not shadow_window_met(selection.exe, selection.repo, cwd=repo_root)
    argv = gate_argv(selection, suite_argv, head=head, shadow=shadow)
    return GateSelection(tuple(argv), SHADOW if shadow else ENFORCED)
