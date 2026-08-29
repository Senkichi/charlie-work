"""Cross-PR base-revert detection gate (issue #390, fail-closed rework #1068).

A squash-merge of a PR branch that contains a ``Revert "<original>"`` commit
whose ``<original>`` subject matches a commit already on the base branch would
silently undo that base change — the PR diff looks clean (the revert only
touches the branch), but the merged result rolls back base work. This module
enumerates the branch-only commits and matches revert subjects against base
commits to flag that hazard before the merge advances.

The gate returns an explicit :class:`CrossPrRevertResult` verdict rather than a
``str | None`` so the caller can distinguish "verified clean" (``CLEAN``) from
"could not verify" (``UNDETERMINED``). Previously both folded into ``None`` and
the caller (``workflow.merge_ready``) treated ``None`` as "proceed toward
merge", so a real cross-PR base revert could merge unflagged whenever the
gate's local-git verification failed transiently — the exact scenario the gate
exists to catch (issue #1068). ``UNDETERMINED`` fails closed: the merge is
held, but the PR is not routed to rework (an unverified gate is a refusal to
merge, not a detected revert).

This was verbatim-moved out of ``src/charlie_work/janitor.py`` (the
over-cap monolith) so the new code lands in a domain module under the cap
rather than growing the monolith past its file-size ratchet mark
(issue #1442). ``workflow.py`` re-exports the public names through its facade
import block, matching the ``.dispatch_selection`` / ``.escalation`` /
``.verdict_parsing`` / ``.rework_prompts`` / ``.ci_findings`` /
``.backlog_reachability`` / ``.stalled_review_reap`` / ``.dead_worker_reap``
extraction lineage.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from charlie_work.safe_ref import require_valid_ref_name
from charlie_work.subprocess_runner import (
    hidden_console_kwargs,
    no_console_window_kwargs,
)

logger = logging.getLogger(__name__)


class CrossPrRevertStatus(Enum):
    """Outcome states for :func:`detect_cross_pr_revert`.

    ``CLEAN``
        Verified: enumerated the branch commits and found no cross-PR base
        revert, OR an explicit ``allow-revert:`` marker line authorized the
        merge. The gate is satisfied and the merge may advance.
    ``REVERT_DETECTED``
        A silent cross-PR base revert was found. ``reason`` carries the
        blocking message; callers route the PR to rework.
    ``UNDETERMINED``
        The local git history required to decide was unavailable — a
        fetch/rev-list/log non-zero exit, a ref/SHA validation failure, an
        OSError, or the repo root / PR refs needed to run the gate were
        absent. The gate is NOT verified, so callers must fail closed: do not
        let the merge advance on an unverified gate (issue #1068). This is
        distinct from ``CLEAN``: a transient local-git failure used to fold
        into the same ``None`` as "verified clean", silently disabling the
        gate and letting a real cross-PR revert merge unflagged.
    """

    CLEAN = "clean"
    REVERT_DETECTED = "revert_detected"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class CrossPrRevertResult:
    """Verdict of :func:`detect_cross_pr_revert`.

    ``reason`` carries the blocking message for ``REVERT_DETECTED`` and a
    diagnostic for ``UNDETERMINED``; it is ``None`` for ``CLEAN``.
    """

    status: CrossPrRevertStatus
    reason: str | None = None

    @property
    def blocks_merge(self) -> bool:
        """True when the merge must not proceed on this gate's verdict.

        Both ``REVERT_DETECTED`` and ``UNDETERMINED`` block: the gate's whole
        purpose is to stop a specific bad merge, so an unverified gate fails
        closed (issue #1068). ``CLEAN`` is the only state that lets the merge
        advance.
        """
        return self.status is not CrossPrRevertStatus.CLEAN


def detect_cross_pr_revert(
    pr: dict[str, Any],
    repo_root: Path | None,
    allow_marker: str = "allow-revert",
) -> CrossPrRevertResult:
    """Return a verdict on whether the PR branch would silently revert a base commit.

    Enumerates non-merge commits on the PR branch that are not reachable from
    the base. A commit whose subject is ``Revert "<original>"`` and whose
    inner ``<original>`` subject matches a commit reachable from the base
    indicates that squashing the branch would undo a change already on the base
    (the cross-PR revert incident, issue #390). An explicit ``allow-revert:``
    marker line in the PR body suppresses the block so legitimate intentional
    reverts can be merged.

    Returns a :class:`CrossPrRevertResult`:

    - ``CLEAN`` when no revert is detected, an ``allow-revert:`` marker line
      is present, or the gate is structurally inapplicable (no ``repo_root``,
      not a git checkout, or the PR lacks ref names). The structural cases are
      "gate not applicable" rather than "verified clean" — but in production
      ``repo_root`` is always a configured git checkout and PRs always carry
      ref names, so those branches are only reached in tests/edge configs and
      folding them into ``CLEAN`` preserves the prior "gate skipped" behavior.
    - ``REVERT_DETECTED`` (with a blocking ``reason``) when a silent cross-PR
      base revert is found.
    - ``UNDETERMINED`` (with a diagnostic ``reason``) when the gate *ran* but
      a transient local-git failure prevented a verdict — a fetch/rev-list/log
      non-zero exit, a ref validation failure, or an OSError. Callers must
      treat ``UNDETERMINED`` as *not verified* and fail closed — never advance
      the merge on an unverified gate (issue #1068).
    """
    if not repo_root:
        return CrossPrRevertResult(CrossPrRevertStatus.CLEAN)

    repo_root_path = Path(repo_root)
    if not repo_root_path.is_dir() or not (repo_root_path / ".git").exists():
        return CrossPrRevertResult(CrossPrRevertStatus.CLEAN)

    body = str(pr.get("body") or "")
    # Require a structural marker line: "allow-revert:" followed by a reason.
    # A bare word/substring (e.g. quoting this guidance) must not bypass the gate.
    marker_re = re.compile(
        rf"^{re.escape(allow_marker)}:\s*\S",
        re.IGNORECASE | re.MULTILINE,
    )
    if marker_re.search(body):
        return CrossPrRevertResult(CrossPrRevertStatus.CLEAN)

    head_ref = pr.get("headRefName")
    base_ref = pr.get("baseRefName")
    if not head_ref or not base_ref:
        return CrossPrRevertResult(CrossPrRevertStatus.CLEAN)

    try:
        # Validate ref names before they reach git argv (issue #659). Both
        # values are passed as plain positionals to ``git fetch origin``, so a
        # flag-like value would be parsed as an option without this guard.
        head_ref = require_valid_ref_name(head_ref, context="detect_cross_pr_revert head_ref")
        base_ref = require_valid_ref_name(base_ref, context="detect_cross_pr_revert base_ref")

        fetch = subprocess.run(
            ["git", "fetch", "origin", str(head_ref), str(base_ref)],
            cwd=repo_root_path,
            capture_output=True,
            text=True,
            check=False,
            **hidden_console_kwargs(),
        )
        if fetch.returncode != 0:
            return CrossPrRevertResult(
                CrossPrRevertStatus.UNDETERMINED,
                f"git fetch origin {head_ref} {base_ref} failed (exit {fetch.returncode})",
            )

        commits = subprocess.run(
            [
                "git",
                "rev-list",
                "--no-merges",
                f"origin/{head_ref}",
                f"^origin/{base_ref}",
            ],
            cwd=repo_root_path,
            capture_output=True,
            text=True,
            check=False,
            **no_console_window_kwargs(),
        )
        if commits.returncode != 0:
            return CrossPrRevertResult(
                CrossPrRevertStatus.UNDETERMINED,
                f"git rev-list origin/{head_ref} ^origin/{base_ref} failed "
                f"(exit {commits.returncode})",
            )

        # Track whether every branch commit could be inspected. A non-zero
        # exit on any per-commit ``git log`` means that commit's subject (and
        # therefore any revert it might carry) could not be verified — the
        # gate is incomplete, not clean. If a revert is found in a later,
        # inspectable commit we still return REVERT_DETECTED (the block is
        # real regardless); only the no-revert-found fallthrough flips to
        # UNDETERMINED when inspection was incomplete (issue #1068).
        verification_incomplete = False
        for sha in commits.stdout.strip().splitlines():
            if not sha:
                continue
            subject_proc = subprocess.run(
                ["git", "log", "-1", "--format=%s", sha],
                cwd=repo_root_path,
                capture_output=True,
                text=True,
                check=False,
                **no_console_window_kwargs(),
            )
            if subject_proc.returncode != 0:
                verification_incomplete = True
                continue
            subject = subject_proc.stdout.strip()
            if subject.startswith('Revert "') and subject.endswith('"'):
                original = subject[len('Revert "') : -1]
                match_proc = subprocess.run(
                    [
                        "git",
                        "log",
                        f"origin/{base_ref}",
                        "--format=%H",
                        "--fixed-strings",
                        "--grep",
                        original,
                    ],
                    cwd=repo_root_path,
                    capture_output=True,
                    text=True,
                    check=False,
                    **no_console_window_kwargs(),
                )
                if match_proc.returncode != 0:
                    verification_incomplete = True
                    continue
                for base_sha in match_proc.stdout.strip().splitlines():
                    if not base_sha:
                        continue
                    base_subject_proc = subprocess.run(
                        ["git", "log", "-1", "--format=%s", base_sha],
                        cwd=repo_root_path,
                        capture_output=True,
                        text=True,
                        check=False,
                        **no_console_window_kwargs(),
                    )
                    if base_subject_proc.returncode != 0:
                        verification_incomplete = True
                        continue
                    if base_subject_proc.stdout.strip() == original:
                        return CrossPrRevertResult(
                            CrossPrRevertStatus.REVERT_DETECTED,
                            f"PR branch contains revert commit {sha[:12]} ({subject}) which "
                            f"would silently undo base commit {base_sha[:12]}; add an explicit "
                            f"'{allow_marker}: <reason>' line to the PR body to proceed",
                        )
    except ValueError as exc:
        # Ref validation failed (issue #659). This is an undetermined gate, not
        # a verified-clean one: log the diagnostic and fail closed (issue #1068).
        logger.warning("detect_cross_pr_revert ref validation failed: %s", exc)
        return CrossPrRevertResult(
            CrossPrRevertStatus.UNDETERMINED,
            f"ref validation failed: {exc}",
        )
    except OSError as exc:
        # OSError previously returned None with no logging at all, making a
        # disk/IO failure indistinguishable from "verified clean". Log and
        # fail closed (issue #1068).
        logger.warning("detect_cross_pr_revert git operation OS error: %s", exc)
        return CrossPrRevertResult(
            CrossPrRevertStatus.UNDETERMINED,
            f"OS error during git operation: {exc}",
        )

    if verification_incomplete:
        return CrossPrRevertResult(
            CrossPrRevertStatus.UNDETERMINED,
            "could not inspect every branch commit (git log non-zero exit); "
            "cross-PR revert gate not fully verified",
        )
    return CrossPrRevertResult(CrossPrRevertStatus.CLEAN)
