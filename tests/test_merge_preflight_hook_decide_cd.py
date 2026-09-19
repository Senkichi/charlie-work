"""Tests for ``charlie_work.merge_preflight_hook`` (#894).

``_decide`` ``cd``-resolution behavior (#1252 defect 1 + review
finding): the ``cd``'d directory -- not the hook's own cwd -- picks
the fleet repo that merge-check runs against, and subshell-scoped
``cd`` must not misattribute the repo.

Everything is mocked: no network, no real fleet.json reads, no subprocesses,
no LLM processes. Split verbatim out of ``tests/test_merge_preflight_hook.py``
for the Track-1 attachment-budget split (#1564).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from charlie_work import merge_preflight_hook as hook

# ---------------------------------------------------------------------------
# #1252 defect 1: repo resolved from command's effective cwd, not hook cwd
# ---------------------------------------------------------------------------


def test_decide_bash_cd_resolves_correct_fleet_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The core defect: a session anchored in charlie-work merging a job-cannon
    # PR by number was checked against charlie-work's PR. After the fix, the
    # cd'd directory determines the repo.
    charlie_root = tmp_path / "charlie-work"
    job_cannon_root = tmp_path / "job-cannon"
    charlie_root.mkdir()
    job_cannon_root.mkdir()
    monkeypatch.setattr(
        hook,
        "_load_fleet_roots",
        lambda: {
            "senkichi/charlie-work": charlie_root,
            "senkichi/job-cannon": job_cannon_root,
        },
    )
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    # Session cwd is charlie-work, but the merge runs in job-cannon.
    reason = hook._decide(
        "Bash",
        {"command": f"cd {job_cannon_root.as_posix()} && gh pr merge 1679 --squash"},
        charlie_root,
    )
    assert reason is None
    assert calls == [(job_cannon_root, 1679)]


def test_decide_bash_cd_to_non_fleet_dir_skips_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the cd'd directory is not a fleet repo, the merge is out of scope —
    # do NOT fall back to the hook cwd (the cd is authoritative).
    charlie_root = tmp_path / "charlie-work"
    outside = tmp_path / "outside"
    charlie_root.mkdir()
    outside.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"senkichi/charlie-work": charlie_root})
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide(
        "Bash",
        {"command": f"cd {outside.as_posix()} && gh pr merge 5 --squash"},
        charlie_root,
    )
    assert reason is None
    assert calls == []


def test_decide_bash_no_cd_warns_on_hook_cwd_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No --repo and no cd: the hook infers from its own cwd and warns.
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide("Bash", {"command": "gh pr merge 5 --squash"}, root)
    assert reason is None
    stderr = capsys.readouterr().err
    assert "inferring repo from hook cwd" in stderr


def test_decide_bash_cd_does_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # With a cd, the repo is resolved confidently — no warning.
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide(
        "Bash", {"command": f"cd {root.as_posix()} && gh pr merge 5 --squash"}, tmp_path
    )
    assert reason is None
    stderr = capsys.readouterr().err
    assert "inferring repo from hook cwd" not in stderr


def test_decide_bash_explicit_repo_does_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # With an explicit --repo, no warning.
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide("Bash", {"command": "gh pr merge -R o/repo 5 --squash"}, tmp_path)
    assert reason is None
    stderr = capsys.readouterr().err
    assert "inferring repo from hook cwd" not in stderr


def test_decide_bash_cd_denied_on_failed_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cd-resolved repo must still be checked: a failing merge-check denies.
    job_cannon_root = tmp_path / "job-cannon"
    charlie_root = tmp_path / "charlie-work"
    job_cannon_root.mkdir()
    charlie_root.mkdir()
    monkeypatch.setattr(
        hook,
        "_load_fleet_roots",
        lambda: {
            "senkichi/charlie-work": charlie_root,
            "senkichi/job-cannon": job_cannon_root,
        },
    )
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (False, "not_approved"))
    reason = hook._decide(
        "Bash",
        {"command": f"cd {job_cannon_root.as_posix()} && gh pr merge 1679 --squash"},
        charlie_root,
    )
    assert reason is not None
    assert "not_approved" in reason
    assert "#1679" in reason
    assert "senkichi/job-cannon" in reason


# ---------------------------------------------------------------------------
# #1252 review finding: cd in a pipeline or backgrounded command runs in a
# subshell -- its cwd change must not persist to later && / ; -joined commands
# in the same chain. Without this, ``echo x | cd <repo> && gh pr merge N``
# and ``cd <repo> & gh pr merge N`` reopen the exact wrong-repo merge-check
# bypass this hook closes.
# ---------------------------------------------------------------------------


def test_decide_bash_cd_piped_does_not_check_wrong_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The wrong-repo bypass this hook closes: a session anchored in
    # charlie-work runs ``echo x | cd job-cannon && gh pr merge 1679``. The
    # cd is piped (subshell), so the merge actually runs in charlie-work.
    # The hook must check charlie-work's PR #1679, NOT job-cannon's. If the
    # cd were wrongly attributed, merge-check would run against job-cannon
    # (the wrong repo), silently bypassing the gate for charlie-work.
    charlie_root = tmp_path / "charlie-work"
    job_cannon_root = tmp_path / "job-cannon"
    charlie_root.mkdir()
    job_cannon_root.mkdir()
    monkeypatch.setattr(
        hook,
        "_load_fleet_roots",
        lambda: {
            "senkichi/charlie-work": charlie_root,
            "senkichi/job-cannon": job_cannon_root,
        },
    )
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide(
        "Bash",
        {"command": f"echo x | cd {job_cannon_root.as_posix()} && gh pr merge 1679 --squash"},
        charlie_root,
    )
    assert reason is None
    # The merge runs in charlie-work (the hook cwd, since the piped cd does
    # not persist), so merge-check is called on charlie_root, not
    # job_cannon_root.
    assert calls == [(charlie_root, 1679)], (
        f"merge-check must run against charlie-work (hook cwd), not job-cannon "
        f"(piped cd); got {calls}"
    )


def test_decide_bash_cd_backgrounded_does_not_check_wrong_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same bypass via backgrounding: ``cd job-cannon & gh pr merge 1679``.
    # The cd is backgrounded (subshell), so the merge runs in charlie-work.
    charlie_root = tmp_path / "charlie-work"
    job_cannon_root = tmp_path / "job-cannon"
    charlie_root.mkdir()
    job_cannon_root.mkdir()
    monkeypatch.setattr(
        hook,
        "_load_fleet_roots",
        lambda: {
            "senkichi/charlie-work": charlie_root,
            "senkichi/job-cannon": job_cannon_root,
        },
    )
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide(
        "Bash",
        {"command": f"cd {job_cannon_root.as_posix()} & gh pr merge 1679 --squash"},
        charlie_root,
    )
    assert reason is None
    assert calls == [(charlie_root, 1679)], (
        f"merge-check must run against charlie-work (hook cwd), not job-cannon "
        f"(backgrounded cd); got {calls}"
    )
