"""Rule 3 (Worker fate): ``_worktree_still_unsafe`` salvages stranded commits
before the ``worktree_unsafe`` escalation clears.

Covers both callers of the gate: the operator ``unescalate`` command and the
mechanical de-escalation sweep. Kept out of test_fix_unescalate.py and
test_deescalation.py, which are at their file-size ratchet marks.
"""

from __future__ import annotations

from pathlib import Path

from _unescalate_fixtures import _events, _stranded_worktree_bed
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state, state_lock


def _seed_stranded_issue(
    app, branch: str, *, reason: str = "worktree_unsafe_local_commits"
) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": reason,
            "reason_class": "judgment" if reason.endswith("local_commits") else "mechanical",
            "branch_name": branch,
        }
        save_state(app.paths.state_file, state)


def _remote_branch_sha(remote: Path, branch: str) -> str | None:
    import subprocess

    out = subprocess.run(
        ["git", "ls-remote", str(remote), f"refs/heads/{branch}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return out.split()[0] if out else None


def test_unescalate_worktree_unsafe_local_commits_salvages_then_clears(tmp_path: Path) -> None:
    """Rule 3: committed-but-unpushed work is published (ff-only push) and the
    escalation then clears, because the post-salvage re-check is clean.
    Clearing without the push would let the next worktree reset discard the
    commits."""
    from _worktree_fixtures import _git

    branch = "agent/issue-123-salvage-clear"
    app, _repo, wt_path, remote = _stranded_worktree_bed(tmp_path, branch)
    _seed_stranded_issue(app, branch)
    assert _remote_branch_sha(remote, branch) is None

    result = app.unescalate(None, 123, dry_run=False)

    assert result.data.get("worktree_still_unsafe") is not True
    assert result.data["changed"] is True
    assert _remote_branch_sha(remote, branch) == _git(wt_path, "rev-parse", "HEAD").stdout.strip()
    state = load_state(app.paths.state_file)
    salvaged = _events(state, "worktree_unsafe_stranded_salvaged")
    assert len(salvaged) == 1
    assert salvaged[0]["payload"]["mode"] == "pushed"
    assert salvaged[0]["payload"]["issue_number"] == 123
    assert _events(state, "worktree_unsafe_stranded_salvage_failed") == []


def test_unescalate_worktree_unsafe_salvage_failure_keeps_label(tmp_path: Path) -> None:
    """A diverged remote branch cannot be fast-forwarded: salvage is skipped,
    nothing is pushed, and the ORIGINAL reason keeps the label."""
    from _worktree_fixtures import _git

    branch = "agent/issue-123-salvage-diverged"
    app, repo, _wt_path, remote = _stranded_worktree_bed(tmp_path, branch)
    # A foreign commit lands on origin/<branch>: same object store, so the
    # remote head is present locally and is NOT an ancestor of the worktree.
    _git(repo, "checkout", "-q", "-b", "other-tip", "main")
    (repo / "other.txt").write_text("foreign work\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-m", "diverging remote commit")
    _git(repo, "push", "origin", f"other-tip:refs/heads/{branch}")
    _git(repo, "checkout", "-q", "main")
    remote_before = _remote_branch_sha(remote, branch)
    _seed_stranded_issue(app, branch)

    result = app.unescalate(None, 123, dry_run=False)

    assert result.data["worktree_still_unsafe"] is True
    assert result.data["changed"] is False
    assert _remote_branch_sha(remote, branch) == remote_before
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    failed = _events(state, "worktree_unsafe_stranded_salvage_failed")
    assert len(failed) == 1
    assert failed[0]["payload"]["skip_reason"] == "diverged"
    assert _events(state, "worktree_unsafe_stranded_salvaged") == []


def test_unescalate_worktree_unsafe_dry_run_never_clears(tmp_path: Path) -> None:
    """``--dry-run`` never pushes and never reports a local-commits blocker as
    cleared: the commits are still only local afterwards."""
    branch = "agent/issue-123-salvage-dry-run"
    app, _repo, _wt_path, remote = _stranded_worktree_bed(tmp_path, branch)
    _seed_stranded_issue(app, branch)

    result = app.unescalate(None, 123, dry_run=True)

    assert result.data["worktree_still_unsafe"] is True
    assert _remote_branch_sha(remote, branch) is None
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert _events(state, "worktree_unsafe_stranded_salvaged") == []
    assert _events(state, "worktree_unsafe_stranded_salvage_failed") == []


def test_unescalate_worktree_unsafe_shim_dirt_unchanged(tmp_path: Path) -> None:
    """Worker-authored dirt is not the stranded case: no salvage is attempted
    (nothing is pushed), the dirt survives, and the label stays."""
    branch = "agent/issue-123-salvage-dirt"
    app, _repo, wt_path, remote = _stranded_worktree_bed(tmp_path, branch)
    (wt_path / "worker_wip.txt").write_text("uncommitted work\n", encoding="utf-8")
    _seed_stranded_issue(app, branch, reason="worktree_unsafe_shim_dirt")

    result = app.unescalate(None, 123, dry_run=False)

    assert result.data["worktree_still_unsafe"] is True
    assert "uncommitted" in result.data["worktree_unsafe_reason"]
    assert _remote_branch_sha(remote, branch) is None
    assert (wt_path / "worker_wip.txt").read_text(encoding="utf-8") == "uncommitted work\n"
    state = load_state(app.paths.state_file)
    assert _events(state, "worktree_unsafe_stranded_salvaged") == []
    assert _events(state, "worktree_unsafe_stranded_salvage_failed") == []


def test_unescalate_worktree_unsafe_no_origin_parks_then_clears(tmp_path: Path) -> None:
    """No origin remote: the #1944 archive park stands in for the push. The
    tip is preserved on ``archive/<branch>-<date>`` before the label clears."""
    from _worktree_fixtures import _git

    branch = "agent/issue-123-salvage-park"
    app, repo, wt_path, _remote = _stranded_worktree_bed(tmp_path, branch, origin=False)
    _seed_stranded_issue(app, branch)
    tip = _git(wt_path, "rev-parse", "HEAD").stdout.strip()

    result = app.unescalate(None, 123, dry_run=False)

    assert result.data.get("worktree_still_unsafe") is not True
    archived = _git(
        repo, "for-each-ref", "--format=%(objectname)", f"refs/heads/archive/{branch}-*"
    )
    assert archived.stdout.split() == [tip]
    state = load_state(app.paths.state_file)
    salvaged = _events(state, "worktree_unsafe_stranded_salvaged")
    assert [e["payload"]["mode"] for e in salvaged] == ["parked"]


def test_mechanical_deescalation_salvages_stranded_before_clear(tmp_path: Path) -> None:
    """Rule 3 on the mechanical path: a ``mechanical`` worktree_unsafe
    escalation whose worktree holds only committed-but-unpushed work is
    salvaged (ff-only push) by ``_worktree_still_unsafe`` and only then
    cleared -- the sweep no longer skips it as ``worktree_still_unsafe``."""
    from _unescalate_fixtures import _stranded_worktree_bed
    from _worktree_fixtures import _git

    branch = "agent/issue-123-mechanical-salvage"
    app, _repo, wt_path, remote = _stranded_worktree_bed(tmp_path, branch)

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "escalation_reason": "worktree_unsafe_local_commits",
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "worktree_unsafe_local_commits",
            "reason_class": "mechanical",
            "branch_name": branch,
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue_123 = state["issues"]["123"]
    assert issue_123["status"] == PASSIVE_OPEN_STATUS
    assert issue_123["auto_deescalation_count"] == 1
    remote_tip = _git(remote, "rev-parse", f"refs/heads/{branch}").stdout.strip()
    assert remote_tip == _git(wt_path, "rev-parse", "HEAD").stdout.strip()
    assert len(_events(state, "worktree_unsafe_stranded_salvaged")) == 1
